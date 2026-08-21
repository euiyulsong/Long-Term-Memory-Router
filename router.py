# ltm_retrieval_ablation_with_em.py
#
# ============================================================
# LTM RETRIEVAL + QA ABLATION
#
# DATA
# ------------------------------------------------------------
# LoCoMo : 100 examples
#   -> MUST retrieve Long-Term Memory
#
# SQuAD  : 100 examples
#   -> MUST NOT retrieve Long-Term Memory
#      because current context already contains the answer
#
# TOTAL: 200 examples
#
#
# ABLATION
# ------------------------------------------------------------
# Query:
#   - multiturn
#   - rewrite
#
# Pre-router:
#   - OFF
#   - ON
#
# Post-router:
#   - OFF
#   - ON
#
# 2 x 2 x 2 = 8 configurations
#
#
# RETRIEVAL METRICS
# ------------------------------------------------------------
# LoCoMo:
#   Recall@5
#   Post-filter Recall
#   Pre-router Recall
#
# SQuAD:
#   Pre-router Specificity
#   Unnecessary Retrieval Rate
#   Post-router Empty Rate
#   Post-router Keep Rate
#
#
# QA METRICS
# ------------------------------------------------------------
# LoCoMo EM / F1
# SQuAD EM / F1
# Overall EM / F1
#
#
# LLM
# ------------------------------------------------------------
# Qwen3-8B via OpenRouter
#
# - rewrite      : few-shot, short output
# - pre-router   : 1 / 0
# - post-router  : 1 / 0
# - answer       : short answer only
#
# API calls use ThreadPoolExecutor(max_workers=100)
#
# ============================================================
#
# pip install \
#   openai \
#   datasets \
#   sentence-transformers \
#   torch \
#   numpy \
#   pandas \
#   requests \
#   tqdm
#
# export OPENROUTER_API_KEY="sk-or-..."
#
# python ltm_retrieval_ablation_with_em.py
#
# ============================================================

import os
import re
import json
import random
import hashlib
import string

from pathlib import Path
from itertools import islice
from collections import Counter
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import requests
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_LOCOMO = 100
N_SQUAD = 100

TOP_K = 5

RECENT_CONTEXT_TURNS = 6

MAX_WORKERS = 100

LLM_MODEL = "qwen/qwen3-8b"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# IMPORTANT:
# Keep the SAME cache directory as the previous experiment.
# Existing rewrite / pre / post cache files will be reused.
CACHE_DIR = Path("./cache_ltm_fast")
CACHE_DIR.mkdir(exist_ok=True)

LOCOMO_PATH = CACHE_DIR / "locomo10.json"

LOCOMO_URL = (
    "https://raw.githubusercontent.com/"
    "snap-research/locomo/main/data/locomo10.json"
)

RAW_CSV = "ltm_em_raw.csv"
SUMMARY_CSV = "ltm_em_summary.csv"

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# OPENROUTER
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_retries=5,
    timeout=60,
)


def llm(
    system,
    user,
    max_tokens,
):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
    )

    text = (
        response
        .choices[0]
        .message
        .content
    )

    return (text or "").strip()


# ============================================================
# DISK CACHE
# ============================================================

def make_cache_path(
    prefix,
    system,
    user,
    max_tokens,
):
    payload = json.dumps(
        {
            "model": LLM_MODEL,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    h = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return (
        CACHE_DIR
        / f"{prefix}_{h}.txt"
    )


def cached_llm(
    prefix,
    system,
    user,
    max_tokens,
):
    path = make_cache_path(
        prefix,
        system,
        user,
        max_tokens,
    )

    if path.exists():
        return path.read_text(
            encoding="utf-8"
        ).strip()

    result = llm(
        system=system,
        user=user,
        max_tokens=max_tokens,
    )

    path.write_text(
        result,
        encoding="utf-8",
    )

    return result


# ============================================================
# PARALLEL LLM
# ============================================================

def parallel_map(
    fn,
    items,
    desc,
    max_workers=MAX_WORKERS,
):
    """
    Up to 100 OpenRouter requests in parallel.

    Results preserve original input order.
    """

    if not items:
        return []

    results = [None] * len(items)

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                fn,
                item,
            ): i
            for i, item in enumerate(items)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
        ):
            idx = futures[future]

            try:
                results[idx] = future.result()

            except Exception as e:
                print(
                    f"\nERROR idx={idx}: {e}"
                )
                results[idx] = None

    return results


# ============================================================
# LOCOMO DOWNLOAD
# ============================================================

def download_locomo():

    if LOCOMO_PATH.exists():
        return

    print("Downloading LoCoMo...")

    response = requests.get(
        LOCOMO_URL,
        timeout=60,
    )

    response.raise_for_status()

    LOCOMO_PATH.write_bytes(
        response.content
    )


def load_locomo():

    download_locomo()

    with open(
        LOCOMO_PATH,
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# LOCOMO PARSING
# ============================================================

def get_sessions(
    conversation
):
    sessions = []

    for key, value in conversation.items():

        m = re.fullmatch(
            r"session_(\d+)",
            key,
        )

        if not m:
            continue

        session_id = int(
            m.group(1)
        )

        date = conversation.get(
            f"session_{session_id}_date_time",
            "",
        )

        turns = []

        for t in value:

            text = (
                t.get("text")
                or t.get("blip_caption")
                or ""
            )

            if not text:
                continue

            turns.append({
                "dia_id":
                    t.get("dia_id"),

                "speaker":
                    t.get(
                        "speaker",
                        "",
                    ),

                "text":
                    text,

                "date":
                    date,

                "session":
                    session_id,
            })

        sessions.append(
            (
                session_id,
                turns,
            )
        )

    sessions.sort(
        key=lambda x: x[0]
    )

    return sessions


def flatten_conversation(
    conversation
):
    output = []

    for _, turns in get_sessions(
        conversation
    ):
        output.extend(
            turns
        )

    return output


def turn_text(
    t
):
    return (
        f"[{t.get('date', '')}] "
        f"{t.get('speaker', '')}: "
        f"{t['text']}"
    )


# ============================================================
# LOCOMO: 100 MEMORY-NEEDED QA
# ============================================================

def build_locomo_examples(data):
    candidates = []

    for conv_idx, conv in enumerate(data):

        conversation_id = str(
            conv.get(
                "sample_id",
                conv_idx,
            )
        )

        turns = flatten_conversation(
            conv["conversation"]
        )

        turn_map = {
            t["dia_id"]: t
            for t in turns
            if t.get("dia_id")
        }

        for qa_idx, qa in enumerate(
            conv.get("qa", [])
        ):

            try:
                category = int(
                    qa.get(
                        "category",
                        -1,
                    )
                )
            except Exception:
                continue

            # conversational-memory QA only
            if category not in {1, 2, 4}:
                continue

            # -----------------------------
            # Question
            # -----------------------------
            raw_question = qa.get(
                "question",
                "",
            )

            question = (
                ""
                if raw_question is None
                else str(raw_question).strip()
            )

            # -----------------------------
            # Answer
            # -----------------------------
            raw_answer = qa.get(
                "answer",
                "",
            )

            if raw_answer is None:
                answer = ""

            elif isinstance(
                raw_answer,
                list,
            ):
                answer = " ".join(
                    str(x)
                    for x in raw_answer
                ).strip()

            else:
                answer = str(
                    raw_answer
                ).strip()

            # -----------------------------
            # Evidence
            # -----------------------------
            evidence_ids = (
                qa.get(
                    "evidence",
                    [],
                )
                or []
            )

            evidence_ids = [
                x
                for x in evidence_ids
                if x in turn_map
            ]

            if not question:
                continue

            if not answer:
                continue

            if not evidence_ids:
                continue

            evidence = set(
                evidence_ids
            )

            # GT evidence must NOT be in
            # current context.
            non_evidence_turns = [
                t
                for t in turns
                if t.get("dia_id")
                not in evidence
            ]

            current_turns = (
                non_evidence_turns[
                    -RECENT_CONTEXT_TURNS:
                ]
            )

            current_context = [
                turn_text(t)
                for t in current_turns
            ]

            # Full conversation = LTM
            memories = [
                {
                    "memory_id":
                        t.get("dia_id"),

                    "text":
                        turn_text(t),
                }
                for t in turns
            ]

            candidates.append({
                "id":
                    (
                        f"locomo_"
                        f"{conversation_id}_"
                        f"{qa_idx}"
                    ),

                "dataset":
                    "locomo",

                "condition":
                    "memory_needed",

                "should_retrieve":
                    1,

                "question":
                    question,

                "answer":
                    answer,

                "current_context":
                    current_context,

                "ltm":
                    memories,

                "evidence_ids":
                    evidence_ids,

                "conversation_id":
                    conversation_id,
            })

    random.Random(
        SEED
    ).shuffle(
        candidates
    )

    print(
        "Eligible LoCoMo QA:",
        len(candidates),
    )

    if len(candidates) < N_LOCOMO:
        raise RuntimeError(
            f"Not enough LoCoMo examples: "
            f"{len(candidates)}"
        )

    return candidates[:N_LOCOMO]
# ============================================================
# SQUAD: 100 NO-LTM-NEEDED QA
# ============================================================

def load_squad_100():

    ds = load_dataset(
        "rajpurkar/squad",
        split="validation",
        streaming=True,
    )

    return list(
        islice(
            ds,
            N_SQUAD,
        )
    )


def build_squad_examples(
    squad_rows,
    locomo_examples,
):
    examples = []

    rng = random.Random(
        SEED + 123
    )

    memory_pools = [
        x["ltm"]
        for x in locomo_examples
    ]

    for i, row in enumerate(
        squad_rows
    ):

        unrelated_ltm = rng.choice(
            memory_pools
        )

        answers = (
            row.get(
                "answers",
                {},
            )
            .get(
                "text",
                [],
            )
        )

        if not answers:
            continue

        examples.append({
            "id":
                f"squad_{i}",

            "dataset":
                "squad",

            "condition":
                "no_ltm_needed",

            "should_retrieve":
                0,

            "question":
                row["question"],

            "answer":
                answers[0],

            # Current context contains the answer.
            "current_context": [
                "Context: "
                + row["context"]
            ],

            # unrelated user-specific LTM noise
            "ltm":
                unrelated_ltm,

            "evidence_ids":
                [],

            "conversation_id":
                None,
        })

    if len(examples) != N_SQUAD:
        raise RuntimeError(
            f"Expected {N_SQUAD} SQuAD "
            f"examples, got {len(examples)}"
        )

    return examples


# ============================================================
# EMBEDDING MODEL
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Loading {EMBED_MODEL} "
    f"on {device}"
)

embedder = SentenceTransformer(
    EMBED_MODEL,
    device=device,
)


def embed_texts(
    texts
):
    return embedder.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


# ============================================================
# MEMORY EMBEDDING CACHE
#
# In-memory cache.
# Recomputed after process restart.
# ============================================================

memory_embedding_cache = {}


def get_memory_embeddings(
    ex
):

    if ex["dataset"] == "locomo":

        key = (
            "locomo_"
            + str(
                ex[
                    "conversation_id"
                ]
            )
        )

    else:

        memory_text = "\n".join(
            m["text"]
            for m in ex["ltm"]
        )

        key = hashlib.md5(
            memory_text.encode(
                "utf-8"
            )
        ).hexdigest()

    if key in memory_embedding_cache:

        return (
            memory_embedding_cache[
                key
            ]
        )

    emb = embed_texts(
        [
            m["text"]
            for m in ex["ltm"]
        ]
    )

    memory_embedding_cache[
        key
    ] = emb

    return emb


# ============================================================
# CONTEXT TEXT
# ============================================================

def current_context_text(
    ex
):

    return "\n".join(
        ex["current_context"]
    )


# ============================================================
# QUERY REWRITE
#
# SAME prompt as previous version
# so old cache can be reused.
# ============================================================

REWRITE_SYSTEM = """
Convert the final question into a short standalone search query
for retrieving long-term user memory.

Resolve references using the current conversation.
Remove irrelevant context.
Do not answer.

Examples:

Conversation:
User: I need treats for Momo.
Question:
What flavor does she like?
Output:
Momo favorite treat flavor

Conversation:
User: I'm booking another flight.
Question:
Which seat do I usually prefer?
Output:
user preferred airplane seat

Conversation:
User: We are debugging alpha-7.
Question:
Which server are we debugging?
Output:
current debugging server

Return only the short search query.
""".strip()


def rewrite_one(
    ex
):

    user = f"""
Conversation:
{current_context_text(ex)}

Question:
{ex["question"]}

Output:
""".strip()

    return cached_llm(
        prefix="rewrite",
        system=REWRITE_SYSTEM,
        user=user,
        max_tokens=32,
    )


# ============================================================
# PRE-ROUTER
#
# SAME prompt as previous version.
#
# 1 = retrieve
# 0 = skip
# ============================================================

PRE_ROUTER_SYSTEM = """
Decide whether LONG-TERM USER MEMORY must be retrieved to answer
the question.

Output:
1 = information is missing from the current context and earlier
    user-specific memory is required.
0 = current context already contains enough information, or the
    question does not require user memory.

Examples:

Context:
User: I am buying treats for Momo.
Question:
What flavor does she usually like?
Output:
1

Context:
User: Momo likes salmon treats.
Question:
What flavor does Momo like?
Output:
0

Context:
Paris is the capital and largest city of France.
Question:
What is the capital of France?
Output:
0

Return only 1 or 0.
""".strip()


def pre_router_one(
    ex
):

    user = f"""
Context:
{current_context_text(ex)}

Question:
{ex["question"]}

Output:
""".strip()

    out = cached_llm(
        prefix="pre",
        system=PRE_ROUTER_SYSTEM,
        user=user,
        max_tokens=2,
    )

    match = re.search(
        r"[01]",
        out,
    )

    if match:
        return int(
            match.group()
        )

    return 0


# ============================================================
# POST-ROUTER
#
# SAME prompt as previous version.
#
# 1 = keep
# 0 = discard
# ============================================================

POST_ROUTER_SYSTEM = """
Judge whether ONE retrieved long-term memory directly contains
information useful for answering the question.

Output:
1 = keep
0 = discard

Examples:

Question:
What treats does Momo like?
Memory:
User: Momo loves salmon treats.
Output:
1

Question:
What treats does Momo like?
Memory:
User: My laptop is a MacBook Pro.
Output:
0

Question:
What is the capital of France?
Memory:
User: I prefer aisle seats on airplanes.
Output:
0

Do not keep a memory only because it shares a topic.
Return only 1 or 0.
""".strip()


def post_router_one(
    item
):

    ex = item["example"]
    memory = item["memory"]

    user = f"""
Question:
{ex["question"]}

Memory:
{memory["text"]}

Output:
""".strip()

    out = cached_llm(
        prefix="post",
        system=POST_ROUTER_SYSTEM,
        user=user,
        max_tokens=2,
    )

    match = re.search(
        r"[01]",
        out,
    )

    if match:
        return int(
            match.group()
        )

    return 0


# ============================================================
# ANSWER GENERATION
#
# NEW.
#
# Few-shot and short output to keep generation cheap.
# ============================================================

ANSWER_SYSTEM = """
Answer the question using the current context and retrieved
long-term memories.

Return only the shortest correct answer.
Do not explain.
Do not write a full sentence unless necessary.

Examples:

Current context:
Momo likes salmon treats.

Long-term memories:
(none)

Question:
What treats does Momo like?

Answer:
salmon treats

Current context:
The capital of France is Paris.

Long-term memories:
(none)

Question:
What is the capital of France?

Answer:
Paris

Current context:
User: I am buying treats for Momo.

Long-term memories:
User: Momo loves salmon treats.

Question:
What flavor does Momo like?

Answer:
salmon
""".strip()


def answer_one(
    item
):

    ex = item["example"]

    memories = item[
        "memories"
    ]

    if memories:

        memory_text = "\n".join(
            m["text"]
            for m in memories
        )

    else:

        memory_text = "(none)"

    user = f"""
Current context:
{current_context_text(ex)}

Long-term memories:
{memory_text}

Question:
{ex["question"]}

Answer:
""".strip()

    # max 32 tokens.
    return cached_llm(
        prefix="answer",
        system=ANSWER_SYSTEM,
        user=user,
        max_tokens=32,
    )


# ============================================================
# RETRIEVER
# ============================================================

def retrieve(
    ex,
    query,
    k=TOP_K,
):

    memories = ex["ltm"]

    memory_emb = (
        get_memory_embeddings(
            ex
        )
    )

    query_emb = (
        embed_texts(
            [query]
        )[0]
    )

    scores = (
        memory_emb
        @ query_emb
    )

    k = min(
        k,
        len(memories),
    )

    order = np.argsort(
        -scores
    )[:k]

    return [
        {
            **memories[i],

            "score":
                float(
                    scores[i]
                ),
        }
        for i in order
    ]


# ============================================================
# RETRIEVAL HIT
# ============================================================

def evidence_hit(
    ex,
    memories,
):

    if (
        ex["dataset"]
        != "locomo"
    ):
        return np.nan

    gt = set(
        ex["evidence_ids"]
    )

    returned = {
        m["memory_id"]
        for m in memories
    }

    return int(
        bool(
            gt
            & returned
        )
    )


# ============================================================
# SQUAD-STYLE NORMALIZATION
# ============================================================

def normalize_answer(
    s
):

    def lower(
        text
    ):
        return text.lower()

    def remove_punc(
        text
    ):

        exclude = set(
            string.punctuation
        )

        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

    def remove_articles(
        text
    ):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def white_space_fix(
        text
    ):
        return " ".join(
            text.split()
        )

    return white_space_fix(
        remove_articles(
            remove_punc(
                lower(
                    str(s)
                )
            )
        )
    )


def exact_match_score(
    prediction,
    gold,
):

    return int(
        normalize_answer(
            prediction
        )
        ==
        normalize_answer(
            gold
        )
    )


def token_f1_score(
    prediction,
    gold,
):

    pred_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    gold_tokens = (
        normalize_answer(
            gold
        ).split()
    )

    if (
        len(pred_tokens) == 0
        and
        len(gold_tokens) == 0
    ):
        return 1.0

    if (
        len(pred_tokens) == 0
        or
        len(gold_tokens) == 0
    ):
        return 0.0

    common = (
        Counter(pred_tokens)
        &
        Counter(gold_tokens)
    )

    num_same = sum(
        common.values()
    )

    if num_same == 0:
        return 0.0

    precision = (
        num_same
        /
        len(pred_tokens)
    )

    recall = (
        num_same
        /
        len(gold_tokens)
    )

    return (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )
    )


# ============================================================
# PRECOMPUTE REWRITE + PRE ROUTER
#
# Existing cache should make this fast.
# ============================================================

def precompute_llm_outputs(
    examples
):

    print()
    print("=" * 80)
    print("PRECOMPUTE REWRITE")
    print("=" * 80)

    rewrites = parallel_map(
        rewrite_one,
        examples,
        desc="rewrite",
    )

    for ex, value in zip(
        examples,
        rewrites,
    ):

        ex[
            "_rewrite"
        ] = (
            value
            or ex["question"]
        )

    print()
    print("=" * 80)
    print("PRECOMPUTE PRE-ROUTER")
    print("=" * 80)

    pre_outputs = parallel_map(
        pre_router_one,
        examples,
        desc="pre-router",
    )

    for ex, value in zip(
        examples,
        pre_outputs,
    ):

        ex[
            "_pre_router"
        ] = (
            0
            if value is None
            else value
        )


# ============================================================
# RAW MULTI-TURN QUERY
# ============================================================

def raw_multiturn_query(
    ex
):

    return (
        current_context_text(ex)
        + "\n"
        + ex["question"]
    )


# ============================================================
# RUN ONE CONFIG
# ============================================================

def run_config(
    examples,
    query_mode,
    use_pre,
    use_post,
):

    temporary = []

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    for ex in tqdm(
        examples,
        desc=(
            f"retrieve "
            f"{query_mode} "
            f"pre={use_pre}"
        ),
    ):

        if query_mode == "rewrite":

            query = ex[
                "_rewrite"
            ]

        else:

            query = (
                raw_multiturn_query(
                    ex
                )
            )

        if use_pre:

            router_output = ex[
                "_pre_router"
            ]

        else:

            router_output = 1

        if router_output == 1:

            raw_memories = retrieve(
                ex,
                query,
                TOP_K,
            )

        else:

            raw_memories = []

        temporary.append({
            "example":
                ex,

            "query":
                query,

            "router_output":
                router_output,

            "raw_memories":
                raw_memories,
        })

    # --------------------------------------------------------
    # POST ROUTER
    # --------------------------------------------------------

    if use_post:

        post_jobs = []

        for item_idx, item in enumerate(
            temporary
        ):

            for memory_idx, memory in enumerate(
                item[
                    "raw_memories"
                ]
            ):

                post_jobs.append({
                    "item_idx":
                        item_idx,

                    "memory_idx":
                        memory_idx,

                    "example":
                        item[
                            "example"
                        ],

                    "memory":
                        memory,
                })

        outputs = parallel_map(
            post_router_one,
            post_jobs,
            desc="post-router",
        )

        keep_map = {}

        for job, output in zip(
            post_jobs,
            outputs,
        ):

            keep_map[
                (
                    job["item_idx"],
                    job["memory_idx"],
                )
            ] = (
                0
                if output is None
                else output
            )

    else:

        keep_map = {}

    # --------------------------------------------------------
    # BUILD FINAL MEMORY SETS
    # --------------------------------------------------------

    for item_idx, item in enumerate(
        temporary
    ):

        raw_memories = item[
            "raw_memories"
        ]

        if use_post:

            final_memories = []

            for memory_idx, memory in enumerate(
                raw_memories
            ):

                keep = keep_map.get(
                    (
                        item_idx,
                        memory_idx,
                    ),
                    0,
                )

                if keep == 1:

                    final_memories.append(
                        memory
                    )

        else:

            final_memories = (
                raw_memories
            )

        item[
            "final_memories"
        ] = final_memories

    # --------------------------------------------------------
    # ANSWER GENERATION
    #
    # 200 jobs/config in parallel.
    #
    # Answer cache also means identical context/memory/query
    # combinations across configs are reused.
    # --------------------------------------------------------

    answer_jobs = [
        {
            "example":
                item["example"],

            "memories":
                item[
                    "final_memories"
                ],
        }
        for item in temporary
    ]

    predictions = parallel_map(
        answer_one,
        answer_jobs,
        desc="answer",
    )

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    rows = []

    for item_idx, (
        item,
        prediction,
    ) in enumerate(
        zip(
            temporary,
            predictions,
        )
    ):

        ex = item[
            "example"
        ]

        raw_memories = item[
            "raw_memories"
        ]

        final_memories = item[
            "final_memories"
        ]

        prediction = (
            prediction
            or ""
        )

        rows.append({
            "id":
                ex["id"],

            "dataset":
                ex["dataset"],

            "condition":
                ex[
                    "condition"
                ],

            "query":
                query_mode,

            "pre":
                use_pre,

            "post":
                use_post,

            "should_retrieve":
                ex[
                    "should_retrieve"
                ],

            "router_output":
                item[
                    "router_output"
                ],

            # --------------------------------------------
            # retrieval
            # --------------------------------------------

            "raw_memories":
                len(
                    raw_memories
                ),

            "final_memories":
                len(
                    final_memories
                ),

            "recall_hit":
                evidence_hit(
                    ex,
                    raw_memories,
                ),

            "post_recall_hit":
                evidence_hit(
                    ex,
                    final_memories,
                ),

            "post_empty":
                int(
                    len(
                        final_memories
                    )
                    == 0
                ),

            "post_keep_rate":
                (
                    len(
                        final_memories
                    )
                    /
                    len(
                        raw_memories
                    )

                    if raw_memories

                    else 0.0
                ),

            # --------------------------------------------
            # QA
            # --------------------------------------------

            "gold":
                ex["answer"],

            "prediction":
                prediction,

            "em":
                exact_match_score(
                    prediction,
                    ex["answer"],
                ),

            "f1":
                token_f1_score(
                    prediction,
                    ex["answer"],
                ),
        })

    return rows


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    df
):

    rows = []

    for (
        query,
        pre,
        post,
    ), g in df.groupby(
        [
            "query",
            "pre",
            "post",
        ]
    ):

        locomo = g[
            g.dataset
            ==
            "locomo"
        ]

        squad = g[
            g.dataset
            ==
            "squad"
        ]

        rows.append({
            # ================================================
            # CONFIG
            # ================================================

            "query":
                query,

            "pre":
                pre,

            "post":
                post,

            # ================================================
            # QA
            # ================================================

            "overall_em":
                g[
                    "em"
                ].mean(),

            "overall_f1":
                g[
                    "f1"
                ].mean(),

            "locomo_em":
                locomo[
                    "em"
                ].mean(),

            "locomo_f1":
                locomo[
                    "f1"
                ].mean(),

            "squad_em":
                squad[
                    "em"
                ].mean(),

            "squad_f1":
                squad[
                    "f1"
                ].mean(),

            # ================================================
            # LOCOMO RETRIEVAL
            # ================================================

            f"recall@{TOP_K}":
                locomo[
                    "recall_hit"
                ].mean(),

            "post_filter_recall":
                locomo[
                    "post_recall_hit"
                ].mean(),

            "pre_router_recall":
                locomo[
                    "router_output"
                ].mean(),

            # ================================================
            # SQUAD ROUTING
            # ================================================

            "pre_router_specificity":
                (
                    1.0
                    -
                    squad[
                        "router_output"
                    ].mean()
                ),

            "unnecessary_retrieval":
                squad[
                    "router_output"
                ].mean(),

            # ================================================
            # SQUAD POST ROUTER
            # ================================================

            "squad_post_empty_rate":
                squad[
                    "post_empty"
                ].mean(),

            "squad_post_keep_rate":
                squad[
                    "post_keep_rate"
                ].mean(),

            # ================================================
            # MEMORY COUNTS
            # ================================================

            "avg_raw_memories":
                g[
                    "raw_memories"
                ].mean(),

            "avg_final_memories":
                g[
                    "final_memories"
                ].mean(),

            "locomo_final_memories":
                locomo[
                    "final_memories"
                ].mean(),

            "squad_final_memories":
                squad[
                    "final_memories"
                ].mean(),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOAD LOCOMO")
    print("=" * 80)

    locomo_data = (
        load_locomo()
    )

    locomo = (
        build_locomo_examples(
            locomo_data
        )
    )

    print()
    print("=" * 80)
    print("LOAD SQUAD")
    print("=" * 80)

    squad_rows = (
        load_squad_100()
    )

    squad = (
        build_squad_examples(
            squad_rows,
            locomo,
        )
    )

    assert (
        len(locomo)
        == N_LOCOMO
    )

    assert (
        len(squad)
        == N_SQUAD
    )

    examples = (
        locomo
        + squad
    )

    print()
    print(
        f"LoCoMo memory-needed : "
        f"{len(locomo)}"
    )

    print(
        f"SQuAD no-LTM-needed  : "
        f"{len(squad)}"
    )

    print(
        f"Total                : "
        f"{len(examples)}"
    )

    # --------------------------------------------------------
    # REWRITE + PRE ROUTER
    #
    # Old cache should mostly handle these immediately.
    # --------------------------------------------------------

    precompute_llm_outputs(
        examples
    )

    # --------------------------------------------------------
    # 8 CONFIGS
    # --------------------------------------------------------

    configs = [
        (
            query,
            pre,
            post,
        )
        for query in [
            "multiturn",
            "rewrite",
        ]
        for pre in [
            False,
            True,
        ]
        for post in [
            False,
            True,
        ]
    ]

    all_rows = []

    for (
        query,
        pre,
        post,
    ) in configs:

        print()
        print(
            "=" * 100
        )

        print(
            f"QUERY={query} "
            f"PRE={pre} "
            f"POST={post}"
        )

        print(
            "=" * 100
        )

        rows = run_config(
            examples=examples,
            query_mode=query,
            use_pre=pre,
            use_post=post,
        )

        all_rows.extend(
            rows
        )

        # Save after every config.
        pd.DataFrame(
            all_rows
        ).to_csv(
            RAW_CSV,
            index=False,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_rows
    )

    summary = summarize(
        df
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print()
    print(
        "=" * 220
    )

    print(
        "LONG-TERM MEMORY RETRIEVAL "
        "+ QA ABLATION RESULTS"
    )

    print(
        "=" * 220
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print("Saved:")
    print(
        f"  {RAW_CSV}"
    )
    print(
        f"  {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
