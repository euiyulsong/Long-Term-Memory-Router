# ltm_router_ablation.py
#
# pip install openai sentence-transformers torch numpy pandas requests tqdm
#
# export OPENROUTER_API_KEY=sk-or-...
# python ltm_router_ablation.py
#
# Dataset:
#   LoCoMo, ACL 2024
#
# Experiment:
#   1000 memory-needed
#   1000 memory-not-needed
#
# Ablation:
#   query = multiturn / rewrite
#   pre_router = False / True
#   post_router = False / True

import os
import re
import json
import random
import hashlib
from pathlib import Path
from collections import Counter

import requests
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_BASE = 1000

TOP_K = 5
RECENT_CONTEXT_TURNS = 6
CURRENT_EVIDENCE_WINDOW = 2

LLM_MODEL = "qwen/qwen3-8b"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CACHE_DIR = Path("./cache_ltm_ablation")
CACHE_DIR.mkdir(exist_ok=True)

LOCOMO_PATH = CACHE_DIR / "locomo10.json"

# Official repository
LOCOMO_URL = (
    "https://raw.githubusercontent.com/"
    "snap-research/locomo/main/data/locomo10.json"
)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# OPENROUTER
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_retries=5,
)


def llm(system, user, max_tokens=128):
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

    text = response.choices[0].message.content

    return (text or "").strip()


# ============================================================
# SIMPLE DISK CACHE
# ============================================================

def cache_key(prefix, payload):
    x = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    h = hashlib.sha256(
        x.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{prefix}_{h}.txt"


def cached_llm(prefix, system, user, max_tokens=128):

    path = cache_key(
        prefix,
        {
            "model": LLM_MODEL,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
        },
    )

    if path.exists():
        return path.read_text()

    result = llm(
        system,
        user,
        max_tokens=max_tokens,
    )

    path.write_text(result)

    return result


# ============================================================
# LOAD LOCOMO
# ============================================================

def download_locomo():

    if LOCOMO_PATH.exists():
        return

    print("Downloading LoCoMo...")

    r = requests.get(
        LOCOMO_URL,
        timeout=60,
    )

    r.raise_for_status()

    LOCOMO_PATH.write_bytes(r.content)


def load_locomo():

    download_locomo()

    with open(LOCOMO_PATH) as f:
        return json.load(f)


# ============================================================
# CONVERSATION PARSING
# ============================================================

def get_sessions(conversation):

    sessions = []

    for key, value in conversation.items():

        m = re.fullmatch(
            r"session_(\d+)",
            key,
        )

        if not m:
            continue

        session_num = int(m.group(1))

        date = conversation.get(
            f"session_{session_num}_date_time",
            "",
        )

        turns = []

        for t in value:

            turns.append({
                "dia_id": t.get("dia_id"),
                "speaker": t.get("speaker", ""),
                "text": t.get(
                    "text",
                    t.get("blip_caption", ""),
                ),
                "session": session_num,
                "date": date,
            })

        sessions.append(
            (
                session_num,
                turns,
            )
        )

    sessions.sort(
        key=lambda x: x[0]
    )

    return sessions


def flatten_conversation(conversation):

    result = []

    for _, turns in get_sessions(
        conversation
    ):
        result.extend(turns)

    return result


# ============================================================
# BUILD 1000 BASE QA
# ============================================================

def build_base_examples(data):

    candidates = []

    for conv in data:

        conversation = conv["conversation"]

        turns = flatten_conversation(
            conversation
        )

        turn_map = {
            t["dia_id"]: t
            for t in turns
            if t["dia_id"]
        }

        for i, qa in enumerate(conv["qa"]):

            category = int(
                qa.get("category", -1)
            )

            # LoCoMo:
            #
            # 1 / 2 / 4 are conversation-memory QA.
            # Exclude:
            # 3 = open-domain
            # 5 = adversarial
            #
            if category not in {1, 2, 4}:
                continue

            evidence_ids = qa.get(
                "evidence",
                []
            )

            if not evidence_ids:
                continue

            evidence_ids = [
                x
                for x in evidence_ids
                if x in turn_map
            ]

            if not evidence_ids:
                continue

            answer = qa.get(
                "answer",
                ""
            )

            question = qa.get(
                "question",
                ""
            )

            if not answer or not question:
                continue

            candidates.append({
                "base_id":
                    f"{conv['sample_id']}_{i}",

                "conversation_id":
                    conv["sample_id"],

                "question":
                    question,

                "answer":
                    answer,

                "category":
                    category,

                "evidence_ids":
                    evidence_ids,

                "turns":
                    turns,
            })

    print(
        "Eligible memory QA:",
        len(candidates),
    )

    random.shuffle(candidates)

    if len(candidates) < N_BASE:

        raise RuntimeError(
            f"Only {len(candidates)} eligible "
            f"examples found."
        )

    return candidates[:N_BASE]


# ============================================================
# CURRENT CONTEXT
# ============================================================

def turn_text(turn):

    return (
        f"[{turn['date']}] "
        f"{turn['speaker']}: "
        f"{turn['text']}"
    )


def make_memory_needed_context(ex):

    evidence = set(
        ex["evidence_ids"]
    )

    # Current context must not contain
    # ground-truth evidence.
    non_evidence = [
        t
        for t in ex["turns"]
        if t["dia_id"] not in evidence
    ]

    context = (
        non_evidence[
            -RECENT_CONTEXT_TURNS:
        ]
    )

    return [
        turn_text(x)
        for x in context
    ]


def make_memory_not_needed_context(ex):

    evidence = set(
        ex["evidence_ids"]
    )

    indices = [
        i
        for i, t in enumerate(ex["turns"])
        if t["dia_id"] in evidence
    ]

    selected_indices = set()

    # Put evidence and neighboring turns
    # into current context.
    for idx in indices:

        start = max(
            0,
            idx - CURRENT_EVIDENCE_WINDOW,
        )

        end = min(
            len(ex["turns"]),
            idx + CURRENT_EVIDENCE_WINDOW + 1,
        )

        selected_indices.update(
            range(start, end)
        )

    context = [
        ex["turns"][i]
        for i in sorted(selected_indices)
    ]

    return [
        turn_text(x)
        for x in context
    ]


# ============================================================
# LTM
#
# one dialogue turn = one memory
# ============================================================

def make_ltm(ex, need_memory):

    evidence = set(
        ex["evidence_ids"]
    )

    memories = []

    for t in ex["turns"]:

        # For no-memory-needed condition,
        # remove GT evidence from Long-Term Memory.
        if (
            not need_memory
            and t["dia_id"] in evidence
        ):
            continue

        memories.append({
            "memory_id": t["dia_id"],
            "text": turn_text(t),
        })

    return memories


# ============================================================
# CREATE MATCHED 2000 EXAMPLES
# ============================================================

def build_paired_examples(base_examples):

    examples = []

    for ex in base_examples:

        # ------------------------------------------
        # memory-needed
        # ------------------------------------------

        examples.append({
            "id":
                ex["base_id"] + "_need",

            "base_id":
                ex["base_id"],

            "need_memory":
                True,

            "question":
                ex["question"],

            "answer":
                ex["answer"],

            "category":
                ex["category"],

            "evidence_ids":
                ex["evidence_ids"],

            "current_context":
                make_memory_needed_context(ex),

            "ltm":
                make_ltm(
                    ex,
                    need_memory=True,
                ),
        })

        # ------------------------------------------
        # memory-not-needed
        # ------------------------------------------

        examples.append({
            "id":
                ex["base_id"] + "_noneed",

            "base_id":
                ex["base_id"],

            "need_memory":
                False,

            "question":
                ex["question"],

            "answer":
                ex["answer"],

            "category":
                ex["category"],

            "evidence_ids":
                ex["evidence_ids"],

            "current_context":
                make_memory_not_needed_context(ex),

            "ltm":
                make_ltm(
                    ex,
                    need_memory=False,
                ),
        })

    return examples


# ============================================================
# EMBEDDING RETRIEVER
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Loading embedding model "
    f"{EMBED_MODEL} on {device}"
)

embedder = SentenceTransformer(
    EMBED_MODEL,
    device=device,
)


def embed_texts(texts):

    return embedder.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


# ============================================================
# QUERY MODE
# ============================================================

def current_context_text(ex):

    return "\n".join(
        ex["current_context"]
    )


def multiturn_query(ex):

    return (
        current_context_text(ex)
        + "\n\n"
        + f"Current user question: "
        + ex["question"]
    )


REWRITE_SYSTEM = """
You rewrite a user's current question into a concise,
standalone search query for retrieving relevant information
from the user's LONG-TERM conversational memory.

Use the current conversation only to resolve references,
pronouns, entities, or omitted context.

Do not answer the question.
Do not add facts.
Return only the search query.
""".strip()


def rewrite_query(ex):

    user = f"""
CURRENT CONVERSATION:
{current_context_text(ex)}

CURRENT QUESTION:
{ex["question"]}
""".strip()

    return cached_llm(
        "rewrite",
        REWRITE_SYSTEM,
        user,
        max_tokens=80,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(ex, query, top_k=TOP_K):

    texts = [
        x["text"]
        for x in ex["ltm"]
    ]

    # Small LoCoMo corpus per user:
    # simple brute-force cosine is enough.
    memory_emb = embed_texts(texts)

    q_emb = embed_texts(
        [query]
    )[0]

    scores = (
        memory_emb @ q_emb
    )

    order = np.argsort(
        -scores
    )[:top_k]

    return [
        {
            **ex["ltm"][i],
            "score":
                float(scores[i]),
        }
        for i in order
    ]


# ============================================================
# PRE-RETRIEVAL ROUTER
# ============================================================

PRE_ROUTER_SYSTEM = """
Decide whether answering the user's CURRENT question requires
retrieving information from LONG-TERM memory.

LONG-TERM memory is needed when required information comes
from conversations before the current conversation.

Do NOT retrieve long-term memory when the current conversation
already contains enough information to answer.

Return exactly one token:

YES
or
NO
""".strip()


def pre_router(ex):

    user = f"""
CURRENT CONVERSATION:
{current_context_text(ex)}

CURRENT QUESTION:
{ex["question"]}
""".strip()

    output = cached_llm(
        "pre_router",
        PRE_ROUTER_SYSTEM,
        user,
        max_tokens=8,
    )

    return (
        output
        .strip()
        .upper()
        .startswith("YES")
    )


# ============================================================
# POST-RETRIEVAL POINTWISE SELF ROUTER
# ============================================================

POST_ROUTER_SYSTEM = """
Determine whether the retrieved long-term memory item is
directly useful for answering the user's current question.

Return exactly:

YES
or
NO
""".strip()


def post_router(ex, memory):

    user = f"""
CURRENT CONVERSATION:
{current_context_text(ex)}

CURRENT QUESTION:
{ex["question"]}

RETRIEVED MEMORY:
{memory["text"]}
""".strip()

    output = cached_llm(
        "post_router",
        POST_ROUTER_SYSTEM,
        user,
        max_tokens=8,
    )

    return (
        output
        .strip()
        .upper()
        .startswith("YES")
    )


# ============================================================
# ANSWER GENERATION
# ============================================================

ANSWER_SYSTEM = """
Answer the user's current question.

Use the current conversation first.
Use retrieved long-term memories only if they are relevant.

Be concise.
Do not explain your reasoning.
""".strip()


def generate_answer(ex, memories):

    if memories:

        memory_text = "\n".join(
            f"- {x['text']}"
            for x in memories
        )

    else:

        memory_text = "(none)"

    user = f"""
CURRENT CONVERSATION:
{current_context_text(ex)}

RETRIEVED LONG-TERM MEMORIES:
{memory_text}

QUESTION:
{ex["question"]}
""".strip()

    return cached_llm(
        "answer",
        ANSWER_SYSTEM,
        user,
        max_tokens=128,
    )


# ============================================================
# QA METRICS
# ============================================================

def normalize_answer(s):

    s = str(s).lower()

    s = re.sub(
        r"[^\w\s]",
        " ",
        s,
    )

    s = re.sub(
        r"\s+",
        " ",
        s,
    ).strip()

    return s


def token_f1(pred, gold):

    p = normalize_answer(
        pred
    ).split()

    g = normalize_answer(
        gold
    ).split()

    if not p or not g:

        return float(p == g)

    pc = Counter(p)
    gc = Counter(g)

    overlap = sum(
        (pc & gc).values()
    )

    if overlap == 0:
        return 0.0

    precision = (
        overlap / len(p)
    )

    recall = (
        overlap / len(g)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def exact_match(pred, gold):

    return int(
        normalize_answer(pred)
        ==
        normalize_answer(gold)
    )


# ============================================================
# OPTIONAL LLM JUDGE
# ============================================================

JUDGE_SYSTEM = """
Judge whether the model response correctly answers the
question according to the reference answer.

Paraphrases are allowed.

Return exactly:

YES
or
NO
""".strip()


def judge_answer(ex, prediction):

    user = f"""
QUESTION:
{ex["question"]}

REFERENCE ANSWER:
{ex["answer"]}

MODEL RESPONSE:
{prediction}
""".strip()

    output = cached_llm(
        "judge",
        JUDGE_SYSTEM,
        user,
        max_tokens=8,
    )

    return int(
        output
        .strip()
        .upper()
        .startswith("YES")
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def evidence_hit(ex, retrieved):

    evidence = set(
        ex["evidence_ids"]
    )

    retrieved_ids = {
        x["memory_id"]
        for x in retrieved
    }

    return int(
        len(
            evidence & retrieved_ids
        ) > 0
    )


# ============================================================
# RUN ONE CONFIG
# ============================================================

def run_example(
    ex,
    query_mode,
    use_pre_router,
    use_post_router,
):

    # ------------------------------------------
    # Query
    # ------------------------------------------

    if query_mode == "multiturn":

        query = multiturn_query(ex)

    elif query_mode == "rewrite":

        query = rewrite_query(ex)

    else:

        raise ValueError(
            query_mode
        )

    # ------------------------------------------
    # Pre-router
    # ------------------------------------------

    router_decision = True

    if use_pre_router:

        router_decision = (
            pre_router(ex)
        )

    raw_retrieved = []

    if router_decision:

        raw_retrieved = retrieve(
            ex,
            query,
            top_k=TOP_K,
        )

    # ------------------------------------------
    # Post-router
    # ------------------------------------------

    if use_post_router:

        final_memories = [
            m
            for m in raw_retrieved
            if post_router(
                ex,
                m,
            )
        ]

    else:

        final_memories = (
            raw_retrieved
        )

    # ------------------------------------------
    # Answer
    # ------------------------------------------

    pred = generate_answer(
        ex,
        final_memories,
    )

    em = exact_match(
        pred,
        ex["answer"],
    )

    f1 = token_f1(
        pred,
        ex["answer"],
    )

    judge = judge_answer(
        ex,
        pred,
    )

    return {
        "id":
            ex["id"],

        "base_id":
            ex["base_id"],

        "need_memory":
            ex["need_memory"],

        "query_mode":
            query_mode,

        "pre_router":
            use_pre_router,

        "post_router":
            use_post_router,

        "query":
            query,

        "router_retrieve":
            router_decision,

        "raw_n_memories":
            len(raw_retrieved),

        "final_n_memories":
            len(final_memories),

        "retrieval_hit":
            (
                evidence_hit(
                    ex,
                    raw_retrieved,
                )
                if ex["need_memory"]
                else np.nan
            ),

        "post_filter_hit":
            (
                evidence_hit(
                    ex,
                    final_memories,
                )
                if ex["need_memory"]
                else np.nan
            ),

        "prediction":
            pred,

        "gold":
            ex["answer"],

        "em":
            em,

        "f1":
            f1,

        "judge_acc":
            judge,
    }


# ============================================================
# FULL ABLATION
# ============================================================

def run_benchmark(examples):

    rows = []

    configs = []

    for query_mode in [
        "multiturn",
        "rewrite",
    ]:

        for pre in [
            False,
            True,
        ]:

            for post in [
                False,
                True,
            ]:

                configs.append(
                    (
                        query_mode,
                        pre,
                        post,
                    )
                )

    for (
        query_mode,
        pre,
        post,
    ) in configs:

        print()
        print("=" * 80)

        print(
            f"query={query_mode} "
            f"pre={pre} "
            f"post={post}"
        )

        print("=" * 80)

        for ex in tqdm(examples):

            row = run_example(
                ex,
                query_mode,
                pre,
                post,
            )

            rows.append(row)

            # checkpoint
            if len(rows) % 100 == 0:

                pd.DataFrame(
                    rows
                ).to_csv(
                    "ltm_ablation_raw.csv",
                    index=False,
                )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def summarize(df):

    rows = []

    group_cols = [
        "query_mode",
        "pre_router",
        "post_router",
    ]

    for config, g in df.groupby(
        group_cols
    ):

        needed = g[
            g.need_memory
        ]

        noneeded = g[
            ~g.need_memory
        ]

        router_recall = (
            needed[
                "router_retrieve"
            ].mean()
        )

        router_specificity = (
            1
            - noneeded[
                "router_retrieve"
            ].mean()
        )

        unnecessary_rate = (
            noneeded[
                "router_retrieve"
            ].mean()
        )

        rows.append({
            "query":
                config[0],

            "pre":
                config[1],

            "post":
                config[2],

            # ------------------------
            # final QA
            # ------------------------

            "overall_judge":
                g.judge_acc.mean(),

            "needed_judge":
                needed.judge_acc.mean(),

            "not_needed_judge":
                noneeded.judge_acc.mean(),

            "overall_f1":
                g.f1.mean(),

            "needed_f1":
                needed.f1.mean(),

            "not_needed_f1":
                noneeded.f1.mean(),

            # ------------------------
            # retrieval
            # ------------------------

            f"recall@{TOP_K}":
                needed[
                    "retrieval_hit"
                ].mean(),

            "post_filter_recall":
                needed[
                    "post_filter_hit"
                ].mean(),

            # ------------------------
            # router
            # ------------------------

            "pre_router_recall":
                router_recall,

            "pre_router_specificity":
                router_specificity,

            "unnecessary_retrieval":
                unnecessary_rate,

            # ------------------------
            # context size
            # ------------------------

            "avg_raw_memories":
                g.raw_n_memories.mean(),

            "avg_final_memories":
                g.final_n_memories.mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    data = load_locomo()

    base = build_base_examples(
        data
    )

    examples = build_paired_examples(
        base
    )

    n_needed = sum(
        x["need_memory"]
        for x in examples
    )

    n_not_needed = (
        len(examples)
        - n_needed
    )

    print()
    print(
        f"memory-needed     : "
        f"{n_needed}"
    )

    print(
        f"memory-not-needed : "
        f"{n_not_needed}"
    )

    assert n_needed == 1000
    assert n_not_needed == 1000

    df = run_benchmark(
        examples
    )

    df.to_csv(
        "ltm_ablation_raw.csv",
        index=False,
    )

    summary = summarize(df)

    summary.to_csv(
        "ltm_ablation_summary.csv",
        index=False,
    )

    print()
    print("=" * 160)
    print(
        "LONG-TERM MEMORY "
        "ABLATION RESULTS"
    )
    print("=" * 160)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()
