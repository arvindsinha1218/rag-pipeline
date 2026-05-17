# ============================================================
# Imports
# ============================================================

from pathlib import Path
from typing import List, Dict, Any, Optional

import os
import json
import pickle
import uuid

import numpy as np

from dotenv import load_dotenv

from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)

from langchain_core.runnables import (
    RunnableLambda,
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

# ============================================================
# Config
# ============================================================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

RESUME_INDEX_NAME = "resume-hybrid-index"

TFIDF_PATH = "tfidf_vectorizer.pkl"

PROMPT_PATH = "prompt.yaml"

# ============================================================
# Models
# ============================================================

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

dense_model = SentenceTransformer(
    DENSE_MODEL_NAME
)

RERANK_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

rerank_model = CrossEncoder(
    RERANK_MODEL_NAME
)

# ============================================================
# Pinecone
# ============================================================

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index(RESUME_INDEX_NAME)

# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=openai_api_key
)

# ============================================================
# Multi Query Expansion
# ============================================================

def multi_query_ext(user_query):

    prompt = f"""
    Imagine you are helping retrieve the most suitable candidate resumes.

    Write 3 variations based on the user query.

    Return only the rewritten queries.

    query:
    {user_query}
    """

    resp = llm.invoke(prompt)

    content = getattr(resp, "content", resp)

    final_query = user_query + "\n" + content

    return final_query

# ============================================================
# Hybrid Search
# ============================================================

def hybrid_query(
    query_text: str,
    alpha: float = 0.5,
    top_k: int = 8
):

    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    q_dense = dense_model.encode(
        [query_text],
        normalize_embeddings=True
    )[0]

    q_dense = (
        np.asarray(q_dense, dtype=float)
        * (1.0 - alpha)
    ).tolist()

    q_sparse_csr = (
        vectorizer.transform([query_text]).tocoo()
    )

    if q_sparse_csr.nnz == 0:

        q_sparse = {
            "indices": [0],
            "values": [0.0]
        }

    else:

        q_sparse = {
            "indices": q_sparse_csr.col.tolist(),
            "values": (
                q_sparse_csr.data.astype(float)
                * alpha
            ).tolist(),
        }

    res = index.query(
        vector=q_dense,
        sparse_vector=q_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    out = []

    for m in res.get("matches", []):

        md = m.get("metadata", {}) or {}

        text = md.get("text", "") or ""

        preview = " ".join(
            text.split()[:120]
        )

        out.append(
            {
                "id": m["id"],
                "score": float(m["score"]),
                "preview": preview,
                "metadata": md,
            }
        )

    return out

# ============================================================
# Reranking
# ============================================================

def reranker_crossencoder(query, results):

    pairs = [
        (query, r["preview"])
        for r in results
    ]

    scores = rerank_model.predict(pairs)

    rescored = []

    for r, s in zip(results, scores):

        r2 = dict(r)

        r2["rerank_cross_score"] = float(s)

        rescored.append(r2)

    return sorted(
        rescored,
        key=lambda x: x["rerank_cross_score"],
        reverse=True
    )

# ============================================================
# Prompt Loader
# ============================================================

def load_prompt(path=PROMPT_PATH):

    import yaml

    text = Path(path).read_text(
        encoding="utf-8"
    )

    cfg = yaml.safe_load(text) or {}

    return cfg

# ============================================================
# Pydantic Models
# ============================================================

class ResumeItem(BaseModel):

    resume_id: str

    filename: Optional[str] = None

    link: Optional[str] = None

    jd_relevance: float

    profile_summary: str

    key_skills: List[str] = []

    risks_or_flags: List[str] = []

class ResumeResponse(BaseModel):

    query: str

    resumes: List[ResumeItem]

resume_parser = PydanticOutputParser(
    pydantic_object=ResumeResponse
)

# ============================================================
# Prompt Rendering
# ============================================================

def render_prompt(
    cfg: dict,
    query: str,
    sources: str
):

    format_instructions = (
        resume_parser.get_format_instructions()
    )

    vars_all = dict(
        cfg.get("vars", {}),
        query=query,
        sources=sources,
        format_instructions=format_instructions,
    )

    return cfg["template"].format(**vars_all)

# ============================================================
# Helper Functions
# ============================================================

def _start(user_q):

    return {
        "query": user_q
    }

def _extract_answer(answer_obj):

    content = getattr(
        answer_obj,
        "content",
        answer_obj
    )

    return str(content).strip()

# ============================================================
# Final Retrieval Chain
# ============================================================

final_chain = (

    RunnableLambda(_start)

    .assign(
        multi_query=
        RunnableLambda(
            lambda q:
            multi_query_ext(q["query"])
        )
    )

    .assign(
        results=
        RunnableLambda(
            lambda q:
            hybrid_query(q["multi_query"])
        )
    )

    .assign(
        re_ranked_results=
        RunnableLambda(
            lambda q:
            reranker_crossencoder(
                q["query"],
                q["results"]
            )
        )
    )

    .assign(
        prompt_temp=
        RunnableLambda(
            lambda q:
            load_prompt(PROMPT_PATH)
        )
    )

    .assign(
        render_prompt=
        RunnableLambda(
            lambda q:
            render_prompt(
                q["prompt_temp"],
                q["multi_query"],
                q["re_ranked_results"]
            )
        )
    )

    .assign(
        output=
        RunnableLambda(
            lambda q:
            llm.invoke(q["render_prompt"])
        )
    )

    .assign(
        answer_raw=
        RunnableLambda(
            lambda q:
            _extract_answer(q["output"])
        )
    )

    .assign(
        answer_parsed=
        RunnableLambda(
            lambda d:
            resume_parser.parse(
                d["answer_raw"]
            )
        )
    )
)

# ============================================================
# Memory
# ============================================================

_session_store = {}

_history_store = {}

def get_history(session_id):

    if session_id not in _history_store:

        _history_store[session_id] = (
            ChatMessageHistory()
        )

    return _history_store[session_id]

# ============================================================
# QA Prompt
# ============================================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant answering questions "
            "about candidate resumes.\n"
            "Use ONLY the Resume JSON as ground truth.\n"
            "If unavailable say you don't know.\n\n"
            "Initial query:\n{initial_query}\n\n"
            "Resume JSON:\n{resume_json}"
        ),

        MessagesPlaceholder("history"),

        ("human", "{question}")
    ]
)

# ============================================================
# QA Input Prep
# ============================================================

def _prep_qa_inputs(d):

    resume_json = d["resume_json"]

    initial_query = resume_json.get(
        "query",
        ""
    )

    return {
        "question": d["question"],
        "history": d.get("history", []),
        "resume_json": json.dumps(
            resume_json,
            ensure_ascii=False
        ),
        "initial_query": initial_query,
    }

# ============================================================
# QA Chain
# ============================================================

resume_qa_chain = (

    RunnableLambda(_prep_qa_inputs)

    | qa_prompt

    | llm

    | StrOutputParser()
)

# ============================================================
# QA Agent
# ============================================================

qa_agent = RunnableWithMessageHistory(
    resume_qa_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ============================================================
# Main User Function
# ============================================================

def handle_user_message(
    user_input: str,
    session_id: str = None
):

    if session_id is None:
        session_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # First Query -> Retrieval Chain
    # --------------------------------------------------------

    if session_id not in _session_store:

        out = final_chain.invoke(user_input)

        parsed: ResumeResponse = (
            out["answer_parsed"]
        )

        _session_store[session_id] = (
            parsed.model_dump()
        )

        _history_store[session_id] = (
            ChatMessageHistory()
        )

        resumes = (
            _session_store[session_id]
            .get("resumes", [])
        )

        lines = [
            f"Parsed {len(resumes)} resumes.\n",
            "Shortlisted resumes:\n"
        ]

        for r in resumes:

            filename = (
                r.get("filename")
                or r["resume_id"]
            )

            rel = float(
                r.get("jd_relevance", 0.0)
            )

            lines.append(
                f"- {filename} "
                f"(relevance = {rel:.2f})"
            )

        return {
            "session_id": session_id,
            "response": "\n".join(lines)
        }

    # --------------------------------------------------------
    # Follow-up Questions -> QA Agent
    # --------------------------------------------------------

    cfg = {
        "configurable": {
            "session_id": session_id
        }
    }

    response = qa_agent.invoke(
        {
            "question": user_input,
            "resume_json":
                _session_store[session_id]
        },
        config=cfg
    )

    return {
        "session_id": session_id,
        "response": response
    }