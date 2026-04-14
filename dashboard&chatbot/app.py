from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Entry-Level Camera Review Dashboard",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"

BRAND_MAP = {
    "佳能": "Canon",
    "富士": "Fujifilm",
    "尼康": "Nikon",
    "索尼": "Sony",
}

TOPIC_ORDER = [
    "Design & Portability",
    "Performance & Quality",
    "Appearance & Satisfaction",
    "Logistics & Service",
]
SENTIMENT_ORDER = ["positive", "neutral", "negative"]
BEHAVIOR_ORDER = ["referral", "community", "discussion"]
SCORE_COL_MAP = {
    "positive": "positive_score",
    "neutral": "neutral_score",
    "negative": "negative_score",
    "referral": "referral_score",
    "community": "community_score",
    "discussion": "discussion_score",
}
KEY_COLS = ["用户ID", "时间", "商品规格", "评论内容", "页面标题", "品牌"]

POSITIVE_KEYWORDS = [
    "好", "棒", "赞", "喜欢", "满意", "推荐", "不错", "很好", "优秀", "完美", "惊喜", "感动", "放心", "友好",
    "清晰", "高清", "可爱", "小巧", "漂亮", "美观", "实用", "方便", "快捷", "迅速", "正品", "信赖", "值得",
    "超值", "划算", "优惠", "性价比", "厉害", "专业", "强大", "稳定", "顺畅", "流畅", "舒服", "舒适",
]
NEGATIVE_KEYWORDS = [
    "差", "坏", "烂", "失望", "问题", "糟糕", "垃圾", "后悔", "缺陷", "不足", "麻烦", "慢", "卡顿", "延迟",
    "发热", "故障", "损坏", "破损", "错", "误", "假货", "昂贵", "贵", "不值", "浪费", "难用", "复杂", "笨重",
    "粗糙", "模糊", "暗淡", "失真", "噪音", "异响", "震动", "过热",
]

LABEL_SOURCE_MAP = {
    "Rule-based label": "rule_sentiment",
    "Model-predicted label": "predicted_sentiment",
}

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_TOP_K = 5
DEFAULT_BIGMODEL_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_BIGMODEL_MODEL = "glm-4.7-flash"

CHATBOT_SYSTEM_PROMPT = """
You are an analytical assistant for a camera review intelligence dashboard.
Answer only from the retrieved evidence.
Do not invent unsupported facts or metrics.
Be concise, comparative, and user-oriented.
If the evidence is limited, say so clearly.
""".strip()

CHATBOT_ANSWER_INSTRUCTION = """
Write the answer in English.

Use this structure:
1. Direct Answer
2. Explanation
3. Supporting Evidence
4. If evidence is weak or incomplete

Rules:
- Keep the answer grounded in the provided evidence only.
- Do not mention embeddings, vector search, or internal retrieval mechanics.
- Do not quote long review text unless necessary.
""".strip()


st.markdown(
    """
    <style>
    .small-note {font-size: 0.92rem; color: #6b7280;}
    .section-title {margin-top: 0.5rem; margin-bottom: 0.25rem;}
    .block-container {padding-top: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)



def get_kb_dir() -> Path | None:
    candidates = [
        APP_DIR / "kb_output",
        APP_DIR.parent / "kb_output",
        Path.cwd() / "kb_output",
    ]
    for path in candidates:
        if (path / "chunks.jsonl").exists() and (path / "embeddings.npy").exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_kb_chunks(kb_dir_str: str) -> list[dict[str, Any]]:
    path = Path(kb_dir_str) / "chunks.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_kb_embeddings(kb_dir_str: str) -> np.ndarray:
    return np.load(Path(kb_dir_str) / "embeddings.npy")


@st.cache_data(show_spinner=False)
def load_kb_meta(kb_dir_str: str) -> dict[str, Any]:
    path = Path(kb_dir_str) / "index_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"backend": "numpy", "metric": "inner_product_on_normalized_embeddings"}


import torch
from transformers import AutoTokenizer, AutoModel


@st.cache_resource
def get_e5_components(model_name: str):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_chat_query(query: str) -> np.ndarray:
    tokenizer, model = get_e5_components(EMBED_MODEL_NAME)

    text = f"query: {query}" if "e5" in EMBED_MODEL_NAME.lower() else query
    batch = tokenizer(
        [text],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


def retrieve_chatbot_hits(question: str, kb_dir: Path, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    qvec = encode_chat_query(question)
    meta = load_kb_meta(str(kb_dir))
    backend = meta.get("backend", "numpy")
    chunks = load_kb_chunks(str(kb_dir))

    if backend == "faiss" and (kb_dir / "faiss.index").exists():
        try:
            import faiss  # type: ignore

            index = faiss.read_index(str(kb_dir / "faiss.index"))
            scores, ids = index.search(qvec, top_k)
            scores = scores[0].tolist()
            ids = ids[0].tolist()
        except Exception:
            embeddings = load_kb_embeddings(str(kb_dir))
            sims = embeddings @ qvec[0]
            ids = np.argsort(-sims)[:top_k].tolist()
            scores = sims[ids].tolist()
    else:
        embeddings = load_kb_embeddings(str(kb_dir))
        sims = embeddings @ qvec[0]
        ids = np.argsort(-sims)[:top_k].tolist()
        scores = sims[ids].tolist()

    hits: list[dict[str, Any]] = []
    for score, idx in zip(scores, ids):
        idx = int(idx)
        if 0 <= idx < len(chunks):
            item = dict(chunks[idx])
            item["score"] = float(score)
            hits.append(item)
    return hits


def build_chatbot_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[Evidence {i}]\n"
            f"Title: {h.get('title', 'Untitled')}\n"
            f"Source type: {h.get('source_type', 'unknown')}\n"
            f"Source file: {h.get('source_file', 'unknown')}\n"
            f"Brand: {h.get('brand', 'ALL')}\n"
            f"Topic: {h.get('topic', 'ALL')}\n"
            f"Sentiment: {h.get('sentiment', 'ALL')}\n"
            f"Similarity score: {h.get('score', 0.0):.4f}\n"
            f"Content:\n{h.get('text', '')}"
        )
    context = "\n\n".join(blocks)
    return f"""
{CHATBOT_ANSWER_INSTRUCTION}

User question:
{question}

Retrieved evidence:
{context}
""".strip()


def extract_llm_text(data: dict[str, Any]) -> str:
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    finish_reason = choice.get("finish_reason", "")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                text_parts.append(block.strip())
            elif isinstance(block, dict):
                for key in ["text", "content"]:
                    value = block.get(key)
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())
        merged = "\n".join(text_parts).strip()
        if merged:
            return merged

    reasoning_content = message.get("reasoning_content", "")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    return f"No usable text returned by the model. finish_reason={finish_reason}; raw={json.dumps(data, ensure_ascii=False)[:1500]}"


def call_bigmodel_answer(question: str, hits: list[dict[str, Any]], api_key: str, model_name: str) -> tuple[str, str]:
    user_prompt = build_chatbot_prompt(question, hits)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "stream": False,
        "response_format": {"type": "text"},
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        os.getenv("BIGMODEL_API_URL", DEFAULT_BIGMODEL_URL),
        headers=headers,
        json=payload,
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    data = response.json()
    return extract_llm_text(data), user_prompt



def format_score(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def ordered_topics(values: list[str]) -> list[str]:
    present = [v for v in values if pd.notna(v)]
    return [t for t in TOPIC_ORDER if t in present] + sorted([t for t in present if t not in TOPIC_ORDER])



def ordered_brands(values: list[str]) -> list[str]:
    preferred = ["Canon", "Fujifilm", "Nikon", "Sony"]
    present = [v for v in values if pd.notna(v)]
    return [b for b in preferred if b in present] + sorted([b for b in present if b not in preferred])



def build_average_score_df(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        score_col = SCORE_COL_MAP[label]
        rows.append({"score_type": label, "average_score": frame[score_col].mean()})
    return pd.DataFrame(rows)



def default_highlight_metric(selected_sentiments: list[str]) -> str:
    if len(selected_sentiments) == 1 and selected_sentiments[0] in SCORE_COL_MAP:
        return SCORE_COL_MAP[selected_sentiments[0]]
    if "negative" in selected_sentiments and "positive" not in selected_sentiments:
        return "negative_score"
    return "positive_score"



def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "时间" in out.columns:
        dt = pd.to_datetime(out["时间"], errors="coerce")
        out["时间"] = dt.dt.strftime("%Y-%m-%d").fillna(out["时间"].astype(str).str.strip())
    for col in KEY_COLS:
        if col in out.columns and col != "时间":
            out[col] = out[col].astype(str).str.strip()
    return out



def find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None



def get_source_paths() -> dict[str, Path | None]:
    project_roots = [
        APP_DIR,
        APP_DIR.parent,
        APP_DIR / "8017_project_test-main",
        APP_DIR.parent / "8017_project_test-main",
        Path.cwd(),
        Path.cwd() / "8017_project_test-main",
        Path("/mnt/data/project_unz/8017_project_test-main"),
    ]

    merged_candidates = [
        APP_DIR / "data" / "camera_sentiment_label.xlsx",
        APP_DIR / "camera_sentiment_label.xlsx",
        APP_DIR.parent / "camera_sentiment_label.xlsx",
    ]
    sent_candidates = []
    topic_candidates = []

    for root in project_roots:
        merged_candidates.append(root / "Brand Analysis part3&5" / "PART3_User Preference Analysis & Competitive Analysis" / "camera_sentiment_label.xlsx")
        sent_candidates.append(root / "Sentitivies Analysis" / "camera_reviews_sentiment_analysis.xlsx")
        topic_candidates.append(root / "Topic Classification" / "camera_labeled.csv")

    return {
        "merged": find_existing_path(merged_candidates),
        "sentiment": find_existing_path(sent_candidates),
        "topic": find_existing_path(topic_candidates),
    }



def classify_rule_sentiment(text: str) -> str:
    text = str(text)
    pos_count = sum(1 for word in POSITIVE_KEYWORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text)
    if pos_count > neg_count:
        return "positive"
    if neg_count > pos_count:
        return "negative"
    return "neutral"



def add_rule_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_sentiment"] = out["评论内容"].astype(str).apply(classify_rule_sentiment)
    out["rule_positive_hits"] = out["评论内容"].astype(str).apply(
        lambda x: sum(1 for word in POSITIVE_KEYWORDS if word in x)
    )
    out["rule_negative_hits"] = out["评论内容"].astype(str).apply(
        lambda x: sum(1 for word in NEGATIVE_KEYWORDS if word in x)
    )
    return out


@st.cache_data(show_spinner=False)
def load_unified_data() -> tuple[pd.DataFrame, str]:
    paths = get_source_paths()

    if paths["merged"] is not None:
        df = pd.read_excel(paths["merged"])
        source_note = f"Loaded unified table from: {paths['merged']}"
    elif paths["sentiment"] is not None and paths["topic"] is not None:
        sent_df = pd.read_excel(paths["sentiment"])
        topic_df = pd.read_csv(paths["topic"])

        sent_df = normalize_key_columns(sent_df)
        topic_df = normalize_key_columns(topic_df)

        topic_extra_cols = [col for col in ["segmented_text", "segmented_list", "topic_id", "topic_label"] if col in topic_df.columns]
        df = sent_df.merge(topic_df[KEY_COLS + topic_extra_cols], on=KEY_COLS, how="left")
        source_note = "Built unified table at runtime from sentiment + topic files."
    else:
        raise FileNotFoundError(
            "Could not find camera_sentiment_label.xlsx or the pair camera_reviews_sentiment_analysis.xlsx + camera_labeled.csv."
        )

    df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
    df["品牌_en"] = df["品牌"].map(BRAND_MAP).fillna(df["品牌"].astype(str))
    df["topic_label"] = df["topic_label"].fillna("Unassigned")
    df["predicted_sentiment"] = df["predicted_sentiment"].fillna("unknown").astype(str).str.lower()
    df = add_rule_sentiment_columns(df)

    numeric_cols = [
        "positive_score",
        "neutral_score",
        "negative_score",
        "referral_score",
        "community_score",
        "discussion_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["时间"].notna().any():
        df["month"] = df["时间"].dt.to_period("M").astype(str)
        df["year"] = df["时间"].dt.year
    else:
        df["month"] = "Unknown"
        df["year"] = pd.NA

    df["comment_length"] = df["评论内容"].astype(str).str.len()
    df["net_sentiment_score"] = df["positive_score"].fillna(0) - df["negative_score"].fillna(0)
    df["score_implied_sentiment"] = (
        df[["positive_score", "neutral_score", "negative_score"]]
        .idxmax(axis=1)
        .str.replace("_score", "", regex=False)
    )
    return df, source_note



def make_distribution_df(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    counts = frame[label_col].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).reset_index()
    counts.columns = ["sentiment", "count"]
    return counts


try:
    df, source_note = load_unified_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

min_date = df["时间"].min()
max_date = df["时间"].max()

st.sidebar.title("📷 Dashboard Controls")
page = st.sidebar.radio(
    "Go to section",
    [
        "Overview",
        "Brand Comparison",
        "Topic & Sentiment",
        "Trend Analysis",
        "Review Explorer",
        "Chatbot",
        "Method Notes",
    ],
)

label_source_name = st.sidebar.radio(
    "Sentiment label source used for filtering",
    list(LABEL_SOURCE_MAP.keys()),
    index=0,
)
active_label_col = LABEL_SOURCE_MAP[label_source_name]

brand_options = ordered_brands(df["品牌_en"].dropna().unique().tolist())
topic_options = ordered_topics(df["topic_label"].dropna().unique().tolist())
sentiment_options = SENTIMENT_ORDER.copy()

selected_brands = st.sidebar.multiselect("Brand", brand_options, default=brand_options)
selected_topics = st.sidebar.multiselect("Topic", topic_options, default=topic_options)
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_options, default=sentiment_options)

if pd.notna(min_date) and pd.notna(max_date):
    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
else:
    selected_dates = None

keyword = st.sidebar.text_input("Keyword search in review text", placeholder="e.g. 轻便 / 画质 / 物流")

filtered = df.copy()
filtered = filtered[filtered["品牌_en"].isin(selected_brands)]
filtered = filtered[filtered["topic_label"].isin(selected_topics)]
filtered = filtered[filtered[active_label_col].isin(selected_sentiments)]

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    filtered = filtered[(filtered["时间"] >= start_date) & (filtered["时间"] <= end_date)]

if keyword:
    mask = filtered["评论内容"].astype(str).str.contains(keyword, case=False, na=False)
    filtered = filtered[mask]


total_reviews = len(filtered)
unique_users = filtered["用户ID"].nunique()
avg_positive = filtered["positive_score"].mean()
avg_neutral = filtered["neutral_score"].mean()
avg_negative = filtered["negative_score"].mean()
avg_net = filtered["net_sentiment_score"].mean()

st.title("Entry-Level Camera Review Intelligence Dashboard")
st.caption(
    "Dashboard table = raw reviews + topic labels + model sentiment probabilities + rule-based sentiment labels."
)
st.caption(f"Current filter uses: **{label_source_name}**")

if total_reviews == 0:
    selected_label_text = ", ".join(selected_sentiments) if selected_sentiments else "none"
    st.warning(
        f"No data available under the current filters. This can happen when the selected label source does not have matching rows for '{selected_label_text}'."
    )
    st.stop()


if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered Reviews", f"{total_reviews:,}")
    c2.metric("Unique Users", f"{unique_users:,}")
    c3.metric("Avg Positive Score", format_score(avg_positive))
    c4.metric("Avg Net Sentiment", format_score(avg_net))

    col1, col2 = st.columns([1.1, 1])
    with col1:
        brand_counts = (
            filtered.groupby("品牌_en", as_index=False)
            .size()
            .rename(columns={"size": "review_count"})
            .sort_values("review_count", ascending=False)
        )
        fig_brand = px.bar(
            brand_counts,
            x="品牌_en",
            y="review_count",
            text="review_count",
            title="Review Volume by Brand",
        )
        fig_brand.update_layout(xaxis_title="Brand", yaxis_title="Reviews")
        st.plotly_chart(fig_brand, use_container_width=True)

    with col2:
        topic_counts = (
            filtered.groupby("topic_label", as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        topic_counts["topic_label"] = pd.Categorical(
            topic_counts["topic_label"],
            categories=ordered_topics(topic_counts["topic_label"].tolist()),
            ordered=True,
        )
        topic_counts = topic_counts.sort_values("topic_label")
        fig_topic = px.pie(
            topic_counts,
            names="topic_label",
            values="count",
            hole=0.45,
            title="Topic Distribution",
        )
        st.plotly_chart(fig_topic, use_container_width=True)

    col3, col4 = st.columns([1.15, 1])
    with col3:
        monthly_reviews = (
            filtered.groupby("month", as_index=False)
            .size()
            .rename(columns={"size": "review_count"})
            .sort_values("month")
        )
        fig_monthly = px.line(
            monthly_reviews,
            x="month",
            y="review_count",
            markers=True,
            title="Monthly Review Trend",
        )
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Reviews")
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col4:
        sentiment_df = build_average_score_df(filtered, SENTIMENT_ORDER)
        fig_sentiment_scores = px.bar(
            sentiment_df,
            x="score_type",
            y="average_score",
            title="Average Sentiment Scores",
            text="average_score",
        )
        fig_sentiment_scores.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_sentiment_scores.update_layout(yaxis_title="Average score", xaxis_title="")
        st.plotly_chart(fig_sentiment_scores, use_container_width=True)

    col5, col6 = st.columns([1, 1])
    with col5:
        behavior_df = build_average_score_df(filtered, BEHAVIOR_ORDER)
        fig_behavior_scores = px.bar(
            behavior_df,
            x="score_type",
            y="average_score",
            title="Average Referral / Community / Discussion Scores",
            text="average_score",
        )
        fig_behavior_scores.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_behavior_scores.update_layout(yaxis_title="Average score", xaxis_title="")
        st.plotly_chart(fig_behavior_scores, use_container_width=True)

    with col6:
        comparison_rows = []
        for label_col, source_name in [("rule_sentiment", "Rule-based"), ("predicted_sentiment", "Model-predicted")]:
            dist = make_distribution_df(filtered, label_col)
            dist["source"] = source_name
            comparison_rows.append(dist)
        label_compare_df = pd.concat(comparison_rows, ignore_index=True)
        fig_compare = px.bar(
            label_compare_df,
            x="sentiment",
            y="count",
            color="source",
            barmode="group",
            title="Rule-based vs Model-predicted Sentiment Counts",
            category_orders={"sentiment": SENTIMENT_ORDER},
        )
        fig_compare.update_layout(xaxis_title="Sentiment label", yaxis_title="Reviews")
        st.plotly_chart(fig_compare, use_container_width=True)

    col7, col8 = st.columns([1.05, 0.95])
    with col7:
        agreement = pd.crosstab(filtered["rule_sentiment"], filtered["predicted_sentiment"]).reindex(
            index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0
        )
        agree_fig = go.Figure(
            data=go.Heatmap(
                z=agreement.values,
                x=list(agreement.columns),
                y=list(agreement.index),
                hovertemplate="Rule=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
            )
        )
        agree_fig.update_layout(title="Rule vs Predicted Agreement Matrix", xaxis_title="Predicted", yaxis_title="Rule")
        st.plotly_chart(agree_fig, use_container_width=True)

    with col8:
        rule_counts = make_distribution_df(filtered, "rule_sentiment").rename(columns={"count": "rule_count"})
        pred_counts = make_distribution_df(filtered, "predicted_sentiment").rename(columns={"count": "predicted_count"})
        summary = rule_counts.merge(pred_counts, on="sentiment")
        st.subheader("Label comparison table")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown(
            '<p class="small-note">The sidebar filter uses the selected label source, but this page shows both so you can compare them directly.</p>',
            unsafe_allow_html=True,
        )

    with st.expander("Show summary table"):
        summary_table = (
            filtered.groupby("品牌_en")
            .agg(
                reviews=("评论内容", "count"),
                avg_positive=("positive_score", "mean"),
                avg_negative=("negative_score", "mean"),
                avg_net=("net_sentiment_score", "mean"),
                rule_negative_share=("rule_sentiment", lambda s: (s == "negative").mean()),
            )
            .reset_index()
            .sort_values("reviews", ascending=False)
        )
        st.dataframe(summary_table, use_container_width=True)


elif page == "Brand Comparison":
    brand_metrics = (
        filtered.groupby("品牌_en")
        .agg(
            reviews=("评论内容", "count"),
            avg_positive=("positive_score", "mean"),
            avg_neutral=("neutral_score", "mean"),
            avg_negative=("negative_score", "mean"),
            avg_net=("net_sentiment_score", "mean"),
            avg_referral=("referral_score", "mean"),
            avg_community=("community_score", "mean"),
            avg_discussion=("discussion_score", "mean"),
        )
        .reset_index()
    )
    brand_metrics["品牌_en"] = pd.Categorical(
        brand_metrics["品牌_en"], categories=ordered_brands(brand_metrics["品牌_en"].tolist()), ordered=True
    )
    brand_metrics = brand_metrics.sort_values("品牌_en")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig_metric = px.bar(
            brand_metrics,
            x="品牌_en",
            y=["avg_positive", "avg_neutral", "avg_negative"],
            barmode="group",
            title="Average Sentiment Probabilities by Brand",
        )
        fig_metric.update_layout(xaxis_title="Brand", yaxis_title="Average score", legend_title="Metric")
        st.plotly_chart(fig_metric, use_container_width=True)

    with c2:
        label_mix = (
            filtered.groupby(["品牌_en", active_label_col], as_index=False)
            .size()
            .rename(columns={"size": "count", active_label_col: "sentiment"})
        )
        label_mix["品牌_en"] = pd.Categorical(
            label_mix["品牌_en"], categories=ordered_brands(label_mix["品牌_en"].tolist()), ordered=True
        )
        label_mix["sentiment"] = pd.Categorical(label_mix["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
        label_mix = label_mix.sort_values(["品牌_en", "sentiment"])
        fig_label_mix = px.bar(
            label_mix,
            x="品牌_en",
            y="count",
            color="sentiment",
            barmode="stack",
            title=f"{label_source_name} Distribution by Brand",
            category_orders={"sentiment": SENTIMENT_ORDER},
        )
        fig_label_mix.update_layout(xaxis_title="Brand", yaxis_title="Reviews")
        st.plotly_chart(fig_label_mix, use_container_width=True)

    c3, c4 = st.columns([1.15, 1])
    with c3:
        topic_mix = (
            filtered.groupby(["品牌_en", "topic_label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        topic_mix["品牌_en"] = pd.Categorical(
            topic_mix["品牌_en"], categories=ordered_brands(topic_mix["品牌_en"].tolist()), ordered=True
        )
        topic_mix["topic_label"] = pd.Categorical(
            topic_mix["topic_label"], categories=ordered_topics(topic_mix["topic_label"].tolist()), ordered=True
        )
        topic_mix = topic_mix.sort_values(["品牌_en", "topic_label"])
        fig_mix = px.bar(
            topic_mix,
            x="品牌_en",
            y="count",
            color="topic_label",
            barmode="stack",
            title="Topic Mix by Brand",
        )
        fig_mix.update_layout(xaxis_title="Brand", yaxis_title="Review count", legend_title="Topic")
        st.plotly_chart(fig_mix, use_container_width=True)

    with c4:
        share = brand_metrics.copy()
        share["share_of_voice"] = share["reviews"] / share["reviews"].sum()
        fig_share = px.bar(
            share,
            x="品牌_en",
            y="share_of_voice",
            text="share_of_voice",
            title="Share of Voice by Brand",
        )
        fig_share.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_share.update_layout(xaxis_title="Brand", yaxis_title="Share of total reviews")
        st.plotly_chart(fig_share, use_container_width=True)

    c5, c6 = st.columns([1.05, 0.95])
    with c5:
        pivot = (
            filtered.pivot_table(index="品牌_en", columns="topic_label", values="positive_score", aggfunc="mean")
            .reindex(index=ordered_brands(filtered["品牌_en"].dropna().unique().tolist()))
            .reindex(columns=ordered_topics(filtered["topic_label"].dropna().unique().tolist()))
        )
        heat = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                hovertemplate="Brand=%{y}<br>Topic=%{x}<br>Avg positive=%{z:.3f}<extra></extra>",
            )
        )
        heat.update_layout(title="Brand × Topic Positive Score Heatmap", xaxis_title="Topic", yaxis_title="Brand")
        st.plotly_chart(heat, use_container_width=True)

    with c6:
        behavior_by_brand = brand_metrics.melt(
            id_vars="品牌_en",
            value_vars=["avg_referral", "avg_community", "avg_discussion"],
            var_name="metric",
            value_name="average_score",
        )
        fig_behavior_brand = px.bar(
            behavior_by_brand,
            x="品牌_en",
            y="average_score",
            color="metric",
            barmode="group",
            title="Behavioral Signals by Brand",
        )
        fig_behavior_brand.update_layout(xaxis_title="Brand", yaxis_title="Average score", legend_title="Metric")
        st.plotly_chart(fig_behavior_brand, use_container_width=True)

    st.subheader("Brand KPI Tables")
    left, right = st.columns(2)
    with left:
        st.markdown("**Sentiment metrics**")
        sentiment_kpi = brand_metrics[["品牌_en", "reviews", "avg_positive", "avg_neutral", "avg_negative", "avg_net"]]
        st.dataframe(
            sentiment_kpi.style.format(
                {
                    "avg_positive": "{:.3f}",
                    "avg_neutral": "{:.3f}",
                    "avg_negative": "{:.3f}",
                    "avg_net": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=215,
        )
    with right:
        st.markdown(f"**{label_source_name} counts**")
        label_table = (
            filtered.groupby(["品牌_en", active_label_col], as_index=False)
            .size()
            .rename(columns={"size": "count", active_label_col: "sentiment"})
            .pivot(index="品牌_en", columns="sentiment", values="count")
            .reindex(index=ordered_brands(filtered["品牌_en"].dropna().unique().tolist()), columns=SENTIMENT_ORDER)
            .fillna(0)
            .astype(int)
        )
        st.dataframe(label_table, use_container_width=True)


elif page == "Topic & Sentiment":
    c1, c2 = st.columns([1.1, 1])
    with c1:
        topic_volume = (
            filtered.groupby("topic_label", as_index=False)
            .size()
            .rename(columns={"size": "review_count"})
        )
        topic_volume["topic_label"] = pd.Categorical(
            topic_volume["topic_label"], categories=ordered_topics(topic_volume["topic_label"].tolist()), ordered=True
        )
        topic_volume = topic_volume.sort_values("topic_label")
        fig_topic_vol = px.bar(
            topic_volume,
            x="topic_label",
            y="review_count",
            text="review_count",
            title="Review Volume by Topic",
        )
        fig_topic_vol.update_layout(xaxis_title="Topic", yaxis_title="Reviews")
        st.plotly_chart(fig_topic_vol, use_container_width=True)

    with c2:
        topic_sent = (
            filtered.groupby("topic_label", as_index=False)[["positive_score", "neutral_score", "negative_score", "net_sentiment_score"]]
            .mean()
        )
        topic_sent["topic_label"] = pd.Categorical(
            topic_sent["topic_label"], categories=ordered_topics(topic_sent["topic_label"].tolist()), ordered=True
        )
        topic_sent = topic_sent.sort_values("topic_label")
        fig_topic_sent = px.bar(
            topic_sent,
            x="topic_label",
            y=["positive_score", "neutral_score", "negative_score"],
            barmode="group",
            title="Average Sentiment Scores by Topic",
        )
        fig_topic_sent.update_layout(xaxis_title="Topic", yaxis_title="Average score", legend_title="Metric")
        st.plotly_chart(fig_topic_sent, use_container_width=True)

    c3, c4 = st.columns([1.15, 1])
    with c3:
        mention_pivot = (
            filtered.groupby(["品牌_en", "topic_label"], as_index=False)
            .size()
            .rename(columns={"size": "review_count"})
            .pivot(index="品牌_en", columns="topic_label", values="review_count")
            .fillna(0)
            .reindex(index=ordered_brands(filtered["品牌_en"].dropna().unique().tolist()))
            .reindex(columns=ordered_topics(filtered["topic_label"].dropna().unique().tolist()))
        )
        mention_heat = go.Figure(
            data=go.Heatmap(
                z=mention_pivot.values,
                x=list(mention_pivot.columns),
                y=list(mention_pivot.index),
                hovertemplate="Brand=%{y}<br>Topic=%{x}<br>Reviews=%{z}<extra></extra>",
            )
        )
        mention_heat.update_layout(title="Brand × Topic Mention Heatmap", xaxis_title="Topic", yaxis_title="Brand")
        st.plotly_chart(mention_heat, use_container_width=True)

    with c4:
        pos_pivot = (
            filtered.pivot_table(index="品牌_en", columns="topic_label", values="positive_score", aggfunc="mean")
            .fillna(0)
            .reindex(index=ordered_brands(filtered["品牌_en"].dropna().unique().tolist()))
            .reindex(columns=ordered_topics(filtered["topic_label"].dropna().unique().tolist()))
        )
        pos_heat = go.Figure(
            data=go.Heatmap(
                z=pos_pivot.values,
                x=list(pos_pivot.columns),
                y=list(pos_pivot.index),
                hovertemplate="Brand=%{y}<br>Topic=%{x}<br>Avg positive=%{z:.3f}<extra></extra>",
            )
        )
        pos_heat.update_layout(title="Brand × Topic Positive Score Heatmap", xaxis_title="Topic", yaxis_title="Brand")
        st.plotly_chart(pos_heat, use_container_width=True)

    st.subheader("Representative comments")
    brand_pick = st.selectbox("Choose a brand for sample comments", brand_options, index=0)
    topic_pick = st.selectbox("Choose a topic", topic_options, index=0)
    sample_metric = st.selectbox(
        "Rank sample comments by",
        ["positive_score", "negative_score", "neutral_score", "discussion_score", "referral_score"],
        index=0,
    )
    sample_pool = filtered[(filtered["品牌_en"] == brand_pick) & (filtered["topic_label"] == topic_pick)].copy()
    sample_pool = sample_pool.sort_values(sample_metric, ascending=False).head(5)

    if sample_pool.empty:
        st.info("No representative comments found for this brand-topic combination under the current filters.")
    else:
        for i, row in enumerate(sample_pool.itertuples(index=False), start=1):
            metric_value = getattr(row, sample_metric)
            st.markdown(
                f"**{i}. {row.品牌_en} | {row.topic_label} | rule={row.rule_sentiment} | predicted={row.predicted_sentiment} | {sample_metric}={metric_value:.3f}**  \n"
                f"{str(row.评论内容)}"
            )


elif page == "Trend Analysis":
    monthly_brand = (
        filtered.groupby(["month", "品牌_en"], as_index=False)
        .agg(
            review_count=("评论内容", "count"),
            avg_positive=("positive_score", "mean"),
            avg_negative=("negative_score", "mean"),
            avg_net=("net_sentiment_score", "mean"),
        )
        .sort_values("month")
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig_vol = px.line(
            monthly_brand,
            x="month",
            y="review_count",
            color="品牌_en",
            markers=True,
            title="Monthly Review Volume by Brand",
        )
        fig_vol.update_layout(xaxis_title="Month", yaxis_title="Reviews")
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        fig_net = px.line(
            monthly_brand,
            x="month",
            y="avg_net",
            color="品牌_en",
            markers=True,
            title="Monthly Net Sentiment by Brand",
        )
        fig_net.update_layout(xaxis_title="Month", yaxis_title="Average net sentiment")
        st.plotly_chart(fig_net, use_container_width=True)

    c3, c4 = st.columns([1.1, 1])
    with c3:
        topic_trend = (
            filtered.groupby(["month", "topic_label"], as_index=False)
            .size()
            .rename(columns={"size": "review_count"})
            .sort_values("month")
        )
        fig_topic_trend = px.line(
            topic_trend,
            x="month",
            y="review_count",
            color="topic_label",
            markers=False,
            title="Topic Attention Trend",
        )
        fig_topic_trend.update_layout(xaxis_title="Month", yaxis_title="Reviews")
        st.plotly_chart(fig_topic_trend, use_container_width=True)

    with c4:
        monthly_scores = (
            filtered.groupby("month", as_index=False)[["positive_score", "neutral_score", "negative_score"]]
            .mean()
            .sort_values("month")
        )
        fig_score_trend = px.line(
            monthly_scores,
            x="month",
            y=["positive_score", "neutral_score", "negative_score"],
            markers=True,
            title="Overall Sentiment Score Trend",
        )
        fig_score_trend.update_layout(xaxis_title="Month", yaxis_title="Average score")
        st.plotly_chart(fig_score_trend, use_container_width=True)

    st.subheader("Existing forecast outputs from the team")
    forecast_cols = st.columns(2)
    img1 = ASSETS_DIR / "Brand_Net_Sentiment_Trend.png"
    img2 = ASSETS_DIR / "Four_Brands_ARIMA_Forecast.png"
    with forecast_cols[0]:
        if img1.exists():
            st.image(str(img1), caption="Original team output: brand net sentiment trend")
        else:
            st.info("Brand trend image not found.")
    with forecast_cols[1]:
        if img2.exists():
            st.image(str(img2), caption="Original team output: ARIMA forecast")
        else:
            st.info("Forecast image not found.")


elif page == "Review Explorer":
    st.subheader("Searchable review table")
    st.markdown(
        '<p class="small-note">Use the global filters in the sidebar, then search the table below for evidence-level comments.</p>',
        unsafe_allow_html=True,
    )

    cols_to_show = [
        "时间",
        "品牌_en",
        "页面标题",
        "商品规格",
        "topic_label",
        "rule_sentiment",
        "predicted_sentiment",
        "positive_score",
        "negative_score",
        "referral_score",
        "discussion_score",
        "评论内容",
    ]

    display_df = filtered[cols_to_show].copy().sort_values("时间", ascending=False)

    explorer_keyword = st.text_input(
        "Optional secondary search inside the filtered table",
        placeholder="e.g. 携带 / 清晰 / 包装",
    )
    if explorer_keyword:
        display_df = display_df[
            display_df["评论内容"].astype(str).str.contains(explorer_keyword, case=False, na=False)
        ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in table", f"{len(display_df):,}")
    c2.metric("Average comment length", f"{filtered['comment_length'].mean():.1f}")
    c3.metric("Average referral score", format_score(filtered["referral_score"].mean()))

    st.download_button(
        label="Download filtered data as CSV",
        data=convert_df_to_csv(display_df),
        file_name="filtered_camera_reviews.csv",
        mime="text/csv",
    )

    st.dataframe(display_df, use_container_width=True, height=520)

    st.subheader("Highlighted comments")
    highlight_metric_options = [
        "positive_score",
        "negative_score",
        "neutral_score",
        "discussion_score",
        "referral_score",
        "community_score",
    ]
    default_metric = default_highlight_metric(selected_sentiments)
    default_index = highlight_metric_options.index(default_metric) if default_metric in highlight_metric_options else 0

    c4, c5 = st.columns([1, 1.2])
    with c4:
        highlight_metric = st.selectbox("Rank highlighted comments by", highlight_metric_options, index=default_index)
    with c5:
        top_n = st.slider("Number of highlighted comments", min_value=3, max_value=15, value=5)

    highlights = (
        filtered.sort_values([highlight_metric, "discussion_score"], ascending=[False, False])
        .head(top_n)
        [["时间", "品牌_en", "topic_label", "rule_sentiment", "predicted_sentiment", highlight_metric, "评论内容"]]
    )
    for row in highlights.itertuples(index=False):
        date_text = row.时间.date() if pd.notna(row.时间) else ""
        metric_value = getattr(row, highlight_metric)
        st.markdown(
            f"**{row.品牌_en} | {row.topic_label} | {date_text} | rule={row.rule_sentiment} | predicted={row.predicted_sentiment} | {highlight_metric}={metric_value:.3f}**  \n"
            f"{row.评论内容}"
        )


elif page == "Chatbot":
    st.subheader("Camera Insight Chatbot")
    st.markdown(
        '<p class="small-note">Ask business-facing questions about brands, topics, sentiment, and review evidence. The chatbot answers from the built knowledge base in <code>kb_output</code>.</p>',
        unsafe_allow_html=True,
    )

    kb_dir = get_kb_dir()
    if kb_dir is None:
        st.error("Could not find a usable kb_output folder. Put kb_output next to app.py or one level above it.")
        st.code("python build_kb_and_index.py --input_xlsx camera_sentiment_label.xlsx --output_dir kb_output")
        st.stop()

    st.caption(f"Knowledge base directory: {kb_dir}")

    example_questions = [
        "Which brand performs best in Performance & Quality?",
        "What are Fuji's main strengths?",
        "Show me negative feedback about Sony.",
        "Which brand has the strongest overall sentiment?",
        "Which topic is most discussed for Canon?",
    ]

    c1, c2 = st.columns([1.7, 1])
    with c1:
        question = st.text_input(
            "Your question",
            value=st.session_state.get("chatbot_question", example_questions[0]),
            placeholder="Ask a question about brands, topics, or review evidence...",
        )
    with c2:
        preset = st.selectbox("Example questions", options=[""] + example_questions, index=0)
        if preset:
            question = preset
            st.session_state["chatbot_question"] = preset

    settings_col1, settings_col2, settings_col3 = st.columns([1, 1, 1.2])
    with settings_col1:
        top_k = st.slider("Retrieved evidence count", min_value=3, max_value=8, value=5)
    with settings_col2:
        model_name = st.text_input("LLM model", value=os.getenv("BIGMODEL_MODEL", DEFAULT_BIGMODEL_MODEL))
    with settings_col3:
        api_key = st.text_input(
            "BIGMODEL API key",
            type="password",
            value="19ca7071a318e06d65ed5f1b3a3402ad.zsdoa0UTlzKPWQVH",
            help="You can also set BIGMODEL_API_KEY in your environment.",
        )

    ask_col1, ask_col2 = st.columns([0.45, 1.55])
    with ask_col1:
        run_chat = st.button("Ask", type="primary", use_container_width=True)
    with ask_col2:
        show_prompt = st.checkbox("Show final prompt sent to the model", value=False)

    if run_chat:
        st.session_state["chatbot_question"] = question
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Retrieving evidence..."):
                hits = retrieve_chatbot_hits(question, kb_dir=kb_dir, top_k=top_k)

            if not hits:
                st.warning("No relevant evidence was retrieved for this question.")
            else:
                answer_text = ""
                final_prompt = ""
                if api_key.strip():
                    try:
                        with st.spinner("Generating answer..."):
                            answer_text, final_prompt = call_bigmodel_answer(question, hits, api_key.strip(), model_name.strip() or DEFAULT_BIGMODEL_MODEL)
                    except Exception as exc:
                        st.error(f"LLM generation failed: {exc}")
                else:
                    st.info("No API key provided. Evidence retrieval still worked; add your BIGMODEL API key to generate the final answer.")
                    final_prompt = build_chatbot_prompt(question, hits)

                if answer_text:
                    st.markdown("### Final Answer")
                    st.write(answer_text)

                st.markdown("### Retrieved Evidence")
                for i, h in enumerate(hits, start=1):
                    title = h.get("title", "Untitled")
                    badge = f"score={h.get('score', 0.0):.4f}"
                    with st.expander(f"Evidence {i}: {title} ({badge})", expanded=(i == 1)):
                        meta_df = pd.DataFrame(
                            {
                                "field": ["source_type", "source_file", "brand", "topic", "sentiment", "score"],
                                "value": [
                                    h.get("source_type", "unknown"),
                                    h.get("source_file", "unknown"),
                                    h.get("brand", "ALL"),
                                    h.get("topic", "ALL"),
                                    h.get("sentiment", "ALL"),
                                    f"{h.get('score', 0.0):.4f}",
                                ],
                            }
                        )
                        st.dataframe(meta_df, use_container_width=True, hide_index=True)
                        st.write(h.get("text", ""))

                if show_prompt and final_prompt:
                    st.markdown("### Prompt Preview")
                    st.code(final_prompt)



elif page == "Method Notes":
    st.subheader("How this dashboard is connected to your project")
    st.markdown(
        f"""
        **Current data source status**  
        {source_note}

        **Unified table logic used here**
        - Start from the merged review-level project table when available
        - Keep the original model outputs: `predicted_sentiment`, `positive_score`, `neutral_score`, `negative_score`
        - Rebuild the notebook-style rule label from the review text: `rule_sentiment`
        - Keep topic labels and behavioral signals in the same row-level table

        This gives you one dashboard table with both sentiment views instead of forcing you to choose only one.
        """
    )

    st.subheader("Why there are two sentiment labels")
    st.markdown(
        """
        - **Rule-based label** comes from the keyword counting logic used in the original sentiment notebook.
        - **Model-predicted label** comes from the Naive Bayes model trained on the rule labels.
        - They are not contradictory: one is the heuristic label, the other is the model output.
        - For dashboard storytelling, keeping both is more transparent and preserves the negative cases that were present in the rule-based stage.
        """
    )

    rule_dist = make_distribution_df(df, "rule_sentiment").rename(columns={"count": "rule_count"})
    pred_dist = make_distribution_df(df, "predicted_sentiment").rename(columns={"count": "predicted_count"})
    compare = rule_dist.merge(pred_dist, on="sentiment")
    st.dataframe(compare, use_container_width=True, hide_index=True)

    heat1 = ASSETS_DIR / "heatmap_positive_score_by_topic.png"
    heat2 = ASSETS_DIR / "heatmap_mention_frequency_by_topic.png"
    show_assets = st.checkbox("Show original team heatmaps", value=True)
    if show_assets:
        cols = st.columns(2)
        with cols[0]:
            if heat1.exists():
                st.image(str(heat1), caption="Original team output: positive score heatmap")
        with cols[1]:
            if heat2.exists():
                st.image(str(heat2), caption="Original team output: mention frequency heatmap")

    st.info(
        "Next step after this dashboard: add a chatbot page that answers questions from the same row-level table, with access to both rule-based and model-predicted sentiment fields."
    )
