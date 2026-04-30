"""
NovaTech RAG Chatbot -- Streamlit Demo (Claude API)
BSAN 6200 | Spring 2026

Run with: python -m streamlit run streamlit_app.py

Setup:
    pip install streamlit chromadb anthropic python-dotenv

    Create .env file with:
        ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

    Get API key at: https://console.anthropic.com
"""

import streamlit as st
import chromadb
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NovaTech RAG Chatbot", page_icon="🤖")

# ── Config ──
# claude-haiku-4-5-20251001  -- cheapest ($0.80/1M input, $4/1M output)
# claude-sonnet-4-20250514   -- best balance ($3/1M input, $15/1M output)
MODEL_ID = "claude-haiku-4-5-20251001"

GROUNDED_PROMPT = """You are a concise assistant for NovaTech, a data analytics consulting firm.
Answer the question using ONLY the information provided in the context below.

Rules:
1. Answer in 1-2 sentences MAX. Be as brief as possible.
2. Only include information that DIRECTLY answers the question. Do not add extra details.
3. If the context does not contain the answer, say: "I don't have that information."
4. Do NOT generate follow-up questions or additional context."""

SAMPLE_DOCUMENTS = [
    {
        "text": "NovaTech is a data analytics consulting firm founded in 2019 and headquartered "
                "in Los Angeles, California. The company specializes in helping mid-size retail "
                "and e-commerce businesses turn raw data into actionable insights. NovaTech employs "
                "45 people across three departments: Data Engineering, Analytics, and Client Services. "
                "The company uses Python, SQL, and Tableau as its primary technology stack.",
        "source": "company_overview",
    },
    {
        "text": "The Data Engineering team at NovaTech is responsible for building and maintaining "
                "ETL pipelines that process over 2 million transactions per day for retail clients. "
                "The team uses Apache Airflow for orchestration, PostgreSQL and BigQuery for data "
                "warehousing, and dbt for data transformations. The team consists of 12 engineers, "
                "led by Senior Director Maria Chen.",
        "source": "data_engineering",
    },
    {
        "text": "NovaTech offers three tiers of analytics services: Basic Reporting (dashboards and "
                "KPI tracking), Advanced Analytics (predictive modeling, customer segmentation, churn "
                "analysis), and Strategic Consulting (C-suite advisory, data strategy roadmaps). The "
                "Analytics team uses scikit-learn and XGBoost for machine learning, along with custom "
                "NLP pipelines built with spaCy.",
        "source": "analytics_services",
    },
    {
        "text": "NovaTech runs a competitive summer internship program for graduate students in data "
                "science, business analytics, and computer science. The program runs from June through "
                "August (10 weeks) and includes a stipend of $6,000 per month. Past interns have come "
                "from USC, UCLA, LMU, and UC Irvine. Approximately 60% of interns receive full-time offers.",
        "source": "internship",
    },
    {
        "text": "RetailMax, a regional chain of 85 home goods stores, partnered with NovaTech in 2023. "
                "The engagement included migrating from Excel-based inventory tracking to an automated "
                "BigQuery pipeline, building a customer segmentation model, and deploying a Tableau "
                "dashboard suite. Results after 6 months: 22% reduction in stockouts, 15% increase in "
                "targeted marketing ROI.",
        "source": "case_study",
    },
]


# ══════════════════════════════════════════
# Load resources (cached -- runs only once)
# ══════════════════════════════════════════

@st.cache_resource
def load_vectorstore():
    client = chromadb.Client()
    collection = client.create_collection("novatech_demo")
    collection.add(
        documents=[d["text"] for d in SAMPLE_DOCUMENTS],
        metadatas=[{"source": d["source"]} for d in SAMPLE_DOCUMENTS],
        ids=[f"chunk_{i}" for i in range(len(SAMPLE_DOCUMENTS))],
    )
    return collection


@st.cache_resource
def load_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it to your .env file.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ══════════════════════════════════════════
# RAG logic
# ══════════════════════════════════════════

def search(collection, query, k=3):
    """Retrieve top-k chunks from the vector store."""
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0], [m["source"] for m in results["metadatas"][0]]


def ask_rag(collection, llm_client, question, k=3):
    """Full RAG pipeline: retrieve -> build prompt -> generate."""
    docs, sources = search(collection, question, k=k)
    context = "\n\n".join(docs)

    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    response = llm_client.messages.create(
        model=MODEL_ID,
        max_tokens=150,
        system=GROUNDED_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = response.content[0].text.strip()
    return answer, sources, docs


# ══════════════════════════════════════════
# UI
# ══════════════════════════════════════════

collection = load_vectorstore()
llm_client = load_llm()

st.title("🤖 NovaTech RAG Chatbot")
st.caption("Ask me anything about NovaTech -- answers are grounded in company documents.")

# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot uses Retrieval-Augmented Generation (RAG) to answer "
        "questions about NovaTech, a fictional analytics company."
    )
    st.write("**Documents in knowledge base:**")
    for d in SAMPLE_DOCUMENTS:
        st.write(f"- {d['source'].replace('_', ' ').title()}")
    st.divider()
    st.write(f"**Model:** {MODEL_ID}")
    st.caption("BSAN 6200 | Spring 2026")

# ── Sample questions ──
st.write("**Try a sample question:**")
samples = [
    "What technology does NovaTech use?",
    "Tell me about the internship program",
    "What results did RetailMax achieve?",
]
cols = st.columns(len(samples))
for i, q in enumerate(samples):
    if cols[i].button(q, key=f"sample_{i}"):
        st.session_state["pending_question"] = q

# ── Chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Retrieved chunks"):
                st.markdown(msg["sources"])

# ── Handle input ──
user_input = st.chat_input("Ask me about NovaTech...")

if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            answer, sources, docs = ask_rag(collection, llm_client, user_input)

            st.markdown(answer)

            source_text = ""
            for i, (doc, src) in enumerate(zip(docs, sources)):
                source_text += f"**Chunk {i+1}** ({src}):\n> {doc[:200]}...\n\n"

            with st.expander("📎 Retrieved chunks"):
                st.markdown(source_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_text,
            })

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
