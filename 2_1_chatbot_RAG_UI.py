import os
import hashlib
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# DocuNest — Streamlit frontend
# Your original backend file remains unchanged.
# This separate app contains a Streamlit-compatible workflow.
# ============================================================

st.set_page_config(
    page_title="DocuNest | Chat with your PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- COMPACT, HIGH-CONTRAST UI ----------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #f4f6fb;
        --muted: #c0c6d2;
        --line: #3a3e48;
        --canvas: #202124;
        --purple: #8b82ff;
        --purple-dark: #7569f0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(32,33,36,.97);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 1.1rem;
        padding-bottom: 6.5rem;
    }

    /* Brand */
    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 3px 0 24px 0;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        color: #fff;
        font-size: 22px;
        background: linear-gradient(135deg, #8276f4, #594bc8);
        box-shadow: 0 5px 14px rgba(101,88,217,.22);
    }

    .brand-title {
        font-family: 'Manrope', sans-serif;
        font-size: 21px;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.6px;
        color: #f5f6fa;
    }

    .brand-caption {
        margin-top: 3px;
        color: #c0c5d0;
        font-size: 11px;
    }

    .side-label {
        margin: 20px 0 9px 0;
        color: #aeb5c2;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.15px;
        text-transform: uppercase;
    }

    .side-note {
        color: #c0c5d0;
        font-size: 12px;
        line-height: 1.55;
        margin: 7px 0 12px;
        overflow-wrap: anywhere;
    }

    .document-status {
        background: #f8f9ff;
        border: 1px solid #e6e8fb;
        border-radius: 12px;
        padding: 12px;
        margin-top: 12px;
        color: #293148;
        overflow-wrap: anywhere;
    }

    .document-status .status-title {
        color: #f0f2f7;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .document-status .status-detail {
        color: #c0c6d2;
        font-size: 11px;
        line-height: 1.55;
    }

    /* Main header: compact, no oversized blank hero */
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        padding: 4px 0 15px;
        border-bottom: 1px solid #3a3e48;
        margin-bottom: 20px;
    }

    .topbar-title {
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -.4px;
        color: #f4f6fb;
    }

    .topbar-subtitle {
        color: #c0c5d0;
        font-size: 12px;
        margin-top: 3px;
    }

    .assistant-pill {
        white-space: nowrap;
        border: 1px solid #e2e5ef;
        border-radius: 99px;
        padding: 6px 10px;
        background: #fff;
        color: #626d82;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .6px;
        position: relative;
        top: 12px;
    }

    .welcome {
        padding: 12px 0 15px;
    }

    .welcome h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3vw, 34px);
        line-height: 1.2;
        letter-spacing: -1px;
        font-weight: 800;
        color: #f5f6fa !important;
        margin: 0 0 9px 0;
    }

    .welcome p {
        color: #c4c9d4 !important;
        font-size: 14px;
        line-height: 1.65;
        margin: 0;
    }

    .mini-label {
        color: #8992a3;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin: 13px 0 10px;
    }

    /* Suggestion buttons */
    div.stButton > button {
        min-height: 42px;
        border-radius: 11px;
        border: 1px solid #e1e5ef;
        background: #ffffff;
        color: #354057;
        font-size: 12px;
        font-weight: 600;
        transition: .15s ease;
    }

    div.stButton > button:hover {
        border-color: #b9b2f5;
        background: #f5f3ff;
        color: #5145bd;
    }

    /* Upload component */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e4e8f0;
        border-radius: 13px;
        padding: 11px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fafbff;
        border: 1px dashed #d4d9e7;
        border-radius: 10px;
    }

    /* Chat bubbles: readable dark text on white */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e6e9f1;
        border-radius: 15px;
        padding: 14px 17px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(25,35,60,.025);
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
        color: #f1f3f8 !important;
        line-height: 1.72;
        font-size: 14px;
    }

    /* Clean white input — no black footer */
    [data-testid="stChatInput"] {
        padding-top: 8px;
    }

    [data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 1px solid #dfe4ee !important;
        border-radius: 15px !important;
        box-shadow: 0 5px 18px rgba(30,40,70,.07) !important;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: #8b82ff !important;
        box-shadow: 0 0 0 3px rgba(101,88,217,.09) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #f5f6fa !important;
        font-size: 14px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #b4bac6 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    [data-testid="stCaptionContainer"] {
        color: #c0c5d0;
    }


    /* Final dark-theme overrides for Streamlit widgets */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    label, .stCaption, small {
        color: #f1f3f8;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #f1f3f8 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
        color: #f4f6fb !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #f4f6fb !important;
        caret-color: #a69fff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #aeb5c2 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #edf0f6 !important;
    }

    [data-testid="stAlert"] {
        background: #2b2d33;
        color: #f4f6fb;
        border-color: #464a55;
    }

    hr {
        border-color: #3a3e48 !important;
    }

    /* TRUE HIGH-CONTRAST DARK MODE: dark surfaces + readable white text */
    :root {
        --ink: #ffffff;
        --muted: #e0e0e0;
        --line: #424242;
        --canvas: #000000;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background: #000000 !important;
        color: #ffffff !important;
    }
    [data-testid="stHeader"], [data-testid="stBottom"] {
        background: #000000 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: #000000 !important;
        border-right: 1px solid #333333 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .brand-title, .brand-caption, .side-label, .side-note,
    .document-status, .document-status .status-title,
    .document-status .status-detail, .topbar-title,
    .topbar-subtitle, .mini-label, .welcome h1, .welcome p,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    label, small, .stCaption, [data-testid="stCaptionContainer"] {
        color: #ffffff !important;
    }
    .document-status, [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stChatMessage"],
    [data-testid="stChatInput"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: #171717 !important;
        color: #ffffff !important;
        border-color: #454545 !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #202020 !important;
        border: 1px dashed #606060 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
    }
    div.stButton > button, [data-testid="stDownloadButton"] button {
        background: #292929 !important;
        color: #ffffff !important;
        border: 1px solid #505050 !important;
    }
    div.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        background: #383838 !important;
        border-color: #8b82ff !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        background: #171717 !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        color: #bdbdbd !important;
        opacity: 1 !important;
    }
    .assistant-pill {
        background: #292929 !important;
        border-color: #505050 !important;
        color: #ffffff !important;
    }
    [data-testid="stAlert"] {
        background: #202020 !important;
        color: #ffffff !important;
        border-color: #505050 !important;
    }
    hr { border-color: #424242 !important; }

    @media (max-width: 700px) {
        .block-container {
            padding: .8rem .7rem 5rem;
        }
        .topbar-title {
            font-size: 16px;
        }
        .welcome h1 {
            font-size: 27px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SESSION STATE ----------------

defaults = {
    "pdf_hash": None,
    "agent": None,
    "llm": None,
    "messages": [],
    "pdf_name": None,
    "pdf_pages": None,
    "pdf_chunks": None,
    "vector_store": None,
    "pending_question": None,
    "chat_titles": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- MODEL HELPER ----------------

def get_llm():
    return ChatOpenAI(
        model="mistralai/mistral-large-4-0",
        base_url="https://api.xkiro.com/v1",
    )


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✳</div>
            <div>
                <div class="brand-title">DocuNest</div>
                <div class="brand-caption">Your AI document space</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown('<div class="side-label">Your conversations</div>', unsafe_allow_html=True)

    if st.session_state.messages:
        first_user_message = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"),
            "Conversation",
        )
        title = first_user_message[:32] + ("…" if len(first_user_message) > 32 else "")
        st.markdown(
            f'<div class="document-status"><div class="status-title">● {title}</div>'
            '<div class="status-detail">Current session</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="side-note">Your current conversation will appear here.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-label">Document source</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a resume, book, report, notes, or any text-based PDF.",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="side-note">PDF, reports, resumes, books and notes are welcome.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.pdf_hash and st.session_state.pdf_name:
        st.markdown(
            f"""
            <div class="document-status">
                <div class="status-title">✓ PDF ready</div>
                <div class="status-detail">{st.session_state.pdf_name}</div>
                <div class="status-detail">{st.session_state.pdf_pages} pages · {st.session_state.pdf_chunks} chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="side-note">No PDF uploaded yet.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="side-note">✦ RAG-powered document chat</div>',
        unsafe_allow_html=True,
    )


# ---------------- MAIN HEADER ----------------

st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="topbar-title">DocuNest Workspace</div>
            <div class="topbar-subtitle">A simpler way to explore your documents</div>
        </div>
        <div class="assistant-pill">● AI ASSISTANT</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- PDF PROCESSING ----------------

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if current_hash != st.session_state.pdf_hash:
        st.session_state.agent = None
        st.session_state.vector_store = None
        st.session_state.pdf_hash = None
        st.session_state.pdf_name = None
        st.session_state.messages = []

        try:
            with st.spinner("Preparing your PDF…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(pdf_bytes)
                    temp_path = temp_pdf.name

                try:
                    docs = PyPDFLoader(temp_path).load()
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if not docs:
                    raise ValueError("No readable pages were found in this PDF.")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2500,
                    chunk_overlap=400,
                ).split_documents(docs)

                if not splitter:
                    raise ValueError(
                        "No readable text was found. Scanned PDFs may need OCR."
                    )

                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )

                vector_store = InMemoryVectorStore.from_documents(
                    embedding=embeddings,
                    documents=splitter,
                )

                llm = get_llm()

                @tool
                def context_window_tool(query: str) -> str:
                    """Retrieve relevant information from the user's uploaded PDF."""
                    found_docs = vector_store.similarity_search(query, k=2)
                    return "\n\n".join(doc.page_content for doc in found_docs)

                system_prompt = """
                You are an intelligent PDF assistant.

                For questions about the uploaded document, use
                context_window_tool and answer using the retrieved PDF content.

                Do not invent facts, figures, quotations, methods, equations,
                or details that are not supported by the retrieved text.

                If the retrieved passages do not contain enough information,
                try a more specific query. If the information is still
                unavailable, clearly say it could not be found in the PDF.

                Keep answers clear, useful, and well structured.
                """

                agent = create_agent(
                    model=llm,
                    tools=[context_window_tool],
                    system_prompt=system_prompt,
                    checkpointer=InMemorySaver(),
                )

                st.session_state.vector_store = vector_store
                st.session_state.llm = llm
                st.session_state.agent = agent
                st.session_state.pdf_hash = current_hash
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_pages = len(docs)
                st.session_state.pdf_chunks = len(splitter)

            st.success(f"{uploaded_pdf.name} is ready to chat with.")

        except Exception as e:
            st.session_state.agent = None
            st.session_state.pdf_hash = None
            st.error(f"Could not process this PDF: {e}")


# ---------------- WELCOME / SUGGESTIONS ----------------

if not st.session_state.messages:
    if st.session_state.pdf_name:
        welcome_title = "Your PDF is ready."
        welcome_subtitle = "Ask a question about the document, or start a general chat."
        suggestions = [
            "Summarize this PDF",
            "What are the key points?",
            "Explain this document simply",
        ]
    else:
        welcome_title = "How can I help today?"
        welcome_subtitle = "Ask a general question or upload a PDF to explore its contents."
        suggestions = [
            "What is artificial intelligence?",
            "Explain RAG in simple words",
            "What can you help me with?",
        ]

    st.markdown(
        f"""
        <div class="welcome">
            <h1>{welcome_title}</h1>
            <p>{welcome_subtitle}</p>
        </div>
        <div class="mini-label">Try asking</div>
        """,
        unsafe_allow_html=True,
    )

    suggestion_cols = st.columns(3)
    for index, suggestion in enumerate(suggestions):
        with suggestion_cols[index]:
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- QUESTION ROUTING ----------------

def is_document_question(question: str) -> bool:
    """
    Lightweight UI-level routing.
    PDF-specific questions use the PDF agent; ordinary questions use the LLM directly.
    """
    q = question.lower().strip()
    document_terms = [
        "pdf", "document", "uploaded", "upload", "file",
        "resume", "résumé", "cv", "this paper", "research paper",
        "this report", "in the report", "in the document",
        "according to", "from the file", "from this",
        "page ", "pages", "in my", "this document", "this resume",
        "this pdf", "summarize this", "summarise this",
        "key points", "in the pdf", "what does it say",
        "whose resume", "uploaded resume",
    ]
    return any(term in q for term in document_terms)


# ---------------- CHAT INPUT / ANSWERS ----------------

pending_question = st.session_state.pending_question
st.session_state.pending_question = None
question = st.chat_input("Message DocuNest…")

if pending_question:
    question = pending_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if st.session_state.agent is not None and is_document_question(question):
                with st.spinner("Checking your document…"):
                    response = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": question}]},
                        {
                            "configurable": {
                                "thread_id": st.session_state.pdf_hash
                            }
                        },
                    )
                    answer = response["messages"][-1].content
            else:
                with st.spinner("Thinking…"):
                    if st.session_state.llm is None:
                        st.session_state.llm = get_llm()

                    general_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are DocuNest, a helpful general-purpose AI assistant. "
                                "Answer general questions directly and clearly. "
                                "Today's date is "
                                f"{datetime.now().strftime('%A, %B %d, %Y')}."
                            ),
                        }
                    ]

                    # Include recent conversational context, excluding the current
                    # message because it is appended separately below.
                    prior_messages = st.session_state.messages[:-1][-8:]
                    for item in prior_messages:
                        general_messages.append(
                            {"role": item["role"], "content": item["content"]}
                        )
                    general_messages.append({"role": "user", "content": question})

                    result = st.session_state.llm.invoke(general_messages)
                    answer = result.content

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error("I couldn't complete that request. Check your model API settings and try again.")
            st.caption(str(e))
