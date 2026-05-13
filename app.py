import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
if "llm" not in st.session_state:
        st.session_state.llm = None

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")

# Load external CSS
# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- HERO ---
st.markdown("""
<div class="hero">
    <h1>📄 RAG Document Assistant</h1>
    <p>Upload any PDF. Ask anything. Get answers grounded in your document.</p>
</div>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {}
if "chain" not in st.session_state:
    st.session_state.chain = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📁 Documents")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")

        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 100, help="Smaller = more precise, Larger = more context")
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 25, help="Overlap between chunks")

        if st.button("🔄 Process Documents"):
            import hashlib
            import pickle

            os.makedirs("cache", exist_ok=True)

            with st.spinner("Processing PDFs..."):
                all_text = ""
                total_pages = 0
                file_names = []

                # Build cache key from all filenames + sizes
                cache_key = hashlib.md5(
                    "".join([f.name + str(f.size) for f in uploaded_files]).encode()
                ).hexdigest()
                cache_path = f"cache/{cache_key}.faiss"
                meta_path = f"cache/{cache_key}.meta"

                # Check cache
                if os.path.exists(cache_path) and os.path.exists(meta_path):
                    st.info("⚡ Loading from cache — instant!")
                    embeddings = get_embeddings()
                    vector_store = FAISS.load_local(
                        cache_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    with open(meta_path, "rb") as f:
                        cached_meta = pickle.load(f)

                    st.session_state.vector_store = vector_store
                    st.session_state.llm = get_llm()
                    st.session_state.chat_history = []
                    st.session_state.doc_meta = cached_meta
                    st.success(f"✦ Loaded from cache in <1s!")

                else:
                    # Process fresh
                    progress_bar = st.progress(0)
                    status = st.empty()

                    for idx, uploaded_file in enumerate(uploaded_files):
                        status.text(f"📄 Reading {uploaded_file.name}...")
                        reader = PdfReader(uploaded_file)
                        total_pages += len(reader.pages)
                        file_names.append(uploaded_file.name)

                        for page_num, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                all_text += f"\n[PAGE {page_num + 1} | {uploaded_file.name}]\n{text}"

                        progress_bar.progress((idx + 1) / (len(uploaded_files) * 3))

                    if not all_text.strip():
                        st.error("No text found in PDFs. Try a different file.")
                    else:
                        status.text("✂️ Splitting into chunks...")
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=["\n\n", "\n", ".", " "]
                        )
                        chunks = splitter.split_text(all_text)
                        progress_bar.progress(0.5)

                        status.text(f"🧠 Embedding {len(chunks)} chunks...")
                        embeddings = get_embeddings()

                        # Batch embedding — 32 chunks at a time
                        # Batch embedding — 32 chunks at a time
                        batch_size = 32
                        total_batches = (len(chunks) + batch_size - 1) // batch_size

                        for batch_num, i in enumerate(range(0, len(chunks), batch_size)):
                            batch = chunks[i:i+batch_size]
                            status.text(f"🧠 Embedding batch {batch_num+1}/{total_batches} ({len(batch)} chunks)...")
                            progress_bar.progress(0.5 + 0.4 * ((batch_num+1) / total_batches))

                        vector_store = FAISS.from_texts(chunks, embeddings)
                        progress_bar.progress(0.95)

                        status.text("💾 Saving to cache...")
                        vector_store.save_local(cache_path)
                        doc_meta = {
                            "files": file_names,
                            "pages": total_pages,
                            "chunks": len(chunks)
                        }
                        with open(meta_path, "wb") as f:
                            pickle.dump(doc_meta, f)

                        progress_bar.progress(1.0)
                        status.text("✅ Done!")

                        st.session_state.vector_store = vector_store
                        st.session_state.llm = get_llm()
                        st.session_state.chat_history = []
                        st.session_state.doc_meta = doc_meta
                        st.success(f"✦ {len(chunks)} chunks indexed + cached!")

    # Doc stats
    if st.session_state.doc_meta:
        meta = st.session_state.doc_meta
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        st.markdown(f"**Files:** {len(meta['files'])}")
        st.markdown(f"**Pages:** {meta['pages']}")
        st.markdown(f"**Chunks:** {meta['chunks']}")
        st.markdown("---")
        for f in meta["files"]:
            st.markdown(f"📄 {f}")

    if st.button("🗑️ Clear Everything"):
        st.session_state.vector_store = None
        st.session_state.chain = None
        st.session_state.chat_history = []
        st.session_state.doc_meta = {}
        st.rerun()

# --- MAIN AREA ---
if not st.session_state.vector_store:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📄</div>
        <div style="font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#1a1a2e; margin-bottom:0.5rem;">No document loaded</div>
        <div style="color:#888; font-size:0.9rem; max-width:400px; margin:0 auto;">Upload a PDF in the sidebar and click <strong>Process Documents</strong> to start asking questions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">✦ What you can do</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="doc-card">
            <h4>📑 Multi-PDF Support</h4>
            <p>Upload multiple documents at once. Query across all of them simultaneously.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="doc-card">
            <h4>💬 Conversational Memory</h4>
            <p>Ask follow-up questions. The assistant remembers your conversation context.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="doc-card">
            <h4>📍 Source Citations</h4>
            <p>Every answer shows which page and document the information came from.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Doc loaded — show stats
    meta = st.session_state.doc_meta
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">{len(meta['files'])}</div><div class="stat-label">Documents</div></div>
        <div class="stat-box"><div class="stat-num">{meta['pages']}</div><div class="stat-label">Pages</div></div>
        <div class="stat-box"><div class="stat-num">{meta['chunks']}</div><div class="stat-label">Chunks Indexed</div></div>
        <div class="stat-box"><div class="stat-num">{len(st.session_state.chat_history)}</div><div class="stat-label">Questions Asked</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    st.markdown('<div class="section-label">✦ Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#888; font-size:0.9rem;">
            ✦ Document indexed. Ask your first question below.
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            sources = msg.get("sources", "")
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}<div class="chat-source">{sources}</div></div>', unsafe_allow_html=True)

    # Question input
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">✦ Ask a Question</div>', unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**Quick questions:**")
    suggestions = [
        "Summarize this document",
        "What are the main conclusions?",
        "What are the key findings?",
        "List the main topics covered"
    ]
    cols = st.columns(4)
    for i, q in enumerate(suggestions):
        with cols[i]:
            if st.button(q, key=f"sq_{i}"):
                st.session_state["pending_q"] = q
                st.rerun()

    question = st.text_input("Your question:", placeholder="e.g. What does the document say about...")

    if st.session_state.get("pending_q"):
        question = st.session_state["pending_q"]
        st.session_state["pending_q"] = None

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask = st.button("✦ Ask", disabled=not question)
    with col_clear:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask and question:
        with st.spinner("Searching document..."):
            try:
                # FAISS similarity search
                docs = st.session_state.vector_store.similarity_search(question, k=4)

                # Extract source page info
                sources_seen = set()
                source_parts = []
                context_chunks = []

                for doc in docs:
                    context_chunks.append(doc.page_content)
                    for line in doc.page_content.split('\n'):
                        if line.startswith('[PAGE'):
                            page_info = line.strip('[]')
                            if page_info not in sources_seen:
                                sources_seen.add(page_info)
                                source_parts.append(f"📍 {page_info}")
                            break

                context = "\n\n---\n\n".join(context_chunks)
                sources_str = " | ".join(source_parts) if source_parts else "📍 Source: document"

                # Build chat history for context
                history_text = ""
                for msg in st.session_state.chat_history[-6:]:  # last 3 exchanges
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"

                # Call Groq directly
                messages = [
                    SystemMessage(content=f"""You are a helpful document assistant. Answer questions based ONLY on the provided document context. 
If the answer is not in the context, say "I couldn't find this in the document."
Be concise and accurate.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}"""),
                    HumanMessage(content=question)
                ]

                response = st.session_state.llm.invoke(messages)
                answer = response.content

                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources_str})
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")

    # Export chat
    if st.session_state.chat_history:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        chat_export = "\n\n".join([
            f"{'USER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}"
            + (f"\n{m.get('sources','')}" if m['role']=='assistant' else "")
            for m in st.session_state.chat_history
        ])
        st.download_button(
            label="⬇ Export Chat as .txt",
            data=chat_export,
            file_name="rag_chat_export.txt",
            mime="text/plain"
        )
        
        