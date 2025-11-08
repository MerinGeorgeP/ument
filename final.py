import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import shutil
import hashlib
from transformers import pipeline


# ============================================================
# 🔐 AUTHENTICATION & USER MANAGEMENT
# ============================================================

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

USERS_DIR = "users"
USERS_DB = os.path.join(USERS_DIR, "users_db.pkl")
os.makedirs(USERS_DIR, exist_ok=True)

if os.path.exists(USERS_DB):
    with open(USERS_DB, "rb") as f:
        users_db = pickle.load(f)
else:
    users_db = {}

def save_users_db():
    with open(USERS_DB, "wb") as f:
        pickle.dump(users_db, f)

def signup(username, password):
    if username in users_db:
        st.error("Username already exists.")
        return False
    users_db[username] = hash_password(password)
    save_users_db()
    os.makedirs(os.path.join(USERS_DIR, username, "uploaded_pdfs"), exist_ok=True)
    st.success("Signup successful! You can now log in.")
    return True

def login(username, password):
    if username not in users_db:
        st.error("Username does not exist.")
        return False
    if not verify_password(password, users_db[username]):
        st.error("Incorrect password.")
        return False
    st.session_state.username = username
    st.session_state.logged_in = True
    st.success(f"Logged in as {username}")
    return True

def get_user_paths(username):
    user_dir = os.path.join(USERS_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return {
        "UPLOAD_DIR": os.path.join(user_dir, "uploaded_pdfs"),
        "INDEX_PATH": os.path.join(user_dir, "faiss_index.index"),
        "DOC_CHUNKS_PATH": os.path.join(user_dir, "doc_chunks.pkl")
    }

# ============================================================
# 🤖 MODEL LOADERS (Cached)
# ============================================================

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

@st.cache_resource
def load_qa_model():
    """Lightweight QA model"""
    return pipeline("text2text-generation", model="google/t5-efficient-tiny", device=-1)

# ============================================================
# 📘 PDF PROCESSING HELPERS
# ============================================================

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ============================================================
# 🧠 SUMMARIZATION & QA HELPERS
# ============================================================

def summarize_text(summarizer, text, max_chunk_length=800):
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_length]) for i in range(0, len(words), max_chunk_length)]
    summaries = []
    for chunk in chunks:
        try:
            inputs = chunk[:3000]
            summary = summarizer(inputs, max_length=200, min_length=60, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Skipping chunk due to error: {e}")
    return " ".join(summaries)

def qa_answer(model, question, context):
    """Generate concise, sensible answer using T5-efficient-tiny"""
    prompt = f"Give a short and clear answer. Question: {question} Context: {context}"
    try:
        result = model(
            prompt,
            max_length=40,
            min_length=3,
            do_sample=False,
            num_beams=4
        )
        answer = result[0]['generated_text']

        # Optional: Refine the answer
        if len(answer.split()) > 25:
            summarizer = load_summarizer()
            refined = summarizer(answer, max_length=40, min_length=10, do_sample=False)[0]['summary_text']
            return refined

        return answer

    except Exception as e:
        print("Error generating answer:", e)
        return "Unable to generate a concise answer."


def get_relevant_chunks(query, index, doc_chunks, top_k=3):
    """Retrieve most relevant text chunks via FAISS"""
    model = load_sentence_model()
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(np.array(q_emb, dtype=np.float32), top_k)
    relevant_texts = []
    for idx in indices[0]:
        if idx < len(doc_chunks):
            _, chunk = doc_chunks[idx]
            relevant_texts.append(chunk)
    return " ".join(relevant_texts)



# ============================================================
# 🗂️ DATA STORAGE / FAISS INDEX
# ============================================================

def build_faiss_index(doc_chunks):
    model = load_sentence_model()
    texts = [chunk for _, chunk in doc_chunks]
    embeddings = model.encode(texts, convert_to_tensor=False)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    return index

def save_data(index, doc_chunks, paths):
    if index is not None:
        faiss.write_index(index, paths["INDEX_PATH"])
    with open(paths["DOC_CHUNKS_PATH"], "wb") as f:
        pickle.dump(doc_chunks, f)

def load_data(paths):
    index = None
    doc_chunks = []
    if os.path.exists(paths["INDEX_PATH"]) and os.path.exists(paths["DOC_CHUNKS_PATH"]):
        index = faiss.read_index(paths["INDEX_PATH"])
        with open(paths["DOC_CHUNKS_PATH"], "rb") as f:
            doc_chunks = pickle.load(f)
    return index, doc_chunks

def clear_all_data(paths):
    st.session_state.index = None
    st.session_state.doc_chunks = []
    if os.path.exists(paths["UPLOAD_DIR"]):
        shutil.rmtree(paths["UPLOAD_DIR"])
    os.makedirs(paths["UPLOAD_DIR"], exist_ok=True)
    st.success("Cleared all stored PDFs.")

# ============================================================
# 🎨 STREAMLIT APP
# ============================================================

st.set_page_config(page_title="AI Librarian", layout="wide")
st.title("📚 AI Librarian + 🧠 Summarizer + 💬 Q&A")

# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Login / Signup ---
if not st.session_state.logged_in:
    st.header("👤 Login / Signup")
    choice = st.radio("Select Action", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        if choice == "Signup":
            signup(username, password)
        else:
            login(username, password)

else:
    st.sidebar.write(f"👋 Logged in as **{st.session_state.username}**")
    user_paths = get_user_paths(st.session_state.username)
    st.session_state.index, st.session_state.doc_chunks = load_data(user_paths)

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Navigation
    menu = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload PDFs", "🔎 Search PDFs", "🧠 Summarize PDF", "💬 Ask Questions"]
    )

    # ---------------- HOME ----------------
    if menu == "🏠 Home":
        st.header("Welcome to AI Librarian")
        st.write("Upload, search, summarize, and ask questions about your PDFs!")

    # ---------------- UPLOAD ----------------
    elif menu == "📤 Upload PDFs":
        st.header("Upload and Manage PDFs")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("Save Files"):
            for file in uploaded_files:
                with open(os.path.join(user_paths["UPLOAD_DIR"], file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded successfully!")

            # ✅ Build FAISS index automatically
            st.info("Building FAISS index for uploaded PDFs...")
            doc_chunks = []
            for file in os.listdir(user_paths["UPLOAD_DIR"]):
                pdf_path = os.path.join(user_paths["UPLOAD_DIR"], file)
                text = extract_text_from_pdf(pdf_path)
                if text.strip():
                    chunks = chunk_text(text)
                    doc_chunks.extend([(file, c) for c in chunks])

            if doc_chunks:
                index = build_faiss_index(doc_chunks)
                save_data(index, doc_chunks, user_paths)
                st.session_state.index = index
                st.session_state.doc_chunks = doc_chunks
                st.success("✅ Index built successfully! You can now search or ask questions.")
            else:
                st.warning("No valid text extracted from PDFs — index not built.")

        existing = os.listdir(user_paths["UPLOAD_DIR"])
        if existing:
            st.subheader("Stored PDFs:")
            for f in existing:
                st.write("•", f)

            if st.button("🔄 Rebuild FAISS Index"):
                st.info("Rebuilding FAISS index...")
                doc_chunks = []
                for file in os.listdir(user_paths["UPLOAD_DIR"]):
                    pdf_path = os.path.join(user_paths["UPLOAD_DIR"], file)
                    text = extract_text_from_pdf(pdf_path)
                    if text.strip():
                        chunks = chunk_text(text)
                        doc_chunks.extend([(file, c) for c in chunks])
                if doc_chunks:
                    index = build_faiss_index(doc_chunks)
                    save_data(index, doc_chunks, user_paths)
                    st.session_state.index = index
                    st.session_state.doc_chunks = doc_chunks
                    st.success("✅ FAISS index rebuilt successfully!")
                else:
                    st.warning("No valid text found to rebuild index.")

            if st.button("🗑️ Clear Library"):
                clear_all_data(user_paths)

    # ---------------- SEARCH ----------------
    elif menu == "🔎 Search PDFs":
        st.header("Search Stored PDFs")

        # ✅ Always reload saved FAISS index
        st.session_state.index, st.session_state.doc_chunks = load_data(user_paths)

        if st.session_state.index is None or not st.session_state.doc_chunks:
            st.warning("No indexed documents found. Please upload PDFs first.")
        else:
            query = st.text_input("Enter search query")
            if st.button("Search"):
                model = load_sentence_model()
                q_emb = model.encode([query])
                faiss.normalize_L2(q_emb)
                distances, indices = st.session_state.index.search(np.array(q_emb, dtype=np.float32), 10)

                seen_files = set()
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    filename, chunk = st.session_state.doc_chunks[idx]
                    if filename not in seen_files:
                        seen_files.add(filename)
                        results.append((filename, chunk, dist))

                if results:
                    st.subheader("🔍 Search Results:")
                    for filename, chunk, dist in results:
                        st.write(f"📄 **{filename}** — (distance: {dist:.2f})")
                        st.write(chunk[:400] + "...")
    
                        # 📎 Add clickable link to open/download PDF
                        pdf_path = os.path.join(user_paths["UPLOAD_DIR"], filename)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label=f"📥 Download {filename}",
                                data=pdf_file,
                                file_name=filename,
                                mime="application/pdf"
                            )

                        st.markdown("---")

                else:
                    st.info("No relevant results found.")

    # ---------------- SUMMARIZE ----------------
    elif menu == "🧠 Summarize PDF":
        st.header("Summarize Your PDFs")
        files = os.listdir(user_paths["UPLOAD_DIR"])
        if not files:
            st.warning("No PDFs available. Please upload first.")
        else:
            selected_file = st.selectbox("Select a PDF to summarize", files)
            if st.button("Generate Summary"):
                pdf_path = os.path.join(user_paths["UPLOAD_DIR"], selected_file)
                text = extract_text_from_pdf(pdf_path)
                if len(text) < 500:
                    st.warning("PDF too short to summarize.")
                else:
                    summarizer = load_summarizer()
                    with st.spinner("Summarizing..."):
                        summary = summarize_text(summarizer, text)
                    st.success("✅ Summary generated!")
                    st.write(summary)
                    
                    

    # ---------------- Q&A SECTION ----------------
    elif menu == "💬 Ask Questions":
        st.header("💬 Ask Questions from Your PDFs (T5-Efficient-Tiny + FAISS)")
        files = os.listdir(user_paths["UPLOAD_DIR"])
        if not files:
            st.warning("No PDFs uploaded yet.")
        else:
            selected_file = st.selectbox("Select a PDF to query", files)
            question = st.text_input("Enter your question")

            if st.button("Get Answer"):
                pdf_path = os.path.join(user_paths["UPLOAD_DIR"], selected_file)
                text = extract_text_from_pdf(pdf_path)
                qa_model = load_qa_model()

                if len(text) < 300:
                    st.warning("PDF too short for Q&A.")
                else:
                    if not st.session_state.doc_chunks or not st.session_state.index:
                        st.info("Building FAISS index...")
                        chunks = chunk_text(text)
                        doc_chunks = [(selected_file, c) for c in chunks]
                        index = build_faiss_index(doc_chunks)
                        st.session_state.doc_chunks = doc_chunks
                        st.session_state.index = index
                        save_data(index, doc_chunks, user_paths)
                    else:
                        index = st.session_state.index
                        doc_chunks = st.session_state.doc_chunks

                    with st.spinner("Retrieving relevant information... 🔍"):
                        context = get_relevant_chunks(question, index, doc_chunks, top_k=3)

                    with st.spinner("Generating answer... 🤖"):
                        answer = qa_answer(qa_model, question, context)

                    st.success("✅ Answer generated!")
                    

                    with st.expander("📘 View Retrieved Context"):
                        st.write(context)

                    
                    



