# main.py
# VAKALAT AI: ULTIMATE EDITION (Diagnostic + Auto-Healing + PDF Report)

import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
from fpdf import FPDF
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ---------------------------------------------------------
# 1. AUTHENTICATION & SETUP
# ---------------------------------------------------------

# Bridge for Secrets (Works on Cloud & Local)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()

# Verify Key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("❌ Critical Error: OpenAI API Key is missing. Check .env or Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="Vakalat AI | Legal Intelligence",
    page_icon="⚖️",
    layout="wide"
)

# PASSWORD BOUNCER
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Access Code:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Access Code:", type="password", on_change=password_entered, key="password")
        st.error("❌ Access Denied")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ---------------------------------------------------------
# 2. THE BRAIN (DATABASE ENGINE)
# ---------------------------------------------------------

@st.cache_resource
def get_vector_db():
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    embedding_function = OpenAIEmbeddings()
    db_path = "./chroma_db"
    
    # CHECK: Does the database exist?
    if not os.path.exists(db_path):
        st.warning("⚠️ Building Brain... (This happens once).")
        
        all_docs = []
        
        # A. DIAGNOSTIC: What files are actually here?
        files_in_root = os.listdir(".")
        st.write(f"📂 Debug: Files in Root: {files_in_root}")
        
        # B. INGEST STATUTES (Root Folder)
        statutes = ["bns.pdf", "bnss.pdf", "bsa.pdf"]
        for pdf in statutes:
            # Case-insensitive check
            found_file = next((f for f in files_in_root if f.lower() == pdf), None)
            if found_file:
                st.info(f"📖 Reading Statute: {found_file}...")
                loader = PyMuPDFLoader(found_file)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_book"] = found_file
                    doc.metadata["source_type"] = "statute"
                all_docs.extend(docs)
            else:
                st.error(f"❌ Missing File: {pdf} (Please upload to GitHub root)")
        
        # C. INGEST CASE LAW (Judgments Folder)
        judgment_folder = "./judgments"
        if os.path.exists(judgment_folder):
            all_files = os.listdir(judgment_folder)
            judgments = [f for f in all_files if f.lower().endswith(".pdf")]
            
            st.write(f"📂 Debug: Found {len(judgments)} judgments in folder.")
            
            for filename in judgments:
                filepath = os.path.join(judgment_folder, filename)
                st.info(f"⚖️ Reading Judgment: {filename}...")
                try:
                    loader = PyMuPDFLoader(filepath)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_type"] = "case_law"
                        doc.metadata["case_name"] = filename
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"Failed to read {filename}: {e}")
        else:
            st.warning(f"⚠️ 'judgments' folder not found at {os.path.abspath(judgment_folder)}")
        
        # D. SPLIT TEXT
        if not all_docs:
            st.error("❌ No documents found! Database will be empty.")
            return Chroma(persist_directory=db_path, embedding_function=embedding_function)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        # E. INJECT PATCH (Mob Lynching)
        chunks.append(Document(
            page_content="BNS Section 103(2) (Mob Lynching): When a group of five or more persons acting in concert commits murder on the ground of race, caste or community, sex, place of birth, language, personal belief or any other like ground, each member of such group shall be punished with death or with imprisonment for life, and shall also be liable to fine.",
            metadata={"source_book": "bns.pdf", "source_type": "statute"} 
        ))

        # F. BUILD & SAVE
        st.info("🧠 Indexing Neural Connections... Please wait.")
        db = Chroma.from_documents(chunks, embedding_function, persist_directory=db_path)
        st.success("✅ Brain Rebuilt! System Online.")
        return db
        
    # Normal Load
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
    return db

# FORCE RESET FUNCTION
def reset_brain():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    st.session_state.clear()
    st.rerun()

# Initialize DB
try:
    vector_db = get_vector_db()
except Exception as e:
    st.error(f"Database Error: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS (PDF & File Reading)
# ---------------------------------------------------------

def read_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def create_pdf(query, response, sources):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Vakalat AI | Legal Opinion Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Generated by Vakalat AI. Not a substitute for professional legal counsel.', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. Title
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, f"Query: {query[:200]}...")
    pdf.ln(5)
    
    # 2. Analysis Body (Sanitize text for basic Latin support)
    pdf.set_font("Arial", "", 11)
    # Replace unsupported characters to prevent crash
    safe_response = response.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_response)
    pdf.ln(10)
    
    # 3. Sources
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Legal Sources Referenced:", 0, 1)
    pdf.set_font("Arial", "I", 9)
    for doc in sources:
        # Sanitize source names too
        raw_source = doc.metadata.get('source_book') or doc.metadata.get('case_name') or "Unknown"
        safe_source = raw_source.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 6, f"- {safe_source}", 0, 1)
        
    # RETURN BYTES (Fix: Do not double-encode)
    return bytes(pdf.output(dest='S'))

# ---------------------------------------------------------
# 4. SIDEBAR & ADMIN
# ---------------------------------------------------------

with st.sidebar:
    st.title("⚖️ Vakalat AI")
    st.caption("Statute + Case Law Engine")
    
    st.markdown("---")
    st.subheader("📂 Analyze Case File")
    uploaded_file = st.file_uploader("Upload FIR / Charge Sheet (PDF)", type="pdf")
    
    st.markdown("---")
    st.subheader("⚙️ System Admin")
    if st.button("🔄 Force Rebuild Brain"):
        st.warning("Deleting old index and rebuilding... (Takes ~2 mins)")
        reset_brain()

# ---------------------------------------------------------
# 5. CORE LOGIC (ROUTER + SEARCH)
# ---------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# A. RETRIEVER
def get_legal_context(query, k=4):
    docs = []
    # 1. Broad Statute Search
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_type": "statute"}))
    
    # 2. Smart Case Law Search
    trigger_words = ["arrest", "police", "custody", "bail", "detention", "torture", "handcuff", "remand", "investigation", "fir", "complaint", "quash"]
    if any(word in query.lower() for word in trigger_words):
        docs.extend(vector_db.similarity_search(query, k=3, filter={"source_type": "case_law"}))
    
    return docs

# B. ROUTER
router_prompt = ChatPromptTemplate.from_template("""
Classify the user's query into one of two categories. Return ONLY the category name.
1. LEGAL_RESEARCH: Questions about laws, crimes, punishments, police, FIRs, courts, bail, or specific sections.
2. GENERAL_CHAT: Greetings, asking "who are you", "what can you do", "help", or general conversation.
Query: {question}
Category:
""")
router_chain = router_prompt | llm | StrOutputParser()

# C. PROMPTS
general_prompt = ChatPromptTemplate.from_template("""
You are Vakalat AI, a specialized legal research assistant for Indian Criminal Law (BNS, BNSS, BSA).
The user is asking a general question. Answer professionally and concisely.
User Query: {question}
Answer:
""")

research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat AI, a senior legal consultant.
Use the Context (Statutes + Case Law) to answer.

CRITICAL RULES:
1. First, state the STATUTE (BNS/BNSS).
2. Second, check if any SUPREME COURT JUDGMENT (Case Law) overrides or clarifies it.
3. Use the Case Name if provided in the source metadata.

Context:
{context}

Question: {question}
Answer:
""")

analysis_prompt = ChatPromptTemplate.from_template("""
You are a Defense Lawyer analyzing a case file.
TASK:
1. Analyze the CLIENT CASE FILE facts.
2. Cross-reference with RETRIEVED LAWS.
3. Find Loopholes (Arrest rules, Section ingredients).

CLIENT CASE FILE:
{case_file}

RETRIEVED LAWS:
{context}

USER QUESTION: {question}

ANALYSIS:
""")

# ---------------------------------------------------------
# 6. MAIN CHAT INTERFACE
# ---------------------------------------------------------

st.subheader("Your Legal Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am online. Ready for legal research or case analysis."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ex: 'Punishment for Section 302' or 'Who are you?'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            if uploaded_file:
                intent = "LEGAL_RESEARCH"
            else:
                intent = router_chain.invoke({"question": user_input}).strip()
        
        if intent == "GENERAL_CHAT":
            chain = general_prompt | llm | StrOutputParser()
            response = chain.invoke({"question": user_input})
            message_placeholder.markdown(response)
            
        else:
            with st.spinner("Consulting Legal Database..."):
                # Prepare Data
                case_text = ""
                if uploaded_file:
                    case_text = read_pdf(uploaded_file)
                    full_query = f"{user_input} {case_text[:1500]}"
                else:
                    full_query = user_input

                # Retrieve
                docs = get_legal_context(full_query, k=4)
                context_text = "\n\n".join(f"[Source: {d.metadata.get('source_book', d.metadata.get('case_name', 'Unknown'))}]\n{d.page_content}" for d in docs)
                
                # Generate
                if uploaded_file:
                    chain = analysis_prompt | llm | StrOutputParser()
                    response = chain.invoke({"case_file": case_text, "context": context_text, "question": user_input})
                else:
                    chain = research_prompt | llm | StrOutputParser()
                    response = chain.invoke({"context": context_text, "question": user_input})
                
                message_placeholder.markdown(response)
                
                # --- PDF REPORT GENERATION ---
                try:
                    pdf_bytes = create_pdf(user_input, response, docs)
                    
                    # Layout: Button on the left
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.download_button(
                            label="📄 Download Official Report",
                            data=pdf_bytes,
                            file_name="Vakalat_Legal_Opinion.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"⚠️ Could not generate PDF: {e}")
                
                # Evidence Inspector
                with st.expander("🔍 Inspect Legal Sources"):
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get('source_book') or doc.metadata.get('case_name') or "Unknown"
                        st.caption(f"**{i+1}. {source}**")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
