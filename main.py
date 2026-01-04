# main.py
# VAKALAT AI: ULTIMATE EDITION
# Features: File Analysis + Supreme Court Precedent + Segmented Search

import streamlit as st
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 1. LOAD SECRETS
load_dotenv()

# 2. PAGE CONFIG
st.set_page_config(
    page_title="Vakalat AI | Legal Intelligence",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 8px; border: 1px solid #e0e0e0; }
    .stChatInput { position: fixed; bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

# 3. ENGINE SETUP
# REPLACE THIS FUNCTION IN main.py

# REPLACE THE ENTIRE 'get_vector_db' FUNCTION IN main.py WITH THIS:

@st.cache_resource
def get_vector_db():
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    embedding_function = OpenAIEmbeddings()
    
    # CHECK: Does the database exist?
    if not os.path.exists("./chroma_db"):
        st.warning("⚠️ Cloud Database missing. Building Brain from PDFs... (This takes ~2 mins)")
        
        all_docs = []
        
        # A. INGEST STATUTES (The PDFs you just uploaded)
        pdf_files = ["bns.pdf", "bnss.pdf", "bsa.pdf"]
        for pdf in pdf_files:
            if os.path.exists(pdf):
                st.info(f"📖 Reading {pdf}...")
                loader = PyMuPDFLoader(pdf)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_book"] = pdf
                all_docs.extend(docs)
        
        # Split the PDFs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        # B. INJECT SUPREME COURT PRECEDENTS (Hardcoded)
        # (We hardcode this so we don't need the 'judgments' folder on Cloud)
        sc_texts = [
            {
                "case": "Arnesh Kumar vs State of Bihar",
                "text": """SUPREME COURT GUIDELINES ON ARREST (Section 41A CrPC / Section 35 BNSS):
                1. No automatic arrest for offenses punishable with imprisonment less than 7 years.
                2. Police must issue a Notice of Appearance (Section 41A) first.
                3. Arrest is only allowed if the accused fails to comply or if there is a specific risk.
                4. Magistrate must not authorize detention mechanically."""
            },
            {
                "case": "D.K. Basu vs State of West Bengal",
                "text": """SUPREME COURT GUIDELINES ON CUSTODY & TORTURE:
                1. Police personnel must bear accurate name tags.
                2. Memo of arrest must be prepared and attested by a witness.
                3. Arrestee has right to inform a relative immediately.
                4. Medical examination every 48 hours."""
            }
        ]
        
        for sc in sc_texts:
            chunks.append(Document(
                page_content=sc["text"],
                metadata={"source_type": "case_law", "case_name": sc["case"]}
            ))

        # C. INJECT PATCH (Mob Lynching)
        chunks.append(Document(
            page_content="BNS Section 103(2) (Mob Lynching): When a group of five or more persons acting in concert commits murder on the ground of race, caste or community, sex, place of birth, language, personal belief or any other like ground, each member of such group shall be punished with death or with imprisonment for life, and shall also be liable to fine.",
            metadata={"source_book": "bns.pdf"} # Tagged as BNS so it shows up in search
        ))

        # D. BUILD & SAVE
        st.info("🧠 Indexing data... Please wait.")
        db = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db")
        st.success("✅ System Ready!")
        return db
        
    # Normal Load (if DB exists)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return db
try:
    vector_db = get_vector_db()
except Exception as e:
    st.error(f"❌ Database Error: {e}")
    st.stop()

# 4. FILE READER
def read_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# 5. SIDEBAR
with st.sidebar:
    st.title("⚖️ Vakalat AI")
    st.caption("Statute + Case Law Engine")
    st.markdown("---")
    
    st.subheader("📂 Case File Analysis")
    uploaded_file = st.file_uploader("Upload FIR / Charge Sheet (PDF)", type="pdf")
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")
        
    st.markdown("---")
    search_mode = st.radio("Research Depth:", ["Standard", "Deep"], horizontal=True)
    k_val = 4 if search_mode == "Standard" else 8
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 6. CORE LOGIC (THE BRAIN)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_legal_context(query, k=4):
    docs = []
    # A. Statutes (The Rules)
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_book": "bns.pdf"}))
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_book": "bnss.pdf"}))
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_book": "bsa.pdf"}))
    
    # B. Case Law (The Override) - CHECKS FOR ARNESH KUMAR ETC
    docs.extend(vector_db.similarity_search(query, k=2, filter={"source_type": "case_law"}))
    
    # C. Patches
    docs.extend(vector_db.similarity_search(query, k=2, filter={"source_book": "manual_patch_v1"}))
    
    return docs

# PROMPTS
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat AI, a senior legal consultant.
Use the Context (Statutes + Case Law) to answer.

CRITICAL RULES:
1. First, state the STATUTE (BNS/BNSS).
2. Second, check if any SUPREME COURT JUDGMENT (Case Law) overrides or clarifies it.
   - Specifically check for 'Arnesh Kumar' if the topic is Arrest.
   - Specifically check for 'D.K. Basu' if the topic is Custody.
3. If there is a conflict, Case Law prevails.

Context:
{context}

Question: {question}
Answer:
""")

analysis_prompt = ChatPromptTemplate.from_template("""
You are a Defense Lawyer analyzing a case file.

TASK:
1. Analyze the CLIENT CASE FILE facts.
2. Cross-reference with RETRIEVED LAWS (Statutes + Precedents).
3. Find Loopholes:
   - Does the FIR violate 'Arnesh Kumar' guidelines (Automatic arrest <7 years)?
   - Does the FIR violate 'D.K. Basu' (No arrest memo)?
   - Do the facts match the BNS Section ingredients?

CLIENT CASE FILE:
{case_file}

RETRIEVED LAWS:
{context}

USER QUESTION: {question}

ANALYSIS:
""")

# 7. UI LOGIC
st.subheader("Your Legal Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am ready. I know BNS, BNSS, BSA, and Supreme Court Guidelines (Arnesh Kumar/D.K. Basu)."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ex: 'Can police arrest for 3-year punishment?'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Consulting Statutes & Supreme Court..."):
            # A. Prepare Data
            case_text = ""
            if uploaded_file:
                case_text = read_pdf(uploaded_file)
                full_query = f"{user_input} {case_text[:1500]}"
            else:
                full_query = user_input

            # B. Search
            docs = get_legal_context(full_query, k=k_val)
            context_text = "\n\n".join(f"[Source: {d.metadata.get('source_book', d.metadata.get('case_name', 'Unknown'))}]\n{d.page_content}" for d in docs)
            
            # C. Generate
            if uploaded_file:
                chain = analysis_prompt | llm | StrOutputParser()
                response = chain.invoke({"case_file": case_text, "context": context_text, "question": user_input})
            else:
                chain = research_prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "question": user_input})
            
            message_placeholder.markdown(response)
            
            # D. Evidence Inspector
            with st.expander("🔍 Inspect Legal Sources"):
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('source_book') or doc.metadata.get('case_name') or "Unknown"
                    st.caption(f"**{i+1}. {source}**")
                    st.text(doc.page_content[:200] + "...")
                    st.divider()


    st.session_state.messages.append({"role": "assistant", "content": response})
