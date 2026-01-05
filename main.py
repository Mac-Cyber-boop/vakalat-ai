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


# --- INSERT THIS AFTER set_page_config ---

# 1. THE BOUNCER (Password Check)
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Define your password here (or get from st.secrets for safety)
    # For now, let's keep it simple.
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input
        st.text_input(
            "🔒 Enter Access Code to enter Vakalat AI:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again
        st.text_input(
            "🔒 Enter Access Code to enter Vakalat AI:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("❌ Access Denied")
        return False
    else:
        # Password correct
        return True

# STOP EVERYTHING if password is wrong
if not check_password():
    st.stop()

# --- END OF BOUNCER ---
# (The rest of your code: Database setup, Sidebar, Chat... follows here)
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
        pdf_files = ["bns.pdf", "bnss.pdf", "bsa.pdf", "it_act.pdf"]
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

# --- A. THE CORTEX (Intent Classifier) ---
# This uses the AI to decide if we need the database, instead of hardcoded rules.
router_prompt = ChatPromptTemplate.from_template("""
Classify the user's query into one of two categories. Return ONLY the category name.

Categories:
1. LEGAL_RESEARCH: Questions about laws, crimes, punishments, police, FIRs, courts, bail, or specific sections.
2. GENERAL_CHAT: Greetings, asking "who are you", "what can you do", "help", or general conversation.

Query: {question}
Category:
""")
router_chain = router_prompt | llm | StrOutputParser()

# --- B. THE RETRIEVER (Search Engine) ---
def get_legal_context(query, k=4):
    docs = []
    # 1. Broad Statute Search
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_book": "bns.pdf"}))
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_book": "bnss.pdf"}))
    docs.extend(vector_db.similarity_search(query, k=k, filter={"source_book": "bsa.pdf"}))
    
    # 2. Smart Case Law Search
    trigger_words = ["arrest", "police", "custody", "bail", "detention", "torture", "handcuff", "remand", "investigation"]
    if any(word in query.lower() for word in trigger_words):
        docs.extend(vector_db.similarity_search(query, k=2, filter={"source_type": "case_law"}))
    
    # 3. Patch Search
    docs.extend(vector_db.similarity_search(query, k=2, filter={"source_book": "manual_patch_v1"}))
    return docs

# --- C. PROMPTS ---
# Prompt 1: For General Chat (No Database)
general_prompt = ChatPromptTemplate.from_template("""
You are Vakalat AI, a specialized legal research assistant for Indian Criminal Law (BNS, BNSS, BSA).
The user is asking a general question. Answer professionally and concisely.
Explain your capabilities (Case Analysis, Statute Search, Case Law Checks) only if asked.

User Query: {question}
Answer:
""")

# Prompt 2: For Legal Research (With Database)
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat AI, a senior legal consultant.
Use the Context (Statutes + Case Law) to answer.

CRITICAL RULES:
1. First, state the STATUTE (BNS/BNSS).
2. Second, check if any SUPREME COURT JUDGMENT (Case Law) overrides it.
3. If no relevant law is found in the context, admit it.

Context:
{context}

Question: {question}
Answer:
""")

# Prompt 3: For Case Analysis (With File)
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

# 7. UI LOGIC
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
        
        # DECISION TIME: What does the user want?
        with st.spinner("Thinking..."):
            # If a file is uploaded, FORCE legal analysis mode
            if uploaded_file:
                intent = "LEGAL_RESEARCH"
            else:
                intent = router_chain.invoke({"question": user_input}).strip()
        
        # BRANCH 1: GENERAL CHAT (Fast, No DB)
        if intent == "GENERAL_CHAT":
            chain = general_prompt | llm | StrOutputParser()
            response = chain.invoke({"question": user_input})
            message_placeholder.markdown(response)
            
        # BRANCH 2: LEGAL RESEARCH (Deep, Uses DB)
        else:
            with st.spinner("Consulting Legal Database..."):
                # A. Prepare Data
                case_text = ""
                if uploaded_file:
                    case_text = read_pdf(uploaded_file)
                    full_query = f"{user_input} {case_text[:1500]}"
                else:
                    full_query = user_input

                # B. Retrieve
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




