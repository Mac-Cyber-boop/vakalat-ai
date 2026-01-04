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

@st.cache_resource
def get_vector_db():
    embedding_function = OpenAIEmbeddings()
    
    # CLOUD FIX: If DB is missing, rebuild it on the fly
    if not os.path.exists("./chroma_db"):
        st.warning("⚠️ Database not found. Rebuilding Brain... (This takes 1 min)")
        
        # 1. Re-inject Supreme Court Rules (Hardcoded)
        from langchain_core.documents import Document
        judgments = [
            {"case_name": "Arnesh Kumar vs State of Bihar", "text": "SUPREME COURT GUIDELINES ON ARREST... [Insert full text from inject_precedents.py here]"},
            {"case_name": "D.K. Basu vs State of West Bengal", "text": "SUPREME COURT GUIDELINES ON CUSTODY... [Insert full text here]"}
        ]
        docs = [Document(page_content=j["text"], metadata={"source_type": "case_law", "case_name": j["case_name"]}) for j in judgments]
        
        # 2. Re-inject Patch
        patch = Document(page_content="BNS Section 103(2)... [Mob Lynching text]", metadata={"source_book": "manual_patch_v1"})
        docs.append(patch)
        
        # 3. Build DB
        db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
        st.success("✅ Brain Rebuilt!")
        return db
        
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