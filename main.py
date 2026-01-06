# main.py
# VAKALAT PRO: FINAL GOLD (v7.0)
# Features: Date-Awareness + Intelligent RAG + Drafting Studio

import streamlit as st
import os
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fpdf import FPDF
from gtts import gTTS
from io import BytesIO
from datetime import date

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Vakalat Pro | Legal OS", page_icon="⚖️", layout="wide")

if "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ Critical: Secrets missing.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    h1, h2, h3 { color: #D4AF37; font-family: 'Merriweather', serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1F242D; border-radius: 5px; color: #FAFAFA; }
    .stTabs [aria-selected="true"] { background-color: #D4AF37; color: #000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOGIC ENGINES
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

@st.cache_resource
def get_vector_store():
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        try: pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        except: pass
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

vector_db = get_vector_store()

def ingest_data():
    status = st.empty()
    status.info("🔍 Scanning Library...")
    from langchain_community.document_loaders import PyMuPDFLoader
    all_docs = []
    for root, dirs, files in os.walk("."):
        if ".git" in root: continue
        for file in files:
            if file.lower().endswith(".pdf"):
                try:
                    loader = PyMuPDFLoader(os.path.join(root, file))
                    docs = loader.load()
                    clean_source = file.replace(".pdf", "").replace("_", " ")
                    for doc in docs: doc.metadata = {"source": clean_source}
                    all_docs.extend(docs)
                except: pass
    
    if not all_docs: return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)
    vector_db.add_documents(splits)
    status.success(f"✅ Indexed {len(splits)} segments!")

# --- DATE-AWARE RESEARCH PROMPT ---
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat Pro, a Senior Legal Researcher.
TODAY'S DATE: {current_date}

Protocol:
1. Search the "LEGAL CONTEXT" for answers.
2. **Time-Bar Check:** If the user mentions dates (e.g., loan date), compare them with TODAY'S DATE. If the Limitation period has expired, EXPLICITLY WARN the user.
3. Cite sources for every claim using [Source: Filename].
4. If info is missing, say "Not found in database."

LEGAL CONTEXT:
{context}

QUERY: {question}

RESPONSE:
1. **Opinion:**
2. **Authority Footer:**
   • [Filename] – [Context]
""")

# B. DRAFTING BRAIN
drafting_prompt = ChatPromptTemplate.from_template("""
You are a Senior Drafter at a Top Law Firm.
Task: Draft a professional {doc_type}.

DETAILS:
{user_details}

REQUIREMENTS:
1. Use professional legal language (Whereas, Therefore, Hereby).
2. Cite relevant laws (e.g., Section 138 NI Act, Section 80 CPC) where applicable.
3. Be aggressive but polite (if it's a notice) or humble (if it's a petition).
4. Strict formatting.

DRAFT:
""")

# ---------------------------------------------------------
# 3. INTERFACE (TABS)
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=50)
    st.title("Vakalat Pro")
    st.caption("Legal OS v7.0")
    if st.button("☁️ Sync Library"): ingest_data()
    st.markdown("---")
    enable_hindi = st.toggle("🇮🇳 Hindi Mode")

tab1, tab2 = st.tabs(["🔍 Research", "📝 Drafting Studio"])

# --- TAB 1: RESEARCH ---
with tab1:
    col1, col2 = st.columns([1, 8])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/751/751463.png", width=50)
    with col2: st.header("Legal Research")
    
    if user_input := st.chat_input("Query Law Database..."):
        with st.spinner("Analyzing..."):
            results = vector_db.similarity_search(user_input, k=6)
            context_text = ""
            for doc in results: context_text += f"\n[Document: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}\n"
            
            chain = research_prompt | llm | StrOutputParser()
            # INJECT TODAY'S DATE HERE
            response = chain.invoke({
                "context": context_text, 
                "question": user_input, 
                "current_date": date.today()
            })
            
            if enable_hindi:
                trans = ChatPromptTemplate.from_template("Translate to Hindi. Keep citations in English.\n\n{text}") | llm | StrOutputParser()
                response = trans.invoke({"text": response})
            
            st.markdown(response)

# --- TAB 2: DRAFTING STUDIO ---
with tab2:
    col1, col2 = st.columns([1, 8])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=50)
    with col2: st.header("Drafting Studio")
    
    doc_type = st.selectbox("Select Document Type", [
        "Legal Notice (Recovery of Money)",
        "Legal Notice (Dishonour of Cheque - Sec 138)",
        "Writ Petition (General)",
        "Reply to Show Cause Notice",
        "Rent Agreement",
        "Custom Document (Describe below)"
    ])
    
    st.divider()
    
    with st.form("drafting_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            client_name = st.text_input("Client Name")
            opponent_name = st.text_input("Opponent Name")
        with col_b:
            date_of_incident = st.date_input("Date of Incident", date.today())
            amount = st.text_input("Amount Involved (₹)", "0")
            
        facts = st.text_area("Key Facts / Narrative", height=150, placeholder="E.g., We supplied goods on 1st Jan, Invoice #123. They have not paid despite reminders.")
        
        generate_btn = st.form_submit_button("⚡ Generate Draft")
    
    if generate_btn:
        with st.spinner("Drafting Document..."):
            details = f"""
            Client: {client_name}
            Opponent: {opponent_name}
            Date: {date_of_incident}
            Amount: {amount}
            Facts: {facts}
            """
            
            d_chain = drafting_prompt | llm | StrOutputParser()
            draft = d_chain.invoke({"doc_type": doc_type, "user_details": details})
            
            st.subheader("📄 Draft Preview")
            st.markdown(draft)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 6, draft.encode('latin-1', 'replace').decode('latin-1'))
            pdf_bytes = bytes(pdf.output(dest='S'))
            
            st.download_button("Download PDF", pdf_bytes, "Draft.pdf", "application/pdf")
