# main.py
# VAKALAT PRO: LOGIC PATCH (v8.0)
# Fixes: Cheque Bounce Math, Mandatory Mediation, and Date Logic

import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF
from gtts import gTTS
from io import BytesIO
from datetime import date, timedelta

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

# --- LOGIC ENHANCED PROMPT ---
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat Pro, a Senior Legal Consultant.
TODAY'S DATE: {current_date}

CRITICAL LOGIC CHECKS (Must Apply):
1. **Commercial Disputes:** If the user asks about a business/vendor dispute > ₹3 Lakhs, CHECK for "Pre-Institution Mediation" (Section 12A, Commercial Courts Act). If they skip it, warn them the suit will be rejected.
2. **Cheque Bounce (Sec 138 NI Act):**
   - Step 1: Notice Service Date.
   - Step 2: ADD 15 Days (Waiting Period).
   - Step 3: Cause of Action arises ONLY on Day 16.
   - **Rule:** A complaint filed BEFORE Day 16 is PREMATURE and illegal (Yogendra Pratap Singh judgment).
3. **Limitation Act:** Compare Incident Date with Today's Date. If > 3 years (for debt), it is Time-Barred.

LEGAL CONTEXT FROM DB:
{context}

QUERY: {question}

RESPONSE STRUCTURE:
1. **Direct Answer:** (Yes/No/Maybe with reasoning).
2. **Procedural Check:** (Did the user miss Mediation? Is the date premature?).
3. **Authority Footer:**
   • [Exact Filename] – [Section/Context]
""")

# DRAFTING BRAIN
drafting_prompt = ChatPromptTemplate.from_template("""
You are a Senior Drafter. Draft a {doc_type}.
DETAILS: {user_details}
REQUIREMENTS: Professional legal language. Cite laws. Strict formatting.
DRAFT:
""")

# ---------------------------------------------------------
# 3. INTERFACE
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=50)
    st.title("Vakalat Pro")
    st.caption("Legal OS v8.0")
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
        with st.spinner("Analyzing Law & Procedure..."):
            # Increased k to 8 to catch Commercial Courts Act if present
            results = vector_db.similarity_search(user_input, k=8)
            context_text = ""
            for doc in results: context_text += f"\n[Document: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}\n"
            
            chain = research_prompt | llm | StrOutputParser()
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
        "Rent Agreement",
        "Custom Document"
    ])
    st.divider()
    with st.form("drafting_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            client_name = st.text_input("Client Name")
            opponent_name = st.text_input("Opponent Name")
        with col_b:
            date_of_incident = st.date_input("Date of Incident", date.today())
            amount = st.text_input("Amount (₹)", "0")
        facts = st.text_area("Facts", height=150)
        generate_btn = st.form_submit_button("⚡ Generate Draft")
    
    if generate_btn:
        with st.spinner("Drafting..."):
            details = f"Client: {client_name}\nOpponent: {opponent_name}\nDate: {date_of_incident}\nAmount: {amount}\nFacts: {facts}"
            d_chain = drafting_prompt | llm | StrOutputParser()
            draft = d_chain.invoke({"doc_type": doc_type, "user_details": details})
            st.markdown(draft)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 6, draft.encode('latin-1', 'replace').decode('latin-1'))
            st.download_button("Download PDF", bytes(pdf.output(dest='S')), "Draft.pdf", "application/pdf")
