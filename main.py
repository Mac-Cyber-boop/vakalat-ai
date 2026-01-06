# main.py
# VAKALAT PRO: COMPLETE SUITE (v11.0)
# Features: Logic Gates + Drafting Studio + Case Watch (Simulated) + Secure Access

import streamlit as st
import os
import time
import random
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF
from datetime import date, timedelta

# ---------------------------------------------------------
# 1. SETUP & AUTHENTICATION
# ---------------------------------------------------------
st.set_page_config(page_title="Vakalat Pro | Legal OS", page_icon="⚖️", layout="wide")

if "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ Critical Error: PINECONE_API_KEY is missing.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# --- PASSWORD GATE ---
def check_password():
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
    return True

if not check_password(): st.stop()

# Theme Styling
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    h1, h2, h3 { color: #D4AF37; font-family: 'Merriweather', serif; }
    div.stButton > button { background: linear-gradient(to right, #D4AF37, #C5A028); color: #000; border: none; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1F242D; border-radius: 5px; color: #FAFAFA; }
    .stTabs [aria-selected="true"] { background-color: #D4AF37; color: #000; font-weight: bold; }
    .metric-card { background-color: #1F242D; padding: 15px; border-radius: 10px; border-left: 4px solid #D4AF37; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. INTELLIGENCE ENGINES
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

@st.cache_resource
def get_vector_store():
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        try: 
            pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
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
    
    if not all_docs: status.warning("⚠️ No PDFs found."); return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)
    vector_db.add_documents(splits)
    status.success(f"✅ Indexed {len(splits)} segments! Brain Updated.")

# --- RESEARCH LOGIC (AGGRESSIVE) ---
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat Pro, a Senior Legal Consultant.
TODAY'S DATE: {current_date}

CRITICAL LOGIC GATES (Overrides Retrieval):
1. **Commercial Disputes (>3 Lakhs):**
   - Rule: Pre-Institution Mediation (Sec 12A) is MANDATORY.
   - Trap: If user wants to file "immediately" without Mediation, the Answer is **NO**, unless they seek "Urgent Interim Relief".
2. **Cheque Bounce (Sec 138 NI Act):**
   - Rule: Filing before 15-day notice period expires is ILLEGAL.
   - Logic: Notice Date + 15 Days = Earliest Filing Date.
3. **Limitation Act:**
   - Rule: Debt recovery limit is 3 Years.
   - Logic: If Incident Date > 3 years ago, Answer is **NO** (Time-Barred).

LEGAL CONTEXT:
{context}

QUERY: {question}

RESPONSE:
1. **Direct Answer:** (Must be "NO" if Logic Gate fails).
2. **Procedural Check:**
3. **Authority Footer:**
   • [Exact Filename] – [Section]
""")

# --- DRAFTING LOGIC ---
drafting_prompt = ChatPromptTemplate.from_template("""
You are a Senior Drafter. Draft a professional {doc_type}.
DETAILS: {user_details}
REQUIREMENTS: Professional legal language. Cite laws. Strict formatting.
DRAFT:
""")

# --- CASE TRACKING SIMULATION ---
def fetch_case_status(c_type, c_num, c_year):
    """
    SIMULATION MODE: Returns realistic dummy data.
    Replace this function with real API calls (e.g., Legistify/RapidAPI) in Production.
    """
    time.sleep(1.5) # Simulate Network Latency
    
    # Randomly generate a status for demo
    statuses = ["Pending Hearing", "Disposed", "Order Reserved", "Awaiting Summons"]
    stages = ["Arguments", "Evidence", "Admission", "Final Hearing"]
    next_dates = [date.today() + timedelta(days=random.randint(5, 60)) for _ in range(5)]
    
    return {
        "status": random.choice(statuses),
        "stage": random.choice(stages),
        "next_date": random.choice(next_dates).strftime("%d %B %Y"),
        "court": "High Court of Delhi",
        "judge": "Hon'ble Justice S. Sharma",
        "petitioner": "Tech Solutions Pvt Ltd",
        "respondent": "State of NCT & Ors"
    }

# ---------------------------------------------------------
# 3. INTERFACE
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=50)
    st.title("Vakalat Pro")
    st.caption("Secure Enterprise | v11.0")
    if st.button("☁️ Sync Library"): ingest_data()
    st.markdown("---")
    enable_hindi = st.toggle("🇮🇳 Hindi Mode")

tab1, tab2, tab3 = st.tabs(["🔍 Research", "📝 Drafting Studio", "📡 Case Watch"])

# --- TAB 1: RESEARCH ---
with tab1:
    col1, col2 = st.columns([1, 8])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/751/751463.png", width=50)
    with col2: st.header("Legal Research")
    
    if user_input := st.chat_input("Query Law Database..."):
        with st.spinner("Running Logic Gates & Analyzing..."):
            results = vector_db.similarity_search(user_input, k=8)
            context_text = ""
            for doc in results: 
                context_text += f"\n[Document: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}\n"
            
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
        "Legal Notice (Dishonour of Cheque)",
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

# --- TAB 3: CASE WATCH (NEW) ---
with tab3:
    col1, col2 = st.columns([1, 8])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/3222/3222625.png", width=50)
    with col2: st.header("Case Watch (Live Radar)")
    
    st.info("ℹ️ Connected to: National Judicial Data Grid (Simulated Mode)")
    
    with st.form("case_track_form"):
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1: case_type = st.selectbox("Case Type", ["W.P.(C) - Writ Petition", "CS(COMM) - Commercial Suit", "Bail App"])
        with c_col2: case_num = st.text_input("Case Number", "1234")
        with c_col3: case_year = st.text_input("Year", "2024")
        
        track_btn = st.form_submit_button("📡 Track Case")
    
    if track_btn:
        with st.spinner("Fetching Live Status from Court Server..."):
            data = fetch_case_status(case_type, case_num, case_year)
            
            st.markdown("### 🏛️ Case Status Report")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-card'><b>Next Hearing</b><br><h2 style='color:#D4AF37'>{data['next_date']}</h2></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-card'><b>Current Stage</b><br><h3>{data['stage']}</h3></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><b>Status</b><br><h3>{data['status']}</h3></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-card'><b>Court</b><br><h5>{data['court']}</h5></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.write(f"**Parties:** {data['petitioner']} V/s {data['respondent']}")
            st.write(f"**Coram:** {data['judge']}")
            st.success("✅ Data Sync Complete")
