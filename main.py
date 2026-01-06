# main.py
# VAKALAT PRO: SAAS UI EDITION (v12.0)
# Features: Premium UI + Logic Gates + Case Watch + Secure Access

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
# 1. CONFIG & AUTH
# ---------------------------------------------------------
st.set_page_config(
    page_title="Vakalat Pro | Legal OS", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Secrets Check
if "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ Critical: PINECONE_API_KEY missing.")
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
        st.markdown("<h2 style='text-align:center; color:#D4AF37;'>Vakalat Pro Enterprise</h2>", unsafe_allow_html=True)
        st.text_input("Enter Access Token", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Access Token", type="password", on_change=password_entered, key="password")
        st.error("❌ Access Denied")
        return False
    return True

if not check_password(): st.stop()

# ---------------------------------------------------------
# 2. THE "SAAS" UI THEME (CSS INJECTION)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* GLOBAL RESET */
    * { font-family: 'Inter', sans-serif !important; }
    
    /* APP BACKGROUND */
    .stApp {
        background-color: #0F172A; /* Slate 900 */
        color: #E2E8F0; /* Slate 200 */
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #1E293B; /* Slate 800 */
        border-right: 1px solid #334155;
    }
    
    /* REMOVE TOP PADDING */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #FBBF24 !important; /* Amber 400 */
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* BUTTONS (Gradient) */
    div.stButton > button {
        background: linear-gradient(135deg, #B45309 0%, #D97706 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3);
    }
    
    /* INPUT FIELDS */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stDateInput > div > div > input {
        background-color: #1E293B;
        color: white;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FBBF24;
    }
    
    /* CHAT BUBBLES */
    .stChatMessage {
        background-color: #1E293B !important;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E293B;
        padding: 0.5rem;
        border-radius: 12px;
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 8px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F172A;
        color: #FBBF24;
        font-weight: bold;
    }
    
    /* METRIC CARDS (DASHBOARD) */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #FBBF24;
    }
    
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. LOGIC & BRAIN
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
    with st.sidebar:
        status = st.empty()
        status.info("⚙️ Syncing...")
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
        status.success(f"✅ Synced {len(splits)} docs")

# LOGIC GATES (Prompt)
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat Pro, a Senior Legal Consultant.
TODAY'S DATE: {current_date}

CRITICAL LOGIC GATES:
1. **Commercial Disputes:** Pre-Institution Mediation (Sec 12A) is MANDATORY. Answer **NO** if skipped, unless "Urgent Relief" is sought.
2. **Cheque Bounce:** Filing before 15-day notice expiry is ILLEGAL.
3. **Limitation:** Debt > 3 years old is TIME-BARRED (Answer NO).

CONTEXT: {context}
QUERY: {question}

RESPONSE:
1. **Direct Opinion:** (Start with YES/NO).
2. **Analysis:** (Explain Logic/Procedure).
3. **Authority:**
   • [Filename] – [Section]
""")

# DRAFTING (Prompt)
drafting_prompt = ChatPromptTemplate.from_template("""
Role: Senior Drafter.
Doc Type: {doc_type}.
Details: {user_details}.
Rules: Professional tone, relevant citations, aggressive but polite.
DRAFT:
""")

# CASE SIMULATION
def fetch_case_status(c_type, c_num, c_year):
    time.sleep(1) 
    return {
        "status": random.choice(["Pending Hearing", "Disposed", "Order Reserved"]),
        "stage": random.choice(["Arguments", "Evidence", "Admission"]),
        "next_date": (date.today() + timedelta(days=random.randint(5, 60))).strftime("%d %b %Y"),
        "court": "High Court of Delhi"
    }

# ---------------------------------------------------------
# 4. MAIN APP LAYOUT
# ---------------------------------------------------------

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=60)
    st.markdown("### Vakalat Pro")
    st.caption("Enterprise OS v12.0")
    st.markdown("---")
    
    st.markdown("**Library Admin**")
    if st.button("🔄 Sync Cloud Database"): ingest_data()
    
    st.markdown("---")
    st.markdown("**Settings**")
    enable_hindi = st.toggle("🇮🇳 Hindi Output")
    
    st.markdown("---")
    st.info("🔒 Secure Connection\n\nLicense: Enterprise")

# MAIN CONTENT
tab1, tab2, tab3 = st.tabs(["🔎 Intelligence", "✍️ Drafting Studio", "📡 Case Radar"])

# --- TAB 1: INTELLIGENCE ---
with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### Legal Intelligence Engine")
        st.markdown("*Ask about Case Laws, Procedures, or limitations.*")
    
    if user_input := st.chat_input("Ex: Can I file a suit without mediation?"):
        with st.spinner("Processing..."):
            results = vector_db.similarity_search(user_input, k=6)
            ctx = "\n".join([f"[Doc: {d.metadata.get('source','Unknown')}]\n{d.page_content}" for d in results])
            
            chain = research_prompt | llm | StrOutputParser()
            resp = chain.invoke({"context": ctx, "question": user_input, "current_date": date.today()})
            
            if enable_hindi:
                trans = ChatPromptTemplate.from_template("Translate to Hindi (Legal).\n\n{text}") | llm | StrOutputParser()
                resp = trans.invoke({"text": resp})
            
            st.markdown(resp)

# --- TAB 2: DRAFTING ---
with tab2:
    st.markdown("### Drafting Studio")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        doc_type = st.selectbox("Document Type", ["Legal Notice (Money)", "Sec 138 Notice", "Writ Petition", "Rent Agreement"])
        client_name = st.text_input("Client Name")
        opponent_name = st.text_input("Opponent Name")
        amount = st.text_input("Amount (₹)")
    
    with c2:
        date_incident = st.date_input("Incident Date")
        facts = st.text_area("Case Facts", height=200, placeholder="Describe the sequence of events...")
        if st.button("⚡ Generate Draft Document"):
            with st.spinner("Drafting..."):
                det = f"Client: {client_name}\nOpp: {opponent_name}\nDate: {date_incident}\nAmt: {amount}\nFacts: {facts}"
                draft = (drafting_prompt | llm | StrOutputParser()).invoke({"doc_type": doc_type, "user_details": det})
                st.markdown("---")
                st.markdown(draft)
                
                pdf = FPDF()
                pdf.add_page(); pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 6, draft.encode('latin-1','replace').decode('latin-1'))
                st.download_button("⬇️ Download PDF", bytes(pdf.output(dest='S')), "Draft.pdf", "application/pdf")

# --- TAB 3: CASE RADAR ---
with tab3:
    st.markdown("### Case Watch Radar")
    
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: ct = st.selectbox("Type", ["W.P.(C)", "CS(COMM)", "Bail App"])
        with c2: cn = st.text_input("Case No")
        with c3: cy = st.text_input("Year", "2024")
        with c4: 
            st.write("")
            st.write("")
            track = st.button("Track")
            
    if track:
        d = fetch_case_status(ct, cn, cy)
        st.markdown("---")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><h5>Next Date</h5><h2 style='color:#FBBF24'>{d['next_date']}</h2></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><h5>Stage</h5><h3>{d['stage']}</h3></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><h5>Status</h5><h3>{d['status']}</h3></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><h5>Court</h5><h4>Delhi HC</h4></div>", unsafe_allow_html=True)
