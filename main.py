# main.py
# VAKALAT PRO: ENTERPRISE EDITION (v3.0)
# Features: Pinecone Cloud DB + Civil Law Engine + Strict Citations

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

# ---------------------------------------------------------
# 1. AUTHENTICATION & SETUP
# ---------------------------------------------------------

st.set_page_config(page_title="Vakalat Pro | Legal OS", page_icon="⚖️", layout="wide")

# Load Secrets
if "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ Critical Error: PINECONE_API_KEY missing in Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# CSS Styling (Dark/Gold Theme)
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1F242D; border: 1px solid #30363D; border-radius: 10px; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #131720; border-left: 4px solid #D4AF37; border-radius: 5px; }
    h1, h2, h3 { font-family: 'Merriweather', serif; color: #E6EDF3; }
    h1 { color: #D4AF37; font-weight: 700; }
    div.stButton > button { background: linear-gradient(to right, #D4AF37, #C5A028); color: #0E1117; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# Password Protection
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Access Code:", type="password", key="password_input", on_change=validate_password)
        return False
    return st.session_state["password_correct"]

def validate_password():
    if st.session_state["password_input"] == st.secrets["APP_PASSWORD"]:
        st.session_state["password_correct"] = True
    else:
        st.session_state["password_correct"] = False
        st.error("❌ Access Denied")

if not check_password():
    st.stop()

# ---------------------------------------------------------
# 2. CLOUD BRAIN (PINECONE ENGINE)
# ---------------------------------------------------------

@st.cache_resource
def get_vector_store():
    """Connects to the Pinecone Cloud Index."""
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.toast("✅ Created New Cloud Index", icon="☁️")
        except Exception as e:
            st.error(f"Failed to create index: {e}")
            st.stop()
            
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

vector_db = get_vector_store()

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS (Ingestion, PDF, Audio)
# ---------------------------------------------------------

def ingest_data():
    """Scans ALL folders (root, library, judgments) and uploads PDFs to Pinecone."""
    status_container = st.empty()
    status_container.info("☁️ Starting Cloud Sync... Scanning all directories.")
    
    from langchain_community.document_loaders import PyMuPDFLoader
    
    all_docs = []
    total_files = 0
    
    # RECURSIVE SCAN (Finds your 20 Acts wherever they are)
    for root, dirs, files in os.walk("."):
        if ".git" in root: continue # Skip git internal files
        
        for file in files:
            if file.lower().endswith(".pdf"):
                filepath = os.path.join(root, file)
                try:
                    loader = PyMuPDFLoader(filepath)
                    docs = loader.load()
                    
                    # Clean filename for metadata
                    clean_name = file.replace(".pdf", "").replace("_", " ").title()
                    for doc in docs:
                        doc.metadata = {"source": clean_name}
                    
                    all_docs.extend(docs)
                    total_files += 1
                    st.toast(f"📖 Queued: {clean_name}")
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    if not all_docs:
        status_container.warning("⚠️ No PDFs found in the project.")
        return

    # Split & Upload
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)
    
    status_container.info(f"🚀 Uploading {len(splits)} segments from {total_files} Acts to Cloud...")
    
    # Add to Pinecone
    try:
        vector_db.add_documents(splits)
        status_container.success(f"✅ Sync Complete! {total_files} Acts are now live in the Brain.")
    except Exception as e:
        status_container.error(f"Upload Failed: {e}")

def create_pdf(query, response, sources):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Vakalat Pro | Legal Opinion', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'AI Generated. Not a substitute for professional counsel.', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, f"Ref: {query[:200]}...")
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    safe_resp = response.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_resp)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Sources Used:", 0, 1)
    pdf.set_font("Arial", "I", 9)
    for doc in sources:
        src = doc.metadata.get('source', 'Unknown')
        safe_src = src.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 6, f"- {safe_src}", 0, 1)
    return bytes(pdf.output(dest='S'))

# ---------------------------------------------------------
# 4. INTELLIGENCE LAYER (Prompts)
# ---------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# STRICT "AUTHORITY" PROMPT
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat Pro, a Senior Legal Consultant.

STRICT RESPONSE STRUCTURE:
1. **Direct Answer:** Start with the specific provision/law.
2. **Analysis:** Apply the law to the user's query.
3. **MANDATORY FOOTER:** You MUST end every single response with the exact citation format below.

MANDATORY FOOTER FORMAT:
Authority:
• [Act Name, Year] – [Section Number]
• [Case Name] – [Ratio/One-line summary]

CONTEXT FROM DATABASE:
{context}

USER QUERY: {question}

YOUR LEGAL OPINION:
""")

# ---------------------------------------------------------
# 5. UI & SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=50)
    st.title("Vakalat Pro")
    st.caption("Enterprise Edition | v3.0")
    
    st.markdown("---")
    st.subheader("📂 Document Lab")
    uploaded_file = st.file_uploader("Upload File (Writ/Notice)", type="pdf")
    
    st.markdown("---")
    st.subheader("⚙️ Admin Console")
    # This button finds your 20 Acts and uploads them
    if st.button("☁️ Sync Library to Cloud"):
        ingest_data() 
    
    st.markdown("---")
    st.subheader("🗣️ Output Options")
    enable_hindi = st.toggle("🇮🇳 Reply in Hindi")
    enable_audio = st.toggle("🔊 Audio Mode")

# ---------------------------------------------------------
# 6. MAIN CHAT ENGINE
# ---------------------------------------------------------

col1, col2 = st.columns([1, 8])
with col1: st.image("https://cdn-icons-png.flaticon.com/512/2237/2237599.png", width=60)
with col2: 
    st.title("Vakalat Pro")
    st.markdown("*Civil, Criminal & Corporate Intelligence System*")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Online. Connected to Pinecone Legal Grid."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about Contract Act, GST, or upload a Notice..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            
            # 1. RETRIEVAL
            docs = []
            context_text = ""
            
            if uploaded_file:
                # Session Mode (Single File)
                try:
                    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                        text = ""
                        for page in doc: text += page.get_text()
                    context_text = text[:15000] 
                    docs = [Document(page_content="Uploaded File Analysis", metadata={"source": uploaded_file.name})]
                except:
                    st.error("Error reading file.")
            else:
                # Cloud Mode (Search Pinecone)
                docs = vector_db.similarity_search(user_input, k=5)
                context_text = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])

            # 2. GENERATION
            chain = research_prompt | llm | StrOutputParser()
            raw_response = chain.invoke({"context": context_text, "question": user_input})
            
            # 3. HINDI TRANSLATION
            final_response = raw_response
            if enable_hindi:
                trans_chain = ChatPromptTemplate.from_template("Translate to Formal Legal Hindi. Keep citations in English.\n\n{text}") | llm | StrOutputParser()
                final_response = trans_chain.invoke({"text": raw_response})
            
            st.markdown(final_response)
            
            # 4. AUDIO
            if enable_audio:
                try:
                    tts = gTTS(text=final_response, lang='hi' if enable_hindi else 'en', slow=False)
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    st.audio(audio_bytes, format='audio/mp3')
                except: pass
                
            # 5. PDF REPORT
            try:
                pdf_data = create_pdf(user_input, final_response, docs)
                st.download_button("📄 Download Legal Opinion", data=pdf_data, file_name="Legal_Opinion.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": final_response})
