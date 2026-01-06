# main.py
# VAKALAT PRO: INTELLIGENT RAG (v5.0)
# Features: Smart Retrieval + "Cite-or-Die" Verification + Pinecone

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
# 1. SETUP & CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Vakalat Pro | Legal OS", page_icon="⚖️", layout="wide")

if "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ Critical: Secrets missing.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Smart UI
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1F242D; border: 1px solid #30363D; border-radius: 10px; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #131720; border-left: 4px solid #D4AF37; border-radius: 5px; }
    h1 { color: #D4AF37; font-family: 'Merriweather', serif; }
    div.stButton > button { background: linear-gradient(to right, #D4AF37, #C5A028); color: #000; border: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. VECTOR DATABASE (Pinecone)
# ---------------------------------------------------------
@st.cache_resource
def get_vector_store():
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Auto-create index if missing
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        try:
            pc.create_index(
                name=INDEX_NAME, 
                dimension=1536, 
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except: pass
            
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

vector_db = get_vector_store()

def ingest_data():
    """Smart Ingestion: Recursively finds PDFs and syncs to Cloud."""
    status = st.empty()
    status.info("🔍 Scanning project for Law Books...")
    
    from langchain_community.document_loaders import PyMuPDFLoader
    all_docs = []
    
    for root, dirs, files in os.walk("."):
        if ".git" in root: continue
        for file in files:
            if file.lower().endswith(".pdf"):
                try:
                    path = os.path.join(root, file)
                    loader = PyMuPDFLoader(path)
                    docs = loader.load()
                    # Store clean filename as metadata
                    clean_source = file.replace(".pdf", "").replace("_", " ")
                    for doc in docs: doc.metadata = {"source": clean_source}
                    all_docs.extend(docs)
                    st.toast(f"Found: {clean_source}")
                except: pass

    if not all_docs:
        status.error("No PDFs found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)
    
    status.info(f"🚀 Indexing {len(splits)} legal segments to Neural Cloud...")
    vector_db.add_documents(splits)
    status.success("✅ Knowledge Base Updated!")

# ---------------------------------------------------------
# 3. THE "CITE-OR-DIE" PROMPT (Intelligent + Strict)
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# This prompt forces the AI to be smart but requires proof.
research_prompt = ChatPromptTemplate.from_template("""
You are Vakalat Pro, an expert Legal Research AI.

Your Intelligence Protocol:
1. **Analyze** the user's query legally (understand intent, legal concepts).
2. **Search** the provided "LEGAL CONTEXT" for answers.
3. **Verify** facts: You must cite the specific document for every major legal claim.
4. **Reject** Hallucinations: If the answer is not in the Context, explicitly state: "The uploaded documents do not contain information regarding [Topic]." DO NOT USE OUTSIDE KNOWLEDGE.

STRICT CITATION RULE:
Every legal assertion must be followed by its source in brackets.
Example: "The limitation period is 3 years [Source: Limitation Act 1963]."

LEGAL CONTEXT (From Database):
{context}

USER QUERY: {question}

RESPONSE FORMAT:
1. **Direct Opinion:** (Smart synthesis of the laws found)
2. **Key Provisions:** (Bullet points of sections found)
3. **Authority Footer:** (List the exact files used)

Authority:
• [Exact Filename] – [Section/Context]
""")

# ---------------------------------------------------------
# 4. INTERFACE
# ---------------------------------------------------------
with st.sidebar:
    st.title("Vakalat Pro")
    st.caption("Intelligent RAG Engine")
    if st.button("☁️ Update Knowledge Base"): ingest_data()
    st.markdown("---")
    enable_hindi = st.toggle("🇮🇳 Hindi Mode")
    enable_audio = st.toggle("🔊 Audio Mode")

# MAIN CHAT
col1, col2 = st.columns([1, 8])
with col1: st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=60)
with col2: st.title("Vakalat Pro"); st.markdown("*Advanced Legal Intelligence*")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Vakalat Pro Online. Ready for complex legal research."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ex: What is the limitation period for property suits?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting Legal Matrix..."):
            
            # 1. SMART RETRIEVAL (No score filter, let the LLM judge)
            # We fetch top 6 matches to give the AI enough context to "think"
            results = vector_db.similarity_search(user_input, k=6)
            
            # Build Context with Source Tags
            context_text = ""
            sources_found = set()
            for doc in results:
                src = doc.metadata.get("source", "Unknown Document")
                sources_found.add(src)
                context_text += f"\n[Document: {src}]\n{doc.page_content}\n"
            
            # 2. GENERATION
            chain = research_prompt | llm | StrOutputParser()
            final_response = chain.invoke({"context": context_text, "question": user_input})

            # 3. TRANSLATION (Optional)
            if enable_hindi:
                trans_chain = ChatPromptTemplate.from_template("Translate to formal Hindi. Keep citations like [Source: ...] in English.\n\n{text}") | llm | StrOutputParser()
                final_response = trans_chain.invoke({"text": final_response})
            
            st.markdown(final_response)

            # 4. AUDIO
            if enable_audio:
                try:
                    tts_lang = 'hi' if enable_hindi else 'en'
                    tts = gTTS(text=final_response, lang=tts_lang, slow=False)
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    st.audio(audio_bytes, format='audio/mp3')
                except: pass

    st.session_state.messages.append({"role": "assistant", "content": final_response})
