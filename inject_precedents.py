# inject_precedents.py
# BYPASSES PDF LOADING. Injects Supreme Court rules directly.

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

print("--- SURGICAL INJECTION: SUPREME COURT PRECEDENTS ---")

# 1. CONNECT TO BRAIN
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 2. THE DATA (Hardcoded Legal Rules)
judgments = [
    {
        "case_name": "Arnesh Kumar vs State of Bihar (2014)",
        "text": """
        SUPREME COURT GUIDELINES ON ARREST (Section 41A CrPC / Section 35 BNSS):
        1. No automatic arrest for offenses punishable with imprisonment less than 7 years.
        2. Police must issue a Notice of Appearance (Section 41A) first.
        3. Arrest is only allowed if the accused fails to comply with the notice or if there is a specific risk (fleeing/tampering).
        4. Magistrate must not authorize detention mechanically; they must verify if the police followed these checks.
        5. Failure to comply renders the police officer liable for departmental action and Contempt of Court.
        """
    },
    {
        "case_name": "D.K. Basu vs State of West Bengal (1997)",
        "text": """
        SUPREME COURT GUIDELINES ON CUSTODY & TORTURE:
        1. Police personnel must bear accurate, visible, and clear identification and name tags.
        2. A memo of arrest must be prepared at the time of arrest, attested by at least one witness (family or local).
        3. The arrestee has the right to have a friend or relative informed about the arrest immediately.
        4. The arrestee must be medically examined every 48 hours during detention.
        5. Copies of all documents must be sent to the Magistrate.
        """
    }
]

# 3. INJECT
docs = []
for judgment in judgments:
    doc = Document(
        page_content=judgment["text"],
        metadata={"source_type": "case_law", "case_name": judgment["case_name"]}
    )
    docs.append(doc)

db.add_documents(docs)
print(f"✅ SUCCESS: {len(docs)} Landmark Judgments injected into the Brain.")