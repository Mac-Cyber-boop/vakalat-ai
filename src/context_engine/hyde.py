"""
HyDE — Hypothetical Document Embeddings for legal retrieval.

Problem with standard RAG:
    User query: "can anticipatory bail be cancelled after grant?"
    → This is a question. Pinecone compares it against answer-shaped legal text.
    → The embedding spaces don't align well.

HyDE solution:
    Generate a hypothetical ideal answer first → embed that → retrieve.
    → The hypothetical answer uses legal language ("Section 438", "Sessions Court",
      "Nimmagadda Prasad") that closely matches actual statute/judgment text.

This significantly improves recall for legal queries, especially procedural ones
where the user's natural language question is far from the document's formal language.

Usage:
    hyde = HyDERetriever(vector_db, llm)
    docs = hyde.retrieve("can anticipatory bail be cancelled?", k=10)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI


_HYDE_PROMPT = ChatPromptTemplate.from_template("""
You are a senior Indian lawyer. Write a short, formal legal passage (3-5 sentences)
that DIRECTLY ANSWERS the following legal question. Use precise legal language,
relevant section numbers, and case names if applicable. Do NOT add caveats or say
"it depends" — write as if this is an excerpt from a legal textbook or judgment.

Question: {query}

Passage:
""")


class HyDERetriever:
    """
    Improves retrieval by embedding a hypothetical answer instead of the raw query.

    Steps:
        1. LLM generates a hypothetical legal passage for the query.
        2. That passage is embedded (it uses formal legal language).
        3. The embedding is used to search Pinecone (better alignment with docs).
        4. Returns real retrieved documents — NOT the hypothetical answer.

    Args:
        vector_store: Connected PineconeVectorStore instance.
        llm: ChatOpenAI instance (should be fast/cheap — gpt-4o-mini is fine here).
    """

    def __init__(self, vector_store: PineconeVectorStore, llm: ChatOpenAI):
        self.vector_store = vector_store
        self._chain = _HYDE_PROMPT | llm | StrOutputParser()

    def retrieve(
        self,
        query: str,
        k: int = 10,
        source_type_filter: str | None = None,
    ) -> list:
        """
        Retrieve documents using HyDE.

        Args:
            query: The user's natural language legal question.
            k: Number of documents to retrieve.
            source_type_filter: Optional Pinecone metadata filter
                                 (e.g., "statute" or "case_law").

        Returns:
            List of Langchain Document objects from Pinecone.
        """
        # Generate hypothetical answer
        hypothetical_passage = self._chain.invoke({"query": query})

        # Use it as the search query (not the original user question)
        search_kwargs = {"k": k}
        if source_type_filter:
            search_kwargs["filter"] = {"source_type": {"$eq": source_type_filter}}

        return self.vector_store.similarity_search(
            query=hypothetical_passage,
            **search_kwargs,
        )

    async def aretrieve(
        self,
        query: str,
        k: int = 10,
        source_type_filter: str | None = None,
    ) -> list:
        """Async version of retrieve."""
        hypothetical_passage = await self._chain.ainvoke({"query": query})

        search_kwargs = {"k": k}
        if source_type_filter:
            search_kwargs["filter"] = {"source_type": {"$eq": source_type_filter}}

        return self.vector_store.similarity_search(
            query=hypothetical_passage,
            **search_kwargs,
        )
