"""
Document generator orchestrating template-first legal document generation.

Combines:
- Template loading (Phase 2 TemplateRepository)
- Citation recommendation (Phase 3 CitationRecommender)
- Citation verification (Phase 1 CitationGate)
- LLM content generation with formal legal language

Pipeline:
1. Load template from repository
2. Retrieve relevant citations (if recommender available)
3. Generate formal content for each text/relief field via LLM
4. Render template with filled fields (Jinja2)
5. Verify citations in output (if gate available)
6. Sanitize blocked citations if needed
7. Return GeneratedDocument with metadata
"""

from typing import Literal, Optional

from jinja2 import Template
from openai import OpenAI
from pydantic import BaseModel, Field

from src.citations.recommender import CitationRecommendation, CitationRecommender
from src.generation.prompts import get_generation_prompt
from src.templates.schemas import CourtLevel, DocumentType, LegalTemplate
from src.templates.storage import TemplateRepository
from src.verification.citation_gate import CitationGate, FilteredCitations


class GeneratedDocument(BaseModel):
    """Output model for generated legal documents."""

    content: str = Field(
        description="Complete rendered document content"
    )
    doc_type: str = Field(
        description="Document type that was generated"
    )
    court_level: str = Field(
        description="Court level for formatting"
    )
    citations_used: list[str] = Field(
        default_factory=list,
        description="Citations integrated into content"
    )
    verification_status: Literal["verified", "sanitized"] = Field(
        description="Citation verification outcome"
    )
    blocked_citations: int = Field(
        default=0,
        description="Count of citations blocked by verification"
    )
    formatting: dict = Field(
        default_factory=dict,
        description="Court-specific formatting metadata"
    )


class DocumentGenerator:
    """
    Orchestrates template-first legal document generation.

    Combines:
    - Template loading (Phase 2 TemplateRepository)
    - Citation recommendation (Phase 3 CitationRecommender)
    - Citation verification (Phase 1 CitationGate)
    - LLM content generation with formal legal language

    Example:
        generator = DocumentGenerator(template_repo, citation_recommender, citation_gate)
        result = generator.generate_document(
            doc_type=DocumentType.BAIL_APPLICATION,
            court_level=CourtLevel.SUPREME_COURT,
            user_facts={"applicant_name": "John Doe", ...}
        )
    """

    # Field types that require LLM expansion
    LLM_FIELD_TYPES = {"text", "relief"}

    # Field types that use user input directly
    DIRECT_FIELD_TYPES = {"party", "date", "case_number", "court"}

    def __init__(
        self,
        template_repo: TemplateRepository,
        citation_recommender: Optional[CitationRecommender] = None,
        citation_gate: Optional[CitationGate] = None,
        llm_client: Optional[OpenAI] = None
    ):
        """
        Initialize the DocumentGenerator.

        Args:
            template_repo: TemplateRepository instance for loading templates
            citation_recommender: Optional CitationRecommender for retrieving relevant citations
            citation_gate: Optional CitationGate for verifying citations in output
            llm_client: Optional OpenAI client. If None, creates default client.
        """
        self.template_repo = template_repo
        self.citation_recommender = citation_recommender
        self.citation_gate = citation_gate

        # Initialize LLM client
        if llm_client is None:
            self.llm_client = OpenAI()
        else:
            self.llm_client = llm_client

        self.model = "gpt-4o-2024-08-06"  # Not deprecated gpt-4o

    def generate_document(
        self,
        doc_type: DocumentType,
        court_level: CourtLevel,
        user_facts: dict
    ) -> GeneratedDocument:
        """
        Generate a legal document using template-first approach.

        Pipeline:
        1. Load template from repository
        2. Retrieve relevant citations (if recommender available)
        3. Generate formal content for each text/relief field via LLM
        4. Render template with filled fields (Jinja2)
        5. Verify citations in output (if gate available)
        6. Sanitize blocked citations if needed
        7. Return GeneratedDocument with metadata

        Args:
            doc_type: Type of document to generate (bail_application, legal_notice, etc.)
            court_level: Court level for formatting (supreme_court, high_court, district_court)
            user_facts: Dictionary of user-provided facts matching template fields

        Returns:
            GeneratedDocument with content, citations, and verification status

        Raises:
            ValueError: If template not found or required fields missing
        """
        # Step 1: Load template
        template = self.template_repo.get_template(doc_type, court_level)
        if template is None:
            raise ValueError(
                f"Template not found for {doc_type.value} at {court_level.value}"
            )

        # Step 2: Retrieve citations for legal issue (if recommender available)
        citations: list[CitationRecommendation] = []
        if self.citation_recommender is not None:
            # Extract legal issue from user facts
            legal_issue = self._extract_legal_issue(user_facts, doc_type)
            if legal_issue:
                citations = self.citation_recommender.recommend_precedents(
                    legal_issue=legal_issue,
                    filing_court=court_level.value,
                    top_k=5
                )

        # Step 3: Generate content for each field
        filled_fields = {}
        citations_used = []

        for field in template.required_fields:
            field_name = field.field_name
            user_input = user_facts.get(field_name, "")

            if not user_input and field.required:
                raise ValueError(f"Required field '{field_name}' not provided")

            if field.field_type in self.LLM_FIELD_TYPES:
                # Generate formal content via LLM
                generated_content = self._generate_field_content(
                    field_name=field_name,
                    user_input=str(user_input),
                    court_level=court_level,
                    citations=citations
                )
                filled_fields[field_name] = generated_content

                # Track citations that were available for this field
                citations_used.extend([c.formatted_citation for c in citations])
            else:
                # Use user input directly for party, date, case_number, court fields
                filled_fields[field_name] = str(user_input)

        # Also process optional fields if provided
        for field in template.optional_fields:
            field_name = field.field_name
            user_input = user_facts.get(field_name)

            if user_input:
                if field.field_type in self.LLM_FIELD_TYPES:
                    filled_fields[field_name] = self._generate_field_content(
                        field_name=field_name,
                        user_input=str(user_input),
                        court_level=court_level,
                        citations=citations
                    )
                else:
                    filled_fields[field_name] = str(user_input)

        # Step 4: Render template with Jinja2
        jinja_template = Template(template.template_content)
        rendered_content = jinja_template.render(**filled_fields)

        # Step 5 & 6: Verify and sanitize citations if gate available
        verification_status: Literal["verified", "sanitized"] = "verified"
        blocked_count = 0

        if self.citation_gate is not None:
            filtered = self.citation_gate.filter_all_citations(rendered_content)

            if filtered.blocked:
                rendered_content = self.citation_gate.sanitize_output(
                    rendered_content, filtered
                )
                verification_status = "sanitized"
                blocked_count = len(filtered.blocked)

            # Update citations_used to only include verified ones
            citations_used = filtered.verified

        # Deduplicate citations used
        citations_used = list(set(citations_used))

        # Step 7: Return GeneratedDocument
        return GeneratedDocument(
            content=rendered_content,
            doc_type=doc_type.value,
            court_level=court_level.value,
            citations_used=citations_used,
            verification_status=verification_status,
            blocked_citations=blocked_count,
            formatting=template.formatting.model_dump()
        )

    def _generate_field_content(
        self,
        field_name: str,
        user_input: str,
        court_level: CourtLevel,
        citations: list[CitationRecommendation]
    ) -> str:
        """
        Generate formal legal content for a template field.

        Uses get_generation_prompt() to build system prompt with
        base tone, court-specific requirements, field-specific guidance,
        and citation integration instructions.

        Args:
            field_name: Name of the template field being generated
            user_input: User's plain language input to expand
            court_level: Court level for court-specific language
            citations: List of citations to potentially integrate

        Returns:
            Formal legal content generated by LLM
        """
        # Extract citation strings for prompt builder
        citation_strings = [c.formatted_citation for c in citations] if citations else None

        # Build system prompt with citations included
        system_prompt = get_generation_prompt(
            court_level=court_level,
            field_name=field_name,
            citations=citation_strings  # Pass citations to prompt builder
        )

        # Call LLM with system prompt containing citation guidance
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        return response.choices[0].message.content or ""

    def _extract_legal_issue(self, user_facts: dict, doc_type: DocumentType) -> str:
        """
        Extract legal issue from user facts for citation retrieval.

        Args:
            user_facts: Dictionary of user-provided facts
            doc_type: Type of document being generated

        Returns:
            Legal issue string for citation recommendation
        """
        # Priority fields that likely contain the legal issue
        issue_fields = [
            "grounds_for_bail",
            "grounds_summary",
            "legal_grounds",
            "subject_matter",
            "facts_summary",
            "purpose",
        ]

        for field in issue_fields:
            if field in user_facts and user_facts[field]:
                return str(user_facts[field])

        # Fallback: combine doc_type with any available description
        return f"{doc_type.value} legal matter"
