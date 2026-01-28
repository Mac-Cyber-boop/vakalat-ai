"""
Document reviser for iterative editing using the "edit trick" pattern.

Instead of regenerating the entire document (slow, expensive), this module:
1. Asks LLM to generate minimal edit instructions
2. Applies only necessary changes to the original
3. Preserves unchanged sections exactly

This approach is ~79% faster than full regeneration (6s vs 30s).

Usage:
    reviser = DocumentReviser()
    result = reviser.revise_document(
        original_content="...",
        user_instruction="Make the grounds more concise"
    )
    print(result.content)
"""

from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator


class DocumentEdit(BaseModel):
    """Represents a single edit operation on a document."""

    paragraph_number: int = Field(
        ge=1,
        description="1-indexed paragraph number to edit"
    )
    action: Literal["replace", "insert_after", "delete"] = Field(
        description="Edit action type"
    )
    new_content: Optional[str] = Field(
        default=None,
        description="New content (required for replace/insert_after)"
    )

    @model_validator(mode='after')
    def validate_content_for_action(self) -> 'DocumentEdit':
        """Ensure new_content provided when action requires it."""
        if self.action in ("replace", "insert_after") and not self.new_content:
            raise ValueError(f"new_content required for action '{self.action}'")
        return self


class EditList(BaseModel):
    """Container for list of edits (needed for structured outputs)."""
    edits: list[DocumentEdit] = Field(
        description="List of edit operations to apply"
    )


class RevisionResult(BaseModel):
    """Output of a document revision operation."""

    content: str = Field(
        description="Revised document content"
    )
    edits_applied: int = Field(
        description="Number of edit operations applied"
    )
    paragraphs_changed: list[int] = Field(
        default_factory=list,
        description="Paragraph numbers that were modified"
    )


EDIT_SYSTEM_PROMPT = """You are a legal document editor. Generate MINIMAL edits to apply the user's instruction.

Return edits as a JSON object with an "edits" array. Each edit specifies:
- paragraph_number: which paragraph to modify (1-indexed)
- action: "replace" (change content), "insert_after" (add new paragraph), "delete" (remove)
- new_content: the new text (for replace/insert_after)

RULES:
1. Make ONLY necessary changes - preserve as much original text as possible
2. Maintain formal legal tone ("Hon'ble Court", "most respectfully submitted")
3. Keep numbered paragraph structure intact
4. Maximum {max_edits} edits allowed
5. Paragraphs are separated by double newlines

DOCUMENT TO EDIT:
{document}

USER INSTRUCTION: {instruction}

Generate the minimal edits needed to fulfill this instruction."""


class DocumentReviser:
    """
    Enables iterative document editing using the "edit trick" pattern.

    Instead of regenerating the entire document for each revision,
    generates minimal edit instructions and applies them to the original.

    Example:
        reviser = DocumentReviser()
        result = reviser.revise_document(
            original_content="IN THE SUPREME COURT...",
            user_instruction="Make the grounds more concise"
        )
        # result.content has the revised document
        # result.edits_applied shows how many changes were made
    """

    def __init__(self, llm_client: Optional[OpenAI] = None):
        """
        Initialize DocumentReviser.

        Args:
            llm_client: OpenAI client. If None, creates default client.
        """
        if llm_client is None:
            self.llm_client = OpenAI()
        else:
            self.llm_client = llm_client

        self.model = "gpt-4o-2024-08-06"

    def revise_document(
        self,
        original_content: str,
        user_instruction: str,
        max_edits: int = 5
    ) -> RevisionResult:
        """
        Apply user's revision instruction to document.

        Uses "edit trick" pattern:
        1. Ask LLM to generate minimal edit instructions
        2. Apply edits to original document
        3. Return revised content with metadata

        Args:
            original_content: Current document content
            user_instruction: User's revision request (e.g., "Make grounds more concise")
            max_edits: Maximum edits to apply (prevents runaway changes)

        Returns:
            RevisionResult with revised content and edit metadata
        """
        if not original_content or not original_content.strip():
            return RevisionResult(
                content=original_content,
                edits_applied=0,
                paragraphs_changed=[]
            )

        # Build system prompt with document and instruction
        system_prompt = EDIT_SYSTEM_PROMPT.format(
            max_edits=max_edits,
            document=original_content,
            instruction=user_instruction
        )

        # Get edit instructions using structured outputs
        completion = self.llm_client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate edits for: {user_instruction}"}
            ],
            response_format=EditList
        )

        # Extract edits from response
        edit_list = completion.choices[0].message.parsed
        if edit_list is None or not edit_list.edits:
            return RevisionResult(
                content=original_content,
                edits_applied=0,
                paragraphs_changed=[]
            )

        # Limit edits to max_edits
        edits = edit_list.edits[:max_edits]

        # Apply edits to content
        revised_content, changed_paragraphs = self._apply_edits(
            original_content, edits
        )

        return RevisionResult(
            content=revised_content,
            edits_applied=len(edits),
            paragraphs_changed=changed_paragraphs
        )

    def _apply_edits(
        self,
        content: str,
        edits: list[DocumentEdit]
    ) -> tuple[str, list[int]]:
        """
        Apply edits to document content.

        Processes edits in reverse order (highest paragraph first)
        to prevent index shifting issues.

        Args:
            content: Original document content
            edits: List of edit operations to apply

        Returns:
            Tuple of (revised_content, changed_paragraph_numbers)
        """
        # Split content into paragraphs
        paragraphs = content.split("\n\n")
        changed = []

        # Sort edits by paragraph number descending to avoid index shifting
        sorted_edits = sorted(edits, key=lambda e: e.paragraph_number, reverse=True)

        for edit in sorted_edits:
            # Convert to 0-indexed
            idx = edit.paragraph_number - 1

            # Validate index is within bounds
            if idx < 0:
                continue

            if edit.action == "replace":
                if idx < len(paragraphs):
                    paragraphs[idx] = edit.new_content or ""
                    changed.append(edit.paragraph_number)

            elif edit.action == "insert_after":
                if idx < len(paragraphs):
                    paragraphs.insert(idx + 1, edit.new_content or "")
                    changed.append(edit.paragraph_number + 1)

            elif edit.action == "delete":
                if idx < len(paragraphs):
                    paragraphs.pop(idx)
                    changed.append(edit.paragraph_number)

        # Rejoin paragraphs
        revised_content = "\n\n".join(paragraphs)

        # Sort changed list for consistent output
        changed.sort()

        return revised_content, changed
