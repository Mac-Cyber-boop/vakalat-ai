# 04-04 Summary: DocumentReviser

## What Was Built
DocumentReviser class implementing the "edit trick" pattern for iterative document editing. Instead of regenerating entire documents (~30s), generates minimal edit instructions and applies them (~6s) - a 79% improvement.

## Files Changed
| File | Change |
|------|--------|
| src/generation/reviser.py | Created with DocumentEdit, EditList, RevisionResult models and DocumentReviser class |
| src/generation/__init__.py | Added exports for DocumentReviser, DocumentEdit, RevisionResult |

## Key Implementation Details

### DocumentEdit Model
Represents a single edit operation:
- `paragraph_number`: 1-indexed paragraph to edit
- `action`: "replace" | "insert_after" | "delete"
- `new_content`: Content for replace/insert_after (validated)

Uses `@model_validator` to ensure `new_content` provided when action requires it.

### RevisionResult Model
Output of revision:
- `content`: Revised document content
- `edits_applied`: Number of edit operations applied
- `paragraphs_changed`: List of paragraph numbers modified

### DocumentReviser Class

**Constructor:**
```python
def __init__(self, llm_client: Optional[OpenAI] = None)
```

**Main Method - revise_document:**
```python
def revise_document(
    self,
    original_content: str,
    user_instruction: str,
    max_edits: int = 5
) -> RevisionResult
```

**Edit Trick Pattern:**
1. Build system prompt with document + instruction
2. Use OpenAI structured outputs (`beta.chat.completions.parse`) to get edit list
3. Apply edits in reverse order (highest paragraph first) to avoid index shifting
4. Return revised content with metadata

### Edit Application Logic
- Split content on `\n\n` to identify paragraphs
- Sort edits by paragraph_number descending
- Apply each edit (replace/insert_after/delete)
- Rejoin paragraphs

## Commits
- `0baa5a5`: feat(04-04): add DocumentReviser with edit trick pattern

## Verification Results
- [x] DocumentEdit model validates correctly (requires content for replace/insert_after)
- [x] RevisionResult model validates correctly
- [x] DocumentReviser imports without error
- [x] Uses gpt-4o-2024-08-06 model
- [x] Module exports work: `from src.generation import DocumentReviser, DocumentEdit, RevisionResult`

## Requirements Addressed
- DOC-04: User can edit and revise generated documents iteratively
- Preserved sections maintained exactly (edit trick preserves unchanged paragraphs)
- Formal legal tone maintained via system prompt instructions
