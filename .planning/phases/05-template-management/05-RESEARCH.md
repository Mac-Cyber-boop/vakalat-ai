# Phase 5: Template Management - Research

**Researched:** 2026-01-27
**Domain:** Template upload, validation, versioning, lifecycle management
**Confidence:** HIGH

## Summary

Phase 5 builds on the existing template infrastructure from Phase 2 to add user template management capabilities. The core technical challenges are: (1) handling JSON file uploads in FastAPI, (2) validating uploaded templates against the existing LegalTemplate Pydantic schema, (3) implementing semantic versioning with changelog tracking, (4) managing template lifecycle states with proper transition rules, and (5) generating template previews showing required inputs.

The existing codebase provides a strong foundation. The `src/templates/schemas.py` already defines `LegalTemplate`, `TemplateMetadata`, `FormattingRequirements`, and `TemplateField` models with comprehensive Pydantic validation. The `TemplateRepository` in `storage.py` provides CRUD operations on JSON files. Phase 5 extends these with upload endpoints, version tracking, lifecycle state management, and preview generation.

FastAPI file upload with `UploadFile` combined with Pydantic's `model_validate_json()` provides a clean pattern for validating user-uploaded templates. Semantic versioning can be implemented with the `semver` library. Lifecycle states (active/deprecated/archived) are best implemented as a simple Enum with validation rules for allowed transitions, avoiding the complexity of a full state machine library.

**Primary recommendation:** Extend existing `LegalTemplate` schema to include lifecycle status and changelog fields, implement upload endpoint using `UploadFile` + `model_validate_json()`, use `semver` library for version parsing/comparison, and add custom validators for lifecycle state transitions.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Pydantic | 2.x | Template schema validation | Already in codebase, `model_validate_json()` for file validation |
| FastAPI | Latest | File upload endpoints | Already in codebase, `UploadFile` for efficient file handling |
| python-multipart | Latest | Form/file data parsing | Required by FastAPI for file uploads (not in requirements.txt) |
| semver | 3.x | Semantic version parsing/comparison | Industry standard, simple API for version operations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python Enum | stdlib | Lifecycle status enum | Type-safe status values with Pydantic integration |
| datetime | stdlib | Changelog timestamps | ISO format timestamps for version history |
| pathlib | stdlib | File path operations | Already used in TemplateRepository |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| semver library | Manual version comparison | Library handles edge cases, prerelease versions |
| Simple Enum + validators | python-statemachine | Full state machine is overkill for 3 states with simple transitions |
| JSON file storage | Database | Phase 2 established JSON for transparency; maintain consistency |

**Installation:**
```bash
pip install python-multipart semver
```

## Architecture Patterns

### Recommended Extension to Project Structure
```
src/
├── templates/
│   ├── schemas.py          # Extended with TemplateStatus, VersionHistory
│   ├── storage.py          # Extended with version-aware operations
│   ├── upload.py           # NEW: Upload validation and processing
│   ├── versioning.py       # NEW: Version comparison, changelog management
│   ├── preview.py          # NEW: Template preview generation
│   └── data/
│       └── *.json          # Templates with added status, changelog fields
```

### Pattern 1: Template Upload with Pydantic Validation
**What:** Use FastAPI `UploadFile` to receive JSON file, then validate with Pydantic `model_validate_json()`
**When to use:** All template upload operations
**Example:**
```python
# Source: https://fastapi.tiangolo.com/tutorial/request-files/
# Source: https://docs.pydantic.dev/latest/examples/files/
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import ValidationError

from src.templates.schemas import LegalTemplate

@app.post("/templates/upload")
async def upload_template(file: UploadFile):
    # Validate content type
    if file.content_type != "application/json":
        raise HTTPException(400, "File must be JSON")

    # Read file contents
    contents = await file.read()

    # Validate against schema
    try:
        template = LegalTemplate.model_validate_json(contents)
    except ValidationError as e:
        # Return actionable error messages
        errors = [
            {"field": ".".join(map(str, err["loc"])), "message": err["msg"]}
            for err in e.errors()
        ]
        raise HTTPException(400, {"errors": errors})

    return {"status": "valid", "template_name": template.metadata.name}
```

### Pattern 2: Lifecycle Status Enum with Transition Validation
**What:** Define allowed lifecycle states and valid transitions using Enum + custom validator
**When to use:** All template status changes
**Example:**
```python
# Source: https://docs.pydantic.dev/latest/concepts/validators/
from enum import Enum
from pydantic import BaseModel, model_validator

class TemplateStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

# Valid transitions: active -> deprecated -> archived
# Cannot go backwards (archived template stays archived)
ALLOWED_TRANSITIONS = {
    TemplateStatus.ACTIVE: {TemplateStatus.DEPRECATED, TemplateStatus.ARCHIVED},
    TemplateStatus.DEPRECATED: {TemplateStatus.ARCHIVED},
    TemplateStatus.ARCHIVED: set(),  # Terminal state
}

class TemplateStatusChange(BaseModel):
    current_status: TemplateStatus
    new_status: TemplateStatus

    @model_validator(mode='after')
    def validate_transition(self) -> 'TemplateStatusChange':
        allowed = ALLOWED_TRANSITIONS.get(self.current_status, set())
        if self.new_status not in allowed and self.current_status != self.new_status:
            raise ValueError(
                f"Cannot transition from {self.current_status.value} to {self.new_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        return self
```

### Pattern 3: Semantic Versioning with Changelog
**What:** Track version history with semantic versioning and changelog entries
**When to use:** All template updates
**Example:**
```python
# Source: https://semver.org/
# Source: https://pypi.org/project/semver/
import semver
from pydantic import BaseModel, field_validator
from typing import List
from datetime import datetime

class ChangelogEntry(BaseModel):
    version: str
    date: str
    changes: List[str]
    author: str = "system"

    @field_validator('version')
    @classmethod
    def validate_semver(cls, v: str) -> str:
        try:
            semver.Version.parse(v)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {v}")
        return v

class VersionedTemplate(BaseModel):
    # ... existing fields ...
    version: str  # "1.0.0"
    changelog: List[ChangelogEntry] = []

    def bump_version(self, bump_type: str, changes: List[str]) -> 'VersionedTemplate':
        """Create new version with changelog entry."""
        current = semver.Version.parse(self.version)
        if bump_type == "major":
            new_ver = current.bump_major()
        elif bump_type == "minor":
            new_ver = current.bump_minor()
        else:
            new_ver = current.bump_patch()

        new_entry = ChangelogEntry(
            version=str(new_ver),
            date=datetime.now().isoformat(),
            changes=changes
        )

        return self.model_copy(update={
            "version": str(new_ver),
            "changelog": [new_entry] + self.changelog
        })
```

### Pattern 4: Template Preview Generation
**What:** Generate preview showing required/optional fields without full template content
**When to use:** Template selection UI, before document generation
**Example:**
```python
from pydantic import BaseModel
from typing import List

class FieldPreview(BaseModel):
    field_name: str
    label: str
    field_type: str
    required: bool
    description: str
    example: str | None

class TemplatePreview(BaseModel):
    doc_type: str
    court_level: str
    name: str
    version: str
    status: str
    description: str
    required_field_count: int
    optional_field_count: int
    required_fields: List[FieldPreview]
    optional_fields: List[FieldPreview]

def generate_preview(template: LegalTemplate) -> TemplatePreview:
    """Generate preview from full template."""
    return TemplatePreview(
        doc_type=template.metadata.doc_type.value,
        court_level=template.metadata.court_level.value,
        name=template.metadata.name,
        version=template.metadata.version,
        status=template.status.value if hasattr(template, 'status') else "active",
        description=template.metadata.description,
        required_field_count=len(template.required_fields),
        optional_field_count=len(template.optional_fields),
        required_fields=[
            FieldPreview(
                field_name=f.field_name,
                label=f.label,
                field_type=f.field_type,
                required=True,
                description=f.description,
                example=f.example
            )
            for f in template.required_fields
        ],
        optional_fields=[
            FieldPreview(
                field_name=f.field_name,
                label=f.label,
                field_type=f.field_type,
                required=False,
                description=f.description,
                example=f.example
            )
            for f in template.optional_fields
        ]
    )
```

### Anti-Patterns to Avoid
- **Storing uploaded files without validation:** Always validate JSON against schema before saving
- **Allowing arbitrary version jumps:** Enforce sequential versioning (1.0.0 -> 1.0.1, not 1.0.0 -> 3.0.0)
- **Skipping lifecycle transition validation:** Archived templates should not become active again
- **Returning full template for preview:** Preview should be lightweight, not include template_content
- **Using database for template storage:** Maintain JSON file pattern from Phase 2 for consistency

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Version comparison | String comparison | `semver.Version.parse()` and comparison operators | Handles prerelease, build metadata correctly |
| JSON validation | Manual field checking | `model_validate_json()` | Gives detailed error messages, type coercion |
| File upload parsing | Manual multipart parsing | FastAPI `UploadFile` | Memory-efficient streaming for large files |
| Version string parsing | Regex parsing | `semver.Version.parse()` | Handles edge cases like "1.0.0-alpha.1+build.123" |
| Error message formatting | Custom error building | Pydantic `ValidationError.errors()` | Structured errors with field paths |

**Key insight:** The combination of FastAPI's `UploadFile`, Pydantic's `model_validate_json()`, and the `semver` library covers all validation needs. Focus implementation effort on business logic (lifecycle rules, changelog format) not parsing/validation infrastructure.

## Common Pitfalls

### Pitfall 1: Missing python-multipart Dependency
**What goes wrong:** FastAPI endpoint for file upload fails with error about form data
**Why it happens:** `python-multipart` is not in requirements.txt but required for file uploads
**How to avoid:** Add `python-multipart` to requirements.txt before implementing upload endpoint
**Warning signs:** Runtime error mentioning "form data" or "multipart"

### Pitfall 2: Overwriting Existing Templates Without Version Check
**What goes wrong:** User uploads template that silently replaces existing one
**Why it happens:** No check for existing template with same doc_type/court_level
**How to avoid:** Check existence first, require explicit version bump for updates
**Warning signs:** Templates disappearing or reverting unexpectedly

### Pitfall 3: Allowing Invalid State Transitions
**What goes wrong:** Archived template becomes active, deprecated template becomes active
**Why it happens:** No validation of transition rules
**How to avoid:** Implement ALLOWED_TRANSITIONS map and validate before state change
**Warning signs:** Template lifecycle becomes inconsistent, deprecated templates reappear

### Pitfall 4: Losing Changelog on Template Updates
**What goes wrong:** Version increments but changelog not updated, history lost
**Why it happens:** Changelog update not atomic with version bump
**How to avoid:** Make version bump and changelog entry creation a single atomic operation
**Warning signs:** Templates have version 2.3.0 but only 2 changelog entries

### Pitfall 5: Blocking Document Generation During Deprecation
**What goes wrong:** Deprecating a template breaks all documents using it
**Why it happens:** Deprecated treated as "unavailable" instead of "warning"
**How to avoid:** Deprecated templates remain usable with warning; only archived blocks usage
**Warning signs:** User complaints about documents failing after template deprecation

### Pitfall 6: Returning Full Template Content in Preview
**What goes wrong:** Preview endpoint is slow, returns too much data
**Why it happens:** Preview returns full LegalTemplate including template_content
**How to avoid:** Create separate TemplatePreview model with only preview-relevant fields
**Warning signs:** Large response payloads for template list/preview endpoints

## Code Examples

Verified patterns from official sources:

### Complete Upload Endpoint with Validation
```python
# Source: https://fastapi.tiangolo.com/tutorial/request-files/
# Source: https://docs.pydantic.dev/latest/examples/files/
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import ValidationError
import json

from src.templates.schemas import LegalTemplate, TemplateStatus
from src.templates.storage import TemplateRepository

MAX_FILE_SIZE = 1024 * 1024  # 1MB limit for template JSON

@app.post("/templates/upload")
async def upload_template(file: UploadFile):
    """
    Upload and validate a custom template.

    Returns validation results with specific error messages if invalid.
    """
    # Validate content type
    if not file.content_type or "json" not in file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File must be JSON (application/json)"
        )

    # Read with size limit
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )

    # Validate JSON syntax
    try:
        json.loads(contents)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON syntax at line {e.lineno}: {e.msg}"
        )

    # Validate against schema
    try:
        template = LegalTemplate.model_validate_json(contents)
    except ValidationError as e:
        errors = [
            {
                "field": ".".join(map(str, err["loc"])),
                "message": err["msg"],
                "type": err["type"]
            }
            for err in e.errors()
        ]
        raise HTTPException(
            status_code=400,
            detail={"validation_errors": errors}
        )

    # Check for existing template
    repo = TemplateRepository()
    existing = repo.get_template(
        template.metadata.doc_type,
        template.metadata.court_level
    )

    if existing:
        # Validate version is higher
        import semver
        existing_ver = semver.Version.parse(existing.metadata.version)
        new_ver = semver.Version.parse(template.metadata.version)
        if new_ver <= existing_ver:
            raise HTTPException(
                status_code=400,
                detail=f"Version {template.metadata.version} must be higher than existing {existing.metadata.version}"
            )

    # Save template
    filepath = repo.save_template(template)

    return {
        "status": "success",
        "message": f"Template uploaded successfully",
        "template": {
            "doc_type": template.metadata.doc_type.value,
            "court_level": template.metadata.court_level.value,
            "version": template.metadata.version,
            "name": template.metadata.name
        },
        "is_update": existing is not None
    }
```

### Lifecycle Status Change Endpoint
```python
# Source: https://docs.pydantic.dev/latest/concepts/validators/
from pydantic import BaseModel, Field
from enum import Enum

class TemplateStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class StatusChangeRequest(BaseModel):
    doc_type: str
    court_level: str
    new_status: TemplateStatus
    reason: str = Field(min_length=10, description="Reason for status change")

@app.post("/templates/status")
async def change_template_status(req: StatusChangeRequest):
    """
    Change template lifecycle status.

    Valid transitions:
    - active -> deprecated (template still usable with warning)
    - active -> archived (template no longer usable)
    - deprecated -> archived (template no longer usable)
    """
    repo = TemplateRepository()
    template = repo.get_template(
        DocumentType(req.doc_type),
        CourtLevel(req.court_level)
    )

    if not template:
        raise HTTPException(404, "Template not found")

    current_status = getattr(template, 'status', TemplateStatus.ACTIVE)

    # Validate transition
    ALLOWED_TRANSITIONS = {
        TemplateStatus.ACTIVE: {TemplateStatus.DEPRECATED, TemplateStatus.ARCHIVED},
        TemplateStatus.DEPRECATED: {TemplateStatus.ARCHIVED},
        TemplateStatus.ARCHIVED: set(),
    }

    if req.new_status not in ALLOWED_TRANSITIONS.get(current_status, set()):
        if req.new_status != current_status:
            raise HTTPException(
                400,
                f"Cannot transition from {current_status.value} to {req.new_status.value}"
            )

    # Update template with new status and changelog entry
    # ... implementation details ...

    return {
        "status": "success",
        "old_status": current_status.value,
        "new_status": req.new_status.value,
        "reason": req.reason
    }
```

### Version History Query
```python
from typing import List
from pydantic import BaseModel

class VersionHistoryEntry(BaseModel):
    version: str
    date: str
    changes: List[str]
    status_at_version: str

class VersionHistoryResponse(BaseModel):
    doc_type: str
    court_level: str
    current_version: str
    history: List[VersionHistoryEntry]

@app.post("/templates/history")
async def get_version_history(req: GetTemplateRequest):
    """
    Get version history with changelog for a template.
    """
    repo = TemplateRepository()
    template = repo.get_template(
        DocumentType(req.doc_type),
        CourtLevel(req.court_level)
    )

    if not template:
        raise HTTPException(404, "Template not found")

    changelog = getattr(template, 'changelog', [])

    return VersionHistoryResponse(
        doc_type=req.doc_type,
        court_level=req.court_level,
        current_version=template.metadata.version,
        history=[
            VersionHistoryEntry(
                version=entry.version,
                date=entry.date,
                changes=entry.changes,
                status_at_version=entry.status if hasattr(entry, 'status') else "active"
            )
            for entry in changelog
        ]
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual version strings | semver library | 2020+ | Reliable version comparison, prerelease handling |
| Full state machine libraries | Simple Enum + validators | N/A | Reduced complexity for simple state graphs |
| UploadFile.file.read() | await UploadFile.read() | FastAPI 0.89+ | Proper async file reading |
| Pydantic v1 `.parse_raw()` | Pydantic v2 `.model_validate_json()` | Pydantic 2.0 (2023) | Cleaner API, better error messages |
| Custom JSON parsing | Pydantic ValidationError | Pydantic 2.0 | Structured errors with field paths |

**Deprecated/outdated:**
- Pydantic v1 `parse_raw()` method: Use `model_validate_json()` in v2
- Pydantic v1 `@validator`: Use `@field_validator` in v2
- Pydantic v1 `class Config`: Use `model_config = ConfigDict()` in v2
- Synchronous file reading with UploadFile: Use async `await file.read()`

## Open Questions

Things that couldn't be fully resolved:

1. **Custom Template Filename Strategy**
   - What we know: Phase 2 uses `{doc_type}_{court_level}.json` naming
   - What's unclear: Should user-uploaded templates use different naming (e.g., `custom_{doc_type}_{court_level}.json`)?
   - Recommendation: Keep same naming; user templates override defaults (latest version wins)

2. **Multi-User Template Isolation**
   - What we know: Current system has no user-specific template storage
   - What's unclear: Should user-uploaded templates be visible to all users or isolated?
   - Recommendation: Phase 5 focuses on single shared template library; defer user isolation to v2

3. **Template Content Migration**
   - What we know: Existing 12 templates lack status/changelog fields
   - What's unclear: Should migration add these fields, or leave existing templates unchanged?
   - Recommendation: Add optional fields with defaults; existing templates treated as "active" with empty changelog

4. **Rollback to Previous Version**
   - What we know: Version history can be tracked
   - What's unclear: Should users be able to rollback to previous template versions?
   - Recommendation: Defer rollback feature; focus on forward versioning for Phase 5

## Sources

### Primary (HIGH confidence)
- FastAPI Request Files - https://fastapi.tiangolo.com/tutorial/request-files/
- Pydantic File Validation - https://docs.pydantic.dev/latest/examples/files/
- Pydantic Validators - https://docs.pydantic.dev/latest/concepts/validators/
- Semantic Versioning Spec - https://semver.org/
- semver PyPI - https://pypi.org/project/semver/

### Secondary (MEDIUM confidence)
- Pydantic Enums - https://docs.pydantic.dev/latest/api/standard_library_types/
- python-statemachine States - https://python-statemachine.readthedocs.io/en/latest/states.html
- FastAPI File Upload Guide - https://betterstack.com/community/guides/scaling-python/uploading-files-using-fastapi/

### Tertiary (LOW confidence)
- pytransitions library - https://github.com/pytransitions/transitions
- React JSON Schema Form - https://rjsf-team.github.io/react-jsonschema-form/docs/api-reference/uiSchema/

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastAPI and Pydantic docs verified, semver is well-established
- Architecture patterns: HIGH - Patterns derived from official docs and existing codebase patterns
- Lifecycle management: HIGH - Simple Enum approach verified, transition validation is standard Pydantic
- Pitfalls: MEDIUM - Based on common FastAPI/Pydantic issues, verified against docs
- Preview generation: MEDIUM - Pattern is straightforward but specific requirements may vary

**Research date:** 2026-01-27
**Valid until:** 2026-02-27 (30 days - stable libraries, well-established patterns)
