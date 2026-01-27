"""
Version comparison and changelog management utilities for template versioning.

Provides:
- Semantic version comparison using semver library
- Version bump utilities (major, minor, patch)
- Changelog entry creation with ISO timestamps
- Version upgrade validation for template updates
"""

from datetime import datetime
from typing import List, Literal, Tuple, Optional

import semver

from src.templates.schemas import ChangelogEntry


def compare_versions(version_a: str, version_b: str) -> int:
    """
    Compare two semantic versions.

    Args:
        version_a: First version string (e.g., "1.0.0")
        version_b: Second version string (e.g., "1.1.0")

    Returns:
        -1 if version_a < version_b
         0 if version_a == version_b
         1 if version_a > version_b

    Raises:
        ValueError: If either version is not a valid semantic version
    """
    try:
        ver_a = semver.Version.parse(version_a)
        ver_b = semver.Version.parse(version_b)
    except ValueError as e:
        raise ValueError(f"Invalid semantic version: {e}")

    return ver_a.compare(ver_b)


def is_version_higher(new_version: str, existing_version: str) -> bool:
    """
    Check if new version is higher than existing version.

    Args:
        new_version: The proposed new version
        existing_version: The current version to compare against

    Returns:
        True if new_version > existing_version, False otherwise

    Raises:
        ValueError: If either version is not a valid semantic version
    """
    return compare_versions(new_version, existing_version) > 0


def bump_version(
    version: str,
    bump_type: Literal["major", "minor", "patch"] = "patch"
) -> str:
    """
    Increment a semantic version by the specified bump type.

    Args:
        version: Current version string (e.g., "1.2.3")
        bump_type: Type of version bump - "major", "minor", or "patch"

    Returns:
        New version string after bump

    Raises:
        ValueError: If version is invalid or bump_type is not recognized

    Examples:
        >>> bump_version("1.2.3", "patch")
        "1.2.4"
        >>> bump_version("1.2.3", "minor")
        "1.3.0"
        >>> bump_version("1.2.3", "major")
        "2.0.0"
    """
    try:
        ver = semver.Version.parse(version)
    except ValueError as e:
        raise ValueError(f"Invalid semantic version '{version}': {e}")

    if bump_type == "major":
        new_ver = ver.bump_major()
    elif bump_type == "minor":
        new_ver = ver.bump_minor()
    elif bump_type == "patch":
        new_ver = ver.bump_patch()
    else:
        raise ValueError(f"Invalid bump_type: {bump_type}. Must be 'major', 'minor', or 'patch'")

    return str(new_ver)


def create_changelog_entry(
    version: str,
    changes: List[str],
    author: str = "system"
) -> ChangelogEntry:
    """
    Create a new changelog entry with current ISO timestamp.

    Args:
        version: Semantic version for this entry
        changes: List of change descriptions
        author: Who made the changes (default: "system")

    Returns:
        ChangelogEntry with ISO timestamp

    Raises:
        ValueError: If version is not valid semver or changes is empty
    """
    if not changes:
        raise ValueError("Changes list cannot be empty")

    # Validate version (ChangelogEntry validator will also check this)
    try:
        semver.Version.parse(version)
    except ValueError:
        raise ValueError(f"Invalid semantic version: {version}. Use format like '1.0.0'")

    return ChangelogEntry(
        version=version,
        date=datetime.now().isoformat(),
        changes=changes,
        author=author
    )


def validate_version_upgrade(
    new_version: str,
    existing_version: str,
    require_higher: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a version upgrade is valid.

    Checks:
    1. Both versions are valid semantic versions
    2. If require_higher=True, new version must be higher than existing

    Args:
        new_version: Proposed new version
        existing_version: Current version
        require_higher: Whether new version must be higher (default: True)

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    """
    # Validate new version format
    try:
        new_ver = semver.Version.parse(new_version)
    except ValueError:
        return (False, f"Invalid new version format: '{new_version}'. Use semantic versioning (e.g., '1.0.0')")

    # Validate existing version format
    try:
        existing_ver = semver.Version.parse(existing_version)
    except ValueError:
        return (False, f"Invalid existing version format: '{existing_version}'")

    # Check version ordering if required
    if require_higher:
        if new_ver <= existing_ver:
            return (
                False,
                f"New version '{new_version}' must be higher than existing version '{existing_version}'"
            )

    return (True, None)


def get_version_parts(version: str) -> Tuple[int, int, int]:
    """
    Extract major, minor, patch components from a version string.

    Args:
        version: Semantic version string

    Returns:
        Tuple of (major, minor, patch)

    Raises:
        ValueError: If version is not valid semver
    """
    try:
        ver = semver.Version.parse(version)
    except ValueError as e:
        raise ValueError(f"Invalid semantic version '{version}': {e}")

    return (ver.major, ver.minor, ver.patch)


def format_version_diff(old_version: str, new_version: str) -> str:
    """
    Generate a human-readable description of version change.

    Args:
        old_version: Previous version
        new_version: New version

    Returns:
        Description like "1.0.0 -> 1.1.0 (minor bump)"

    Raises:
        ValueError: If either version is invalid
    """
    old_parts = get_version_parts(old_version)
    new_parts = get_version_parts(new_version)

    # Determine bump type
    if new_parts[0] > old_parts[0]:
        bump_type = "major bump"
    elif new_parts[1] > old_parts[1]:
        bump_type = "minor bump"
    elif new_parts[2] > old_parts[2]:
        bump_type = "patch bump"
    else:
        comparison = compare_versions(new_version, old_version)
        if comparison < 0:
            bump_type = "downgrade"
        else:
            bump_type = "no change"

    return f"{old_version} -> {new_version} ({bump_type})"
