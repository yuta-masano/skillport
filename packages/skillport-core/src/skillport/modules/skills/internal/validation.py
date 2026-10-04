"""Skill validation rules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from skillport.shared.types import ValidationIssue
from skillport.shared.utils import parse_frontmatter

SKILL_LINE_THRESHOLD = 500
NAME_MAX_LENGTH = 64
DESCRIPTION_MAX_LENGTH = 1024
COMPATIBILITY_MAX_LENGTH = 500

# Reserved words that cannot appear in skill names
RESERVED_WORDS: frozenset[str] = frozenset({"anthropic", "claude"})

# Pattern to detect XML-like tags (e.g., <tag>, </tag>, <tag attr="x"/>)
# Tags must start with a letter or "/" (for closing tags)
_XML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")


def _is_valid_name_char(char: str) -> bool:
    """Check if a character is valid for skill names (lowercase letter, digit, or hyphen)."""
    if char == "-":
        return True
    category = unicodedata.category(char)
    # Ll = lowercase letter, Nd = decimal digit
    return category in ("Ll", "Nd")


def _validate_name_chars(name: str) -> bool:
    """Validate that all characters in name are lowercase letters, digits, or hyphens."""
    normalized = unicodedata.normalize("NFKC", name)
    return all(_is_valid_name_char(c) for c in normalized)


def _contains_reserved_word(name: str) -> str | None:
    """Return the first reserved word found in name, or None."""
    name_lower = name.lower()
    for word in RESERVED_WORDS:
        if word in name_lower:
            return word
    return None


def _contains_xml_tags(text: str) -> bool:
    """Check if text contains XML-like tags."""
    return bool(_XML_TAG_PATTERN.search(text))


def _strip_xml_tag_delimiters(text: str) -> str:
    """Remove only the delimiters of XML-like tags before the name charset rule.

    The tag name and attributes stay in the text so that characters the XML tag
    pattern would otherwise hide (spaces, quotes, control codes) are still
    rejected by the name charset rule.
    """
    return _XML_TAG_PATTERN.sub(lambda match: match.group(0)[1:-1].lstrip("/"), text)


# Allowed top-level frontmatter properties
# Standard keys are validated by value; vendor keys are accepted as known
# product extensions and only reported as a warning.
STANDARD_FRONTMATTER_KEYS: frozenset[str] = frozenset(
    {
        # agentskills.io specification
        "name",
        "description",
        "license",
        "allowed-tools",
        "metadata",
        "compatibility",
    }
)

CLAUDE_CODE_PRODUCT = "Claude Code"
CURSOR_PRODUCT = "Cursor"

_CLAUDE_CODE_SKILLS_DOC = "https://code.claude.com/docs/en/skills"
_CURSOR_SKILLS_DOC = "https://cursor.com/docs/skills"


@dataclass(frozen=True)
class VendorKeySource:
    """Primary source and confirmed version for one product's frontmatter field."""

    product: str
    url: str
    version: str


# Product-specific top-level keys, each mapped to the products that define it.
# The values are owned by each product and are deliberately not validated here.
# Keys with no confirmed product source (Codex, GitHub Copilot, Gemini CLI,
# Google Antigravity) are not registered.
VENDOR_FRONTMATTER_KEYS: dict[str, tuple[VendorKeySource, ...]] = {
    "when_to_use": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "argument-hint": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "arguments": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "disable-model-invocation": (
        VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),
        VendorKeySource(CURSOR_PRODUCT, _CURSOR_SKILLS_DOC, "未確認"),
    ),
    "user-invocable": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "disallowed-tools": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "model": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "effort": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "context": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "agent": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "background": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "2.1.218+"),),
    "hooks": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "paths": (
        VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),
        VendorKeySource(CURSOR_PRODUCT, _CURSOR_SKILLS_DOC, "未確認"),
    ),
    "shell": (VendorKeySource(CLAUDE_CODE_PRODUCT, _CLAUDE_CODE_SKILLS_DOC, "未確認"),),
    "icon": (VendorKeySource(CURSOR_PRODUCT, _CURSOR_SKILLS_DOC, "未確認"),),
    "color": (VendorKeySource(CURSOR_PRODUCT, _CURSOR_SKILLS_DOC, "未確認"),),
    "globs": (VendorKeySource(CURSOR_PRODUCT, _CURSOR_SKILLS_DOC, "未確認"),),
}

ALLOWED_FRONTMATTER_KEYS: set[str] = set(STANDARD_FRONTMATTER_KEYS) | set(VENDOR_FRONTMATTER_KEYS)


def _build_vendor_warning(vendor_keys: list[str]) -> ValidationIssue:
    """Build one aggregated warning for the vendor keys used by a skill."""
    used = set(vendor_keys)
    products: list[str] = []
    for key, sources in VENDOR_FRONTMATTER_KEYS.items():
        if key not in used:
            continue
        for source in sources:
            if source.product not in products:
                products.append(source.product)
    return ValidationIssue(
        severity="warning",
        message=(
            f"frontmatter: vendor-specific field(s) for {', '.join(products)}: "
            f"{', '.join(vendor_keys)}"
        ),
        field="frontmatter",
    )


def validate_skill_record(
    skill: dict,
    *,
    strict: bool = False,
    meta: dict | None = None,
    allow_xml_tags: bool = False,
) -> list[ValidationIssue]:
    """Validate a skill dict; returns issue list.

    Args:
        skill: Skill data dict (name, description, lines, path).
        strict: If True, informational issues are omitted. Fatal and warning
                issues are returned either way, so callers can report
                non-fatal warnings along with the validation result.
        meta: Raw frontmatter dict from parse_frontmatter(). If provided,
              enables key existence checks (A1/A2). Used by add command.
        allow_xml_tags: If True, XML tag violations in name/description are
                demoted to warnings. All other fatal rules stay fatal.

    Returns:
        List of validation issues.
    """
    issues: list[ValidationIssue] = []
    name = skill.get("name", "")
    description = skill.get("description", "")
    lines = skill.get("lines", 0)
    path = skill.get("path", "")
    dir_name = Path(path).name if path else ""

    # A1/A2: Key existence checks (only when meta is provided)
    if meta is not None:
        if "name" not in meta:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message="frontmatter: 'name' key is missing",
                    field="name",
                )
            )
        if "description" not in meta:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message="frontmatter: 'description' key is missing",
                    field="description",
                )
            )

    # Type and required field checks
    name_is_str = isinstance(name, str)
    desc_is_str = isinstance(description, str)

    # name: must be non-empty string
    if not name_is_str:
        issues.append(
            ValidationIssue(
                severity="fatal",
                message=f"frontmatter.name: must be a string (got {type(name).__name__})",
                field="name",
            )
        )
    elif not name:
        issues.append(
            ValidationIssue(severity="fatal", message="frontmatter.name: missing", field="name")
        )

    # description: must be non-empty string
    if not desc_is_str:
        issues.append(
            ValidationIssue(
                severity="fatal",
                message=f"frontmatter.description: must be a string (got {type(description).__name__})",
                field="description",
            )
        )
    elif not description:
        issues.append(
            ValidationIssue(
                severity="fatal",
                message="frontmatter.description: missing",
                field="description",
            )
        )

    # Name vs directory
    if name and dir_name and name != dir_name:
        issues.append(
            ValidationIssue(
                severity="fatal",
                message=f"frontmatter.name '{name}' doesn't match directory '{dir_name}'",
                field="name",
            )
        )

    if lines and lines > SKILL_LINE_THRESHOLD:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=f"SKILL.md: {lines} lines (recommended ≤{SKILL_LINE_THRESHOLD})",
                field="lines",
            )
        )

    if name and name_is_str:
        if len(name) > NAME_MAX_LENGTH:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message=f"frontmatter.name: {len(name)} chars (max {NAME_MAX_LENGTH})",
                    field="name",
                )
            )
        name_for_charset = _strip_xml_tag_delimiters(name) if allow_xml_tags else name
        if not _validate_name_chars(name_for_charset):
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message="frontmatter.name: invalid chars (use lowercase letters, digits, hyphens)",
                    field="name",
                )
            )
        if name_for_charset.startswith("-") or name_for_charset.endswith("-"):
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message="frontmatter.name: cannot start or end with hyphen",
                    field="name",
                )
            )
        if "--" in name:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message="frontmatter.name: cannot contain consecutive hyphens",
                    field="name",
                )
            )
        reserved = _contains_reserved_word(name)
        if reserved:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message=f"frontmatter.name: cannot contain reserved word '{reserved}'",
                    field="name",
                )
            )
        if _contains_xml_tags(name):
            issues.append(
                ValidationIssue(
                    severity="warning" if allow_xml_tags else "fatal",
                    message="frontmatter.name: cannot contain XML tags",
                    field="name",
                )
            )

    if description and desc_is_str:
        if len(description) > DESCRIPTION_MAX_LENGTH:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    message=f"frontmatter.description: {len(description)} chars (max {DESCRIPTION_MAX_LENGTH})",
                    field="description",
                )
            )
        if _contains_xml_tags(description):
            issues.append(
                ValidationIssue(
                    severity="warning" if allow_xml_tags else "fatal",
                    message="frontmatter.description: cannot contain XML tags",
                    field="description",
                )
            )

    # Check for unexpected frontmatter keys and compatibility (requires reading SKILL.md)
    if path:
        skill_md = Path(path) / "SKILL.md"
        if skill_md.exists():
            try:
                parsed_meta, _ = parse_frontmatter(skill_md)
                if isinstance(parsed_meta, dict):
                    # Unexpected keys → fatal (per Agent Skills spec)
                    unexpected_keys = set(parsed_meta.keys()) - ALLOWED_FRONTMATTER_KEYS
                    if unexpected_keys:
                        issues.append(
                            ValidationIssue(
                                severity="fatal",
                                message=f"frontmatter: unexpected field(s): {', '.join(sorted(unexpected_keys))}",
                                field="frontmatter",
                            )
                        )
                    # Registered vendor keys → one aggregated warning
                    vendor_keys = sorted(set(parsed_meta.keys()) & set(VENDOR_FRONTMATTER_KEYS))
                    if vendor_keys:
                        issues.append(_build_vendor_warning(vendor_keys))
                    # Compatibility validation (optional, max 500 chars, string type)
                    compatibility = parsed_meta.get("compatibility")
                    if compatibility is not None:
                        if not isinstance(compatibility, str):
                            issues.append(
                                ValidationIssue(
                                    severity="fatal",
                                    message="frontmatter.compatibility: must be a string",
                                    field="compatibility",
                                )
                            )
                        elif len(compatibility) > COMPATIBILITY_MAX_LENGTH:
                            issues.append(
                                ValidationIssue(
                                    severity="fatal",
                                    message=f"frontmatter.compatibility: {len(compatibility)} chars (max {COMPATIBILITY_MAX_LENGTH})",
                                    field="compatibility",
                                )
                            )
            except Exception:
                pass  # Skip if file cannot be parsed

    # strict mode: informational issues are dropped; fatal and warning issues
    # stay so callers such as add can report non-fatal warnings with the result
    if strict:
        return [i for i in issues if i.severity != "info"]
    return issues
