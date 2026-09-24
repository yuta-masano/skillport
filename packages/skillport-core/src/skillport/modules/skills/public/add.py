from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from skillport.modules.skills.internal import (
    SkillInfo,
    acquire_local_source_snapshot,
    compute_content_hash,
    detect_skills,
    extract_zip,
    fetch_github_source_with_info,
    has_hidden_or_excluded_component,
    parse_github_url,
    record_origin,
    rename_single_skill_dir,
    resolve_source,
    validate_skill_name_for_path,
)
from skillport.modules.skills.internal import (
    add_builtin as _add_builtin,
)
from skillport.modules.skills.internal import (
    add_local as _add_local,
)
from skillport.shared.config import Config
from skillport.shared.types import SourceType
from skillport.shared.utils import SymlinkPathError, find_symlink_path_component

from .types import AddResult, AddResultItem


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PrepareResult:
    """Result of preparing a source for skill addition."""

    source_path: Path
    source_label: str
    origin_payload: dict
    temp_dir: Path | None = None
    cleanup_temp_dir: bool = False
    commit_sha: str = ""


@dataclass
class AddContext:
    """Context for skill addition operation."""

    source_type: SourceType
    prepare: PrepareResult
    config: Config
    force: bool
    keep_structure: bool | None
    namespace: str | None
    name: str | None
    allow_symlinks: bool = False
    # Accumulated results
    details: list[AddResultItem] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    messages_added: list[str] = field(default_factory=list)
    messages_skipped: list[str] = field(default_factory=list)
    zip_added_ids: set[str] = field(default_factory=set)
    source_relative_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source preparation functions
# ---------------------------------------------------------------------------
def _prepare_github(
    resolved: str,
    pre_fetched_dir: Path | None,
    pre_fetched_commit_sha: str,
    *,
    allow_symlinks: bool = False,
    cleanup_pre_fetched_dir: bool = True,
) -> PrepareResult:
    """Prepare GitHub source for skill addition."""
    parsed = parse_github_url(resolved)
    if pre_fetched_dir:
        source_path = Path(pre_fetched_dir)
        temp_dir = source_path if cleanup_pre_fetched_dir else None
        commit_sha = pre_fetched_commit_sha
    else:
        fetch_result = fetch_github_source_with_info(resolved, allow_symlinks=allow_symlinks)
        source_path = fetch_result.extracted_path
        temp_dir = source_path
        commit_sha = fetch_result.commit_sha

    return PrepareResult(
        source_path=source_path,
        source_label=Path(parsed.normalized_path or parsed.repo).name,
        origin_payload={
            "source": resolved,
            "kind": "github",
            "ref": parsed.ref,
            "path": parsed.normalized_path or "",
            "commit_sha": commit_sha,
        },
        temp_dir=temp_dir,
        cleanup_temp_dir=temp_dir is not None,
        commit_sha=commit_sha,
    )


def _prepare_zip(resolved: str, *, allow_symlinks: bool = False) -> PrepareResult:
    """Prepare ZIP source for skill addition."""
    zip_path = Path(resolved)
    extract_result = extract_zip(zip_path, allow_symlinks=allow_symlinks)
    return PrepareResult(
        source_path=extract_result.extracted_path,
        source_label=zip_path.stem,
        origin_payload={
            "source": resolved,
            "kind": "zip",
            "path": "",
            "source_mtime": extract_result.source_mtime_ns,
        },
        temp_dir=extract_result.extracted_path,
        cleanup_temp_dir=True,
    )


def _prepare_local(resolved: str) -> PrepareResult:
    """Prepare local directory source for skill addition.

    The source tree is materialized through a no-follow descriptor chain and
    every later step (detection, validation, snapshot, copy) reads that
    snapshot. The origin keeps the original local path, never the snapshot.
    """
    source = acquire_local_source_snapshot(Path(resolved))
    return PrepareResult(
        source_path=source.snapshot,
        source_label=source.snapshot.name,
        origin_payload={"source": resolved, "kind": "local", "path": ""},
        temp_dir=source.temp_root,
        cleanup_temp_dir=True,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_zip_skills(skills: list[SkillInfo], source_path: Path) -> AddResult | None:
    """Validate ZIP contains exactly one skill. Returns error result or None."""
    if not skills:
        return AddResult(success=False, skill_id="", message=f"No skills found in {source_path}")
    if len(skills) != 1:
        return AddResult(
            success=False,
            skill_id="",
            message=(
                f"Zip must contain exactly one skill (found {len(skills)}). "
                "Split the archive into separate zip files."
            ),
        )
    return None


def _drop_hidden_or_excluded_symlink_candidates(
    temp_dir: Path, skills: list[SkillInfo]
) -> list[SkillInfo]:
    """Drop root candidates whose own path is a hidden or excluded symlink.

    Such a link is not an archive entry that should participate in selecting the
    single skill. Symlinks inside the selected skill remain in the list and are
    validated by the normal per-skill copy checks.
    """
    kept: list[SkillInfo] = []
    for skill in skills:
        if skill.source_path.is_symlink() and has_hidden_or_excluded_component(
            skill.source_path.relative_to(temp_dir).parts
        ):
            continue
        kept.append(skill)
    return kept


def _handle_single_skill_rename(
    prepare: PrepareResult,
    skills: list[SkillInfo],
    source_type: SourceType,
    *,
    allow_symlinks: bool = False,
) -> tuple[PrepareResult, list[SkillInfo]]:
    """Prepare a single-skill source for addition."""
    if source_type not in (SourceType.GITHUB, SourceType.ZIP) or len(skills) != 1:
        return prepare, skills

    single = skills[0]
    if single.error is not None:
        return prepare, skills
    if not prepare.cleanup_temp_dir and single.source_path == prepare.source_path:
        try:
            validate_skill_name_for_path(single.name)
        except ValueError:
            return prepare, skills
        staging_root: Path | None = None
        try:
            staging_root = Path(
                tempfile.mkdtemp(prefix="skillport-single-", dir=prepare.source_path.parent)
            )
            nested_source = staging_root / single.name
            shutil.copytree(prepare.source_path, nested_source, symlinks=True)
        except Exception:
            if staging_root is not None:
                shutil.rmtree(staging_root, ignore_errors=True)
            return prepare, skills
        try:
            new_skills = detect_skills(nested_source, allow_symlinks=allow_symlinks)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        return PrepareResult(
            source_path=nested_source,
            source_label=prepare.source_label,
            origin_payload=dict(prepare.origin_payload),
            temp_dir=staging_root,
            cleanup_temp_dir=True,
            commit_sha=prepare.commit_sha,
        ), new_skills
    try:
        new_source_path = rename_single_skill_dir(prepare.source_path, single.name)
    except ValueError:
        return prepare, skills
    new_skills = detect_skills(new_source_path, allow_symlinks=allow_symlinks)

    return PrepareResult(
        source_path=new_source_path,
        source_label=prepare.source_label,
        origin_payload=prepare.origin_payload,
        temp_dir=new_source_path,
        cleanup_temp_dir=prepare.cleanup_temp_dir,
        commit_sha=prepare.commit_sha,
    ), new_skills


# ---------------------------------------------------------------------------
# Structure options
# ---------------------------------------------------------------------------
def _determine_structure_options(
    skills: list[SkillInfo],
    source_label: str,
    keep_structure: bool | None,
    namespace: str | None,
) -> tuple[bool, str | None]:
    """Determine effective keep_structure and namespace_override."""
    effective_keep_structure = keep_structure
    namespace_override = namespace

    if len(skills) == 1:
        effective_keep_structure = (
            False if effective_keep_structure is None else effective_keep_structure
        )
    else:
        if effective_keep_structure is None:
            effective_keep_structure = True
        if effective_keep_structure and namespace_override is None:
            namespace_override = source_label

    return bool(effective_keep_structure), namespace_override


# ---------------------------------------------------------------------------
# Skill processing
# ---------------------------------------------------------------------------
def _is_zip_candidate(path: Path) -> bool:
    if path.suffix.lower() != ".zip":
        return False
    return find_symlink_path_component(path) is not None or path.is_file()


def _process_directory_skills(
    ctx: AddContext,
    skills: list[SkillInfo],
    effective_keep_structure: bool,
    namespace_override: str | None,
) -> None:
    """Add directory skills and update context with results."""
    if not skills:
        return

    results = _add_local(
        source_path=ctx.prepare.source_path,
        skills=skills,
        config=ctx.config,
        keep_structure=effective_keep_structure,
        force=ctx.force,
        namespace_override=namespace_override,
        rename_single_to=ctx.name,
        allow_symlinks=ctx.allow_symlinks,
    )

    ctx.details = [
        AddResultItem(
            skill_id=r.skill_id,
            success=r.success,
            message=r.message,
            warnings=r.warnings,
        )
        for r in results
    ]
    ctx.added_ids = [r.skill_id for r in results if r.success]
    ctx.skipped_ids = [r.skill_id for r in results if not r.success]
    ctx.messages_added = [r.message for r in results if r.success and r.message]
    ctx.messages_skipped = [r.message for r in results if not r.success and r.message]

    for skill, result in zip(skills, results, strict=True):
        if not result.success:
            continue
        relative = skill.source_path.relative_to(ctx.prepare.source_path).as_posix()
        ctx.source_relative_paths[result.skill_id] = "" if relative == "." else relative


def _process_nested_zips(ctx: AddContext) -> None:
    """Process ZIP files in LOCAL directory (recursive)."""
    source_path = ctx.prepare.source_path
    if (
        ctx.source_type != SourceType.LOCAL
        or source_path.is_symlink()
        or (source_path / "SKILL.md").exists()
    ):
        return

    # The recursive add re-opens the original path (never the snapshot) so the
    # recorded origin points at the caller's source; the boundary is acquired
    # again for that path.
    origin_root = Path(str(ctx.prepare.origin_payload.get("source", "")))
    zip_files = sorted(f for f in source_path.iterdir() if _is_zip_candidate(f))

    for zip_file in zip_files:
        try:
            add_path = origin_root / zip_file.relative_to(source_path)
        except ValueError:
            add_path = zip_file
        # Use user-specified namespace only (not directory-derived namespace_override)
        try:
            zip_result = add_skill(
                str(add_path),
                config=ctx.config,
                force=ctx.force,
                namespace=ctx.namespace,
                keep_structure=ctx.namespace is not None,
                allow_symlinks=ctx.allow_symlinks,
            )
        except SymlinkPathError as exc:
            candidate_id = zip_file.name
            zip_result = AddResult(
                success=False,
                skill_id=candidate_id,
                message=str(exc),
                skipped=[candidate_id],
                details=[
                    AddResultItem(
                        skill_id=candidate_id,
                        success=False,
                        message=str(exc),
                    )
                ],
            )
        # Merge results
        if zip_result.details:
            ctx.details.extend(zip_result.details)
        ctx.added_ids.extend(zip_result.added)
        ctx.zip_added_ids.update(zip_result.added)
        ctx.skipped_ids.extend(zip_result.skipped)
        if zip_result.added:
            ctx.messages_added.extend(
                d.message for d in (zip_result.details or []) if d.success and d.message
            )
        if zip_result.skipped or (not zip_result.success and not zip_result.added):
            if zip_result.message:
                ctx.messages_skipped.append(zip_result.message)


# ---------------------------------------------------------------------------
# Origin recording
# ---------------------------------------------------------------------------
def _record_skill_origins(ctx: AddContext) -> None:
    """Record origin for all added skills."""
    if not ctx.added_ids:
        return

    origin_payload = ctx.prepare.origin_payload

    for sid in ctx.added_ids:
        # Skip skills added via nested ZIP (already recorded in recursive call)
        if sid in ctx.zip_added_ids:
            continue
        try:
            skill_path = ctx.config.skills_dir / sid
            content_hash = compute_content_hash(skill_path)

            rel_path = ctx.source_relative_paths[sid]

            # Build enriched payload
            enriched_payload = dict(origin_payload)
            if origin_payload.get("kind") == "github":
                prefix = origin_payload.get("path", "").rstrip("/")
                if (
                    prefix
                    and rel_path
                    and rel_path != prefix
                    and not prefix.endswith(f"/{rel_path}")
                ):
                    enriched_payload["path"] = f"{prefix}/{rel_path}"
                elif prefix:
                    enriched_payload["path"] = prefix
                else:
                    enriched_payload["path"] = rel_path
            else:
                enriched_payload["path"] = rel_path
            enriched_payload["content_hash"] = content_hash

            record_origin(sid, enriched_payload, config=ctx.config)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------
def _summarize_skipped(reasons: list[str]) -> str:
    """Return a concise summary for skipped skills."""
    if not reasons:
        return "No skills added"

    exists = [r for r in reasons if "exists" in r]
    invalid = [r for r in reasons if "Invalid SKILL.md" in r]
    others = [r for r in reasons if r not in exists and r not in invalid]

    parts: list[str] = []
    if exists:
        parts.append(f"{len(exists)} already exist")
    if invalid:
        parts.append(f"{len(invalid)} invalid SKILL.md")
    if others:
        first_other = others[0]
        extra = len(others) - 1
        parts.append(first_other if extra == 0 else f"{first_other} (+{extra} more)")

    return "; ".join(parts) if parts else "No skills added"


def _build_add_result(ctx: AddContext) -> AddResult:
    """Build final AddResult from context."""
    success_all = len(ctx.skipped_ids) == 0

    if ctx.messages_skipped:
        message = _summarize_skipped(ctx.messages_skipped)
    elif ctx.messages_added:
        # Deduplicate added messages but keep order
        seen: set[str] = set()
        uniq_added = []
        for msg in ctx.messages_added:
            if msg not in seen:
                uniq_added.append(msg)
                seen.add(msg)
        message = "; ".join(uniq_added)
    else:
        message = "No skills added"

    overall_id = ctx.added_ids[0] if len(ctx.added_ids) == 1 else ",".join(ctx.added_ids)

    return AddResult(
        success=success_all,
        skill_id=overall_id,
        message=message,
        added=ctx.added_ids,
        skipped=ctx.skipped_ids,
        details=ctx.details,
    )


# ---------------------------------------------------------------------------
# Builtin handling
# ---------------------------------------------------------------------------
def _add_builtin_with_origin(resolved: str, config: Config, force: bool) -> AddResult:
    """Add builtin skill and record origin."""
    result = _add_builtin(resolved, config=config, force=force)
    if result.success:
        try:
            record_origin(
                resolved,
                {"source": resolved, "kind": "builtin"},
                config=config,
            )
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_skill(
    source: str,
    *,
    config: Config,
    force: bool = False,
    keep_structure: bool | None = None,
    namespace: str | None = None,
    name: str | None = None,
    pre_fetched_dir: Path | None = None,
    pre_fetched_commit_sha: str = "",
    allow_symlinks: bool = False,
    cleanup_pre_fetched_dir: bool = True,
) -> AddResult:
    """Add a skill from builtin/local/github source."""
    # 1. Resolve source type
    try:
        source_type, resolved = resolve_source(source)
    except SymlinkPathError:
        raise
    except Exception as exc:
        return AddResult(success=False, skill_id="", message=str(exc))

    # 2. Handle BUILTIN early return
    if source_type == SourceType.BUILTIN:
        return _add_builtin_with_origin(resolved, config, force)

    # 3. Prepare source
    if source_type == SourceType.GITHUB:
        prepare = _prepare_github(
            resolved,
            pre_fetched_dir,
            pre_fetched_commit_sha,
            allow_symlinks=allow_symlinks,
            cleanup_pre_fetched_dir=cleanup_pre_fetched_dir,
        )
    elif source_type == SourceType.ZIP:
        prepare = _prepare_zip(resolved, allow_symlinks=allow_symlinks)
    else:
        prepare = _prepare_local(resolved)

    try:
        # 4. Detect skills
        skills = detect_skills(prepare.source_path, allow_symlinks=allow_symlinks)
        if source_type == SourceType.ZIP and allow_symlinks:
            # Ignore archive-level hidden or excluded links before selecting its skill.
            skills = _drop_hidden_or_excluded_symlink_candidates(prepare.source_path, skills)

        # 5. Validate ZIP single-skill constraint
        if source_type == SourceType.ZIP:
            error = _validate_zip_skills(skills, prepare.source_path)
            if error:
                return error

        # 6. Handle single skill directory rename
        prepare, skills = _handle_single_skill_rename(
            prepare,
            skills,
            source_type,
            allow_symlinks=allow_symlinks,
        )
        if source_type == SourceType.ZIP and allow_symlinks:
            # Renaming the temporary ZIP root runs detection again, so apply the
            # archive-level filter to that second candidate list as well.
            skills = _drop_hidden_or_excluded_symlink_candidates(prepare.source_path, skills)

        # 7. Check for nested ZIP files in LOCAL directory
        root_detection_error = any(
            skill.source_path == prepare.source_path and skill.error is not None for skill in skills
        )
        can_process_nested_zips = (
            source_type == SourceType.LOCAL
            and not root_detection_error
            and not prepare.source_path.is_symlink()
        )
        has_zip_files = (
            can_process_nested_zips
            and not (prepare.source_path / "SKILL.md").exists()
            and any(_is_zip_candidate(f) for f in prepare.source_path.iterdir())
        )

        if not skills and not has_zip_files:
            return AddResult(
                success=False,
                skill_id="",
                message=f"No skills found in {prepare.source_path}",
            )

        # 8. Determine structure options
        effective_keep_structure, namespace_override = _determine_structure_options(
            skills,
            prepare.source_label,
            keep_structure,
            namespace,
        )

        # 9. Create context
        ctx = AddContext(
            source_type=source_type,
            prepare=prepare,
            config=config,
            force=force,
            keep_structure=keep_structure,
            namespace=namespace,
            name=name,
            allow_symlinks=allow_symlinks,
        )

        # 10. Process directory skills
        _process_directory_skills(ctx, skills, effective_keep_structure, namespace_override)

        # 11. Process nested ZIPs (LOCAL only)
        if can_process_nested_zips:
            _process_nested_zips(ctx)

        # 12. Record origins
        _record_skill_origins(ctx)

        # 13. Build and return result
        return _build_add_result(ctx)

    finally:
        if prepare.cleanup_temp_dir and prepare.temp_dir and prepare.temp_dir.exists():
            shutil.rmtree(prepare.temp_dir, ignore_errors=True)
