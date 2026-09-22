"""Update skills from their original sources.

This module uses function-based dispatch to handle different update sources
(local, GitHub, zip) with shared helper functions for common operations.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from skillport.modules.skills.internal import (
    ParsedGitHubURL,
    SkillInfo,
    StableLocalSource,
    acquire_local_source_snapshot,
    capture_origin_state,
    compute_content_hash,
    compute_content_hash_with_reason,
    copy_skill_dir,
    copy_skill_snapshot,
    detect_skills,
    extract_opened_zip,
    fail_on_symlinks,
    fetch_github_source_with_info,
    get_all_origins,
    get_origin,
    get_remote_tree_hash,
    get_remote_tree_symlinks,
    has_hidden_or_excluded_component,
    open_zip_source,
    parse_github_url,
    rename_single_skill_dir,
    restore_origin_state,
    snapshot_skill_dir,
    update_origin,
    zip_has_symlink_entries,
    zip_hidden_or_excluded_symlink_entries,
)
from skillport.shared.auth import resolve_github_token
from skillport.shared.config import Config

from .types import Origin, UpdateResult, UpdateResultItem

# =============================================================================
# Public API
# =============================================================================


def detect_local_modification(skill_id: str, *, config: Config) -> bool:
    """Check if a skill has local modifications.

    Compares the current SKILL.md content hash against the stored hash.
    Returns False if origin info is missing or doesn't have content_hash.
    """
    origin = get_origin(skill_id, config=config)
    if not origin:
        return False

    stored_hash = origin.get("content_hash")
    if not stored_hash:
        return False

    skill_path = config.skills_dir / skill_id
    if _installed_path_symlink_reason(config.skills_dir, skill_path):
        return False
    current_hash = compute_content_hash(skill_path)

    return stored_hash != current_hash


def check_update_available(skill_id: str, *, config: Config) -> dict[str, Any]:
    """Check if an update is available for a skill."""
    origin = get_origin(skill_id, config=config)

    if not origin:
        return {
            "available": False,
            "reason": "No origin info (cannot update)",
            "origin": None,
            "new_commit": "",
        }

    kind = origin.get("kind", "")

    if kind == "builtin":
        return {
            "available": False,
            "reason": "Built-in skill cannot be updated",
            "origin": origin,
            "new_commit": "",
        }

    skill_path = config.skills_dir / skill_id
    if symlink_reason := _installed_path_symlink_reason(config.skills_dir, skill_path):
        return {
            "available": False,
            "reason": symlink_reason,
            "origin": origin,
            "new_commit": "",
        }

    source_hash, source_reason = _compute_source_hash(origin, skill_id, config=config)
    if source_reason:
        return {"available": False, "reason": source_reason, "origin": origin, "new_commit": ""}

    installed_hash, installed_reason = compute_content_hash_with_reason(
        config.skills_dir / skill_id
    )
    if installed_reason:
        return {
            "available": False,
            "reason": f"Installed skill unreadable: {installed_reason}",
            "origin": origin,
            "new_commit": "",
        }

    if source_hash == installed_hash:
        return {
            "available": False,
            "reason": "Already at latest content",
            "origin": origin,
            "new_commit": "",
        }

    return {
        "available": True,
        "reason": "Remote content differs" if kind == "github" else "Local source changed",
        "origin": origin,
        "new_commit": source_hash.split(":", 1)[-1][:7]
        if source_hash.startswith("sha256:")
        else source_hash[:7],
    }


def update_skill(
    skill_id: str,
    *,
    config: Config,
    force: bool = False,
    dry_run: bool = False,
    allow_symlinks: bool = False,
) -> UpdateResult:
    """Update a single skill from its original source."""
    skill_path = config.skills_dir / skill_id
    if symlink_reason := _installed_path_symlink_reason(config.skills_dir, skill_path):
        return UpdateResult(
            success=False,
            skill_id=skill_id,
            message=symlink_reason,
        )
    if not skill_path.exists():
        return UpdateResult(
            success=False, skill_id=skill_id, message=f"Skill '{skill_id}' not found"
        )

    origin = get_origin(skill_id, config=config)
    if not origin:
        return UpdateResult(
            success=False,
            skill_id=skill_id,
            message=f"Skill '{skill_id}' has no origin info (cannot update)",
        )

    kind = origin.get("kind", "")

    if kind == "builtin":
        return UpdateResult(
            success=False, skill_id=skill_id, message="Built-in skill cannot be updated"
        )

    # The flag is transient: a skill containing symlinks (only installable with
    # --allow-symlinks) is rejected again on every flagless update attempt
    if not allow_symlinks:
        try:
            fail_on_symlinks(skill_path)
        except ValueError as exc:
            return UpdateResult(
                success=False,
                skill_id=skill_id,
                message=f"{exc} (updating requires --allow-symlinks)",
            )

    ctx = UpdateContext(
        skill_id=skill_id,
        origin=origin,
        config=config,
        force=force,
        dry_run=dry_run,
        allow_symlinks=allow_symlinks,
    )

    handler = _UPDATE_HANDLERS.get(kind)
    if handler is None:
        return UpdateResult(
            success=False, skill_id=skill_id, message=f"Unknown origin kind: {kind}"
        )

    return handler(ctx)


def _installed_path_symlink_reason(skills_dir: Path, skill_path: Path) -> str | None:
    try:
        relative_parts = skill_path.relative_to(skills_dir).parts
    except ValueError:
        return f"Installed skill path is outside skills directory: {skill_path}"

    current = skills_dir
    if current.is_symlink():
        return f"Installed skills directory is a symlink: {current}"

    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            if current == skill_path:
                return f"Installed skill root is a symlink: {current}"
            return f"Installed skill path contains a symlink component: {current}"

    return None


def update_all_skills(
    *,
    config: Config,
    force: bool = False,
    dry_run: bool = False,
    skill_ids: list[str] | None = None,
    allow_symlinks: bool = False,
) -> UpdateResult:
    """Update all updatable skills (optionally limited to skill_ids)."""
    origins = get_all_origins(config=config)

    if skill_ids is not None:
        origins = {k: v for k, v in origins.items() if k in skill_ids}

    if not origins:
        return UpdateResult(success=True, skill_id="", message="No skills to update")

    updated: list[str] = []
    skipped: list[str] = []
    details: list[UpdateResultItem] = []
    errors: list[str] = []

    for skill_id, origin in origins.items():
        if origin.get("kind") == "builtin":
            continue

        result = update_skill(
            skill_id,
            config=config,
            force=force,
            dry_run=dry_run,
            allow_symlinks=allow_symlinks,
        )

        if result.updated:
            updated.extend(result.updated)
        if result.skipped:
            skipped.extend(result.skipped)
        if result.details:
            details.extend(result.details)
        if not result.success and not result.skipped:
            errors.append(f"{skill_id}: {result.message}")
            details.append(
                UpdateResultItem(skill_id=skill_id, success=False, message=result.message)
            )

    parts = []
    if updated:
        parts.append(f"Updated {len(updated)} skill(s)")
    if skipped:
        parts.append(f"Skipped {len(skipped)} (up to date)")
    if errors:
        parts.append(f"{len(errors)} error(s)")

    return UpdateResult(
        success=len(errors) == 0,
        skill_id=",".join(updated) if updated else "",
        message=", ".join(parts) if parts else "No skills to update",
        updated=updated,
        skipped=skipped,
        details=details,
        errors=errors,
    )


# =============================================================================
# Update Context
# =============================================================================


@dataclass
class UpdateContext:
    """Context for update operations."""

    skill_id: str
    origin: Origin
    config: Config
    force: bool
    dry_run: bool
    allow_symlinks: bool = False

    @property
    def stored_hash(self) -> str:
        return self.origin.get("content_hash", "")

    @property
    def dest_path(self) -> Path:
        return self.config.skills_dir / self.skill_id


# =============================================================================
# Common Helpers
# =============================================================================


def _error(ctx: UpdateContext, message: str) -> UpdateResult:
    """Create an error result."""
    return UpdateResult(success=False, skill_id=ctx.skill_id, message=message)


def _already_up_to_date(ctx: UpdateContext) -> UpdateResult:
    """Create an 'already up to date' result."""
    return UpdateResult(
        success=True, skill_id=ctx.skill_id, message="Already up to date", skipped=[ctx.skill_id]
    )


def _local_modification_error(ctx: UpdateContext) -> UpdateResult:
    """Create a local modification error result."""
    return UpdateResult(
        success=False,
        skill_id=ctx.skill_id,
        message="Local modifications detected. Use --force to overwrite",
        local_modified=True,
    )


def _sync_stored_hash_if_needed(ctx: UpdateContext, current_hash: str) -> None:
    """Sync stored hash if outdated."""
    if ctx.stored_hash != current_hash:
        update_origin(ctx.skill_id, {"content_hash": current_hash}, config=ctx.config)


def _has_local_modifications(ctx: UpdateContext, current_hash: str) -> bool:
    """Check if local modifications exist."""
    return bool(ctx.stored_hash and ctx.stored_hash != current_hash)


def _check_update_needed(
    ctx: UpdateContext, source_hash: str, current_hash: str
) -> UpdateResult | None:
    """Check if update is needed.

    Returns None if update should proceed, otherwise returns early-exit result.
    """
    if source_hash == current_hash:
        _sync_stored_hash_if_needed(ctx, current_hash)
        return _already_up_to_date(ctx)

    if _has_local_modifications(ctx, current_hash) and not ctx.force:
        return _local_modification_error(ctx)

    return None


def _reject_source_symlinks(source_path: Path, *, allow_symlinks: bool) -> str | None:
    """Source-side symlink gate, run before any hash/mtime early-exit path.

    The skill root itself must never be a symlink regardless of the flag (it
    would make hashing and copying follow to external content). Without the
    flag, any symlink in the source tree is rejected; with the flag, per-entry
    validation happens later in the staged preserve copy.
    Returns an error message, or None when the source is acceptable.
    """
    if source_path.is_symlink():
        return f"Skill source root is a symlink: {source_path}"
    if not allow_symlinks:
        try:
            fail_on_symlinks(source_path)
        except ValueError as exc:
            return f"{exc} (updating requires --allow-symlinks)"
    return None


def _github_tree_path(
    origin: Origin, parsed: ParsedGitHubURL, skill_id: str
) -> tuple[str, bool]:
    """Resolve the remote tree path for a GitHub origin.

    An explicit ``path`` (including the empty string for a skill at the
    repository root) is authoritative. Only origins without the key keep the
    historical skill-name fallback; the returned flag marks those.
    """
    if "path" in origin:
        return origin.get("path") or "", False
    return parsed.normalized_path or skill_id.split("/")[-1], True


def _hidden_or_excluded_symlink(symlink_paths: list[str]) -> str | None:
    """Return the first symlink path whose own path is hidden or excluded."""
    for path in symlink_paths:
        if has_hidden_or_excluded_component(PurePosixPath(path).parts):
            return path
    return None


def _github_source_symlinks(origin: Origin, skill_id: str) -> list[str]:
    """List visible symlink entries of the remote tree for the skill's path.

    Mirrors the path resolution of _github_source_hash so the gate scans the
    same tree the hash covers.
    """
    source_url = origin.get("source", "")
    if not source_url:
        return []

    auth = resolve_github_token()
    parsed = parse_github_url(source_url, resolve_default_branch=True, auth=auth)
    path, legacy_fallback = _github_tree_path(origin, parsed, skill_id)

    symlink_paths = get_remote_tree_symlinks(parsed, auth.token, path)
    if not symlink_paths and legacy_fallback:
        skill_tail = skill_id.split("/")[-1]
        candidate = "/".join(p for p in [parsed.normalized_path, skill_tail] if p)
        if candidate != path:
            symlink_paths = get_remote_tree_symlinks(parsed, auth.token, candidate)
    return symlink_paths


def _copy_and_update_origin(
    ctx: UpdateContext,
    source_path: Path,
    success_message: str,
    extra_fields: dict[str, Any] | None = None,
    history_entry: dict[str, Any] | None = None,
    details: list[UpdateResultItem] | None = None,
    source_is_snapshot: bool = False,
) -> UpdateResult:
    """Common update: staged validated copy + origin update, then replace destination.

    The existing skill directory is only removed after the staged copy and the
    origin update both succeeded, so a symlink/validation failure leaves the
    installed skill unchanged.
    """
    staging_root: Path | None = None
    backup_path: Path | None = None
    backup_retained = False
    old_destination_moved = False
    new_destination_installed = False
    metadata_attempted = False
    origin_state: tuple[bool, dict[str, Any]] | None = None
    try:
        origin_state = capture_origin_state(config=ctx.config)
        staging_root = Path(tempfile.mkdtemp(prefix="skillport-update-", dir=ctx.dest_path.parent))
        staged = staging_root / ctx.dest_path.name
        if source_is_snapshot:
            copy_skill_snapshot(source_path, staged, allow_symlinks=ctx.allow_symlinks)
        else:
            copy_skill_dir(source_path, staged, allow_symlinks=ctx.allow_symlinks)

        new_hash, _ = compute_content_hash_with_reason(staged)
        origin_updates: dict[str, Any] = {
            "content_hash": new_hash,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "local_modified": False,
        }
        if extra_fields:
            origin_updates.update(extra_fields)

        backup_path = Path(
            tempfile.mkdtemp(
                prefix=f".{ctx.dest_path.name}-backup-",
                dir=ctx.dest_path.parent,
            )
        )
        backup_path.rmdir()
        ctx.dest_path.rename(backup_path)
        old_destination_moved = True
        staged.rename(ctx.dest_path)
        new_destination_installed = True

        metadata_attempted = True
        update_origin(
            ctx.skill_id, origin_updates, config=ctx.config, add_history_entry=history_entry
        )
        shutil.rmtree(backup_path, ignore_errors=True)
        backup_path = None

        return UpdateResult(
            success=True,
            skill_id=ctx.skill_id,
            message=success_message,
            updated=[ctx.skill_id],
            details=details or [],
        )
    except Exception as e:
        rollback_errors: list[str] = []
        try:
            if new_destination_installed:
                if ctx.dest_path.is_symlink() or ctx.dest_path.is_file():
                    ctx.dest_path.unlink()
                elif ctx.dest_path.exists():
                    shutil.rmtree(ctx.dest_path)
            if old_destination_moved and backup_path and backup_path.exists():
                backup_path.rename(ctx.dest_path)
                backup_path = None
        except Exception as rollback_error:
            if backup_path is not None and backup_path.exists():
                # The backup now holds the only copy of the previous skill
                # content, so it is kept for recovery instead of being cleaned
                # up with the other temporary resources.
                backup_retained = True
                rollback_errors.append(
                    f"destination rollback failed: {rollback_error}; "
                    f"previous skill content is kept at {backup_path}"
                )
            else:
                rollback_errors.append(f"destination rollback failed: {rollback_error}")

        if metadata_attempted and origin_state is not None:
            try:
                restore_origin_state(origin_state, config=ctx.config)
            except Exception as rollback_error:
                rollback_errors.append(f"origin rollback failed: {rollback_error}")

        if rollback_errors:
            return _error(ctx, f"Failed to update: {e}; {'; '.join(rollback_errors)}")
        return _error(ctx, f"Failed to update: {e}")
    finally:
        if staging_root and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        if backup_path and backup_path.exists() and not backup_retained:
            shutil.rmtree(backup_path, ignore_errors=True)


# =============================================================================
# Update Handlers (one per origin kind)
# =============================================================================


def _update_local(ctx: UpdateContext) -> UpdateResult:
    """Update from local source directory."""
    source_base = Path(ctx.origin.get("source", ""))

    stable: StableLocalSource | None = None
    snapshot: Path | None = None
    try:
        # Pin the source root and read only its snapshot: replacing a source
        # ancestor after this boundary must not redirect the update.
        try:
            stable = acquire_local_source_snapshot(source_base)
        except FileNotFoundError:
            return _error(ctx, f"Source path not found: {source_base}")
        except NotADirectoryError:
            return _error(ctx, f"Source is not a directory: {source_base}")
        source_root = stable.snapshot

        # Resolve skill path within the pinned source
        try:
            source_path = _resolve_local_source_path(
                ctx.origin,
                source_root,
                ctx.skill_id,
                allow_final_skill_symlink=ctx.allow_symlinks,
            )
        except ValueError as exc:
            return _error(ctx, str(exc))

        if source_path is None:
            return _error(ctx, f"Could not find skill in source: {source_base}")

        # Source symlink gate before the hash comparison, so a symlinked source
        # cannot early-exit with "Already up to date" (installed skill stays intact)
        if reject_reason := _reject_source_symlinks(source_path, allow_symlinks=ctx.allow_symlinks):
            return _error(ctx, reject_reason)

        snapshot = snapshot_skill_dir(source_path, allow_symlinks=ctx.allow_symlinks)
        source_hash, reason = compute_content_hash_with_reason(snapshot)
        if reason:
            return _error(ctx, f"Source not readable: {reason}")

        current_hash, reason = compute_content_hash_with_reason(ctx.dest_path)
        if reason:
            return _error(ctx, f"Installed skill unreadable: {reason}")

        origin_updates: dict[str, Any] = {}
        if "path" not in ctx.origin:
            try:
                rel = source_path.relative_to(source_root).as_posix()
                if rel != ".":
                    origin_updates["path"] = rel
            except Exception:
                pass

        if result := _check_update_needed(ctx, source_hash, current_hash):
            if result.skipped and origin_updates:
                try:
                    update_origin(ctx.skill_id, origin_updates, config=ctx.config)
                except Exception:
                    pass
            return result

        if ctx.dry_run:
            return UpdateResult(
                success=True,
                skill_id=ctx.skill_id,
                message=f"Would update from {source_path}",
                updated=[ctx.skill_id],
            )

        return _copy_and_update_origin(
            ctx,
            snapshot,
            "Updated from local source",
            extra_fields=origin_updates or None,
            source_is_snapshot=True,
        )
    except Exception as e:
        return _error(ctx, f"Source not readable: {e}")
    finally:
        if snapshot and snapshot.exists():
            shutil.rmtree(snapshot.parent, ignore_errors=True)
        if stable is not None:
            stable.cleanup()


def _update_github(ctx: UpdateContext) -> UpdateResult:
    """Update from GitHub repository."""
    source_url = ctx.origin.get("source", "")
    if not source_url:
        return _error(ctx, "Missing GitHub source URL")

    old_commit = ctx.origin.get("commit_sha", "")[:7] or "unknown"

    # Compute installed hash
    current_hash, reason = compute_content_hash_with_reason(ctx.dest_path)
    if reason:
        return _error(ctx, f"Installed skill unreadable: {reason}")

    # Get remote hash via tree API (no download yet)
    remote_hash, reason = _github_source_hash(ctx.origin, ctx.skill_id, config=ctx.config)
    if reason:
        return _error(ctx, f"Cannot check remote: {reason}")

    # Source symlink gate before the hash comparison: a flagless update must
    # not early-exit with "Already up to date" for a symlinked remote source
    if not ctx.allow_symlinks:
        symlink_paths = _github_source_symlinks(ctx.origin, ctx.skill_id)
        if symlink_paths:
            return _error(
                ctx,
                f"Symlink detected in GitHub source: {symlink_paths[0]}"
                " (updating requires --allow-symlinks)",
            )

    # The remote tree hash does not cover hidden/excluded paths, so a matching
    # hash with such a link must fail before the stored hash is synced.
    if ctx.allow_symlinks and remote_hash == current_hash:
        hidden_link = _hidden_or_excluded_symlink(_github_source_symlinks(ctx.origin, ctx.skill_id))
        if hidden_link is not None:
            return _error(ctx, f"Symlink path is hidden or excluded: {hidden_link}")

    # Check if update needed
    if result := _check_update_needed(ctx, remote_hash, current_hash):
        return result

    # Dry run
    if ctx.dry_run:
        return UpdateResult(
            success=True,
            skill_id=ctx.skill_id,
            message=f"Would update ({old_commit} -> latest)",
            updated=[ctx.skill_id],
            details=[
                UpdateResultItem(
                    skill_id=ctx.skill_id,
                    success=True,
                    message="Would update",
                    from_commit=old_commit,
                    to_commit="latest",
                )
            ],
        )

    # Download and apply update
    temp_dir: Path | None = None
    try:
        fetch_result = fetch_github_source_with_info(source_url, allow_symlinks=ctx.allow_symlinks)
        temp_dir = fetch_result.extracted_path
        new_commit = fetch_result.commit_sha[:7] if fetch_result.commit_sha else ""

        source_path = _resolve_github_source_path(
            temp_dir,
            ctx.origin,
            source_url,
            allow_symlinks=ctx.allow_symlinks,
        )
        if not temp_dir.exists():
            temp_dir = source_path

        history_entry = {
            "from_commit": old_commit,
            "to_commit": new_commit or "latest",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        return _copy_and_update_origin(
            ctx,
            source_path,
            f"Updated ({old_commit} -> {new_commit or 'latest'})",
            extra_fields={"commit_sha": fetch_result.commit_sha},
            history_entry=history_entry,
            details=[
                UpdateResultItem(
                    skill_id=ctx.skill_id,
                    success=True,
                    message="Updated",
                    from_commit=old_commit,
                    to_commit=new_commit or "latest",
                )
            ],
        )

    except Exception as e:
        return _error(ctx, f"Failed to fetch from GitHub: {e}")
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _update_zip(ctx: UpdateContext) -> UpdateResult:
    """Update from zip file source."""
    source_zip = Path(ctx.origin.get("source", ""))

    # Source no-follow gate: the zip is opened through a pinned descriptor, so
    # a path replaced afterwards cannot swap in another archive. The same
    # descriptor is used for the mtime, the entry scan, and the extraction.
    try:
        source = open_zip_source(source_zip)
    except Exception as exc:
        return _error(ctx, str(exc))

    temp_dir: Path | None = None
    snapshot: Path | None = None
    try:
        current_mtime = source.mtime_ns
        symlink_entries = zip_has_symlink_entries(source)

        # Source symlink gate before the mtime fast path: a flagless update must
        # not early-exit with "Already up to date" for a symlinked zip source
        if not ctx.allow_symlinks and symlink_entries:
            return _error(
                ctx,
                f"Symlink detected in zip: {source_zip} (updating requires --allow-symlinks)",
            )

        # Compute installed hash
        current_hash, reason = compute_content_hash_with_reason(ctx.dest_path)
        if reason:
            return _error(ctx, f"Installed skill unreadable: {reason}")

        # A zip carrying symlink entries that the flag allows must have every
        # link validated before the mtime fast path can report "Already up to
        # date": the per-skill policy allows only InsideSkill targets, so a
        # missing, absolute, outside-skill or cyclic target must fail the skill
        # even when the archive is unchanged.
        validate_links = ctx.allow_symlinks and symlink_entries

        # Fast path: mtime unchanged. Zips that still need link validation make
        # this decision after the validated snapshot is built.
        stored_mtime = ctx.origin.get("source_mtime")
        if not validate_links and stored_mtime == current_mtime and ctx.stored_hash:
            if ctx.stored_hash == current_hash:
                return _already_up_to_date(ctx)
            if not ctx.force:
                return _local_modification_error(ctx)

        # Extract and compute source hash
        try:
            extract_result = extract_opened_zip(source, allow_symlinks=ctx.allow_symlinks)
            temp_dir = extract_result.extracted_path

            skills = detect_skills(temp_dir, allow_symlinks=ctx.allow_symlinks)
            if ctx.allow_symlinks:
                # A root symlink directory whose own path is hidden or excluded
                # is not an entry of this archive, so it must not take part in
                # the single-skill selection.
                skills = _drop_hidden_or_excluded_symlink_candidates(temp_dir, skills)
            if not skills:
                return _error(ctx, "No skills found in zip source")
            if len(skills) != 1:
                return _error(ctx, f"Zip must contain exactly one skill (found {len(skills)})")

            skill_source_path = _resolve_zip_skill_path(
                temp_dir,
                ctx.origin,
                skills,
                allow_final_skill_symlink=ctx.allow_symlinks,
            )

            if validate_links:
                # The hidden/excluded scan is scoped to the resolved skill: a
                # link outside it follows the normal exclusion rules and must
                # not fail this skill. An in-skill link must not reach the
                # mtime fast path, so the scan runs before the validated
                # snapshot builds.
                hidden_links = zip_hidden_or_excluded_symlink_entries(
                    source, prefix=_zip_skill_prefix(temp_dir, skill_source_path)
                )
                if hidden_links:
                    return _error(ctx, f"Symlink path is hidden or excluded: {hidden_links[0]}")

                # Full validation of every link (relative target, resolvable
                # chain, existing target inside the skill directory), then reuse
                # the validated snapshot for the hash and the update copy.
                snapshot = snapshot_skill_dir(skill_source_path, allow_symlinks=True)
                skill_source_path = snapshot

                if stored_mtime == current_mtime and ctx.stored_hash:
                    if ctx.stored_hash == current_hash:
                        return _already_up_to_date(ctx)
                    if not ctx.force:
                        return _local_modification_error(ctx)

            source_hash, reason = compute_content_hash_with_reason(skill_source_path)
            if reason:
                return _error(ctx, f"Source not readable: {reason}")

            # Check if update needed
            if result := _check_update_needed(ctx, source_hash, current_hash):
                # Update mtime for fast path next time
                if result.skipped:
                    update_origin(
                        ctx.skill_id,
                        {"source_mtime": current_mtime, "content_hash": current_hash},
                        config=ctx.config,
                    )
                return result

            # Dry run
            if ctx.dry_run:
                return UpdateResult(
                    success=True,
                    skill_id=ctx.skill_id,
                    message=f"Would update from {source_zip.name}",
                    updated=[ctx.skill_id],
                )

            # Apply update
            return _copy_and_update_origin(
                ctx,
                skill_source_path,
                "Updated from zip source",
                extra_fields={"source_mtime": current_mtime},
                source_is_snapshot=snapshot is not None,
            )

        except Exception as e:
            return _error(ctx, f"Failed to extract zip: {e}")
    finally:
        source.close()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if snapshot and snapshot.exists():
            shutil.rmtree(snapshot.parent, ignore_errors=True)


# Handler dispatch table
_UPDATE_HANDLERS: dict[str, Callable[[UpdateContext], UpdateResult]] = {
    "local": _update_local,
    "github": _update_github,
    "zip": _update_zip,
}


# =============================================================================
# Path Resolution Helpers
# =============================================================================


def _source_path_symlink_reason(source_root: Path, candidate: Path) -> str | None:
    root = Path(os.path.abspath(source_root))
    target = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(target.anchor)
    target_parts = target.parts[1:]

    for index, part in enumerate(target_parts):
        if part in ("", "."):
            continue
        if part == "..":
            if not current.exists() or not current.is_dir():
                return None
            current = current.parent
            continue

        next_path = current / part
        if next_path.is_symlink():
            if next_path == root:
                return f"Skill source root is a symlink: {next_path}"
            return f"Skill source path contains a symlink component: {next_path}"
        if not next_path.exists():
            return None
        if index < len(target_parts) - 1 and not next_path.is_dir():
            return None
        current = next_path

    if not current.is_relative_to(root):
        return f"Skill source path is outside source root: {candidate}"
    return None


def _validate_source_path(source_root: Path, candidate: Path) -> None:
    if reason := _source_path_symlink_reason(source_root, candidate):
        raise ValueError(reason)


def _candidate_contains_skill(
    source_root: Path,
    candidate: Path,
    *,
    allow_final_skill_symlink: bool = False,
) -> bool:
    _validate_source_path(source_root, candidate)
    skill_file = candidate / "SKILL.md"
    if allow_final_skill_symlink and skill_file.is_symlink():
        return skill_file.exists()
    _validate_source_path(source_root, skill_file)
    return skill_file.exists()


def _resolve_local_source_path(
    origin: Origin,
    source_root: Path,
    skill_id: str,
    *,
    allow_final_skill_symlink: bool,
) -> Path | None:
    """Resolve the skill directory of a local origin within the pinned source.

    An explicit ``path`` (including the empty string for a skill at the source
    root) is authoritative. Only origins without the key keep the historical
    skill-name fallback used by pre-path records.
    """
    if "path" in origin:
        candidate = source_root / (origin.get("path") or "")
        return (
            candidate
            if _candidate_contains_skill(
                source_root,
                candidate,
                allow_final_skill_symlink=allow_final_skill_symlink,
            )
            else None
        )

    return _resolve_local_skill_path(
        source_root,
        skill_id,
        allow_final_skill_symlink=allow_final_skill_symlink,
    )


def _resolve_local_skill_path(
    source: Path,
    skill_id: str,
    *,
    allow_final_skill_symlink: bool = False,
) -> Path | None:
    """Resolve skill directory within a local source."""
    skill_name = skill_id.split("/")[-1]

    for candidate in [source / skill_id, source / skill_name, source]:
        if _candidate_contains_skill(
            source,
            candidate,
            allow_final_skill_symlink=allow_final_skill_symlink,
        ):
            return candidate

    return None


def _resolve_github_source_path(
    temp_dir: Path,
    origin: Origin,
    source_url: str,
    *,
    allow_symlinks: bool = False,
) -> Path:
    """Resolve skill directory within extracted GitHub tarball."""
    _validate_source_path(temp_dir, temp_dir)
    parsed = parse_github_url(source_url, resolve_default_branch=True)
    url_prefix = parsed.normalized_path
    origin_path = origin.get("path") or ""

    # Strip URL prefix from origin.path if present
    if url_prefix and origin_path.startswith(url_prefix + "/"):
        relative_path = origin_path[len(url_prefix) + 1 :]
    elif url_prefix and origin_path == url_prefix:
        relative_path = ""
    else:
        relative_path = origin_path

    if relative_path:
        candidate = temp_dir / relative_path
        _validate_source_path(temp_dir, candidate)
        if candidate.exists():
            return candidate
        return temp_dir

    skills = detect_skills(temp_dir, allow_symlinks=allow_symlinks)
    if skills and len(skills) == 1:
        if skills[0].error is not None:
            raise ValueError(skills[0].error)
        renamed = rename_single_skill_dir(temp_dir, skills[0].name)
        _validate_source_path(renamed.parent, renamed)
        return renamed

    return temp_dir


def _resolve_zip_skill_path(
    temp_dir: Path,
    origin: Origin,
    skills: list,
    *,
    allow_final_skill_symlink: bool = False,
) -> Path:
    """Resolve skill directory within extracted zip."""
    _validate_source_path(temp_dir, temp_dir)
    origin_path = origin.get("path", "")
    if origin_path:
        candidate = temp_dir / origin_path
        if _candidate_contains_skill(
            temp_dir,
            candidate,
            allow_final_skill_symlink=allow_final_skill_symlink,
        ):
            return candidate
    skill = skills[0]
    if skill.error is not None:
        raise ValueError(skill.error)
    _validate_source_path(temp_dir, skill.source_path)
    return skill.source_path


def _zip_skill_prefix(temp_dir: Path, skill_source_path: Path) -> str:
    """Posix prefix of the resolved skill directory inside the extraction root."""
    relative = skill_source_path.relative_to(temp_dir).as_posix()
    return "" if relative == "." else relative


def _drop_hidden_or_excluded_symlink_candidates(
    temp_dir: Path, skills: list[SkillInfo]
) -> list[SkillInfo]:
    """Drop skill candidates whose own path is a hidden or excluded symlink.

    detect_skills() reports a symlinked skill root as a per-skill error
    candidate, because copying through it would read content from outside the
    collection. A link whose own path is hidden or in EXCLUDE_NAMES is not an
    entry of the archive at all: the normal exclusion rules apply, so it must
    not take part in the single-skill selection. A visible symlink root keeps
    its existing rejection.
    """
    kept: list[SkillInfo] = []
    for skill in skills:
        if skill.source_path.is_symlink() and has_hidden_or_excluded_component(
            skill.source_path.relative_to(temp_dir).parts
        ):
            continue
        kept.append(skill)
    return kept


# =============================================================================
# Source Hash Computation (for check_update_available)
# =============================================================================


def _compute_source_hash(
    origin: Origin, skill_id: str, *, config: Config
) -> tuple[str, str | None]:
    """Compute source-side hash. Returns (hash, error_reason)."""
    kind = origin.get("kind", "")

    if kind == "local":
        return _local_source_hash(origin, skill_id, config=config)
    if kind == "github":
        return _github_source_hash(origin, skill_id, config=config)
    if kind == "zip":
        return _zip_source_hash(origin, skill_id, config=config)

    return "", f"Unknown origin kind: {kind}"


def _local_source_hash(origin: Origin, skill_id: str, *, config: Config) -> tuple[str, str | None]:
    """Compute source hash for local origin."""
    source_base = Path(origin.get("source", ""))
    stable: StableLocalSource | None = None
    try:
        # Pin the source root and hash only its snapshot, so a source ancestor
        # replaced after this boundary is not hashed as the installed content.
        try:
            stable = acquire_local_source_snapshot(source_base)
        except FileNotFoundError:
            return "", f"Source path not found: {source_base}"
        except NotADirectoryError:
            return "", f"Source is not a directory: {source_base}"
        except ValueError as exc:
            return "", str(exc)

        source_root = stable.snapshot
        try:
            source_path = _resolve_local_source_path(
                origin,
                source_root,
                skill_id,
                allow_final_skill_symlink=True,
            )
        except ValueError as exc:
            return "", str(exc)

        if source_path is None:
            return "", f"Could not find skill in source: {source_base}"

        return compute_content_hash_with_reason(source_path)
    finally:
        if stable is not None:
            stable.cleanup()


def _github_source_hash(origin: Origin, skill_id: str, *, config: Config) -> tuple[str, str | None]:
    """Compute source hash for GitHub origin via tree API."""
    source_url = origin.get("source", "")
    if not source_url:
        return "", "Missing source URL"

    auth = resolve_github_token()
    parsed = parse_github_url(source_url, resolve_default_branch=True, auth=auth)
    path, legacy_fallback = _github_tree_path(origin, parsed, skill_id)

    remote_hash = get_remote_tree_hash(parsed, auth.token, path)

    # Try narrowing path if initial attempt failed (legacy origins only)
    if legacy_fallback and (not remote_hash or path == parsed.normalized_path):
        skill_tail = skill_id.split("/")[-1]
        candidate = "/".join(p for p in [parsed.normalized_path, skill_tail] if p)
        if candidate != path:
            alt_hash = get_remote_tree_hash(parsed, auth.token, candidate)
            if alt_hash:
                remote_hash = alt_hash
                try:
                    update_origin(skill_id, {"path": candidate}, config=config)
                except Exception:
                    pass

    if not remote_hash:
        return "", "Could not fetch remote tree (treated as unknown)"
    return remote_hash, None


def _zip_source_hash(origin: Origin, skill_id: str, *, config: Config) -> tuple[str, str | None]:
    """Compute source hash for zip origin."""
    source_path = Path(origin.get("source", ""))

    # The zip is opened through a pinned descriptor, so a path replaced
    # afterwards cannot swap in another archive for the mtime fast path or the
    # extraction used by this check.
    try:
        source = open_zip_source(source_path)
    except Exception as exc:
        return "", str(exc)

    temp_dir: Path | None = None
    try:
        # Fast path: mtime unchanged
        current_mtime = source.mtime_ns
        stored_mtime = origin.get("source_mtime")
        stored_hash = origin.get("content_hash", "")

        if stored_mtime == current_mtime and stored_hash:
            return stored_hash, None

        # Need to extract. The check must judge content, not enforce the flag
        # (which is not part of check_update_available's contract), so symlink
        # entries are materialized here and hashed with Git blob semantics; this
        # never installs or writes outside the temporary extraction directory.
        try:
            extract_result = extract_opened_zip(source, allow_symlinks=True)
            temp_dir = extract_result.extracted_path

            skills = detect_skills(temp_dir, allow_symlinks=True)
            # A root symlink directory whose own path is hidden or excluded is
            # not an entry of this archive, so it must not take part in the
            # single-skill selection.
            skills = _drop_hidden_or_excluded_symlink_candidates(temp_dir, skills)
            if not skills:
                return "", "No skills found in zip source"
            if len(skills) != 1:
                return "", f"Zip must contain exactly one skill (found {len(skills)})"

            skill_path = _resolve_zip_skill_path(
                temp_dir,
                origin,
                skills,
                allow_final_skill_symlink=True,
            )
            return compute_content_hash_with_reason(skill_path)

        except Exception as e:
            return "", f"Failed to extract zip: {e}"
    finally:
        source.close()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
