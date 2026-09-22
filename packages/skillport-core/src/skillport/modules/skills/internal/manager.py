from __future__ import annotations

import errno
import ntpath
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from skillport.modules.skills.public.types import AddResult, RemoveResult
from skillport.shared.config import Config
from skillport.shared.types import SourceType
from skillport.shared.utils import (
    SymlinkPathError,
    checked_components,
    identity_signature,
    no_follow_open_supported,
    open_no_follow,
    parse_frontmatter,
    resolve_inside,
    safe_basename,
    stat_no_follow,
    stat_signature,
    verify_checked_path,
    verify_no_follow_identity,
)

from .validation import validate_skill_record

# GitHub shorthand pattern: owner/repo (no slashes in owner or repo)
GITHUB_SHORTHAND_RE = re.compile(r"^(?P<owner>[a-zA-Z0-9_-]+)/(?P<repo>[a-zA-Z0-9_.-]+)$")

# Built-in skills
BUILTIN_SKILLS = {
    "hello-world": """\
---
name: hello-world
description: A simple hello world skill for testing SkillPort.
metadata:
  skillport:
    category: examples
    tags: [hello, test, demo]
---
# Hello World Skill

This is a sample skill to verify your SkillPort installation is working.

## Usage

When the user asks to test SkillPort or says "hello", respond with a friendly greeting
and confirm that the skill system is operational.

## Example Response

"Hello! The hello-world skill is working correctly."
""",
    "template": """\
---
name: template
description: Replace this with a description of what your skill does.
metadata:
  skillport:
    category: custom
    tags: [template, starter]
---
# My Custom Skill

Replace this content with instructions for the AI agent.

## When to Use

Describe the situations when this skill should be activated.

## Instructions

1. Step one...
2. Step two...
3. Step three...

## Examples

Provide example inputs and expected outputs.
""",
}

EXCLUDE_NAMES = {".git", ".env", ".DS_Store", "__pycache__", "node_modules"}


def has_hidden_or_excluded_component(parts: Iterable[str]) -> bool:
    """Return True when any path component is hidden or in EXCLUDE_NAMES.

    ``.`` and ``..`` are path navigation, not names, so they are ignored.
    """
    return any(
        part not in (".", "..") and (part.startswith(".") or part in EXCLUDE_NAMES)
        for part in parts
    )


@dataclass
class SkillInfo:
    name: str
    source_path: Path
    # Set when the skill root itself is unusable (e.g. a symlinked directory);
    # consumers must fail this skill without reading or copying source_path.
    error: str | None = None


def is_github_shorthand(source: str) -> bool:
    """Check if source matches GitHub shorthand format (owner/repo)."""
    return bool(GITHUB_SHORTHAND_RE.match(source))


def parse_github_shorthand(source: str) -> tuple[str, str] | None:
    """Parse GitHub shorthand format. Returns (owner, repo) or None."""
    match = GITHUB_SHORTHAND_RE.match(source)
    if match:
        return match.group("owner"), match.group("repo")
    return None


def resolve_source(source: str) -> tuple[SourceType, str]:
    """Determine source type and resolved value."""
    if not source:
        raise ValueError("Source is required")
    if source in BUILTIN_SKILLS:
        return SourceType.BUILTIN, source
    if source.startswith("https://github.com/"):
        return SourceType.GITHUB, source

    # Check local path first (priority over GitHub shorthand). The path is
    # inspected without following a symlink component (descriptor chain where
    # available), so a symlink that appears after this check cannot change the
    # classification or a later read.
    candidate = Path(source).expanduser().absolute()
    info: os.stat_result | None = None
    try:
        info = stat_no_follow(candidate)
    except SymlinkPathError as exc:
        if candidate.suffix.lower() == ".zip":
            raise SymlinkPathError(
                f"Zip source path traverses a symlink: {exc.component}", component=exc.component
            ) from None
        raise ValueError(f"Source path traverses a symlink: {exc.component}") from None
    except OSError:
        info = None

    if info is not None:
        if stat.S_ISDIR(info.st_mode):
            return SourceType.LOCAL, str(candidate)
        if stat.S_ISREG(info.st_mode) and candidate.suffix.lower() == ".zip":
            return SourceType.ZIP, str(candidate)
        raise ValueError(f"Source is not a directory or zip file: {candidate}")

    # GitHub shorthand: owner/repo (only if not a local path)
    parsed = parse_github_shorthand(source)
    if parsed:
        owner, repo = parsed
        return SourceType.GITHUB, f"https://github.com/{owner}/{repo}"

    raise ValueError(f"Source not found: {source}")


@dataclass
class StableLocalSource:
    """A local source directory pinned by a no-follow boundary.

    ``snapshot`` is a copy materialized through the pinned descriptor or the
    checked path walk. Every later read must use the snapshot; the original
    path is never resolved again, so a symlink swapped in after the boundary
    check cannot redirect the operation to an external tree.
    """

    snapshot: Path
    temp_root: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)


class _HardlinkMarkers:
    """Marks materialized files whose source reported ``st_nlink > 1``.

    The per-skill snapshot rejects hardlinked regular files with
    ``--allow-symlinks``. The materialized copy must report the same condition
    even when the other link lives outside the source tree, so one extra link
    is created inside a private marker directory that is never part of the
    source tree.
    """

    def __init__(self, parent: Path) -> None:
        self._parent = parent
        self._root: Path | None = None
        self._count = 0

    def mark(self, file: Path) -> None:
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix=".nlink-", dir=self._parent))
        self._count += 1
        os.link(file, self._root / str(self._count))


def acquire_local_source_snapshot(source: Path) -> StableLocalSource:
    """Pin a local source directory and materialize a stable copy of its tree.

    The directory and every ancestor are opened no-follow relative to their
    parent descriptor where available; otherwise the parent chain is checked
    without following a link and the tree is read by path, where every entry
    must match the identity its directory reported. The tree is copied without
    following any entry. A path replaced while the copy is made is detected
    before the snapshot is returned, and a path whose identity cannot be
    verified is rejected instead of being read through a normal path
    resolution.

    Raises:
        SymlinkPathError: a component of the source path is a symlink.
        NotADirectoryError: the source path is not a directory.
        OSError: the source directory cannot be opened.
    """
    target = source if source.is_absolute() else Path.cwd() / source
    if not no_follow_open_supported():
        return _acquire_checked_path_snapshot(target)

    fd = open_no_follow(target, directory=True)
    try:
        before = os.fstat(fd)
        temp_root = Path(tempfile.mkdtemp(prefix="skillport-source-"))
        try:
            # The snapshot always lives under a fixed child of the temporary
            # root and uses a normalized single-component name, so a path such
            # as ``/tmp/source/..`` can neither materialize outside the root nor
            # turn the namespace into ``..``.
            snapshot = temp_root / "source" / safe_basename(target)
            _materialize_tree(fd, snapshot, markers=_HardlinkMarkers(temp_root))
            verify_no_follow_identity(target, before, directory=True)
        except BaseException:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise
        return StableLocalSource(snapshot=snapshot, temp_root=temp_root)
    finally:
        os.close(fd)


def _acquire_checked_path_snapshot(target: Path) -> StableLocalSource:
    """Materialize a local source through checked paths.

    Fallback for platforms without descriptor-relative no-follow opens. The
    source root is inspected without following a link, and the tree is then
    read by path: every directory must still report the identity its parent
    listed and every entry must match the identity its directory reported, so
    a component replaced during the copy fails instead of reading another
    tree. The inspected components are verified once more before the snapshot
    is returned, so a path swapped and restored during the copy is reported
    too.
    """
    records = checked_components(target)
    resolved = records[-1][0]
    before = records[-1][1]
    if not stat.S_ISDIR(before.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(resolved))

    temp_root = Path(tempfile.mkdtemp(prefix="skillport-source-"))
    try:
        snapshot = temp_root / "source" / safe_basename(target)
        _materialize_tree_checked(resolved, snapshot, before, markers=_HardlinkMarkers(temp_root))
        verify_checked_path(target, records)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return StableLocalSource(snapshot=snapshot, temp_root=temp_root)


def _materialize_tree_checked(
    source_path: Path, dest: Path, before: os.stat_result, *, markers: _HardlinkMarkers
) -> None:
    """Materialize a tree read through checked paths.

    Fallback counterpart of :func:`_materialize_tree` for platforms where
    directory descriptors cannot be opened. The directory must still be the
    object the caller inspected, and it is inspected again after the read, so
    a path replaced by a symlink or another directory while the copy runs
    fails instead of being followed. Each entry is opened by path and must
    match the identity the directory reported before it is read.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        current = source_path.lstat()
    except OSError as exc:
        raise ValueError(f"Source changed while being read: {source_path}: {exc}") from exc
    if stat.S_ISLNK(current.st_mode):
        raise SymlinkPathError(
            f"Source path traverses a symlink: {source_path}", component=source_path
        )
    if not stat.S_ISDIR(current.st_mode) or identity_signature(current) != identity_signature(
        before
    ):
        raise ValueError(f"Source changed while being read: {source_path}")
    with os.scandir(source_path) as entries:
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            entry_path = Path(entry.path)
            dest_entry = dest / entry.name
            if stat.S_ISLNK(entry_stat.st_mode):
                _snapshot_symlink_checked(entry_path, dest_entry, entry_stat)
                continue
            if entry.name in EXCLUDE_NAMES:
                if stat.S_ISDIR(entry_stat.st_mode):
                    _materialize_excluded_symlinks_checked(entry_path, dest_entry)
                continue
            _materialize_entry_checked(entry_path, entry_stat, dest_entry, markers=markers)
    try:
        after = source_path.lstat()
    except OSError as exc:
        raise ValueError(f"Source changed while being read: {source_path}: {exc}") from exc
    if stat_signature(after) != stat_signature(current):
        raise ValueError(f"Source changed while being read: {source_path}")


def _materialize_excluded_symlinks_checked(source_path: Path, dest: Path) -> None:
    """Materialize only the symlinks of an excluded subtree through checked paths."""
    with os.scandir(source_path) as entries:
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            entry_path = Path(entry.path)
            if stat.S_ISLNK(entry_stat.st_mode):
                _snapshot_symlink_checked(entry_path, dest / entry.name, entry_stat)
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                _materialize_excluded_symlinks_checked(entry_path, dest / entry.name)


def _materialize_entry_checked(
    source: Path, before: os.stat_result, dest: Path, *, markers: _HardlinkMarkers
) -> None:
    if stat.S_ISDIR(before.st_mode):
        _materialize_tree_checked(source, dest, before, markers=markers)
        return
    if stat.S_ISREG(before.st_mode):
        _snapshot_regular_file(source, dest, before)
        if before.st_nlink > 1:
            markers.mark(dest)
        return
    if hasattr(os, "mkfifo"):
        # Keep the entry type that the per-skill snapshot rejects, so the
        # unsupported entry fails only the skill that contains it.
        os.mkfifo(dest)
        return
    raise ValueError(f"Unsupported file type in skill source: {dest}")


def _snapshot_symlink_checked(source: Path, dest: Path, before: os.stat_result) -> None:
    """Materialize one symlink entry whose identity the checked directory reported."""
    target = os.readlink(source)
    try:
        actual = source.lstat()
    except OSError as exc:
        raise ValueError(f"Source changed while being read: {source}: {exc}") from exc
    if stat_signature(actual) != stat_signature(before):
        raise ValueError(f"Source changed while being read: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, dest)


def _open_child(dir_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SymlinkPathError(f"Source path traverses a symlink: {name}") from None
        raise


def _materialize_tree(source_fd: int, dest: Path, *, markers: _HardlinkMarkers) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with os.scandir(source_fd) as entries:
        for entry in entries:
            # Symlinks are materialized even under excluded names: dropping one
            # here would hide it from the per-skill symlink policy that rejects
            # or preserves entries based on the flag.
            if entry.is_symlink():
                _materialize_entry(source_fd, entry, dest / entry.name, markers=markers)
                continue
            if entry.name in EXCLUDE_NAMES:
                if entry.is_dir(follow_symlinks=False):
                    child_fd = _open_child(source_fd, entry.name)
                    try:
                        _materialize_excluded_symlinks(child_fd, dest / entry.name, markers=markers)
                    finally:
                        os.close(child_fd)
                continue
            _materialize_entry(source_fd, entry, dest / entry.name, markers=markers)


def _materialize_excluded_symlinks(
    source_fd: int, dest: Path, *, markers: _HardlinkMarkers
) -> None:
    """Materialize only the symlinks of an excluded subtree.

    The regular content of an excluded directory stays out of the source
    snapshot, but a symlink inside one must still reach the per-skill symlink
    policy, which rejects links whose own path is hidden or excluded.
    """
    with os.scandir(source_fd) as entries:
        for entry in entries:
            if entry.is_symlink():
                dest_entry = dest / entry.name
                dest_entry.parent.mkdir(parents=True, exist_ok=True)
                _materialize_entry(source_fd, entry, dest_entry, markers=markers)
                continue
            if entry.is_dir(follow_symlinks=False):
                child_fd = _open_child(source_fd, entry.name)
                try:
                    _materialize_excluded_symlinks(child_fd, dest / entry.name, markers=markers)
                finally:
                    os.close(child_fd)


def _materialize_entry(
    source_fd: int, entry: os.DirEntry, dest: Path, *, markers: _HardlinkMarkers
) -> None:
    entry_stat = entry.stat(follow_symlinks=False)
    if stat.S_ISLNK(entry_stat.st_mode):
        os.symlink(os.readlink(entry.name, dir_fd=source_fd), dest)
        return
    if stat.S_ISDIR(entry_stat.st_mode):
        child_fd = _open_child(source_fd, entry.name)
        try:
            _materialize_tree(child_fd, dest, markers=markers)
        finally:
            os.close(child_fd)
        return
    if stat.S_ISREG(entry_stat.st_mode):
        _materialize_regular_file(source_fd, entry.name, dest, entry_stat, markers=markers)
        return
    if hasattr(os, "mkfifo"):
        # Keep the entry type that the per-skill snapshot rejects, so the
        # unsupported entry fails only the skill that contains it.
        os.mkfifo(dest)
        return
    raise ValueError(f"Unsupported file type in skill source: {dest}")


def _materialize_regular_file(
    source_fd: int, name: str, dest: Path, before: os.stat_result, *, markers: _HardlinkMarkers
) -> None:
    source_file_fd = _open_child(source_fd, name)
    try:
        if stat_signature(os.fstat(source_file_fd)) != stat_signature(before):
            raise ValueError(f"Source changed while being read: {name}")
        with os.fdopen(source_file_fd, "rb") as source_file:
            source_file_fd = -1
            with dest.open("wb") as dest_file:
                shutil.copyfileobj(source_file, dest_file)
    finally:
        if source_file_fd >= 0:
            os.close(source_file_fd)

    try:
        os.chmod(dest, stat.S_IMODE(before.st_mode))
        os.utime(dest, ns=(before.st_atime_ns, before.st_mtime_ns))
    except OSError as exc:
        raise ValueError(f"Snapshot metadata could not be preserved: {dest}: {exc}") from exc

    if before.st_nlink > 1:
        markers.mark(dest)


def _load_skill_info(skill_dir: Path, *, allow_symlinks: bool = False) -> SkillInfo:
    skill_md = skill_dir / "SKILL.md"
    if skill_md.is_symlink():
        try:
            if not allow_symlinks:
                raise ValueError(f"Symlinks are not allowed in skills: {skill_md}")
            _validate_preserved_symlink(skill_md, skill_dir)
        except (ValueError, RuntimeError) as exc:
            return SkillInfo(name=skill_dir.name, source_path=skill_dir, error=str(exc))
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")
    meta, _ = parse_frontmatter(skill_md)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid SKILL.md in {skill_dir}: frontmatter must be a mapping")
    name = meta.get("name") or ""
    return SkillInfo(name=name, source_path=skill_dir)


def detect_skills(path: Path, *, allow_symlinks: bool = False) -> list[SkillInfo]:
    """Detect skills under the given path (root or one-level children)."""
    if path.is_symlink():
        return [
            SkillInfo(
                name=path.name,
                source_path=path,
                error=f"Skill source root is a symlink: {path}",
            )
        ]
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Source must be a directory: {path}")

    skills: list[SkillInfo] = []
    root_skill = path / "SKILL.md"
    if root_skill.is_symlink() or root_skill.exists():
        skills.append(_load_skill_info(path, allow_symlinks=allow_symlinks))
        return skills

    for child in sorted(path.iterdir()):
        if child.is_symlink():
            # A symlinked skill root would read SKILL.md and copy content from
            # outside the collection; report it as a per-skill error instead.
            if child.is_dir() and (child / "SKILL.md").exists():
                skills.append(
                    SkillInfo(
                        name=child.name,
                        source_path=child,
                        error=f"Skill root is a symlink: {child}",
                    )
                )
            continue
        child_skill_md = child / "SKILL.md"
        if child.is_dir() and (child_skill_md.is_symlink() or child_skill_md.exists()):
            skills.append(_load_skill_info(child, allow_symlinks=allow_symlinks))
    return skills


def _ensure_frontmatter_name(raw_content: str, target_name: str) -> str:
    """Rewrite frontmatter.name to match target directory for lint compliance."""
    if not raw_content.startswith("---"):
        return raw_content
    try:
        parts = raw_content.split("---", 2)
        if len(parts) < 3:
            return raw_content
        meta = yaml.safe_load(parts[1]) or {}
        if not isinstance(meta, dict):
            return raw_content
        meta["name"] = target_name
        new_meta = yaml.safe_dump(meta, sort_keys=False).strip()
        body = parts[2].lstrip("\n")
        return f"---\n{new_meta}\n---\n{body}"
    except Exception:
        return raw_content


def fail_on_symlinks(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        for entry in dirs + files:
            candidate = Path(root) / entry
            if candidate.is_symlink():
                raise ValueError(f"Symlinks are not allowed in skills: {candidate}")


def _hop_names_directory(hop: str) -> bool:
    """Return True when the raw link target names a directory as its final path.

    ``Path(hop).parts`` drops a trailing separator and a final ``.`` component,
    so ``target.txt/`` and ``target.txt/.`` would otherwise be classified as the
    regular file ``target.txt``. The raw string is inspected before that
    normalization; on Windows both separators are recognized.
    """
    separators = "/\\" if os.sep == "\\" else "/"
    stripped = hop.rstrip(separators)
    if stripped != hop:
        return True
    if stripped == ".":
        return True
    return any(stripped.endswith(separator + ".") for separator in separators)


def _resolve_hop_link(base: Path, hop: str) -> Path:
    """Physically resolve ``base / hop``, leaving only the final component unresolved.

    Directory components (including symlinked ones) are resolved before ``..``
    is applied, so a chain like ``alias -> .`` cannot mask an escape through
    ``alias/../outside.txt``: the ``..`` applies to the resolved directory, not
    to the lexical path. A target that names its final component as a directory
    (a trailing separator or a final ``.``) resolves that component too and
    fails when it is not a directory, so the directory requirement is not lost
    to path normalization.
    """
    current = base.resolve()
    parts = Path(hop).parts
    for part in parts[:-1]:
        if part in (".", ""):
            continue
        if part == "..":
            if not current.exists() or not current.is_dir():
                raise ValueError(f"Symlink path component is not a directory: {base / hop}")
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            current = current.resolve()
        if not current.exists() or not current.is_dir():
            raise ValueError(f"Symlink path component is not a directory: {base / hop}")
    last = parts[-1] if parts else "."
    if last == "..":
        return current.parent
    if last in (".", ""):
        return current
    resolved = current / last
    if _hop_names_directory(hop) and not resolved.is_dir():
        raise ValueError(f"Symlink path component is not a directory: {base / hop}")
    return resolved


def _validate_preserved_symlink(link: Path, skill_root: Path) -> None:
    """Validate one symlink for --allow-symlinks copying.

    Requires: relative link, fully resolvable chain without cycles, final target
    existing inside the same individual skill directory, and no hidden or
    EXCLUDE_NAMES component in the link's own path or in any link target path of
    the chain. Each hop is resolved physically so the skill boundary is checked
    against the real filesystem, not against a lexically normalized path.
    """
    if has_hidden_or_excluded_component(link.relative_to(skill_root).parts):
        raise ValueError(f"Symlink path is hidden or excluded: {link}")
    skill_root_real = skill_root.resolve()
    current = link
    seen: set[Path] = set()
    while current.is_symlink():
        if current in seen:
            raise ValueError(f"Symlink chain cycles: {link}")
        seen.add(current)
        hop = os.readlink(current)
        if not hop:
            raise ValueError(f"Symlink target is empty: {current}")
        if os.path.isabs(hop) or ntpath.isabs(hop) or ntpath.splitdrive(hop)[0]:
            raise ValueError(f"Symlinks must be relative: {current} -> {hop}")
        for part in Path(hop).parts:
            if part in (".", ".."):
                continue
            if part.startswith("."):
                raise ValueError(f"Symlink target is a hidden path: {current} -> {hop}")
            if part in EXCLUDE_NAMES:
                raise ValueError(f"Symlink target is excluded: {current} -> {hop}")
        try:
            current = _resolve_hop_link(current.parent, hop)
        except RuntimeError as exc:
            raise ValueError(f"Symlink chain cycles: {link}") from exc
        except ValueError as exc:
            raise ValueError(f"Symlink target is invalid: {link} -> {hop}") from exc
        if not current.is_relative_to(skill_root_real):
            raise ValueError(f"Symlink target is outside the skill directory: {link} -> {hop}")

    if not current.exists():
        raise ValueError(f"Symlink target does not exist: {link} -> {os.readlink(link)}")
    if not current.resolve().is_relative_to(skill_root_real):
        raise ValueError(
            f"Symlink target is outside the skill directory: {link} -> {os.readlink(link)}"
        )
    try:
        target_stat = current.lstat()
    except OSError as exc:
        raise ValueError(f"Symlink target cannot be inspected: {link}") from exc
    if not stat.S_ISREG(target_stat.st_mode):
        raise ValueError(f"Symlink target is not a regular file: {link} -> {current}")


def _validate_skill_tree_for_preserve(source: Path) -> None:
    """Validate symlinks and hardlinks of a skill tree before preserve copy.

    Every symlink is validated wherever it lives, because the per-skill policy
    rejects a link whose own path is hidden or excluded. Hardlink detection only
    applies to regular files that the preserve copy keeps.
    """
    for root, dirs, files in os.walk(source, followlinks=False):
        for entry in dirs:
            candidate = Path(root) / entry
            if candidate.is_symlink():
                _validate_preserved_symlink(candidate, source)
        for entry in files:
            candidate = Path(root) / entry
            if candidate.is_symlink():
                _validate_preserved_symlink(candidate, source)
                continue
            if entry in EXCLUDE_NAMES or entry.startswith("."):
                continue
            st = candidate.lstat()
            # Directories grow st_nlink from child counts; only regular files are hardlinks
            if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                raise ValueError(f"Hardlinks are not allowed in skills: {candidate}")


def _snapshot_regular_file(source: Path, dest: Path, before: os.stat_result) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(source, flags)
        if stat_signature(os.fstat(fd)) != stat_signature(before):
            raise ValueError(f"Source changed while snapshotting: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(fd, "rb") as source_file, dest.open("wb") as dest_file:
            fd = None
            shutil.copyfileobj(source_file, dest_file)
    except OSError as exc:
        raise ValueError(f"Source could not be snapshotted: {source}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)

    try:
        os.chmod(dest, stat.S_IMODE(before.st_mode))
        os.utime(dest, ns=(before.st_atime_ns, before.st_mtime_ns))
    except OSError as exc:
        raise ValueError(f"Snapshot metadata could not be preserved: {source}: {exc}") from exc

    try:
        after = source.lstat()
    except OSError as exc:
        raise ValueError(f"Source changed while snapshotting: {source}") from exc
    if stat_signature(after) != stat_signature(before):
        raise ValueError(f"Source changed while snapshotting: {source}")


def _snapshot_symlink(source: Path, dest: Path) -> None:
    target = os.readlink(source)
    if target != os.readlink(source):
        raise ValueError(f"Source changed while snapshotting: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, dest)


def _snapshot_excluded_symlinks(source: Path, dest: Path) -> None:
    """Materialize only the symlinks of an excluded subtree.

    The regular content of a hidden or EXCLUDE_NAMES directory stays out of the
    preserve copy, but a symlink inside one must still reach the per-skill
    symlink policy, which rejects links whose own path is hidden or excluded.
    """
    with os.scandir(source) as entries:
        for entry in entries:
            source_entry = Path(entry.path)
            if entry.is_symlink():
                _snapshot_symlink(source_entry, dest / entry.name)
                continue
            if entry.is_dir(follow_symlinks=False):
                _snapshot_excluded_symlinks(source_entry, dest / entry.name)


def _snapshot_tree(source: Path, dest: Path, *, allow_symlinks: bool) -> None:
    try:
        before = source.lstat()
    except OSError as exc:
        raise ValueError(f"Source could not be snapshotted: {source}: {exc}") from exc

    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"Skill source is not a directory: {source}")

    dest.mkdir(parents=True, exist_ok=True)
    try:
        with os.scandir(source) as entries:
            for entry in entries:
                source_entry = Path(entry.path)
                dest_entry = dest / entry.name
                entry_stat = entry.stat(follow_symlinks=False)
                excluded = entry.name in EXCLUDE_NAMES or entry.name.startswith(".")

                if stat.S_ISLNK(entry_stat.st_mode):
                    # Materialized before the excluded filter: a link whose own
                    # path is hidden or excluded must be rejected by the
                    # per-skill policy instead of being dropped here.
                    _snapshot_symlink(source_entry, dest_entry)
                elif excluded:
                    if stat.S_ISDIR(entry_stat.st_mode):
                        _snapshot_excluded_symlinks(source_entry, dest_entry)
                elif stat.S_ISDIR(entry_stat.st_mode):
                    _snapshot_tree(source_entry, dest_entry, allow_symlinks=allow_symlinks)
                elif stat.S_ISREG(entry_stat.st_mode):
                    if allow_symlinks and entry_stat.st_nlink > 1:
                        raise ValueError(f"Hardlinks are not allowed in skills: {source_entry}")
                    _snapshot_regular_file(source_entry, dest_entry, entry_stat)
                else:
                    raise ValueError(f"Unsupported file type in skill source: {source_entry}")
    except OSError as exc:
        raise ValueError(f"Source could not be snapshotted: {source}: {exc}") from exc

    try:
        after = source.lstat()
    except OSError as exc:
        raise ValueError(f"Source changed while snapshotting: {source}") from exc
    if stat_signature(after) != stat_signature(before):
        raise ValueError(f"Source changed while snapshotting: {source}")


def snapshot_skill_dir(source: Path, *, allow_symlinks: bool = False) -> Path:
    if source.is_symlink():
        raise ValueError(f"Skill source root is a symlink: {source}")
    if not source.exists() or not source.is_dir():
        raise ValueError(f"Skill source is not a directory: {source}")
    if not allow_symlinks:
        fail_on_symlinks(source)

    snapshot_parent = Path(tempfile.mkdtemp(prefix="skillport-snapshot-"))
    snapshot = snapshot_parent / source.name
    try:
        _snapshot_tree(source, snapshot, allow_symlinks=allow_symlinks)
        if allow_symlinks:
            _validate_skill_tree_for_preserve(snapshot)
        else:
            fail_on_symlinks(snapshot)
        return snapshot
    except BaseException:
        shutil.rmtree(snapshot_parent, ignore_errors=True)
        raise


def copy_skill_snapshot(source: Path, dest: Path, *, allow_symlinks: bool) -> None:
    def _ignore(_src, names):
        return {n for n in names if n in EXCLUDE_NAMES or n.startswith(".")}

    if source.is_symlink():
        raise ValueError(f"Skill source root is a symlink: {source}")

    shutil.copytree(source, dest, dirs_exist_ok=False, ignore=_ignore, symlinks=allow_symlinks)


def copy_skill_dir(source: Path, dest: Path, *, allow_symlinks: bool = False) -> None:
    snapshot = snapshot_skill_dir(source, allow_symlinks=allow_symlinks)
    try:
        copy_skill_snapshot(snapshot, dest, allow_symlinks=allow_symlinks)
    finally:
        shutil.rmtree(snapshot.parent, ignore_errors=True)


def _validate_skill_file(skill_dir: Path) -> None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_dir}")
    meta, body = parse_frontmatter(skill_md)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid SKILL.md in {skill_dir}: frontmatter must be a mapping")

    name = meta.get("name")
    description = meta.get("description", "")

    # Spec: frontmatter.name/description are必須
    if not name or not str(name).strip():
        raise ValueError(f"Invalid SKILL.md in {skill_dir}: frontmatter.name is required")
    if not description or not str(description).strip():
        raise ValueError(f"Invalid SKILL.md in {skill_dir}: frontmatter.description is required")

    name = str(name).strip()
    description = str(description)
    lines = body.count("\n") + (1 if body and not body.endswith("\n") else 0)

    # Spec: frontmatter.name/description are必須
    if "name" not in meta or not str(meta.get("name", "")).strip():
        raise ValueError(f"Invalid SKILL.md in {skill_dir}: frontmatter.name is required")
    if "description" not in meta or not str(meta.get("description", "")).strip():
        raise ValueError(f"Invalid SKILL.md in {skill_dir}: frontmatter.description is required")

    issues = validate_skill_record(
        {
            "name": name,
            "description": description,
            "lines": lines,
            "path": str(skill_dir),
        },
        strict=True,
        meta=meta,
    )
    # strict=True returns only fatal issues
    if issues:
        raise ValueError("; ".join([i.message for i in issues]))
    # Warnings printed but non-fatal
    for issue in issues:
        if issue.severity != "fatal":
            print(f"[WARN] {skill_dir}: {issue.message}", file=sys.stderr)

    if name != skill_dir.name:
        raise ValueError(
            f"Invalid SKILL.md in {skill_dir}: name '{name}' must match directory '{skill_dir.name}'"
        )


def add_builtin(name: str, *, config: Config, force: bool) -> AddResult:
    if name not in BUILTIN_SKILLS:
        raise ValueError(f"Unknown built-in skill: {name}")

    dest_root = config.skills_dir
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / name
    if dest.exists():
        if not force:
            return AddResult(
                success=False,
                skill_id=name,
                message=f"Skill '{name}' exists. Use --force to overwrite.",
                skipped=[name],
            )
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)
    content = _ensure_frontmatter_name(BUILTIN_SKILLS[name], name)
    (dest / "SKILL.md").write_text(content, encoding="utf-8")
    return AddResult(
        success=True,
        skill_id=name,
        message=f"Added '{name}' to {dest_root}",
        added=[name],
    )


def add_local(
    source_path: Path,
    skills: list[SkillInfo],
    *,
    config: Config,
    keep_structure: bool,
    force: bool,
    namespace_override: str | None = None,
    rename_single_to: str | None = None,
    allow_symlinks: bool = False,
) -> list[AddResult]:
    target_root = config.skills_dir
    target_root.mkdir(parents=True, exist_ok=True)

    results: list[AddResult] = []
    namespace = namespace_override or safe_basename(source_path)
    seen_ids: set[str] = set()

    for skill in skills:
        if skill.error is not None:
            skill_name = skill.name
            if rename_single_to and len(skills) == 1:
                skill_name = rename_single_to
            skill_id = skill_name if not keep_structure else f"{namespace}/{skill_name}"
            results.append(
                AddResult(
                    success=False,
                    skill_id=skill_id,
                    message=skill.error,
                )
            )
            continue

        snapshot: Path | None = None
        skill_name = rename_single_to if rename_single_to and len(skills) == 1 else skill.name
        skill_id = skill_name if not keep_structure else f"{namespace}/{skill_name}"
        dest: Path | None = None
        try:
            snapshot = snapshot_skill_dir(skill.source_path, allow_symlinks=allow_symlinks)
            _validate_skill_file(snapshot)

            if skill_id in seen_ids:
                raise ValueError(f"Duplicate skill id detected: {skill_id}")
            seen_ids.add(skill_id)

            dest = target_root / skill_id
            if dest.exists():
                if not force:
                    results.append(
                        AddResult(
                            success=False,
                            skill_id=skill_id,
                            message=f"Skill '{skill_id}' exists.",
                        )
                    )
                    continue
                shutil.rmtree(dest)

            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_skill_snapshot(snapshot, dest, allow_symlinks=allow_symlinks)
            if rename_single_to and len(skills) == 1:
                skill_md_path = dest / "SKILL.md"
                raw = skill_md_path.read_text(encoding="utf-8")
                skill_md_path.write_text(
                    _ensure_frontmatter_name(raw, skill_name), encoding="utf-8"
                )
                source_skill_md = snapshot / "SKILL.md"
                if source_skill_md.exists() and not source_skill_md.is_symlink():
                    source_stat = source_skill_md.stat()
                    os.chmod(skill_md_path, stat.S_IMODE(source_stat.st_mode))
                    os.utime(
                        skill_md_path,
                        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
                    )
            results.append(
                AddResult(success=True, skill_id=skill_id, message=f"Added '{skill_id}'")
            )
        except Exception as exc:
            if dest is not None and dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            results.append(
                AddResult(
                    success=False,
                    skill_id=skill_id,
                    message=f"Failed to add '{skill_id}': {exc}",
                )
            )
        finally:
            if snapshot is not None:
                shutil.rmtree(snapshot.parent, ignore_errors=True)

    return results


def remove_skill(skill_id: str, *, config: Config) -> RemoveResult:
    dest = config.skills_dir / skill_id
    resolve_inside(config.skills_dir, skill_id)  # traversal guard
    if not dest.exists():
        return RemoveResult(
            success=False, skill_id=skill_id, message=f"Skill not found: {skill_id}"
        )
    if not dest.is_dir():
        return RemoveResult(success=False, skill_id=skill_id, message=f"Not a directory: {dest}")
    shutil.rmtree(dest)
    return RemoveResult(success=True, skill_id=skill_id, message=f"Removed '{skill_id}'")
