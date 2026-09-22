"""Pure utility helpers shared across modules."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Any

import yaml

# Re-export normalize_token from filters for backwards compatibility
from .filters import normalize_token


class SymlinkPathError(ValueError):
    def __init__(self, message: str, *, component: Path | None = None) -> None:
        super().__init__(message)
        self.component = component


def parse_frontmatter(file_path: Path) -> tuple[dict[str, Any], str]:
    """Parse a Markdown file with YAML frontmatter.

    Returns (metadata, body). If frontmatter is absent or invalid, metadata is {}.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                if not isinstance(meta, dict):
                    meta = {}
            except yaml.YAMLError:
                meta = {}
            body = parts[2].lstrip("\n")
            return meta, body
    return {}, text


def resolve_inside(base: Path, relative_path: str) -> Path:
    """Resolve relative_path within base; prevent traversal."""
    target = (base / relative_path).resolve()
    try:
        if not target.is_relative_to(base.resolve()):
            raise PermissionError(f"Path traversal detected: {relative_path}")
    except AttributeError:
        base_resolved = str(base.resolve())
        try:
            common = os.path.commonpath([base_resolved, str(target)])
        except ValueError:
            raise PermissionError(f"Path traversal detected: {relative_path}") from None

        if os.path.normcase(common) != os.path.normcase(base_resolved):
            raise PermissionError(f"Path traversal detected: {relative_path}")
    return target


def reject_symlink_ancestor(base: Path, relative_path: str) -> None:
    """Reject archive paths whose parent directory is an already-extracted symlink.

    During incremental extraction, writing through such a parent would place
    content outside the extraction root.
    """
    current = base
    for part in relative_path.replace("\\", "/").split("/")[:-1]:
        if not part or part == ".":
            continue
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Archive path traverses a symlink directory: {relative_path}")


def find_symlink_path_component(path: Path) -> Path | None:
    target = path if path.is_absolute() else Path.cwd() / path
    current = Path(target.anchor)
    for part in target.parts[1:]:
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            return current
    return None


def safe_basename(path: str | Path) -> str:
    """Return a single-component name derived from ``path``.

    The name comes from the lexically normalized path, never from the raw final
    component, so a valid path such as ``/tmp/source/..`` cannot produce ``..``
    (or an empty name) when the value is used as a directory or namespace.
    ``.``, the filesystem root, and an empty path fall back to ``"source"``.
    """
    normalized = Path(os.path.normpath(str(path)))
    name = normalized.name
    if name in ("", ".", ".."):
        return "source"
    return name


def no_follow_open_supported() -> bool:
    """Return True when descriptor-relative no-follow opens are available."""
    return os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW")


def open_no_follow(path: Path, *, directory: bool | None = None) -> int:
    """Open a path without following any symlink component.

    Where descriptor-relative no-follow opens are available, each component is
    opened relative to the descriptor of its parent with ``O_NOFOLLOW``, so a
    component replaced by a symlink between the check and the read cannot
    redirect the descriptor to another tree. Otherwise every component is
    checked without following it and the final entry must match the identity
    the check reported; a replaced component fails the operation instead of
    being followed. The fallback opens the target by path, so it cannot return
    a descriptor for a directory on a platform that does not allow directory
    descriptors (for example Windows); callers that need a directory must read
    it through checked paths instead. The caller owns the returned descriptor
    and must close it.

    Args:
        path: Path to open. A relative path is anchored at the current
            working directory.
        directory: ``True`` requires the final component to be a directory,
            ``False`` requires it not to be one, ``None`` accepts either.

    Returns:
        An open file descriptor.

    Raises:
        SymlinkPathError: a component of ``path`` is a symlink.
        OSError: the path cannot be opened (missing, no permission, not a
            directory component, ...).
    """
    if not no_follow_open_supported():
        return _open_checked_path(path, directory=directory)

    target = path if path.is_absolute() else Path.cwd() / path
    parts = target.parts
    if not parts:
        raise ValueError(f"Invalid source path: {path}")

    fd = os.open(parts[0], os.O_RDONLY | os.O_NONBLOCK)
    try:
        for index in range(1, len(parts)):
            try:
                next_fd = os.open(
                    parts[index],
                    os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    component = Path(*parts[: index + 1])
                    raise SymlinkPathError(
                        f"Source path traverses a symlink: {component}",
                        component=component,
                    ) from None
                raise
            os.close(fd)
            fd = next_fd

        mode = os.fstat(fd).st_mode
        if directory is True and not stat.S_ISDIR(mode):
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(target))
        if directory is False and stat.S_ISDIR(mode):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", str(target))
    except BaseException:
        os.close(fd)
        raise
    return fd


def _checked_target(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return the identity fields compared when a path must not change under a read."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def identity_signature(value: os.stat_result) -> tuple[int, int, int]:
    """Return the fields that identify one filesystem object.

    Directories legitimately gain and lose entries while an operation runs
    (for example the snapshot root created next to the source), so a parent
    directory is only compared by identity. The final target is compared with
    the full :func:`stat_signature` where a change must fail the read.
    """
    return (value.st_dev, value.st_ino, value.st_mode)


def checked_components(path: Path) -> list[tuple[Path, os.stat_result]]:
    """Inspect every component of ``path`` without following a link.

    Each component is inspected with ``lstat`` and a symlink/reparse point is
    rejected, so the returned records describe the real directory chain and
    the last record is the resolved final target (a trailing ``..`` moves the
    target back to its parent, exactly as path resolution does once no
    component is a link). Comparing the records again with
    :func:`verify_checked_path` after a read reports a component that was
    renamed away and back during the read instead of silently accepting it.

    Raises:
        SymlinkPathError: a component of ``path`` is a symlink.
        OSError: the path cannot be inspected.
    """
    target = _checked_target(path)
    if not target.parts:
        raise ValueError(f"Invalid source path: {path}")
    anchor = Path(target.anchor)
    records: list[tuple[Path, os.stat_result]] = [(anchor, os.lstat(anchor))]
    for part in target.parts[1:]:
        if part in ("", "."):
            continue
        if part == "..":
            if len(records) > 1:
                records.pop()
            continue
        current = records[-1][0] / part
        result = os.lstat(current)
        if stat.S_ISLNK(result.st_mode):
            raise SymlinkPathError(f"Source path traverses a symlink: {current}", component=current)
        records.append((current, result))
    return records


def verify_checked_path(path: Path, records: list[tuple[Path, os.stat_result]]) -> None:
    """Fail when ``path`` no longer matches records captured by checked_components.

    A component replaced by a link is reported by the walk; a component
    renamed away and restored during the read is reported by its changed
    identity.

    Raises:
        SymlinkPathError: a component of ``path`` is a symlink now.
        ValueError: the path no longer resolves to the recorded components.
    """
    current = checked_components(path)
    if len(current) != len(records):
        raise ValueError(f"Source path changed while being read: {path}")
    last = len(records) - 1
    for index, ((recorded_path, recorded_stat), (current_path, current_stat)) in enumerate(
        zip(records, current)
    ):
        if recorded_path != current_path:
            raise ValueError(f"Source path changed while being read: {path}")
        if index == last:
            unchanged = stat_signature(recorded_stat) == stat_signature(current_stat)
        else:
            unchanged = identity_signature(recorded_stat) == identity_signature(current_stat)
        if not unchanged:
            raise ValueError(f"Source path changed while being read: {path}")


def _open_descriptor(path: Path, flags: int) -> int:
    return os.open(path, flags)


def _open_entry_descriptor(path: Path, flags: int) -> int:
    try:
        return _open_descriptor(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SymlinkPathError(
                f"Source path traverses a symlink: {path}", component=path
            ) from None
        raise


def _entry_flags(expected: os.stat_result) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if stat.S_ISDIR(expected.st_mode) and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    return flags


def open_checked_entry(parent: Path, name: str, expected: os.stat_result) -> int:
    """Open ``parent / name`` and verify it is the entry the checked walk reported.

    The path is opened normally, but the descriptor must match ``expected``
    (the entry stat captured by the component walk), so a path replaced before
    the open fails the operation instead of reading another tree. The caller
    owns the returned descriptor.

    Raises:
        SymlinkPathError: the path is a symlink (loop).
        ValueError: the opened entry is not the recorded one.
    """
    path = parent / name
    try:
        fd = _open_entry_descriptor(path, _entry_flags(expected))
    except OSError as exc:
        raise ValueError(f"Source changed while being read: {path}: {exc}") from exc
    try:
        if identity_signature(os.fstat(fd)) != identity_signature(expected):
            raise ValueError(f"Source changed while being read: {path}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_checked_path(path: Path, *, directory: bool | None) -> int:
    """Open ``path`` after checking every component without following a link.

    Fallback for platforms without ``O_NOFOLLOW`` and descriptor-relative
    opens (for example Windows, where directory descriptors cannot be opened
    either). The component walk rejects an existing link and is repeated just
    before the open, and the opened entry must match the identity the walk
    reported, so a component replaced between the check and the open fails the
    operation instead of being followed.
    """
    records = checked_components(path)
    open_target = records[-1][0]
    expected = records[-1][1]
    if directory is True and not stat.S_ISDIR(expected.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(open_target))
    if directory is False and stat.S_ISDIR(expected.st_mode):
        raise IsADirectoryError(errno.EISDIR, "Is a directory", str(open_target))
    verify_checked_path(path, records)
    return open_checked_entry(open_target.parent, open_target.name, expected)


def stat_no_follow(path: Path) -> os.stat_result:
    """Return the identity of ``path``, rejecting a symlink component.

    Uses the descriptor-relative no-follow open where available; otherwise a
    checked component walk that inspects each component with ``lstat``.

    Raises:
        SymlinkPathError: a component of ``path`` is a symlink.
        OSError: the path cannot be inspected.
    """
    if not no_follow_open_supported():
        return checked_components(path)[-1][1]

    fd = open_no_follow(path)
    try:
        return os.fstat(fd)
    finally:
        os.close(fd)


def verify_no_follow_identity(
    path: Path, expected: os.stat_result, *, directory: bool | None = None
) -> None:
    """Fail when ``path`` no longer opens to the identity recorded in ``expected``.

    The path is reopened through the no-follow chain and its device and inode
    are compared, so a path replaced after the first acquisition is reported
    instead of being silently used.
    """
    if no_follow_open_supported():
        fd = open_no_follow(path, directory=directory)
        try:
            actual = os.fstat(fd)
        finally:
            os.close(fd)
    else:
        actual = stat_no_follow(path)
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError(f"Source path changed while being read: {path}")


__all__ = [
    "SymlinkPathError",
    "checked_components",
    "find_symlink_path_component",
    "identity_signature",
    "no_follow_open_supported",
    "normalize_token",
    "open_checked_entry",
    "open_no_follow",
    "parse_frontmatter",
    "reject_symlink_ancestor",
    "resolve_inside",
    "safe_basename",
    "stat_no_follow",
    "stat_signature",
    "verify_checked_path",
    "verify_no_follow_identity",
]
