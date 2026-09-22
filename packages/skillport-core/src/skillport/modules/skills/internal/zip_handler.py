"""Zip file handling for skill extraction."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from skillport.shared.utils import (
    SymlinkPathError,
    open_no_follow,
    reject_symlink_ancestor,
    verify_no_follow_identity,
)

from .manager import has_hidden_or_excluded_component

# Security limits (consistent with GitHub tarball handling)
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024  # 100MB total (multimodal files: images, PDFs)
MAX_ZIP_FILES = 1000  # Maximum number of files in zip
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25MB per file (large PDFs)


@dataclass
class ZipExtractResult:
    """Result of extracting a zip file."""

    extracted_path: Path
    file_count: int
    source_mtime_ns: int = 0


class ZipSource:
    """A ZIP file opened without following a symlink component.

    The descriptor, the reported mtime, and the ``ZipFile`` all refer to the
    same opened file, so a path replaced after acquisition cannot redirect an
    entry scan, an extraction, or a hash to another archive.
    """

    def __init__(
        self,
        fd: int,
        file: BinaryIO,
        zip_file: zipfile.ZipFile,
        mtime_ns: int,
    ) -> None:
        self.mtime_ns = mtime_ns
        self.zip_file = zip_file
        self._fd = fd
        self._file = file

    def close(self) -> None:
        try:
            self.zip_file.close()
        finally:
            self._file.close()


def open_zip_source(zip_path: Path) -> ZipSource:
    """Open a ZIP source without following symlinks in any path component.

    Raises:
        SymlinkPathError: a component of the zip path is a symlink.
        FileNotFoundError: the zip file does not exist.
        ValueError: the source is not a zip file.
    """
    try:
        fd = open_no_follow(zip_path, directory=False)
    except SymlinkPathError as exc:
        raise SymlinkPathError(
            f"Zip source path traverses a symlink: {exc.component}", component=exc.component
        ) from None
    except (FileNotFoundError, NotADirectoryError):
        raise FileNotFoundError(f"Zip file not found: {zip_path}") from None
    except OSError as exc:
        raise FileNotFoundError(f"Zip file not found: {zip_path}") from exc

    try:
        source_stat = os.fstat(fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Not a valid zip file: {zip_path}")
        verify_no_follow_identity(zip_path, source_stat, directory=False)
    except BaseException:
        os.close(fd)
        raise

    file = os.fdopen(fd, "rb")
    try:
        zip_file = zipfile.ZipFile(file, "r")
    except zipfile.BadZipFile:
        file.close()
        raise ValueError(f"Not a valid zip file: {zip_path}") from None
    except BaseException:
        file.close()
        raise

    return ZipSource(
        fd=fd,
        file=file,
        zip_file=zip_file,
        mtime_ns=source_stat.st_mtime_ns,
    )


def zip_has_symlink_entries(source: ZipSource) -> bool:
    """Return True if the opened zip contains any 0xA000 symlink entry.

    Mirrors extract_zip, which rejects flagless extraction on any symlink
    entry, including hidden ones.
    """
    for info in source.zip_file.infolist():
        if is_zip_symlink_entry(info):
            return True
    return False


def zip_hidden_or_excluded_symlink_entries(source: ZipSource, *, prefix: str) -> list[str]:
    """List 0xA000 entries under ``prefix`` whose own path or target is hidden or excluded.

    Mirrors the per-skill symlink policy for the selected skill only: such a
    link must fail the skill instead of being skipped during extraction or
    reported as up to date, while an entry outside ``prefix`` follows the
    normal exclusion rules and must not fail the skill. The extraction resource
    limits guard every payload read. The member name is normalized exactly like
    extraction, and every inspected symlink entry's target string is checked,
    so a chain reaching a hidden/excluded name through another link is caught
    by that link's own entry.
    """
    infos = source.zip_file.infolist()
    if len(infos) > MAX_ZIP_FILES:
        raise ValueError(f"Zip contains too many files: {len(infos)} > {MAX_ZIP_FILES}")

    names: list[str] = []
    total_size = 0
    for info in infos:
        if not is_zip_symlink_entry(info):
            continue
        rel_posix = _zip_rel_posix_path(info.filename)
        if not _zip_path_within_prefix(rel_posix, prefix):
            continue
        if has_hidden_or_excluded_component(PurePosixPath(rel_posix).parts):
            names.append(info.filename)
            continue
        if info.file_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File too large: {info.filename} ({info.file_size} > {MAX_FILE_BYTES})"
            )
        total_size += info.file_size
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError(f"Extracted size exceeds limit: {total_size} > {MAX_EXTRACTED_BYTES}")
        target = _zip_symlink_target(info, source.zip_file)
        if has_hidden_or_excluded_component(Path(target).parts):
            names.append(info.filename)
    return names


def _zip_path_within_prefix(rel_posix: str, prefix: str) -> bool:
    """Return True when a normalized entry path is ``prefix`` itself or below it."""
    if not prefix:
        return True
    return rel_posix == prefix or rel_posix.startswith(prefix + "/")


def _zip_rel_posix_path(name: str) -> str:
    """Normalize a zip member name to a safe POSIX-style relative path."""
    if not name:
        raise ValueError("Invalid zip entry: empty name")

    normalized = name.replace("\\", "/")
    p = PurePosixPath(normalized)

    # Reject absolute paths (covers "/x" and UNC-like "\\\\server\\share" after normalization)
    if p.is_absolute():
        raise ValueError(f"Path traversal detected: {name}")

    parts = [part for part in p.parts if part not in (".", "")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Path traversal detected: {name}")

    # Reject drive-like prefixes / invalid Windows characters early (':' is invalid on Windows)
    if any(":" in part for part in parts):
        raise ValueError(f"Path traversal detected: {name}")

    return "/".join(parts)


def is_zip_symlink_entry(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0xF000 == 0xA000


def _zip_symlink_target(info: zipfile.ZipInfo, zf: zipfile.ZipFile) -> str:
    """Read the link target string stored as a 0xA000 entry's content bytes."""
    if info.file_size > MAX_FILE_BYTES:
        raise ValueError(f"File too large: {info.filename} ({info.file_size} > {MAX_FILE_BYTES})")
    with zf.open(info, "r") as src:
        return os.fsdecode(src.read())


def _materialize_zip_symlink(
    dest_root: Path, rel_posix: str, info: zipfile.ZipInfo, zf: zipfile.ZipFile
) -> None:
    """Create a zip 0xA000 entry as a real symlink from its content bytes."""
    target = _zip_symlink_target(info, zf)
    dest_path = dest_root.joinpath(*PurePosixPath(rel_posix).parts)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, dest_path)
    except OSError as e:
        raise ValueError(f"Failed to create symlink {info.filename}: {e}") from e


def extract_zip(zip_path: Path, *, allow_symlinks: bool = False) -> ZipExtractResult:
    """Extract a zip file to a temporary directory.

    Args:
        zip_path: Path to the zip file to extract
        allow_symlinks: Materialize 0xA000 symlink entries instead of rejecting them

    Returns:
        ZipExtractResult: Path to extracted directory, file count, and the
        mtime of the opened source file

    Raises:
        FileNotFoundError: If zip file does not exist
        ValueError: If zip is invalid or violates security constraints
        SymlinkPathError: If a component of the zip path is a symlink
    """
    source = open_zip_source(zip_path)
    try:
        return extract_opened_zip(source, allow_symlinks=allow_symlinks)
    finally:
        source.close()


def extract_opened_zip(source: ZipSource, *, allow_symlinks: bool = False) -> ZipExtractResult:
    """Extract an already opened zip source through its opened descriptor."""
    temp_dir = Path(tempfile.mkdtemp(prefix="skillport-zip-"))
    total_size = 0
    file_count = 0
    # Normalized paths already written: a second entry collapsing to the same
    # path (e.g. "link" then "./link") must not overwrite the first, because a
    # materialized symlink at that path would redirect the write outside the
    # extraction root.
    written_paths: set[str] = set()

    try:
        zf = source.zip_file
        # Check total file count
        if len(zf.namelist()) > MAX_ZIP_FILES:
            raise ValueError(f"Zip contains too many files: {len(zf.namelist())} > {MAX_ZIP_FILES}")

        for info in zf.infolist():
            symlink_entry = is_zip_symlink_entry(info)
            if not symlink_entry and info.is_dir():
                continue

            name = info.filename
            rel_posix = _zip_rel_posix_path(name)

            parts = PurePosixPath(rel_posix).parts

            # Symlink detection (posix mode 0xA000)
            if symlink_entry:
                if not allow_symlinks:
                    raise ValueError(f"Symlink detected in zip: {name}")
                # Materialized before the hidden/excluded filter: a link whose
                # own path is hidden or excluded must be rejected per skill by
                # the common preserve validation instead of being dropped here.
                if rel_posix in written_paths:
                    raise ValueError(f"Duplicate zip entry: {name}")
                total_size += info.file_size
                if total_size > MAX_EXTRACTED_BYTES:
                    raise ValueError(
                        f"Extracted size exceeds limit: {total_size} > {MAX_EXTRACTED_BYTES}"
                    )
                reject_symlink_ancestor(temp_dir, rel_posix)
                _materialize_zip_symlink(temp_dir, rel_posix, info, zf)
                written_paths.add(rel_posix)
                file_count += 1
                continue

            # Skip hidden files and excluded names
            if has_hidden_or_excluded_component(parts):
                continue

            # Single file size check
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError(f"File too large: {name} ({info.file_size} > {MAX_FILE_BYTES})")

            # Cumulative size check
            total_size += info.file_size
            if total_size > MAX_EXTRACTED_BYTES:
                raise ValueError(
                    f"Extracted size exceeds limit: {total_size} > {MAX_EXTRACTED_BYTES}"
                )

            if rel_posix in written_paths:
                raise ValueError(f"Duplicate zip entry: {name}")

            # Extract file (manual to avoid platform-specific path quirks)
            reject_symlink_ancestor(temp_dir, rel_posix)
            dest_path = temp_dir.joinpath(*PurePosixPath(rel_posix).parts)
            if dest_path.is_symlink():
                raise ValueError(f"Archive path traverses a symlink: {name}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written_paths.add(rel_posix)
            file_count += 1

        return ZipExtractResult(
            extracted_path=temp_dir,
            file_count=file_count,
            source_mtime_ns=source.mtime_ns,
        )

    except Exception:
        # Cleanup temp dir on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


__all__ = [
    "ZipExtractResult",
    "ZipSource",
    "extract_opened_zip",
    "extract_zip",
    "is_zip_symlink_entry",
    "open_zip_source",
    "zip_has_symlink_entries",
    "zip_hidden_or_excluded_symlink_entries",
    "MAX_EXTRACTED_BYTES",
    "MAX_ZIP_FILES",
    "MAX_FILE_BYTES",
]
