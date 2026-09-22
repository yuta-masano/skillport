"""Unit tests for zip_handler module."""

import os
import shutil
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

import pytest

from skillport.modules.skills.internal.zip_handler import (
    MAX_EXTRACTED_BYTES,
    MAX_FILE_BYTES,
    MAX_ZIP_FILES,
    extract_zip,
    open_zip_source,
    zip_hidden_or_excluded_symlink_entries,
)


def _symlink_capable() -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink("target", os.path.join(tmp, "probe"))
    except (OSError, NotImplementedError):
        return False
    return True


@pytest.fixture
def require_symlink():
    if not _symlink_capable():
        pytest.skip("symlinks cannot be created on this platform")


class TestExtractZip:
    """Tests for extract_zip function."""

    def test_extract_single_skill_zip(self, tmp_path):
        """Single skill zip is extracted correctly."""
        # Create a zip with SKILL.md
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: my-skill\n---\ncontent")
            zf.writestr("README.md", "# My Skill")

        result = extract_zip(zip_path)

        assert result.extracted_path.exists()
        assert result.file_count == 2
        assert (result.extracted_path / "SKILL.md").exists()
        assert (result.extracted_path / "README.md").exists()

        # Cleanup
        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_extract_multiple_skills_zip(self, tmp_path):
        """Multiple skills in zip are extracted correctly."""
        zip_path = tmp_path / "skills.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("skill-a/SKILL.md", "---\nname: skill-a\n---\na")
            zf.writestr("skill-b/SKILL.md", "---\nname: skill-b\n---\nb")
            zf.writestr("skill-c/SKILL.md", "---\nname: skill-c\n---\nc")

        result = extract_zip(zip_path)

        assert result.extracted_path.exists()
        assert result.file_count == 3
        assert (result.extracted_path / "skill-a" / "SKILL.md").exists()
        assert (result.extracted_path / "skill-b" / "SKILL.md").exists()
        assert (result.extracted_path / "skill-c" / "SKILL.md").exists()

        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_extract_preserves_directory_structure(self, tmp_path):
        """Directory structure is preserved after extraction."""
        zip_path = tmp_path / "nested.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("my-skill/SKILL.md", "---\nname: my-skill\n---\n")
            zf.writestr("my-skill/lib/utils.py", "# utils")
            zf.writestr("my-skill/assets/logo.txt", "logo")

        result = extract_zip(zip_path)

        assert (result.extracted_path / "my-skill" / "SKILL.md").exists()
        assert (result.extracted_path / "my-skill" / "lib" / "utils.py").exists()
        assert (result.extracted_path / "my-skill" / "assets" / "logo.txt").exists()

        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)


class TestExtractZipSecurity:
    """Security tests for extract_zip function."""

    def test_rejects_path_traversal(self, tmp_path):
        """Zip with path traversal is rejected."""
        zip_path = tmp_path / "malicious.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Create entry with path traversal
            info = zipfile.ZipInfo("../../../etc/passwd")
            zf.writestr(info, "malicious content")

        with pytest.raises(ValueError, match="Path traversal"):
            extract_zip(zip_path)

    def test_rejects_absolute_path(self, tmp_path):
        """Zip with absolute path is rejected."""
        zip_path = tmp_path / "absolute.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("/etc/passwd")
            zf.writestr(info, "malicious content")

        with pytest.raises(ValueError, match="Path traversal"):
            extract_zip(zip_path)

    def test_rejects_unc_absolute_path(self, tmp_path):
        """Zip with UNC-style absolute path (Windows) is rejected."""
        zip_path = tmp_path / "unc.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo(r"\\server\\share\\evil.txt")
            zf.writestr(info, "malicious content")

        with pytest.raises(ValueError, match="Path traversal"):
            extract_zip(zip_path)

    def test_rejects_drive_prefixed_path(self, tmp_path):
        """Zip with drive-prefixed path (Windows) is rejected."""
        zip_path = tmp_path / "drive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo(r"C:\\Windows\\evil.txt")
            zf.writestr(info, "malicious content")

        with pytest.raises(ValueError, match="Path traversal"):
            extract_zip(zip_path)

    def test_rejects_too_many_files(self, tmp_path):
        """Zip with too many files is rejected."""
        zip_path = tmp_path / "many_files.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(MAX_ZIP_FILES + 1):
                zf.writestr(f"file_{i}.txt", f"content {i}")

        with pytest.raises(ValueError, match="too many files"):
            extract_zip(zip_path)

    def test_rejects_oversized_file(self, tmp_path):
        """Zip with oversized single file is rejected."""
        zip_path = tmp_path / "large_file.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Create a file larger than MAX_FILE_BYTES
            large_content = "x" * (MAX_FILE_BYTES + 1)
            zf.writestr("large.txt", large_content)

        with pytest.raises(ValueError, match="File too large"):
            extract_zip(zip_path)

    def test_rejects_oversized_total(self, tmp_path):
        """Zip with total size over limit is rejected."""
        zip_path = tmp_path / "total_large.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Create multiple files that exceed total limit
            chunk_size = MAX_FILE_BYTES - 1000  # Just under single file limit
            num_chunks = (MAX_EXTRACTED_BYTES // chunk_size) + 2
            for i in range(num_chunks):
                zf.writestr(f"chunk_{i}.txt", "x" * chunk_size)

        with pytest.raises(ValueError, match="exceeds limit"):
            extract_zip(zip_path)

    def test_rejects_symlink(self, tmp_path):
        """Zip containing symlink is rejected."""
        zip_path = tmp_path / "symlink.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16  # POSIX symlink
            zf.writestr(info, "target")

        with pytest.raises(ValueError, match="Symlink"):
            extract_zip(zip_path)


class TestExtractZipAllowSymlinks:
    """allow_symlinks=True: 0xA000 entries are materialized as symlinks."""

    def test_symlink_entry_materialized_with_flag(self, tmp_path, require_symlink):
        """Zip symlink entry becomes a real symlink whose target is the entry content."""
        zip_path = tmp_path / "linked.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            info = zipfile.ZipInfo("docs/link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "../SKILL.md")

        result = extract_zip(zip_path, allow_symlinks=True)

        link = result.extracted_path / "docs" / "link"
        assert link.is_symlink()
        assert os.readlink(link) == "../SKILL.md"
        assert link.resolve() == (result.extracted_path / "SKILL.md").resolve()
        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_directory_named_symlink_entry_is_materialized_with_flag(
        self, tmp_path, require_symlink
    ):
        zip_path = tmp_path / "linked-directory-name.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            info = zipfile.ZipInfo("link/")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        result = extract_zip(zip_path, allow_symlinks=True)

        link = result.extracted_path / "link"
        assert link.is_symlink()
        assert not link.is_dir()
        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_hidden_and_excluded_symlink_entries_materialized_with_flag(
        self, tmp_path, require_symlink
    ):
        """Hidden/excluded link entries are materialized so per-skill validation sees them."""
        zip_path = tmp_path / "hidden-links.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            for name in (".hidden-link", "node_modules/link"):
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                zf.writestr(info, "SKILL.md")
            zf.writestr(".hidden-regular", "hidden content")

        result = extract_zip(zip_path, allow_symlinks=True)

        assert (result.extracted_path / ".hidden-link").is_symlink()
        assert (result.extracted_path / "node_modules" / "link").is_symlink()
        assert not (result.extracted_path / ".hidden-regular").exists()
        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_directory_named_symlink_entry_is_rejected_without_flag(self, tmp_path):
        zip_path = tmp_path / "linked-directory-name.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("link/")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(zip_path)

    def test_directory_entry_with_trailing_slash_is_skipped(self, tmp_path):
        zip_path = tmp_path / "directory.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("directory/", "")
            zf.writestr("file.txt", "content")

        result = extract_zip(zip_path)

        assert result.file_count == 1
        assert not (result.extracted_path / "directory").exists()
        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_symlink_creation_failure_is_clear_error(self, tmp_path, monkeypatch):
        """os.symlink failure surfaces as an error naming the symlink problem."""
        zip_path = tmp_path / "linked.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        def _fail_symlink(src, dst):
            raise OSError("simulated permission failure")

        monkeypatch.setattr(os, "symlink", _fail_symlink)

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(zip_path, allow_symlinks=True)

    def test_empty_symlink_target_is_rejected_before_add(
        self, tmp_path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.internal import add_local, detect_skills
        from skillport.modules.skills.internal import manager as manager_module
        from skillport.shared.config import Config

        zip_path = tmp_path / "empty-target.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        result = extract_zip(zip_path, allow_symlinks=True)
        try:
            real_readlink = manager_module.os.readlink

            def readlink(path):
                if any(part.startswith("skillport-snapshot-") for part in Path(path).parts):
                    return ""
                return real_readlink(path)

            monkeypatch.setattr(manager_module.os, "readlink", readlink)
            target = tmp_path / "target"
            results = add_local(
                source_path=result.extracted_path,
                skills=detect_skills(result.extracted_path),
                config=Config(skills_dir=target),
                keep_structure=False,
                force=False,
                allow_symlinks=True,
            )

            assert len(results) == 1
            assert results[0].success is False
            assert not (target / "z").exists()
        finally:
            shutil.rmtree(result.extracted_path, ignore_errors=True)


class TestZipSymlinkPreflightLimits:
    """The hidden/excluded preflight applies the extraction resource limits."""

    @staticmethod
    def _write_zip(zip_path: Path, links: list[tuple[str, str]]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            for name, target in links:
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                zf.writestr(info, target)

    def test_within_limits_returns_hidden_and_excluded_entries(self, tmp_path):
        """Within every limit the preflight returns the violating link entries."""
        zip_path = tmp_path / "links.zip"
        self._write_zip(
            zip_path,
            [
                (".hidden-link", "SKILL.md"),
                ("link-to-hidden", ".env"),
                ("visible", "SKILL.md"),
            ],
        )

        with closing(open_zip_source(zip_path)) as source:
            names = zip_hidden_or_excluded_symlink_entries(source, prefix="")

        assert names == [".hidden-link", "link-to-hidden"]

    def test_entry_count_limit_is_checked_before_reading_payloads(self, tmp_path, monkeypatch):
        """An over-count archive fails before any symlink payload is read."""
        from skillport.modules.skills.internal import zip_handler as zip_handler_module

        zip_path = tmp_path / "many.zip"
        self._write_zip(zip_path, [("link", "SKILL.md"), ("extra", "SKILL.md")])

        monkeypatch.setattr(zip_handler_module, "MAX_ZIP_FILES", 2)
        reads: list[str] = []

        def _record_read(info, zf):
            reads.append(info.filename)
            return "SKILL.md"

        monkeypatch.setattr(zip_handler_module, "_zip_symlink_target", _record_read)

        with closing(open_zip_source(zip_path)) as source:
            with pytest.raises(ValueError, match="too many files"):
                zip_hidden_or_excluded_symlink_entries(source, prefix="")

        assert reads == []

    def test_oversized_single_payload_is_rejected_before_reading(self, tmp_path, monkeypatch):
        """A payload over the per-entry limit fails before the payload is read."""
        from skillport.modules.skills.internal import zip_handler as zip_handler_module

        zip_path = tmp_path / "oversized.zip"
        self._write_zip(zip_path, [("link", "SKILL.md")])

        monkeypatch.setattr(zip_handler_module, "MAX_FILE_BYTES", 4)
        reads: list[str] = []

        def _record_read(info, zf):
            reads.append(info.filename)
            return "SKILL.md"

        monkeypatch.setattr(zip_handler_module, "_zip_symlink_target", _record_read)

        with closing(open_zip_source(zip_path)) as source:
            with pytest.raises(ValueError, match="File too large"):
                zip_hidden_or_excluded_symlink_entries(source, prefix="")

        assert reads == []

    def test_cumulative_payload_limit_stops_before_the_next_read(self, tmp_path, monkeypatch):
        """The payload that would cross the cumulative limit is not read."""
        from skillport.modules.skills.internal import zip_handler as zip_handler_module

        zip_path = tmp_path / "cumulative.zip"
        self._write_zip(zip_path, [("link-a", "SKILL.md"), ("link-b", "SKILL.md")])

        monkeypatch.setattr(zip_handler_module, "MAX_EXTRACTED_BYTES", 10)
        reads: list[str] = []

        def _record_read(info, zf):
            reads.append(info.filename)
            return "SKILL.md"

        monkeypatch.setattr(zip_handler_module, "_zip_symlink_target", _record_read)

        with closing(open_zip_source(zip_path)) as source:
            with pytest.raises(ValueError, match="exceeds limit"):
                zip_hidden_or_excluded_symlink_entries(source, prefix="")

        assert reads == ["link-a"]


class TestExtractZipDuplicateEntries:
    """Duplicate normalized paths and symlinked destinations cannot redirect writes."""

    def test_symlink_then_duplicate_regular_entry_rejected(self, tmp_path):
        """A regular entry collapsing onto a materialized symlink is rejected, not written through."""
        outside = tmp_path / "outside-target.txt"
        outside.write_text("original-content", encoding="utf-8")

        zip_path = tmp_path / "dup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, str(outside))
            zf.writestr("./link", "attacker-content")

        with pytest.raises(ValueError, match="(?i)duplicate zip entry"):
            extract_zip(zip_path, allow_symlinks=True)

        # The external target was not overwritten through the symlink
        assert outside.read_text(encoding="utf-8") == "original-content"

    def test_regular_then_duplicate_symlink_entry_rejected(self, tmp_path):
        """A symlink entry collapsing onto an already-written path is rejected."""
        zip_path = tmp_path / "dup2.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            zf.writestr("link", "regular content")
            info = zipfile.ZipInfo("./link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        with pytest.raises(ValueError, match="(?i)duplicate zip entry|symlink"):
            extract_zip(zip_path, allow_symlinks=True)

    def test_regular_entry_under_symlink_directory_rejected(self, tmp_path):
        """A file entry whose parent directory is a materialized symlink is rejected."""
        outside_dir = tmp_path / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "existing.txt").write_text("original", encoding="utf-8")

        zip_path = tmp_path / "parent-link.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")
            info = zipfile.ZipInfo("docs")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, str(outside_dir))
            zf.writestr("docs/new.txt", "attacker-content")

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(zip_path, allow_symlinks=True)

        # Nothing was written through the symlinked directory
        assert sorted(p.name for p in outside_dir.iterdir()) == ["existing.txt"]
        assert (outside_dir / "existing.txt").read_text(encoding="utf-8") == "original"


class TestZipSourceSymlinkBoundary:
    """The zip source path itself (final component or ancestors) must not be a symlink."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_symlinked_zip_file_is_rejected(self, tmp_path, allow: bool, require_symlink):
        """A zip path whose final component is a symlink is rejected before reading."""
        outside = tmp_path / "outside" / "valid.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")

        alias = tmp_path / "alias.zip"
        alias.symlink_to(outside)

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(alias, allow_symlinks=allow)

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_symlinked_ancestor_directory_is_rejected(self, tmp_path, allow: bool, require_symlink):
        """A zip path under a symlinked ancestor directory is rejected before reading."""
        outside = tmp_path / "outside" / "nested" / "skill.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")

        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(outside.parent, target_is_directory=True)

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(source / "alias" / "skill.zip", allow_symlinks=allow)

    def test_dangling_symlink_zip_is_rejected(self, tmp_path, require_symlink):
        """A symlink final component is rejected even when its target is missing."""
        alias = tmp_path / "dangling.zip"
        alias.symlink_to(tmp_path / "missing.zip")

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(alias)

    def test_regular_zip_path_still_extracts(self, tmp_path):
        """A regular zip path with no symlinked component extracts as before."""
        zip_path = tmp_path / "plain.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: z\ndescription: Z\n---\ncontent")

        result = extract_zip(zip_path)

        assert (result.extracted_path / "SKILL.md").exists()
        shutil.rmtree(result.extracted_path, ignore_errors=True)


class TestExtractZipEdgeCases:
    """Edge case tests for extract_zip function."""

    def test_empty_zip(self, tmp_path):
        """Empty zip is handled gracefully."""
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass  # Create empty zip

        result = extract_zip(zip_path)

        assert result.extracted_path.exists()
        assert result.file_count == 0

        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_zip_without_skill_md(self, tmp_path):
        """Zip without SKILL.md is extracted (detect_skills handles validation)."""
        zip_path = tmp_path / "no_skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("README.md", "# README")
            zf.writestr("data.json", "{}")

        result = extract_zip(zip_path)

        assert result.extracted_path.exists()
        assert result.file_count == 2
        assert not (result.extracted_path / "SKILL.md").exists()

        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_hidden_files_skipped(self, tmp_path):
        """Hidden files (.gitignore, etc.) are skipped."""
        zip_path = tmp_path / "hidden.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: test\n---\n")
            zf.writestr(".gitignore", "*.pyc")
            zf.writestr(".env", "SECRET=xxx")
            zf.writestr("normal.txt", "normal")

        result = extract_zip(zip_path)

        # Only non-hidden files should be extracted
        assert result.file_count == 2
        assert (result.extracted_path / "SKILL.md").exists()
        assert (result.extracted_path / "normal.txt").exists()
        assert not (result.extracted_path / ".gitignore").exists()
        assert not (result.extracted_path / ".env").exists()

        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_excluded_directories_skipped(self, tmp_path):
        """Excluded directories (__pycache__, .git) are skipped."""
        zip_path = tmp_path / "excluded.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: test\n---\n")
            zf.writestr("__pycache__/module.pyc", "bytecode")
            zf.writestr(".git/config", "git config")
            zf.writestr("src/main.py", "# main")

        result = extract_zip(zip_path)

        assert (result.extracted_path / "SKILL.md").exists()
        assert (result.extracted_path / "src" / "main.py").exists()
        assert not (result.extracted_path / "__pycache__").exists()
        assert not (result.extracted_path / ".git").exists()

        import shutil

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_nonexistent_zip_raises(self, tmp_path):
        """Non-existent zip file raises FileNotFoundError."""
        zip_path = tmp_path / "nonexistent.zip"

        with pytest.raises(FileNotFoundError):
            extract_zip(zip_path)

    def test_invalid_zip_raises(self, tmp_path):
        """Invalid zip file raises ValueError."""
        invalid_path = tmp_path / "not_a_zip.zip"
        invalid_path.write_text("this is not a zip file")

        with pytest.raises(ValueError, match="Not a valid zip"):
            extract_zip(invalid_path)


class TestZipSourceBoundarySwap:
    """A zip path swapped for a symlink at the no-follow open is rejected."""

    def test_ancestor_swapped_at_open_is_rejected(self, tmp_path, monkeypatch, require_symlink):
        from skillport.modules.skills.internal import zip_handler as zip_handler_module

        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "skill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: E\n---\nexternal")

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        with zipfile.ZipFile(source / "alias" / "skill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: O\n---\noriginal")

        real_open = zip_handler_module.open_no_follow
        state = {"swapped": False}

        def swapping_open(path, *, directory=None):
            if not state["swapped"]:
                state["swapped"] = True
                (source / "alias").rename(source / "alias-original")
                (source / "alias").symlink_to(outside, target_is_directory=True)
            return real_open(path, directory=directory)

        monkeypatch.setattr(zip_handler_module, "open_no_follow", swapping_open)

        with pytest.raises(ValueError, match="(?i)symlink"):
            extract_zip(source / "alias" / "skill.zip")

        assert sorted(p.name for p in outside.iterdir()) == ["skill.zip"]


class TestNoFollowUnsupportedPlatform:
    """ZIP sources still open and reject symlinked paths without O_NOFOLLOW/dir_fd."""

    def test_regular_zip_extracts(self, tmp_path, no_follow_unavailable):
        zip_path = tmp_path / "skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: D\n---\nbody")

        result = extract_zip(zip_path)

        assert (result.extracted_path / "SKILL.md").read_text(encoding="utf-8").endswith("body")
        assert result.source_mtime_ns == zip_path.stat().st_mtime_ns

        shutil.rmtree(result.extracted_path, ignore_errors=True)

    def test_symlinked_zip_path_is_rejected(self, tmp_path, no_follow_unavailable, require_symlink):
        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "skill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: E\n---\nexternal")
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")

        alias = tmp_path / "alias.zip"
        alias.symlink_to(outside / "skill.zip")

        with pytest.raises(ValueError, match="traverses a symlink"):
            extract_zip(alias)

        assert sorted(p.name for p in outside.iterdir()) == ["external-marker.txt", "skill.zip"]

    def test_ancestor_swapped_at_open_is_rejected(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        from skillport.modules.skills.internal import zip_handler as zip_handler_module

        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "skill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: E\n---\nexternal")

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        with zipfile.ZipFile(source / "alias" / "skill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: O\n---\noriginal")

        real_open = zip_handler_module.open_no_follow
        state = {"swapped": False}

        def swapping_open(path, *, directory=None):
            if not state["swapped"]:
                state["swapped"] = True
                (source / "alias").rename(source / "alias-original")
                (source / "alias").symlink_to(outside, target_is_directory=True)
            return real_open(path, directory=directory)

        monkeypatch.setattr(zip_handler_module, "open_no_follow", swapping_open)

        with pytest.raises(ValueError, match="traverses a symlink"):
            extract_zip(source / "alias" / "skill.zip")

        assert sorted(p.name for p in outside.iterdir()) == ["skill.zip"]

    def test_archive_swapped_during_open_is_rejected(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        """A swap between the component check and the archive open is rejected."""
        from skillport.shared import utils as utils_module

        outside = tmp_path / "outside"
        outside.mkdir()
        external_zip = outside / "skill.zip"
        with zipfile.ZipFile(external_zip, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: E\n---\nexternal")
        external_zip_before = external_zip.read_bytes()

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        archive = source / "alias" / "skill.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: O\n---\noriginal")

        real_open = utils_module._open_descriptor
        state = {"swapped": False}
        displaced = archive.parent / "skill-original.zip"

        def swapping_open(path, flags):
            if path == archive and not state["swapped"]:
                state["swapped"] = True
                archive.rename(displaced)
                archive.symlink_to(external_zip)
                try:
                    return real_open(path, flags)
                finally:
                    archive.unlink()
                    displaced.rename(archive)
            return real_open(path, flags)

        monkeypatch.setattr(utils_module, "_open_descriptor", swapping_open)

        with pytest.raises(ValueError, match="(?i)changed|symlink"):
            extract_zip(archive)

        assert external_zip.read_bytes() == external_zip_before
        assert sorted(p.name for p in outside.iterdir()) == ["skill.zip"]
