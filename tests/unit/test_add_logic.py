"""Unit tests for add command logic (SPEC2-CLI Section 3.3)."""

import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path

import pytest

from skillport.modules.skills.internal.manager import (
    BUILTIN_SKILLS,
    _validate_skill_file,
    add_builtin,
    add_local,
    detect_skills,
    snapshot_skill_dir,
)
from skillport.shared.config import Config


def _symlink_capable() -> bool:
    """Probe whether this platform can create symlinks (skip policy of order.md)."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink("target", Path(tmp) / "probe")
    except (OSError, NotImplementedError):
        return False
    return True


@pytest.fixture
def require_symlink():
    """Skip the target test (instead of failing setup) when symlinks cannot be created."""
    if not _symlink_capable():
        pytest.skip("symlinks cannot be created on this platform")


class TestSymlinkCapabilityProbe:
    """The capability probe itself."""

    def test_probe_reports_unavailable_when_os_symlink_fails(self, monkeypatch):
        """An OSError from os.symlink makes the probe report unavailable (skip, not fail)."""

        def _fail(src, dst, **kwargs):
            raise OSError("simulated symlink permission failure")

        monkeypatch.setattr(os, "symlink", _fail)
        assert _symlink_capable() is False


def _create_skill(path: Path, name: str, description: str = "Test description") -> Path:
    """Helper to create a valid skill directory."""
    skill_dir = path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody content", encoding="utf-8"
    )
    return skill_dir


def _create_skill_with_frontmatter(
    path: Path, name: str, frontmatter: str, *, body: str = "Body content"
) -> Path:
    """Helper to create a skill directory with extra top-level frontmatter keys."""
    skill_dir = path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test description\n{frontmatter}\n---\n{body}",
        encoding="utf-8",
    )
    return skill_dir


class TestDetectSkills:
    """Skill detection tests."""

    def test_single_skill_at_root(self, tmp_path: Path):
        """Single SKILL.md at root → 1 skill."""
        (tmp_path / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: Root skill\n---\nbody", encoding="utf-8"
        )
        skills = detect_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "root-skill"

    def test_multiple_skills_in_children(self, tmp_path: Path):
        """Multiple child dirs with SKILL.md → N skills."""
        _create_skill(tmp_path, "skill-a")
        _create_skill(tmp_path, "skill-b")
        _create_skill(tmp_path, "skill-c")

        skills = detect_skills(tmp_path)
        assert len(skills) == 3
        names = {s.name for s in skills}
        assert names == {"skill-a", "skill-b", "skill-c"}

    def test_no_skill_md_returns_empty(self, tmp_path: Path):
        """No SKILL.md → empty list."""
        (tmp_path / "some-file.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        skills = detect_skills(tmp_path)
        assert len(skills) == 0

    def test_mixed_dirs_only_detects_skills(self, tmp_path: Path):
        """Only dirs with SKILL.md are detected."""
        _create_skill(tmp_path, "valid-skill")
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "not-a-skill" / "README.md").write_text("readme", encoding="utf-8")

        skills = detect_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "valid-skill"

    def test_detects_skill_name_from_frontmatter(self, tmp_path: Path):
        """Skill name comes from frontmatter, not dir name."""
        skill_dir = tmp_path / "dir-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: frontmatter-name\ndescription: desc\n---\nbody", encoding="utf-8"
        )
        skills = detect_skills(tmp_path)
        assert len(skills) == 1
        # Note: name comes from frontmatter
        assert skills[0].name == "frontmatter-name"

    def test_missing_required_frontmatter_is_fatal(self, tmp_path: Path):
        """Missing name/description should raise at validation time."""
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: \n---\nbody", encoding="utf-8")
        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(tmp_path)

        with pytest.raises(ValueError):
            _validate_skill_file(skill_dir)

        results = add_local(
            source_path=tmp_path,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )
        assert len(results) == 1
        assert not results[0].success
        assert "Invalid SKILL.md" in results[0].message

    def test_source_not_found_raises(self, tmp_path: Path):
        """Non-existent path → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_skills(tmp_path / "nonexistent")

    def test_source_is_file_raises(self, tmp_path: Path):
        """File path → ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match="directory"):
            detect_skills(file_path)

    def test_skill_md_symlink_is_validated_before_frontmatter_read(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.internal import manager as manager_module

        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        docs = skill / "docs"
        docs.mkdir()
        (docs / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Valid\n---\nbody", encoding="utf-8"
        )
        (skill / "SKILL.md").unlink()
        (skill / "SKILL.md").symlink_to(Path("docs") / "SKILL.md")

        calls: list[Path] = []
        original_parse_frontmatter = manager_module.parse_frontmatter

        def parse_frontmatter(path: Path):
            calls.append(path)
            return original_parse_frontmatter(path)

        monkeypatch.setattr(manager_module, "parse_frontmatter", parse_frontmatter)

        skills = detect_skills(source, allow_symlinks=True)

        assert len(skills) == 1
        assert skills[0].error is None
        assert calls == [skill / "SKILL.md"]

    def test_invalid_skill_md_symlink_is_rejected_without_reading_target(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.internal import manager as manager_module

        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        outside = tmp_path / "outside" / "SKILL.md"
        outside.parent.mkdir()
        outside.write_text("---\nname: skill-a\ndescription: External\n---\nbody", encoding="utf-8")
        (skill / "SKILL.md").unlink()
        (skill / "SKILL.md").symlink_to(outside)

        calls: list[Path] = []

        def parse_frontmatter(path: Path):
            calls.append(path)
            raise AssertionError("frontmatter must not be read")

        monkeypatch.setattr(manager_module, "parse_frontmatter", parse_frontmatter)

        skills = detect_skills(source, allow_symlinks=True)

        assert len(skills) == 1
        assert skills[0].error is not None
        assert "outside" in skills[0].error.lower()
        assert calls == []


class TestAddLocalNamespace:
    """Namespace handling in add_local."""

    def test_keep_structure_true_uses_namespace(self, tmp_path: Path):
        """keep_structure=True → skills/<namespace>/<skill>/"""
        source = tmp_path / "source"
        _create_skill(source, "skill-a")
        _create_skill(source, "skill-b")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=True,
            force=False,
        )

        assert all(r.success for r in results)
        # Default namespace is source dir name
        assert (target / "source" / "skill-a" / "SKILL.md").exists()
        assert (target / "source" / "skill-b" / "SKILL.md").exists()

    def test_keep_structure_false_flattens(self, tmp_path: Path):
        """keep_structure=False → skills/<skill>/"""
        source = tmp_path / "source"
        _create_skill(source, "skill-a")
        _create_skill(source, "skill-b")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )

        assert all(r.success for r in results)
        assert (target / "skill-a" / "SKILL.md").exists()
        assert (target / "skill-b" / "SKILL.md").exists()
        # No namespace directory
        assert not (target / "source").exists()

    def test_custom_namespace_override(self, tmp_path: Path):
        """namespace_override → uses custom namespace."""
        source = tmp_path / "source"
        _create_skill(source, "skill-a")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=True,
            force=False,
            namespace_override="my-team",
        )

        assert all(r.success for r in results)
        assert results[0].skill_id == "my-team/skill-a"
        assert (target / "my-team" / "skill-a" / "SKILL.md").exists()


class TestAddLocalOverwrite:
    """Overwrite behavior in add_local."""

    def test_existing_skill_without_force_skipped(self, tmp_path: Path):
        """Existing skill without --force → skipped."""
        source = tmp_path / "source"
        _create_skill(source, "skill-a")

        target = tmp_path / "target"
        # Pre-create existing skill
        _create_skill(target, "skill-a")
        original_content = (target / "skill-a" / "SKILL.md").read_text()

        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert not results[0].success
        assert "exists" in results[0].message.lower()
        # Content unchanged
        assert (target / "skill-a" / "SKILL.md").read_text() == original_content

    def test_existing_skill_with_force_overwritten(self, tmp_path: Path):
        """Existing skill with --force → overwritten."""
        source = tmp_path / "source"
        source_skill = _create_skill(source, "skill-a")
        new_content = "---\nname: skill-a\ndescription: Updated\n---\nNew body"
        (source_skill / "SKILL.md").write_text(new_content, encoding="utf-8")

        target = tmp_path / "target"
        _create_skill(target, "skill-a")

        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=True,
        )

        assert len(results) == 1
        assert results[0].success
        # Content updated
        assert "Updated" in (target / "skill-a" / "SKILL.md").read_text()

    def test_mixed_new_and_existing(self, tmp_path: Path):
        """Some new, some existing → partial success."""
        source = tmp_path / "source"
        _create_skill(source, "skill-a")
        _create_skill(source, "skill-b")
        _create_skill(source, "skill-c")

        target = tmp_path / "target"
        _create_skill(target, "skill-a")  # exists

        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )

        success = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(success) == 2  # skill-b, skill-c
        assert len(failed) == 1  # skill-a
        assert failed[0].skill_id == "skill-a"


class TestAddBuiltin:
    """Built-in skill add tests."""

    @pytest.mark.parametrize("builtin_name", list(BUILTIN_SKILLS.keys()))
    def test_add_builtin_skills(self, tmp_path: Path, builtin_name: str):
        """All built-in skills can be added."""
        cfg = Config(skills_dir=tmp_path)
        result = add_builtin(builtin_name, config=cfg, force=False)

        assert result.success
        assert (tmp_path / builtin_name / "SKILL.md").exists()

    def test_add_unknown_builtin_raises(self, tmp_path: Path):
        """Unknown built-in name → ValueError."""
        cfg = Config(skills_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown"):
            add_builtin("nonexistent-builtin", config=cfg, force=False)

    def test_builtin_exists_without_force_fails(self, tmp_path: Path):
        """Existing built-in without --force → fails."""
        cfg = Config(skills_dir=tmp_path)

        # Add first time
        result1 = add_builtin("hello-world", config=cfg, force=False)
        assert result1.success

        # Add again without force
        result2 = add_builtin("hello-world", config=cfg, force=False)
        assert not result2.success
        assert "exists" in result2.message.lower()

    def test_builtin_exists_with_force_overwrites(self, tmp_path: Path):
        """Existing built-in with --force → overwrites."""
        cfg = Config(skills_dir=tmp_path)

        # Add first time
        add_builtin("hello-world", config=cfg, force=False)

        # Modify the file
        (tmp_path / "hello-world" / "SKILL.md").write_text("modified", encoding="utf-8")

        # Add again with force
        result = add_builtin("hello-world", config=cfg, force=True)
        assert result.success
        # Content restored to original
        content = (tmp_path / "hello-world" / "SKILL.md").read_text()
        assert "Hello World Skill" in content


class TestSkillRenameSingle:
    """Single skill rename with --name option."""

    def test_rename_single_skill(self, tmp_path: Path):
        """--name renames single skill."""
        source = tmp_path / "source"
        _create_skill(source, "original-name")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
            rename_single_to="new-name",
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].skill_id == "new-name"
        assert (target / "new-name" / "SKILL.md").exists()
        # frontmatter.name should be updated
        content = (target / "new-name" / "SKILL.md").read_text()
        assert "name: new-name" in content

    def test_rename_ignored_for_multiple(self, tmp_path: Path):
        """--name ignored when multiple skills."""
        source = tmp_path / "source"
        _create_skill(source, "skill-a")
        _create_skill(source, "skill-b")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
            rename_single_to="ignored-name",
        )

        # Names not changed because len(skills) > 1
        skill_ids = {r.skill_id for r in results}
        assert skill_ids == {"skill-a", "skill-b"}


class TestSymlinkRejection:
    """Symlink security tests."""

    def test_symlink_in_skill_rejected(self, tmp_path: Path, require_symlink):
        """Symlinks in skill directory → rejected."""
        source = tmp_path / "source"
        skill_dir = _create_skill(source, "skill-with-link")

        # Create a symlink
        link_path = skill_dir / "link.txt"
        target_file = tmp_path / "outside.txt"
        target_file.write_text("secret", encoding="utf-8")
        link_path.symlink_to(target_file)

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )

        # Should fail due to symlink
        assert len(results) == 1
        assert not results[0].success
        assert "symlink" in results[0].message.lower()

    def test_hidden_symlink_is_rejected_before_snapshot(self, tmp_path: Path, require_symlink):
        source = tmp_path / "source"
        skill_dir = _create_skill(source, "skill-with-hidden-link")
        (skill_dir / "target.txt").write_text("target", encoding="utf-8")
        (skill_dir / ".hidden-link").symlink_to("target.txt")

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "symlink" in results[0].message.lower()
        assert not (target / "skill-with-hidden-link").exists()


class TestExcludedNameSymlinkRejection:
    """A symlink named like an EXCLUDE_NAMES entry still reaches the symlink gate.

    The local source snapshot materializes excluded names before per-skill
    validation, so a flagless add of ``.env -> SKILL.md`` fails instead of
    silently dropping the link; a regular file with the same name stays ignored.
    """

    def test_public_add_rejects_excluded_name_symlink_without_flag(
        self, tmp_path: Path, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        _create_skill(tmp_path / "src", "skill-a")
        source = tmp_path / "src" / "skill-a"
        (source / ".env").symlink_to("SKILL.md")

        target = tmp_path / "installed"
        cfg = Config(skills_dir=target, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, keep_structure=False)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert not (target / "skill-a").exists()
        assert get_origin("skill-a", config=cfg) is None

    def test_public_add_ignores_excluded_name_regular_file(self, tmp_path: Path):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        _create_skill(tmp_path / "src", "skill-a")
        source = tmp_path / "src" / "skill-a"
        (source / ".env").write_text("ignored", encoding="utf-8")

        target = tmp_path / "installed"
        cfg = Config(skills_dir=target, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, keep_structure=False)

        assert result.success, result.message
        assert (target / "skill-a" / "SKILL.md").exists()
        assert not (target / "skill-a" / ".env").exists()
        assert get_origin("skill-a", config=cfg) is not None


class TestAllowSymlinksLocalPreserve:
    """allow_symlinks=True: compliant relative symlinks are preserved as links."""

    def test_compliant_symlink_chain_preserved(self, tmp_path: Path, require_symlink):
        """Relative in-skill link chain (2 hops) is copied preserving links."""
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        assets = skill / "assets"
        assets.mkdir()
        (assets / "manual.md").write_text("manual content", encoding="utf-8")
        (assets / "local.md").symlink_to("manual.md")
        docs = skill / "docs"
        docs.mkdir()
        (docs / "current.md").symlink_to(Path("../assets") / "local.md")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert len(results) == 1
        assert results[0].success, results[0].message
        dest = target / "skill-a"
        assert (dest / "assets" / "local.md").is_symlink()
        current = dest / "docs" / "current.md"
        assert current.is_symlink()
        assert os.readlink(current) == "../assets/local.md"
        assert current.resolve() == (dest / "assets" / "manual.md").resolve()
        assert current.read_text(encoding="utf-8") == "manual content"

    def test_skill_md_symlink_is_preserved(self, tmp_path: Path, require_symlink):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        docs = skill / "docs"
        docs.mkdir()
        skill_md = skill / "SKILL.md"
        docs_skill_md = docs / "SKILL.md"
        skill_md.replace(docs_skill_md)
        skill_md.symlink_to(Path("docs") / "SKILL.md")

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source, allow_symlinks=True),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert results[0].success, results[0].message
        installed_skill_md = target / "skill-a" / "SKILL.md"
        assert installed_skill_md.is_symlink()
        assert os.readlink(installed_skill_md) == "docs/SKILL.md"
        assert installed_skill_md.read_text(encoding="utf-8").endswith("Body content")

    def test_directory_final_target_is_rejected(self, tmp_path: Path, require_symlink):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        (skill / "assets").mkdir()
        (skill / "link").symlink_to("assets", target_is_directory=True)

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "regular file" in results[0].message.lower()
        assert not (target / "skill-a").exists()

    @pytest.mark.parametrize(
        "target", ["C:relative.txt", "C:/absolute.txt", r"\\server\share\file.txt"]
    )
    def test_windows_style_target_is_rejected_on_posix(
        self, tmp_path: Path, target: str, require_symlink
    ):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        (skill / "link.txt").symlink_to(target)
        # A regular file with the same name exists, so only the Windows-style
        # classification can reject the link (not a missing target).
        if os.name == "posix":
            same_named = skill / target
            same_named.parent.mkdir(parents=True, exist_ok=True)
            same_named.write_text("same-named regular file", encoding="utf-8")

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=tmp_path / "target"),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert results[0].success is False
        assert "relative" in results[0].message.lower()


class TestSnapshotMetadata:
    def test_regular_file_metadata_is_preserved_in_snapshot_and_install(self, tmp_path: Path):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        data = skill / "data.txt"
        data.write_text("data", encoding="utf-8")
        data.chmod(0o640)
        fixed_mtime = 1_600_000_000_123_456_789
        os.utime(data, ns=(fixed_mtime, fixed_mtime))

        snapshot = snapshot_skill_dir(skill)
        try:
            snapshot_data = snapshot / "data.txt"
            assert stat.S_IMODE(snapshot_data.stat().st_mode) == 0o640
            assert snapshot_data.stat().st_mtime_ns == fixed_mtime
        finally:
            import shutil

            shutil.rmtree(snapshot.parent, ignore_errors=True)

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
        )

        assert results[0].success, results[0].message
        installed_data = target / "skill-a" / "data.txt"
        assert stat.S_IMODE(installed_data.stat().st_mode) == 0o640
        assert installed_data.stat().st_mtime_ns == fixed_mtime

    def test_renamed_skill_md_preserves_metadata(self, tmp_path: Path):
        source = tmp_path / "source"
        skill = _create_skill(source, "original")
        skill_md = skill / "SKILL.md"
        skill_md.chmod(0o640)
        fixed_mtime = 1_600_000_000_987_654_321
        os.utime(skill_md, ns=(fixed_mtime, fixed_mtime))

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            rename_single_to="renamed",
        )

        assert results[0].success, results[0].message
        installed_skill_md = target / "renamed" / "SKILL.md"
        assert stat.S_IMODE(installed_skill_md.stat().st_mode) == 0o640
        assert installed_skill_md.stat().st_mtime_ns == fixed_mtime


class TestEmptySymlinkTarget:
    @pytest.mark.parametrize("allow_symlinks", [False, True], ids=["flagless", "with-flag"])
    def test_empty_target_is_rejected(
        self, tmp_path: Path, allow_symlinks: bool, monkeypatch, require_symlink
    ):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        (skill / "target.txt").write_text("target", encoding="utf-8")
        os.symlink("target.txt", skill / "empty.link")

        if allow_symlinks:
            from skillport.modules.skills.internal import manager as manager_module

            real_readlink = manager_module.os.readlink

            def readlink(path):
                if any(part.startswith("skillport-snapshot-") for part in Path(path).parts):
                    return ""
                return real_readlink(path)

            monkeypatch.setattr(manager_module.os, "readlink", readlink)

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=allow_symlinks,
        )

        assert len(results) == 1
        assert results[0].success is False
        assert not (target / "skill-a").exists()


def _create_violating_link(kind: str, skill_dir: Path, source_root: Path) -> list[str]:
    """Create a violating symlink inside skill_dir. Returns created link file names."""
    link = skill_dir / "bad.link"
    if kind == "absolute":
        outside = source_root.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link.symlink_to(outside)
    elif kind == "cycle":
        link.symlink_to("other.link")
        (skill_dir / "other.link").symlink_to("bad.link")
        return ["bad.link", "other.link"]
    elif kind == "outside-skill":
        link.symlink_to(Path("../skill-ok") / "data.txt")
    elif kind == "hidden":
        (skill_dir / ".secret.txt").write_text("secret", encoding="utf-8")
        link.symlink_to(".secret.txt")
    elif kind == "excluded-name":
        pkg_dir = skill_dir / "node_modules"
        pkg_dir.mkdir()
        (pkg_dir / "pkg.txt").write_text("pkg", encoding="utf-8")
        link.symlink_to(Path("node_modules") / "pkg.txt")
    elif kind == "missing-target":
        link.symlink_to("does-not-exist.txt")
    elif kind == "trailing-separator":
        (skill_dir / "target.txt").write_text("target", encoding="utf-8")
        link.symlink_to("target.txt/")
    else:
        raise ValueError(f"unknown violation kind: {kind}")
    return ["bad.link"]


class TestAllowSymlinksViolations:
    """allow_symlinks=True: violating links fail only the skill that contains them."""

    @pytest.mark.parametrize(
        "violation",
        [
            "absolute",
            "cycle",
            "outside-skill",
            "hidden",
            "excluded-name",
            "missing-target",
            "trailing-separator",
        ],
    )
    def test_violating_link_fails_only_that_skill(
        self, tmp_path: Path, violation: str, require_symlink
    ):
        """Multi-skill source: violating skill fails, the other skill is added."""
        source = tmp_path / "source"
        ok_skill = _create_skill(source, "skill-ok")
        (ok_skill / "data.txt").write_text("ok data", encoding="utf-8")
        bad_skill = _create_skill(source, "skill-bad")
        link_names = _create_violating_link(violation, bad_skill, source)

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )
        by_id = {r.skill_id: r for r in results}

        assert not by_id["skill-bad"].success
        assert any(name in by_id["skill-bad"].message for name in link_names)
        assert not (target / "skill-bad").exists()

        assert by_id["skill-ok"].success, by_id["skill-ok"].message
        assert (target / "skill-ok" / "SKILL.md").exists()
        assert (target / "skill-ok" / "data.txt").read_text(encoding="utf-8") == "ok data"


class TestAllowSymlinksTargetDirectorySuffix:
    """allow_symlinks=True: a target string that names a directory is rejected.

    Path normalization drops a trailing separator or a final ``.`` component,
    so the raw target string is inspected: ``target.txt/`` and ``target.txt/.``
    name a directory instead of the regular file the link policy requires, and
    installing them would leave a broken link.
    """

    def test_regular_file_target_is_preserved_as_link(self, tmp_path: Path, require_symlink):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        (skill / "target.txt").write_text("target", encoding="utf-8")
        (skill / "link").symlink_to("target.txt")

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert len(results) == 1
        assert results[0].success, results[0].message
        installed = target / "skill-a" / "link"
        assert installed.is_symlink()
        assert os.readlink(installed) == "target.txt"

    @pytest.mark.parametrize("target_string", ["target.txt/", "target.txt/.", "target.txt//"])
    def test_directory_suffix_target_fails_only_that_skill(
        self, tmp_path: Path, target_string: str, require_symlink
    ):
        source = tmp_path / "source"
        ok_skill = _create_skill(source, "skill-ok")
        (ok_skill / "data.txt").write_text("ok data", encoding="utf-8")
        bad_skill = _create_skill(source, "skill-bad")
        (bad_skill / "target.txt").write_text("target", encoding="utf-8")
        (bad_skill / "link").symlink_to(target_string)

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )
        by_id = {r.skill_id: r for r in results}

        assert by_id["skill-bad"].success is False
        assert "link" in by_id["skill-bad"].message
        assert not (target / "skill-bad").exists()
        assert by_id["skill-ok"].success, by_id["skill-ok"].message
        assert (target / "skill-ok" / "data.txt").read_text(encoding="utf-8") == "ok data"


class TestAllowSymlinksHiddenPathViolations:
    """allow_symlinks=True: a link whose own path is hidden or excluded fails the skill.

    The link is classified before the hidden/EXCLUDE_NAMES filter, so the
    violation is reported instead of the link being dropped silently.
    """

    @pytest.mark.parametrize(
        "link_path",
        [".hidden-link", ".env", "docs/.hidden-link", ".hidden/link", "node_modules/link"],
        ids=[
            "hidden-name",
            "excluded-name",
            "hidden-in-visible-dir",
            "hidden-parent",
            "excluded-parent",
        ],
    )
    def test_hidden_or_excluded_link_path_fails_only_that_skill(
        self, tmp_path: Path, link_path: str, require_symlink
    ):
        source = tmp_path / "source"
        ok_skill = _create_skill(source, "skill-ok")
        (ok_skill / "data.txt").write_text("ok data", encoding="utf-8")
        bad_skill = _create_skill(source, "skill-bad")
        (bad_skill / "target.txt").write_text("target", encoding="utf-8")
        link = bad_skill / link_path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(bad_skill / "target.txt", link.parent))

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )
        by_id = {r.skill_id: r for r in results}

        assert by_id["skill-bad"].success is False
        assert "hidden or excluded" in by_id["skill-bad"].message.lower()
        assert link_path in by_id["skill-bad"].message
        assert not (target / "skill-bad").exists()

        assert by_id["skill-ok"].success, by_id["skill-ok"].message
        assert (target / "skill-ok" / "data.txt").read_text(encoding="utf-8") == "ok data"

    def test_public_add_with_flag_rejects_hidden_link_without_origin(
        self, tmp_path: Path, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        _create_skill(tmp_path / "src", "skill-a")
        source = tmp_path / "src" / "skill-a"
        (source / "target.txt").write_text("target", encoding="utf-8")
        (source / ".hidden-link").symlink_to("target.txt")

        target = tmp_path / "installed"
        cfg = Config(skills_dir=target, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            str(source), config=cfg, force=False, keep_structure=False, allow_symlinks=True
        )

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert not (target / "skill-a").exists()
        assert get_origin("skill-a", config=cfg) is None

    def test_compliant_link_preserved_while_excluded_regular_file_stays_ignored(
        self, tmp_path: Path, require_symlink
    ):
        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        (skill / "assets").mkdir()
        (skill / "assets" / "manual.md").write_text("manual content", encoding="utf-8")
        (skill / "docs").mkdir()
        (skill / "docs" / "current.md").symlink_to(Path("../assets") / "manual.md")
        (skill / ".env").write_text("ignored", encoding="utf-8")

        target = tmp_path / "target"
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert results[0].success, results[0].message
        installed = target / "skill-a"
        current = installed / "docs" / "current.md"
        assert current.is_symlink()
        assert os.readlink(current) == "../assets/manual.md"
        assert current.read_text(encoding="utf-8") == "manual content"
        assert not (installed / ".env").exists()


class TestSkillRootSymlinkBoundary:
    """A symlinked individual skill root is rejected without reading through it."""

    def _make_collection_with_root_symlink(self, tmp_path: Path) -> Path:
        source = tmp_path / "source"
        _create_skill(source, "skill-a")
        outside = tmp_path / "outside" / "skill-b"
        outside.mkdir(parents=True)
        (outside / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: External\n---\nexternal body", encoding="utf-8"
        )
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")
        (source / "skill-b").symlink_to(outside)
        return source

    def test_detect_skills_reports_root_symlink_as_error(self, tmp_path: Path, require_symlink):
        """A symlinked skill child is reported as an unusable skill, not loaded."""
        source = self._make_collection_with_root_symlink(tmp_path)

        skills = detect_skills(source)
        by_name = {s.name: s for s in skills}

        assert by_name["skill-a"].error is None
        assert by_name["skill-b"].error is not None
        assert "symlink" in by_name["skill-b"].error.lower()

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_root_symlink_skill_fails_and_other_skills_continue(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        """The symlinked skill fails per-skill; the real skill and external content stay safe."""
        source = self._make_collection_with_root_symlink(tmp_path)
        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=allow,
        )
        by_id = {r.skill_id: r for r in results}

        assert by_id["skill-b"].success is False
        assert "symlink" in by_id["skill-b"].message.lower()
        assert by_id["skill-a"].success, by_id["skill-a"].message
        assert (target / "skill-a" / "SKILL.md").exists()
        assert not (target / "skill-b").exists()
        assert not any(target.rglob("external-marker.txt"))

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_public_add_rejects_source_root_symlink(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        from skillport.modules.skills import add_skill

        real_skill = _create_skill(tmp_path / "outside", "skill-a")
        (real_skill / "external-marker.txt").write_text("secret", encoding="utf-8")
        source = tmp_path / "source"
        source.symlink_to(real_skill)
        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        result = add_skill(
            str(source),
            config=cfg,
            force=False,
            keep_structure=False,
            allow_symlinks=allow,
        )

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert not any(target.rglob("external-marker.txt"))

    def test_root_detection_error_does_not_process_nested_zip(
        self, tmp_path: Path, require_symlink
    ):
        from skillport.modules.skills import add_skill

        source = tmp_path / "source"
        source.mkdir()
        (source / "left").symlink_to("right", target_is_directory=True)
        (source / "right").symlink_to("left", target_is_directory=True)
        (source / "SKILL.md").symlink_to("left/SKILL.md")
        with zipfile.ZipFile(source / "nested.zip", "w") as archive:
            archive.writestr(
                "nested-skill/SKILL.md",
                "---\nname: nested-skill\ndescription: nested\n---\nbody",
            )

        target = tmp_path / "target"
        result = add_skill(
            str(source),
            config=Config(skills_dir=target),
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )

        assert result.success is False
        assert "nested-skill" not in result.added
        assert not (target / "nested-skill").exists()

    @staticmethod
    def _make_external_zip_collection(tmp_path: Path) -> Path:
        external = tmp_path / "external" / "collection"
        external.mkdir(parents=True)
        with zipfile.ZipFile(external / "inner.zip", "w") as archive:
            archive.writestr(
                "inner/SKILL.md",
                "---\nname: inner\ndescription: Inner\n---\nbody",
            )
        return external

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_source_root_symlink_does_not_install_nested_zip(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        """A symlinked source root never traverses the target for nested ZIPs."""
        from skillport.modules.skills import add_skill

        external = self._make_external_zip_collection(tmp_path)
        source = tmp_path / "source"
        source.symlink_to(external)

        target = tmp_path / "target"
        result = add_skill(
            str(source),
            config=Config(skills_dir=target),
            force=False,
            keep_structure=False,
            allow_symlinks=allow,
        )

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert "inner" not in result.added
        assert not (target / "inner").exists()

    def test_child_symlink_failure_keeps_independent_nested_zip(
        self, tmp_path: Path, require_symlink
    ):
        """A per-skill child symlink failure does not stop nested ZIP processing."""
        from skillport.modules.skills import add_skill

        source = tmp_path / "source"
        source.mkdir()
        _create_skill(source, "good")
        outside = tmp_path / "outside" / "bad"
        outside.mkdir(parents=True)
        (outside / "SKILL.md").write_text(
            "---\nname: bad\ndescription: External\n---\nexternal body", encoding="utf-8"
        )
        (source / "bad").symlink_to(outside)
        with zipfile.ZipFile(source / "inner.zip", "w") as archive:
            archive.writestr(
                "inner/SKILL.md",
                "---\nname: inner\ndescription: Inner\n---\nbody",
            )

        target = tmp_path / "target"
        result = add_skill(
            str(source), config=Config(skills_dir=target), force=False, keep_structure=False
        )

        assert "good" in result.added
        assert "inner" in result.added
        assert "bad" in result.skipped
        assert not (target / "bad").exists()


class TestSkillMdHopCycle:
    """A cyclic SKILL.md hop fails only that skill; the rest of the add continues."""

    @staticmethod
    def _make_cycle_collection(tmp_path: Path) -> Path:
        source = tmp_path / "source"
        skill_a = _create_skill(source, "skill-a")
        (skill_a / "SKILL.md").unlink()
        (skill_a / "loop").symlink_to("loop", target_is_directory=True)
        (skill_a / "SKILL.md").symlink_to("loop/file")
        _create_skill(source, "skill-b")
        return source

    def test_hop_cycle_fails_only_that_skill(self, tmp_path: Path, require_symlink):
        from skillport.modules.skills import add_skill

        source = self._make_cycle_collection(tmp_path)
        target = tmp_path / "target"

        result = add_skill(
            str(source),
            config=Config(skills_dir=target),
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )

        assert "skill-b" in result.added
        assert "skill-a" in result.skipped
        assert (target / "skill-b" / "SKILL.md").exists()
        assert not (target / "skill-a").exists()

    def test_resolve_runtime_error_becomes_per_skill_failure(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        """A RuntimeError from physical resolution surfaces as a per-skill error."""
        source = self._make_cycle_collection(tmp_path)
        loop_dir = source / "skill-a" / "loop"

        real_resolve = Path.resolve

        def resolve_raising_on_loop(self, strict=False):
            if self == loop_dir:
                raise RuntimeError(f"Symlink loop from {self!r}")
            return real_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", resolve_raising_on_loop)

        skills = detect_skills(source, allow_symlinks=True)
        by_name = {s.name: s for s in skills}

        assert by_name["skill-a"].error is not None
        assert "cycle" in by_name["skill-a"].error.lower()
        assert by_name["skill-b"].error is None


class TestSymlinkChainPhysicalResolution:
    """Chain hops resolve physically; lexical normalization cannot mask escapes."""

    def test_chain_through_alias_dot_escaping_outside_rejected(
        self, tmp_path: Path, require_symlink
    ):
        """`alias -> .` plus `link -> alias/../outside.txt` resolves outside the skill."""
        source = tmp_path / "source"
        _create_skill(source, "ok-skill")
        chain_skill = _create_skill(source, "chain-skill")
        (chain_skill / "alias").symlink_to(".")
        (source / "outside.txt").write_text("outside-secret", encoding="utf-8")
        (chain_skill / "link").symlink_to("alias/../outside.txt")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )
        by_id = {r.skill_id: r for r in results}

        assert by_id["chain-skill"].success is False
        assert "link" in by_id["chain-skill"].message
        assert not (target / "chain-skill").exists()
        assert by_id["ok-skill"].success, by_id["ok-skill"].message
        assert (target / "ok-skill" / "SKILL.md").exists()

    def test_chain_through_regular_file_is_rejected(self, tmp_path: Path, require_symlink):
        source = tmp_path / "source"
        ok_skill = _create_skill(source, "skill-ok")
        (ok_skill / "data.txt").write_text("ok data", encoding="utf-8")
        bad_skill = _create_skill(source, "skill-bad")
        (bad_skill / "file").write_text("regular file", encoding="utf-8")
        (bad_skill / "bad").symlink_to("file/../SKILL.md")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )
        by_id = {r.skill_id: r for r in results}

        assert by_id["skill-bad"].success is False
        assert "bad" in by_id["skill-bad"].message
        assert not (target / "skill-bad").exists()
        assert by_id["skill-ok"].success, by_id["skill-ok"].message
        assert (target / "skill-ok" / "SKILL.md").exists()

    def test_copy_skill_dir_rejects_symlink_source_root(self, tmp_path: Path, require_symlink):
        """copy_skill_dir refuses a symlinked source root regardless of the flag."""
        from skillport.modules.skills.internal.manager import copy_skill_dir

        real = _create_skill(tmp_path / "real", "real-skill")
        link_dir = tmp_path / "linked"
        link_dir.mkdir()
        (link_dir / "my-skill").symlink_to(real)

        for allow in (False, True):
            with pytest.raises(ValueError, match="(?i)symlink"):
                copy_skill_dir(
                    link_dir / "my-skill", tmp_path / f"dest-{allow}", allow_symlinks=allow
                )


def _create_hardlink_skill(source: Path, name: str) -> Path:
    """Create a skill containing two names for the same inode."""
    skill = _create_skill(source, name)
    data = skill / "data.txt"
    data.write_text("shared content", encoding="utf-8")
    os.link(data, skill / "data-copy.txt")
    return skill


class TestLocalHardlinkPolicy:
    """st_nlink > 1 detection is gated by allow_symlinks."""

    def test_hardlink_without_flag_copied_as_regular_file(self, tmp_path: Path):
        """Without the flag, hardlinked file is copied as regular content (status quo)."""
        source = tmp_path / "source"
        _create_hardlink_skill(source, "hard-skill")
        assert (source / "hard-skill" / "data.txt").stat().st_nlink == 2

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=cfg,
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert results[0].success, results[0].message
        copy = target / "hard-skill" / "data-copy.txt"
        assert copy.exists()
        assert not copy.is_symlink()
        assert copy.read_text(encoding="utf-8") == "shared content"
        assert copy.stat().st_nlink == 1

    def test_hardlink_with_flag_rejected(self, tmp_path: Path):
        """With the flag, st_nlink > 1 regular files fail that skill."""
        source = tmp_path / "source"
        _create_hardlink_skill(source, "hard-skill")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=cfg,
            keep_structure=False,
            force=False,
            allow_symlinks=True,
        )

        assert len(results) == 1
        assert not results[0].success
        message = results[0].message.lower()
        assert "hardlink" in message or "hard link" in message
        assert not (target / "hard-skill").exists()


class TestAllowSymlinksFlagNotPersisted:
    """The flag is transient: never stored in Config or origin JSON."""

    def test_config_has_no_allow_symlinks_field(self):
        assert "allow_symlinks" not in Config.model_fields

    def test_origin_json_has_no_allow_symlinks(self, tmp_path: Path, require_symlink):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        source = tmp_path / "source"
        skill = _create_skill(source, "skill-a")
        (skill / "target.txt").write_text("target", encoding="utf-8")
        (skill / "link.txt").symlink_to("target.txt")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            str(source), config=cfg, force=False, keep_structure=False, allow_symlinks=True
        )

        assert result.success, result.message
        origin = get_origin("skill-a", config=cfg)
        assert origin is not None
        assert "allow_symlinks" not in origin
        origins_raw = json.loads((cfg.meta_dir / "origins.json").read_text(encoding="utf-8"))
        assert origins_raw
        assert all("allow_symlinks" not in entry for entry in origins_raw.values())


class TestZipSourceSymlinkAdd:
    """Direct and nested zip adds reject a symlinked zip source path."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_direct_add_of_symlinked_zip_is_rejected(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        """add_skill on a symlinked zip path fails without installing or recording origin."""
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        outside = tmp_path / "outside" / "valid.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: inner\ndescription: Inner\n---\nbody")

        alias = tmp_path / "alias.zip"
        alias.symlink_to(outside)

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        with pytest.raises(ValueError, match="(?i)symlink"):
            add_skill(str(alias), config=cfg, force=False, allow_symlinks=allow)

        assert not (skills_dir / "inner").exists()
        assert get_origin("inner", config=cfg) is None

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_nested_add_of_symlinked_zip_is_rejected_without_blocking_siblings(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        """A nested symlinked zip fails without blocking a regular sibling zip."""
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        outside = tmp_path / "outside" / "valid.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: inner\ndescription: Inner\n---\nbody")

        source = tmp_path / "collection"
        source.mkdir()
        (source / "alias.zip").symlink_to(outside)
        with zipfile.ZipFile(source / "normal.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: normal\ndescription: Normal\n---\nbody")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, allow_symlinks=allow)

        assert not (skills_dir / "inner").exists()
        assert get_origin("inner", config=cfg) is None
        assert "normal" in result.added
        assert (skills_dir / "normal" / "SKILL.md").exists()
        assert "alias.zip" in result.skipped
        alias_detail = next(detail for detail in result.details if detail.skill_id == "alias.zip")
        assert alias_detail.success is False
        assert "symlink" in alias_detail.message.lower()

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_direct_add_under_symlinked_ancestor_is_rejected(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        """add_skill on a zip behind a symlinked ancestor directory fails without installing."""
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        outside = tmp_path / "outside" / "bundle"
        outside.mkdir(parents=True)
        with zipfile.ZipFile(outside / "inner.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: inner\ndescription: Inner\n---\nbody")

        source = tmp_path / "source"
        source.symlink_to(outside, target_is_directory=True)

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        with pytest.raises(ValueError, match="(?i)symlink"):
            add_skill(str(source / "inner.zip"), config=cfg, force=False, allow_symlinks=allow)

        assert not (skills_dir / "inner").exists()
        assert get_origin("inner", config=cfg) is None


class TestLocalSourceBoundarySymlink:
    """A local source behind a symlinked final component or ancestor is rejected."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_direct_add_under_symlinked_ancestor_is_rejected(
        self, tmp_path: Path, allow: bool, require_symlink
    ):
        """add_skill on a directory behind a symlinked ancestor installs nothing."""
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        external = _create_skill(tmp_path / "outside", "skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(tmp_path / "outside", target_is_directory=True)

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            str(source / "alias" / "skill"), config=cfg, force=False, allow_symlinks=allow
        )

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert result.added == []
        assert not (skills_dir / "skill").exists()
        assert get_origin("skill", config=cfg) is None
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_resolve_source_rejects_ancestor_symlink_before_stat(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        """resolve_source rejects the symlinked ancestor before any stat on the path."""
        from skillport.modules.skills.internal.manager import resolve_source

        outside = _create_skill(tmp_path / "outside", "skill")
        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(tmp_path / "outside", target_is_directory=True)
        target = source / "alias" / "skill"

        original_exists = Path.exists

        def exists(path: Path) -> bool:
            if path == target:
                raise AssertionError("symlinked source must be rejected before stat")
            return original_exists(path)

        monkeypatch.setattr(Path, "exists", exists)

        with pytest.raises(ValueError, match="(?i)symlink"):
            resolve_source(str(target))

        assert (outside / "SKILL.md").exists()

    def test_resolve_source_accepts_regular_local_directory(self, tmp_path: Path):
        """A regular local directory still resolves to LOCAL."""
        from skillport.modules.skills.internal.manager import resolve_source
        from skillport.shared.types import SourceType

        skill = _create_skill(tmp_path / "source", "skill")

        source_type, resolved = resolve_source(str(skill))

        assert source_type == SourceType.LOCAL
        assert resolved == str(skill)


class TestAllowSymlinksNestedZip:
    """Nested zip recursion passes the flag to the inner add_skill call."""

    def test_nested_zip_symlink_added_with_flag(self, tmp_path: Path, require_symlink):
        import zipfile

        from skillport.modules.skills import add_skill

        source_dir = tmp_path / "bundle"
        source_dir.mkdir()
        zip_path = source_dir / "inner.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: inner-skill\ndescription: Inner\n---\nbody")
            info = zipfile.ZipInfo("docs/link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "../SKILL.md")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source_dir), config=cfg, force=False, allow_symlinks=True)

        assert "inner-skill" in result.added, result.message
        link = skills_dir / "inner-skill" / "docs" / "link"
        assert link.is_symlink()
        assert os.readlink(link) == "../SKILL.md"


class TestAllowSymlinksGithubHandoff:
    """_prepare_github passes the flag to fetch_github_source_with_info."""

    def test_fetch_receives_allow_symlinks_and_skill_preserved(
        self, tmp_path, monkeypatch, require_symlink
    ):
        from types import SimpleNamespace

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.public import add as add_module

        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text(
            "---\nname: gh-skill\ndescription: GH\n---\nbody", encoding="utf-8"
        )
        assets = prepared / "assets"
        assets.mkdir()
        (assets / "m.md").write_text("m", encoding="utf-8")
        (prepared / "link.md").symlink_to("assets/m.md")

        recorded: dict = {}

        def fake_fetch(url, allow_symlinks=False):
            recorded["url"] = url
            recorded["allow_symlinks"] = allow_symlinks
            return SimpleNamespace(extracted_path=prepared, commit_sha="abc1234")

        monkeypatch.setattr(add_module, "fetch_github_source_with_info", fake_fetch)

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            "https://github.com/user/repo",
            config=cfg,
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )

        assert recorded["allow_symlinks"] is True
        assert result.success, result.message
        link = skills_dir / "gh-skill" / "link.md"
        assert link.is_symlink()
        assert os.readlink(link) == "assets/m.md"


class TestGithubAddRootSymlinkPrefetched:
    """Root symlinks inside a fetched GitHub tree fail only the offending skill."""

    def test_prefetched_tree_root_symlink_fails_only_that_skill(
        self, tmp_path, monkeypatch, require_symlink
    ):
        """A symlinked skill root in the extracted tree is rejected, others continue."""
        from types import SimpleNamespace

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.public import add as add_module

        prepared = tmp_path / "gh-extract"
        (prepared / "gh-a").mkdir(parents=True)
        (prepared / "gh-a" / "SKILL.md").write_text(
            "---\nname: gh-a\ndescription: A\n---\nbody", encoding="utf-8"
        )
        outside = tmp_path / "outside" / "gh-b"
        outside.mkdir(parents=True)
        (outside / "SKILL.md").write_text(
            "---\nname: gh-b\ndescription: B\n---\nexternal", encoding="utf-8"
        )
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")
        (prepared / "gh-b").symlink_to(outside)

        monkeypatch.setattr(
            add_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            "https://github.com/user/repo",
            config=cfg,
            force=False,
            keep_structure=True,
            namespace="user",
            allow_symlinks=True,
        )

        by_id = {d.skill_id: d for d in result.details}
        assert "user/gh-a" in result.added
        assert by_id["user/gh-b"].success is False
        assert "symlink" in by_id["user/gh-b"].message.lower()
        assert (skills_dir / "user" / "gh-a" / "SKILL.md").exists()
        assert not (skills_dir / "user" / "gh-b").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))


class TestFrontmatterKeyValidation:
    """Tests for frontmatter key existence validation on add."""

    def test_add_rejects_missing_name_key(self, tmp_path: Path):
        """SKILL.md without 'name' key in frontmatter → rejected."""
        source = tmp_path / "source"
        skill_dir = source / "bad-skill"
        skill_dir.mkdir(parents=True)
        # No 'name' key, only 'description'
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: A test skill\n---\nBody", encoding="utf-8"
        )

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert not results[0].success
        assert "name" in results[0].message.lower()

    def test_add_rejects_missing_description_key(self, tmp_path: Path):
        """SKILL.md without 'description' key in frontmatter → rejected."""
        source = tmp_path / "source"
        skill_dir = source / "bad-skill"
        skill_dir.mkdir(parents=True)
        # No 'description' key, only 'name'
        (skill_dir / "SKILL.md").write_text("---\nname: bad-skill\n---\nBody", encoding="utf-8")

        target = tmp_path / "target"
        cfg = Config(skills_dir=target)
        skills = detect_skills(source)

        results = add_local(
            source_path=source,
            skills=skills,
            config=cfg,
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert not results[0].success
        assert "description" in results[0].message.lower()


class TestResolveSourceZip:
    """Tests for resolve_source with zip files."""

    def test_resolve_zip_file(self, tmp_path: Path):
        """Zip file is resolved as SourceType.ZIP."""
        import zipfile

        from skillport.modules.skills.internal.manager import resolve_source
        from skillport.shared.types import SourceType

        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: my-skill\n---\n")

        source_type, resolved = resolve_source(str(zip_path))

        assert source_type == SourceType.ZIP
        assert resolved == str(zip_path)

    def test_resolve_zip_case_insensitive(self, tmp_path: Path):
        """Zip detection is case-insensitive (.ZIP)."""
        import zipfile

        from skillport.modules.skills.internal.manager import resolve_source
        from skillport.shared.types import SourceType

        zip_path = tmp_path / "my-skill.ZIP"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "content")

        source_type, resolved = resolve_source(str(zip_path))

        assert source_type == SourceType.ZIP

    def test_resolve_symlinked_zip_rejects_before_stat(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.internal.manager import resolve_source

        outside = tmp_path / "outside.zip"
        outside.write_bytes(b"not a zip")
        alias = tmp_path / "alias.zip"
        alias.symlink_to(outside)

        original_exists = Path.exists

        def exists(path: Path) -> bool:
            if path == alias:
                raise AssertionError("symlinked zip must be rejected before stat")
            return original_exists(path)

        monkeypatch.setattr(Path, "exists", exists)

        with pytest.raises(ValueError, match="(?i)symlink"):
            resolve_source(str(alias))


class TestAddSkillFromZip:
    """Tests for add_skill with zip files."""

    def test_add_single_skill_from_zip(self, tmp_path: Path):
        """Single skill zip is added correctly."""
        import zipfile

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        # Create zip
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "SKILL.md",
                "---\nname: my-skill\ndescription: A test skill\n---\nContent",
            )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(zip_path), config=cfg, force=False)

        assert result.success
        assert "my-skill" in result.added
        assert (skills_dir / "my-skill" / "SKILL.md").exists()

        # Check origin
        origin = get_origin("my-skill", config=cfg)
        assert origin is not None
        assert origin["kind"] == "zip"
        assert "source_mtime" in origin
        assert origin["source"] == str(zip_path)

    @staticmethod
    def _write_zip_with_symlink(zip_path: Path, link_name: str, target: str = "zskill") -> None:
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "zskill/SKILL.md",
                "---\nname: zskill\ndescription: Test skill\n---\nbody",
            )
            info = zipfile.ZipInfo(link_name)
            info.external_attr = 0o120777 << 16
            archive.writestr(info, target)

    @pytest.mark.parametrize("link_name", [".env-link", "node_modules"], ids=["hidden", "excluded"])
    def test_add_zip_ignores_hidden_or_excluded_root_symlink(
        self, tmp_path: Path, link_name: str, require_symlink, monkeypatch
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        zip_path = tmp_path / "zskill.zip"
        self._write_zip_with_symlink(zip_path, link_name)
        target = tmp_path / "skills"
        cfg = Config(skills_dir=target, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(zip_path), config=cfg, force=False, allow_symlinks=True)

        assert result.success, result.message
        assert result.added == ["zskill"]
        assert result.skipped == []
        assert (target / "zskill" / "SKILL.md").exists()
        assert not (target / "zskill" / link_name).exists()
        assert list(sandbox.iterdir()) == []
        origin = get_origin("zskill", config=cfg)
        assert origin is not None
        assert origin["kind"] == "zip"

    def test_add_zip_rejects_visible_root_symlink(
        self, tmp_path: Path, require_symlink, monkeypatch
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        zip_path = tmp_path / "zskill.zip"
        self._write_zip_with_symlink(zip_path, "alias")
        target = tmp_path / "skills"
        cfg = Config(skills_dir=target, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(zip_path), config=cfg, force=False, allow_symlinks=True)

        assert result.success is False
        assert "found 2" in result.message
        assert result.added == []
        assert not (target / "zskill").exists()
        assert list(sandbox.iterdir()) == []
        assert get_origin("zskill", config=cfg) is None

    def test_add_zip_rejects_symlink_without_flag(
        self, tmp_path: Path, require_symlink, monkeypatch
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        zip_path = tmp_path / "zskill.zip"
        self._write_zip_with_symlink(zip_path, ".env-link")
        cfg = Config(skills_dir=tmp_path / "skills", db_path=tmp_path / "db.lancedb")

        with pytest.raises(ValueError, match="(?i)symlink"):
            add_skill(str(zip_path), config=cfg, force=False)

        assert list(sandbox.iterdir()) == []
        assert get_origin("zskill", config=cfg) is None

    def test_add_zip_rejects_hidden_or_excluded_symlink_inside_skill(
        self, tmp_path: Path, require_symlink, monkeypatch
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "zskill/SKILL.md",
                "---\nname: zskill\ndescription: Test skill\n---\nbody",
            )
            info = zipfile.ZipInfo("zskill/.hidden-link")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "SKILL.md")

        target = tmp_path / "skills"
        cfg = Config(skills_dir=target, db_path=tmp_path / "db.lancedb")
        result = add_skill(str(zip_path), config=cfg, force=False, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert result.added == []
        assert not (target / "zskill").exists()
        assert list(sandbox.iterdir()) == []
        assert get_origin("zskill", config=cfg) is None

    def test_add_multiple_skills_from_zip_rejected(self, tmp_path: Path):
        """Multiple skills in a single zip are rejected (1 zip = 1 skill)."""
        import zipfile

        from skillport.modules.skills import add_skill

        # Create zip with multiple skills
        zip_path = tmp_path / "skills.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "skill-a/SKILL.md",
                "---\nname: skill-a\ndescription: Skill A\n---\nA",
            )
            zf.writestr(
                "skill-b/SKILL.md",
                "---\nname: skill-b\ndescription: Skill B\n---\nB",
            )
            zf.writestr(
                "skill-c/SKILL.md",
                "---\nname: skill-c\ndescription: Skill C\n---\nC",
            )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(zip_path), config=cfg, force=False, keep_structure=False)

        assert not result.success
        assert not result.added
        assert "exactly one skill" in result.message.lower()

    def test_add_zip_with_namespace_rejected_when_multiple(self, tmp_path: Path):
        """Even with namespace, multi-skill zip is rejected."""
        import zipfile

        from skillport.modules.skills import add_skill

        # Create zip
        zip_path = tmp_path / "skills.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "skill-a/SKILL.md",
                "---\nname: skill-a\ndescription: Skill A\n---\nA",
            )
            zf.writestr(
                "skill-b/SKILL.md",
                "---\nname: skill-b\ndescription: Skill B\n---\nB",
            )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            str(zip_path),
            config=cfg,
            force=False,
            keep_structure=True,
            namespace="my-ns",
        )

        assert not result.success
        assert not result.added
        assert "exactly one skill" in result.message.lower()

    def test_add_zip_origin_has_source_mtime(self, tmp_path: Path):
        """Zip origin includes source_mtime for update detection."""
        import zipfile

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "SKILL.md",
                "---\nname: my-skill\ndescription: Test\n---\n",
            )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        add_skill(str(zip_path), config=cfg)

        origin = get_origin("my-skill", config=cfg)

        assert origin is not None
        assert origin["kind"] == "zip"
        assert "source_mtime" in origin
        assert isinstance(origin["source_mtime"], int)
        # source_mtime should match the actual file mtime
        assert origin["source_mtime"] == zip_path.stat().st_mtime_ns

    def test_add_zip_no_skills_found(self, tmp_path: Path):
        """Zip without SKILL.md returns no skills found error."""
        import zipfile

        from skillport.modules.skills import add_skill

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("README.md", "# No skills here")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(zip_path), config=cfg)

        assert not result.success
        assert "no skills found" in result.message.lower()


class TestAddMixedDirectory:
    """Tests for adding from directories containing both zips and skill directories."""

    def test_add_directory_with_zips_and_dirs(self, tmp_path: Path):
        """Directory containing both zip files and skill directories adds all."""
        import zipfile

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        # Create mixed directory
        source_dir = tmp_path / "mixed"
        source_dir.mkdir()

        # Create zip files
        zip_a = source_dir / "a.zip"
        with zipfile.ZipFile(zip_a, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-a\ndescription: A\n---\nA")

        zip_b = source_dir / "b.zip"
        with zipfile.ZipFile(zip_b, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-b\ndescription: B\n---\nB")

        # Create skill directories
        skill_c = source_dir / "skill-c"
        skill_c.mkdir()
        (skill_c / "SKILL.md").write_text("---\nname: skill-c\ndescription: C\n---\nC")

        skill_d = source_dir / "skill-d"
        skill_d.mkdir()
        (skill_d / "SKILL.md").write_text("---\nname: skill-d\ndescription: D\n---\nD")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source_dir), config=cfg, force=False, keep_structure=False)

        assert result.success
        assert set(result.added) == {"skill-a", "skill-b", "skill-c", "skill-d"}
        assert (skills_dir / "skill-a" / "SKILL.md").exists()
        assert (skills_dir / "skill-b" / "SKILL.md").exists()
        assert (skills_dir / "skill-c" / "SKILL.md").exists()
        assert (skills_dir / "skill-d" / "SKILL.md").exists()

        # Check origins
        origin_a = get_origin("skill-a", config=cfg)
        origin_c = get_origin("skill-c", config=cfg)
        assert origin_a is not None and origin_a["kind"] == "zip"
        assert origin_c is not None and origin_c["kind"] == "local"

    def test_add_directory_with_only_zips(self, tmp_path: Path):
        """Directory containing only zip files adds all zips."""
        import zipfile

        from skillport.modules.skills import add_skill

        source_dir = tmp_path / "zips-only"
        source_dir.mkdir()

        zip_a = source_dir / "a.zip"
        with zipfile.ZipFile(zip_a, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-a\ndescription: A\n---\nA")

        zip_b = source_dir / "b.zip"
        with zipfile.ZipFile(zip_b, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-b\ndescription: B\n---\nB")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source_dir), config=cfg, force=False)

        assert result.success
        assert set(result.added) == {"skill-a", "skill-b"}

    def test_add_mixed_directory_with_namespace(self, tmp_path: Path):
        """Namespace is applied to both zip and directory skills."""
        import zipfile

        from skillport.modules.skills import add_skill

        source_dir = tmp_path / "mixed"
        source_dir.mkdir()

        # Zip file
        zip_a = source_dir / "a.zip"
        with zipfile.ZipFile(zip_a, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-a\ndescription: A\n---\nA")

        # Directory
        skill_b = source_dir / "skill-b"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text("---\nname: skill-b\ndescription: B\n---\nB")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            str(source_dir), config=cfg, force=False, keep_structure=True, namespace="my-ns"
        )

        assert result.success
        assert "my-ns/skill-a" in result.added
        assert "my-ns/skill-b" in result.added
        assert (skills_dir / "my-ns" / "skill-a" / "SKILL.md").exists()
        assert (skills_dir / "my-ns" / "skill-b" / "SKILL.md").exists()

    def test_add_mixed_directory_zip_error_continues(self, tmp_path: Path):
        """Invalid zip in directory doesn't block other skills."""
        import zipfile

        from skillport.modules.skills import add_skill

        source_dir = tmp_path / "mixed"
        source_dir.mkdir()

        # Valid zip
        zip_a = source_dir / "a.zip"
        with zipfile.ZipFile(zip_a, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-a\ndescription: A\n---\nA")

        # Invalid zip (no SKILL.md)
        zip_invalid = source_dir / "invalid.zip"
        with zipfile.ZipFile(zip_invalid, "w") as zf:
            zf.writestr("README.md", "# No skill here")

        # Directory skill
        skill_b = source_dir / "skill-b"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text("---\nname: skill-b\ndescription: B\n---\nB")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source_dir), config=cfg, force=False, keep_structure=False)

        # skill-a and skill-b should be added, invalid.zip should be skipped
        assert "skill-a" in result.added
        assert "skill-b" in result.added
        assert len(result.added) == 2

    def test_add_root_skill_ignores_zips(self, tmp_path: Path):
        """If root has SKILL.md, zip files in same directory are ignored."""
        import zipfile

        from skillport.modules.skills import add_skill

        # Create directory that is itself a skill
        source_dir = tmp_path / "root-skill"
        source_dir.mkdir()
        (source_dir / "SKILL.md").write_text("---\nname: root-skill\ndescription: Root\n---\nRoot")

        # Add a zip file (should be ignored)
        zip_a = source_dir / "a.zip"
        with zipfile.ZipFile(zip_a, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill-a\ndescription: A\n---\nA")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source_dir), config=cfg, force=False)

        # Only root-skill should be added
        assert result.success
        assert result.added == ["root-skill"]
        assert "skill-a" not in result.added


def _swap_source_at_boundary(monkeypatch, module, name: str, path: Path, external: Path) -> None:
    """Replace ``path`` with a symlink to ``external`` on the first boundary call.

    The swap lands after the source classification check and at the exact point
    where the source boundary opens the path, which is the TOCTOU window the
    boundary must close.
    """
    real_boundary = getattr(module, name)
    state = {"swapped": False}

    def swapping_boundary(*args, **kwargs):
        if not state["swapped"]:
            state["swapped"] = True
            path.rename(path.parent / f"{path.name}-original")
            path.symlink_to(external, target_is_directory=external.is_dir())
        return real_boundary(*args, **kwargs)

    monkeypatch.setattr(module, name, swapping_boundary)


def _swap_source_during_checked_copy(monkeypatch, module, path: Path, external: Path) -> None:
    """Swap ``path`` for a symlink to ``external`` around the checked tree copy.

    The swap lands after the source boundary was inspected and while the
    fallback reads the tree, then the original path is restored: the window
    between the component check and the read that a path-only snapshot would
    leave open.
    """
    real_copy = module._materialize_tree_checked
    state = {"swapped": False}
    original = path.parent / f"{path.name}-original"

    def swapping_copy(source_path, dest, before, *, markers):
        if not state["swapped"]:
            state["swapped"] = True
            path.rename(original)
            path.symlink_to(external, target_is_directory=external.is_dir())
            try:
                return real_copy(source_path, dest, before, markers=markers)
            finally:
                path.unlink()
                original.rename(path)
        return real_copy(source_path, dest, before, markers=markers)

    monkeypatch.setattr(module, "_materialize_tree_checked", swapping_copy)


class TestSourceBoundarySwap:
    """A local or zip source swapped for a symlink at the boundary is never read."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_local_add_does_not_follow_ancestor_swapped_at_boundary(
        self, tmp_path: Path, allow: bool, monkeypatch, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin
        from skillport.modules.skills.public import add as add_module

        external = _create_skill(tmp_path / "outside", "skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        original = _create_skill(source, "skill")
        (original / "original.txt").write_text("original", encoding="utf-8")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        _swap_source_at_boundary(
            monkeypatch,
            add_module,
            "acquire_local_source_snapshot",
            source,
            tmp_path / "outside",
        )

        with pytest.raises(ValueError, match="(?i)symlink"):
            add_skill(str(source), config=cfg, force=False, allow_symlinks=allow)

        assert not (skills_dir / "skill").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert get_origin("skill", config=cfg) is None
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_zip_add_does_not_follow_ancestor_swapped_at_boundary(
        self, tmp_path: Path, allow: bool, monkeypatch, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin
        from skillport.modules.skills.internal import zip_handler as zip_handler_module

        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "skill.zip", "w") as zf:
            zf.writestr(
                "SKILL.md",
                "---\nname: skill\ndescription: External\n---\nexternal body",
            )
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        with zipfile.ZipFile(source / "alias" / "skill.zip", "w") as zf:
            zf.writestr(
                "SKILL.md",
                "---\nname: skill\ndescription: Original\n---\noriginal body",
            )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        _swap_source_at_boundary(
            monkeypatch,
            zip_handler_module,
            "open_zip_source",
            source / "alias",
            outside,
        )

        with pytest.raises(ValueError, match="(?i)symlink"):
            add_skill(
                str(source / "alias" / "skill.zip"),
                config=cfg,
                force=False,
                allow_symlinks=allow,
            )

        assert not (skills_dir / "skill").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert get_origin("skill", config=cfg) is None
        assert (outside / "external-marker.txt").read_text(encoding="utf-8") == "secret"


class TestSourcePathParentComponent:
    """A local source path ending in ``..`` stays inside its destination roots."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    @pytest.mark.parametrize("path_style", ["absolute", "relative"])
    def test_local_add_with_parent_component_installs_inside_skills_dir(
        self, tmp_path: Path, allow: bool, path_style: str, monkeypatch
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        base = tmp_path / "base"
        _create_skill(base, "skill-a")
        _create_skill(base, "skill-b")
        (base / "child").mkdir()
        marker = tmp_path / "external-marker.txt"
        marker.write_text("secret", encoding="utf-8")

        if path_style == "relative":
            monkeypatch.chdir(tmp_path)
            source_arg = str(Path("base") / "child" / "..")
        else:
            source_arg = str(base / "child" / "..")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(source_arg, config=cfg, force=False, allow_symlinks=allow)

        assert result.success, result.message
        assert sorted(result.added) == ["base/skill-a", "base/skill-b"]
        assert (skills_dir / "base" / "skill-a" / "SKILL.md").exists()
        assert (skills_dir / "base" / "skill-b" / "SKILL.md").exists()
        assert not (skills_dir.parent / "skill-a").exists()
        assert not (skills_dir.parent / "skill-b").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert marker.read_text(encoding="utf-8") == "secret"
        assert list(sandbox.iterdir()) == []
        origin = get_origin("base/skill-a", config=cfg)
        assert origin is not None
        assert origin["source"] == str(base / "child" / "..")

    def test_add_local_with_parent_component_uses_safe_namespace(self, tmp_path: Path):
        base = tmp_path / "base"
        _create_skill(base, "skill-a")
        _create_skill(base, "skill-b")
        (base / "child").mkdir()

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        skills = detect_skills(base / "child" / "..")

        results = add_local(
            source_path=base / "child" / "..",
            skills=skills,
            config=cfg,
            keep_structure=True,
            force=False,
        )

        assert all(r.success for r in results), [r.message for r in results]
        assert sorted(r.skill_id for r in results) == ["base/skill-a", "base/skill-b"]
        assert (skills_dir / "base" / "skill-a" / "SKILL.md").exists()
        assert (skills_dir / "base" / "skill-b" / "SKILL.md").exists()
        assert not (skills_dir.parent / "skill-a").exists()
        assert not (skills_dir.parent / "skill-b").exists()

    def test_snapshot_is_materialized_inside_dedicated_temp_root(self, tmp_path: Path, monkeypatch):
        from skillport.modules.skills.internal import acquire_local_source_snapshot
        from skillport.modules.skills.internal import manager as manager_module

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        base = tmp_path / "base"
        _create_skill(base, "skill-a")
        (base / "child").mkdir()

        captured: dict[str, Path] = {}
        real_materialize = manager_module._materialize_tree

        def capturing_materialize(fd, dest, *, markers):
            captured.setdefault("snapshot", dest)
            return real_materialize(fd, dest, markers=markers)

        monkeypatch.setattr(manager_module, "_materialize_tree", capturing_materialize)

        stable = acquire_local_source_snapshot(base / "child" / "..")
        try:
            snapshot = captured["snapshot"]
            assert stable.snapshot == snapshot
            assert snapshot.name == "base"
            assert snapshot.resolve().is_relative_to(stable.temp_root.resolve())
        finally:
            stable.cleanup()

    def test_local_add_with_parent_component_processes_nested_zip(
        self, tmp_path: Path, monkeypatch
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        base = tmp_path / "base"
        (base / "child").mkdir(parents=True)
        with zipfile.ZipFile(base / "inner.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: inner\ndescription: I\n---\ninner body")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(base / "child" / ".."), config=cfg, force=False)

        assert result.success, result.message
        assert result.added == ["inner"]
        assert "inner body" in (skills_dir / "inner" / "SKILL.md").read_text(encoding="utf-8")
        assert not (skills_dir.parent / "inner").exists()
        assert not (skills_dir.parent / "base" / "inner").exists()
        assert list(sandbox.iterdir()) == []
        origin = get_origin("inner", config=cfg)
        assert origin is not None
        assert origin["kind"] == "zip"
        assert Path(origin["source"]).name == "inner.zip"

    def test_safe_basename_rejects_parent_and_empty_names(self):
        from skillport.shared.utils import safe_basename

        assert safe_basename("/tmp/source/..") == "tmp"
        assert safe_basename("/tmp/source/../other") == "other"
        assert safe_basename("..") == "source"
        assert safe_basename("/") == "source"
        assert safe_basename("") == "source"


class TestGithubTarballPerSkillSymlink:
    """A GitHub tarball with a violating symlink fails only that skill."""

    @staticmethod
    def _make_multi_skill_tar(tmp_path: Path, link_target: str) -> Path:
        import io
        import tarfile

        tar_path = tmp_path / "repo.tar.gz"
        root = "user-repo-abc1234"
        with tarfile.open(tar_path, "w:gz") as tar:
            for name, data in (
                ("skills/bad/SKILL.md", b"---\nname: bad\ndescription: B\n---\nbad body"),
                ("skills/good/SKILL.md", b"---\nname: good\ndescription: G\n---\ngood body"),
                ("skills/good/notes.txt", b"good notes"),
            ):
                info = tarfile.TarInfo(f"{root}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            link = tarfile.TarInfo(f"{root}/skills/bad/evil")
            link.type = tarfile.SYMTYPE
            link.linkname = link_target
            tar.addfile(link)
        return tar_path

    def test_violating_symlink_fails_only_that_skill(self, tmp_path: Path, monkeypatch):
        from types import SimpleNamespace

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin, parse_github_url
        from skillport.modules.skills.internal.github import extract_tarball
        from skillport.modules.skills.public import add as add_module

        external = tmp_path / "outside"
        external.mkdir()
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        tar_path = self._make_multi_skill_tar(tmp_path, str(external / "external-marker.txt"))
        parsed = parse_github_url("https://github.com/user/repo/tree/main/skills")
        extracted, commit_sha = extract_tarball(tar_path, parsed, allow_symlinks=True)

        monkeypatch.setattr(
            add_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=extracted, commit_sha=commit_sha
            ),
        )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            "https://github.com/user/repo/tree/main/skills",
            config=cfg,
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )

        by_id = {d.skill_id: d for d in result.details}
        assert "good" in result.added
        assert "bad" in result.skipped
        assert by_id["bad"].success is False
        assert "relative" in by_id["bad"].message.lower()
        assert (skills_dir / "good" / "SKILL.md").exists()
        assert "good body" in (skills_dir / "good" / "SKILL.md").read_text(encoding="utf-8")
        assert not (skills_dir / "bad").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert get_origin("bad", config=cfg) is None
        good_origin = get_origin("good", config=cfg)
        assert good_origin is not None and good_origin["kind"] == "github"
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"


class TestOriginSourceRelativePath:
    """Origin path records the actual source-relative skill directory."""

    def test_root_local_skill_records_empty_origin_path(self, tmp_path, require_symlink):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        source = tmp_path / "root-skill"
        (source / "assets").mkdir(parents=True)
        (source / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nbody", encoding="utf-8"
        )
        (source / "assets" / "manual.md").write_text("v1\n", encoding="utf-8")
        (source / "docs").mkdir()
        os.symlink("../assets/manual.md", source / "docs" / "current.md")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, allow_symlinks=True)

        assert result.success, result.message
        origin = get_origin("root-skill", config=cfg)
        assert origin is not None
        assert origin["path"] == ""
        assert (skills_dir / "root-skill" / "docs" / "current.md").is_symlink()

    def test_child_local_skill_records_relative_origin_path(self, tmp_path):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin
        from skillport.modules.skills.public import update_skill

        source = tmp_path / "collection"
        child = source / "the-skill"
        child.mkdir(parents=True)
        (child / "SKILL.md").write_text(
            "---\nname: the-skill\ndescription: C\n---\nbody", encoding="utf-8"
        )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False)

        assert result.success, result.message
        origin = get_origin("the-skill", config=cfg)
        assert origin is not None
        assert origin["path"] == "the-skill"

        (child / "SKILL.md").write_text(
            "---\nname: the-skill\ndescription: C\n---\nbody2", encoding="utf-8"
        )
        update_result = update_skill("the-skill", config=cfg)

        assert update_result.success, update_result.message
        assert (skills_dir / "the-skill" / "SKILL.md").read_text(encoding="utf-8").endswith("body2")

    def test_root_github_skill_records_empty_origin_path(
        self, tmp_path, monkeypatch, require_symlink
    ):
        from types import SimpleNamespace

        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin
        from skillport.modules.skills.public import add as add_module

        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text(
            "---\nname: gh-root\ndescription: G\n---\nbody", encoding="utf-8"
        )
        os.symlink("SKILL.md", prepared / "link.md")

        monkeypatch.setattr(
            add_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            "https://github.com/user/repo",
            config=cfg,
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )

        assert result.success, result.message
        origin = get_origin("gh-root", config=cfg)
        assert origin is not None
        assert origin["path"] == ""


class TestNoFollowUnsupportedPlatform:
    """Adds still work when O_NOFOLLOW and descriptor-relative opens are unavailable."""

    def test_local_add_succeeds(self, tmp_path: Path, no_follow_unavailable):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        source = tmp_path / "source"
        _create_skill(source, "skill")
        (source / "skill" / "notes.txt").write_text("notes", encoding="utf-8")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, allow_symlinks=False)

        assert result.success, result.message
        assert result.added == ["skill"]
        assert (skills_dir / "skill" / "SKILL.md").exists()
        assert (skills_dir / "skill" / "notes.txt").read_text(encoding="utf-8") == "notes"
        origin = get_origin("skill", config=cfg)
        assert origin is not None
        assert origin["source"] == str(source)

    def test_zip_add_succeeds(self, tmp_path: Path, no_follow_unavailable):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        zip_path = tmp_path / "skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: D\n---\nbody")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(zip_path), config=cfg, force=False, allow_symlinks=False)

        assert result.success, result.message
        assert result.added == ["skill"]
        assert (skills_dir / "skill" / "SKILL.md").read_text(encoding="utf-8").endswith("body")
        origin = get_origin("skill", config=cfg)
        assert origin is not None
        assert origin["kind"] == "zip"

    def test_local_add_rejects_symlink_ancestor(
        self, tmp_path: Path, no_follow_unavailable, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        external = _create_skill(tmp_path / "outside", "skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(tmp_path / "outside", target_is_directory=True)

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source / "alias" / "skill"), config=cfg, force=False)

        assert result.success is False
        assert "traverses a symlink" in result.message.lower()
        assert not any(skills_dir.rglob("skill"))
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert get_origin("skill", config=cfg) is None
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_zip_add_rejects_symlinked_archive(
        self, tmp_path: Path, no_follow_unavailable, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin

        zip_path = tmp_path / "real.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: skill\ndescription: D\n---\nbody")

        alias = tmp_path / "alias.zip"
        alias.symlink_to(zip_path)

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        with pytest.raises(ValueError, match="traverses a symlink"):
            add_skill(str(alias), config=cfg, force=False)

        assert not any(skills_dir.rglob("skill"))
        assert get_origin("skill", config=cfg) is None

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_local_add_does_not_follow_ancestor_swapped_at_boundary(
        self, tmp_path: Path, allow: bool, no_follow_unavailable, monkeypatch, require_symlink
    ):
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin
        from skillport.modules.skills.public import add as add_module

        external = _create_skill(tmp_path / "outside", "skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        original = _create_skill(source, "skill")
        (original / "original.txt").write_text("original", encoding="utf-8")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        _swap_source_at_boundary(
            monkeypatch,
            add_module,
            "acquire_local_source_snapshot",
            source,
            tmp_path / "outside",
        )

        with pytest.raises(ValueError, match="traverses a symlink"):
            add_skill(str(source), config=cfg, force=False, allow_symlinks=allow)

        assert not (skills_dir / "skill").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert get_origin("skill", config=cfg) is None
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_local_add_rejects_ancestor_swapped_during_checked_copy(
        self, tmp_path: Path, allow: bool, no_follow_unavailable, monkeypatch, require_symlink
    ):
        """A swap after the boundary check but during the tree read is rejected."""
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.internal import get_origin
        from skillport.modules.skills.internal import manager as manager_module

        external = _create_skill(tmp_path / "outside", "skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        original = _create_skill(source, "skill")
        (original / "original.txt").write_text("original", encoding="utf-8")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        _swap_source_during_checked_copy(monkeypatch, manager_module, source, tmp_path / "outside")

        with pytest.raises(ValueError, match="(?i)changed|symlink"):
            add_skill(str(source), config=cfg, force=False, allow_symlinks=allow)

        assert not (skills_dir / "skill").exists()
        assert not any(skills_dir.rglob("external-marker.txt"))
        assert get_origin("skill", config=cfg) is None
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"
        assert (original / "original.txt").read_text(encoding="utf-8") == "original"

    def test_local_add_with_flag_rejects_hardlink(self, tmp_path: Path, no_follow_unavailable):
        from skillport.modules.skills import add_skill

        source = tmp_path / "source"
        skill = _create_skill(source, "skill")
        hardlinked = tmp_path / "hardlinked.txt"
        hardlinked.write_text("hard", encoding="utf-8")
        os.link(hardlinked, skill / "linked.txt")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, allow_symlinks=True)

        assert result.success is False
        assert "hardlink" in result.message.lower()
        assert not (skills_dir / "skill").exists()

    def test_local_add_without_flag_copies_hardlink(self, tmp_path: Path, no_follow_unavailable):
        from skillport.modules.skills import add_skill

        source = tmp_path / "source"
        skill = _create_skill(source, "skill")
        hardlinked = tmp_path / "hardlinked.txt"
        hardlinked.write_text("hard", encoding="utf-8")
        os.link(hardlinked, skill / "linked.txt")

        skills_dir = tmp_path / "skills"
        cfg = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, allow_symlinks=False)

        assert result.success, result.message
        assert (skills_dir / "skill" / "linked.txt").read_text(encoding="utf-8") == "hard"


class TestAddLocalVendorWarnings:
    """Successful adds carry non-fatal warnings; failures and skips do not."""

    def test_successful_add_carries_vendor_warning(self, tmp_path: Path):
        source = tmp_path / "source"
        _create_skill_with_frontmatter(source, "skill-a", "model: sonnet\nicon: toolbox")
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=tmp_path / "target"),
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert results[0].success, results[0].message
        warnings = [w for w in results[0].warnings if w.severity == "warning"]
        assert len(warnings) == 1
        message = warnings[0].message
        assert "model" in message
        assert "icon" in message
        assert "Claude Code" in message
        assert "Cursor" in message

    def test_failed_add_carries_no_warnings(self, tmp_path: Path):
        source = tmp_path / "source"
        _create_skill_with_frontmatter(source, "bad-skill", "bogus-field: value")
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=tmp_path / "target"),
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert not results[0].success
        assert results[0].warnings == []

    def test_skipped_existing_skill_carries_no_warnings(self, tmp_path: Path):
        source = tmp_path / "source"
        _create_skill_with_frontmatter(source, "skill-a", "model: sonnet")
        target = tmp_path / "target"
        _create_skill(target, "skill-a")

        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=target),
            keep_structure=False,
            force=False,
        )

        assert len(results) == 1
        assert not results[0].success
        assert results[0].warnings == []

    def test_standard_only_skill_carries_no_warnings(self, tmp_path: Path):
        source = tmp_path / "source"
        _create_skill(source, "skill-a")
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=tmp_path / "target"),
            keep_structure=False,
            force=False,
        )

        assert results[0].success, results[0].message
        assert results[0].warnings == []

    def test_line_count_warning_is_carried(self, tmp_path: Path):
        source = tmp_path / "source"
        body = "\n".join(["line"] * 501)
        _create_skill_with_frontmatter(source, "skill-a", "", body=body)
        results = add_local(
            source_path=source,
            skills=detect_skills(source),
            config=Config(skills_dir=tmp_path / "target"),
            keep_structure=False,
            force=False,
        )

        assert results[0].success, results[0].message
        assert any("lines" in w.message.lower() for w in results[0].warnings)


class TestAddSkillVendorWarningPropagation:
    """Public add result forwards per-skill warnings to its details (order.md §3.3)."""

    def test_success_detail_carries_warning(self, tmp_path: Path):
        from skillport.modules.skills import add_skill

        source = tmp_path / "source"
        _create_skill_with_frontmatter(source, "vendor-skill", "model: sonnet")
        cfg = Config(skills_dir=tmp_path / "installed", db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, keep_structure=False)

        assert result.success, result.message
        assert len(result.details) == 1
        assert result.details[0].success
        warnings = result.details[0].warnings
        assert len(warnings) == 1
        assert "model" in warnings[0].message
        assert "Claude Code" in warnings[0].message
        assert "warnings" not in result.model_dump()

    def test_failed_detail_carries_no_warning(self, tmp_path: Path):
        from skillport.modules.skills import add_skill

        source = tmp_path / "source"
        _create_skill_with_frontmatter(source, "bad-skill", "bogus-field: value")
        cfg = Config(skills_dir=tmp_path / "installed", db_path=tmp_path / "db.lancedb")

        result = add_skill(str(source), config=cfg, force=False, keep_structure=False)

        assert not result.success
        assert len(result.details) == 1
        assert not result.details[0].success
        assert result.details[0].warnings == []
        assert "warnings" not in result.model_dump()
