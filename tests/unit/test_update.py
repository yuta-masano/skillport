"""Unit tests for skill update functionality."""

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillport.modules.skills import (
    check_update_available,
    detect_local_modification,
    update_all_skills,
    update_skill,
)
from skillport.modules.skills.internal import (
    compute_content_hash,
    get_missing_skill_ids,
    get_origin,
    get_tracked_skill_ids,
    get_untracked_skill_ids,
    record_origin,
    scan_installed_skill_ids,
)
from skillport.shared.config import Config


def _git_blob_sha(data: bytes) -> str:
    """Git blob object SHA-1: sha1(b\"blob <len>\\0\" + data)."""
    return hashlib.sha1(f"blob {len(data)}\x00".encode() + data).hexdigest()


def _tree_hash_of_entries(entries) -> str:
    """Outer hash in the same shape get_remote_tree_hash builds from tree API blobs."""
    hasher = hashlib.sha256()
    for rel, data in sorted(entries, key=lambda e: e[0]):
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(_git_blob_sha(data).encode("utf-8"))
        hasher.update(b"\x00")
    return f"sha256:{hasher.hexdigest()}"


def _installed_tree_state(root: Path) -> dict[str, tuple]:
    """Snapshot every entry below root by lstat, keeping dangling/cyclic links visible."""
    state: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                state[relative] = ("symlink", os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                state[relative] = ("dir",)
            elif stat.S_ISREG(info.st_mode):
                state[relative] = ("file", path.read_bytes())
            else:
                state[relative] = ("other", stat.S_IFMT(info.st_mode))
    return state


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


class TestDetectLocalModification:
    """Tests for local modification detection."""

    def test_no_origin_returns_false(self, tmp_path):
        """No origin info means no tracking, returns False."""
        config = Config(skills_dir=tmp_path / "skills", db_path=tmp_path / "db.lancedb")

        result = detect_local_modification("nonexistent", config=config)

        assert result is False

    def test_no_content_hash_returns_false(self, tmp_path):
        """Origin without content_hash (v1) returns False."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record origin without content_hash (simulating v1)
        record_origin("my-skill", {"source": "test", "kind": "local"}, config=config)

        result = detect_local_modification("my-skill", config=config)

        # Migration adds empty content_hash, which means "unknown", so no modification detected
        assert result is False

    def test_matching_hash_returns_false(self, tmp_path):
        """Matching content_hash means no modification."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        content_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {"source": str(skill_dir), "kind": "local", "content_hash": content_hash},
            config=config,
        )

        result = detect_local_modification("my-skill", config=config)

        assert result is False

    def test_different_hash_returns_true(self, tmp_path):
        """Different content_hash means modification detected."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {"source": str(skill_dir), "kind": "local", "content_hash": "sha256:old_hash"},
            config=config,
        )

        result = detect_local_modification("my-skill", config=config)

        assert result is True


class TestCheckUpdateAvailable:
    """Tests for check_update_available function."""

    def test_no_origin_not_available(self, tmp_path):
        """No origin info means not updatable."""
        config = Config(skills_dir=tmp_path / "skills", db_path=tmp_path / "db.lancedb")

        result = check_update_available("nonexistent", config=config)

        assert result["available"] is False
        assert "no origin" in result["reason"].lower()

    def test_builtin_not_available(self, tmp_path):
        """Builtin skills cannot be updated."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("hello-world", {"source": "hello-world", "kind": "builtin"}, config=config)

        result = check_update_available("hello-world", config=config)

        assert result["available"] is False
        assert "built-in" in result["reason"].lower()

    def test_local_missing_source_not_available(self, tmp_path):
        """Local skill with missing source path is not updatable."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {"source": "/nonexistent/path", "kind": "local"},
            config=config,
        )

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "not found" in result["reason"].lower()

    def test_local_with_source_available(self, tmp_path):
        """Local skill with valid source is updatable."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "SKILL.md").write_text("source body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # installed copy differs
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("installed body")

        record_origin(
            "my-skill",
            {"source": str(source_dir), "kind": "local"},
            config=config,
        )

        result = check_update_available("my-skill", config=config)

        assert result["available"] is True

    def test_github_same_content_not_available(self, tmp_path, monkeypatch):
        """GitHub skill with same tree hash is up to date."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "commit_sha": "abc1234567890",
            },
            config=config,
        )

        # create installed content
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("body")

        def mock_get_remote_tree_hash(parsed, token, path=None):
            return compute_content_hash(skill_dir)

        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(update_module, "get_remote_tree_hash", mock_get_remote_tree_hash)

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "latest" in result["reason"].lower()

    def test_github_different_content_available(self, tmp_path, monkeypatch):
        """GitHub skill with different tree hash has update available."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "commit_sha": "abc1234567890",
            },
            config=config,
        )

        # create installed content
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("old")

        from skillport.modules.skills.public import update as update_module

        def mock_get_remote_tree_hash(parsed, token, path=None):
            return "sha256:remotehash"

        monkeypatch.setattr(update_module, "get_remote_tree_hash", mock_get_remote_tree_hash)

        result = check_update_available("my-skill", config=config)

        assert result["available"] is True
        assert "remote" in result["reason"].lower()

    def test_github_api_failure_not_available(self, tmp_path, monkeypatch):
        """GitHub API failure should not mark as available immediately after add."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "commit_sha": "abc1234567890",
            },
            config=config,
        )

        # Mock get_remote_tree_hash to return empty string (API failure)
        def mock_get_remote_tree_hash(parsed, token, path=None):
            return ""

        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(update_module, "get_remote_tree_hash", mock_get_remote_tree_hash)

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "remote tree" in result["reason"].lower()

    @pytest.mark.parametrize("layout", ["skill-root", "ancestor"])
    def test_installed_symlink_is_rejected_before_hashing(
        self, tmp_path, layout: str, monkeypatch, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        real_branch = tmp_path / "real-branch"
        real_skill = real_branch / "my-skill"
        real_skill.mkdir(parents=True)
        (real_skill / "SKILL.md").write_text("installed")

        if layout == "skill-root":
            (skills_dir / "my-skill").symlink_to(real_skill, target_is_directory=True)
            skill_id = "my-skill"
        else:
            (skills_dir / "alias").symlink_to(real_branch, target_is_directory=True)
            skill_id = "alias/my-skill"

        source = tmp_path / "source" / "my-skill"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("source")
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            skill_id,
            {
                "source": str(source.parent),
                "kind": "local",
                "path": skill_id,
                "content_hash": "sha256:installed",
            },
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        def fail_hash(*args, **kwargs):
            raise AssertionError("installed symlink was hashed")

        monkeypatch.setattr(update_module, "compute_content_hash", fail_hash)
        monkeypatch.setattr(update_module, "_compute_source_hash", fail_hash)

        assert detect_local_modification(skill_id, config=config) is False
        result = check_update_available(skill_id, config=config)

        assert result["available"] is False
        assert "symlink" in result["reason"].lower()

    def test_show_available_updates_classifies_installed_symlink(self, tmp_path, require_symlink):
        from skillport.interfaces.cli.commands.update import _show_available_updates

        real_skills = tmp_path / "real-skills"
        real_skill = real_skills / "my-skill"
        real_skill.mkdir(parents=True)
        (real_skill / "SKILL.md").write_text("installed")
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "my-skill").symlink_to(real_skill, target_is_directory=True)
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {"source": "/nonexistent/source", "kind": "local"},
            config=config,
        )

        result = _show_available_updates(config, json_output=False)

        assert [item["skill_id"] for item in result["not_updatable"]] == ["my-skill"]
        assert "symlink" in result["not_updatable"][0]["reason"].lower()


class TestUpdateSkill:
    """Tests for update_skill function."""

    def test_update_nonexistent_skill_fails(self, tmp_path):
        """Updating non-existent skill fails."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = update_skill("nonexistent", config=config)

        assert result.success is False
        assert "not found" in result.message.lower()

    def test_update_skill_without_origin_fails(self, tmp_path):
        """Updating skill without origin fails."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = update_skill("my-skill", config=config)

        assert result.success is False
        assert "no origin" in result.message.lower()

    def test_update_builtin_fails(self, tmp_path):
        """Updating builtin skill fails."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "hello-world"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: hello-world\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("hello-world", {"source": "hello-world", "kind": "builtin"}, config=config)

        result = update_skill("hello-world", config=config)

        assert result.success is False
        assert "built-in" in result.message.lower()

    def test_update_local_modified_without_force_fails(self, tmp_path):
        """Updating locally modified skill without force fails."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nmodified body")

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "SKILL.md").write_text("---\nname: my-skill\n---\noriginal body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record with original hash
        original_hash = compute_content_hash(source_dir)
        record_origin(
            "my-skill",
            {"source": str(source_dir), "kind": "local", "content_hash": original_hash},
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert result.success is False
        assert result.local_modified is True
        assert "--force" in result.message

    def test_update_local_modified_with_force_succeeds(self, tmp_path):
        """Updating locally modified skill with force succeeds."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nmodified body")

        source_dir = tmp_path / "source" / "my-skill"
        source_dir.mkdir(parents=True)
        (source_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nnew body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record with different hash
        record_origin(
            "my-skill",
            {"source": str(source_dir), "kind": "local", "content_hash": "sha256:old"},
            config=config,
        )

        result = update_skill("my-skill", config=config, force=True)

        assert result.success is True
        assert "my-skill" in result.updated

        # Verify content was updated
        assert (skill_dir / "SKILL.md").read_text() == "---\nname: my-skill\n---\nnew body"

    def test_update_local_already_up_to_date(self, tmp_path):
        """Local skill with matching hash is already up to date."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        source_dir = tmp_path / "source" / "my-skill"
        source_dir.mkdir(parents=True)
        (source_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")  # Same content

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        content_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {"source": str(source_dir), "kind": "local", "content_hash": content_hash},
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert result.success is True
        assert "my-skill" in result.skipped
        assert "up to date" in result.message.lower()

    def test_update_dry_run_no_changes(self, tmp_path):
        """Dry run shows what would be updated without changes."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nold body")

        source_dir = tmp_path / "source" / "my-skill"
        source_dir.mkdir(parents=True)
        (source_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nnew body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        content_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {"source": str(source_dir), "kind": "local", "content_hash": content_hash},
            config=config,
        )

        result = update_skill("my-skill", config=config, dry_run=True)

        assert result.success is True
        assert "my-skill" in result.updated
        assert "would" in result.message.lower()

        # Content should NOT be changed
        assert (skill_dir / "SKILL.md").read_text() == "---\nname: my-skill\n---\nold body"


class TestScanInstalledSkillIds:
    """Tests for scan_installed_skill_ids function (T1)."""

    def test_flat_skill_detected(self, tmp_path):
        """Flat skill (my-skill/) is detected."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = scan_installed_skill_ids(config=config)

        assert result == {"my-skill"}

    def test_nested_skill_detected(self, tmp_path):
        """Nested skill (ns/my-skill/) is detected."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "ns" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = scan_installed_skill_ids(config=config)

        assert result == {"ns/my-skill"}

    def test_hidden_directories_skipped(self, tmp_path):
        """Hidden directories (.git, .venv) are skipped."""
        skills_dir = tmp_path / "skills"
        # Valid skill
        valid_skill = skills_dir / "valid-skill"
        valid_skill.mkdir(parents=True)
        (valid_skill / "SKILL.md").write_text("---\nname: valid\n---\nbody")
        # Hidden directory skill (should be skipped)
        hidden_skill = skills_dir / ".git" / "hooks-skill"
        hidden_skill.mkdir(parents=True)
        (hidden_skill / "SKILL.md").write_text("---\nname: hidden\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = scan_installed_skill_ids(config=config)

        assert result == {"valid-skill"}

    def test_node_modules_skipped(self, tmp_path):
        """node_modules directory is skipped."""
        skills_dir = tmp_path / "skills"
        # Valid skill
        valid_skill = skills_dir / "valid-skill"
        valid_skill.mkdir(parents=True)
        (valid_skill / "SKILL.md").write_text("---\nname: valid\n---\nbody")
        # node_modules skill (should be skipped)
        node_skill = skills_dir / "node_modules" / "some-skill"
        node_skill.mkdir(parents=True)
        (node_skill / "SKILL.md").write_text("---\nname: node\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = scan_installed_skill_ids(config=config)

        assert result == {"valid-skill"}

    def test_nonexistent_skills_dir_returns_empty(self, tmp_path):
        """Non-existent skills_dir returns empty set."""
        config = Config(skills_dir=tmp_path / "nonexistent", db_path=tmp_path / "db.lancedb")

        result = scan_installed_skill_ids(config=config)

        assert result == set()

    def test_empty_skills_dir_returns_empty(self, tmp_path):
        """Existing but empty skills_dir returns empty set."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = scan_installed_skill_ids(config=config)

        assert result == set()


class TestGetTrackedSkillIds:
    """Tests for get_tracked_skill_ids function (T1)."""

    def test_tracked_skill_returned(self, tmp_path):
        """Skills in origins.json are returned."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("my-skill", {"source": "test", "kind": "local"}, config=config)

        result = get_tracked_skill_ids(config=config)

        assert result == {"my-skill"}

    def test_different_skills_dir_excluded(self, tmp_path):
        """Skills with different skills_dir are excluded."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        other_dir = tmp_path / "other"
        other_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record skill for current skills_dir
        record_origin("my-skill", {"source": "test", "kind": "local"}, config=config)

        # Manually add skill for different skills_dir
        origins_path = config.meta_dir / "origins.json"
        with open(origins_path, encoding="utf-8") as f:
            data = json.load(f)
        data["other-skill"] = {"source": "test", "kind": "local", "skills_dir": str(other_dir)}
        with open(origins_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = get_tracked_skill_ids(config=config)

        assert result == {"my-skill"}

    def test_legacy_entry_without_skills_dir_included(self, tmp_path):
        """Legacy entries without skills_dir field are included."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Manually create legacy entry without skills_dir
        origins_path = config.meta_dir / "origins.json"
        origins_path.parent.mkdir(parents=True, exist_ok=True)
        with open(origins_path, "w", encoding="utf-8") as f:
            json.dump({"legacy-skill": {"source": "test", "kind": "local"}}, f)

        result = get_tracked_skill_ids(config=config)

        assert result == {"legacy-skill"}


class TestGetUntrackedSkillIds:
    """Tests for get_untracked_skill_ids function (T1)."""

    def test_tracked_skill_not_in_untracked(self, tmp_path):
        """Skills in origins.json are not in untracked list."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("my-skill", {"source": "test", "kind": "local"}, config=config)

        result = get_untracked_skill_ids(config=config)

        assert result == []

    def test_untracked_skill_in_list(self, tmp_path):
        """Skills not in origins.json are in untracked list."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "untracked-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: untracked\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = get_untracked_skill_ids(config=config)

        assert result == ["untracked-skill"]

    def test_untracked_sorted_alphabetically(self, tmp_path):
        """Untracked skills are sorted alphabetically."""
        skills_dir = tmp_path / "skills"
        for name in ["zebra", "alpha", "beta"]:
            skill_dir = skills_dir / name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = get_untracked_skill_ids(config=config)

        assert result == ["alpha", "beta", "zebra"]

    def test_mixed_tracked_and_untracked(self, tmp_path):
        """Mix of tracked and untracked skills returns only untracked."""
        skills_dir = tmp_path / "skills"
        for name in ["tracked-1", "untracked-1", "tracked-2", "untracked-2"]:
            skill_dir = skills_dir / name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("tracked-1", {"source": "test", "kind": "local"}, config=config)
        record_origin("tracked-2", {"source": "test", "kind": "local"}, config=config)

        result = get_untracked_skill_ids(config=config)

        assert result == ["untracked-1", "untracked-2"]


class TestGetMissingSkillIds:
    """Tests for get_missing_skill_ids function (T1)."""

    def test_missing_skill_detected(self, tmp_path):
        """Tracked but not installed skills are detected."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("missing-skill", {"source": "test", "kind": "local"}, config=config)

        result = get_missing_skill_ids(config=config)

        assert result == {"missing-skill"}

    def test_installed_skill_not_missing(self, tmp_path):
        """Installed skills are not in missing set."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("my-skill", {"source": "test", "kind": "local"}, config=config)

        result = get_missing_skill_ids(config=config)

        assert result == set()

    def test_untracked_and_missing_coexist(self, tmp_path):
        """Untracked and missing skills can exist simultaneously."""
        skills_dir = tmp_path / "skills"

        # Create untracked skill
        untracked_dir = skills_dir / "untracked-skill"
        untracked_dir.mkdir(parents=True)
        (untracked_dir / "SKILL.md").write_text("---\nname: untracked\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record missing skill (tracked but not installed)
        record_origin("missing-skill", {"source": "test", "kind": "local"}, config=config)

        untracked = get_untracked_skill_ids(config=config)
        missing = get_missing_skill_ids(config=config)

        assert untracked == ["untracked-skill"]
        assert missing == {"missing-skill"}


class TestShowAvailableUpdatesJSON:
    """Tests for CLI JSON output with untracked field (T2)."""

    def test_untracked_field_in_json_output(self, tmp_path):
        """JSON output includes untracked field with expected skill IDs."""
        from skillport.interfaces.cli.commands.update import _show_available_updates

        skills_dir = tmp_path / "skills"

        # Create untracked skill
        untracked_dir = skills_dir / "untracked-skill"
        untracked_dir.mkdir(parents=True)
        (untracked_dir / "SKILL.md").write_text("---\nname: untracked\n---\nbody")

        # Create tracked skill
        tracked_dir = skills_dir / "tracked-skill"
        tracked_dir.mkdir(parents=True)
        (tracked_dir / "SKILL.md").write_text("---\nname: tracked\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("tracked-skill", {"source": "test", "kind": "local"}, config=config)

        # Get the data directly (skip actual JSON printing)
        result = _show_available_updates(config, json_output=False)

        assert "untracked" in result
        assert result["untracked"] == ["untracked-skill"]

    def test_untracked_empty_list_when_none(self, tmp_path):
        """JSON output has empty untracked list when no untracked skills."""
        from skillport.interfaces.cli.commands.update import _show_available_updates

        skills_dir = tmp_path / "skills"

        # Create only tracked skill
        tracked_dir = skills_dir / "tracked-skill"
        tracked_dir.mkdir(parents=True)
        (tracked_dir / "SKILL.md").write_text("---\nname: tracked\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("tracked-skill", {"source": "test", "kind": "local"}, config=config)

        result = _show_available_updates(config, json_output=False)

        assert "untracked" in result
        assert result["untracked"] == []

    def test_json_output_preserves_other_fields(self, tmp_path):
        """Adding untracked field does not affect other fields."""
        from skillport.interfaces.cli.commands.update import _show_available_updates

        skills_dir = tmp_path / "skills"

        # Create tracked builtin skill
        builtin_dir = skills_dir / "hello-world"
        builtin_dir.mkdir(parents=True)
        (builtin_dir / "SKILL.md").write_text("---\nname: hello-world\n---\nbody")

        # Create untracked skill
        untracked_dir = skills_dir / "untracked-skill"
        untracked_dir.mkdir(parents=True)
        (untracked_dir / "SKILL.md").write_text("---\nname: untracked\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin("hello-world", {"source": "hello-world", "kind": "builtin"}, config=config)

        result = _show_available_updates(config, json_output=False)

        # Check all expected fields exist
        assert "updates_available" in result
        assert "up_to_date" in result
        assert "not_updatable" in result
        assert "untracked" in result

        # Verify structure
        assert isinstance(result["updates_available"], list)
        assert isinstance(result["up_to_date"], list)
        assert isinstance(result["not_updatable"], list)
        assert isinstance(result["untracked"], list)

        # Verify builtin is in not_updatable
        not_updatable_ids = [item["skill_id"] for item in result["not_updatable"]]
        assert "hello-world" in not_updatable_ids

        # Verify untracked skill
        assert result["untracked"] == ["untracked-skill"]


class TestCheckUpdateAvailableZip:
    """Tests for check_update_available with zip sources."""

    def test_zip_missing_source_not_available(self, tmp_path):
        """Zip skill with missing source file is not updatable."""

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record origin pointing to non-existent zip
        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "nonexistent.zip"),
                "kind": "zip",
                "source_mtime": 123456789,
            },
            config=config,
        )

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "not found" in result["reason"].lower()

    def test_zip_unchanged_mtime_not_available(self, tmp_path):
        """Zip skill with unchanged mtime is up to date."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        # Create zip file
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record origin with matching mtime and hash
        content_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": content_hash,
            },
            config=config,
        )

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "latest" in result["reason"].lower()


class TestUpdateSkillZip:
    """Tests for update_skill with zip sources."""

    def test_update_zip_skill_success(self, tmp_path):
        """Updating zip skill when zip content changed succeeds."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nold body")

        # Create zip with new content
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: my-skill\n---\nnew body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record origin with matching hash (no local modifications)
        # The hash should match the installed skill's content
        installed_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,  # Different from current to trigger re-check
                "content_hash": installed_hash,
            },
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert result.success
        assert "my-skill" in result.updated

        # Verify content was updated
        content = (skill_dir / "SKILL.md").read_text()
        assert "new body" in content

    def test_update_zip_with_nested_dir_does_not_double_nest(self, tmp_path):
        """Zip packaged with top-level directory updates without nested output."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nold")

        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("my-skill/SKILL.md", "---\nname: my-skill\n---\nupdated content")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        installed_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": installed_hash,
            },
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert result.success
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "SKILL.md").read_text().strip().endswith("updated content")
        # No extra nested copy
        assert not (skill_dir / "my-skill" / "SKILL.md").exists()

    def test_update_zip_already_up_to_date(self, tmp_path):
        """Zip skill with matching content is already up to date."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        content = "---\nname: my-skill\n---\nsame body"
        (skill_dir / "SKILL.md").write_text(content)

        # Create zip with same content
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", content)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        content_hash = compute_content_hash(skill_dir)
        record_origin(
            "my-skill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": content_hash,
            },
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert result.success
        assert "my-skill" in result.skipped
        assert "up to date" in result.message.lower()

    def test_update_zip_with_multiple_skills_rejected(self, tmp_path):
        """Zip containing multiple skills is rejected (1 zip = 1 skill)."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "skill-a"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: skill-a\n---\nold")

        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("skill-a/SKILL.md", "---\nname: skill-a\n---\nnew")
            zf.writestr("skill-b/SKILL.md", "---\nname: skill-b\n---\nother")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "skill-a",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(skill_dir),
            },
            config=config,
        )

        result = update_skill("skill-a", config=config)

        assert not result.success
        assert "exactly one skill" in result.message.lower()

    def test_update_zip_missing_source_fails(self, tmp_path):
        """Updating zip skill with missing source fails."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "nonexistent.zip"),
                "kind": "zip",
            },
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert not result.success
        assert "not found" in result.message.lower()

    def test_update_zip_local_modified_without_force_fails(self, tmp_path):
        """Updating locally modified zip skill without force fails."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nlocally modified")

        # Create zip with different content
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: my-skill\n---\nnew content")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        # Record origin with original hash (different from current)
        record_origin(
            "my-skill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": "sha256:original_hash",
            },
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert not result.success
        assert result.local_modified is True
        assert "--force" in result.message

    def test_update_zip_local_modified_with_force_succeeds(self, tmp_path):
        """Updating locally modified zip skill with force succeeds."""
        import zipfile

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nlocally modified")

        # Create zip with different content
        zip_path = tmp_path / "my-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: my-skill\n---\nfrom zip")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        record_origin(
            "my-skill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": "sha256:original_hash",
            },
            config=config,
        )

        result = update_skill("my-skill", config=config, force=True)

        assert result.success
        assert "my-skill" in result.updated

        # Verify content was updated
        content = (skill_dir / "SKILL.md").read_text()
        assert "from zip" in content


def _make_symlink_skill_dir(base: Path, skill_body: str) -> Path:
    """Create a skill dir with SKILL.md, target.txt and link.txt -> target.txt."""
    skill_dir = base / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: my-skill\ndescription: Symlink skill\n---\n{skill_body}"
    )
    (skill_dir / "target.txt").write_text("target-content\n")
    os.symlink("target.txt", skill_dir / "link.txt")
    return skill_dir


class TestUpdateSymlinkSkills:
    """allow_symlinks behavior for update and check flows on symlink skills."""

    def _setup_local_update(self, tmp_path: Path, installed_body: str, source_body: str):
        skills_dir = tmp_path / "skills"
        installed = _make_symlink_skill_dir(skills_dir, installed_body)
        _make_symlink_skill_dir(tmp_path / "src", source_body)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "src"),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed

    def test_no_local_modification_right_after_compliant_add(self, tmp_path, require_symlink):
        """Skill added with the flag reports no local modification immediately."""
        from skillport.modules.skills import add_skill

        source_parent = tmp_path / "source"
        _make_symlink_skill_dir(source_parent, "body")

        skills_dir = tmp_path / "skills"
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        result = add_skill(
            str(source_parent),
            config=config,
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )
        assert result.success, result.message

        assert detect_local_modification("my-skill", config=config) is False

    def test_check_reports_latest_with_blob_semantics_remote_hash(
        self, tmp_path, monkeypatch, require_symlink
    ):
        """Remote tree hash built from mode 120000 blob SHAs matches the local hash."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("body\n")
        (skill_dir / "notes.txt").write_text("notes-content\n")
        os.symlink("notes.txt", skill_dir / "link")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(skill_dir),
            },
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        def mock_get_remote_tree_hash(parsed, token, path=None):
            return _tree_hash_of_entries(
                [
                    ("SKILL.md", b"body\n"),
                    ("link", b"notes.txt"),
                    ("notes.txt", b"notes-content\n"),
                ]
            )

        monkeypatch.setattr(update_module, "get_remote_tree_hash", mock_get_remote_tree_hash)

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "latest" in result["reason"].lower()

    def test_update_local_with_flag_preserves_symlink(self, tmp_path, require_symlink):
        """Update with the flag applies new content and keeps the link a symlink."""
        config, installed = self._setup_local_update(tmp_path, "old body", "new body")

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert "my-skill" in result.updated
        assert (installed / "link.txt").is_symlink()
        assert os.readlink(installed / "link.txt") == "target.txt"
        assert (
            installed / "SKILL.md"
        ).read_text() == "---\nname: my-skill\ndescription: Symlink skill\n---\nnew body"

    def test_update_local_with_flag_accepts_skill_md_symlink(self, tmp_path, require_symlink):
        config, installed = self._setup_local_update(tmp_path, "old body", "new body")
        source_skill = tmp_path / "src" / "my-skill"
        docs = source_skill / "docs"
        docs.mkdir()
        skill_md = source_skill / "SKILL.md"
        docs_skill_md = docs / "SKILL.md"
        skill_md.replace(docs_skill_md)
        skill_md.symlink_to(Path("docs") / "SKILL.md")

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success, result.message
        installed_skill_md = installed / "SKILL.md"
        assert installed_skill_md.is_symlink()
        assert os.readlink(installed_skill_md) == "docs/SKILL.md"
        assert installed_skill_md.read_text().endswith("new body")

    def test_update_all_skills_accepts_allow_symlinks(self, tmp_path, require_symlink):
        """update_all_skills forwards the flag to the per-skill update."""
        config, installed = self._setup_local_update(tmp_path, "old body", "new body")

        result = update_all_skills(config=config, allow_symlinks=True)

        assert result.success, result.message
        assert "my-skill" in result.updated
        assert (installed / "link.txt").is_symlink()

    def test_update_without_flag_rejects_unchanged_symlink_source(self, tmp_path, require_symlink):
        """Flagless update rejects a symlink source even when content is unchanged."""
        config, installed = self._setup_local_update(tmp_path, "same body", "same body")

        result = update_skill("my-skill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (
            installed / "SKILL.md"
        ).read_text() == "---\nname: my-skill\ndescription: Symlink skill\n---\nsame body"
        assert (installed / "link.txt").is_symlink()

    def test_update_without_flag_keeps_installed_skill_unchanged(self, tmp_path, require_symlink):
        """Flagless update rejection leaves the installed skill untouched."""
        config, installed = self._setup_local_update(tmp_path, "old body", "new body")

        result = update_skill("my-skill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (
            installed / "SKILL.md"
        ).read_text() == "---\nname: my-skill\ndescription: Symlink skill\n---\nold body"
        assert (installed / "target.txt").read_text() == "target-content\n"
        assert (installed / "link.txt").is_symlink()

    def test_update_with_flag_violating_source_keeps_installed_skill_unchanged(
        self, tmp_path, require_symlink
    ):
        """Flag-on update from a source with a cycling link fails without touching the install."""
        skills_dir = tmp_path / "skills"
        installed = _make_symlink_skill_dir(skills_dir, "old body")

        src_skill = tmp_path / "src" / "my-skill"
        src_skill.mkdir(parents=True)
        (src_skill / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Symlink skill\n---\nnew body"
        )
        (src_skill / "loop-a").symlink_to("loop-b")
        (src_skill / "loop-b").symlink_to("loop-a")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "src"),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "loop-a" in result.message or "loop-b" in result.message
        assert (installed / "SKILL.md").read_text().endswith("old body")
        assert (installed / "target.txt").read_text() == "target-content\n"
        assert (installed / "link.txt").is_symlink()
        # No staging directory leaks next to the installed skill
        assert not any(p.name.startswith("skillport-update-") for p in skills_dir.iterdir())

    def test_update_with_flag_rejects_hidden_link_despite_hash_match(
        self, tmp_path, require_symlink
    ):
        """A hidden link fails the update even when the source hash matches."""
        config, installed = self._setup_local_update(tmp_path, "same body", "same body")
        source_skill = tmp_path / "src" / "my-skill"
        (source_skill / ".hidden-link").symlink_to("target.txt")
        origin_before = get_origin("my-skill", config=config)

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("same body")
        assert (installed / "link.txt").is_symlink()
        assert not (installed / ".hidden-link").exists()
        assert get_origin("my-skill", config=config) == origin_before

    def test_update_with_flag_rejects_excluded_link_despite_hash_match(
        self, tmp_path, require_symlink
    ):
        """An excluded-name link fails the update even when the source hash matches."""
        config, installed = self._setup_local_update(tmp_path, "same body", "same body")
        source_skill = tmp_path / "src" / "my-skill"
        (source_skill / "node_modules").mkdir()
        (source_skill / "node_modules" / "link").symlink_to("../target.txt")
        origin_before = get_origin("my-skill", config=config)

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("same body")
        assert not (installed / "node_modules").exists()
        assert get_origin("my-skill", config=config) == origin_before

    def test_update_with_flag_rejects_directory_symlink_target(self, tmp_path, require_symlink):
        config, installed = self._setup_local_update(tmp_path, "old body", "new body")
        source_skill = tmp_path / "src" / "my-skill"
        (source_skill / "assets").mkdir()
        (source_skill / "directory-link").symlink_to("assets", target_is_directory=True)

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "regular file" in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("old body")
        assert not (installed / "directory-link").exists()

    def test_update_local_preserves_regular_file_metadata(self, tmp_path, require_symlink):
        config, installed = self._setup_local_update(tmp_path, "old body", "new body")
        source_data = tmp_path / "src" / "my-skill" / "data.txt"
        source_data.write_text("data", encoding="utf-8")
        source_data.chmod(0o640)
        fixed_mtime = 1_600_000_000_246_813_579
        os.utime(source_data, ns=(fixed_mtime, fixed_mtime))

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success, result.message
        installed_data = installed / "data.txt"
        assert stat.S_IMODE(installed_data.stat().st_mode) == 0o640
        assert installed_data.stat().st_mtime_ns == fixed_mtime

    def test_update_zip_with_flag_preserves_symlink(self, tmp_path, require_symlink):
        """Zip update with the flag installs new content and the symlink."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\n---\nold body")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\n---\nnew body")
            info = zipfile.ZipInfo("link.txt")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert "zskill" in result.updated
        assert (installed / "link.txt").is_symlink()
        assert os.readlink(installed / "link.txt") == "SKILL.md"
        assert (installed / "SKILL.md").read_text().endswith("new body")

    def test_update_zip_with_flag_preserves_inside_skill_link_chain(
        self, tmp_path, require_symlink
    ):
        """A zip link chain resolving inside the skill is installed as links."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\n---\nold body")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\n---\nnew body")
            for name, target in (("alias", "SKILL.md"), ("link", "alias")):
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                zf.writestr(info, target)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert "zskill" in result.updated
        assert (installed / "alias").is_symlink()
        assert os.readlink(installed / "alias") == "SKILL.md"
        assert (installed / "link").is_symlink()
        assert os.readlink(installed / "link") == "alias"
        assert (installed / "SKILL.md").read_text().endswith("new body")

    def test_update_github_with_flag_preserves_symlink(
        self, tmp_path, monkeypatch, require_symlink
    ):
        """GitHub update with the flag fetches with the flag and preserves the link."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: gh-skill\n---\nold body")

        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text("---\nname: gh-skill\n---\nnew body")
        assets = prepared / "assets"
        assets.mkdir()
        (assets / "m.md").write_text("m")
        os.symlink("assets/m.md", prepared / "link.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo/tree/main",
                "kind": "github",
                "path": "",
                "commit_sha": "old1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        recorded: dict = {}

        def fake_fetch(url, allow_symlinks=False):
            recorded["allow_symlinks"] = allow_symlinks
            return SimpleNamespace(extracted_path=prepared, commit_sha="new1234abcdef")

        monkeypatch.setattr(
            update_module, "get_remote_tree_hash", lambda parsed, token, path=None: "sha256:remote"
        )
        monkeypatch.setattr(update_module, "fetch_github_source_with_info", fake_fetch)

        result = update_skill("gh-skill", config=config, allow_symlinks=True)

        # The flag must reach the GitHub fetch handoff
        assert recorded["allow_symlinks"] is True
        assert result.success, result.message
        assert (installed / "link.md").is_symlink()
        assert os.readlink(installed / "link.md") == "assets/m.md"
        assert (installed / "SKILL.md").read_text().endswith("new body")
        assert not prepared.exists()
        assert not (prepared.parent / "gh-skill").exists()

    def test_update_github_cleans_renamed_root_after_copy_failure(
        self, tmp_path, monkeypatch, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nold")

        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nnew")
        (prepared / "loop-a").symlink_to("loop-b")
        (prepared / "loop-b").symlink_to("loop-a")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "",
                "commit_sha": "old1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module, "get_remote_tree_hash", lambda parsed, token, path=None: "sha256:remote"
        )
        monkeypatch.setattr(
            update_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="new1234abcdef"
            ),
        )

        result = update_skill("gh-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert not prepared.exists()
        assert not (prepared.parent / "gh-skill").exists()
        assert (installed / "SKILL.md").read_text().endswith("old")

    def test_update_github_flagless_fetch_receives_default_false(
        self, tmp_path, monkeypatch, require_symlink
    ):
        """Without the flag, the GitHub fetch handoff receives the default False."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nold body")

        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nnew body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "",
                "commit_sha": "old1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.internal.github import ParsedGitHubURL
        from skillport.modules.skills.public import update as update_module

        recorded: dict = {}

        def fake_fetch(url, allow_symlinks=False):
            recorded["allow_symlinks"] = allow_symlinks
            return SimpleNamespace(extracted_path=prepared, commit_sha="new1234abcdef")

        monkeypatch.setattr(
            update_module,
            "parse_github_url",
            lambda url, **kwargs: ParsedGitHubURL(owner="user", repo="repo", ref="main", path=""),
        )
        monkeypatch.setattr(
            update_module, "get_remote_tree_hash", lambda parsed, token, path=None: "sha256:remote"
        )
        monkeypatch.setattr(
            update_module, "get_remote_tree_symlinks", lambda parsed, token, path: []
        )
        monkeypatch.setattr(update_module, "fetch_github_source_with_info", fake_fetch)

        result = update_skill("gh-skill", config=config)

        assert result.success, result.message
        assert recorded["allow_symlinks"] is False
        assert (installed / "SKILL.md").read_text().endswith("new body")


class TestExcludedNameSymlinkUpdate:
    """A flagless update rejects a source symlink named like an EXCLUDE_NAMES entry.

    The local source snapshot materializes excluded names before the source
    symlink gate, so ``.env -> SKILL.md`` fails the update and leaves the
    installed skill and origin untouched; a regular file with the same name
    stays ignored and the update applies.
    """

    def _setup_local_update(self, tmp_path: Path, *, symlink_env: bool) -> tuple[Config, Path]:
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "skill-a"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: S\n---\nold", encoding="utf-8"
        )
        (installed / "keep.txt").write_text("keep", encoding="utf-8")

        source = tmp_path / "src" / "skill-a"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: S\n---\nnew", encoding="utf-8"
        )
        (source / "keep.txt").write_text("keep", encoding="utf-8")
        if symlink_env:
            (source / ".env").symlink_to("SKILL.md")
        else:
            (source / ".env").write_text("ignored", encoding="utf-8")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "skill-a",
            {
                "source": str(source),
                "kind": "local",
                "path": "",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed

    def test_flagless_update_rejects_excluded_name_symlink(
        self, tmp_path: Path, require_symlink
    ):
        config, installed = self._setup_local_update(tmp_path, symlink_env=True)
        origin_before = get_origin("skill-a", config=config)

        result = update_skill("skill-a", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("old")
        assert (installed / "keep.txt").read_text(encoding="utf-8") == "keep"
        assert not (installed / ".env").exists()
        assert get_origin("skill-a", config=config) == origin_before

    def test_flagless_update_ignores_excluded_name_regular_file(self, tmp_path: Path):
        config, installed = self._setup_local_update(tmp_path, symlink_env=False)

        result = update_skill("skill-a", config=config)

        assert result.success, result.message
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("new")
        assert (installed / "keep.txt").read_text(encoding="utf-8") == "keep"
        assert not (installed / ".env").exists()


class TestUpdateSourcePreflight:
    """Source symlinks are rejected before hash/mtime early-exit paths."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_local_source_root_symlink_is_rejected(self, tmp_path, allow: bool, require_symlink):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        installed_content = "---\nname: my-skill\ndescription: D\n---\ninstalled"
        (installed / "SKILL.md").write_text(installed_content)

        real_source = tmp_path / "real" / "my-skill"
        real_source.mkdir(parents=True)
        (real_source / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nsource")
        source = tmp_path / "src"
        source.symlink_to(real_source)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {"source": str(source), "kind": "local", "path": ""},
            config=config,
        )

        result = update_skill("my-skill", config=config, allow_symlinks=allow)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text() == installed_content

    def test_flagless_local_root_symlink_source_rejected_despite_hash_match(
        self, tmp_path, require_symlink
    ):
        """A symlinked local source root is rejected even when content hashes match."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nsame body")

        real_source = tmp_path / "real" / "my-skill"
        real_source.mkdir(parents=True)
        (real_source / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nsame body")

        source = tmp_path / "src"
        source.mkdir()
        (source / "my-skill").symlink_to(real_source)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(source),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("my-skill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("same body")

    def test_local_origin_path_symlink_ancestor_is_rejected(self, tmp_path, require_symlink):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: my-skill\ndescription: D\n---\ninstalled")

        outside = tmp_path / "outside" / "my-skill"
        outside.mkdir(parents=True)
        (outside / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\noutside")
        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(outside.parent, target_is_directory=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(source),
                "kind": "local",
                "path": "alias/my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert (outside / "SKILL.md").read_text().endswith("outside")

    def test_local_origin_path_dotdot_cannot_bypass_symlink_ancestor(
        self, tmp_path, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: my-skill\ndescription: D\n---\ninstalled")

        outside = tmp_path / "outside"
        (outside / "branch").mkdir(parents=True)
        external_skill = outside / "my-skill"
        external_skill.mkdir()
        (external_skill / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\noutside"
        )
        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(outside / "branch", target_is_directory=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(source),
                "kind": "local",
                "path": "alias/../my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert (external_skill / "SKILL.md").read_text().endswith("outside")

    def test_flagless_zip_symlink_source_rejected_despite_mtime_match(self, tmp_path):
        """A flagless zip update cannot early-exit 'up to date' on a symlinked zip."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("link.txt")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        # mtime fast path would report up-to-date without extraction
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")

    def test_zip_with_flag_early_exit_is_unaffected(self, tmp_path, require_symlink):
        """With the flag, an unchanged symlinked zip still reports up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("link.txt")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["zskill"]

    def test_zip_with_flag_hidden_symlink_rejected_despite_mtime_match(
        self, tmp_path, require_symlink
    ):
        """With the flag, a zip with a hidden link cannot early-exit as up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo(".env")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert not (installed / ".env").exists()
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_visible_symlink_entry_still_early_exits(
        self, tmp_path, require_symlink
    ):
        """With the flag, a visible link entry keeps the mtime fast path."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["zskill"]
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert not (installed / "link").exists()
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_ignores_hidden_link_outside_the_selected_skill(
        self, tmp_path, require_symlink
    ):
        """A hidden link outside the selected skill cannot fail its up-to-date update."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("zskill/SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo(".env-link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "zskill/SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "zskill",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["zskill"]
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_rejects_hidden_link_inside_the_selected_skill(
        self, tmp_path, require_symlink
    ):
        """The same hidden link inside the selected skill still fails before up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("zskill/SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("zskill/.env-link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "zskill",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_visible_link_to_hidden_target_rejected_despite_mtime_match(
        self, tmp_path, require_symlink
    ):
        """A visible link whose target is hidden cannot early-exit as up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, ".env")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert not (installed / "link").exists()
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_hidden_target_via_link_chain_rejected_despite_mtime_match(
        self, tmp_path, require_symlink
    ):
        """A chain reaching a hidden target cannot early-exit as up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            for name, target in (("link", "alias"), ("alias", ".env")):
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                zf.writestr(info, target)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert not (installed / "link").exists()
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_backslash_excluded_path_rejected_despite_mtime_match(
        self, tmp_path, require_symlink
    ):
        """A backslash-separated excluded link path cannot early-exit as up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("node_modules\\link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert not (installed / "node_modules").exists()
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_with_flag_unnormalizable_symlink_entry_rejected_despite_mtime_match(
        self, tmp_path
    ):
        """A link entry whose name cannot be normalized fails before the fast path."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("/absolute/link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "/absolute/link" in result.message
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert get_origin("zskill", config=config) == origin_before

    @pytest.mark.parametrize(
        ("links", "message_part"),
        [
            ([("link", "missing-target")], "does not exist"),
            ([("link", "/outside.txt")], "must be relative"),
            ([("link", "../outside.txt")], "outside the skill directory"),
            ([("link", "alias"), ("alias", "link")], "cycles"),
        ],
        ids=["missing-target", "absolute-path", "outside-skill", "cycle-chain"],
    )
    def test_zip_with_flag_invalid_symlink_target_rejected_despite_mtime_match(
        self, tmp_path, links, message_part, require_symlink
    ):
        """Every non-InsideSkill target fails the skill before the mtime fast path."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")
            for name, target in links:
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o120777 << 16
                zf.writestr(info, target)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        # mtime/hash fast path would report up-to-date without validating links
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert message_part in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before

    def test_zip_origin_path_symlink_ancestor_is_rejected(self, tmp_path, require_symlink):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: zskill\ndescription: D\n---\ninstalled")

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\noutside")
        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("alias")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, str(outside))

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "alias",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert (outside / "SKILL.md").read_text().endswith("outside")

    def test_zip_origin_path_dotdot_cannot_bypass_symlink_ancestor(self, tmp_path, require_symlink):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: zskill\ndescription: D\n---\ninstalled")

        outside = tmp_path / "outside"
        (outside / "branch").mkdir(parents=True)
        external_skill = outside / "zskill"
        external_skill.mkdir()
        (external_skill / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\noutside")
        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\narchive")
            info = zipfile.ZipInfo("alias")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, str(outside / "branch"))

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "alias/../zskill",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert (external_skill / "SKILL.md").read_text().endswith("outside")

    def test_flagless_github_symlink_remote_rejected_despite_hash_match(
        self, tmp_path, monkeypatch
    ):
        """A flagless GitHub update cannot early-exit when the remote tree has symlinks."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "gh-skill",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.internal.github import ParsedGitHubURL
        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module,
            "parse_github_url",
            lambda url, **kwargs: ParsedGitHubURL(owner="user", repo="repo", ref="main", path=""),
        )
        # Remote hash matches the installed hash: only the source gate can reject
        monkeypatch.setattr(
            update_module,
            "get_remote_tree_hash",
            lambda parsed, token, path=None: compute_content_hash(installed),
        )
        monkeypatch.setattr(
            update_module, "get_remote_tree_symlinks", lambda parsed, token, path: ["link"]
        )

        result = update_skill("gh-skill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")

    def test_github_with_flag_early_exit_is_unaffected(self, tmp_path, monkeypatch):
        """With the flag, a matching remote hash still reports up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "gh-skill",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.internal.github import ParsedGitHubURL
        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module,
            "parse_github_url",
            lambda url, **kwargs: ParsedGitHubURL(owner="user", repo="repo", ref="main", path=""),
        )
        monkeypatch.setattr(
            update_module,
            "get_remote_tree_hash",
            lambda parsed, token, path=None: compute_content_hash(installed),
        )
        monkeypatch.setattr(
            update_module, "get_remote_tree_symlinks", lambda parsed, token, path: ["link"]
        )

        result = update_skill("gh-skill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["gh-skill"]

    def test_github_with_flag_hidden_symlink_remote_rejected_despite_hash_match(
        self, tmp_path, monkeypatch
    ):
        """With the flag, a remote tree with a hidden link cannot report up to date."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "gh-skill",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("gh-skill", config=config)

        from skillport.modules.skills.internal.github import ParsedGitHubURL
        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module,
            "parse_github_url",
            lambda url, **kwargs: ParsedGitHubURL(owner="user", repo="repo", ref="main", path=""),
        )
        monkeypatch.setattr(
            update_module,
            "get_remote_tree_hash",
            lambda parsed, token, path=None: compute_content_hash(installed),
        )
        monkeypatch.setattr(
            update_module, "get_remote_tree_symlinks", lambda parsed, token, path: [".hidden-link"]
        )

        result = update_skill("gh-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert (installed / "SKILL.md").read_text().endswith("body")
        assert get_origin("gh-skill", config=config) == origin_before

    def test_github_origin_path_symlink_ancestor_is_rejected(
        self, tmp_path, monkeypatch, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: gh-skill\ndescription: G\n---\ninstalled")

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: gh-skill\ndescription: G\n---\noutside")
        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "alias").symlink_to(outside, target_is_directory=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "alias",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module, "get_remote_tree_hash", lambda parsed, token, path=None: "sha256:remote"
        )
        monkeypatch.setattr(
            update_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="new1234abcdef"
            ),
        )

        result = update_skill("gh-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert (outside / "SKILL.md").read_text().endswith("outside")
        assert not prepared.exists()

    def test_github_origin_path_dotdot_cannot_bypass_symlink_ancestor(
        self, tmp_path, monkeypatch, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-skill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: gh-skill\ndescription: G\n---\ninstalled")

        outside = tmp_path / "outside"
        (outside / "branch").mkdir(parents=True)
        external_skill = outside / "gh-skill"
        external_skill.mkdir()
        (external_skill / "SKILL.md").write_text(
            "---\nname: gh-skill\ndescription: G\n---\noutside"
        )
        prepared = tmp_path / "gh-extract"
        prepared.mkdir()
        (prepared / "alias").symlink_to(outside / "branch", target_is_directory=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-skill",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "alias/../gh-skill",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module, "get_remote_tree_hash", lambda parsed, token, path=None: "sha256:remote"
        )
        monkeypatch.setattr(
            update_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="new1234abcdef"
            ),
        )

        result = update_skill("gh-skill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert (external_skill / "SKILL.md").read_text().endswith("outside")
        assert not prepared.exists()


class TestZipRootSymlinkDirectoryCandidates:
    """A root hidden/excluded symlink directory is not a skill candidate in a ZIP update.

    detect_skills() reports a symlinked skill root as a per-skill error
    candidate. A link whose own path is hidden or in EXCLUDE_NAMES follows the
    normal exclusion rules instead, so it must not fail the single-skill
    selection; the same link inside the selected skill is still rejected.
    """

    @staticmethod
    def _setup(
        tmp_path: Path,
        *,
        link_name: str,
        link_target: str,
        installed_body: str,
        source_body: str,
        source_mtime: int | None = None,
    ) -> tuple[Config, Path]:
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(installed_body)

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("zskill/SKILL.md", source_body)
            info = zipfile.ZipInfo(link_name)
            info.external_attr = 0o120777 << 16
            zf.writestr(info, link_target)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "zskill",
                "source_mtime": (
                    zip_path.stat().st_mtime_ns if source_mtime is None else source_mtime
                ),
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed

    def test_hidden_directory_link_at_root_is_ignored_and_update_skips(
        self, tmp_path, require_symlink
    ):
        """A root .env-link -> zskill/ is not a candidate, so the matching skill skips."""
        body = "---\nname: zskill\ndescription: D\n---\nbody"
        config, installed = self._setup(
            tmp_path,
            link_name=".env-link",
            link_target="zskill/",
            installed_body=body,
            source_body=body,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["zskill"]
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before

    def test_excluded_directory_link_at_root_is_ignored_and_skill_updates(
        self, tmp_path, require_symlink
    ):
        """A root node_modules -> zskill/ is not a candidate, so the changed skill updates."""
        config, installed = self._setup(
            tmp_path,
            link_name="node_modules",
            link_target="zskill/",
            installed_body="---\nname: zskill\ndescription: D\n---\nold body",
            source_body="---\nname: zskill\ndescription: D\n---\nnew body",
            source_mtime=0,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.updated == ["zskill"]
        assert (installed / "SKILL.md").read_text().endswith("new body")
        assert "node_modules" not in _installed_tree_state(installed)

    def test_hidden_link_inside_the_selected_skill_is_rejected(self, tmp_path, require_symlink):
        """A hidden link inside zskill/ fails the update before the up-to-date check."""
        body = "---\nname: zskill\ndescription: D\n---\nbody"
        config, installed = self._setup(
            tmp_path,
            link_name="zskill/.env-link",
            link_target="SKILL.md",
            installed_body=body,
            source_body=body,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before

    def test_excluded_link_inside_the_selected_skill_is_rejected(self, tmp_path, require_symlink):
        """An excluded-name link inside zskill/ fails the update before the up-to-date check."""
        body = "---\nname: zskill\ndescription: D\n---\nbody"
        config, installed = self._setup(
            tmp_path,
            link_name="zskill/node_modules",
            link_target="SKILL.md",
            installed_body=body,
            source_body=body,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "hidden or excluded" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before

    def test_visible_directory_link_at_root_still_fails_single_skill_check(
        self, tmp_path, require_symlink
    ):
        """A visible symlink root keeps its existing rejection."""
        body = "---\nname: zskill\ndescription: D\n---\nbody"
        config, installed = self._setup(
            tmp_path,
            link_name="alias",
            link_target="zskill/",
            installed_body=body,
            source_body=body,
        )
        origin_before = get_origin("zskill", config=config)
        installed_before = _installed_tree_state(installed)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "exactly one skill" in result.message.lower()
        assert _installed_tree_state(installed) == installed_before
        assert get_origin("zskill", config=config) == origin_before


class TestZipSourceSymlinkUpdateCheck:
    """A symlinked zip source path is rejected before mtime/hash fast paths."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_update_symlinked_zip_source_rejected_despite_mtime_match(
        self, tmp_path, allow: bool, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: zskill\ndescription: D\n---\nbody")

        outside = tmp_path / "outside" / "zskill.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")

        alias = tmp_path / "zskill.zip"
        alias.symlink_to(outside)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        # Stored mtime/hash match the followed source: without the gate the
        # mtime fast path would early-exit as up to date
        record_origin(
            "zskill",
            {
                "source": str(alias),
                "kind": "zip",
                "source_mtime": alias.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=allow)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert "up to date" not in result.message.lower()
        assert installed_file.read_text().endswith("body")
        assert get_origin("zskill", config=config) == origin_before

    def test_update_zip_source_symlinked_ancestor_is_rejected(self, tmp_path, require_symlink):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text("---\nname: zskill\ndescription: D\n---\ninstalled")

        outside = tmp_path / "outside" / "zskill.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nsource")

        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(outside.parent, target_is_directory=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(source / "alias" / "zskill.zip"),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)
        zip_bytes_before = outside.read_bytes()

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text().endswith("installed")
        assert get_origin("zskill", config=config) == origin_before
        assert outside.read_bytes() == zip_bytes_before

    def test_check_symlinked_zip_source_is_not_updatable_despite_mtime_match(
        self, tmp_path, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        outside = tmp_path / "outside" / "zskill.zip"
        outside.parent.mkdir(parents=True)
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")

        alias = tmp_path / "zskill.zip"
        alias.symlink_to(outside)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(alias),
                "kind": "zip",
                "source_mtime": alias.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = check_update_available("zskill", config=config)

        assert result["available"] is False
        assert "symlink" in result["reason"].lower()
        assert "latest" not in result["reason"].lower()
        assert get_origin("zskill", config=config) == origin_before

    def test_regular_zip_source_check_still_reports_latest(self, tmp_path):
        """A regular zip source keeps the mtime fast path (latest) on check."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zskill\ndescription: D\n---\nbody")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nbody")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is False
        assert "latest" in result["reason"].lower()

    def test_trailing_slash_symlink_entry_is_rejected_before_mtime_fast_path(
        self, tmp_path, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        content = "---\nname: zskill\ndescription: D\n---\nbody"
        (installed / "SKILL.md").write_text(content)

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", content)
            info = zipfile.ZipInfo("link/")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert "up to date" not in result.message.lower()

    def test_trailing_slash_symlink_entry_is_preserved_with_flag(self, tmp_path, require_symlink):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        old_content = "---\nname: zskill\ndescription: D\n---\nold"
        (installed / "SKILL.md").write_text(old_content)

        zip_path = tmp_path / "zskill.zip"
        new_content = "---\nname: zskill\ndescription: D\n---\nnew"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", new_content)
            info = zipfile.ZipInfo("link/")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert (installed / "link").is_symlink()
        assert os.readlink(installed / "link") == "SKILL.md"

    def test_directory_suffix_symlink_target_is_rejected_with_rollback(
        self, tmp_path, require_symlink
    ):
        """A zip target naming a directory (``SKILL.md/``) fails and keeps the installed skill."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        old_content = "---\nname: zskill\ndescription: D\n---\nold"
        (installed / "SKILL.md").write_text(old_content)
        installed_before = _installed_tree_state(installed)

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nnew")
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md/")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "symlink target is invalid" in result.message.lower()
        assert _installed_tree_state(installed) == installed_before
        assert (installed / "SKILL.md").read_text() == old_content
        assert get_origin("zskill", config=config) == origin_before


class TestZipUpdateTemporaryResourceCleanup:
    """Every zip update terminal releases its extraction, snapshot, staging, and backup resources."""

    @staticmethod
    def _stray_resources(controlled_tmp: Path, skills_dir: Path) -> list[str]:
        stray = sorted(path.name for path in controlled_tmp.iterdir())
        stray += sorted(
            path.name
            for path in skills_dir.iterdir()
            if path.name.startswith("skillport-") or "-backup-" in path.name
        )
        return stray

    @staticmethod
    def _setup_installed(tmp_path: Path) -> tuple[Config, Path, Path, str]:
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        body = "---\nname: zskill\ndescription: D\n---\nbody"
        (installed / "SKILL.md").write_text(body)
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        return config, skills_dir, installed, body

    @staticmethod
    def _write_zip(zip_path: Path, skill_md: str, link_target: str) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", skill_md)
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, link_target)

    def test_invalid_link_terminal_releases_temporary_resources(
        self, tmp_path, monkeypatch, require_symlink
    ):
        controlled_tmp = tmp_path / "controlled-tmp"
        controlled_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))

        config, skills_dir, installed, body = self._setup_installed(tmp_path)
        zip_path = tmp_path / "zskill.zip"
        self._write_zip(zip_path, body, "missing-target")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success is False
        assert "does not exist" in result.message.lower()
        assert self._stray_resources(controlled_tmp, skills_dir) == []

    def test_skipped_update_terminal_releases_temporary_resources(
        self, tmp_path, monkeypatch, require_symlink
    ):
        controlled_tmp = tmp_path / "controlled-tmp"
        controlled_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))

        config, skills_dir, installed, body = self._setup_installed(tmp_path)
        zip_path = tmp_path / "zskill.zip"
        self._write_zip(zip_path, body, "SKILL.md")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["zskill"]
        assert self._stray_resources(controlled_tmp, skills_dir) == []

    def test_applied_update_terminal_releases_temporary_resources(
        self, tmp_path, monkeypatch, require_symlink
    ):
        controlled_tmp = tmp_path / "controlled-tmp"
        controlled_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))

        config, skills_dir, installed, _ = self._setup_installed(tmp_path)
        zip_path = tmp_path / "zskill.zip"
        new_body = "---\nname: zskill\ndescription: D\n---\nnew body"
        self._write_zip(zip_path, new_body, "SKILL.md")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.updated == ["zskill"]
        assert (installed / "link").is_symlink()
        assert self._stray_resources(controlled_tmp, skills_dir) == []

    def test_root_hidden_directory_link_skip_releases_temporary_resources(
        self, tmp_path, monkeypatch, require_symlink
    ):
        controlled_tmp = tmp_path / "controlled-tmp"
        controlled_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))

        config, skills_dir, installed, body = self._setup_installed(tmp_path)
        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("zskill/SKILL.md", body)
            info = zipfile.ZipInfo(".env-link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "zskill/")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "zskill",
                "source_mtime": zip_path.stat().st_mtime_ns,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert result.skipped == ["zskill"]
        assert self._stray_resources(controlled_tmp, skills_dir) == []


class TestUpdateLocalHardlink:
    """Local update rejects st_nlink > 1 sources only when the flag is set."""

    @staticmethod
    def _setup(tmp_path: Path):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nold body")

        source_skill = tmp_path / "src" / "my-skill"
        source_skill.mkdir(parents=True)
        (source_skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nnew body")
        original = tmp_path / "original.txt"
        original.write_text("shared content", encoding="utf-8")
        os.link(original, source_skill / "data.txt")
        assert (source_skill / "data.txt").stat().st_nlink == 2

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "src"),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed, original

    def test_update_with_flag_rejects_hardlink_source(self, tmp_path):
        config, installed, original = self._setup(tmp_path)
        origin_before = get_origin("my-skill", config=config)

        result = update_skill("my-skill", config=config, allow_symlinks=True)

        assert result.success is False
        message = result.message.lower()
        assert "hardlink" in message or "hard link" in message
        assert (installed / "SKILL.md").read_text().endswith("old body")
        assert not (installed / "data.txt").exists()
        assert get_origin("my-skill", config=config) == origin_before
        assert original.read_text(encoding="utf-8") == "shared content"

    def test_update_without_flag_copies_hardlink_as_regular_file(self, tmp_path):
        config, installed, original = self._setup(tmp_path)

        result = update_skill("my-skill", config=config)

        assert result.success, result.message
        assert "my-skill" in result.updated
        assert (installed / "SKILL.md").read_text().endswith("new body")
        copied = installed / "data.txt"
        assert copied.exists()
        assert not copied.is_symlink()
        assert copied.read_text(encoding="utf-8") == "shared content"
        assert copied.stat().st_nlink == 1
        assert original.read_text(encoding="utf-8") == "shared content"


class TestUpdateTransaction:
    @staticmethod
    def _setup(tmp_path: Path):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\ninstalled")

        source = tmp_path / "source" / "my-skill"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nupdated")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "source"),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
                "update_history": [{"from_commit": "before", "to_commit": "current"}],
            },
            config=config,
        )
        return config, installed

    def test_destination_commit_failure_restores_tree_and_origin(self, tmp_path: Path, monkeypatch):
        config, installed = self._setup(tmp_path)
        before_origin = get_origin("my-skill", config=config)

        from skillport.modules.skills.public import update as update_module

        original_rename = Path.rename

        def fail_staged_rename(path, target):
            if path.parent.name.startswith("skillport-update-"):
                raise OSError("destination commit failed")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_staged_rename)

        result = update_module.update_skill("my-skill", config=config)

        assert result.success is False
        assert installed.joinpath("SKILL.md").read_text().endswith("installed")
        assert get_origin("my-skill", config=config) == before_origin

    def test_metadata_failure_restores_tree_and_origin(self, tmp_path: Path, monkeypatch):
        config, installed = self._setup(tmp_path)
        before_origin = get_origin("my-skill", config=config)

        from skillport.modules.skills.public import update as update_module

        def fail_update_origin(*args, **kwargs):
            raise OSError("metadata commit failed")

        monkeypatch.setattr(update_module, "update_origin", fail_update_origin)

        result = update_module.update_skill("my-skill", config=config)

        assert result.success is False
        assert installed.joinpath("SKILL.md").read_text().endswith("installed")
        assert get_origin("my-skill", config=config) == before_origin

    def test_successful_update_leaves_no_backup(self, tmp_path: Path):
        """A normal update removes its temporary backup from the skills directory."""
        config, installed = self._setup(tmp_path)

        from skillport.modules.skills.public import update as update_module

        result = update_module.update_skill("my-skill", config=config)

        assert result.success, result.message
        assert installed.joinpath("SKILL.md").read_text().endswith("updated")
        assert list((tmp_path / "skills").glob(".my-skill-backup-*")) == []

    def test_rollback_failure_keeps_backup_and_reports_recovery_path(
        self, tmp_path: Path, monkeypatch
    ):
        """A failed destination rollback keeps the previous content in its backup."""
        config, installed = self._setup(tmp_path)
        before_origin = get_origin("my-skill", config=config)

        from skillport.modules.skills.public import update as update_module

        original_rename = Path.rename

        def fail_staged_and_rollback_rename(path, target):
            if path.parent.name.startswith("skillport-update-"):
                raise OSError("destination commit failed")
            if path.name.startswith(".my-skill-backup-"):
                raise OSError("destination rollback failed")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_staged_and_rollback_rename)

        result = update_module.update_skill("my-skill", config=config)

        assert result.success is False
        assert "destination rollback failed" in result.message
        backups = list((tmp_path / "skills").glob(".my-skill-backup-*"))
        assert len(backups) == 1
        backup = backups[0]
        assert str(backup) in result.message
        assert backup.joinpath("SKILL.md").read_text().endswith("installed")
        assert not installed.exists()
        assert get_origin("my-skill", config=config) == before_origin


class TestInstalledRootSymlinkPreflight:
    def _setup(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nexternal")
        (skills_dir).mkdir()
        (skills_dir / "my-skill").symlink_to(outside, target_is_directory=True)
        source = tmp_path / "source" / "my-skill"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("---\nname: my-skill\ndescription: D\n---\nsource")
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {"source": str(tmp_path / "source"), "kind": "local", "path": "my-skill"},
            config=config,
        )
        return config, outside

    def test_single_update_rejects_installed_root_symlink(self, tmp_path, require_symlink):
        config, outside = self._setup(tmp_path)

        result = update_skill("my-skill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (outside / "SKILL.md").read_text().endswith("external")

    def test_update_all_rejects_installed_root_symlink(self, tmp_path, require_symlink):
        config, outside = self._setup(tmp_path)

        result = update_all_skills(config=config)

        assert result.success is False
        assert any("my-skill" in error for error in result.errors)
        assert (outside / "SKILL.md").read_text().endswith("external")

    def test_single_update_rejects_symlinked_namespace_before_origin_access(
        self, tmp_path, monkeypatch, require_symlink
    ):
        skills_dir = tmp_path / "skills"
        outside = tmp_path / "outside"
        (outside / "skill").mkdir(parents=True)
        installed_file = outside / "skill" / "SKILL.md"
        installed_file.write_text(
            "---\nname: skill\ndescription: D\n---\nexternal", encoding="utf-8"
        )
        skills_dir.mkdir()
        (skills_dir / "namespace").symlink_to(outside, target_is_directory=True)
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "namespace/skill",
            {"source": str(tmp_path / "source"), "kind": "local", "path": "skill"},
            config=config,
        )

        from skillport.modules.skills.public import update as update_module

        monkeypatch.setattr(
            update_module,
            "get_origin",
            lambda *args, **kwargs: pytest.fail("origin access must not occur"),
        )

        result = update_module.update_skill("namespace/skill", config=config)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text(encoding="utf-8").endswith("external")

    def test_update_all_continues_after_symlinked_namespace(self, tmp_path, require_symlink):
        skills_dir = tmp_path / "skills"
        outside = tmp_path / "outside"
        (outside / "skill").mkdir(parents=True)
        outside_file = outside / "skill" / "SKILL.md"
        outside_file.write_text("---\nname: skill\ndescription: D\n---\nexternal", encoding="utf-8")
        good = skills_dir / "good"
        good.mkdir(parents=True)
        good_file = good / "SKILL.md"
        good_file.write_text("---\nname: good\ndescription: D\n---\ngood", encoding="utf-8")
        good_source = tmp_path / "good-source" / "good"
        good_source.mkdir(parents=True)
        (good_source / "SKILL.md").write_text(
            "---\nname: good\ndescription: D\n---\ngood", encoding="utf-8"
        )
        skills_dir.mkdir(exist_ok=True)
        (skills_dir / "namespace").symlink_to(outside, target_is_directory=True)

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "namespace/skill",
            {"source": str(tmp_path / "source"), "kind": "local", "path": "skill"},
            config=config,
        )
        record_origin(
            "good",
            {
                "source": str(tmp_path / "good-source"),
                "kind": "local",
                "path": "good",
                "content_hash": compute_content_hash(good),
            },
            config=config,
        )

        result = update_all_skills(config=config)

        assert result.success is False
        assert any("namespace/skill" in error for error in result.errors)
        assert result.skipped == ["good"]
        assert outside_file.read_text(encoding="utf-8").endswith("external")


class TestZipCheckHashPath:
    """check_update_available hashes symlinked zip sources with blob semantics."""

    def test_check_zip_after_mtime_change_reports_latest(self, tmp_path, require_symlink):
        """A compliant symlinked zip whose mtime changed is judged by content hash, not rejected."""
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zs"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text("---\nname: zs\ndescription: D\n---\nbody")
        os.symlink("SKILL.md", installed / "link.txt")

        zip_path = tmp_path / "zs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zs\ndescription: D\n---\nbody")
            info = zipfile.ZipInfo("link.txt")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        # Stored mtime differs from the current one, forcing the extraction path
        old_mtime = zip_path.stat().st_mtime_ns - 10_000_000_000
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zs",
            {
                "source": str(zip_path),
                "kind": "zip",
                "source_mtime": old_mtime,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = check_update_available("zs", config=config)

        assert result["available"] is False
        assert "latest" in result["reason"].lower()
        assert "symlink detected" not in result["reason"].lower()

    @staticmethod
    def _check_setup(
        tmp_path: Path,
        *,
        link_name: str,
        link_target: str,
        installed_body: str,
        source_body: str,
    ) -> Config:
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(installed_body, encoding="utf-8")

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("zskill/SKILL.md", source_body)
            info = zipfile.ZipInfo(link_name)
            info.external_attr = 0o120777 << 16
            zf.writestr(info, link_target)

        # Stored mtime differs from the current one, forcing the extraction path
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "zskill",
                "source_mtime": zip_path.stat().st_mtime_ns - 10_000_000_000,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config

    def test_root_hidden_link_is_ignored_and_changed_skill_is_available(
        self, tmp_path, require_symlink
    ):
        """A root .env-link is not a candidate, so the changed skill is available."""
        config = self._check_setup(
            tmp_path,
            link_name=".env-link",
            link_target="zskill/",
            installed_body="---\nname: zskill\ndescription: D\n---\nold body",
            source_body="---\nname: zskill\ndescription: D\n---\nnew body",
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is True
        assert "found 2" not in result["reason"]

    def test_root_excluded_link_is_ignored_and_changed_skill_is_available(
        self, tmp_path, require_symlink
    ):
        """A root node_modules link is not a candidate, so the changed skill is available."""
        config = self._check_setup(
            tmp_path,
            link_name="node_modules",
            link_target="zskill/",
            installed_body="---\nname: zskill\ndescription: D\n---\nold body",
            source_body="---\nname: zskill\ndescription: D\n---\nnew body",
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is True
        assert "found 2" not in result["reason"]

    def test_root_visible_link_still_fails_single_skill_check(self, tmp_path, require_symlink):
        """A visible symlink root keeps its existing rejection in the check path."""
        config = self._check_setup(
            tmp_path,
            link_name="alias",
            link_target="zskill/",
            installed_body="---\nname: zskill\ndescription: D\n---\nold body",
            source_body="---\nname: zskill\ndescription: D\n---\nnew body",
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is False
        assert "found 2" in result["reason"]


class TestGithubAddOriginCheckContinuousFlow:
    """add -> origin hash -> update --check -> local modification in one flow."""

    def test_add_check_and_modification_are_consistent(
        self, tmp_path, monkeypatch, require_symlink
    ):
        """A deterministic GitHub add makes the immediate check report latest and unmodified."""
        from skillport.modules.skills import add_skill
        from skillport.modules.skills.public import add as add_module
        from skillport.modules.skills.public import update as update_module

        prepared = tmp_path / "gh-tree"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text(
            "---\nname: gh-cont\ndescription: C\n---\ngh body", encoding="utf-8"
        )
        (prepared / "notes.txt").write_text("notes-content\n", encoding="utf-8")
        os.symlink("notes.txt", prepared / "link")

        def fake_fetch(url, allow_symlinks=False):
            return SimpleNamespace(extracted_path=prepared, commit_sha="abc1234")

        monkeypatch.setattr(add_module, "fetch_github_source_with_info", fake_fetch)

        skills_dir = tmp_path / "skills"
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        add_result = add_skill(
            "https://github.com/user/repo",
            config=config,
            force=False,
            keep_structure=False,
            allow_symlinks=True,
        )
        assert add_result.success, add_result.message
        assert "gh-cont" in add_result.added

        # The skill was installed with the preserved link
        installed = skills_dir / "gh-cont"
        assert (installed / "link").is_symlink()

        # Remote tree hash built with mode 120000 blob semantics over the same tree
        monkeypatch.setattr(
            update_module,
            "get_remote_tree_hash",
            lambda parsed, token, path=None: _tree_hash_of_entries(
                [
                    ("SKILL.md", b"---\nname: gh-cont\ndescription: C\n---\ngh body"),
                    ("link", b"notes.txt"),
                    ("notes.txt", b"notes-content\n"),
                ]
            ),
        )

        check = check_update_available("gh-cont", config=config)
        assert check["available"] is False
        assert "latest" in check["reason"].lower()
        assert detect_local_modification("gh-cont", config=config) is False


class TestRootSkillOriginPath:
    """An explicit empty origin path addresses the source root, not the skill name."""

    def test_root_local_update_with_flag_preserves_symlink(self, tmp_path, require_symlink):
        """A skill added from a source root updates in place and keeps the link a symlink."""
        from skillport.modules.skills import add_skill

        source = tmp_path / "root-skill"
        (source / "assets").mkdir(parents=True)
        (source / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nold body", encoding="utf-8"
        )
        (source / "assets" / "manual.md").write_text("v1\n", encoding="utf-8")
        (source / "docs").mkdir()
        os.symlink("../assets/manual.md", source / "docs" / "current.md")

        skills_dir = tmp_path / "skills"
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        add_result = add_skill(str(source), config=config, force=False, allow_symlinks=True)
        assert add_result.success, add_result.message
        assert get_origin("root-skill", config=config)["path"] == ""

        (source / "assets" / "manual.md").write_text("v2\n", encoding="utf-8")
        result = update_skill("root-skill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert "root-skill" in result.updated
        link = skills_dir / "root-skill" / "docs" / "current.md"
        assert link.is_symlink()
        assert os.readlink(link) == "../assets/manual.md"
        assert link.read_text(encoding="utf-8") == "v2\n"
        assert get_origin("root-skill", config=config)["path"] == ""

    def test_root_local_check_hashes_source_root_over_same_named_child(self, tmp_path):
        """A same-named child appearing after add does not replace the source root."""
        from skillport.modules.skills import add_skill

        source = tmp_path / "root-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nsame body", encoding="utf-8"
        )

        skills_dir = tmp_path / "skills"
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        add_result = add_skill(str(source), config=config, force=False)
        assert add_result.success, add_result.message
        assert get_origin("root-skill", config=config)["path"] == ""

        child = source / "root-skill"
        child.mkdir()
        (child / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nsame body", encoding="utf-8"
        )

        result = check_update_available("root-skill", config=config)

        assert result["available"] is True
        assert get_origin("root-skill", config=config)["path"] == ""

    def test_root_local_update_ignores_same_named_child_skill(self, tmp_path):
        """The explicit empty path updates from the source root, not a same-named child."""
        from skillport.modules.skills import add_skill

        source = tmp_path / "root-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nsame body", encoding="utf-8"
        )

        skills_dir = tmp_path / "skills"
        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")

        add_result = add_skill(str(source), config=config, force=False)
        assert add_result.success, add_result.message
        assert get_origin("root-skill", config=config)["path"] == ""

        child = source / "root-skill"
        child.mkdir()
        (child / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nsame body", encoding="utf-8"
        )
        (source / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: R\n---\nroot v2", encoding="utf-8"
        )

        result = update_skill("root-skill", config=config, allow_symlinks=True)

        assert result.success, result.message
        assert "root-skill" in result.updated
        installed = skills_dir / "root-skill" / "SKILL.md"
        assert installed.read_text(encoding="utf-8").endswith("root v2")
        assert get_origin("root-skill", config=config)["path"] == ""

    def test_root_github_check_reports_latest_for_explicit_empty_path(
        self, tmp_path, monkeypatch, require_symlink
    ):
        """The tree hash for an explicit empty path covers the repository root."""
        from skillport.modules.skills.public import update as update_module

        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-root"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(
            "---\nname: gh-root\ndescription: G\n---\nbody", encoding="utf-8"
        )
        (installed / "notes.txt").write_text("notes\n", encoding="utf-8")
        os.symlink("notes.txt", installed / "link")

        root_hash = _tree_hash_of_entries(
            [
                ("SKILL.md", b"---\nname: gh-root\ndescription: G\n---\nbody"),
                ("link", b"notes.txt"),
                ("notes.txt", b"notes\n"),
            ]
        )

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-root",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        seen: list[str] = []

        def fake_get_remote_tree_hash(parsed, token, path):
            seen.append(path)
            return root_hash if path == "" else "sha256:another-subtree"

        monkeypatch.setattr(update_module, "get_remote_tree_hash", fake_get_remote_tree_hash)

        result = check_update_available("gh-root", config=config)

        assert seen == [""]
        assert result["available"] is False
        assert "latest" in result["reason"].lower()
        assert get_origin("gh-root", config=config)["path"] == ""
        assert detect_local_modification("gh-root", config=config) is False

    def test_missing_path_origin_keeps_skill_name_fallback(self, tmp_path, monkeypatch):
        """An origin without a path key keeps the historical skill-name fallback."""
        from skillport.modules.skills.public import update as update_module

        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-root"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(
            "---\nname: gh-root\ndescription: G\n---\nbody", encoding="utf-8"
        )

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-root",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        seen: list[str] = []

        def fake_get_remote_tree_hash(parsed, token, path):
            seen.append(path)
            return compute_content_hash(installed)

        monkeypatch.setattr(update_module, "get_remote_tree_hash", fake_get_remote_tree_hash)

        result = check_update_available("gh-root", config=config)

        assert seen == ["gh-root"]
        assert result["available"] is False

    def test_flagless_root_github_update_rejects_symlink_source(self, tmp_path, monkeypatch):
        """The flagless gate scans the repository root recorded for the skill."""
        from skillport.modules.skills.public import update as update_module

        skills_dir = tmp_path / "skills"
        installed = skills_dir / "gh-root"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(
            "---\nname: gh-root\ndescription: G\n---\nbody", encoding="utf-8"
        )

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "gh-root",
            {
                "source": "https://github.com/user/repo",
                "kind": "github",
                "path": "",
                "commit_sha": "abc1234",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        scanned: list[str] = []

        def fake_get_remote_tree_symlinks(parsed, token, path):
            scanned.append(path)
            return ["link"] if path == "" else []

        monkeypatch.setattr(
            update_module,
            "get_remote_tree_hash",
            lambda parsed, token, path: compute_content_hash(installed),
        )
        monkeypatch.setattr(
            update_module, "get_remote_tree_symlinks", fake_get_remote_tree_symlinks
        )

        result = update_skill("gh-root", config=config)

        assert scanned == [""]
        assert result.success is False
        assert "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("body")


def _swap_source_at_boundary(monkeypatch, module, name: str, path: Path, external: Path) -> None:
    """Replace ``path`` with a symlink to ``external`` on the first boundary call.

    The swap lands after the origin was read and at the exact point where the
    source boundary opens the path, which is the TOCTOU window the boundary
    must close.
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
    """Local and zip sources swapped for a symlink at the boundary are never read."""

    @staticmethod
    def _local_setup(tmp_path: Path):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text(
            "---\nname: my-skill\ndescription: D\n---\ninstalled", encoding="utf-8"
        )

        external = tmp_path / "outside" / "my-skill"
        external.mkdir(parents=True)
        (external / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\nexternal", encoding="utf-8"
        )
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        original = source / "my-skill"
        original.mkdir(parents=True)
        (original / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\nsource", encoding="utf-8"
        )

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(source),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed, installed_file, source, external

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_local_update_does_not_follow_ancestor_swapped_at_boundary(
        self, tmp_path: Path, allow: bool, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.public import update as update_module

        config, installed, installed_file, source, external = self._local_setup(tmp_path)
        origin_before = get_origin("my-skill", config=config)

        _swap_source_at_boundary(
            monkeypatch,
            update_module,
            "acquire_local_source_snapshot",
            source,
            tmp_path / "outside",
        )

        result = update_skill("my-skill", config=config, allow_symlinks=allow)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text(encoding="utf-8").endswith("installed")
        assert not (installed / "external-marker.txt").exists()
        assert get_origin("my-skill", config=config) == origin_before
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_local_check_does_not_hash_ancestor_swapped_at_boundary(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.public import update as update_module

        config, installed, installed_file, source, external = self._local_setup(tmp_path)
        origin_before = get_origin("my-skill", config=config)

        _swap_source_at_boundary(
            monkeypatch,
            update_module,
            "acquire_local_source_snapshot",
            source,
            tmp_path / "outside",
        )

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "symlink" in result["reason"].lower()
        assert "latest" not in result["reason"].lower()
        assert installed_file.read_text(encoding="utf-8").endswith("installed")
        assert get_origin("my-skill", config=config) == origin_before
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    @staticmethod
    def _zip_setup(tmp_path: Path):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text(
            "---\nname: zskill\ndescription: D\n---\ninstalled", encoding="utf-8"
        )

        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "zskill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nexternal")
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        with zipfile.ZipFile(source / "alias" / "zskill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nsource")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(source / "alias" / "zskill.zip"),
                "kind": "zip",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed, installed_file, source, outside

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_zip_update_does_not_read_ancestor_swapped_at_boundary(
        self, tmp_path: Path, allow: bool, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.public import update as update_module

        config, installed, installed_file, source, outside = self._zip_setup(tmp_path)
        origin_before = get_origin("zskill", config=config)
        external_zip_before = (outside / "zskill.zip").read_bytes()

        _swap_source_at_boundary(
            monkeypatch,
            update_module,
            "open_zip_source",
            source / "alias",
            outside,
        )

        result = update_skill("zskill", config=config, allow_symlinks=allow)

        assert result.success is False
        assert "symlink" in result.message.lower()
        assert installed_file.read_text(encoding="utf-8").endswith("installed")
        assert not (installed / "external-marker.txt").exists()
        assert get_origin("zskill", config=config) == origin_before
        assert (outside / "zskill.zip").read_bytes() == external_zip_before
        assert (outside / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_zip_check_does_not_hash_ancestor_swapped_at_boundary(
        self, tmp_path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.public import update as update_module

        config, installed, installed_file, source, outside = self._zip_setup(tmp_path)
        origin_before = get_origin("zskill", config=config)

        _swap_source_at_boundary(
            monkeypatch,
            update_module,
            "open_zip_source",
            source / "alias",
            outside,
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is False
        assert "symlink" in result["reason"].lower()
        assert "latest" not in result["reason"].lower()
        assert get_origin("zskill", config=config) == origin_before
        assert (outside / "external-marker.txt").read_text(encoding="utf-8") == "secret"


class TestNoFollowUnsupportedPlatform:
    """Updates still work when O_NOFOLLOW and descriptor-relative opens are unavailable."""

    @staticmethod
    def _local_setup(tmp_path: Path) -> tuple[Config, Path]:
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "my-skill"
        installed.mkdir(parents=True)
        (installed / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\nold body", encoding="utf-8"
        )

        source = tmp_path / "source" / "my-skill"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\nnew body", encoding="utf-8"
        )

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "my-skill",
            {
                "source": str(tmp_path / "source"),
                "kind": "local",
                "path": "my-skill",
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        return config, installed

    def test_local_update_succeeds(self, tmp_path, no_follow_unavailable):
        config, installed = self._local_setup(tmp_path)

        result = update_skill("my-skill", config=config, allow_symlinks=False)

        assert result.success, result.message
        assert result.updated == ["my-skill"]
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("new body")

    def test_local_check_reports_available(self, tmp_path, no_follow_unavailable):
        config, installed = self._local_setup(tmp_path)

        result = check_update_available("my-skill", config=config)

        assert result["available"] is True
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("old body")

    def test_zip_update_succeeds(self, tmp_path, no_follow_unavailable):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text(
            "---\nname: zskill\ndescription: D\n---\nold body", encoding="utf-8"
        )

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nnew body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "",
                "source_mtime": zip_path.stat().st_mtime_ns - 10_000_000_000,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = update_skill("zskill", config=config, allow_symlinks=False)

        assert result.success, result.message
        assert result.updated == ["zskill"]
        assert installed_file.read_text(encoding="utf-8").endswith("new body")

    def test_zip_check_reports_available(self, tmp_path, no_follow_unavailable):
        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text(
            "---\nname: zskill\ndescription: D\n---\nold body", encoding="utf-8"
        )

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nnew body")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(zip_path),
                "kind": "zip",
                "path": "",
                "source_mtime": zip_path.stat().st_mtime_ns - 10_000_000_000,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is True
        assert installed_file.read_text(encoding="utf-8").endswith("old body")

    def test_local_update_does_not_follow_ancestor_swapped_at_boundary(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        from skillport.modules.skills.public import update as update_module

        config, installed = self._local_setup(tmp_path)
        origin_before = get_origin("my-skill", config=config)
        external = tmp_path / "outside"
        external.mkdir()
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        _swap_source_at_boundary(
            monkeypatch,
            update_module,
            "acquire_local_source_snapshot",
            tmp_path / "source",
            external,
        )

        result = update_skill("my-skill", config=config, allow_symlinks=False)

        assert result.success is False
        assert "traverses a symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("old body")
        assert not (installed / "external-marker.txt").exists()
        assert get_origin("my-skill", config=config) == origin_before
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_zip_check_does_not_hash_ancestor_swapped_at_boundary(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        from skillport.modules.skills.public import update as update_module

        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text(
            "---\nname: zskill\ndescription: D\n---\ninstalled", encoding="utf-8"
        )

        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "zskill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nexternal")
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        with zipfile.ZipFile(source / "alias" / "zskill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nsource")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(source / "alias" / "zskill.zip"),
                "kind": "zip",
                "path": "",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        _swap_source_at_boundary(
            monkeypatch,
            update_module,
            "open_zip_source",
            source / "alias",
            outside,
        )

        result = check_update_available("zskill", config=config)

        assert result["available"] is False
        assert "traverses a symlink" in result["reason"].lower()
        assert "latest" not in result["reason"].lower()
        assert get_origin("zskill", config=config) == origin_before
        assert (outside / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_local_update_rejects_ancestor_swapped_during_checked_copy(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        """A swap after the boundary check but during the tree read is rejected."""
        from skillport.modules.skills.internal import manager as manager_module

        config, installed = self._local_setup(tmp_path)
        origin_before = get_origin("my-skill", config=config)

        external = tmp_path / "outside"
        external_skill = external / "my-skill"
        external_skill.mkdir(parents=True)
        (external_skill / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\nexternal body", encoding="utf-8"
        )
        (external_skill / "external-marker.txt").write_text("secret", encoding="utf-8")

        _swap_source_during_checked_copy(monkeypatch, manager_module, tmp_path / "source", external)

        result = update_skill("my-skill", config=config, allow_symlinks=False)

        assert result.success is False
        assert "changed" in result.message.lower() or "symlink" in result.message.lower()
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("old body")
        assert not (installed / "external-marker.txt").exists()
        assert get_origin("my-skill", config=config) == origin_before
        assert (external_skill / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_local_check_rejects_ancestor_swapped_during_checked_copy(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        """--check does not hash a tree read through a source ancestor swapped mid-read."""
        from skillport.modules.skills.internal import manager as manager_module

        config, installed = self._local_setup(tmp_path)
        origin_before = get_origin("my-skill", config=config)

        external = tmp_path / "outside"
        external_skill = external / "my-skill"
        external_skill.mkdir(parents=True)
        (external_skill / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: D\n---\nexternal body", encoding="utf-8"
        )
        (external_skill / "external-marker.txt").write_text("secret", encoding="utf-8")

        _swap_source_during_checked_copy(monkeypatch, manager_module, tmp_path / "source", external)

        result = check_update_available("my-skill", config=config)

        assert result["available"] is False
        assert "changed" in result["reason"].lower() or "symlink" in result["reason"].lower()
        assert "latest" not in result["reason"].lower()
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("old body")
        assert get_origin("my-skill", config=config) == origin_before
        assert (external_skill / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_zip_check_rejects_archive_swapped_during_open(
        self, tmp_path, monkeypatch, no_follow_unavailable, require_symlink
    ):
        """--check does not read an archive whose path was swapped inside the open."""
        from skillport.shared import utils as utils_module

        skills_dir = tmp_path / "skills"
        installed = skills_dir / "zskill"
        installed.mkdir(parents=True)
        installed_file = installed / "SKILL.md"
        installed_file.write_text(
            "---\nname: zskill\ndescription: D\n---\ninstalled", encoding="utf-8"
        )

        outside = tmp_path / "outside"
        outside.mkdir()
        with zipfile.ZipFile(outside / "zskill.zip", "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nexternal")
        (outside / "external-marker.txt").write_text("secret", encoding="utf-8")
        external_zip_before = (outside / "zskill.zip").read_bytes()

        source = tmp_path / "source"
        (source / "alias").mkdir(parents=True)
        archive = source / "alias" / "zskill.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zskill\ndescription: D\n---\nsource")

        config = Config(skills_dir=skills_dir, db_path=tmp_path / "db.lancedb")
        record_origin(
            "zskill",
            {
                "source": str(archive),
                "kind": "zip",
                "path": "",
                "source_mtime": 0,
                "content_hash": compute_content_hash(installed),
            },
            config=config,
        )
        origin_before = get_origin("zskill", config=config)

        real_open = utils_module._open_descriptor
        state = {"swapped": False}
        displaced = archive.parent / "zskill-original.zip"

        def swapping_open(path, flags):
            if path == archive and not state["swapped"]:
                state["swapped"] = True
                archive.rename(displaced)
                archive.symlink_to(outside / "zskill.zip")
                try:
                    return real_open(path, flags)
                finally:
                    archive.unlink()
                    displaced.rename(archive)
            return real_open(path, flags)

        monkeypatch.setattr(utils_module, "_open_descriptor", swapping_open)

        result = check_update_available("zskill", config=config)

        assert result["available"] is False
        assert "changed" in result["reason"].lower() or "symlink" in result["reason"].lower()
        assert "latest" not in result["reason"].lower()
        assert (installed / "SKILL.md").read_text(encoding="utf-8").endswith("installed")
        assert get_origin("zskill", config=config) == origin_before
        assert (outside / "zskill.zip").read_bytes() == external_zip_before
        assert (outside / "external-marker.txt").read_text(encoding="utf-8") == "secret"
