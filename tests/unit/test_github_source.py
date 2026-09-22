"""Unit tests for GitHub URL parsing and extraction (SPEC2-CLI Section 3.3)."""

import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest

from skillport.modules.skills.internal.github import (
    GITHUB_URL_RE,
    ParsedGitHubURL,
    extract_tarball,
    get_remote_tree_symlinks,
    parse_github_url,
)


def _symlink_capable() -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink("target", Path(tmp) / "probe")
    except (OSError, NotImplementedError):
        return False
    return True


@pytest.fixture
def require_symlink():
    if not _symlink_capable():
        pytest.skip("symlinks cannot be created on this platform")


def _make_tar(tmp_path: Path, structure: dict) -> Path:
    """Create a tar.gz with given structure under root folder."""
    tar_path = tmp_path / "repo.tar.gz"
    root = "owner-repo-sha"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel, content in structure.items():
            full_name = f"{root}/{rel}"
            data = content.encode("utf-8")
            info = tarfile.TarInfo(full_name)
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
    return tar_path


class TestParseGitHubURL:
    """GitHub URL parsing tests."""

    def test_url_root_defaults_to_main(self):
        """URL without branch → defaults to main."""
        parsed = parse_github_url("https://github.com/user/repo")
        assert parsed.owner == "user"
        assert parsed.repo == "repo"
        assert parsed.ref == "main"
        assert parsed.normalized_path == ""

    def test_url_with_ref_and_path(self):
        """URL with branch and path → parsed correctly."""
        parsed = parse_github_url("https://github.com/user/repo/tree/feat/skills/path")
        assert parsed.owner == "user"
        assert parsed.repo == "repo"
        assert parsed.ref == "feat"
        assert parsed.normalized_path == "skills/path"

    def test_url_with_trailing_slash(self):
        """URL with trailing slash → handled correctly."""
        parsed = parse_github_url("https://github.com/user/repo/")
        assert parsed.owner == "user"
        assert parsed.repo == "repo"
        assert parsed.ref == "main"

    def test_url_tree_with_trailing_slash(self):
        """URL /tree/branch/ with trailing slash → handled correctly."""
        parsed = parse_github_url("https://github.com/user/repo/tree/main/")
        assert parsed.owner == "user"
        assert parsed.repo == "repo"
        assert parsed.ref == "main"
        assert parsed.normalized_path == ""

    def test_url_with_deep_path(self):
        """URL with deep path → parsed correctly."""
        parsed = parse_github_url("https://github.com/user/repo/tree/main/path/to/skills")
        assert parsed.normalized_path == "path/to/skills"

    def test_url_blob_format(self):
        """URL with /blob/ format → parsed same as /tree/."""
        parsed = parse_github_url("https://github.com/user/repo/blob/main/skills/path")
        assert parsed.owner == "user"
        assert parsed.repo == "repo"
        assert parsed.ref == "main"
        assert parsed.normalized_path == "skills/path"

    def test_url_blob_with_trailing_slash(self):
        """URL /blob/branch/ with trailing slash → handled correctly."""
        parsed = parse_github_url("https://github.com/user/repo/blob/main/")
        assert parsed.ref == "main"
        assert parsed.normalized_path == ""

    def test_url_rejects_traversal(self):
        """URL with path traversal → rejected."""
        with pytest.raises(ValueError, match="traversal"):
            parse_github_url("https://github.com/user/repo/tree/main/../secret")

    def test_url_invalid_format_rejected(self):
        """Invalid URL format → rejected."""
        invalid_urls = [
            "https://gitlab.com/user/repo",
            "http://github.com/user/repo",  # http not https
            "github.com/user/repo",  # missing https
            "https://github.com/user",  # missing repo
            "https://github.com/",  # missing owner/repo
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Unsupported"):
                parse_github_url(url)


class TestGitHubURLRegex:
    """GITHUB_URL_RE regex pattern tests."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            # Basic URLs
            (
                "https://github.com/owner/repo",
                {"owner": "owner", "repo": "repo", "ref": None, "path": None},
            ),
            (
                "https://github.com/owner/repo/",
                {"owner": "owner", "repo": "repo", "ref": None, "path": None},
            ),
            # With branch
            (
                "https://github.com/owner/repo/tree/main",
                {"owner": "owner", "repo": "repo", "ref": "main", "path": None},
            ),
            (
                "https://github.com/owner/repo/tree/main/",
                {"owner": "owner", "repo": "repo", "ref": "main", "path": "/"},
            ),
            (
                "https://github.com/owner/repo/tree/develop",
                {"owner": "owner", "repo": "repo", "ref": "develop", "path": None},
            ),
            # With path
            (
                "https://github.com/owner/repo/tree/main/skills",
                {"owner": "owner", "repo": "repo", "ref": "main", "path": "/skills"},
            ),
            (
                "https://github.com/owner/repo/tree/main/path/to/dir",
                {"owner": "owner", "repo": "repo", "ref": "main", "path": "/path/to/dir"},
            ),
            # Blob URLs (same parsing as tree)
            (
                "https://github.com/owner/repo/blob/main",
                {"owner": "owner", "repo": "repo", "ref": "main", "path": None},
            ),
            (
                "https://github.com/owner/repo/blob/main/skills/.experimental/create-plan",
                {
                    "owner": "owner",
                    "repo": "repo",
                    "ref": "main",
                    "path": "/skills/.experimental/create-plan",
                },
            ),
            # Special characters in owner/repo
            (
                "https://github.com/my-org/my-repo",
                {"owner": "my-org", "repo": "my-repo", "ref": None, "path": None},
            ),
            (
                "https://github.com/org123/repo456",
                {"owner": "org123", "repo": "repo456", "ref": None, "path": None},
            ),
        ],
    )
    def test_regex_matches(self, url: str, expected: dict):
        """Valid URLs should match the pattern."""
        match = GITHUB_URL_RE.match(url)
        assert match is not None
        assert match.group("owner") == expected["owner"]
        assert match.group("repo") == expected["repo"]
        assert match.group("ref") == expected["ref"]
        assert match.group("path") == expected["path"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://gitlab.com/owner/repo",
            "http://github.com/owner/repo",
            "github.com/owner/repo",
            "https://github.com/owner",
            "https://github.com/",
            "https://github.com",
        ],
    )
    def test_regex_rejects_invalid(self, url: str):
        """Invalid URLs should not match."""
        assert GITHUB_URL_RE.match(url) is None


class TestExtractTarball:
    """Tarball extraction tests."""

    def test_extract_subpath(self, tmp_path):
        """Extract specific subdirectory."""
        structure = {
            "skills/a/SKILL.md": "---\nname: a\n---\nbody",
            "skills/b/SKILL.md": "---\nname: b\n---\nbody",
        }
        tar_path = _make_tar(tmp_path, structure)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

        dest, commit_sha = extract_tarball(tar_path, parsed)

        assert (dest / "a" / "SKILL.md").exists()
        assert (dest / "b" / "SKILL.md").exists()
        assert commit_sha == "sha"  # From "owner-repo-sha" root

    def test_extract_rejects_symlink(self, tmp_path):
        """Symlinks in tarball → rejected."""
        tar_path = tmp_path / "repo.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(f"{root}/skills/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "evil"
            tar.addfile(info)

        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
        with pytest.raises(ValueError, match="Symlinks are not allowed in GitHub source"):
            extract_tarball(tar_path, parsed)

    def test_extract_rejects_hardlink(self, tmp_path, monkeypatch):
        """Hardlinks in tarball → rejected with the historical symlink error."""
        extracted = tmp_path / "extracted"
        monkeypatch.setattr(
            "skillport.modules.skills.internal.github.tempfile.mkdtemp",
            lambda **_kwargs: str(extracted),
        )
        tar_path = tmp_path / "repo.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            data = b"body"
            skill = tarfile.TarInfo(f"{root}/skills/a/SKILL.md")
            skill.size = len(data)
            tar.addfile(skill, io.BytesIO(data))
            info = tarfile.TarInfo(f"{root}/skills/hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "target"
            tar.addfile(info)

        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
        with pytest.raises(ValueError, match="Symlinks are not allowed in GitHub source"):
            extract_tarball(tar_path, parsed)

        assert not extracted.exists()

    def test_extract_excludes_dotfiles(self, tmp_path):
        """Dotfiles/dirs excluded from extraction."""
        structure = {
            "skills/a/SKILL.md": "---\nname: a\n---\nbody",
            "skills/a/.hidden": "hidden content",
            "skills/.git/config": "git config",
        }
        tar_path = _make_tar(tmp_path, structure)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

        dest, _ = extract_tarball(tar_path, parsed)

        assert (dest / "a" / "SKILL.md").exists()
        assert not (dest / "a" / ".hidden").exists()
        assert not (dest / ".git").exists()

    def test_extract_rejects_hidden_symlink(self, tmp_path):
        tar_path = tmp_path / "repo.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(f"{root}/skills/.hidden-link")
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            tar.addfile(info)

        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            extract_tarball(tar_path, parsed)

    def test_extract_root_path(self, tmp_path):
        """Extract from repository root."""
        structure = {
            "skill-a/SKILL.md": "---\nname: skill-a\n---\nbody",
            "skill-b/SKILL.md": "---\nname: skill-b\n---\nbody",
        }
        tar_path = _make_tar(tmp_path, structure)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="")

        dest, _ = extract_tarball(tar_path, parsed)

        assert (dest / "skill-a" / "SKILL.md").exists()
        assert (dest / "skill-b" / "SKILL.md").exists()

    def test_extract_rejects_path_traversal_member(self, tmp_path):
        """Traversal inside tar members is rejected (important on Windows too)."""
        structure = {
            "skills/../evil.txt": "evil",
            "skills/a/SKILL.md": "---\nname: a\n---\nbody",
        }
        tar_path = _make_tar(tmp_path, structure)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

        with pytest.raises(ValueError, match="Path traversal"):
            extract_tarball(tar_path, parsed)

    def test_extract_rejects_drive_prefixed_member(self, tmp_path):
        """Drive-prefixed tar members are rejected."""
        structure = {
            "skills/C:/Windows/evil.txt": "evil",
        }
        tar_path = _make_tar(tmp_path, structure)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

        with pytest.raises(ValueError, match="Path traversal"):
            extract_tarball(tar_path, parsed)


class TestExtractTarballAllowSymlinks:
    """allow_symlinks=True: issym members are materialized, islnk stays rejected."""

    def _make_symlink_tar(self, tmp_path: Path, link_target: str = "SKILL.md") -> Path:
        tar_path = tmp_path / "repo.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            data = b"---\nname: a\ndescription: A\n---\nbody"
            info = tarfile.TarInfo(f"{root}/skills/a/SKILL.md")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            link = tarfile.TarInfo(f"{root}/skills/a/link.md")
            link.type = tarfile.SYMTYPE
            link.linkname = link_target
            tar.addfile(link)
        return tar_path

    def test_sym_member_materialized_with_flag(self, tmp_path, require_symlink):
        """issym member is extracted as a real symlink when allowed."""
        tar_path = self._make_symlink_tar(tmp_path)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

        dest, _commit_sha = extract_tarball(tar_path, parsed, allow_symlinks=True)

        link = dest / "a" / "link.md"
        assert link.is_symlink()
        assert os.readlink(link) == "SKILL.md"
        assert link.resolve() == (dest / "a" / "SKILL.md").resolve()
        shutil.rmtree(dest, ignore_errors=True)

    def test_hidden_and_excluded_sym_members_materialized_with_flag(
        self, tmp_path, require_symlink
    ):
        """Hidden/excluded link members are materialized so per-skill validation sees them."""
        tar_path = tmp_path / "repo.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            data = b"---\nname: a\ndescription: A\n---\nbody"
            skill = tarfile.TarInfo(f"{root}/skills/a/SKILL.md")
            skill.size = len(data)
            tar.addfile(skill, io.BytesIO(data))
            for name in (".hidden-link", "node_modules/link"):
                link = tarfile.TarInfo(f"{root}/skills/a/{name}")
                link.type = tarfile.SYMTYPE
                link.linkname = "SKILL.md"
                tar.addfile(link)
            hidden_regular = tarfile.TarInfo(f"{root}/skills/a/.hidden")
            hidden_data = b"hidden content"
            hidden_regular.size = len(hidden_data)
            tar.addfile(hidden_regular, io.BytesIO(hidden_data))

        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

        dest, _commit_sha = extract_tarball(tar_path, parsed, allow_symlinks=True)
        try:
            assert (dest / "a" / ".hidden-link").is_symlink()
            assert (dest / "a" / "node_modules" / "link").is_symlink()
            assert not (dest / "a" / ".hidden").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_empty_symlink_target_is_rejected_before_add(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        from skillport.modules.skills.internal import add_local, detect_skills
        from skillport.modules.skills.internal import manager as manager_module
        from skillport.shared.config import Config

        tar_path = self._make_symlink_tar(tmp_path)
        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
        dest, _commit_sha = extract_tarball(tar_path, parsed, allow_symlinks=True)
        try:
            real_readlink = manager_module.os.readlink

            def readlink(path):
                if any(part.startswith("skillport-snapshot-") for part in Path(path).parts):
                    return ""
                return real_readlink(path)

            monkeypatch.setattr(manager_module.os, "readlink", readlink)
            target = tmp_path / "target"
            results = add_local(
                source_path=dest,
                skills=detect_skills(dest),
                config=Config(skills_dir=target),
                keep_structure=False,
                force=False,
                allow_symlinks=True,
            )

            assert len(results) == 1
            assert results[0].success is False
            assert not (target / "a").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_hardlink_rejected_even_with_flag(self, tmp_path, monkeypatch):
        """islnk member is rejected with the hardlink error even when symlinks are allowed."""
        extracted = tmp_path / "extracted"
        monkeypatch.setattr(
            "skillport.modules.skills.internal.github.tempfile.mkdtemp",
            lambda **_kwargs: str(extracted),
        )
        tar_path = tmp_path / "repo.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            data = b"body"
            skill = tarfile.TarInfo(f"{root}/skills/a/SKILL.md")
            skill.size = len(data)
            tar.addfile(skill, io.BytesIO(data))
            info = tarfile.TarInfo(f"{root}/skills/hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "target"
            tar.addfile(info)

        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
        with pytest.raises(ValueError, match="Hard links are not allowed in GitHub source"):
            extract_tarball(tar_path, parsed, allow_symlinks=True)

        assert not extracted.exists()

    def test_duplicate_normalized_path_does_not_overwrite_other_entry(
        self, tmp_path: Path, monkeypatch, require_symlink
    ):
        extracted = tmp_path / "extracted"
        monkeypatch.setattr(
            "skillport.modules.skills.internal.github.tempfile.mkdtemp",
            lambda **_kwargs: str(extracted),
        )

        tar_path = tmp_path / "duplicate.tar.gz"
        root = "owner-repo-sha"
        with tarfile.open(tar_path, "w:gz") as tar:
            link = tarfile.TarInfo(f"{root}/skills/a/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../b/target"
            tar.addfile(link)

            target = tarfile.TarInfo(f"{root}/skills/b/target")
            target_data = b"original"
            target.size = len(target_data)
            tar.addfile(target, io.BytesIO(target_data))

            duplicate = tarfile.TarInfo(f"{root}/skills/a/./link")
            duplicate_data = b"attacker"
            duplicate.size = len(duplicate_data)
            tar.addfile(duplicate, io.BytesIO(duplicate_data))

        parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
        with pytest.raises(ValueError, match="(?i)duplicate tar entry"):
            extract_tarball(tar_path, parsed, allow_symlinks=True)

        assert not extracted.exists()


def test_remote_tree_symlinks_include_hidden_and_excluded(monkeypatch):
    from skillport.modules.skills.internal import github as github_module

    monkeypatch.setattr(
        github_module,
        "_fetch_tree",
        lambda parsed, token: {
            "tree": [
                {"type": "blob", "mode": "120000", "path": "skills/.hidden/link"},
                {"type": "blob", "mode": "120000", "path": "skills/node_modules/link"},
                {"type": "blob", "mode": "100644", "path": "skills/regular.txt"},
            ]
        },
    )

    parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

    assert get_remote_tree_symlinks(parsed, None, "") == [
        ".hidden/link",
        "node_modules/link",
    ]


# Backward compatibility - keep original test function names
def test_parse_github_url_root_defaults_to_main():
    parsed = parse_github_url("https://github.com/user/repo")
    assert parsed.owner == "user"
    assert parsed.repo == "repo"
    assert parsed.ref == "main"
    assert parsed.normalized_path == ""


def test_parse_github_url_with_ref_and_path():
    parsed = parse_github_url("https://github.com/user/repo/tree/feat/skills/path")
    assert parsed.ref == "feat"
    assert parsed.normalized_path == "skills/path"


def test_parse_github_url_rejects_traversal():
    with pytest.raises(ValueError):
        parse_github_url("https://github.com/user/repo/tree/main/../secret")


def test_extract_tarball_subpath(tmp_path):
    structure = {
        "skills/a/SKILL.md": "---\nname: a\n---\nbody",
        "skills/b/SKILL.md": "---\nname: b\n---\nbody",
    }
    tar_path = _make_tar(tmp_path, structure)
    parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")

    dest, commit_sha = extract_tarball(tar_path, parsed)

    assert (dest / "a" / "SKILL.md").exists()
    assert (dest / "b" / "SKILL.md").exists()
    assert commit_sha == "sha"


def test_extract_tarball_rejects_symlink(tmp_path):
    tar_path = tmp_path / "repo.tar.gz"
    root = "owner-repo-sha"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo(f"{root}/skills/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "evil"
        tar.addfile(info)

    parsed = ParsedGitHubURL(owner="user", repo="repo", ref="main", path="/skills")
    with pytest.raises(ValueError):
        extract_tarball(tar_path, parsed)
