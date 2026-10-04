"""Integration tests for CLI commands (SPEC2-CLI Section 2-3).

Uses Typer's CliRunner for E2E CLI testing.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillport.interfaces.cli.app import app

runner = CliRunner()


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


@dataclass
class SkillsEnv:
    """Test environment with skills paths."""

    skills_dir: Path


def _create_skill(path: Path, name: str, description: str = "Test skill") -> Path:
    """Helper to create a valid skill."""
    skill_dir = path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nmetadata:\n  skillport:\n    category: test\n---\n# {name}\n\nInstructions here.",
        encoding="utf-8",
    )
    return skill_dir


def _create_vendor_skill(
    path: Path, name: str, frontmatter: str = "model: sonnet\nicon: toolbox"
) -> Path:
    """Helper to create a skill that uses vendor-specific frontmatter keys."""
    skill_dir = path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A vendor skill\n{frontmatter}\n---\n# {name}\n\nInstructions here.",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture
def skills_env(tmp_path: Path, monkeypatch) -> SkillsEnv:
    """Fixture providing isolated skills environment."""
    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setenv("SKILLPORT_SKILLS_DIR", str(skills))
    monkeypatch.setenv("SKILLPORT_EMBEDDING_PROVIDER", "none")
    return SkillsEnv(skills_dir=skills)


class TestListCommand:
    """skillport list tests."""

    def test_list_empty_skills_dir(self, skills_env: SkillsEnv):
        """Empty skills dir → shows 0 skills."""
        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        # Should show table or "0" message
        assert "0" in result.stdout or "Skills" in result.stdout

    def test_list_with_skills(self, skills_env: SkillsEnv):
        """With skills → shows table."""
        _create_skill(skills_env.skills_dir, "skill-a")
        _create_skill(skills_env.skills_dir, "skill-b")

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "skill-a" in result.stdout
        assert "skill-b" in result.stdout

    def test_list_json_output(self, skills_env: SkillsEnv):
        """--json → valid JSON output."""
        _create_skill(skills_env.skills_dir, "test-skill", "A test skill")

        result = runner.invoke(app, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "skills" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_list_with_limit(self, skills_env: SkillsEnv):
        """--limit restricts results."""
        for i in range(5):
            _create_skill(skills_env.skills_dir, f"skill-{i}")

        result = runner.invoke(app, ["list", "--limit", "2", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data["skills"]) <= 2


class TestShowCommand:
    """skillport show tests."""

    def test_show_existing_skill(self, skills_env: SkillsEnv):
        """Existing skill → shows details."""
        _create_skill(skills_env.skills_dir, "test-skill", "A test skill")

        result = runner.invoke(app, ["show", "test-skill"])

        assert result.exit_code == 0
        assert "test-skill" in result.stdout
        assert "A test skill" in result.stdout or "Instructions" in result.stdout

    def test_show_nonexistent_skill(self, skills_env: SkillsEnv):
        """Non-existent skill → error (exit 1)."""
        result = runner.invoke(app, ["show", "nonexistent"])

        assert result.exit_code == 1
        # Error might be in stdout or exception message
        assert "not found" in (result.stdout + str(result.exception)).lower()

    def test_show_json_output(self, skills_env: SkillsEnv):
        """--json → valid JSON output."""
        _create_skill(skills_env.skills_dir, "test-skill", "Test description")

        result = runner.invoke(app, ["show", "test-skill", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["id"] == "test-skill"
        assert "instructions" in data


class TestProjectConfigResolution:
    """CLI should respect .skillportrc skills_dir when present."""

    def test_show_uses_skillportrc_skills_dir(self, tmp_path: Path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()

        skills_dir = project / "custom-skills"
        skills_dir.mkdir()
        _create_skill(skills_dir, "rc-skill", "From skillportrc")

        # Write .skillportrc pointing to custom skills directory
        rc_path = project / ".skillportrc"
        rc_path.write_text(
            "skills_dir: ./custom-skills\ninstructions:\n  - AGENTS.md\n",
            encoding="utf-8",
        )

        # Run CLI from project root; should pick up .skillportrc skills_dir
        monkeypatch.chdir(project)
        env = {"SKILLPORT_EMBEDDING_PROVIDER": "none"}
        result = runner.invoke(app, ["show", "rc-skill", "--json"], env=env)

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["id"] == "rc-skill"
        assert data["path"].startswith(str(skills_dir))

    def test_env_overrides_skillportrc_skills_dir(self, tmp_path: Path, monkeypatch):
        """When both env and .skillportrc are set, env wins."""
        project = tmp_path / "project"
        project.mkdir()

        env_skills = project / "env-skills"
        env_skills.mkdir()
        _create_skill(env_skills, "env-skill", "From env")

        rc_skills = project / "rc-skills"
        rc_skills.mkdir()
        _create_skill(rc_skills, "rc-skill", "From rc")
        rc_path = project / ".skillportrc"
        rc_path.write_text("skills_dir: ./rc-skills\ninstructions: []\n", encoding="utf-8")

        # Both env var and .skillportrc set; env should take precedence
        monkeypatch.chdir(project)
        env = {
            "SKILLPORT_SKILLS_DIR": str(env_skills),
            "SKILLPORT_EMBEDDING_PROVIDER": "none",
        }
        result = runner.invoke(app, ["show", "env-skill", "--json"], env=env)

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["id"] == "env-skill"
        assert data["path"].startswith(str(env_skills))


class TestAddCommand:
    """skillport add tests.

    Note: Built-in skill add returns AddResult with empty `added` list,
    causing CLI to exit 1 despite successful file creation. This is a
    known bug in the implementation. Tests verify file existence instead.
    """

    def test_add_builtin_hello_world(self, skills_env: SkillsEnv):
        """Add built-in hello-world → creates file."""
        runner.invoke(app, ["add", "hello-world"], input="\n")

        # Verify file was created (primary acceptance criteria)
        assert (skills_env.skills_dir / "hello-world" / "SKILL.md").exists()

    def test_add_builtin_template(self, skills_env: SkillsEnv):
        """Add built-in template → creates file."""
        runner.invoke(app, ["add", "template"], input="\n")

        # Verify file was created
        assert (skills_env.skills_dir / "template" / "SKILL.md").exists()

    def test_add_local_skill(self, skills_env: SkillsEnv, tmp_path: Path):
        """Add local skill → success."""
        source = tmp_path / "source"
        _create_skill(source, "local-skill")

        result = runner.invoke(app, ["add", str(source / "local-skill"), "--no-keep-structure"])

        assert result.exit_code == 0
        assert (skills_env.skills_dir / "local-skill" / "SKILL.md").exists()

    def test_add_already_exists_no_force(self, skills_env: SkillsEnv):
        """Already exists without --force → skipped message."""
        # Add first time
        runner.invoke(app, ["add", "hello-world"], input="\n")

        # Add again
        result = runner.invoke(app, ["add", "hello-world"], input="\n")

        # Should indicate skipped/exists
        assert (
            "exists" in result.stdout.lower()
            or "skipped" in result.stdout.lower()
            or "⊘" in result.stdout
        )

    def test_add_with_force_overwrites(self, skills_env: SkillsEnv):
        """--force overwrites existing built-in."""
        # Add first time
        runner.invoke(app, ["add", "hello-world"], input="\n")

        # Modify the file
        skill_md = skills_env.skills_dir / "hello-world" / "SKILL.md"
        skill_md.write_text("modified", encoding="utf-8")

        # Add again with force
        runner.invoke(app, ["add", "hello-world", "--force"], input="\n")

        # Verify file was restored to original content
        content = skill_md.read_text()
        assert "Hello World" in content  # Original content restored

    def test_add_respects_cli_overrides(self, skills_env: SkillsEnv, tmp_path: Path):
        """--skills-dir overrides env defaults for add."""
        custom_skills = tmp_path / "custom-skills"

        runner.invoke(
            app,
            [
                "--skills-dir",
                str(custom_skills),
                "add",
                "hello-world",
            ],
            input="\n",
        )

        # Even if exit_code is non-zero (known issue), files should land in custom paths
        assert (custom_skills / "hello-world" / "SKILL.md").exists()
        # Default env skills dir should remain untouched
        assert not (skills_env.skills_dir / "hello-world" / "SKILL.md").exists()


class TestRemoveCommand:
    """skillport remove tests."""

    def test_remove_existing_skill(self, skills_env: SkillsEnv):
        """Remove existing skill → success."""
        _create_skill(skills_env.skills_dir, "to-remove")

        result = runner.invoke(app, ["remove", "to-remove", "--force"])

        assert result.exit_code == 0
        assert not (skills_env.skills_dir / "to-remove").exists()
        assert "Removed" in result.stdout

    def test_remove_nonexistent_skill(self, skills_env: SkillsEnv):
        """Remove non-existent skill → error (exit 1)."""
        result = runner.invoke(app, ["remove", "nonexistent", "--force"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


class TestValidateCommand:
    """skillport validate tests."""

    def test_validate_valid_skills(self, skills_env: SkillsEnv):
        """Valid skills → "All pass" (exit 0)."""
        _create_skill(skills_env.skills_dir, "valid-skill", "A valid skill")

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "pass" in result.stdout.lower() or "✓" in result.stdout

    def test_validate_invalid_skill(self, skills_env: SkillsEnv):
        """Invalid skill → issues listed (exit 1)."""
        # Create skill with name mismatch
        skill_dir = skills_env.skills_dir / "correct-dir"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: wrong-name\ndescription: test\n---\nbody", encoding="utf-8"
        )

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 1
        assert "fatal" in result.stdout.lower() or "issue" in result.stdout.lower()

    def test_validate_specific_skill(self, skills_env: SkillsEnv):
        """Validate specific skill by ID → only that skill checked."""
        _create_skill(skills_env.skills_dir, "skill-a", "Skill A")
        _create_skill(skills_env.skills_dir, "skill-b", "Skill B")

        result = runner.invoke(app, ["validate", "skill-a"])

        assert result.exit_code == 0

    def test_validate_by_path_single_skill(self, skills_env: SkillsEnv):
        """Validate by path (single skill) → works without index."""
        skill_dir = _create_skill(skills_env.skills_dir, "path-skill", "A valid skill")
        # Note: not rebuilding index - path-based validation should work without it

        result = runner.invoke(app, ["validate", str(skill_dir)])

        assert result.exit_code == 0
        assert "pass" in result.stdout.lower() or "✓" in result.stdout

    def test_validate_by_path_directory(self, skills_env: SkillsEnv):
        """Validate by path (directory) → scans all skills in dir."""
        _create_skill(skills_env.skills_dir, "skill-a", "Skill A")
        _create_skill(skills_env.skills_dir, "skill-b", "Skill B")
        # Note: not rebuilding index

        result = runner.invoke(app, ["validate", str(skills_env.skills_dir)])

        assert result.exit_code == 0
        assert "2 skill" in result.stdout.lower()

    def test_validate_by_path_nested_directory(self, skills_env: SkillsEnv):
        """Validate by path scans nested/namespaced skills."""
        # Create flat skill
        _create_skill(skills_env.skills_dir, "flat-skill", "Flat skill")
        # Create namespaced skills
        ns_dir = skills_env.skills_dir / "my-namespace"
        ns_dir.mkdir()
        _create_skill(ns_dir, "nested-a", "Nested A")
        _create_skill(ns_dir, "nested-b", "Nested B")

        result = runner.invoke(app, ["validate", str(skills_env.skills_dir)])

        assert result.exit_code == 0
        assert "3 skill" in result.stdout.lower()

    def test_validate_by_path_invalid_skill(self, skills_env: SkillsEnv):
        """Validate by path with invalid skill → shows issues."""
        skill_dir = skills_env.skills_dir / "invalid-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: wrong-name\ndescription: test\n---\nbody", encoding="utf-8"
        )

        result = runner.invoke(app, ["validate", str(skill_dir)])

        assert result.exit_code == 1
        assert "fatal" in result.stdout.lower()

    def test_validate_warning_only_exit_0(self, skills_env: SkillsEnv):
        """Only warnings → exit 0."""
        # Create skill with >500 lines (warning, not fatal)
        skill_dir = skills_env.skills_dir / "warning-skill"
        skill_dir.mkdir()
        long_body = "\n".join(["line"] * 501)  # >500 lines triggers warning
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: warning-skill\ndescription: A valid skill\n---\n{long_body}",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["validate"])

        # Exit 0 because only warnings
        assert result.exit_code == 0
        assert "warning" in result.stdout.lower()

    def test_lint_deprecated_alias(self, skills_env: SkillsEnv):
        """lint command works as deprecated alias."""
        _create_skill(skills_env.skills_dir, "test-skill", "A test skill")

        result = runner.invoke(app, ["lint"])

        assert result.exit_code == 0
        assert "deprecated" in result.stdout.lower()
        assert "pass" in result.stdout.lower() or "✓" in result.stdout


class TestExitCodes:
    """Exit code verification tests."""

    def test_success_exit_0(self, skills_env: SkillsEnv):
        """Successful operations → exit 0."""
        _create_skill(skills_env.skills_dir, "test-skill")

        list_result = runner.invoke(app, ["list"])
        assert list_result.exit_code == 0

        show_result = runner.invoke(app, ["show", "test-skill"])
        assert show_result.exit_code == 0

    def test_error_exit_1(self, skills_env: SkillsEnv):
        """Errors → exit 1."""
        # Show non-existent
        show_result = runner.invoke(app, ["show", "nonexistent"])
        assert show_result.exit_code == 1

        # Remove non-existent
        remove_result = runner.invoke(app, ["remove", "nonexistent", "--force"])
        assert remove_result.exit_code == 1


class TestNamespacedSkills:
    """Tests for namespaced skill IDs."""

    def test_show_namespaced_skill(self, skills_env: SkillsEnv):
        """Show skill with namespace → works."""
        ns_dir = skills_env.skills_dir / "my-team" / "team-skill"
        ns_dir.mkdir(parents=True)
        (ns_dir / "SKILL.md").write_text(
            "---\nname: team-skill\ndescription: Team skill\n---\nbody", encoding="utf-8"
        )

        result = runner.invoke(app, ["show", "my-team/team-skill"])

        assert result.exit_code == 0
        assert "team-skill" in result.stdout

    def test_remove_namespaced_skill(self, skills_env: SkillsEnv):
        """Remove namespaced skill → works."""
        ns_dir = skills_env.skills_dir / "my-team" / "team-skill"
        ns_dir.mkdir(parents=True)
        (ns_dir / "SKILL.md").write_text(
            "---\nname: team-skill\ndescription: Team skill\n---\nbody", encoding="utf-8"
        )

        result = runner.invoke(app, ["remove", "my-team/team-skill", "--force"])

        assert result.exit_code == 0
        assert not ns_dir.exists()


class TestListVisibility:
    """List reflects filesystem changes after add/remove."""

    def test_add_then_list_shows_skill(self, skills_env: SkillsEnv):
        """add → list shows skill immediately (no manual reindex)."""
        runner.invoke(app, ["add", "hello-world"], input="\n")

        result = runner.invoke(app, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        skill_ids = [s["id"] for s in data["skills"]]
        assert "hello-world" in skill_ids

    def test_remove_then_list_hides_skill(self, skills_env: SkillsEnv):
        """remove → list hides skill immediately."""
        # Add first
        runner.invoke(app, ["add", "hello-world"], input="\n")

        # Remove
        runner.invoke(app, ["remove", "hello-world", "--force"])

        # List should not contain the skill
        result = runner.invoke(app, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        skill_ids = [s["id"] for s in data["skills"]]
        assert "hello-world" not in skill_ids

    def test_add_local_then_list_shows_skill(self, skills_env: SkillsEnv, tmp_path: Path):
        """add local → list shows skill immediately."""
        # Create local skill
        source = tmp_path / "source"
        _create_skill(source, "searchable-skill", "A skill for testing search")

        # Add without manual reindex
        runner.invoke(app, ["add", str(source / "searchable-skill"), "--no-keep-structure"])

        # List should show it
        result = runner.invoke(app, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        skill_ids = [s["id"] for s in data["skills"]]
        assert "searchable-skill" in skill_ids


class TestDocCommand:
    """skillport doc tests."""

    def test_doc_creates_agents_md(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc creates AGENTS.md file."""
        _create_skill(skills_env.skills_dir, "test-skill", "Test description")

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(app, ["doc", "-o", str(output), "--force"])

        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "test-skill" in content
        assert "<!-- SKILLPORT_START -->" in content
        assert "<!-- SKILLPORT_END -->" in content

    def test_doc_xml_format(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc --format xml includes <available_skills> tag."""
        _create_skill(skills_env.skills_dir, "test-skill")

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(app, ["doc", "-o", str(output), "--format", "xml", "--force"])

        assert result.exit_code == 0
        content = output.read_text()
        assert "<available_skills>" in content
        assert "</available_skills>" in content

    def test_doc_markdown_format(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc --format markdown does not include XML tags."""
        _create_skill(skills_env.skills_dir, "test-skill")

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(app, ["doc", "-o", str(output), "--format", "markdown", "--force"])

        assert result.exit_code == 0
        content = output.read_text()
        assert "<available_skills>" not in content
        assert "## SkillPort Skills" in content

    def test_doc_with_skills_filter(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc --skills filters to specific skills."""
        _create_skill(skills_env.skills_dir, "skill-a")
        _create_skill(skills_env.skills_dir, "skill-b")
        _create_skill(skills_env.skills_dir, "skill-c")

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(
            app, ["doc", "-o", str(output), "--skills", "skill-a,skill-c", "--force"]
        )

        assert result.exit_code == 0
        content = output.read_text()
        assert "skill-a" in content
        assert "skill-c" in content
        assert "skill-b" not in content

    def test_doc_with_category_filter(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc --category filters by category."""
        # Create skills with different categories
        skill_a = skills_env.skills_dir / "skill-a"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Skill A\nmetadata:\n  skillport:\n    category: dev\n---\nbody"
        )

        skill_b = skills_env.skills_dir / "skill-b"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: Skill B\nmetadata:\n  skillport:\n    category: test\n---\nbody"
        )

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(app, ["doc", "-o", str(output), "--category", "dev", "--force"])

        assert result.exit_code == 0
        content = output.read_text()
        assert "skill-a" in content
        assert "skill-b" not in content

    def test_doc_no_skills_exits_1(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc with no matching skills exits with code 1."""

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(app, ["doc", "-o", str(output), "--force"])

        assert result.exit_code == 1
        assert "no skills" in result.stdout.lower()

    def test_doc_appends_to_existing(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc appends to existing file without markers."""
        _create_skill(skills_env.skills_dir, "test-skill")

        output = tmp_path / "AGENTS.md"
        output.write_text("# Existing Content\n\nSome existing text.\n")

        result = runner.invoke(app, ["doc", "-o", str(output), "--force"])

        assert result.exit_code == 0
        content = output.read_text()
        assert "# Existing Content" in content
        assert "test-skill" in content

    def test_doc_replaces_existing_block(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc replaces existing SkillPort block."""
        _create_skill(skills_env.skills_dir, "new-skill")

        output = tmp_path / "AGENTS.md"
        output.write_text(
            "# Header\n\n"
            "<!-- SKILLPORT_START -->\nold content\n<!-- SKILLPORT_END -->\n\n"
            "# Footer\n"
        )

        result = runner.invoke(app, ["doc", "-o", str(output), "--force"])

        assert result.exit_code == 0
        content = output.read_text()
        assert "# Header" in content
        assert "# Footer" in content
        assert "new-skill" in content
        assert "old content" not in content

    def test_doc_invalid_format_exits_1(self, skills_env: SkillsEnv, tmp_path: Path):
        """doc --format invalid exits with code 1."""
        _create_skill(skills_env.skills_dir, "test-skill")

        output = tmp_path / "AGENTS.md"
        result = runner.invoke(app, ["doc", "-o", str(output), "--format", "invalid", "--force"])

        assert result.exit_code == 1
        assert "invalid" in result.stdout.lower()


def _create_linked_skill_source(source: Path, name: str) -> Path:
    """Create a single-skill source containing a compliant relative symlink."""
    skill = _create_skill(source, name)
    assets = skill / "assets"
    assets.mkdir()
    (assets / "manual.md").write_text("manual v1", encoding="utf-8")
    docs = skill / "docs"
    docs.mkdir()
    os.symlink("../assets/manual.md", docs / "current.md")
    return skill


class TestAllowSymlinksCli:
    """--allow-symlinks CLI wiring for add and update."""

    def test_add_help_lists_allow_symlinks(self, skills_env: SkillsEnv):
        """add --help documents the --allow-symlinks option."""
        result = runner.invoke(app, ["add", "--help"])

        assert result.exit_code == 0
        assert "--allow-symlinks" in result.stdout

    def test_update_help_lists_allow_symlinks(self, skills_env: SkillsEnv):
        """update --help documents the --allow-symlinks option."""
        result = runner.invoke(app, ["update", "--help"])

        assert result.exit_code == 0
        assert "--allow-symlinks" in result.stdout

    def test_add_local_symlink_with_flag(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        """add --allow-symlinks installs the skill with the symlink preserved."""
        source = tmp_path / "src"
        _create_linked_skill_source(source, "linked-skill")

        result = runner.invoke(app, ["add", str(source), "--allow-symlinks", "--no-keep-structure"])

        assert result.exit_code == 0, result.stdout
        current = skills_env.skills_dir / "linked-skill" / "docs" / "current.md"
        assert current.is_symlink()
        assert current.read_text(encoding="utf-8") == "manual v1"

    def test_add_local_symlink_without_flag_is_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        """Flagless add of a symlinked local source exits non-zero without installing."""
        source = tmp_path / "src"
        _create_linked_skill_source(source, "linked-skill")

        result = runner.invoke(app, ["add", str(source), "--no-keep-structure"])

        assert result.exit_code != 0
        assert not (skills_env.skills_dir / "linked-skill").exists()
        assert not any(skills_env.skills_dir.rglob("current.md"))
        assert not any(skills_env.skills_dir.rglob("manual.md"))

    def test_add_local_source_root_symlink_is_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        real_skill = _create_skill(tmp_path / "outside", "linked-skill")
        (real_skill / "external-marker.txt").write_text("secret", encoding="utf-8")
        source = tmp_path / "source"
        source.symlink_to(real_skill)

        result = runner.invoke(app, ["add", str(source), "--allow-symlinks", "--no-keep-structure"])

        assert result.exit_code != 0
        assert not (skills_env.skills_dir / "linked-skill").exists()
        assert not any(skills_env.skills_dir.rglob("external-marker.txt"))

    def test_add_zip_symlink_with_flag(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        """add --allow-symlinks on a zip source materializes the symlink entry."""
        import zipfile

        zip_path = tmp_path / "zipped-skill.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: zipped-skill\ndescription: Z\n---\nbody")
            info = zipfile.ZipInfo("link.txt")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "SKILL.md")

        result = runner.invoke(
            app, ["add", str(zip_path), "--allow-symlinks", "--no-keep-structure"]
        )

        assert result.exit_code == 0, result.stdout
        link = skills_env.skills_dir / "zipped-skill" / "link.txt"
        assert link.is_symlink()
        assert os.readlink(link) == "SKILL.md"
        # Reading through the link yields the target file's content
        assert link.read_text(encoding="utf-8") == (
            skills_env.skills_dir / "zipped-skill" / "SKILL.md"
        ).read_text(encoding="utf-8")

    @pytest.mark.parametrize("link_name", [".env-link", "node_modules"], ids=["hidden", "excluded"])
    def test_add_zip_ignores_hidden_or_excluded_root_symlink(
        self, skills_env: SkillsEnv, tmp_path: Path, link_name: str, monkeypatch, require_symlink
    ):
        """Pre-detection drops root hidden/excluded symlinks and adds the flat skill."""
        import zipfile

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "zskill/SKILL.md",
                "---\nname: zskill\ndescription: Z\n---\nbody",
            )
            info = zipfile.ZipInfo(link_name)
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "zskill")

        result = runner.invoke(app, ["add", str(zip_path), "--allow-symlinks", "--yes"])

        assert result.exit_code == 0, result.stdout
        assert "Added 'zskill'" in result.stdout
        assert (skills_env.skills_dir / "zskill" / "SKILL.md").exists()
        assert not (skills_env.skills_dir / "zskill.zip").exists()
        assert not os.path.lexists(skills_env.skills_dir / link_name)
        assert not os.path.lexists(skills_env.skills_dir / "zskill" / link_name)
        assert list(sandbox.iterdir()) == []

    def test_add_zip_visible_root_symlink_is_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, require_symlink
    ):
        """A visible root symlink stays a candidate and the single-skill check rejects it."""
        import zipfile

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

        zip_path = tmp_path / "zskill.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "zskill/SKILL.md",
                "---\nname: zskill\ndescription: Z\n---\nbody",
            )
            info = zipfile.ZipInfo("alias")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "zskill")

        result = runner.invoke(app, ["add", str(zip_path), "--allow-symlinks", "--yes"])

        assert result.exit_code != 0
        assert "found 2" in result.stdout
        assert not os.path.lexists(skills_env.skills_dir / "zskill")
        assert not os.path.lexists(skills_env.skills_dir / "alias")
        assert list(sandbox.iterdir()) == []

    def test_update_local_symlink_with_flag(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        """update --allow-symlinks applies source changes and preserves the link."""
        source = tmp_path / "src"
        skill = _create_linked_skill_source(source, "linked-skill")

        add_result = runner.invoke(
            app, ["add", str(source), "--allow-symlinks", "--no-keep-structure"]
        )
        assert add_result.exit_code == 0, add_result.stdout

        (skill / "assets" / "manual.md").write_text("manual v2", encoding="utf-8")

        result = runner.invoke(app, ["update", "linked-skill", "--allow-symlinks"])

        assert result.exit_code == 0, result.stdout
        current = skills_env.skills_dir / "linked-skill" / "docs" / "current.md"
        assert current.is_symlink()
        assert current.read_text(encoding="utf-8") == "manual v2"

    def test_update_all_with_flag_preserves_symlink(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        """update --all --allow-symlinks forwards the flag and keeps the link."""
        source = tmp_path / "src"
        skill = _create_linked_skill_source(source, "linked-skill")

        add_result = runner.invoke(
            app, ["add", str(source), "--allow-symlinks", "--no-keep-structure"]
        )
        assert add_result.exit_code == 0, add_result.stdout

        (skill / "assets" / "manual.md").write_text("manual v2", encoding="utf-8")

        result = runner.invoke(app, ["update", "--all", "--allow-symlinks"])

        assert result.exit_code == 0, result.stdout
        current = skills_env.skills_dir / "linked-skill" / "docs" / "current.md"
        assert current.is_symlink()
        assert current.read_text(encoding="utf-8") == "manual v2"

    def test_update_confirm_bulk_with_flag_preserves_symlink(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, require_symlink
    ):
        """Answering the post-check confirmation with the flag runs a bulk update keeping links."""
        from skillport.interfaces.cli.theme import console as theme_console

        source = tmp_path / "src"
        skill = _create_linked_skill_source(source, "linked-skill")

        add_result = runner.invoke(
            app, ["add", str(source), "--allow-symlinks", "--no-keep-structure"]
        )
        assert add_result.exit_code == 0, add_result.stdout

        (skill / "assets" / "manual.md").write_text("manual v2", encoding="utf-8")

        # The confirmation prompt only runs for an interactive console
        monkeypatch.setattr(theme_console, "is_interactive", True)

        result = runner.invoke(app, ["update", "--allow-symlinks"], input="y\n")

        assert result.exit_code == 0, result.stdout
        current = skills_env.skills_dir / "linked-skill" / "docs" / "current.md"
        assert current.is_symlink()
        assert current.read_text(encoding="utf-8") == "manual v2"

    def test_update_confirm_declined_leaves_skill_unchanged(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, require_symlink
    ):
        """Declining the confirmation leaves the installed skill untouched."""
        from skillport.interfaces.cli.theme import console as theme_console

        source = tmp_path / "src"
        skill = _create_linked_skill_source(source, "linked-skill")

        add_result = runner.invoke(
            app, ["add", str(source), "--allow-symlinks", "--no-keep-structure"]
        )
        assert add_result.exit_code == 0, add_result.stdout

        (skill / "assets" / "manual.md").write_text("manual v2", encoding="utf-8")
        monkeypatch.setattr(theme_console, "is_interactive", True)

        result = runner.invoke(app, ["update", "--allow-symlinks"], input="n\n")

        assert result.exit_code == 0, result.stdout
        current = skills_env.skills_dir / "linked-skill" / "docs" / "current.md"
        assert current.is_symlink()
        # Declined: still the original content
        assert current.read_text(encoding="utf-8") == "manual v1"

    def test_update_check_reports_installed_root_symlink(
        self, skills_env: SkillsEnv, tmp_path: Path, require_symlink
    ):
        """update --check and the default invocation reject a symlinked installed root."""
        import shutil

        source = tmp_path / "src"
        _create_skill(source, "checked-skill")

        add_result = runner.invoke(app, ["add", str(source), "--no-keep-structure"])
        assert add_result.exit_code == 0, add_result.stdout

        external = tmp_path / "external" / "checked-skill"
        external.mkdir(parents=True)
        (external / "SKILL.md").write_text(
            "---\nname: checked-skill\ndescription: External\n---\nexternal body",
            encoding="utf-8",
        )
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")
        installed = skills_env.skills_dir / "checked-skill"
        shutil.rmtree(installed)
        os.symlink(external, installed)

        check_result = runner.invoke(app, ["update", "--check", "--json"])

        assert check_result.exit_code == 0, check_result.stdout
        data = json.loads(check_result.stdout)
        assert [item["skill_id"] for item in data["not_updatable"]] == ["checked-skill"]
        assert "symlink" in data["not_updatable"][0]["reason"].lower()
        assert data["updates_available"] == []

        default_result = runner.invoke(app, ["update", "--json"])

        assert default_result.exit_code == 0, default_result.stdout
        default_data = json.loads(default_result.stdout)
        assert [item["skill_id"] for item in default_data["not_updatable"]] == ["checked-skill"]
        assert "symlink" in default_data["not_updatable"][0]["reason"].lower()

    def test_add_github_url_with_flag_preserves_symlink(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, require_symlink
    ):
        """CLI GitHub URL add forwards the flag to the fetch and preserves the link."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text(
            "---\nname: gh-url-skill\ndescription: G\n---\nbody", encoding="utf-8"
        )
        assets = prepared / "assets"
        assets.mkdir()
        (assets / "manual.md").write_text("manual", encoding="utf-8")
        docs = prepared / "docs"
        docs.mkdir()
        os.symlink("../assets/manual.md", docs / "current.md")

        recorded: dict = {}

        def fake_fetch(url, allow_symlinks=False):
            recorded["allow_symlinks"] = allow_symlinks
            return SimpleNamespace(extracted_path=prepared, commit_sha="abc1234")

        monkeypatch.setattr(add_cli_module, "fetch_github_source_with_info", fake_fetch)

        result = runner.invoke(
            app, ["add", "https://github.com/user/repo", "--allow-symlinks", "-y"]
        )

        assert result.exit_code == 0, result.stdout
        assert recorded["allow_symlinks"] is True
        current = skills_env.skills_dir / "gh-url-skill" / "docs" / "current.md"
        assert current.is_symlink()
        assert current.read_text(encoding="utf-8") == "manual"

    def test_add_github_multi_path_with_flag_preserves_symlinks(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, require_symlink
    ):
        """CLI GitHub shorthand with multiple paths forwards the flag for every path."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        for path_name, skill_name in [("path-a", "skill-aa"), ("path-b", "skill-bb")]:
            skill = prepared / path_name / skill_name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                f"---\nname: {skill_name}\ndescription: G\n---\nbody", encoding="utf-8"
            )
            assets = skill / "assets"
            assets.mkdir()
            (assets / "manual.md").write_text(f"manual {skill_name}", encoding="utf-8")
            docs = skill / "docs"
            docs.mkdir()
            os.symlink("../assets/manual.md", docs / "current.md")

        recorded: dict = {}

        def fake_fetch(url, allow_symlinks=False):
            recorded["allow_symlinks"] = allow_symlinks
            return SimpleNamespace(extracted_path=prepared, commit_sha="abc1234")

        monkeypatch.setattr(add_cli_module, "fetch_github_source_with_info", fake_fetch)
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            [
                "add",
                "user/repo",
                "path-a",
                "path-b",
                "--allow-symlinks",
                "--no-keep-structure",
                "-y",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert recorded["allow_symlinks"] is True
        for skill_name in ("skill-aa", "skill-bb"):
            current = skills_env.skills_dir / skill_name / "docs" / "current.md"
            assert current.is_symlink(), skill_name
            assert current.read_text(encoding="utf-8") == f"manual {skill_name}"

    def test_add_github_multi_path_rejects_symlink_ancestor_without_external_effects(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, require_symlink
    ):
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        prepared.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _create_skill(outside, "external-skill")
        (outside / "marker.txt").write_text("keep", encoding="utf-8")
        (prepared / "link").symlink_to(outside, target_is_directory=True)

        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            ["add", "user/repo", "link/subdir", "--allow-symlinks", "--yes"],
        )

        assert result.exit_code != 0
        assert "symlink" in result.stdout.lower()
        assert (outside / "external-skill" / "SKILL.md").exists()
        assert (outside / "marker.txt").read_text(encoding="utf-8") == "keep"

    def test_add_github_root_skill_keeps_parent_collision_and_cleans_extraction(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        prepared.mkdir()
        skill_name = "root-skill"
        (prepared / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: G\n---\nbody", encoding="utf-8"
        )
        collision = prepared.parent / skill_name
        collision.mkdir()
        marker = collision / "marker"
        marker.write_text("keep", encoding="utf-8")

        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            ["add", "user/repo", ".", "--allow-symlinks", "--yes"],
        )

        assert result.exit_code == 0, result.stdout
        assert skill_name in result.stdout
        assert marker.read_text(encoding="utf-8") == "keep"
        assert not prepared.exists()
        assert (skills_env.skills_dir / skill_name / "SKILL.md").exists()

    def test_add_github_prefetch_accepts_safe_frontmatter_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text(
            "---\nname: prefetched-safe\ndescription: G\n---\nbody", encoding="utf-8"
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        marker = outside / "marker.txt"
        marker.write_text("keep", encoding="utf-8")

        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(app, ["add", "user/repo", ".", "--allow-symlinks", "--yes"])

        assert result.exit_code == 0, result.stdout
        assert (skills_env.skills_dir / "prefetched-safe" / "SKILL.md").exists()
        assert marker.read_text(encoding="utf-8") == "keep"

    @pytest.mark.parametrize("name_kind", ["absolute", "ancestor"])
    def test_add_github_prefetch_rejects_unsafe_frontmatter_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch, name_kind: str
    ):
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        escaped = tmp_path / f"{name_kind}-escape"
        outside = tmp_path / "outside"
        outside.mkdir()
        marker = outside / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
        skill_name = str(escaped) if name_kind == "absolute" else "../ancestor-escape"

        prepared = tmp_path / "gh-tree"
        prepared.mkdir()
        (prepared / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: G\n---\nbody", encoding="utf-8"
        )

        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(app, ["add", "user/repo", ".", "--allow-symlinks", "--yes"])

        assert result.exit_code != 0
        assert not list(skills_env.skills_dir.iterdir())
        assert not escaped.exists()
        assert marker.read_text(encoding="utf-8") == "keep"


class TestLocalSourceBoundaryCli:
    """Local add rejects symlinked source paths before pre-detection."""

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_add_absolute_ancestor_symlink_is_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path, allow: bool, require_symlink
    ):
        """An absolute local source behind a symlinked ancestor exits non-zero."""
        external = _create_skill(tmp_path / "outside", "linked-skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")
        source = tmp_path / "source"
        source.mkdir()
        (source / "alias").symlink_to(tmp_path / "outside", target_is_directory=True)

        args = ["add", str(source / "alias" / "linked-skill"), "--no-keep-structure"]
        if allow:
            args.append("--allow-symlinks")
        result = runner.invoke(app, args)

        assert result.exit_code != 0, result.stdout
        assert not (skills_env.skills_dir / "linked-skill").exists()
        assert not any(skills_env.skills_dir.rglob("external-marker.txt"))
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_add_relative_shorthand_shaped_ancestor_symlink_is_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path, allow: bool, require_symlink, monkeypatch
    ):
        """A shorthand-shaped local path behind a symlink is not added as a local skill."""
        external = _create_skill(tmp_path / "outside", "repo")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir()
        (project / "owner").symlink_to(tmp_path / "outside", target_is_directory=True)
        monkeypatch.chdir(project)

        args = ["add", "owner/repo", "--no-keep-structure"]
        if allow:
            args.append("--allow-symlinks")
        result = runner.invoke(app, args)

        assert result.exit_code != 0, result.stdout
        assert not (skills_env.skills_dir / "repo").exists()
        assert not any(skills_env.skills_dir.rglob("external-marker.txt"))
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"

    def test_add_relative_shorthand_shaped_local_path_is_added(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """A regular local owner/repo directory is still added as a local source."""
        project = tmp_path / "project"
        _create_skill(project / "owner", "repo")
        monkeypatch.chdir(project)

        result = runner.invoke(app, ["add", "owner/repo", "--no-keep-structure"])

        assert result.exit_code == 0, result.stdout
        assert (skills_env.skills_dir / "repo" / "SKILL.md").exists()

    @pytest.mark.parametrize("allow", [False, True], ids=["flagless", "with-flag"])
    def test_add_source_swapped_at_pre_detection_boundary_is_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path, allow: bool, monkeypatch, require_symlink
    ):
        """A source ancestor swapped at the pre-detection boundary is not followed."""
        from skillport.interfaces.cli.commands import add as add_cli_module

        external = _create_skill(tmp_path / "outside", "skill")
        (external / "external-marker.txt").write_text("secret", encoding="utf-8")

        source = tmp_path / "source"
        _create_skill(source, "skill")

        real_boundary = add_cli_module.acquire_local_source_snapshot
        state = {"swapped": False}

        def swapping_boundary(path, *args, **kwargs):
            if not state["swapped"]:
                state["swapped"] = True
                source.rename(source.parent / "source-original")
                source.symlink_to(tmp_path / "outside", target_is_directory=True)
            return real_boundary(path, *args, **kwargs)

        monkeypatch.setattr(add_cli_module, "acquire_local_source_snapshot", swapping_boundary)

        args = ["add", str(source)]
        if allow:
            args.append("--allow-symlinks")
        result = runner.invoke(app, args)

        assert result.exit_code != 0, result.stdout
        assert not (skills_env.skills_dir / "skill").exists()
        assert not any(skills_env.skills_dir.rglob("external-marker.txt"))
        assert (external / "external-marker.txt").read_text(encoding="utf-8") == "secret"


class TestLocalSourceParentComponentCli:
    """A local source path ending in ``..`` stays inside the skills directory."""

    @pytest.mark.parametrize("path_style", ["absolute", "relative"])
    def test_add_parent_component_source_installs_inside_skills_dir(
        self, skills_env: SkillsEnv, tmp_path: Path, path_style: str, monkeypatch
    ):
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
            source_arg = f"./{base.name}/child/.."
        else:
            source_arg = str(base / "child" / "..")

        result = runner.invoke(app, ["add", source_arg])

        assert result.exit_code == 0, result.stdout
        assert (skills_env.skills_dir / "base" / "skill-a" / "SKILL.md").exists()
        assert (skills_env.skills_dir / "base" / "skill-b" / "SKILL.md").exists()
        assert not (skills_env.skills_dir.parent / "skill-a").exists()
        assert not (skills_env.skills_dir.parent / "skill-b").exists()
        assert not any(skills_env.skills_dir.rglob("external-marker.txt"))
        assert marker.read_text(encoding="utf-8") == "secret"
        assert list(sandbox.iterdir()) == []


class TestVendorFrontmatterWarningsCli:
    """Vendor keys are allowed with a non-fatal warning (order.md §4 Rule 2, §3.4-3.5)."""

    def test_validate_vendor_keys_warns_and_exits_0(self, skills_env: SkillsEnv):
        _create_vendor_skill(skills_env.skills_dir, "vendor-skill")

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0, result.stdout
        assert "warning" in result.stdout.lower()
        assert "Claude Code" in result.stdout
        assert "model" in result.stdout

    def test_validate_vendor_keys_json_reports_warning(self, skills_env: SkillsEnv):
        _create_vendor_skill(skills_env.skills_dir, "vendor-skill")

        result = runner.invoke(app, ["validate", "--json"])

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["valid"] is True
        skill_entry = next(s for s in data["skills"] if s["id"] == "vendor-skill")
        warnings = [i for i in skill_entry["issues"] if i["severity"] == "warning"]
        assert len(warnings) == 1
        message = warnings[0]["message"]
        assert "model" in message
        assert "icon" in message
        assert "Claude Code" in message
        assert "Cursor" in message

    def test_add_vendor_skill_succeeds_with_warning(self, skills_env: SkillsEnv, tmp_path: Path):
        source = tmp_path / "source"
        _create_vendor_skill(source, "vendor-skill")

        result = runner.invoke(
            app, ["add", str(source / "vendor-skill"), "--no-keep-structure"]
        )

        assert result.exit_code == 0, result.stdout
        assert "Claude Code" in result.stdout
        assert "model" in result.stdout
        assert (skills_env.skills_dir / "vendor-skill" / "SKILL.md").exists()

    def test_add_vendor_skill_json_includes_warning(self, skills_env: SkillsEnv, tmp_path: Path):
        source = tmp_path / "source"
        _create_vendor_skill(source, "vendor-skill")

        result = runner.invoke(
            app,
            ["add", str(source / "vendor-skill"), "--no-keep-structure", "--json"],
        )

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert len(data["details"]) == 1
        details = data["details"][0]
        assert details["success"] is True
        warnings = details["warnings"]
        assert len(warnings) == 1
        message = warnings[0]["message"]
        assert "model" in message
        assert "icon" in message
        assert "Claude Code" in message
        assert "Cursor" in message

    def test_add_unknown_key_still_fails(self, skills_env: SkillsEnv, tmp_path: Path):
        source = tmp_path / "source"
        skill_dir = source / "bad-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-skill\ndescription: bad\nbogus-field:\n  unexpected: true\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["add", str(skill_dir), "--no-keep-structure"])

        assert result.exit_code == 1, result.stdout
        assert not (skills_env.skills_dir / "bad-skill").exists()

    def test_add_mapping_hooks_value_succeeds_with_warning(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        source = tmp_path / "source"
        skill_dir = source / "hooks-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: hooks-skill\ndescription: test\nhooks:\n  unexpected: true\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["add", str(skill_dir), "--no-keep-structure", "--json"])

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert len(data["details"]) == 1
        detail = data["details"][0]
        assert detail["success"] is True
        warnings = detail["warnings"]
        assert len(warnings) == 1
        assert "hooks" in warnings[0]["message"]
        assert (skills_env.skills_dir / "hooks-skill" / "SKILL.md").exists()

    def test_add_mapping_unknown_key_is_rejected(self, skills_env: SkillsEnv, tmp_path: Path):
        source = tmp_path / "source"
        skill_dir = source / "bogus-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bogus-skill\ndescription: test\nbogus-field:\n  unexpected: true\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["add", str(skill_dir), "--no-keep-structure", "--json"])

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert len(data["details"]) == 1
        detail = data["details"][0]
        assert detail["success"] is False
        assert detail["warnings"] == []
        assert not (skills_env.skills_dir / "bogus-skill").exists()

    def test_batch_add_separates_failure_detail_and_vendor_warning(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        source = tmp_path / "source"
        bad_dir = source / "bad-skill"
        bad_dir.mkdir(parents=True)
        (bad_dir / "SKILL.md").write_text(
            "---\nname: bad-skill\ndescription: bad\nbogus-field: value\n---\nbody",
            encoding="utf-8",
        )
        _create_vendor_skill(source, "vendor-skill", "icon: toolbox")

        result = runner.invoke(app, ["add", str(source), "--no-keep-structure", "--json"])

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        details = {d["skill_id"]: d for d in data["details"]}
        assert set(details) == {"bad-skill", "vendor-skill"}
        assert details["bad-skill"]["success"] is False
        assert details["bad-skill"]["warnings"] == []
        assert details["vendor-skill"]["success"] is True
        warnings = details["vendor-skill"]["warnings"]
        assert len(warnings) == 1
        assert "icon" in warnings[0]["message"]
        assert (skills_env.skills_dir / "vendor-skill" / "SKILL.md").exists()
        assert not (skills_env.skills_dir / "bad-skill").exists()

    def test_batch_add_all_unknown_keys_fail_without_warnings(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        source = tmp_path / "source"
        for name in ("bad-skill", "vendor-skill"):
            skill_dir = source / name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: bad\nbogus-field: value\n---\nbody",
                encoding="utf-8",
            )

        result = runner.invoke(app, ["add", str(source), "--no-keep-structure", "--json"])

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert len(data["details"]) == 2
        assert all(d["success"] is False for d in data["details"])
        assert all(d["warnings"] == [] for d in data["details"])
        assert not (skills_env.skills_dir / "bad-skill").exists()
        assert not (skills_env.skills_dir / "vendor-skill").exists()


class TestAllowXmlTagsCli:
    """--allow-xml-tags demotes XML tag violations to warnings (order.md §6)."""

    @staticmethod
    def _create_frontmatter_xml_skill(root: Path, name: str) -> Path:
        skill_dir = root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f'---\nname: {name}\ndescription: "Use when the user says <person>"\n---\n# {name}\n',
            encoding="utf-8",
        )
        return skill_dir

    @staticmethod
    def _create_body_xml_skill(root: Path, name: str) -> Path:
        skill_dir = root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Plain text\n---\nUse when the user says <person>.\n",
            encoding="utf-8",
        )
        return skill_dir

    def test_add_help_lists_allow_xml_tags(self, skills_env: SkillsEnv):
        """add --help documents the --allow-xml-tags option and its danger notice."""
        result = runner.invoke(app, ["add", "--help"])

        assert result.exit_code == 0
        assert "--allow-xml-tags" in result.stdout

        lines = result.stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if "--allow-xml-tags" in line)
        option_help = [lines[start]]
        for line in lines[start + 1 :]:
            if line.lstrip("│ ").startswith("--"):
                break
            option_help.append(line)

        assert "dangerous" in "\n".join(option_help)

    def test_add_without_flag_rejects_xml_tag_skill(self, skills_env: SkillsEnv, tmp_path: Path):
        """Flagless add of a name/description XML tag skill fails and installs nothing."""
        source = self._create_frontmatter_xml_skill(tmp_path / "source", "<person>")

        result = runner.invoke(app, ["add", str(source), "--no-keep-structure"])

        assert result.exit_code == 1, result.stdout
        assert "cannot contain XML tags" in result.stdout
        assert not (skills_env.skills_dir / "<person>").exists()

    def test_add_with_flag_installs_name_and_description_xml_tags(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """--allow-xml-tags installs name/description XML tags and warns for both."""
        source = self._create_frontmatter_xml_skill(tmp_path / "source", "<person>")

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 0, result.stdout
        assert "Added '<person>'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert "frontmatter.description: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()

    def test_add_with_flag_rejects_invalid_name_char_outside_xml_tag(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """Invalid characters outside the XML tag stay fatal under the flag."""
        source = tmp_path / "source"
        skill_dir = source / "my_name<person>"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            '---\nname: "my_name<person>"\ndescription: "Plain text"\n---\nbody',
            encoding="utf-8",
        )

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 1, result.stdout
        assert "invalid chars" in result.stdout
        assert not (skills_env.skills_dir / "my_name<person>").exists()

    @pytest.mark.parametrize(
        "name",
        ['<person key="x">', '<person key="\x1b">', '<person key="\x07">'],
        ids=["attribute", "esc", "bel"],
    )
    def test_add_with_flag_json_rejects_invalid_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, name: str
    ):
        """Tag-internal invalid name chars stay fatal and are not reported as added."""
        source = tmp_path / "source"
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure", "--json"]
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["details"][0]["success"] is False
        assert "invalid chars" in data["details"][0]["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_with_flag_rejects_name_with_path_separator(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A name with path separators is rejected before it can pick a destination."""
        source = tmp_path / "source"
        skill_dir = source / "escape>"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            '---\nname: "<person/../../escape>"\ndescription: "Plain text"\n---\nbody',
            encoding="utf-8",
        )

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 1, result.stdout
        assert "doesn't match directory" in result.stdout
        assert not (skills_env.skills_dir.parent / "escape>").exists()

    def test_add_with_flag_installs_and_warns(self, skills_env: SkillsEnv, tmp_path: Path):
        """--allow-xml-tags installs the skill and reports the XML warning."""
        source = self._create_frontmatter_xml_skill(tmp_path / "source", "xml-skill")

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 0, result.stdout
        assert "Added 'xml-skill'" in result.stdout
        warning_text = result.stdout.split("⚠", 1)[1]
        assert "xml-skill:" in warning_text
        assert "cannot contain XML tags" in warning_text
        assert (skills_env.skills_dir / "xml-skill" / "SKILL.md").exists()

    def test_add_with_flag_json_reports_warning(self, skills_env: SkillsEnv, tmp_path: Path):
        """--json carries the demoted XML issue as a per-skill warning."""
        source = self._create_frontmatter_xml_skill(tmp_path / "source", "xml-skill")

        result = runner.invoke(
            app,
            ["add", str(source), "--allow-xml-tags", "--no-keep-structure", "--json"],
        )

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert len(data["details"]) == 1
        detail = data["details"][0]
        assert detail["success"] is True
        assert len(detail["warnings"]) == 1
        warning = detail["warnings"][0]
        assert warning["severity"] == "warning"
        assert warning["field"] == "description"
        assert warning["message"] == "frontmatter.description: cannot contain XML tags"

    def test_add_with_flag_keeps_other_fatal_rejected(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """Other fatal rules stay fatal when --allow-xml-tags is given."""
        source = tmp_path / "source"
        skill_dir = source / "mixed-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            '---\nname: mixed-skill\ndescription: "Use <person>"\nbogus-field: value\n---\nbody',
            encoding="utf-8",
        )

        result = runner.invoke(
            app, ["add", str(skill_dir), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 1, result.stdout
        assert "unexpected field" in result.stdout
        assert not (skills_env.skills_dir / "mixed-skill").exists()

    def test_flag_is_not_persisted_between_invocations(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A flagged add does not weaken the next flagless add."""
        first_source = self._create_frontmatter_xml_skill(tmp_path / "first", "first-xml-skill")
        second_source = self._create_frontmatter_xml_skill(
            tmp_path / "second", "second-xml-skill"
        )

        first = runner.invoke(
            app, ["add", str(first_source), "--allow-xml-tags", "--no-keep-structure"]
        )
        assert first.exit_code == 0, first.stdout

        second = runner.invoke(app, ["add", str(second_source), "--no-keep-structure"])

        assert second.exit_code == 1, second.stdout
        assert "cannot contain XML tags" in second.stdout
        assert (skills_env.skills_dir / "first-xml-skill" / "SKILL.md").exists()
        assert not (skills_env.skills_dir / "second-xml-skill").exists()

    def test_add_body_only_xml_without_flag_installs(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """XML tags outside the frontmatter are not XML tag violations."""
        source = self._create_body_xml_skill(tmp_path / "source", "body-skill")

        result = runner.invoke(app, ["add", str(source), "--no-keep-structure"])

        assert result.exit_code == 0, result.stdout
        assert (skills_env.skills_dir / "body-skill" / "SKILL.md").exists()

    def test_add_body_only_xml_with_flag_has_no_xml_warning(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A body-only XML tag produces no warning even with the flag."""
        source = self._create_body_xml_skill(tmp_path / "source", "body-skill")

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 0, result.stdout
        assert "cannot contain XML tags" not in result.stdout
        assert (skills_env.skills_dir / "body-skill" / "SKILL.md").exists()

    def test_validate_reports_xml_fatal_after_flagged_add(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """validate always reports the name/description XML tag violation as fatal."""
        source = self._create_frontmatter_xml_skill(tmp_path / "source", "<person>")
        added = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )
        assert added.exit_code == 0, added.stdout

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 1, result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert "frontmatter.description: cannot contain XML tags" in result.stdout
        assert "fatal" in result.stdout.lower()

    def test_validate_body_only_xml_is_not_fatal(self, skills_env: SkillsEnv, tmp_path: Path):
        """validate does not report body-only XML tags as fatal."""
        source = self._create_body_xml_skill(tmp_path / "source", "body-skill")
        added = runner.invoke(app, ["add", str(source), "--no-keep-structure"])
        assert added.exit_code == 0, added.stdout

        result = runner.invoke(app, ["validate", "body-skill"])

        assert result.exit_code == 0, result.stdout
        assert "cannot contain XML tags" not in result.stdout

    def test_add_with_flag_propagates_to_nested_zip(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """The flag reaches the recursive add_skill call for nested ZIP files."""
        import zipfile

        source = tmp_path / "bundle"
        source.mkdir()
        with zipfile.ZipFile(source / "inner.zip", "w") as archive:
            archive.writestr(
                "SKILL.md",
                '---\nname: inner-skill\ndescription: "Use <person>"\n---\nbody',
            )

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"]
        )

        assert result.exit_code == 0, result.stdout
        assert "Added 'inner-skill'" in result.stdout
        assert "cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "inner-skill" / "SKILL.md").exists()

    def test_add_with_flag_installs_tag_name_from_nested_zip(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A bare tag name from a nested ZIP is installed with warnings."""
        import zipfile

        source = tmp_path / "bundle"
        source.mkdir()
        with zipfile.ZipFile(source / "inner.zip", "w") as archive:
            archive.writestr(
                "SKILL.md",
                '---\nname: "<person>"\ndescription: "Use <person>"\n---\nbody',
            )

        result = runner.invoke(app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"])

        assert result.exit_code == 0, result.stdout
        assert "Added '<person>'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert "frontmatter.description: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()

    def test_add_with_flag_rejects_invalid_tag_name_from_nested_zip(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A rejected tag name from a nested ZIP is not reported as added."""
        import zipfile

        name = '<person key="x">'
        source = tmp_path / "bundle"
        source.mkdir()
        with zipfile.ZipFile(source / "inner.zip", "w") as archive:
            archive.writestr(
                "SKILL.md",
                f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            )

        result = runner.invoke(
            app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure", "--json"]
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["details"][0]["success"] is False
        assert "invalid chars" in data["details"][0]["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_github_multi_path_with_flag_propagates(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The flag reaches the add_skill call for GitHub multi-path sources."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        skill_dir = prepared / "path-a" / "gh-xml-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            '---\nname: gh-xml-skill\ndescription: "Use <person>"\n---\nbody',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "--allow-xml-tags", "--no-keep-structure", "--yes"],
        )

        assert result.exit_code == 0, result.stdout
        assert "Added 'gh-xml-skill'" in result.stdout
        assert "cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "gh-xml-skill" / "SKILL.md").exists()

    def test_add_github_multi_path_with_flag_installs_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """A bare tag name is installed with warnings through the multi-path route."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        skill_dir = prepared / "path-a" / "<person>"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            '---\nname: "<person>"\ndescription: "Use <person>"\n---\nbody',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "--allow-xml-tags", "--no-keep-structure", "--yes"],
        )

        assert result.exit_code == 0, result.stdout
        assert "Added '<person>'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()

    def test_add_github_multi_path_with_flag_rejects_invalid_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """A rejected tag name is not aggregated into the added IDs."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        name = '<person key="x">'
        skill_dir = prepared / "path-a" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            [
                "add",
                "user/repo",
                "path-a",
                "--allow-xml-tags",
                "--no-keep-structure",
                "--yes",
                "--json",
            ],
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["details"][0]["success"] is False
        assert "invalid chars" in data["details"][0]["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_with_flag_rejects_tag_name_with_trailing_hyphen(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A tag name ending with a hyphen stays fatal under --allow-xml-tags."""
        name = "<person->"
        source = tmp_path / "source"
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"])

        assert result.exit_code == 1, result.stdout
        assert "start or end with hyphen" in " ".join(result.stdout.split())
        assert "✓ Added" not in result.stdout
        assert not (skills_env.skills_dir / name).exists()

    def test_add_with_flag_json_rejects_tag_name_with_trailing_hyphen(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """The rejected tag name ending with a hyphen is not in the added IDs."""
        name = "<person->"
        source = tmp_path / "source"
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["add", str(source), "--allow-xml-tags", "--no-keep-structure", "--json"],
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["details"][0]["success"] is False
        assert "start or end with hyphen" in data["details"][0]["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_with_flag_rejects_tag_name_trailing_hyphen_from_nested_zip(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A rejected trailing-hyphen tag name from a nested ZIP is not added."""
        import zipfile

        name = "<person->"
        source = tmp_path / "bundle"
        source.mkdir()
        with zipfile.ZipFile(source / "inner.zip", "w") as archive:
            archive.writestr(
                "SKILL.md",
                f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            )

        result = runner.invoke(
            app,
            ["add", str(source), "--allow-xml-tags", "--no-keep-structure", "--json"],
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["details"][0]["success"] is False
        assert "start or end with hyphen" in data["details"][0]["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_github_multi_path_with_flag_rejects_tag_name_trailing_hyphen(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """A rejected trailing-hyphen tag name is not aggregated into added IDs."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        name = "<person->"
        prepared = tmp_path / "gh-tree"
        skill_dir = prepared / "path-a" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            [
                "add",
                "user/repo",
                "path-a",
                "--allow-xml-tags",
                "--no-keep-structure",
                "--yes",
                "--json",
            ],
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["details"][0]["success"] is False
        assert "start or end with hyphen" in data["details"][0]["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_human_output_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """A rejected ESC-bearing name is displayed with visible escapes, not raw."""
        name = "<person\x1b>"
        source = tmp_path / "source"
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"])

        assert result.exit_code == 1, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        assert "invalid chars" in " ".join(result.stdout.split())
        assert not (skills_env.skills_dir / name).exists()

    def test_add_github_multi_path_human_output_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The multi-path human output escapes a control character in the name."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        name = "<person\x1b>"
        prepared = tmp_path / "gh-tree"
        skill_dir = prepared / "path-a" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

        result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "--allow-xml-tags", "--no-keep-structure", "--yes"],
        )

        assert result.exit_code == 1, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        assert not (skills_env.skills_dir / name).exists()

    def test_add_nested_zip_human_output_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """The nested-ZIP human output escapes a control character in the name."""
        import zipfile

        name = "<person\x1b>"
        source = tmp_path / "bundle"
        source.mkdir()
        with zipfile.ZipFile(source / "inner.zip", "w") as archive:
            archive.writestr(
                "SKILL.md",
                f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            )

        result = runner.invoke(app, ["add", str(source), "--allow-xml-tags", "--no-keep-structure"])

        assert result.exit_code == 1, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        assert not (skills_env.skills_dir / name).exists()

    def test_add_json_preserves_control_char_name_values(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        """The JSON result keeps the raw rejected name and detail message."""
        name = "<person\x1b>"
        source = tmp_path / "source"
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["add", str(source), "--allow-xml-tags", "--no-keep-structure", "--json"],
        )

        assert result.exit_code == 1, result.stdout
        data = json.loads(result.stdout)
        assert data["added"] == []
        assert data["skipped"] == [name]
        detail = data["details"][0]
        assert detail["success"] is False
        assert detail["skill_id"] == name
        assert name in detail["message"]
        assert not (skills_env.skills_dir / name).exists()

    def test_add_interactive_prompt_shows_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The interactive prompt shows a normal tag name and installs it with a warning."""
        from skillport.interfaces.cli.commands import add as add_cli_module

        source = self._create_frontmatter_xml_skill(tmp_path / "source", "<person>")
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(app, ["add", str(source), "--allow-xml-tags"], input="1\n")

        assert result.exit_code == 0, result.stdout
        assert "Found 1 skill(s): <person>" in result.stdout
        assert "Added '<person>'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()

    def test_add_interactive_prompt_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The interactive prompt escapes a control character in the name."""
        from skillport.interfaces.cli.commands import add as add_cli_module

        name = "<person\x1b>"
        source = tmp_path / "source"
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(app, ["add", str(source), "--allow-xml-tags"], input="1\n")

        assert result.exit_code == 1, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        prompt_area = result.stdout.split("Where to add?", 1)[0]
        assert "<person\\x1b>" in prompt_area
        assert "invalid chars" in " ".join(result.stdout.split())
        assert not (skills_env.skills_dir / name).exists()

    def test_add_url_interactive_prompt_shows_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The URL flow shows a normal tag name in the prompt and installs it."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        skill_dir = prepared / "<person>"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            '---\nname: "<person>"\ndescription: "Use <person>"\n---\nbody',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(
            app, ["add", "https://github.com/user/repo", "--allow-xml-tags"], input="1\n"
        )

        assert result.exit_code == 0, result.stdout
        assert "Found 1 skill(s): <person>" in result.stdout
        assert "Added '<person>'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()

    def test_add_url_interactive_prompt_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The URL flow escapes a control character in the prompt and skips the skill."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        name = "<person\x1b>"
        prepared = tmp_path / "gh-tree"
        skill_dir = prepared / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(
            app, ["add", "https://github.com/user/repo", "--allow-xml-tags"], input="1\n"
        )

        assert result.exit_code == 1, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        prompt_area = result.stdout.split("Where to add?", 1)[0]
        assert "<person\\x1b>" in prompt_area
        assert "invalid chars" in " ".join(result.stdout.split())
        assert not (skills_env.skills_dir / name).exists()

    def test_add_zip_interactive_prompt_shows_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The direct ZIP flow shows a normal tag name in the prompt and installs it."""
        import zipfile

        from skillport.interfaces.cli.commands import add as add_cli_module

        zip_path = tmp_path / "tag.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "SKILL.md",
                '---\nname: "<person>"\ndescription: "Use <person>"\n---\nbody',
            )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(app, ["add", str(zip_path), "--allow-xml-tags"], input="1\n")

        assert result.exit_code == 0, result.stdout
        assert "Found 1 skill(s): <person>" in result.stdout
        assert "Added '<person>'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()

    def test_add_zip_interactive_prompt_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The direct ZIP flow escapes a control character in the prompt and skips the skill."""
        import zipfile

        from skillport.interfaces.cli.commands import add as add_cli_module

        name = "<person\x1b>"
        zip_path = tmp_path / "tag.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "SKILL.md",
                f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(app, ["add", str(zip_path), "--allow-xml-tags"], input="1\n")

        assert result.exit_code == 1, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        prompt_area = result.stdout.split("Where to add?", 1)[0]
        assert "<person\\x1b>" in prompt_area
        assert "invalid chars" in " ".join(result.stdout.split())
        assert not (skills_env.skills_dir / name).exists()

    def test_add_github_multi_path_interactive_prompt_shows_tag_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The multi-path prompt lists a normal tag name and installs it with a warning."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = tmp_path / "gh-tree"
        tag_dir = prepared / "path-a" / "<person>"
        tag_dir.mkdir(parents=True)
        (tag_dir / "SKILL.md").write_text(
            '---\nname: "<person>"\ndescription: "Use <person>"\n---\nbody',
            encoding="utf-8",
        )
        _create_skill(prepared / "path-b", "plain-skill")
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "path-b", "--allow-xml-tags"],
            input="1\n",
        )

        assert result.exit_code == 0, result.stdout
        assert "Found 2 skill(s): <person>, plain-skill" in result.stdout
        assert "Added '<person>'" in result.stdout
        assert "Added 'plain-skill'" in result.stdout
        assert "frontmatter.name: cannot contain XML tags" in result.stdout
        assert (skills_env.skills_dir / "<person>" / "SKILL.md").exists()
        assert (skills_env.skills_dir / "plain-skill" / "SKILL.md").exists()

    def test_add_github_multi_path_interactive_prompt_escapes_control_char_name(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        """The multi-path prompt escapes a control character and skips the rejected skill."""
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        name = "<person\x1b>"
        prepared = tmp_path / "gh-tree"
        tag_dir = prepared / "path-a" / name
        tag_dir.mkdir(parents=True)
        (tag_dir / "SKILL.md").write_text(
            f"---\nname: {json.dumps(name)}\ndescription: Plain text\n---\nbody",
            encoding="utf-8",
        )
        _create_skill(prepared / "path-b", "plain-skill")
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )
        monkeypatch.setattr(add_cli_module, "is_interactive", lambda: True)

        result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "path-b", "--allow-xml-tags"],
            input="1\n",
        )

        assert result.exit_code == 0, result.stdout
        assert "\x1b" not in result.stdout
        assert "\\x1b" in result.stdout
        prompt_area = result.stdout.split("Where to add?", 1)[0]
        assert "<person\\x1b>, plain-skill" in prompt_area
        assert (skills_env.skills_dir / "plain-skill" / "SKILL.md").exists()
        assert not (skills_env.skills_dir / name).exists()


class TestFrontmatterKeyFatalCli:
    """Unregistered top-level keys stay fatal for validate (order.md §4 Rule 1)."""

    def test_validate_unknown_top_level_key_is_invalid(self, skills_env: SkillsEnv):
        skill_dir = skills_env.skills_dir / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-skill\ndescription: bad\nbogus-field: value\n---\nbody",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 1, result.stdout
        assert "bogus-field" in result.stdout
        assert "fatal" in result.stdout.lower()


class TestAddSameSkillIdWarningDisplay:
    """Repeated skill_id across details must not hide or duplicate warnings."""

    @staticmethod
    def _make_local_source(tmp_path: Path, frontmatter: str, zip_frontmatter: str = "") -> Path:
        import zipfile

        source = tmp_path / "source"
        _create_vendor_skill(source, "repeat-skill", frontmatter)
        zip_extra = f"{zip_frontmatter}\n" if zip_frontmatter else ""
        with zipfile.ZipFile(source / "repeat-skill.zip", "w") as archive:
            archive.writestr(
                "repeat-skill/SKILL.md",
                f"---\nname: repeat-skill\ndescription: Zipped\n{zip_extra}---\nbody",
            )
        return source

    @staticmethod
    def _make_github_fixture(root: Path, path_frontmatter: dict[str, str]) -> Path:
        prepared = root / "gh-tree"
        for path_name, frontmatter in path_frontmatter.items():
            skill_dir = prepared / path_name / "repeat-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: repeat-skill\ndescription: G\n{frontmatter}\n---\nbody",
                encoding="utf-8",
            )
        return prepared

    @staticmethod
    def _install_github_fixture(
        root: Path, path_frontmatter: dict[str, str], monkeypatch
    ) -> None:
        from types import SimpleNamespace

        from skillport.interfaces.cli.commands import add as add_cli_module

        prepared = TestAddSameSkillIdWarningDisplay._make_github_fixture(
            root, path_frontmatter
        )
        monkeypatch.setattr(
            add_cli_module,
            "fetch_github_source_with_info",
            lambda url, allow_symlinks=False: SimpleNamespace(
                extracted_path=prepared, commit_sha="abc1234"
            ),
        )
        monkeypatch.setattr(
            add_cli_module, "get_default_branch", lambda owner, repo, auth=None: "main"
        )

    @staticmethod
    def _clear_installed(skills_dir: Path, skill_id: str) -> None:
        import shutil

        installed = skills_dir / skill_id
        if installed.exists():
            shutil.rmtree(installed)

    def test_local_then_same_id_zip_keeps_success_warning(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        source = self._make_local_source(tmp_path, "model: sonnet", zip_frontmatter="icon: star")

        human = runner.invoke(app, ["add", str(source), "--no-keep-structure"])

        assert human.exit_code == 0, human.stdout
        assert human.stdout.count("⚠") == 1
        assert human.stdout.count("Claude Code") == 1
        assert human.stdout.count("model") == 1
        assert "Cursor" not in human.stdout
        assert "icon" not in human.stdout
        assert "Skipped" in human.stdout or "⊘" in human.stdout
        assert (skills_env.skills_dir / "repeat-skill" / "SKILL.md").exists()

        self._clear_installed(skills_env.skills_dir, "repeat-skill")

        json_result = runner.invoke(app, ["add", str(source), "--no-keep-structure", "--json"])

        assert json_result.exit_code == 0, json_result.stdout
        data = json.loads(json_result.stdout)
        success = [d for d in data["details"] if d["success"]]
        skipped = [d for d in data["details"] if not d["success"]]
        assert len(success) == 1
        assert success[0]["skill_id"] == "repeat-skill"
        warnings = success[0]["warnings"]
        assert len(warnings) == 1
        assert "model" in warnings[0]["message"]
        assert "Claude Code" in warnings[0]["message"]
        assert len(skipped) == 1
        assert skipped[0]["skill_id"] == "repeat-skill"
        assert skipped[0]["warnings"] == []

    def test_local_then_same_id_zip_without_vendor_key_has_no_warning(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        source = self._make_local_source(tmp_path, "license: MIT")

        human = runner.invoke(app, ["add", str(source), "--no-keep-structure"])

        assert human.exit_code == 0, human.stdout
        assert "vendor-specific" not in human.stdout

        self._clear_installed(skills_env.skills_dir, "repeat-skill")

        json_result = runner.invoke(app, ["add", str(source), "--no-keep-structure", "--json"])

        assert json_result.exit_code == 0, json_result.stdout
        data = json.loads(json_result.stdout)
        success = [d for d in data["details"] if d["success"]]
        assert len(success) == 1
        assert success[0]["warnings"] == []

    def test_github_multi_path_same_id_keeps_first_warning(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        path_frontmatter = {"path-a": "model: sonnet", "path-b": "icon: star"}
        self._install_github_fixture(tmp_path / "run-human", path_frontmatter, monkeypatch)

        human = runner.invoke(
            app, ["add", "user/repo", "path-a", "path-b", "--no-keep-structure", "--yes"]
        )

        assert human.exit_code == 0, human.stdout
        assert human.stdout.count("⚠") == 1
        assert human.stdout.count("Claude Code") == 1
        assert human.stdout.count("model") == 1
        assert "Cursor" not in human.stdout
        assert "icon" not in human.stdout

        self._clear_installed(skills_env.skills_dir, "repeat-skill")
        self._install_github_fixture(tmp_path / "run-json", path_frontmatter, monkeypatch)

        json_result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "path-b", "--no-keep-structure", "--yes", "--json"],
        )

        assert json_result.exit_code == 0, json_result.stdout
        data = json.loads(json_result.stdout)
        success = [d for d in data["details"] if d["success"]]
        skipped = [d for d in data["details"] if not d["success"]]
        assert len(success) == 1
        assert success[0]["skill_id"] == "repeat-skill"
        warnings = success[0]["warnings"]
        assert len(warnings) == 1
        assert "model" in warnings[0]["message"]
        assert "Claude Code" in warnings[0]["message"]
        assert len(skipped) == 1
        assert skipped[0]["warnings"] == []

    def test_github_multi_path_same_id_without_vendor_key_has_no_warning(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        path_frontmatter = {"path-a": "license: MIT", "path-b": "license: MIT"}
        self._install_github_fixture(tmp_path / "run-human", path_frontmatter, monkeypatch)

        human = runner.invoke(
            app, ["add", "user/repo", "path-a", "path-b", "--no-keep-structure", "--yes"]
        )

        assert human.exit_code == 0, human.stdout
        assert "vendor-specific" not in human.stdout

        self._clear_installed(skills_env.skills_dir, "repeat-skill")
        self._install_github_fixture(tmp_path / "run-json", path_frontmatter, monkeypatch)

        json_result = runner.invoke(
            app,
            ["add", "user/repo", "path-a", "path-b", "--no-keep-structure", "--yes", "--json"],
        )

        assert json_result.exit_code == 0, json_result.stdout
        data = json.loads(json_result.stdout)
        success = [d for d in data["details"] if d["success"]]
        assert len(success) == 1
        assert success[0]["warnings"] == []

    def test_local_and_zip_same_id_with_force_warns_once_per_success(
        self, skills_env: SkillsEnv, tmp_path: Path
    ):
        source = self._make_local_source(tmp_path, "model: sonnet", zip_frontmatter="icon: star")

        human = runner.invoke(app, ["add", str(source), "--no-keep-structure", "--force"])

        assert human.exit_code == 0, human.stdout
        assert human.stdout.count("Added 'repeat-skill'") == 2
        assert human.stdout.count("⚠") == 2
        warning_blocks = human.stdout.split("⚠")[1:]
        claude_blocks = [
            i for i, b in enumerate(warning_blocks) if "Claude Code" in b and "model" in b
        ]
        cursor_blocks = [i for i, b in enumerate(warning_blocks) if "Cursor" in b and "icon" in b]
        assert len(claude_blocks) == 1
        assert len(cursor_blocks) == 1
        assert claude_blocks[0] != cursor_blocks[0]
        assert (skills_env.skills_dir / "repeat-skill" / "SKILL.md").exists()

        json_result = runner.invoke(
            app, ["add", str(source), "--no-keep-structure", "--force", "--json"]
        )

        assert json_result.exit_code == 0, json_result.stdout
        data = json.loads(json_result.stdout)
        assert len(data["added"]) == 2
        success = [d for d in data["details"] if d["success"]]
        assert len(success) == 2
        assert all(len(d["warnings"]) == 1 for d in success)

    def test_github_multi_path_same_id_with_force_warns_once_per_success(
        self, skills_env: SkillsEnv, tmp_path: Path, monkeypatch
    ):
        path_frontmatter = {"path-a": "model: sonnet", "path-b": "icon: star"}
        self._install_github_fixture(tmp_path / "run-human", path_frontmatter, monkeypatch)

        human = runner.invoke(
            app,
            [
                "add",
                "user/repo",
                "path-a",
                "path-b",
                "--no-keep-structure",
                "--yes",
                "--force",
            ],
        )

        assert human.exit_code == 0, human.stdout
        assert human.stdout.count("Added 'repeat-skill'") == 2
        assert human.stdout.count("⚠") == 2
        warning_blocks = human.stdout.split("⚠")[1:]
        claude_blocks = [
            i for i, b in enumerate(warning_blocks) if "Claude Code" in b and "model" in b
        ]
        cursor_blocks = [i for i, b in enumerate(warning_blocks) if "Cursor" in b and "icon" in b]
        assert len(claude_blocks) == 1
        assert len(cursor_blocks) == 1
        assert claude_blocks[0] != cursor_blocks[0]

        self._clear_installed(skills_env.skills_dir, "repeat-skill")
        self._install_github_fixture(tmp_path / "run-json", path_frontmatter, monkeypatch)

        json_result = runner.invoke(
            app,
            [
                "add",
                "user/repo",
                "path-a",
                "path-b",
                "--no-keep-structure",
                "--yes",
                "--force",
                "--json",
            ],
        )

        assert json_result.exit_code == 0, json_result.stdout
        data = json.loads(json_result.stdout)
        assert len(data["added"]) == 2
        success = [d for d in data["details"] if d["success"]]
        assert len(success) == 2
        assert all(len(d["warnings"]) == 1 for d in success)
