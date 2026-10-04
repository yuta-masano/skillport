"""Unit-test-only pytest fixtures for SkillPort."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_skillport_indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure unit tests don't write under ~/.skillport/indexes.

    Tests that build Config() or Config(skills_dir=...) without an explicit
    db_path/meta_dir would otherwise derive paths under ~/.skillport/indexes and
    write real index data there.
    """
    monkeypatch.setenv("SKILLPORT_DB_PATH", str(tmp_path / "index" / "skills.lancedb"))
    monkeypatch.setenv("SKILLPORT_META_DIR", str(tmp_path / "index" / "meta"))
