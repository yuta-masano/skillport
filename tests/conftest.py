"""Shared pytest fixtures for SkillPort tests."""

import errno
import os
import shutil
from pathlib import Path

import pytest

from skillport.shared.config import Config


@pytest.fixture
def no_follow_unavailable(monkeypatch):
    """Simulate a platform without O_NOFOLLOW, descriptor-relative opens, or
    directory descriptors (for example Windows)."""
    monkeypatch.setattr(os, "supports_dir_fd", set())
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    # shutil only uses descriptor-relative removal where the platform offers
    # it; keep its own cleanup off the emulated directory descriptors.
    monkeypatch.setattr(shutil, "_use_fd_functions", False)

    real_open = os.open

    def open_without_directory_descriptors(path, flags, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and os.path.isdir(path):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_without_directory_descriptors)


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Create a Config that uses temporary directories.

    This ensures tests don't pollute ~/.skillport/ with indexes or metadata.
    """
    return Config(
        skills_dir=tmp_path / "skills",
        db_path=tmp_path / "index" / "skills.lancedb",
    )
