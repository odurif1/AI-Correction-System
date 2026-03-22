"""
Tests for storage layout and CLI/session resolution invariants.
"""

from pathlib import Path

from src.core.models import GradingSession
from src.main import resolve_session_user_id
from src.scripts.read_debug import find_debug_log
from src.storage.file_store import SessionIndex, SessionStore


def test_session_store_writes_index_under_sessions_root(tmp_path):
    store = SessionStore("session-1", user_id="cli_user", base_dir=str(tmp_path))
    store.save_session(GradingSession(session_id="session-1", user_id="cli_user"))

    assert (tmp_path / "sessions" / "cli_user" / "_index.json").exists()
    assert not (tmp_path / "cli_user" / "_index.json").exists()


def test_session_index_reads_user_specific_index_under_sessions_root(tmp_path):
    store = SessionStore("session-1", user_id="teacher-1", base_dir=str(tmp_path))
    store.save_session(GradingSession(session_id="session-1", user_id="teacher-1"))

    index = SessionIndex(user_id="teacher-1", base_dir=str(tmp_path))

    assert index.list_sessions() == ["session-1"]
    assert index.get_session_info("session-1")["user_id"] == "teacher-1"


def test_resolve_session_user_id_scans_session_roots(monkeypatch, tmp_path):
    session_dir = tmp_path / "sessions" / "teacher-1" / "session-abc"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("src.main.DATA_DIR", str(tmp_path))

    assert resolve_session_user_id("session-abc") == "teacher-1"


def test_find_debug_log_searches_all_user_roots(monkeypatch, tmp_path):
    debug_log = tmp_path / "sessions" / "teacher-1" / "session-abc" / "debug" / "debug_log.json"
    debug_log.parent.mkdir(parents=True)
    debug_log.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("src.scripts.read_debug.DATA_DIR", str(tmp_path))

    assert find_debug_log("session-abc") == debug_log
