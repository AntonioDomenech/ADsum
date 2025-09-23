import sqlite3

from adsum.data.models import NoteDocument
from adsum.data.storage import SessionStore


def test_save_notes_upserts_on_session_id(tmp_path):
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    store.initialize()

    original = NoteDocument(
        session_id="session-123",
        title="Initial",
        summary="First summary",
        action_items=["a"],
    )
    updated = NoteDocument(
        session_id="session-123",
        title="Updated",
        summary="Revised summary",
        action_items=["b"],
    )

    store.save_notes(original)
    store.save_notes(updated)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE session_id = ?", ("session-123",)
        ).fetchone()[0]

    assert count == 1

    fetched = store.fetch_notes("session-123")
    assert fetched is not None
    assert fetched.title == updated.title
    assert fetched.summary == updated.summary
    assert fetched.action_items == updated.action_items

