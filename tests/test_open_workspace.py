from pathlib import Path

import pytest

from cua_jev.models import Channel, Observation
from cua_jev.open_browser import BrowserElement, BrowserSnapshot
from cua_jev.open_workspace import OpenWorkspaceTask, WorkspacePlan, WorkspaceSnapshot


def snapshot(*, editor_open: bool = False, note_exists: bool = False) -> WorkspaceSnapshot:
    browser = BrowserSnapshot(
        "https://example.org/releases", "Releases", "Version 5.2, released May 4, 2026",
        (BrowserElement("e2", 2, "a", "", "Details", "", False,
                        "https://example.org/releases/5.2"),),
    )
    return WorkspaceSnapshot(
        browser, "C:/workspace/note.md", note_exists, "Version 5.2" if note_exists else "",
        editor_open, "note.md - Visual Studio Code" if editor_open else "",
    )


def test_workspace_plan_grounded_browser_link() -> None:
    plan = WorkspacePlan.from_dict(
        {"subgoal": "Open details", "options": [{"operation": "browser_click", "ref": "e2"}]},
        snapshot(),
    )
    assert plan.options[0].ref == "e2"
    with pytest.raises(ValueError, match="live link"):
        WorkspacePlan.from_dict(
            {"subgoal": "Open details", "options": [{"operation": "browser_click", "ref": "e9"}]},
            snapshot(),
        )


def test_workspace_plan_respects_live_operation_gate() -> None:
    current = snapshot()
    gated = WorkspaceSnapshot(
        current.browser, current.note_path, False, "", False, "", ("browser_click",)
    )
    with pytest.raises(ValueError, match="unavailable now"):
        WorkspacePlan.from_dict(
            {"subgoal": "Write", "options": [{"operation": "write_note", "text": "Version 5.2",
             "evidence": "Version 5.2"}]}, gated,
        )


def test_workspace_plan_rejects_unseen_fact_and_overwrite() -> None:
    payload = {
        "subgoal": "Write note",
        "options": [{"operation": "write_note", "text": "Version 5.2", "evidence": "Version 5.2"}],
    }
    assert WorkspacePlan.from_dict(payload, snapshot()).options[0].operation == "write_note"
    with pytest.raises(ValueError, match="already exists"):
        WorkspacePlan.from_dict(payload, snapshot(note_exists=True))
    payload["options"][0]["evidence"] = "Version 6.0"
    with pytest.raises(ValueError, match="visible browser"):
        WorkspacePlan.from_dict(payload, snapshot())


def test_workspace_candidates_offer_real_hybrid_routes(tmp_path: Path) -> None:
    class Planner:
        def plan(self, goal, snap, recent):
            return WorkspacePlan.from_dict(
                {"subgoal": "Write", "options": [{"operation": "write_note",
                 "text": "Version 5.2", "evidence": "Version 5.2"}]}, snap,
            )

    task = OpenWorkspaceTask(
        goal="Write a note", start_url="https://example.org/releases",
        note_path=tmp_path / "note.md", planner=Planner(), editor_probe=lambda: "editor",
    )
    task._snapshot = snapshot(editor_open=True)
    observation = Observation("Write a note", "Write", task._snapshot.to_dict())
    candidates = task.candidates(observation, [])
    assert {item.channel for item in candidates} == {Channel.API, Channel.GUI}
    assert all(item.arguments["text"].endswith("Source: https://example.org/releases\n")
               for item in candidates)


def test_workspace_refuses_to_overwrite_existing_note(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("user data", encoding="utf-8")
    task = OpenWorkspaceTask(
        goal="Write note", start_url="https://example.org/", note_path=note,
        planner=object(), editor_probe=lambda: "",
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        task.reset()
    assert note.read_text(encoding="utf-8") == "user data"
