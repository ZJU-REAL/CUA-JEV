from pathlib import Path

import pytest

from cua_jev.models import Channel, Observation
from cua_jev.open_browser import BrowserElement, BrowserSnapshot
from cua_jev.open_workspace import OpenWorkspaceTask, WorkspacePlan, WorkspaceSnapshot, _ground_quote


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


def test_research_evidence_is_grounded_in_distinct_heading_and_page() -> None:
    base = snapshot()
    research = WorkspaceSnapshot(
        base.browser, base.note_path, False, "", False, "", ("record_evidence",),
        "Using the interpreter starts a Python prompt for interactive commands.",
        "Using the Python Interpreter", ("Using the Python Interpreter", "Data Structures"),
    )
    payload = {"subgoal": "Save a source quote", "options": [{
        "operation": "record_evidence", "topic": "Using the Python Interpreter",
        "evidence": "Using the interpreter starts a Python prompt",
    }]}
    assert WorkspacePlan.from_dict(payload, research).options[0].topic == (
        "Using the Python Interpreter"
    )
    payload["options"][0]["evidence"] = "Invented quote"
    with pytest.raises(ValueError, match="main-page text"):
        WorkspacePlan.from_dict(payload, research)
    payload["options"][0]["evidence"] = "Using the interpreter starts a Python prompt"
    payload["options"][0]["topic"] = "Data Structures"
    with pytest.raises(ValueError, match="heading"):
        WorkspacePlan.from_dict(payload, research)


def test_research_note_requires_all_topics_and_appends_citations(tmp_path: Path) -> None:
    base = snapshot()
    evidence = ({
        "topic": "Using the Python Interpreter", "quote": "An observed exact quotation",
        "url": "https://example.org/releases",
    },)
    incomplete = WorkspaceSnapshot(
        base.browser, str(tmp_path / "note.md"), False, "", False, "", ("write_note",),
        "", "", ("Using the Python Interpreter", "Data Structures"), evidence,
    )
    payload = {"subgoal": "Write summary", "options": [{
        "operation": "write_note", "text": "A source-backed guide.",
    }]}
    with pytest.raises(ValueError, match="all research topics"):
        WorkspacePlan.from_dict(payload, incomplete)
    complete = WorkspaceSnapshot(
        base.browser, str(tmp_path / "note.md"), False, "", False, "", ("write_note",),
        "", "", ("Using the Python Interpreter",), evidence,
    )

    class Planner:
        def plan(self, goal, snap, recent):
            return WorkspacePlan.from_dict(payload, snap)

    task = OpenWorkspaceTask(
        goal="Write a guide", start_url="https://example.org/releases",
        note_path=tmp_path / "note.md", planner=Planner(),
        research_topics=["Using the Python Interpreter"], editor_probe=lambda: "",
    )
    task._research_evidence = list(evidence)
    task._snapshot = complete
    candidates = task.candidates(Observation("Write a guide", "Write", complete.to_dict()), [])
    assert len(candidates) == 1
    assert '"An observed exact quotation"' in candidates[0].arguments["text"]
    assert "https://example.org/releases" in candidates[0].arguments["text"]


def test_quote_grounding_preserves_literal_curly_apostrophe() -> None:
    source = "Read the interpreter’s output.\nThen continue."
    assert _ground_quote("the interpreter's output. Then", source) == (
        "the interpreter’s output.\nThen"
    )
    assert _ground_quote("an invented sentence", source) is None
