import pytest

from cua_jev.artifact_surface import ArtifactSurface


def test_artifact_requires_source_gates_and_exact_citations(tmp_path):
    surface = ArtifactSurface(tmp_path / "guide.md")
    surface.reset()
    with pytest.raises(ValueError, match="source gates"):
        surface.validate(surface.capture(), "a0", "write", "premature")
    surface.set_acceptance(
        ready=True, citations=("https://docs.example.test/topic",),
    )
    with pytest.raises(ValueError, match="cite each visited"):
        surface.validate(surface.capture(), "a0", "write", "No citation")
    text = "# Guide\nhttps://docs.example.test/topic\n"
    candidate = surface.compile(surface.capture(), "a0", "write", text, "Write", 0)[0]
    receipt = surface.execute(candidate, "obs", "decision")
    assert receipt.success, receipt.error
    assert surface._verify(candidate, receipt).passed
    assert surface.capture().to_dict()["sha256"]
    assert "# Guide" not in str(surface.capture().to_dict())
    with pytest.raises(ValueError, match="refusing to overwrite"):
        surface.reset()


def test_scoped_artifact_can_open_in_verified_editor_after_creation(tmp_path, monkeypatch):
    editor = {"title": ""}
    calls = []
    surface = ArtifactSurface(
        tmp_path / "new-guide.md", allow_open_vscode=True,
        editor_probe=lambda: editor["title"],
    )
    surface.reset()
    surface.set_acceptance(ready=True, citations=())
    with pytest.raises(ValueError, match="editor open is not currently offered"):
        surface.validate(surface.capture(), "a0", "open", "")
    write = surface.compile(surface.capture(), "a0", "write", "# Guide\n", "Write", 0)[0]
    assert surface.execute(write, "obs", "decision").success

    monkeypatch.setattr("cua_jev.artifact_surface.shutil.which", lambda _name: "code")

    def launch(argv, *, shell):
        calls.append((argv, shell))
        editor["title"] = "new-guide.md - Visual Studio Code"

    monkeypatch.setattr("cua_jev.artifact_surface.subprocess.Popen", launch)
    candidate = surface.compile(surface.capture(), "a0", "open", "", "Open", 1)[0]
    assert candidate.capability == "artifact.open_vscode"
    receipt = surface.execute(candidate, "obs", "decision")
    assert receipt.success, receipt.error
    assert surface._verify_editor_open(candidate, receipt).passed
    assert calls == [(["code", "--new-window", str(surface.path)], False)]
    with pytest.raises(ValueError, match="editor open is not currently offered"):
        surface.validate(surface.capture(), "a0", "open", "")


def test_existing_same_name_editor_is_not_mistaken_for_new_artifact(tmp_path, monkeypatch):
    windows = {7: "guide.md - Visual Studio Code"}

    def launch():
        windows[8] = "guide.md - Visual Studio Code"

    surface = ArtifactSurface(
        tmp_path / "guide.md", allow_open_vscode=True, editor_launcher=launch,
    )
    monkeypatch.setattr(surface, "_editor_windows", lambda: windows.copy())
    surface.reset()
    surface.set_acceptance(ready=True, citations=())
    write = surface.compile(surface.capture(), "a0", "write", "# New guide\n", "Write", 0)[0]
    assert surface.execute(write, "obs", "write").success
    assert not surface.capture().editor_open
    open_action = surface.compile(surface.capture(), "a0", "open", "", "Open", 1)[0]
    receipt = surface.execute(open_action, "obs", "open")
    assert receipt.success, receipt.error
    assert surface.opened_window_handle == 8
    assert surface._verify_editor_open(open_action, receipt).passed


def test_notepad_handoff_is_a_distinct_guarded_editor_route(tmp_path, monkeypatch):
    visible = {"title": ""}
    launched = []
    surface = ArtifactSurface(
        tmp_path / "public-note.txt", allow_open_notepad=True,
        editor_probe=lambda: visible["title"],
    )
    surface.reset()
    surface.set_acceptance(ready=True, citations=())
    write = surface.compile(surface.capture(), "a0", "write", "Public note", "Write", 0)[0]
    assert surface.execute(write, "obs", "write").success
    monkeypatch.setattr("cua_jev.artifact_surface.shutil.which", lambda name: name)

    def launch(argv, *, shell):
        launched.append((argv, shell))
        visible["title"] = "public-note.txt - Notepad"

    monkeypatch.setattr("cua_jev.artifact_surface.subprocess.Popen", launch)
    action = surface.compile(surface.capture(), "a0", "open", "", "Open", 1)[0]
    assert action.capability == "artifact.open_notepad"
    receipt = surface.execute(action, "obs", "open")
    assert receipt.success, receipt.error
    assert surface._verify_editor_open(action, receipt).passed
    assert launched == [(["notepad.exe", str(surface.path)], False)]
    with pytest.raises(ValueError, match="choose exactly one"):
        ArtifactSurface(tmp_path / "other.txt", allow_open_vscode=True, allow_open_notepad=True)
