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
