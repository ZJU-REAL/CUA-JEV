import pytest

from cua_jev.cli import _parser, main


def test_mac_cli_defaults_to_chromium_and_has_native_gates(monkeypatch):
    monkeypatch.setattr("cua_jev.cli.sys.platform", "darwin")
    args = _parser().parse_args([
        "open-computer", "--goal", "test", "--url", "https://example.test",
        "--model-base-url", "https://model.test/v1", "--model", "fixture",
        "--window-title", "Fixture", "--app-bundle-id", "test.fixture",
        "--require-window-text-contains", "Saved", "--record-desktop", "runs/native.mp4",
    ])
    assert args.browser_channel == "chromium"
    assert args.app_bundle_id == "test.fixture"
    assert args.require_window_text_contains == "Saved"


def test_mac_unsupported_vision_is_rejected_before_any_network_call(monkeypatch):
    monkeypatch.setattr("cua_jev.cli.sys.platform", "darwin")
    with pytest.raises(SystemExit, match="desktop VLM is not yet enabled"):
        main([
            "open-desktop", "--goal", "test", "--window-title", "Fixture",
            "--model-base-url", "https://unused.test/v1", "--model", "test", "--vision-model", "vision",
        ])


def test_mac_standalone_desktop_cannot_rely_on_model_success_claim(monkeypatch):
    monkeypatch.setattr("cua_jev.cli.sys.platform", "darwin")
    with pytest.raises(SystemExit, match="caller-supplied"):
        main([
            "open-desktop", "--goal", "test", "--window-title", "Fixture",
            "--model-base-url", "https://unused.test/v1", "--model", "test",
        ])
