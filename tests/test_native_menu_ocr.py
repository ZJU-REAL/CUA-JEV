"""Worker receipts, coordinate mappings and deadlines, with a local subprocess fixture."""

import hashlib
import io
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cua_jev.native_menu_ocr import MenuOcrUnavailable, NativeMenuOcr


@pytest.fixture
def setup(tmp_path, monkeypatch):
    binary = tmp_path / "worker.exe"
    binary.write_bytes(b"non-executed worker fixture")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    output = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(output, "PNG")
    raw = output.getvalue()
    calls = []

    def child(command, *, capture_output, timeout):
        assert capture_output and 0 < timeout <= 6
        calls.append(command)
        assert command[1] == "--recognize-saved"
        source, receipt = Path(command[-2]), Path(command[-1])
        with Image.open(source) as image:
            width, height = image.size
        data = {"completed": True, "ocr_recognition_calls": 1, "helper_binary_sha256": digest,
                "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "width": width, "height": height, "lines": [{"words": [{
                    "text": "Example", "x": 10, "y": 20, "width": 100, "height": 24,
                }]}]}
        receipt.write_text(json.dumps(data), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", child)
    provider = NativeMenuOcr(binary, digest, tmp_path / "ocr")
    return provider, raw, calls, child


def test_tiled_receipts_map_back_to_original_pixels_and_exact_image_cache_avoids_processes(setup):
    provider, raw, calls, _ = setup
    words = provider.recognize(raw)
    assert words.image_sha256 == hashlib.sha256(raw).hexdigest() and words.size == (1920, 1080)
    assert [line[0].box for line in words.lines] == [
        (5, 10, 55, 22), (933, 10, 983, 22), (5, 518, 55, 530), (933, 518, 983, 530),
    ]
    assert len(calls) == 4 and provider.metrics["native_recognitions"] == 4
    assert provider.recognize(raw) is words and len(calls) == 4
    assert provider.metrics["cache_hits"] == 1
    provenance = json.loads(next(provider.artifact_dir.glob("*/provenance.json")).read_bytes())
    assert provenance["completed"] and len(provenance["tiles"]) == 4
    assert all(t["receipt_sha256"] for t in provenance["tiles"])


@pytest.mark.parametrize("change", ["image", "worker", "dimensions", "nonfinite", "bounds", "word_budget"])
def test_wrong_or_unbounded_worker_receipts_fail_without_partial_image_evidence(setup, monkeypatch, change):
    provider, raw, calls, child = setup

    def altered(command, **kwargs):
        result = child(command, **kwargs)
        path = Path(command[-1])
        data = json.loads(path.read_bytes())
        if change == "image":
            data["source_image_sha256"] = "0" * 64
        elif change == "worker":
            data["helper_binary_sha256"] = "0" * 64
        elif change == "dimensions":
            data["width"] += 1
        elif change == "nonfinite":
            data["lines"][0]["words"][0]["x"] = float("nan")
        elif change == "bounds":
            data["lines"][0]["words"][0]["width"] = 10000
        else:
            data["lines"] = [{"words": data["lines"][0]["words"] * 256}] * 8
        path.write_text(json.dumps(data), encoding="utf-8")
        return result

    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", altered)
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert len(calls) == 1 and provider.metrics["failures"] == 1
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert len(calls) == 1  # An identical failed image never launches an implicit retry.
    assert not provider._cache
    assert not json.loads(next(provider.artifact_dir.glob("*/provenance.json")).read_bytes())["completed"]


def test_expired_input_deadline_and_worker_hash_mismatch_never_launch_process(setup):
    provider, raw, calls, _ = setup
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw, deadline=time.perf_counter() - 1)
    assert calls == []
    provider.binary.write_bytes(b"changed")
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert calls == []


def test_timeout_retains_unknown_attempt_provenance_and_never_returns_a_partial_tile(setup, monkeypatch):
    provider, raw, _, _ = setup

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("local worker", 0.001, output=b"partial", stderr=b"timeout")

    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", timeout)
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert provider.metrics["unknown_recognition_attempts"] == 1 and not provider._cache
    folder = next(p for p in provider.artifact_dir.iterdir() if p.is_dir())
    assert (folder / "tile-0-0-stdout.txt").read_bytes() == b"partial"
    assert not json.loads((folder / "provenance.json").read_bytes())["completed"]


@pytest.mark.parametrize("digest", ["", "0" * 64, True])
def test_worker_must_be_explicitly_pinned_before_creating_artifacts(tmp_path, digest):
    binary = tmp_path / "worker.exe"
    binary.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="SHA256"):
        NativeMenuOcr(binary, digest, tmp_path / "ocr")
    assert not (tmp_path / "ocr").exists()


def batch_child(setup):
    provider, _raw, calls, child = setup

    def run(command, **kwargs):
        assert command[1] == "--recognize-saved-batch"
        manifest_path, receipt = Path(command[-2]), Path(command[-1])
        manifest = json.loads(manifest_path.read_bytes())
        assert manifest["kind"] == "native_menu_ocr_batch_v1"
        proofs = []
        for tile in manifest["tiles"]:
            child([command[0], "--recognize-saved", command[2], tile["input"], tile["output"]], **kwargs)
            output = Path(tile["output"])
            proofs.append({"source_image_sha256": tile["source_image_sha256"], "output": str(output),
                           "receipt_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        # The fixture produces four tile receipts but models a single process invocation.
        calls.clear()
        calls.append(command)
        receipt.write_text(json.dumps({
            "kind": "windows_native_ocr_saved_batch_v1", "completed": True,
            "helper_binary_sha256": provider.binary_sha256,
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "ocr_recognition_calls": len(proofs), "tiles": proofs,
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=b"batch", stderr=b"")

    return run


def test_batch_uses_one_process_and_preserves_identical_geometry_and_all_tile_proofs(setup, monkeypatch):
    provider, raw, calls, _ = setup
    provider.batch_mode = True
    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", batch_child(setup))
    result = provider.recognize(raw)
    assert [line[0].box for line in result.lines] == [
        (5, 10, 55, 22), (933, 10, 983, 22), (5, 518, 55, 530), (933, 518, 983, 530),
    ]
    assert len(calls) == 1 and provider.metrics["worker_processes"] == 1
    assert provider.metrics["native_recognitions"] == 4
    assert provider.recognize(raw) is result and len(calls) == 1
    proof = json.loads(next(provider.artifact_dir.glob("*/provenance.json")).read_bytes())
    assert proof["completed"] and proof["process_layout"] == "frame_batch_v1"
    assert proof["batch_input_sha256"] and proof["batch_receipt_sha256"]
    assert len(proof["tiles"]) == 4 and all(t["receipt_sha256"] for t in proof["tiles"])


@pytest.mark.parametrize("corruption", [
    "missing_tile", "reordered", "duplicate", "manifest", "worker", "calls", "calls_bool",
    "incomplete", "missing_receipt", "tile_hash", "tile_bounds", "total_word_budget",
])
def test_batch_rejects_partial_forged_and_unbounded_frame_evidence(setup, monkeypatch, corruption):
    provider, raw, calls, _ = setup
    provider.batch_mode = True
    run = batch_child(setup)

    def altered(command, **kwargs):
        result = run(command, **kwargs)
        receipt = Path(command[-1])
        data = json.loads(receipt.read_bytes())
        if corruption == "missing_tile":
            data["tiles"].pop()
        elif corruption == "reordered":
            data["tiles"].reverse()
        elif corruption == "duplicate":
            data["tiles"][1] = data["tiles"][0]
        elif corruption == "manifest":
            data["source_manifest_sha256"] = "0" * 64
        elif corruption == "worker":
            data["helper_binary_sha256"] = "0" * 64
        elif corruption == "calls":
            data["ocr_recognition_calls"] = 3
        elif corruption == "calls_bool":
            data["ocr_recognition_calls"] = True
        elif corruption == "incomplete":
            data["completed"] = False
        elif corruption == "missing_receipt":
            Path(data["tiles"][-1]["output"]).unlink()
        elif corruption == "tile_hash":
            data["tiles"][0]["receipt_sha256"] = "0" * 64
        else:
            for tile in data["tiles"]:
                path = Path(tile["output"])
                content = json.loads(path.read_bytes())
                if corruption == "tile_bounds":
                    content["lines"][0]["words"][0]["x"] = -1
                else:
                    content["lines"] = content["lines"] * 501  # Per-tile valid, complete frame exceeds2000.
                path.write_text(json.dumps(content), encoding="utf-8")
                tile["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt.write_text(json.dumps(data), encoding="utf-8")
        return result

    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", altered)
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert not provider._cache and provider.metrics["failures"] == 1
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert len(calls) == 1


def test_batch_timeout_preserves_partial_logs_with_unknown_tile_call_upper_bound(setup, monkeypatch):
    provider, raw, _, _ = setup
    provider.batch_mode = True

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 0.001, output=b"partial batch", stderr=b"timeout")

    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", timeout)
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw)
    assert provider.metrics["unknown_recognition_attempts"] == 4
    assert provider.metrics["worker_processes"] == 1 and not provider._cache
    folder = next(provider.artifact_dir.iterdir())
    assert (folder / "batch-stdout.txt").read_bytes() == b"partial batch"
    assert not json.loads((folder / "provenance.json").read_bytes())["completed"]


def test_receipt_validation_time_is_part_of_the_original_frame_deadline(setup, monkeypatch):
    provider, raw, _, child = setup
    now = time.perf_counter()

    def slow(command, **kwargs):
        result = child(command, **kwargs)
        monkeypatch.setattr("cua_jev.native_menu_ocr.time.perf_counter", lambda: now + 10)
        return result

    monkeypatch.setattr("cua_jev.native_menu_ocr.subprocess.run", slow)
    with pytest.raises(MenuOcrUnavailable):
        provider.recognize(raw, deadline=now + 5)
    assert not provider._cache
