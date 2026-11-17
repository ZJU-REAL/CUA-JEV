"""Optional local native OCR worker; no UI input, network or dependency install."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import subprocess
import time
from pathlib import Path

from .desktop_menu_grounding import ImageWord, ImageWords


class MenuOcrUnavailable(RuntimeError):
    pass


class NativeMenuOcr:
    """Recognize bounded enlarged tiles and preserve source/worker receipts.

    The executable is explicitly configured and pinned. All observations and
    outputs stay in a new per-episode artifact directory; nothing is installed.
    """

    def __init__(self, binary: Path, binary_sha256: str, artifact_dir: Path, *, timeout_s: float = 6.0,
                 batch_mode: bool = False):
        self.binary = Path(binary).resolve(strict=True)
        if (not self.binary.is_file() or not isinstance(binary_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", binary_sha256)
                or hashlib.sha256(self.binary.read_bytes()).hexdigest() != binary_sha256):
            raise ValueError("native OCR binary does not match its declared SHA256")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
            raise ValueError("native OCR timeout must be positive and at most30 seconds")
        if type(batch_mode) is not bool:
            raise ValueError("native OCR batch mode must be boolean")
        self.binary_sha256 = binary_sha256
        self.artifact_dir = Path(artifact_dir).absolute()
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.timeout_s = timeout_s
        self.batch_mode = batch_mode
        self._cache: dict[str, ImageWords] = {}
        self._failed: set[str] = set()
        self.metrics = {"requests": 0, "cache_hits": 0, "frames": 0, "native_recognitions": 0,
                        "unknown_recognition_attempts": 0, "worker_processes": 0,
                        "failures": 0, "wall_ms": 0.0}

    def recognize(self, raw: bytes, *, deadline: float | None = None) -> ImageWords:
        from PIL import Image

        self.metrics["requests"] += 1
        started = time.perf_counter()
        if (not isinstance(raw, bytes) or not raw or len(raw) > 24_000_000
                or (deadline is not None and (not math.isfinite(deadline) or deadline <= started))):
            raise MenuOcrUnavailable("native OCR source or remaining input budget is unavailable")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self._failed:
            raise MenuOcrUnavailable("native OCR previously failed on this exact image")
        if digest in self._cache:
            self.metrics["cache_hits"] += 1
            return self._cache[digest]
        if self.metrics["frames"] >= 64:
            raise MenuOcrUnavailable("native OCR episode image budget exhausted")
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != self.binary_sha256:
            raise MenuOcrUnavailable("native OCR worker changed")
        with Image.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            if min(width, height) < 40 or width * height > 16_000_000:
                raise MenuOcrUnavailable("native OCR image dimensions exceed budget")
            image = opened.convert("RGB")
        columns, rows = math.ceil(width / 1024), math.ceil(height / 768)
        if columns * rows > 16:
            raise MenuOcrUnavailable("native OCR tile budget exceeded")
        folder = self.artifact_dir / digest
        folder.mkdir(exist_ok=False)
        self.metrics["frames"] += 1
        (folder / "source-image.bin").write_bytes(raw)
        record = {"kind": "local_native_menu_ocr", "completed": False,
                  "source_image_sha256": digest, "source_size": [width, height],
                  "process_layout": "frame_batch_v1" if self.batch_mode else "tile_process_v1",
                  "worker_sha256": self.binary_sha256, "tiles": [], "environment_inputs": 0}
        until = min(started + self.timeout_s, deadline) if deadline is not None else started + self.timeout_s
        lines = []
        count = 0
        prepared = []
        process_count_before = self.metrics["worker_processes"]
        calls_before = self.metrics["native_recognitions"]
        unknown_before = self.metrics["unknown_recognition_attempts"]
        try:
            for row in range(rows):
                for column in range(columns):
                    remaining = until - time.perf_counter()
                    if remaining <= 0:
                        raise MenuOcrUnavailable("native OCR frame budget expired")
                    right = width * (column + 1) // columns + (32 if column + 1 < columns else 0)
                    region = (max(0, width * column // columns - (32 if column else 0)),
                              max(0, height * row // rows - (32 if row else 0)),
                              min(width, right),
                              min(height, height * (row + 1) // rows + (32 if row + 1 < rows else 0)))
                    cropped = image.crop(region)
                    resized = (cropped.width * 2, cropped.height * 2)
                    cropped = cropped.resize(resized, Image.Resampling.LANCZOS)
                    stem = f"tile-{row}-{column}"
                    source, receipt = folder / (stem + ".png"), folder / (stem + ".json")
                    cropped.save(source)
                    tile = {"region": list(region), "scale": 2,
                            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                    record["tiles"].append(tile)
                    prepared.append((source, receipt, resized, region, stem, tile))
            if self.batch_mode:
                manifest = folder / "batch-input.json"
                manifest.write_text(json.dumps({"kind": "native_menu_ocr_batch_v1", "tiles": [
                    {"input": str(source), "output": str(receipt),
                     "source_image_sha256": tile["input_sha256"]}
                    for source, receipt, _size, _region, _stem, tile in prepared
                ]}), encoding="utf-8")
                record["batch_input_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
                batch_receipt = folder / "batch-result.json"
                process = self._launch(folder, "batch", "--recognize-saved-batch", manifest,
                                       batch_receipt, until, unknown_calls=len(prepared))
                batch = self._read_json(batch_receipt)
                if (process.returncode != 0 or batch.get("completed") is not True
                        or batch.get("kind") != "windows_native_ocr_saved_batch_v1"
                        or batch.get("source_manifest_sha256") != record["batch_input_sha256"]
                        or batch.get("helper_binary_sha256") != self.binary_sha256
                        or type(batch.get("ocr_recognition_calls")) is not int
                        or batch["ocr_recognition_calls"] != len(prepared)
                        or not isinstance(batch.get("tiles"), list)
                        or len(batch["tiles"]) != len(prepared)):
                    raise MenuOcrUnavailable("native OCR batch receipt does not match the frame")
                for proof, (_source, receipt, _size, _region, _stem, tile) in zip(
                    batch["tiles"], prepared, strict=True,
                ):
                    raw_receipt = self._receipt_bytes(receipt)
                    if (not isinstance(proof, dict)
                            or proof.get("source_image_sha256") != tile["input_sha256"]
                            or proof.get("receipt_sha256") != hashlib.sha256(raw_receipt).hexdigest()
                            or proof.get("output") != str(receipt)):
                        raise MenuOcrUnavailable("native OCR batch tile proof does not match")
                    tile["returncode"] = process.returncode
                record["batch_receipt_sha256"] = hashlib.sha256(batch_receipt.read_bytes()).hexdigest()
            for source, receipt, resized, region, stem, tile in prepared:
                if not self.batch_mode:
                    # Recheck after image encoding; preprocessing consumes the same budget.
                    process = self._launch(folder, stem, "--recognize-saved", source, receipt, until)
                    tile["returncode"] = process.returncode
                data = self._read_json(receipt)
                calls = data.get("ocr_recognition_calls")
                if type(calls) is not int or calls not in (0, 1):
                    raise MenuOcrUnavailable("native OCR receipt has invalid recognition count")
                self.metrics["native_recognitions"] += calls
                if (process.returncode != 0 or data.get("completed") is not True
                        or calls != 1
                        or data.get("source_image_sha256") != tile["input_sha256"]
                        or data.get("helper_binary_sha256") != self.binary_sha256
                        or type(data.get("width")) is not int or type(data.get("height")) is not int
                        or (data["width"], data["height"]) != resized
                        or not isinstance(data.get("lines"), list) or len(data["lines"]) > 2000):
                    raise MenuOcrUnavailable("native OCR receipt does not match the observation")
                tile["receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
                for line in data["lines"]:
                    if not isinstance(line, dict) or not isinstance(line.get("words"), list):
                        raise MenuOcrUnavailable("native OCR words must be bounded records")
                    if len(line["words"]) > 256:
                        raise MenuOcrUnavailable("native OCR line word budget exceeded")
                    words = []
                    for word in line["words"]:
                        count += 1
                        if count > 2000 or not isinstance(word, dict):
                            raise MenuOcrUnavailable("native OCR frame word budget exceeded")
                        text = word.get("text")
                        values = [word.get(key) for key in ("x", "y", "width", "height")]
                        if (not isinstance(text, str) or not text.strip() or len(text) > 240
                                or "\x00" in text
                                or any(type(v) not in (int, float) or not math.isfinite(v)
                                       for v in values)):
                            raise MenuOcrUnavailable("native OCR word record is invalid")
                        x, y, w, h = values
                        if not (0 <= x < x + w <= resized[0] and 0 <= y < y + h <= resized[1]):
                            raise MenuOcrUnavailable("native OCR word bounds are invalid")
                        words.append(ImageWord(text, (x / 2 + region[0], y / 2 + region[1],
                                                      (x + w) / 2 + region[0], (y + h) / 2 + region[1])))
                    if words:
                        lines.append(tuple(words))
            if time.perf_counter() >= until:
                raise MenuOcrUnavailable("native OCR frame budget expired during receipt validation")
            result = ImageWords(digest, (width, height), tuple(lines))
            self._cache[digest] = result
            record["completed"] = True
            return result
        except Exception as error:
            if self.batch_mode and self.metrics["worker_processes"] > process_count_before:
                unknown = len(prepared) - (self.metrics["native_recognitions"] - calls_before)
                recorded = self.metrics["unknown_recognition_attempts"] - unknown_before
                self.metrics["unknown_recognition_attempts"] += max(0, unknown - recorded)
                record["unknown_tile_call_count_is_upper_bound"] = True
            self._failed.add(digest)
            self.metrics["failures"] += 1
            record["error_type"] = type(error).__name__
            raise MenuOcrUnavailable("native OCR did not produce complete matching image evidence") from None
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.metrics["wall_ms"] += elapsed
            record["wall_ms"] = elapsed
            (folder / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def _launch(self, folder, stem, mode, source, receipt, until, *, unknown_calls=1):
        remaining = until - time.perf_counter()
        if remaining <= 0:
            raise MenuOcrUnavailable("native OCR frame budget expired")
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != self.binary_sha256:
            raise MenuOcrUnavailable("native OCR worker changed before launch")
        self.metrics["worker_processes"] += 1
        try:
            process = subprocess.run(
                [str(self.binary), mode, str(folder), str(source), str(receipt)],
                capture_output=True, timeout=remaining,
            )
        except subprocess.TimeoutExpired as error:
            # An upper bound of unknown tile calls, never reported as zero completed work.
            self.metrics["unknown_recognition_attempts"] += unknown_calls
            (folder / (stem + "-stdout.txt")).write_bytes(error.stdout or b"")
            (folder / (stem + "-stderr.txt")).write_bytes(error.stderr or b"")
            raise MenuOcrUnavailable("native OCR worker exceeded frame budget") from None
        (folder / (stem + "-stdout.txt")).write_bytes(process.stdout)
        (folder / (stem + "-stderr.txt")).write_bytes(process.stderr)
        return process

    def _receipt_bytes(self, receipt):
        if not receipt.exists() or receipt.stat().st_size > 2_000_000:
            self.metrics["unknown_recognition_attempts"] += 1
            raise MenuOcrUnavailable("native OCR receipt is unavailable or too large")
        raw = receipt.read_bytes()
        if len(raw) > 2_000_000:
            raise MenuOcrUnavailable("native OCR receipt grew beyond budget")
        return raw

    def _read_json(self, receipt):
        data = json.loads(self._receipt_bytes(receipt))
        if not isinstance(data, dict):
            raise MenuOcrUnavailable("native OCR receipt must be a record")
        return data
