"""Build the optional read-only OCR worker using existing Windows system libraries."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new build folder inside this project")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise ValueError("the optional native OCR worker requires Windows")
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError("a new build folder inside the project is required")
    windows = Path(os.environ.get("WINDIR", "C:/Windows"))
    framework = windows / "Microsoft.NET/Framework64/v4.0.30319"
    compiler = framework / "csc.exe"
    if not compiler.is_file():
        raise ValueError("existing .NET Framework compiler is unavailable; no installation attempted")
    gac = windows / "Microsoft.NET/assembly/GAC_MSIL"
    names = ("System.Runtime", "System.Runtime.InteropServices.WindowsRuntime",
             "System.Runtime.WindowsRuntime", "System.ObjectModel", "System.Web.Extensions")
    references = []
    for name in names:
        matches = sorted((gac / name).glob("*/" + name + ".dll"))
        if len(matches) != 1:
            raise ValueError("expected native OCR system assembly is unavailable or ambiguous")
        references.append(matches[0])
    references += [framework / "System.dll", framework / "System.Core.dll"]
    references += [windows / "System32/WinMetadata" / ("Windows." + name + ".winmd")
                   for name in ("Foundation", "Globalization", "Graphics", "Media", "Storage")]
    if not all(p.is_file() for p in references):
        raise ValueError("required installed Windows metadata is unavailable")
    source = ROOT / "scripts/native_menu_ocr.cs"
    destination.mkdir(parents=True)
    binary = destination / "NativeMenuOcr.exe"
    command = [str(compiler), "/nologo", "/noconfig", "/utf8output", "/target:exe", "/out:" + str(binary)]
    command += ["/reference:" + str(p) for p in references] + [str(source)]
    result = subprocess.run(command, capture_output=True, timeout=30)
    log = destination / "compiler.log"
    log.write_bytes(result.stdout + result.stderr)
    record = {"kind": "native_menu_ocr_build", "exit_code": result.returncode,
              "source_sha256": sha(source), "compiler_sha256": sha(compiler),
              "references": {str(p): sha(p) for p in references}, "compiler_log_sha256": sha(log),
              "executed": False, "environment_inputs": 0, "ocr_recognition_calls": 0,
              "host_execution_policy_changed": False, "dependencies_installed": False}
    if result.returncode == 0:
        record["binary_sha256"] = sha(binary)
    (destination / "build.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exit_code": result.returncode, "binary": str(binary),
                      "binary_sha256": record.get("binary_sha256")}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
