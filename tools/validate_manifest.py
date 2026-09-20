#!/usr/bin/env python3
"""Validate MMS-LLaMA TSV/WRD pairs before expensive training."""

import argparse
import csv
import json
import wave
from pathlib import Path


def validate_split(manifest_dir: Path, split: str, probe_audio: int) -> dict:
    tsv_path = manifest_dir / f"{split}.tsv"
    wrd_path = manifest_dir / f"{split}.wrd"
    rows = list(csv.reader(tsv_path.read_text(encoding="utf-8").splitlines()[1:], delimiter="\t"))
    labels = wrd_path.read_text(encoding="utf-8").splitlines()
    report = {
        "split": split,
        "rows": len(rows),
        "labels": len(labels),
        "bad_columns": 0,
        "missing_video": 0,
        "missing_audio": 0,
        "empty_label": 0,
        "bad_numeric": 0,
        "bad_audio_probe": 0,
    }
    if len(rows) != len(labels):
        report["row_label_mismatch"] = True
    for index, row in enumerate(rows):
        if len(row) != 6:
            report["bad_columns"] += 1
            continue
        _, video, audio, frames, samples, rate = row
        if not Path(video).is_file():
            report["missing_video"] += 1
        if not Path(audio).is_file():
            report["missing_audio"] += 1
        if index >= len(labels) or not labels[index].strip():
            report["empty_label"] += 1
        try:
            int(frames)
            int(samples)
            float(rate)
        except ValueError:
            report["bad_numeric"] += 1
        if index < probe_audio and Path(audio).is_file():
            try:
                with wave.open(audio, "rb") as handle:
                    if handle.getframerate() != 16000 or handle.getnchannels() != 1:
                        report["bad_audio_probe"] += 1
            except (wave.Error, EOFError):
                report["bad_audio_probe"] += 1
    report["ok"] = not any(
        report.get(key, 0)
        for key in (
            "row_label_mismatch",
            "bad_columns",
            "missing_video",
            "missing_audio",
            "empty_label",
            "bad_numeric",
            "bad_audio_probe",
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--probe-audio", type=int, default=100)
    args = parser.parse_args()
    reports = [validate_split(args.manifest_dir, split, args.probe_audio) for split in args.splits]
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
