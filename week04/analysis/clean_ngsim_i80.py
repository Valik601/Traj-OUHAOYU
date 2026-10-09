#!/usr/bin/env python3
"""Full-file audit and conservative cleaning for the NGSIM I-80 SI CSV.

The source file is read-only. Missing values are reported, not imputed. An
optional derived CSV removes only exact duplicate (vehicle_id, frame_id) rows;
conflicting duplicate keys stop that export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


MAX_BYTES = 512 * 1024 * 1024
COLS = [
    "vehicle_id", "frame_id", "time_s", "global_time_ms", "local_x_m",
    "local_y_m", "vehicle_length_m", "vehicle_class", "speed_m_s",
    "acceleration_m_s2", "lane_id",
]
INT_COLS = {"vehicle_id", "frame_id", "global_time_ms", "vehicle_class", "lane_id"}
FIELD_INFO = {
    "vehicle_id": ("车辆匿名编号", "标识符", "整数"),
    "frame_id": ("采样帧编号", "帧", "整数"),
    "time_s": ("相对采样起点时间", "s", "有限数值"),
    "global_time_ms": ("全局时间戳", "ms", "整数"),
    "local_x_m": ("横向位置", "m", "有限数值"),
    "local_y_m": ("纵向位置", "m", "有限数值"),
    "vehicle_length_m": ("车辆长度", "m", "有限数值"),
    "vehicle_class": ("车辆类别代码", "类别代码", "整数"),
    "speed_m_s": ("速度", "m/s", "有限数值"),
    "acceleration_m_s2": ("加速度", "m/s²", "有限数值"),
    "lane_id": ("车道代码", "类别代码", "整数"),
}


class AuditError(RuntimeError):
    pass


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit missing and duplicate NGSIM records")
    p.add_argument("--input", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--metadata")
    p.add_argument("--json-report", required=True)
    p.add_argument("--md-report", required=True)
    p.add_argument("--missing-token", action="append", default=[])
    p.add_argument("--cleaned-output")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def local_path(root: Path, value: str, suffix: str, *, output: bool, force: bool) -> Path:
    raw = Path(value)
    if "~" in value or ".." in raw.parts:
        raise AuditError("Paths may not contain '~' or '..'")
    path = (raw if raw.is_absolute() else root / raw).resolve(strict=not output)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AuditError("Path must remain inside --root") from exc
    if path.suffix.lower() != suffix:
        raise AuditError(f"Expected {suffix}: {path.name}")
    if output:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (path.is_symlink() or not path.is_file() or not force):
            raise AuditError(f"Refusing existing output {path.name}; use --force")
    else:
        if path.is_symlink() or not path.is_file():
            raise AuditError("Input must be a regular non-symlink file")
        stat = path.stat()
        if getattr(stat, "st_nlink", 1) > 1 or stat.st_size > MAX_BYTES:
            raise AuditError("Input violates link or 512 MiB size limit")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_row(row: list[str]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for cell in row:
        encoded = cell.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    return h.digest()


def parse_value(column: str, text: str) -> int | float:
    if column in INT_COLS:
        value = int(text)
        if text.lstrip("+") != str(value):
            raise ValueError("non-canonical integer")
        return value
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite")
    return value


def audit_csv(path: Path, missing_tokens: set[str]) -> dict[str, Any]:
    missing = Counter({c: 0 for c in COLS})
    invalid = Counter({c: 0 for c in COLS})
    mins: dict[str, int | float | None] = {c: None for c in COLS}
    maxs: dict[str, int | float | None] = {c: None for c in COLS}
    key_digest: dict[int, bytes] = {}
    vehicle_last_tick: dict[int, int] = {}
    vehicles: set[int] = set()
    lanes: Counter[int] = Counter()
    counts = Counter()
    previous_global: int | None = None
    time_origin: int | None = None
    frame_offset: int | None = None
    rows = 0

    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as f:
        reader = csv.reader(f, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise AuditError("CSV is empty") from exc
        if header != COLS:
            raise AuditError("CSV columns/order do not match the declared SI schema")

        for row in reader:
            rows += 1
            if len(row) != len(COLS):
                counts["nonrectangular_rows"] += 1
                continue
            parsed: dict[str, int | float] = {}
            for i, column in enumerate(COLS):
                text = row[i].strip()
                if text == "" or text in missing_tokens:
                    missing[column] += 1
                    continue
                try:
                    value = parse_value(column, text)
                except ValueError:
                    invalid[column] += 1
                    continue
                parsed[column] = value
                mins[column] = value if mins[column] is None else min(mins[column], value)
                maxs[column] = value if maxs[column] is None else max(maxs[column], value)

            vehicle = parsed.get("vehicle_id")
            frame = parsed.get("frame_id")
            if isinstance(vehicle, int):
                vehicles.add(vehicle)
            lane = parsed.get("lane_id")
            if isinstance(lane, int):
                lanes[lane] += 1

            if isinstance(vehicle, int) and isinstance(frame, int) and 0 <= vehicle < 2**32 and 0 <= frame < 2**32:
                key = (vehicle << 32) | frame
                current_digest = digest_row(row)
                old_digest = key_digest.get(key)
                if old_digest is None:
                    key_digest[key] = current_digest
                else:
                    counts["duplicate_key_extra_rows"] += 1
                    if old_digest == current_digest:
                        counts["exact_duplicate_extra_rows"] += 1
                    else:
                        counts["conflicting_key_extra_rows"] += 1
            else:
                counts["invalid_key_rows"] += 1

            global_time = parsed.get("global_time_ms")
            if isinstance(global_time, int):
                if previous_global is not None and global_time < previous_global:
                    counts["global_time_nonmonotonic_rows"] += 1
                previous_global = global_time
                if global_time % 100:
                    counts["off_100ms_grid_rows"] += 1

            time_s = parsed.get("time_s")
            if isinstance(time_s, float):
                tick = round(time_s * 10)
                if abs(time_s * 10 - tick) > 1e-6:
                    counts["off_100ms_grid_rows"] += 1
                if isinstance(global_time, int):
                    origin = global_time - round(time_s * 1000)
                    time_origin = origin if time_origin is None else time_origin
                    counts["time_origin_mismatch_rows"] += int(origin != time_origin)
                if isinstance(frame, int):
                    offset = frame - tick
                    frame_offset = offset if frame_offset is None else frame_offset
                    counts["frame_offset_mismatch_rows"] += int(offset != frame_offset)
                if isinstance(vehicle, int):
                    last = vehicle_last_tick.get(vehicle)
                    if last is not None:
                        gap = tick - last
                        if gap == 0:
                            counts["duplicate_vehicle_time_rows"] += 1
                        elif gap < 0:
                            counts["vehicle_time_nonmonotonic_rows"] += 1
                        elif gap > 1:
                            counts["trajectory_gap_events"] += 1
                            counts["internal_missing_ticks"] += gap - 1
                    if last is None or tick > last:
                        vehicle_last_tick[vehicle] = tick

    missing_total = sum(missing.values())
    invalid_total = sum(invalid.values())
    review_values = [
        missing_total, invalid_total, counts["nonrectangular_rows"],
        counts["duplicate_key_extra_rows"], counts["invalid_key_rows"],
        counts["trajectory_gap_events"], counts["off_100ms_grid_rows"],
    ]
    return {
        "status": "PASS" if not any(review_values) else "REVIEW",
        "schema": {
            "columns": COLS,
            "column_count": len(COLS),
            "observational_unit": "one vehicle observation at one 0.1-second frame",
            "primary_key": ["vehicle_id", "frame_id"],
            "fields": {
                c: {"meaning": FIELD_INFO[c][0], "unit": FIELD_INFO[c][1], "type": FIELD_INFO[c][2]}
                for c in COLS
            },
        },
        "scope": {
            "rows_scanned": rows,
            "full_file_scanned": True,
            "distinct_vehicles": len(vehicles),
            "lane_record_counts": {str(k): lanes[k] for k in sorted(lanes)},
        },
        "missingness": {
            "policy": "empty/whitespace plus explicit --missing-token values",
            "missing_cells": missing_total,
            "by_column": dict(missing),
            "imputation_performed": False,
        },
        "validity": {
            "invalid_numeric_cells": invalid_total,
            "by_column": dict(invalid),
            "nonrectangular_rows": counts["nonrectangular_rows"],
            "ranges": {c: {"min": mins[c], "max": maxs[c]} for c in COLS},
        },
        "duplicates": {
            "method": "exact primary key plus 128-bit row digest",
            "invalid_key_rows": counts["invalid_key_rows"],
            "duplicate_key_extra_rows": counts["duplicate_key_extra_rows"],
            "exact_duplicate_extra_rows": counts["exact_duplicate_extra_rows"],
            "conflicting_key_extra_rows": counts["conflicting_key_extra_rows"],
        },
        "temporal_continuity": {
            "trajectory_gap_events": counts["trajectory_gap_events"],
            "internal_missing_0_1s_ticks": counts["internal_missing_ticks"],
            "duplicate_vehicle_time_rows": counts["duplicate_vehicle_time_rows"],
            "vehicle_time_nonmonotonic_rows": counts["vehicle_time_nonmonotonic_rows"],
            "global_time_nonmonotonic_rows": counts["global_time_nonmonotonic_rows"],
            "off_100ms_grid_rows": counts["off_100ms_grid_rows"],
            "time_origin_ms": time_origin,
            "time_origin_mismatch_rows": counts["time_origin_mismatch_rows"],
            "frame_tick_offset": frame_offset,
            "frame_offset_mismatch_rows": counts["frame_offset_mismatch_rows"],
        },
    }


def read_metadata(path: Path) -> dict[str, Any]:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError("Metadata JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise AuditError(f"Metadata JSON contains {value}")

    with path.open("r", encoding="utf-8", errors="strict") as f:
        value = json.load(f, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise AuditError("Metadata JSON root must be an object")
    return value


def compare_metadata(
    metadata: dict[str, Any], input_path: Path, file_hash: str, audit: dict[str, Any]
) -> dict[str, bool]:
    files = metadata.get("files", {})
    entry = files.get(input_path.name, {}) if isinstance(files, dict) else {}
    declared_time = metadata.get("global_time_ms")
    measured_time = audit["validity"]["ranges"]["global_time_ms"]
    return {
        "row_count_matches": metadata.get("rows") == audit["scope"]["rows_scanned"],
        "vehicle_count_matches": metadata.get("vehicles") == audit["scope"]["distinct_vehicles"],
        "duplicate_key_count_matches": metadata.get("duplicate_vehicle_frame_keys")
        == audit["duplicates"]["duplicate_key_extra_rows"],
        "file_bytes_match": entry.get("bytes") == input_path.stat().st_size,
        "sha256_matches": entry.get("sha256") == file_hash,
        "global_time_min_matches": isinstance(declared_time, list)
        and len(declared_time) == 2
        and declared_time[0] == measured_time["min"],
        "global_time_max_matches": isinstance(declared_time, list)
        and len(declared_time) == 2
        and declared_time[1] == measured_time["max"],
    }


def atomic_text(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp",
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def write_cleaned(source: Path, target: Path, audit: dict[str, Any]) -> int:
    dup = audit["duplicates"]
    if dup["invalid_key_rows"] or dup["conflicting_key_extra_rows"]:
        raise AuditError("Cleaned export refused: invalid or conflicting primary keys")
    seen: set[int] = set()
    rows_written = 0
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=target.parent,
        prefix=f".{target.name}.", suffix=".tmp",
    )
    temp = Path(handle.name)
    try:
        with source.open("r", encoding="utf-8-sig", errors="strict", newline="") as f:
            reader = csv.reader(f, strict=True)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(next(reader))
            for row in reader:
                vehicle = int(row[0].strip())
                frame = int(row[1].strip())
                key = (vehicle << 32) | frame
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(row)
                rows_written += 1
        handle.close()
        os.replace(temp, target)
    except Exception:
        handle.close()
        temp.unlink(missing_ok=True)
        raise
    return rows_written


def make_markdown(report: dict[str, Any]) -> str:
    audit = report["audit"]
    miss = audit["missingness"]
    valid = audit["validity"]
    dup = audit["duplicates"]
    time = audit["temporal_continuity"]
    source = report["provenance"]
    lines = [
        "# NGSIM I-80 数据缺失与重复检查报告", "",
        f"- 检查状态：**{audit['status']}**",
        f"- 扫描范围：完整文件，共 {audit['scope']['rows_scanned']:,} 条记录",
        f"- 车辆数：{audit['scope']['distinct_vehicles']:,}",
        f"- 字段数：{audit['schema']['column_count']}",
        f"- 文件大小：{source['input_bytes']:,} 字节",
        f"- SHA-256：`{source['input_sha256']}`", "",
        "## 表格属性与单位", "",
        "每行表示一辆车在一个 0.1 秒采样帧上的观测，主键为 `vehicle_id + frame_id`。", "",
        "| 字段 | 含义 | 单位 | 类型 | 最小值 | 最大值 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for column in COLS:
        info = audit["schema"]["fields"][column]
        limits = valid["ranges"][column]
        lines.append(
            f"| `{column}` | {info['meaning']} | {info['unit']} | {info['type']} | "
            f"{limits['min']} | {limits['max']} |"
        )
    lines += [
        "", "## 缺失与字段有效性", "",
        f"- 缺失单元格：{miss['missing_cells']:,}。",
        f"- 类型错误或非有限数：{valid['invalid_numeric_cells']:,}。",
        f"- 非矩形记录：{valid['nonrectangular_rows']:,}。",
        "- 未执行自动插补；数值 0 保留为真实观测。", "",
        "| 字段 | 缺失数 | 类型/数值异常数 |", "|---|---:|---:|",
    ]
    for column in COLS:
        lines.append(f"| `{column}` | {miss['by_column'][column]:,} | {valid['by_column'][column]:,} |")
    lines += [
        "", "## 重复记录", "",
        f"- 主键重复额外记录：{dup['duplicate_key_extra_rows']:,}。",
        f"- 其中完整重复行：{dup['exact_duplicate_extra_rows']:,}。",
        f"- 主键相同但内容冲突：{dup['conflicting_key_extra_rows']:,}。",
        f"- 无效主键记录：{dup['invalid_key_rows']:,}。", "",
        "## 时间连续性补充检查", "",
        f"- 车辆内部时间缺口事件：{time['trajectory_gap_events']:,}。",
        f"- 缺口对应的 0.1 秒采样点：{time['internal_missing_0_1s_ticks']:,}。",
        f"- 同一车辆同一时间重复：{time['duplicate_vehicle_time_rows']:,}。",
        f"- 不在 0.1 秒网格上的记录：{time['off_100ms_grid_rows']:,}。",
        "", "轨迹时间缺口不同于空单元格。报告只标记，不自动插值。",
    ]
    metadata = report.get("metadata_comparison")
    if metadata:
        labels = {
            "row_count_matches": "记录数一致", "vehicle_count_matches": "车辆数一致",
            "duplicate_key_count_matches": "主键重复数一致", "file_bytes_match": "文件大小一致",
            "sha256_matches": "SHA-256 一致", "global_time_min_matches": "最小全局时间一致",
            "global_time_max_matches": "最大全局时间一致",
        }
        lines += ["", "## 与来源检查文件对照", ""]
        lines += [f"- {label}：{'是' if metadata[key] else '否'}。" for key, label in labels.items()]
        if not metadata["global_time_min_matches"]:
            lines.append("- 来源 JSON 的最小 `global_time_ms` 与 CSV 实测值不一致，建议修正元数据。")
    lines += [
        "", "## 清洗结论", "", report["cleaning"]["decision"], "",
        "原始 CSV 未被覆盖。只有显式传入 `--cleaned-output` 才会写派生文件；脚本只移除已验证的完整重复主键记录，不自动处理缺失值。",
        "", "## 方法与限制", "",
        "- 使用 Python 标准库流式扫描完整 CSV，未抽样。",
        "- 缺失定义为空字符串/纯空白以及明确传入的缺失标记。",
        "- 重复检查使用精确主键和 128 位行摘要；摘要仅用于比较同一主键下的记录。",
        "- 时间缺口可能来自采集或轨迹截断，需结合领域规则判断。", "",
        "## 方法来源", "",
        "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
        "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. "
        "arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065", "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = arguments()
    root = Path(args.root).resolve(strict=True)
    if not root.is_dir():
        raise AuditError("--root must be an existing directory")
    source = local_path(root, args.input, ".csv", output=False, force=False)
    json_report = local_path(root, args.json_report, ".json", output=True, force=args.force)
    md_report = local_path(root, args.md_report, ".md", output=True, force=args.force)
    metadata_path = (
        local_path(root, args.metadata, ".json", output=False, force=False)
        if args.metadata else None
    )
    cleaned_path = (
        local_path(root, args.cleaned_output, ".csv", output=True, force=args.force)
        if args.cleaned_output else None
    )

    file_hash = sha256(source)
    audit = audit_csv(source, {token.strip() for token in args.missing_token})
    report: dict[str, Any] = {
        "report_schema_version": "1.0",
        "provenance": {
            "input_basename": source.name,
            "input_bytes": source.stat().st_size,
            "input_sha256": file_hash,
            "raw_source_overwritten": False,
        },
        "audit": audit,
        "cleaning": {
            "decision": (
                "未发现需要删除或插补的缺失/重复问题，因此不改写数据记录。"
                if audit["status"] == "PASS"
                else "发现需要复核的问题；在确认处理规则前不自动删除或插补。"
            ),
            "cleaned_output_written": False,
        },
    }
    if metadata_path:
        report["metadata_comparison"] = compare_metadata(
            read_metadata(metadata_path), source, file_hash, audit
        )
    if cleaned_path:
        report["cleaning"]["cleaned_rows"] = write_cleaned(source, cleaned_path, audit)
        report["cleaning"]["cleaned_output_written"] = True
        report["cleaning"]["cleaned_output_basename"] = cleaned_path.name

    atomic_text(json_report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_text(md_report, make_markdown(report))
    print(json.dumps({
        "status": audit["status"],
        "rows_scanned": audit["scope"]["rows_scanned"],
        "missing_cells": audit["missingness"]["missing_cells"],
        "duplicate_key_extra_rows": audit["duplicates"]["duplicate_key_extra_rows"],
        "exact_duplicate_extra_rows": audit["duplicates"]["exact_duplicate_extra_rows"],
        "json_report": json_report.name,
        "md_report": md_report.name,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, csv.Error, UnicodeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
