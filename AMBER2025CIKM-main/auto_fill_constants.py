#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-detect dataset stats for statics2011/xes3g5m and update Constants.py.

Detected values:
- numbers[dataset]: question count from Dataset/<dataset>/<dataset>_pid_train.csv
- skill[dataset]: concept/skill count from raw Data files
- datasets[dataset]: ensure key exists
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, Iterable, Set


def iter_pid_question_ids(pid_train_path: Path) -> Iterable[int]:
    with pid_train_path.open("r", encoding="utf-8-sig") as f:
        while True:
            len_line = f.readline()
            if not len_line:
                break
            q_line = f.readline()
            _skill_line = f.readline()
            _ans_line = f.readline()
            if not q_line:
                break
            for t in q_line.strip().strip(",").split(","):
                if not t:
                    continue
                qid = int(t)
                if qid > 0:
                    yield qid


def detect_question_count(dataset_root: Path, dataset: str) -> int:
    pid = dataset_root / dataset / f"{dataset}_pid_train.csv"
    if not pid.exists():
        raise FileNotFoundError(f"Missing pid file: {pid}")
    max_q = 0
    for q in iter_pid_question_ids(pid):
        if q > max_q:
            max_q = q
    if max_q <= 0:
        raise ValueError(f"No valid question ids found in {pid}")
    return max_q


def detect_statics_skill_count(data_root: Path) -> int:
    csv_path = data_root / "Statics2011" / "AllData_student_step_2011F.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing raw data: {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    col = "KC (F2011)" if "KC (F2011)" in df.columns else None
    if col is None:
        # fallback to common alternatives
        for c in ["KC (Unique-step)", "KC (Single-KC)", "skill", "Skill"]:
            if c in df.columns:
                col = c
                break
    if col is None:
        raise ValueError("No skill column found for Statics2011")

    skills: Set[str] = set()
    for v in df[col].fillna("").astype(str).tolist():
        s = v.strip()
        if not s or s == ".":
            continue
        parts = s.split("~~") if "~~" in s else re.split(r"[;,|]", s)
        for p in parts:
            t = p.strip()
            if t and t != ".":
                skills.add(t)
    return len(skills)


def detect_xes_skill_count(data_root: Path) -> int:
    import csv

    skills: Set[str] = set()
    for name in ["train.csv", "test.csv"]:
        path = data_root / "XES3G5M" / name
        if not path.exists():
            raise FileNotFoundError(f"Missing raw data: {path}")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "concepts" not in (reader.fieldnames or []):
                raise ValueError(f"Column 'concepts' not found in {path}")
            for row in reader:
                for t in str(row.get("concepts", "")).split(","):
                    c = t.strip()
                    if c and c not in {"0", "-1"}:
                        skills.add(c)
    return len(skills)


def replace_dict_block(text: str, var_name: str, new_dict: Dict[str, int | str]) -> str:
    pattern = re.compile(rf"(?ms)^{var_name}\s*=\s*\{{.*?^\}}")
    formatted_items = []
    for k, v in new_dict.items():
        if isinstance(v, str):
            formatted_items.append(f"    '{k}' : '{v}',")
        else:
            formatted_items.append(f"    '{k}' : {v},")
    new_block = f"{var_name} = {{\n" + "\n".join(formatted_items) + "\n}"
    if not pattern.search(text):
        raise ValueError(f"Cannot find dict block for {var_name} in Constants.py")
    return pattern.sub(new_block, text, count=1)


def parse_dict(text: str, var_name: str) -> Dict:
    m = re.search(rf"(?ms)^{var_name}\s*=\s*(\{{.*?\}})", text)
    if not m:
        raise ValueError(f"Cannot parse dict: {var_name}")
    return ast.literal_eval(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--constants", default="KnowledgeTracing/Constant/Constants.py")
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    constants_path = (root / args.constants).resolve()
    dataset_root = root / "Dataset"
    data_root = root / "Data"

    question_counts = {
        "statics2011": detect_question_count(dataset_root, "statics2011"),
        "xes3g5m": detect_question_count(dataset_root, "xes3g5m"),
    }
    skill_counts = {
        "statics2011": detect_statics_skill_count(data_root),
        "xes3g5m": detect_xes_skill_count(data_root),
    }

    text = constants_path.read_text(encoding="utf-8")
    datasets_dict = parse_dict(text, "datasets")
    numbers_dict = parse_dict(text, "numbers")
    skill_dict = parse_dict(text, "skill")

    datasets_dict["statics2011"] = "statics2011"
    datasets_dict["xes3g5m"] = "xes3g5m"
    numbers_dict.update(question_counts)
    skill_dict.update(skill_counts)

    # Keep existing key order, append new keys at end if absent.
    def ordered_merge(old: Dict, updates: Dict):
        out = dict(old)
        for k, v in updates.items():
            out[k] = v
        return out

    datasets_dict = ordered_merge(parse_dict(text, "datasets"), {"statics2011": "statics2011", "xes3g5m": "xes3g5m"})
    numbers_dict = ordered_merge(parse_dict(text, "numbers"), question_counts)
    skill_dict = ordered_merge(parse_dict(text, "skill"), skill_counts)

    text = replace_dict_block(text, "datasets", datasets_dict)
    text = replace_dict_block(text, "numbers", numbers_dict)
    text = replace_dict_block(text, "skill", skill_dict)

    constants_path.write_text(text, encoding="utf-8")

    print("Updated Constants.py")
    print(f"  statics2011: questions={question_counts['statics2011']}, skills={skill_counts['statics2011']}")
    print(f"  xes3g5m:     questions={question_counts['xes3g5m']}, skills={skill_counts['xes3g5m']}")


if __name__ == "__main__":
    main()
