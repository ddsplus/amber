#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare raw KT datasets into AMBER format.

Outputs per dataset:
1) Dataset/<dataset>/<dataset>_pid_train.csv
   - 4 lines per user sequence:
     line1: sequence length
     line2: comma separated question ids (1-based, remapped)
     line3: placeholder skill line (kept for compatibility)
     line4: comma separated correctness labels (0/1)

2) Dataset/H/<h_tag>.csv
   - Hypergraph incidence matrix H with shape [num_questions, num_users]
   - H[q, u] = 1 if user u interacted with question q at least once
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Event = Tuple[str, int, Optional[float]]  # (item_key, correct, timestamp)


def parse_bool_to_int(v) -> int:
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true", "t", "yes", "y", "correct"}:
        return 1
    return 0


def safe_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def detect_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def sort_user_events(users: Dict[str, List[Event]]) -> None:
    for uid, seq in users.items():
        if any(ts is not None for _, _, ts in seq):
            users[uid] = sorted(seq, key=lambda x: x[2] if x[2] is not None else float("inf"))


def write_pid_train(out_csv: Path, users: Dict[str, List[Event]], min_seq: int = 3) -> Tuple[int, int]:
    all_items = set()
    kept_users: List[Tuple[str, List[Event]]] = []
    for uid, seq in users.items():
        if len(seq) < min_seq:
            continue
        kept_users.append((uid, seq))
        for item, _, _ in seq:
            all_items.add(item)

    qmap = {q: i + 1 for i, q in enumerate(sorted(all_items))}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        for _, seq in kept_users:
            qids = [str(qmap[item]) for item, _, _ in seq]
            ans = [str(corr) for _, corr, _ in seq]
            f.write(str(len(qids)) + "\n")
            f.write(",".join(qids) + "\n")
            f.write(",".join(["0"] * len(qids)) + "\n")
            f.write(",".join(ans) + "\n")

    return len(qmap), len(kept_users)


def write_h_matrix(h_csv: Path, users: Dict[str, List[Event]], min_seq: int = 3) -> None:
    kept = [(uid, seq) for uid, seq in users.items() if len(seq) >= min_seq]
    if not kept:
        raise ValueError("No users left after filtering by min_seq.")

    all_items = sorted({item for _, seq in kept for item, _, _ in seq})
    qmap = {q: i for i, q in enumerate(all_items)}

    H = np.zeros((len(all_items), len(kept)), dtype=np.int64)
    for uidx, (_, seq) in enumerate(kept):
        seen = set()
        for item, _, _ in seq:
            qidx = qmap[item]
            if qidx not in seen:
                H[qidx, uidx] = 1
                seen.add(qidx)

    h_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(H).to_csv(h_csv, index=False, header=False)


def _read_csv_with_fallback(path: Path, encodings: Sequence[str]) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(path)


def load_assist(path: Path) -> Dict[str, List[Event]]:
    # ASSIST2009 is often latin1; keep fallback for robustness.
    df = _read_csv_with_fallback(path, encodings=("utf-8-sig", "utf-8", "latin1"))
    cols = list(df.columns)

    user_col = detect_col(cols, ["user_id", "uid", "user", "anon_id", "Anon Student Id", "studentId"])
    item_col = detect_col(cols, ["problem_id", "problemId", "problem", "item_id", "skill_id", "Problem Name"])
    correct_col = detect_col(cols, ["correct", "Correct", "is_correct", "Outcome"])
    time_col = detect_col(cols, ["order_id", "timestamp", "start_time", "startTime", "Time"])

    if user_col is None or item_col is None or correct_col is None:
        raise ValueError(f"Cannot detect required columns in {path}. columns={cols}")

    users: Dict[str, List[Event]] = defaultdict(list)
    for _, row in df.iterrows():
        uid = str(row[user_col]).strip()
        item = str(row[item_col]).strip()
        if not uid or not item or uid.lower() == "nan" or item.lower() == "nan":
            continue
        corr = parse_bool_to_int(row[correct_col])
        ts = safe_float(row[time_col]) if time_col else None
        users[uid].append((item, corr, ts))

    sort_user_events(users)
    return users


def load_statics2011(path: Path) -> Dict[str, List[Event]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["Anon Student Id", "Problem Name", "Step Name"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")

    users: Dict[str, List[Event]] = defaultdict(list)
    for _, row in df.iterrows():
        uid = str(row.get("Anon Student Id", "")).strip()
        problem = str(row.get("Problem Name", "")).strip()
        step = str(row.get("Step Name", "")).strip()
        item = f"{problem}::{step}" if problem and step else ""
        if not uid or not item or uid.lower() == "nan":
            continue

        first_attempt = str(row.get("First Attempt", "")).strip().lower()
        if first_attempt == "correct":
            corr = 1
        elif first_attempt in {"incorrect", "hint"}:
            corr = 0
        else:
            corr = 1 if str(row.get("Corrects", "0")).strip() not in {"", "0", "0.0"} else 0

        ts = safe_float(row.get("First Transaction Time", None))
        users[uid].append((item, corr, ts))

    sort_user_events(users)
    return users


def load_xes(train_csv: Path, test_csv: Path) -> Dict[str, List[Event]]:
    users: Dict[str, List[Event]] = defaultdict(list)

    def ingest(csv_path: Path) -> None:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = str(row.get("uid", "")).strip()
                if not uid:
                    continue
                questions = [x.strip() for x in str(row.get("questions", "")).split(",") if x.strip()]
                responses = [x.strip() for x in str(row.get("responses", "")).split(",") if x.strip()]
                if len(questions) != len(responses):
                    continue
                for idx, (q, r) in enumerate(zip(questions, responses)):
                    if q in {"", "0", "-1"}:
                        continue
                    corr = parse_bool_to_int(r)
                    users[uid].append((q, corr, float(idx)))

    ingest(train_csv)
    ingest(test_csv)
    sort_user_events(users)
    return users


def process_one(dataset: str, users: Dict[str, List[Event]], dataset_dir: Path, h_dir: Path, h_tag: str, min_seq: int) -> None:
    out_train = dataset_dir / dataset / f"{dataset}_pid_train.csv"
    num_q, num_u = write_pid_train(out_train, users, min_seq=min_seq)
    write_h_matrix(h_dir / f"{h_tag}.csv", users, min_seq=min_seq)
    print(f"[{dataset}] done: questions={num_q}, users={num_u}, pid_train={out_train}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare raw datasets for AMBER")
    p.add_argument("--project-root", default=".", help="Project root path")
    p.add_argument("--data-root", default="Data", help="Raw data root under project")
    p.add_argument("--dataset-root", default="Dataset", help="Output Dataset root")
    p.add_argument("--seed", type=int, default=216)
    p.add_argument("--min-seq", type=int, default=3)
    p.add_argument("--only", choices=["assist2009", "assist2017", "statics2011", "xes3g5m", "all"], default="all")
    args = p.parse_args()

    random.seed(args.seed)

    project_root = Path(args.project_root).resolve()
    data_root = project_root / args.data_root
    dataset_root = project_root / args.dataset_root
    h_root = dataset_root / "H"

    tasks = []
    if args.only in {"assist2009", "all"}:
        tasks.append(("assist2009", data_root / "ASSIST2009" / "skill_builder_data.csv", "2009"))
    if args.only in {"assist2017", "all"}:
        tasks.append(("assist2017", data_root / "ASSIST2017" / "anonymized_full_release_competition_dataset.csv", "2017"))
    if args.only in {"statics2011", "all"}:
        tasks.append(("statics2011", data_root / "Statics2011" / "AllData_student_step_2011F.csv", "2011"))

    for name, path, h_tag in tasks:
        if not path.exists():
            print(f"[{name}] skip: file not found -> {path}")
            continue
        if name in {"assist2009", "assist2017"}:
            users = load_assist(path)
        else:
            users = load_statics2011(path)
        process_one(name, users, dataset_root, h_root, h_tag, args.min_seq)

    if args.only in {"xes3g5m", "all"}:
        tr = data_root / "XES3G5M" / "train.csv"
        te = data_root / "XES3G5M" / "test.csv"
        if tr.exists() and te.exists():
            users = load_xes(tr, te)
            process_one("xes3g5m", users, dataset_root, h_root, "xes3g5m", args.min_seq)
        else:
            print(f"[xes3g5m] skip: missing {tr} or {te}")


if __name__ == "__main__":
    main()
