#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AllData_student_step_2011F.csv into model-ready files"
    )
    p.add_argument("--input", default="AllData_student_step_2011F.csv")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--dataset", default="all_data_student_step_2011f_trainable")
    p.add_argument("--encoding", default="utf-8-sig")

    p.add_argument(
        "--skill-source",
        choices=["unique", "single", "f2011"],
        default="f2011",
        help="Which KC column to use as skill list.",
    )
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--min-seq", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use-prefix", action="store_true")
    p.add_argument("--hg-max-members", type=int, default=0)
    p.add_argument("--hg-sample-mode", choices=["topk", "random"], default="topk")
    return p.parse_args()


def pick_skill_col(skill_source: str) -> str:
    if skill_source == "unique":
        return "KC (Unique-step)"
    if skill_source == "single":
        return "KC (Single-KC)"
    return "KC (F2011)"


def parse_timestamp(ts: str) -> Optional[float]:
    s = (ts or "").strip()
    if not s:
        return None
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return None


def parse_int(v: str) -> int:
    s = (v or "").strip()
    if not s or s == ".":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def split_skills(skill_text: str) -> List[str]:
    s = (skill_text or "").strip()
    if not s or s == ".":
        return []
    parts = [x.strip() for x in s.split("~~")]
    return [x for x in parts if x and x != "."]


def derive_correct(row: Dict[str, str]) -> int:
    first_attempt = (row.get("First Attempt") or "").strip().lower()
    if first_attempt == "correct":
        return 1
    if first_attempt in {"incorrect", "hint"}:
        return 0

    corrects = parse_int(row.get("Corrects", "0"))
    incorrects = parse_int(row.get("Incorrects", "0"))
    hints = parse_int(row.get("Hints", "0"))

    if corrects > 0 and incorrects == 0 and hints == 0:
        return 1
    return 0


def make_question(problem_name: str, step_name: str) -> str:
    p = (problem_name or "").strip()
    s = (step_name or "").strip()
    if not p or not s:
        return ""
    return f"{p}::{s}"


def load_sequences(
    path: str,
    encoding: str,
    skill_col: str,
) -> Dict[str, List[Tuple[str, List[str], int, Optional[float]]]]:
    users: Dict[str, List[Tuple[str, List[str], int, Optional[float]]]] = defaultdict(list)

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        required = ["Anon Student Id", "Problem Name", "Step Name", "First Transaction Time", skill_col]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            user = (row.get("Anon Student Id") or "").strip()
            item = make_question(row.get("Problem Name", ""), row.get("Step Name", ""))
            if not user or not item:
                continue

            skill_list = split_skills(row.get(skill_col, ""))
            correct = derive_correct(row)
            ts = parse_timestamp(row.get("First Transaction Time", ""))
            users[user].append((item, skill_list, correct, ts))

    for u, seq in users.items():
        if any(x[3] is not None for x in seq):
            users[u] = sorted(seq, key=lambda x: x[3] if x[3] is not None else float("inf"))

    return users


def build_q2skills(
    users: Dict[str, List[Tuple[str, List[str], int, Optional[float]]]],
    train_users: List[str],
    use_prefix: bool,
) -> Dict[str, Set[str]]:
    q2skills: Dict[str, Set[str]] = defaultdict(set)
    for u in train_users:
        seq = users[u][:-1] if use_prefix else users[u]
        for item, skills, _, _ in seq:
            for sk in skills:
                q2skills[item].add(sk)
    return q2skills


def write_sequences(
    path: str,
    users: Dict[str, List[Tuple[str, List[str], int, Optional[float]]]],
    user_list: List[str],
    qmap: Dict[str, int],
    min_seq: int,
    drop_unseen: bool,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for u in user_list:
            seq = users[u]
            if drop_unseen:
                seq = [x for x in seq if x[0] in qmap]

            if len(seq) < min_seq:
                continue
            qids = [str(qmap[item]) for item, _, _, _ in seq]
            ans = [str(corr) for _, _, corr, _ in seq]
            f.write(str(len(qids)) + "\n")
            f.write(",".join(qids) + "\n")
            f.write(",".join(ans) + "\n")


def sample_members(d: Dict[int, int], max_members: int, mode: str) -> List[int]:
    if max_members <= 0 or len(d) <= max_members:
        return sorted(d.keys())
    if mode == "topk":
        return [k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:max_members]]
    keys = list(d.keys())
    return random.sample(keys, max_members)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    skill_col = pick_skill_col(args.skill_source)
    users = load_sequences(args.input, args.encoding, skill_col)

    user_list = list(users.keys())
    random.shuffle(user_list)
    ntest = int(len(user_list) * args.test_ratio)
    test_users = set(user_list[:ntest])
    train_users = [u for u in user_list if u not in test_users]
    test_users = list(test_users)

    train_items = set()
    for u in train_users:
        for item, _, _, _ in users[u]:
            train_items.add(item)

    qmap = {q: i for i, q in enumerate(sorted(train_items))}
    unk = len(qmap)

    q2skills = build_q2skills(users, train_users, args.use_prefix)

    all_skills = set()
    for skill_set in q2skills.values():
        all_skills.update(skill_set)
    smap = {s: i for i, s in enumerate(sorted(all_skills))}

    out_path = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_path, exist_ok=True)

    write_sequences(
        os.path.join(out_path, "train_ques.txt"),
        users,
        train_users,
        qmap,
        args.min_seq,
        drop_unseen=False,
    )
    write_sequences(
        os.path.join(out_path, "test_ques.txt"),
        users,
        test_users,
        qmap,
        args.min_seq,
        drop_unseen=True,
    )

    with open(os.path.join(out_path, "kg_pk.edgelist"), "w", encoding="utf-8") as f:
        for q, skills in q2skills.items():
            if q not in qmap:
                continue
            for sk in skills:
                f.write(f"{qmap[q]},{len(qmap) + 1 + smap[sk]}\n")

    user_map = {u: i for i, u in enumerate(train_users)}
    hg_pos: Dict[int, Dict[int, int]] = defaultdict(dict)
    hg_neg: Dict[int, Dict[int, int]] = defaultdict(dict)

    for u in train_users:
        sidx = user_map[u]
        seq = users[u][:-1] if args.use_prefix else users[u]
        for item, _, corr, _ in seq:
            if item not in qmap:
                continue
            qid = qmap[item]
            target = hg_pos if corr else hg_neg
            target[qid][sidx] = target[qid].get(sidx, 0) + 1

    hg_pos_out = {str(q): sample_members(m, args.hg_max_members, args.hg_sample_mode) for q, m in hg_pos.items()}
    hg_neg_out = {str(q): sample_members(m, args.hg_max_members, args.hg_sample_mode) for q, m in hg_neg.items()}

    with open(os.path.join(out_path, "hg_pos.json"), "w", encoding="utf-8") as f:
        json.dump(hg_pos_out, f)
    with open(os.path.join(out_path, "hg_neg.json"), "w", encoding="utf-8") as f:
        json.dump(hg_neg_out, f)
    with open(os.path.join(out_path, "user_map.json"), "w", encoding="utf-8") as f:
        json.dump(user_map, f)
    with open(os.path.join(out_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(qmap) + 1,
                "num_skills": len(smap),
                "num_students": len(train_users),
                "unk_qid": unk,
            },
            f,
        )

    print("done")
    print(f"output_dir={out_path}")
    print(f"students={len(users)} train={len(train_users)} test={len(test_users)}")
    print(f"questions(train)={len(qmap)} skills={len(smap)}")


if __name__ == "__main__":
    main()
