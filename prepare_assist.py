#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import csv
import os
import sys
import random
import multiprocessing as mp
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any


COMMON_USER_COLS = ['Anon Student Id', 'anon_id', 'user_id', 'student_id', 'student', 'user']
COMMON_ITEM_COLS = ['Problem Name', 'Problem ID', 'problem_id', 'problem', 'item_id', 'item']
COMMON_SKILL_COLS = ['skill', 'Skill', 'KC', 'skill_id', 'Knowledge Component']
COMMON_CORRECT_COLS = ['Correct', 'correct', 'CorrectFirstAttempt', 'correct_first', 'is_correct', 'Outcome']
COMMON_TIME_COLS = ['Timestamp', 'timestamp', 'Time', 'time']


def detect_column(fieldnames, candidates):
    low = [f.lower() for f in fieldnames]
    for cand in candidates:
        if cand in fieldnames:
            return cand
        if cand.lower() in low:
            return fieldnames[low.index(cand.lower())]
    return None


def coerce_bool_to_int(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in {'1', '1.0', 'true', 't', 'yes', 'y', 'correct'}:
        return 1
    return 0


def read_rows(path, delimiter=None, encoding='utf-8'):
    try_encodings = [encoding, 'latin1']
    for enc in try_encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                sample = f.read(4096)
                f.seek(0)
                if delimiter is None:
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        delim = dialect.delimiter
                    except:
                        delim = ','
                else:
                    delim = delimiter
                reader = csv.DictReader(f, delimiter=delim)
                return reader.fieldnames, list(reader)
        except:
            continue
    raise RuntimeError("读取失败")


def build_sequences(rows, user_col, item_col, skill_col, correct_col, time_col):
    users = defaultdict(list)
    for r in rows:
        u = r[user_col]
        it = r[item_col]
        sk = r.get(skill_col, '') if skill_col else ''
        corr = coerce_bool_to_int(r.get(correct_col, 0))
        ts = r.get(time_col, None) if time_col else None
        users[u].append((it, sk, corr, ts))

    for u in users:
        if any(x[3] for x in users[u]):
            try:
                users[u] = sorted(users[u], key=lambda x: float(x[3]) if x[3] else 0)
            except:
                pass
    return users


# ===== 仅用于 q2skills（严格：只用训练集 + 可选 prefix）=====
def build_q2skills(users, train_users, use_prefix=False):
    q2skills = defaultdict(set)
    for u in train_users:
        seq = users[u]
        if use_prefix:
            seq = seq[:-1]  # 去掉最后一步，避免未来信息
        for item, skill, *_ in seq:
            if not item or not skill:
                continue
            for s in skill.replace('|', ';').split(';'):
                s = s.strip()
                if s:
                    q2skills[item].add(s)
    return q2skills


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True)
    parser.add_argument('--out-dir', default='../Data')
    parser.add_argument('--dataset', default='ASSIST09')
    parser.add_argument('--delimiter', default=None)
    parser.add_argument('--encoding', default='utf-8')

    parser.add_argument('--user-col')
    parser.add_argument('--item-col')
    parser.add_argument('--skill-col')
    parser.add_argument('--correct-col')
    parser.add_argument('--time-col')

    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--min-seq', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--hg-max-members', type=int, default=0)
    parser.add_argument('--hg-sample-mode', choices=['topk', 'random'], default='topk')

    parser.add_argument('--use-prefix', action='store_true', help='是否使用前缀构图（严格无未来信息）')

    args = parser.parse_args()

    fieldnames, rows = read_rows(args.input, args.delimiter, args.encoding)

    user_col = args.user_col or detect_column(fieldnames, COMMON_USER_COLS)
    item_col = args.item_col or detect_column(fieldnames, COMMON_ITEM_COLS)
    skill_col = args.skill_col or detect_column(fieldnames, COMMON_SKILL_COLS)
    correct_col = args.correct_col or detect_column(fieldnames, COMMON_CORRECT_COLS)
    time_col = args.time_col or detect_column(fieldnames, COMMON_TIME_COLS)

    users = build_sequences(rows, user_col, item_col, skill_col, correct_col, time_col)

    # ===== split =====
    user_list = list(users.keys())
    random.seed(args.seed)
    random.shuffle(user_list)

    ntest = int(len(user_list) * args.test_ratio)
    test_users = set(user_list[:ntest])
    train_users = [u for u in user_list if u not in test_users]
    test_users = list(test_users)

    # ===== qmap（只用训练集）=====
    train_items = set()
    for u in train_users:
        for item, *_ in users[u]:
            train_items.add(item)

    qmap = {q: i for i, q in enumerate(sorted(train_items))}
    UNK = len(qmap)

    # ===== skill 只用训练 =====
    q2skills = build_q2skills(users, train_users, args.use_prefix)

    all_skills = set()
    for vs in q2skills.values():
        all_skills.update(vs)
    smap = {s: i for i, s in enumerate(sorted(all_skills))}

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ===== 写序列（未见题 → UNK）=====
    def write(path, user_list):
        with open(path, 'w') as f:
            for u in user_list:
                seq = users[u]
                if len(seq) < args.min_seq:
                    continue
                qids = [str(qmap[it[0]] if it[0] in qmap else UNK) for it in seq]
                ans = [str(it[2]) for it in seq]
                f.write(str(len(qids)) + '\n')
                f.write(','.join(qids) + '\n')
                f.write(','.join(ans) + '\n')

    write(os.path.join(out_dir, 'train_ques.txt'), train_users)
    write(os.path.join(out_dir, 'test_ques.txt'), test_users)

    # ===== KG（仅训练）=====
    with open(os.path.join(out_dir, 'kg_pk.edgelist'), 'w') as f:
        for q, skills in q2skills.items():
            if q not in qmap:
                continue
            for sk in skills:
                f.write(f"{qmap[q]},{len(qmap)+1+smap[sk]}\n")

    # ===== 超图（仅训练）=====
    user_map = {u: i for i, u in enumerate(train_users)}
    hg_pos, hg_neg = defaultdict(dict), defaultdict(dict)

    for u in train_users:
        sidx = user_map[u]
        seq = users[u][:-1] if args.use_prefix else users[u]
        for item, _, corr, _ in seq:
            if item not in qmap:
                continue
            qid = qmap[item]
            if corr:
                hg_pos[qid][sidx] = hg_pos[qid].get(sidx, 0) + 1
            else:
                hg_neg[qid][sidx] = hg_neg[qid].get(sidx, 0) + 1

    def sample(d):
        if args.hg_max_members <= 0:
            return sorted(d.keys())
        if args.hg_sample_mode == 'topk':
            return [k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:args.hg_max_members]]
        else:
            ks = list(d.keys())
            return random.sample(ks, min(len(ks), args.hg_max_members))

    hg_pos_out = {str(k): sample(v) for k, v in hg_pos.items()}
    hg_neg_out = {str(k): sample(v) for k, v in hg_neg.items()}

    json.dump(hg_pos_out, open(os.path.join(out_dir, 'hg_pos.json'), 'w'))
    json.dump(hg_neg_out, open(os.path.join(out_dir, 'hg_neg.json'), 'w'))
    json.dump(user_map, open(os.path.join(out_dir, 'user_map.json'), 'w'))

    json.dump({
        'num_questions': len(qmap) + 1,  # 包含 UNK
        'num_skills': len(smap),
        'num_students': len(train_users),
        'unk_qid': UNK
    }, open(os.path.join(out_dir, 'meta.json'), 'w'))

    print("✅ 论文级无泄露版本完成")


if __name__ == "__main__":
    main()