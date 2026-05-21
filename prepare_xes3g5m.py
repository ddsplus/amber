#!/usr/bin/env python3
"""
prepare_xes3g5m.py

从 XES3G5M CSV 文件生成模型可用的数据格式
输出文件：
  - train_ques.txt / test_ques.txt
  - train_skill.txt / test_skill.txt  
  - kg_pk.edgelist
  - user_map.json, hg_pos.json, hg_neg.json, meta.json (超图)

用法: python prepare_xes3g5m.py --train-csv train.csv --test-csv test.csv --out-dir ./output
"""

import argparse, csv, os, json, random
from collections import defaultdict
from typing import Dict, List, Tuple


def read_xes3g5m_csv(csv_path: str) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
    users = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['uid']
            questions = [int(q) for q in row['questions'].split(',') if q.strip()]
            concepts = [int(c) for c in row['concepts'].split(',') if c.strip()]
            responses = [int(r) for r in row['responses'].split(',') if r.strip()]
            if len(questions) == len(concepts) == len(responses):
                users[uid] = (questions, concepts, responses)
    return users


def write_ques_skill_files(out_ques: str, out_skill: str, users: Dict, qid_map: Dict, cid_map: Dict):
    """写入文件时必须使用映射后的连续ID（0-based），否则CUDA索引会越界
    过滤规则：
    - 题目ID <= 0：过滤（无效数据）
    - 响应 < 0：过滤（未作答）
    - 知识点ID <= 0：过滤（无效数据）
    """
    with open(out_ques, 'w', encoding='utf-8') as fq, open(out_skill, 'w', encoding='utf-8') as fs:
        for uid, (questions, concepts, responses) in sorted(users.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            # 过滤无效数据后映射ID
            valid_indices = [i for i, (q, r, c) in enumerate(zip(questions, responses, concepts))
                            if q > 0 and r >= 0 and c > 0 and q in qid_map and c in cid_map]
            
            if len(valid_indices) == 0:
                continue
            
            mapped_q = [str(qid_map[questions[i]]) for i in valid_indices]
            mapped_c = [str(cid_map[concepts[i]]) for i in valid_indices]
            mapped_r = [str(responses[i]) for i in valid_indices]
            
            fq.write(f"{len(mapped_q)}\n{','.join(mapped_q)}\n{','.join(mapped_r)}\n")
            fs.write(f"{len(mapped_c)}\n{','.join(mapped_c)}\n\n")


def build_q_to_c_map(train_users: Dict, test_users: Dict = None) -> Tuple[Dict[int, set], Dict[int, int], Dict[int, int]]:
    """构建题目->知识点映射关系（q2c）仅基于训练数据，但映射ID包括训练+测试中的所有有效ID。

    说明：为了避免测试序列在语义上被截断，我们把测试集中出现但训练集中未见的题目/概念也加入映射（qid_map / cid_map），
    但 KG（q2c）与超图仍然仅使用训练数据构建，不会泄露测试标签或关系。
    """
    q2c = defaultdict(set)
    train_q, train_c = set(), set()

    # 从训练数据收集关系与有效ID
    for uid, (questions, concepts, responses) in train_users.items():
        for q, c, r in zip(questions, concepts, responses):
            # 过滤掉无效数据：问题ID为-1或0，或响应为-1
            if q <= 0 or r < 0:
                continue
            if c <= 0:
                continue
            q2c[q].add(c)
            train_q.add(q)
            train_c.add(c)

    # 从测试数据收集ID以扩展映射，但不构建关系
    test_q, test_c = set(), set()
    if test_users:
        for uid, (questions, concepts, responses) in test_users.items():
            for q, c in zip(questions, concepts):
                if q <= 0 or c <= 0:
                    continue
                test_q.add(q)
                test_c.add(c)

    all_q = train_q.union(test_q)
    all_c = train_c.union(test_c)

    print(f"  [过滤后] 训练集有效问题数: {len(train_q)}, 有效知识点数: {len(train_c)}; 映射总计: {len(all_q)} 问题, {len(all_c)} 知识点")

    qid_map = {q: i for i, q in enumerate(sorted(all_q))}
    cid_map = {c: i for i, c in enumerate(sorted(all_c))}
    return dict(q2c), qid_map, cid_map


def write_kg_edgelist(out_path: str, q2c: Dict[int, set], qid_map: Dict, cid_map: Dict):
    num_questions = len(qid_map)
    with open(out_path, 'w', encoding='utf-8') as f:
        for q in sorted(q2c.keys()):
            if q not in qid_map:
                continue
            qid = qid_map[q]
            for c in sorted(q2c[q]):
                if c not in cid_map:
                    continue
                cid = cid_map[c]
                f.write(f"{qid},{num_questions + cid}\n")


def build_hypergraph(train_users: Dict, qid_map: Dict, cid_map: Dict) -> Tuple[Dict, Dict, Dict, int]:
    """构建超图，使用过滤后的有效数据"""
    user_list = sorted(train_users.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    user_map = {uid: idx for idx, uid in enumerate(user_list)}
    
    hg_pos_counts = defaultdict(lambda: defaultdict(int))
    hg_neg_counts = defaultdict(lambda: defaultdict(int))
    
    for uid, (questions, concepts, responses) in train_users.items():
        sidx = user_map[uid]
        for q, resp, c in zip(questions, responses, concepts):
            # 应用相同的过滤规则
            if q <= 0 or resp < 0 or c <= 0:
                continue
            if q not in qid_map or c not in cid_map:
                continue
            qid = qid_map[q]
            if resp == 1:
                hg_pos_counts[qid][sidx] += 1
            else:
                hg_neg_counts[qid][sidx] += 1
    
    hg_pos, hg_neg = {}, {}
    for qid in range(len(qid_map)):
        hg_pos[str(qid)] = sorted(list(hg_pos_counts[qid].keys()))
        hg_neg[str(qid)] = sorted(list(hg_neg_counts[qid].keys()))
    
    return user_map, hg_pos, hg_neg, len(qid_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    print("="*60)
    print("从 XES3G5M 生成模型数据\n")
    
    train_users = read_xes3g5m_csv(args.train_csv)
    test_users = read_xes3g5m_csv(args.test_csv)
    print(f"[1/4] 读取数据: {len(train_users)} 训练用户, {len(test_users)} 测试用户")
    
    q2c, qid_map, cid_map = build_q_to_c_map(train_users, test_users)
    print(f"[2/4] 分配 ID: {len(qid_map)} 问题, {len(cid_map)} 知识点")
    
    print(f"[3/4] 写入 ques/skill 文件（使用映射后的连续ID）")
    write_ques_skill_files(os.path.join(args.out_dir, 'train_ques.txt'), 
                          os.path.join(args.out_dir, 'train_skill.txt'), train_users, qid_map, cid_map)
    write_ques_skill_files(os.path.join(args.out_dir, 'test_ques.txt'), 
                          os.path.join(args.out_dir, 'test_skill.txt'), test_users, qid_map, cid_map)
    
    write_kg_edgelist(os.path.join(args.out_dir, 'kg_pk.edgelist'), q2c, qid_map, cid_map)
    print(f"  KG: {sum(len(v) for v in q2c.values())} 条边")
    
    print(f"[4/4] 生成超图数据")
    user_map, hg_pos, hg_neg, num_q = build_hypergraph(train_users, qid_map, cid_map)
    
    with open(os.path.join(args.out_dir, 'user_map.json'), 'w') as f:
        json.dump({'user_map': {str(k): v for k, v in user_map.items()}, 'num_students': len(user_map)}, f, indent=2)
    
    with open(os.path.join(args.out_dir, 'hg_pos.json'), 'w') as f:
        json.dump(hg_pos, f, indent=2)
    
    with open(os.path.join(args.out_dir, 'hg_neg.json'), 'w') as f:
        json.dump(hg_neg, f, indent=2)
    
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({
            'num_questions': num_q,
            'num_concepts': len(cid_map),
            'num_students': len(user_map),
            'q_node_offset': 0,
            'student_node_offset': num_q
        }, f, indent=2)
    
    print(f"  超图: {len(user_map)} 学生, {num_q} 问题")
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()

