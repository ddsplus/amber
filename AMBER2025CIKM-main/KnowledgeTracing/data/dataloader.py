import sys
import os
import random

sys.path.append('../')
import torch.utils.data as Data
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.preprocess import DataReader
from KnowledgeTracing.data.OneHot import OneHot,OneHotM


def _split_by_user_8_2(source_path, train_out, test_out, seed=216, train_ratio=0.8):
    """Split dataset by user records (4 lines per user) into train/test files."""
    if os.path.exists(train_out) and os.path.exists(test_out):
        return

    with open(source_path, 'r', encoding='UTF-8-sig') as f:
        lines = f.readlines()

    if len(lines) % 4 != 0:
        raise ValueError(f"Invalid dataset format: {source_path}, expected 4 lines per user.")

    users = [lines[i:i + 4] for i in range(0, len(lines), 4)]
    rng = random.Random(seed)
    rng.shuffle(users)

    split_idx = int(len(users) * train_ratio)
    split_idx = max(1, min(split_idx, len(users) - 1))
    train_users = users[:split_idx]
    test_users = users[split_idx:]

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    with open(train_out, 'w', encoding='UTF-8') as f_train:
        for user_block in train_users:
            f_train.writelines(user_block)

    with open(test_out, 'w', encoding='UTF-8') as f_test:
        for user_block in test_users:
            f_test.writelines(user_block)


def _resolve_dataset_paths(dataset):
    base = os.path.join(C.Dpath, dataset)
    original_train = os.path.join(base, f'{dataset}_pid_train.csv')

    split_train = os.path.join(base, f'{dataset}_pid_user8_2_train.csv')
    split_test = os.path.join(base, f'{dataset}_pid_user8_2_test.csv')

    _split_by_user_8_2(original_train, split_train, split_test, seed=216, train_ratio=0.8)
    return split_train, split_test


def getTrainLoader(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP)
    trainques, trainans = handle.getTrainData()
    dtrain = OneHot(trainques, trainans)
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True)
    return trainLoader


def getTestLoader(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP)
    testques, testans = handle.getTestData()
    dtest = OneHot(testques, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False, drop_last=True)
    return testLoader


def getLoader(dataset):
    if dataset not in {'assist2009', 'assist2017', 'assistednet'}:
        raise ValueError(f'Unsupported dataset: {dataset}')

    train_path, test_path = _resolve_dataset_paths(dataset)
    trainLoader = getTrainLoader(train_path)
    testLoader = getTestLoader(test_path)
    return trainLoader, testLoader


def getTrainDataset(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP)
    trainques, trainans = handle.getTrainData()
    dtrain = OneHotM(trainques, trainans)
    return dtrain


def getTestDataset(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP)
    testques, testans = handle.getTestData()
    dtest = OneHotM(testques, testans)
    return dtest


def getHeldDataset(held_data_path):
    handle = DataReader(held_data_path, C.MAX_STEP)
    testques, testans = handle.getTrainData()
    dtest = OneHotM(testques, testans)
    return dtest
