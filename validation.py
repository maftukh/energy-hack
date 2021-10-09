import numpy as np
import pandas as pd


def plant_q_rule(series, threshold=1e-1):
    return (series >= threshold).astype(int)


def private_q_rule(series, threshold=0.6):
    name = series.name
    if name in ['Q_1', 'Q_7']:
        threshold *= 2
    if name in ['Q_6']:
        threshold *= 4
    return (series >= threshold).astype(int)


def get_q_by_p(pressure):
    if pressure.name in ['P_1', 'P_8']:
        return -0.3 + pressure * 7 / 1e6

    elif pressure.name in ['P_2', 'P_3', 'P_4', 'P_6']:
        return -0.2 + pressure * 4 / 1e6


def validate_plant(preds):
    res = plant_q_rule(preds)
    return res


def validate_private(preds):
    res = private_q_rule(preds)
    return res


def check_distribution(preds):
    p_cols = [i for i in preds.columns if i.startswith('P')]
    p_mapping = {
        i: 400000 for i in p_cols
    }
    q_mapping = {
        'Q_1': 2,
        'Q_2': 1,
        'Q_3': 1,
        'Q_4': 1,
        'Q_5': 1,
        'Q_6': 5,
        'Q_7': 2,
        'Q_8': 2,
    }
    mapping = {**p_mapping, **q_mapping}
    for col in preds.columns:
        if col in mapping:
            bad_index = preds[preds[col] > mapping[col]].index
            if bad_index.shape[0] > 0:
                print(f'WARNING in rows {bad_index.values}: {col} is out of upper bound of {mapping[col]}')


def check_pressure_order(idx, preds):
    determined_pairs = [
        ('P_9', 'P_8'),
        ('P_7', 'P_8'),
        ('P_7', 'P_6'),
        ('P_7', 'P_5'),
        ('P_4', 'P_3'),
        ('P_6', 'P_3'),
        ('P_6', 'P_2'),
    ]

    most_probable_pairs = [
        ('P_9', 'P_1'),
        ('P_7', 'P_4'),
    ]

    for pair in determined_pairs:
        if preds[pair[0]] < preds[pair[1]]:
            print(f"WARNING in row {idx}: There's no such case in the dataset: "
                  f"{pair[1]} is greater than {pair[0]}. Please check your predictions")

    for pair in most_probable_pairs:
        if preds[pair[0]] < preds[pair[1]]:
            print(f"WARNING in row {idx}: There are very few such case in the dataset: "
                  f"{pair[1]} is greater than {pair[0]}. Please check your predictions")
    return


def check_p_q_relationship(preds):
    mapping = {
        'P_1': 'Q_1',
        'P_4': 'Q_4',
        'P_6': 'Q_5',
        'P_8': 'Q_7',
        'P_2': 'Q_2',
        'P_3': 'Q_3',
    }

    for i in range(1, 10):
        col = f'P_{i}'
        # If not checked, continue
        if col not in mapping:
            continue

        # Preprocess column
        series = preds[col]
        if i == 2:
            series /= 2
            series += preds['P_1'] / 2

        # Check the difference between predicted and original
        diff_allowed = 0.05
        if i in [2, 3]:
            diff_allowed = 0.07
        elif i in [1, 8]:
            diff_allowed = 0.1

        q_pred = get_q_by_p(series)
        diff = (preds[mapping[col]] - q_pred).abs()
        bad_index = diff[diff > diff_allowed].index
        if bad_index.shape[0] > 0:
            print(f'WARNING in rows {bad_index.values}: {col} does not follow {mapping[col]} relationship')


def check_validity(preds):
    # Check "Q exceeds min level"
    for i in range(1, 8):
        preds[f'validPrivate_{i}'] = validate_private(preds[f'Q_{i}'])

    for i in range(1, 5):
        preds[f'validPlant_{i}'] = validate_plant(preds[f'QPlant_{i}'])

    # Check "P_9 > 200 000 for Plant 4"
    preds['validP9'] = (preds['P_9'] > 200000).astype(int)

    # Check order of Pressures
    for idx, row in preds.iterrows():
        check_pressure_order(idx, row)

    # Check that some specific Q and P follow their linear relationship
    check_p_q_relationship(preds)

    # Check upper bounds of the Q-s
    check_distribution(preds)

    cols = [i for i in preds.columns if i.startswith('valid')]
    validity_score = preds[cols].mean(axis=1)
    validity_binary = preds[cols].product(axis=1)

    return validity_binary, validity_score
