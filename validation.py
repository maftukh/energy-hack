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
#
#
# def get_q7_q1(pressure):
#     return -0.3 + pressure * 7 / 1e6
#
#
# def get_q4_q5(pressure):
#     return -0.2 + pressure * 4 / 1e6


def validate_plant(preds):
    res = plant_q_rule(preds)
    return res


def validate_private(preds):
    res = private_q_rule(preds)
    return res


def check_validity(preds):
    # Check "Q exceeds min level"
    for i in range(1, 8):
        preds[f'validPrivate_{i}'] = validate_private(preds[f'Q_{i}'])

    for i in range(1, 5):
        preds[f'validPlant_{i}'] = validate_plant(preds[f'QPlant_{i}'])

    # Check "P_9 > 200 000 for Plant 4"
    preds['validP9'] = (preds['P_9'] > 200000).astype(int)

    cols = [i for i in preds.columns if i.startswith('valid')]
    validity_score = preds[cols].mean(axis=1)
    validity_binary = preds[cols].product(axis=1)

    return validity_binary, validity_score
