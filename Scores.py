import numpy as np
import os
from scipy.ndimage import uniform_filter


# All EF3
times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
         '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
         '2025-05-19-23UTC']

models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
parameters = [
    'cape',
    'cin',
    'positive_helicity',
    'negative_helicity',
    'lcl',
    'k_idx',
    'lfc',
    'stp']


def compute_event_metrics(truth, pred, thresh, neigh_size=3):
    obs_evt = (abs(truth) >= abs(thresh)).astype(int)
    pred_evt = (abs(pred) >= abs(thresh)).astype(int)

    # Hits / Misses / False Alarms
    hits = (obs_evt == 1) & (pred_evt == 1)
    misses = (obs_evt == 1) & (pred_evt == 0)
    false_alarms = (obs_evt == 0) & (pred_evt == 1)

    # POD (Probability of Detection)
    if hits.sum() + misses.sum() > 0:
        POD = hits.sum() / (hits.sum() + misses.sum())
    else:
        POD = np.nan

    # CSI (Critical Success Index)
    if hits.sum() + misses.sum() + false_alarms.sum() > 0:
        CSI = hits.sum() / (hits.sum() + misses.sum() + false_alarms.sum())
    else:
        CSI = np.nan

    # SR (Success Ratio)
    if hits.sum() + false_alarms.sum() > 0:
        SR = hits.sum() / (hits.sum() + false_alarms.sum())
    else:
        SR = np.nan

    # FSS (Fractional Skill Score)
    f_obs = uniform_filter(
        obs_evt.astype(float),
        size=neigh_size,
        mode='nearest')
    f_pred = uniform_filter(
        pred_evt.astype(float),
        size=neigh_size,
        mode='nearest')
    denom = np.sum(f_pred ** 2 + f_obs ** 2)
    if denom > 0:
        FSS = 1 - np.sum((f_pred - f_obs) ** 2) / denom
    else:
        FSS = np.nan

    return POD, CSI, FSS, SR


def get_tornado_thresholds(time, parameter, model):
    if model != 'gfs':
        real = np.load(os.path.join(time, f'{parameter}_real.npy'))
    else:
        real = np.load(os.path.join(time, 'gfs', f"{parameter}_gfs.npy"))[0]
    if parameter == 'negative_helicity' or parameter == 'cin':
        threash = np.nanpercentile(real, 25)
    else:
        threash = np.nanpercentile(real, 75)
    return threash


POD_list, CSI_list, FSS_list, SR_list = [], [], [], []

for time in times:
    for model in models:
        for parameter in parameters:
            output_path = os.path.join(time, model, 'scores')
            os.makedirs(output_path, exist_ok=True)
            preds = np.load(
                os.path.join(
                    time,
                    model,
                    f"{parameter}_{model}.npy"))
            if model != 'gfs':
                truth = np.load(os.path.join(time, f"{parameter}_real.npy"))
            else:
                truth = np.load(
                    os.path.join(
                        time,
                        'gfs',
                        f"{parameter}_gfs.npy"))[0]
            scores = []
            threash = get_tornado_thresholds(time, parameter, model)
            print(f"{model}_{parameter} = {threash}")
            for pred in preds:
                POD, CSI, FSS, SR = compute_event_metrics(truth, pred, threash)
                POD_list.append(POD)
                CSI_list.append(CSI)
                FSS_list.append(FSS)
                SR_list.append(SR)
                scores.append([POD, CSI, FSS, SR])

            # Replace NaN with 0 in scores
            scores = np.nan_to_num(scores, nan=0)

            np.save(
                os.path.join(
                    output_path,
                    f'scores_{parameter}_{model}.npy'),
                scores)
