import matplotlib.pyplot as plt
import os
import numpy as np


def score_plot(lead_times, scores):
    plt.figure(figsize=(10, 6))
    trans_score = scores.transpose(1, 0)
    plt.plot(lead_times, trans_score[0], 'o-', label='POD', color='green')
    plt.plot(lead_times, trans_score[1], 's-', label='CSI', color='red')
    plt.plot(lead_times, trans_score[2], '^-', label='FSS', color='blue')
    plt.plot(lead_times, trans_score[3], 'o-', label='SR', color='purple')
    plt.xlabel("Lead Time")
    plt.ylabel("Score")
    plt.title(
        f"{model} output {parameter} (thresh 75th percentile) "
        "Event Forecast Metrics")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


parameter = 'cape'
time = '2015-05-07-02UTC'


model = 'fuxi'
if model != 'gfs':
    lead_times = ['24h', '48h', '72h', '120h', '168h']
    scores = np.load(
        os.path.join(
            time,
            model,
            'scores',
            f'scores_{parameter}_{model}.npy'))
    print(scores)
else:
    lead_times = []
    for file in os.listdir(os.path.join(time, model)):
        if file.endswith('.nc') and '._' not in file:
            lead_times.append(file[-6:-3])
    lead_times.sort()
    lead_times = lead_times[1:]
    scores = np.load(
        os.path.join(
            time,
            model,
            'scores',
            f'scores_{parameter}_{model}.npy'))[
        1:]


# scores = np.load('scores_mean_positive_helicity_pangu.npy')
score_plot(lead_times, scores)
