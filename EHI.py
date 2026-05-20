import numpy as np
import os

# All EF3 tornado events
times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
         '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
         '2025-05-19-23UTC']


models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi', 'real']
for model in models:
    if model != 'real':
        for time in times:
            cape = np.load(os.path.join(time, model, f'cape_{model}.npy'))
            srh = np.load(
                os.path.join(
                    time,
                    model,
                    f'positive_helicity_{model}.npy'))
            ehi = cape * srh / 160000
            np.save(os.path.join(time, model, f'ehi_{model}.npy'), ehi)
    else:
        for time in times:
            cape = np.load(os.path.join(time, f'cape_{model}.npy'))
            srh = np.load(os.path.join(time, f'positive_helicity_{model}.npy'))
            ehi = cape * srh / 160000
            np.save(os.path.join(time, f'ehi_{model}.npy'), ehi)
