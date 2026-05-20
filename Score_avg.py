import numpy as np
import os
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def assess_distribution(data, alpha=0.05, z_general=1.96, z_extreme=3.0):
    """
    Normality and skewness assessment (suitable for small samples)
    Returns dict:
      dist_type: 'normal' | 'non_normal' | 'insufficient'
      skew_type: 'symmetric' | 'mild_left' | 'mild_right' |
      'extreme_left' | 'extreme_right' | 'unknown'
      kurt_type: 'normal_tail' | 'mild_heavy' | 'mild_light' |
      'extreme_heavy' | 'extreme_light' | 'unknown'
      shapiro_p, skewness, kurtosis(excess), z_skew, z_kurt, n
    """
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    n = len(data_flat)

    if n < 8:
        return {
            'dist_type': 'insufficient',
            'skew_type': 'unknown',
            'kurt_type': 'unknown',
            'shapiro_p': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'z_skew': np.nan,
            'z_kurt': np.nan,
            'n': n
        }

    # Shapiro-Wilk (recommended for small samples)
    shapiro_p = stats.shapiro(data_flat)[1]

    # Skewness/kurtosis (unbiased estimation for small samples)
    skewness = stats.skew(data_flat, bias=False)
    kurtosis = stats.kurtosis(
        data_flat,
        fisher=True,
        bias=False)  # excess kurtosis

    z_skew = skewness / np.sqrt(6 / n)
    z_kurt = kurtosis / np.sqrt(24 / n)

    # Skewness classification
    if abs(z_skew) <= z_general:
        skew_type = 'symmetric'
    elif abs(z_skew) <= z_extreme:
        skew_type = 'mild_right' if skewness > 0 else 'mild_left'
    else:
        skew_type = 'extreme_right' if skewness > 0 else 'extreme_left'

    # Kurtosis classification (tails)
    if abs(z_kurt) <= z_general:
        kurt_type = 'normal_tail'
    elif abs(z_kurt) <= z_extreme:
        kurt_type = 'mild_heavy' if kurtosis > 0 else 'mild_light'
    else:
        kurt_type = 'extreme_heavy' if kurtosis > 0 else 'extreme_light'

    # Normality判定：Shapiro不拒绝 + 偏度/峰度不显著
    if (shapiro_p >= alpha) and (abs(z_skew) <=
                                 z_general) and (abs(z_kurt) <= z_general):
        dist_type = 'normal'
    else:
        dist_type = 'non_normal'

    return {
        'dist_type': dist_type,
        'skew_type': skew_type,
        'kurt_type': kurt_type,
        'shapiro_p': shapiro_p,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'z_skew': z_skew,
        'z_kurt': z_kurt,
        'n': n
    }


def check_distribution_type(data):
    info = assess_distribution(data)
    return info['dist_type'], info


def calculate_dispersion_metrics(data):
    """
    Calculate dispersion metrics based on normality/skewness assessment
    """
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]

    if len(data_flat) == 0:
        return 0, 0, 'no_data'

    mean_val = np.mean(data_flat)
    dist_type, info = check_distribution_type(data_flat)

    if dist_type == 'normal':
        dispersion = np.std(data_flat, ddof=1)
        dispersion_name = 'std'
    else:
        # Mildly skewed: try trimmed standard deviation, otherwise use robust estimation
        q1, q3 = np.percentile(data_flat, [25, 75])
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        filtered_data = data_flat[(data_flat >= lower) & (data_flat <= upper)]

        if info['skew_type'] in [
                'mild_left', 'mild_right'] and abs(
                info['z_kurt']) <= 3.0:
            if len(filtered_data) > len(data_flat) * 0.9:
                dispersion = np.std(filtered_data, ddof=1)
                dispersion_name = 'std_trimmed'
            else:
                mad = np.median(np.abs(data_flat - np.median(data_flat)))
                dispersion = mad * 1.4826 if mad > 0 else iqr / 1.35
                dispersion_name = 'robust_std'
        else:
            mad = np.median(np.abs(data_flat - np.median(data_flat)))
            dispersion = mad * 1.4826 if mad > 0 else iqr / 1.35
            dispersion_name = 'robust_mad'

    return mean_val, dispersion, dispersion_name


def asymmetric_bounds(data):
    """Calculate asymmetric error bounds: mean -> P16 (lower) / P84 (upper)."""
    clean = data.flatten()
    clean = clean[~np.isnan(clean)]
    if len(clean) < 3:
        return 0.0, 0.0
    m = float(np.mean(clean))
    p16, p84 = np.percentile(clean, [16, 84])
    return max(0.0, m - p16), max(0.0, p84 - m)


# All EF3
times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
         '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
         '2025-05-19-23UTC']
models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
parameters = [
    'cape',
    'positive_helicity',
    'lcl',
    'lfc',
    'k_idx',
    'negative_helicity',
    'cin',
    'ehi']

# Create output directory
os.makedirs('avg_scores', exist_ok=True)

for parameter in parameters:
    for model in models:
        all_ds = []
        distribution_info = {}  # Record distribution type info

        for time in times:
            path = os.path.join(
                time,
                model,
                'scores',
                f'scores_{parameter}_{model}.npy')
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue

            ds = np.load(path)  # (5,4) [POD, CSI, FSS, SR] columns as metrics

            # Replace NaN with 0 (real field always has data, NaN comes from boundary cases in computation)
            ds = np.nan_to_num(ds, nan=0.0)
            all_ds.append(ds)

        if len(all_ds) == 0:
            print(f"No data for {parameter} - {model}")
            continue

        all_ds = np.stack(all_ds, axis=0)  # (n_times, 5, 4)

        # Calculate mean and dispersion for each metric and lead time
        mean_scores = np.zeros((5, 4))
        dispersion_scores = np.zeros((5, 4))
        dispersion_names = np.empty((5, 4), dtype=object)
        lower_err = np.zeros((5, 4))
        upper_err = np.zeros((5, 4))

        for i in range(5):  # Lead times
            for j in range(4):  # Metrics
                data_slice = all_ds[:, i, j]
                mean_val, dispersion, disp_name = calculate_dispersion_metrics(
                    data_slice)
                mean_scores[i, j] = mean_val
                dispersion_scores[i, j] = dispersion
                dispersion_names[i, j] = disp_name
                lo, hi = asymmetric_bounds(data_slice)
                lower_err[i, j] = lo
                upper_err[i, j] = hi

        # Save results
        save_path = os.path.join('avg_scores', model)
        os.makedirs(save_path, exist_ok=True)

        np.savez(
            os.path.join(
                save_path,
                f'scores_meanvar_{parameter}_{model}.npz'),
            mean=mean_scores,
            dispersion=dispersion_scores,
            dispersion_names=dispersion_names,
            lower_err=lower_err,
            upper_err=upper_err)

        # Optionally print distribution type statistics
        unique_names, counts = np.unique(dispersion_names, return_counts=True)
        print(
            f"{parameter} - {model}: Distribution types - "
            f"{dict(zip(unique_names, counts))}")
