import numpy as np
import os
import pandas as pd
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
    """
    Calculate asymmetric error bounds: mean -> P16 (lower) / P84 (upper).
    Returns (lower, upper), i.e., distances from mean to P16 and P84.
    """
    clean = data.flatten()
    clean = clean[~np.isnan(clean)]
    if len(clean) < 3:
        return 0.0, 0.0
    m = float(np.mean(clean))
    p16, p84 = np.percentile(clean, [16, 84])
    lower = max(0.0, m - p16)
    upper = max(0.0, p84 - m)
    return lower, upper


# ==================== Configuration ====================
# Lead times (should match MODE output)
lead_times = [24, 48, 72, 120, 168]

# Time list (same as used when generating CSV)
times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
         '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
         '2025-05-19-23UTC']
'''
# GFS times (can be replaced if needed)
times = ['2015-05-07-02UTC', '2015-05-25-20UTC', '2016-05-09-21UTC',
         '2023-04-20-02UTC', '2024-04-28-04UTC', '2024-05-07-02UTC',
         '2024-05-26-04UTC', '2025-05-19-23UTC']
'''

models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
# models = ['gfs']
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
os.makedirs('avg_mode', exist_ok=True)

# ==================== Main Loop ====================
for parameter in parameters:
    for model in models:
        print(f"\nProcessing MODE summary: {parameter} - {model}")

        # Initialize data lists for each lead time
        data_by_lead = {lt: {
            'interest': [],
            'cent_dist': [],
            'bound_dist': [],
            'area': [],
            'angle': [],
            'int_area': [],
            'fcst_obj': [],
            'obs_obj': [],
            'matches': []
        } for lt in lead_times}

        # Iterate over all time points to collect data
        for time in times:
            csv_path = os.path.join(
                time, model, 'MODE', f'MODE_summary_{parameter}_{model}.csv')
            if not os.path.exists(csv_path):
                print(f"  Warning: CSV not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
                continue

            # Ensure CSV contains required columns (MET compatible fields)
            required_cols = [
                'Lead Time (h)',
                'Mean Total Interest',
                'Mean Centroid Dist (grid)',
                'Mean Boundary Dist (grid)',
                'Mean Area Ratio',
                'Mean Angle Interest',
                'Mean Int Area Ratio',
                '#Fcst Obj',
                '#Obs Obj',
                '#Matches',
            ]
            if not all(col in df.columns for col in required_cols):
                print(
                    f"  Warning: {csv_path} missing required columns, "
                    "skipping")
                continue

            # Iterate over each row in the CSV (each lead time)
            for _, row in df.iterrows():
                lt = row['Lead Time (h)']
                if lt not in data_by_lead:
                    continue  # Skip unknown lead times

                # Extract data (handle possible NaN)
                interest = row['Mean Total Interest'] if not pd.isna(
                    row['Mean Total Interest']) else 0
                cent_dist = row['Mean Centroid Dist (grid)'] if not pd.isna(
                    row['Mean Centroid Dist (grid)']) else np.hypot(21, 25)
                bound_dist = row['Mean Boundary Dist (grid)'] if not pd.isna(
                    row['Mean Boundary Dist (grid)']) else np.hypot(21, 25)
                area = row['Mean Area Ratio'] if not pd.isna(
                    row['Mean Area Ratio']) else 0
                angle = row['Mean Angle Interest'] if not pd.isna(
                    row['Mean Angle Interest']) else 0
                int_area = row['Mean Int Area Ratio'] if not pd.isna(
                    row['Mean Int Area Ratio']) else 0
                fcst_obj = row['#Fcst Obj'] if not pd.isna(
                    row['#Fcst Obj']) else 0
                obs_obj = row['#Obs Obj'] if not pd.isna(
                    row['#Obs Obj']) else 0
                matches = row['#Matches'] if not pd.isna(
                    row['#Matches']) else 0

                data_by_lead[lt]['interest'].append(interest)
                data_by_lead[lt]['cent_dist'].append(cent_dist)
                data_by_lead[lt]['bound_dist'].append(bound_dist)
                data_by_lead[lt]['area'].append(area)
                data_by_lead[lt]['angle'].append(angle)
                data_by_lead[lt]['int_area'].append(int_area)
                data_by_lead[lt]['fcst_obj'].append(fcst_obj)
                data_by_lead[lt]['obs_obj'].append(obs_obj)
                data_by_lead[lt]['matches'].append(matches)

        # Check if there is any data
        total_samples = sum(
            len(data_by_lead[lt]['interest']) for lt in lead_times)
        if total_samples == 0:
            print(f"  No data for {parameter} - {model}, skipping")
            continue

        # Calculate statistics for each lead time
        summary_rows = []
        for lt in lead_times:
            interest_data = np.array(data_by_lead[lt]['interest'])
            cent_dist_data = np.array(data_by_lead[lt]['cent_dist'])
            bound_dist_data = np.array(data_by_lead[lt]['bound_dist'])
            area_data = np.array(data_by_lead[lt]['area'])
            angle_data = np.array(data_by_lead[lt]['angle'])
            int_area_data = np.array(data_by_lead[lt]['int_area'])
            fcst_obj_data = np.array(data_by_lead[lt]['fcst_obj'])
            obs_obj_data = np.array(data_by_lead[lt]['obs_obj'])
            matches_data = np.array(data_by_lead[lt]['matches'])

            row_dict = {'Lead Time (h)': lt}

            # --- Mean Interest ---
            if len(interest_data) > 0:
                mean_val, disp, name = calculate_dispersion_metrics(interest_data)
                row_dict['interest_mean'] = mean_val
                row_dict['interest_disp'] = disp
                row_dict['interest_disp_name'] = name
                lo, hi = asymmetric_bounds(interest_data)
                row_dict['interest_lower'] = lo
                row_dict['interest_upper'] = hi
            else:
                row_dict['interest_mean'] = np.nan
                row_dict['interest_disp'] = np.nan
                row_dict['interest_disp_name'] = 'no_data'
                row_dict['interest_lower'] = 0.0
                row_dict['interest_upper'] = 0.0

            # --- Mean Centroid Dist ---
            if len(cent_dist_data) > 0:
                mean_val, disp, name = calculate_dispersion_metrics(cent_dist_data)
                row_dict['cent_dist_mean'] = mean_val
                row_dict['cent_dist_disp'] = disp
                row_dict['cent_dist_disp_name'] = name
                lo, hi = asymmetric_bounds(cent_dist_data)
                row_dict['cent_dist_lower'] = lo
                row_dict['cent_dist_upper'] = hi
            else:
                row_dict['cent_dist_mean'] = np.nan
                row_dict['cent_dist_disp'] = np.nan
                row_dict['cent_dist_disp_name'] = 'no_data'
                row_dict['cent_dist_lower'] = 0.0
                row_dict['cent_dist_upper'] = 0.0

            # --- Mean Boundary Dist ---
            if len(bound_dist_data) > 0:
                mean_val, disp, name = calculate_dispersion_metrics(bound_dist_data)
                row_dict['bound_dist_mean'] = mean_val
                row_dict['bound_dist_disp'] = disp
                row_dict['bound_dist_disp_name'] = name
                lo, hi = asymmetric_bounds(bound_dist_data)
                row_dict['bound_dist_lower'] = lo
                row_dict['bound_dist_upper'] = hi
            else:
                row_dict['bound_dist_mean'] = np.nan
                row_dict['bound_dist_disp'] = np.nan
                row_dict['bound_dist_disp_name'] = 'no_data'
                row_dict['bound_dist_lower'] = 0.0
                row_dict['bound_dist_upper'] = 0.0

            # --- Mean Area Ratio ---
            if len(area_data) > 0:
                mean_val, disp, name = calculate_dispersion_metrics(area_data)
                row_dict['area_mean'] = mean_val
                row_dict['area_disp'] = disp
                row_dict['area_disp_name'] = name
                lo, hi = asymmetric_bounds(area_data)
                row_dict['area_lower'] = lo
                row_dict['area_upper'] = hi
            else:
                row_dict['area_mean'] = np.nan
                row_dict['area_disp'] = np.nan
                row_dict['area_disp_name'] = 'no_data'
                row_dict['area_lower'] = 0.0
                row_dict['area_upper'] = 0.0

            # --- Mean Angle Interest ---
            if len(angle_data) > 0:
                mean_val, disp, name = calculate_dispersion_metrics(angle_data)
                row_dict['angle_mean'] = mean_val
                row_dict['angle_disp'] = disp
                row_dict['angle_disp_name'] = name
                lo, hi = asymmetric_bounds(angle_data)
                row_dict['angle_lower'] = lo
                row_dict['angle_upper'] = hi
            else:
                row_dict['angle_mean'] = np.nan
                row_dict['angle_disp'] = np.nan
                row_dict['angle_disp_name'] = 'no_data'
                row_dict['angle_lower'] = 0.0
                row_dict['angle_upper'] = 0.0

            # --- Mean Int Area Ratio ---
            if len(int_area_data) > 0:
                mean_val, disp, name = calculate_dispersion_metrics(int_area_data)
                row_dict['int_area_mean'] = mean_val
                row_dict['int_area_disp'] = disp
                row_dict['int_area_disp_name'] = name
                lo, hi = asymmetric_bounds(int_area_data)
                row_dict['int_area_lower'] = lo
                row_dict['int_area_upper'] = hi
            else:
                row_dict['int_area_mean'] = np.nan
                row_dict['int_area_disp'] = np.nan
                row_dict['int_area_disp_name'] = 'no_data'
                row_dict['int_area_lower'] = 0.0
                row_dict['int_area_upper'] = 0.0

            # Object counts (mean only, dispersion is optional)
            if len(fcst_obj_data) > 0:
                row_dict['fcst_obj_mean'] = np.mean(fcst_obj_data)
                row_dict['obs_obj_mean'] = np.mean(obs_obj_data)
                row_dict['matches_mean'] = np.mean(matches_data)
            else:
                row_dict['fcst_obj_mean'] = np.nan
                row_dict['obs_obj_mean'] = np.nan
                row_dict['matches_mean'] = np.nan

            summary_rows.append(row_dict)

        # Create summary DataFrame and save
        df_summary = pd.DataFrame(summary_rows)
        output_dir = os.path.join('avg_mode', model)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f'mode_summary_{parameter}_{model}.csv')
        df_summary.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")

print("\nAll MODE summaries processed.")
