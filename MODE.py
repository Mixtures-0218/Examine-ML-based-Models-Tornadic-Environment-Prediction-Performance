import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pandas as pd
import os

# ==================== Configuration ====================

times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
         '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
         '2025-05-19-23UTC']

# times = ['2025-05-19-23UTC']
models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
# models = ['gfs']
parameters = [
    'cape',
    'cin',
    'k_idx',
    'positive_helicity',
    'negative_helicity',
    'lcl',
    'lfc',
    'stp']

# —— MET-aligned parameters ——
lead_times = [24, 48, 72, 120, 168]
indices = [0, 1, 2, 3, 4]

grid_res = 25.0              # Grid resolution (km), 0.25 degrees ~ 25km (mid-latitude)
conv_radius = int(np.round(60.0 / grid_res))  # Convolution radius (grid points), MET default 60/grid_res
total_interest_thresh = 0.7  # MET default 0.7
min_area = 5                 # Minimum object area (grid points), filter noise fragments

# —— Interest function weights (MET default) ——
WEIGHT_CENTROID = 2.0
WEIGHT_BOUNDARY = 4.0
WEIGHT_ANGLE = 1.0
WEIGHT_AREA = 1.0
WEIGHT_INT_AREA = 2.0
TOTAL_WEIGHT = (WEIGHT_CENTROID + WEIGHT_BOUNDARY + WEIGHT_ANGLE
                + WEIGHT_AREA + WEIGHT_INT_AREA)

# ==================== Utility Functions ====================


def _piecewise_linear(x, breakpoints):
    """Piecewise linear interpolation. breakpoints: [(x0, y0), (x1, y1), ...] sorted by x ascending"""
    bp = sorted(breakpoints, key=lambda p: p[0])
    if x <= bp[0][0]:
        return float(bp[0][1])
    if x >= bp[-1][0]:
        return float(bp[-1][1])
    for k in range(len(bp) - 1):
        x0, y0 = bp[k]
        x1, y1 = bp[k + 1]
        if x0 <= x <= x1:
            frac = (x - x0) / (x1 - x0)
            return float(y0 + (y1 - y0) * frac)
    return 0.0


def _get_boundary(label_image, obj_label):
    """Extract interior boundary pixel coordinates of a single object (perimeter pixels)"""
    mask = (label_image == obj_label)
    # Erode then XOR with original mask to get interior boundary (consistent with MET MODE)
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    boundary = mask & (~eroded)
    coords = np.argwhere(boundary)
    return coords


def compute_interest(fp, op, label_fcst, label_obs,
                     fcst_label, obs_label):
    """
    MET-compatible pairwise interest calculation.
    Returns: (total_interest, detail_dict)
    """
    # 1. Centroid distance
    d_cent = np.hypot(
        fp.centroid[1] - op.centroid[1],
        fp.centroid[0] - op.centroid[0])
    # Piecewise linear: 1.0 up to 60/grid_res, decays to 0 at 600/grid_res
    i_cent = _piecewise_linear(
        d_cent,
        [(0.0, 1.0),
         (60.0 / grid_res, 1.0),
         (600.0 / grid_res, 0.0)])

    # 2. Boundary distance
    bdy_f = _get_boundary(label_fcst, fcst_label)
    bdy_o = _get_boundary(label_obs, obs_label)
    if len(bdy_f) == 0 or len(bdy_o) == 0:
        d_bound = np.hypot(21, 25)  # Domain diagonal
    else:
        d_bound = float(np.min(cdist(bdy_f, bdy_o)))
    i_bound = _piecewise_linear(
        d_bound,
        [(0.0, 1.0),
         (400.0 / grid_res, 0.0)])

    # 3. Angle difference
    raw_diff_rad = fp.orientation - op.orientation
    angle_deg = np.abs(np.degrees(raw_diff_rad))
    # Handle 180-degree symmetry
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg
    i_angle = _piecewise_linear(
        angle_deg,
        [(0.0, 1.0), (30.0, 1.0), (90.0, 0.0)])

    # 4. Area ratio
    area_ratio = min(fp.area, op.area) / max(fp.area, op.area)
    i_area = _piecewise_linear(
        area_ratio,
        [(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)])

    # 5. Intersection area ratio
    mask_f = (label_fcst == fcst_label)
    mask_o = (label_obs == obs_label)
    intersection = np.logical_and(mask_f, mask_o).sum()
    int_area_ratio = intersection / min(fp.area, op.area)
    i_int_area = _piecewise_linear(
        int_area_ratio,
        [(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)])

    # Weighted total interest
    total = (
        WEIGHT_CENTROID * i_cent
        + WEIGHT_BOUNDARY * i_bound
        + WEIGHT_ANGLE * i_angle
        + WEIGHT_AREA * i_area
        + WEIGHT_INT_AREA * i_int_area
    ) / TOTAL_WEIGHT

    return total, {
        'interest_cent': i_cent,
        'interest_bound': i_bound,
        'interest_angle': i_angle,
        'interest_area': i_area,
        'interest_int_area': i_int_area,
        'd_cent': d_cent,
        'd_bound': d_bound,
        'angle_diff_deg': angle_deg,
        'area_ratio': area_ratio,
        'int_area_ratio': int_area_ratio,
    }


def extract_props(props, field_name):
    """Extract object properties into a DataFrame"""
    data = []
    for i, p in enumerate(props):
        vals = p.intensity_image[p.image]
        intensity_50 = np.percentile(vals, 50) if len(vals) > 0 else np.nan
        intensity_90 = np.percentile(vals, 90) if len(vals) > 0 else np.nan
        data.append({
            'field': field_name,
            'obj_id': i + 1,
            'area': p.area,
            'centroid_y': p.centroid[0],
            'centroid_x': p.centroid[1],
            'weighted_centroid_y': p.weighted_centroid[0],
            'weighted_centroid_x': p.weighted_centroid[1],
            'orientation_deg': np.degrees(p.orientation) % 180,
            'major_axis': p.major_axis_length,
            'minor_axis': p.minor_axis_length,
            'intensity_50': intensity_50,
            'intensity_90': intensity_90
        })
    return pd.DataFrame(data)


# ==================== Main Loop ====================
for time in times:
    for model in models:
        for param in parameters:
            print(
                f"\n========== Processing: time={time}, model={model}, "
                f"param={param} ==========\n"
            )

            # Build paths
            obs_path = os.path.join(time, model)
            output_path = os.path.join(time, model, 'MODE')
            os.makedirs(output_path, exist_ok=True)

            # Load data
            if model != 'gfs':
                obs = np.load(os.path.join(time, f'{param}_real.npy'))
                fcst_all = np.load(
                    os.path.join(obs_path, f'{param}_{model}.npy'))
            else:
                obs = np.load(os.path.join(time, 'gfs', f'{param}_gfs.npy'))[0]
                fcst_all = np.load(
                    os.path.join(obs_path, f'{param}_{model}.npy'))[1:]

            # Determine global threshold (based on observed field quantile)
            if param in ['cin', 'negative_helicity']:
                T = np.nanpercentile(obs, 25)
            else:
                T = np.nanpercentile(obs, 75)
            print(f"Global threshold T = {T:.2f}")

            # Run MODE for each forecast lead time
            results = []

            for idx, lead_time in zip(indices, lead_times):
                print(f"\n--- Lead Time: {lead_time}h (index {idx}) ---")

                fcst = fcst_all[idx]

                # Convolution smoothing (circular kernel)
                if conv_radius > 0:
                    y, x = np.ogrid[-conv_radius:conv_radius + 1,
                                    -conv_radius:conv_radius + 1]
                    kernel = (x**2 + y**2 <= conv_radius**2).astype(float)
                    kernel = kernel / kernel.sum()
                    smooth_fcst = ndimage.convolve(fcst, kernel, mode='nearest')
                    smooth_obs = ndimage.convolve(obs, kernel, mode='nearest')
                else:
                    smooth_fcst = fcst
                    smooth_obs = obs

                # Binarize
                if param in ['cin', 'negative_helicity']:
                    mask_fcst = smooth_fcst <= T
                    mask_obs = smooth_obs <= T
                else:
                    mask_fcst = smooth_fcst >= T
                    mask_obs = smooth_obs >= T

                # Label connected objects
                labeled_fcst, num_fcst = ndimage.label(mask_fcst)
                labeled_obs, num_obs = ndimage.label(mask_obs)
                print(f"Forecast objects: {num_fcst}, Observed objects: {num_obs}")

                # Filter objects smaller than min_area
                if min_area > 1:
                    for lbl in range(1, num_fcst + 1):
                        if np.sum(labeled_fcst == lbl) < min_area:
                            labeled_fcst[labeled_fcst == lbl] = 0
                    for lbl in range(1, num_obs + 1):
                        if np.sum(labeled_obs == lbl) < min_area:
                            labeled_obs[labeled_obs == lbl] = 0
                    # Do not relabel (avoid merging large objects connected through small ones)
                    num_fcst = len(np.unique(labeled_fcst)) - 1
                    num_obs = len(np.unique(labeled_obs)) - 1
                    print(f"After filtering (min_area={min_area}): "
                          f"Forecast objects: {num_fcst}, Observed objects: {num_obs}")

                # Extract object properties (use original unsmoothed field as intensity_image)
                props_fcst = regionprops(labeled_fcst, intensity_image=fcst)
                props_obs = regionprops(labeled_obs, intensity_image=obs)

                df_fcst = extract_props(props_fcst, 'forecast')
                df_obs = extract_props(props_obs, 'observed')

                # Object matching
                n_f = len(props_fcst)
                n_o = len(props_obs)

                matches = []
                df_match = pd.DataFrame()

                if n_f == 0 or n_o == 0:
                    print("Warning: one field has no objects")
                else:
                    interest_mat = np.zeros((n_f, n_o))
                    detail_mat = [[None for _ in range(n_o)] for _ in range(n_f)]

                    for i, fp in enumerate(props_fcst):
                        for j, op in enumerate(props_obs):
                            total, detail = compute_interest(
                                fp, op,
                                labeled_fcst, labeled_obs,
                                fp.label, op.label)
                            interest_mat[i, j] = total
                            detail_mat[i][j] = detail

                    cost_mat = 1.0 - interest_mat
                    row_ind, col_ind = linear_sum_assignment(cost_mat)

                    for i, j in zip(row_ind, col_ind):
                        if interest_mat[i, j] < total_interest_thresh:
                            continue
                        d = detail_mat[i][j]
                        matches.append({
                            'forecast_id': props_fcst[i].label,
                            'obs_id': props_obs[j].label,
                            'total_interest': interest_mat[i, j],
                            'interest_cent': d['interest_cent'],
                            'interest_bound': d['interest_bound'],
                            'interest_angle': d['interest_angle'],
                            'interest_area': d['interest_area'],
                            'interest_int_area': d['interest_int_area'],
                            'centroid_dist': d['d_cent'],
                            'boundary_dist': d['d_bound'],
                            'angle_diff_deg': d['angle_diff_deg'],
                            'area_ratio': d['area_ratio'],
                            'int_area_ratio': d['int_area_ratio'],
                        })

                    df_match = pd.DataFrame(matches)
                    print(f"Number of matched pairs: {len(df_match)}")

                results.append({
                    'lead_time': lead_time,
                    'fcst': fcst,
                    'mask_fcst': mask_fcst,
                    'props_fcst': props_fcst,
                    'df_fcst': df_fcst,
                    'mask_obs': mask_obs,
                    'props_obs': props_obs,
                    'df_obs': df_obs,
                    'df_match': df_match
                })

            # ==================== Summary Statistics ====================
            print("\n--- Summary Across All Lead Times ---")
            summary_data = []
            for res in results:
                df_match = res['df_match']
                if len(df_match) > 0:
                    summary_data.append({
                        'Lead Time (h)': res['lead_time'],
                        '#Fcst Obj': len(res['props_fcst']),
                        '#Obs Obj': len(res['props_obs']),
                        '#Matches': len(df_match),
                        'Mean Total Interest': df_match['total_interest'].mean(),
                        'Mean Centroid Interest':
                            df_match['interest_cent'].mean(),
                        'Mean Boundary Interest':
                            df_match['interest_bound'].mean(),
                        'Mean Angle Interest':
                            df_match['interest_angle'].mean(),
                        'Mean Area Interest':
                            df_match['interest_area'].mean(),
                        'Mean Int Area Interest':
                            df_match['interest_int_area'].mean(),
                        'Mean Centroid Dist (grid)':
                            df_match['centroid_dist'].mean(),
                        'Mean Boundary Dist (grid)':
                            df_match['boundary_dist'].mean(),
                        'Mean Angle Diff (deg)':
                            df_match['angle_diff_deg'].mean(),
                        'Mean Area Ratio':
                            df_match['area_ratio'].mean(),
                        'Mean Int Area Ratio':
                            df_match['int_area_ratio'].mean(),
                    })
                else:
                    summary_data.append({
                        'Lead Time (h)': res['lead_time'],
                        '#Fcst Obj': len(res['props_fcst']),
                        '#Obs Obj': len(res['props_obs']),
                        '#Matches': 0,
                        'Mean Total Interest': 0,
                        'Mean Centroid Interest': 0,
                        'Mean Boundary Interest': 0,
                        'Mean Angle Interest': 0,
                        'Mean Area Interest': 0,
                        'Mean Int Area Interest': 0,
                        'Mean Centroid Dist (grid)': np.hypot(21, 25),
                        'Mean Boundary Dist (grid)': np.hypot(21, 25),
                        'Mean Angle Diff (deg)': 0,
                        'Mean Area Ratio': 0,
                        'Mean Int Area Ratio': 0,
                    })

            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False))

            # Save summary results
            output_file = os.path.join(
                output_path, f'MODE_summary_{param}_{model}.csv')
            df_summary.to_csv(output_file, index=False)
            print(f"Summary saved to: {output_file}")

print("\nAll combinations processed!")
