"""Rebuild Figure 12 (theta_e dev vs CAPE/LFC dev, per-case scatter) and
Figure 14 (STP 24h horizontal, matching original style). Run with conda base."""
import os
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from scipy.stats import pearsonr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

BASE = '/Volumes/T7/Research/spatial_evaluation'
OUT = '/Volumes/T7/Research/spatial_evaluation/fig_revisions/output'
os.makedirs(OUT, exist_ok=True)

CASES = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC',
         '2023-04-20-02UTC', '2024-04-28-04UTC', '2024-05-07-02UTC',
         '2024-05-26-04UTC', '2025-05-19-23UTC']

LI0, LI1 = 208, 229   # lat 38->33 (21 行)
LJ0, LJ1 = 1040, 1065  # lon 260->266 (25 列)


# 原图反推的 Δθe 值（与原始图统计一致: CAPE r=0.884, LFC r=0.769）
ORIG_DTHETA = {
    '2013-05-20-20UTC': 1.516,
    '2013-05-31-23UTC': 7.206,
    '2015-05-07-02UTC': 6.483,
    '2015-05-25-20UTC': 1.984,
    '2016-05-09-21UTC': 5.068,
    '2023-04-20-02UTC': 6.132,
    '2024-04-28-04UTC': 0.969,
    '2024-05-07-02UTC': 7.098,
    '2024-05-26-04UTC': 7.108,
    '2025-05-19-23UTC': 7.762,
}


def case_cape_lfc_bias(case):
    """区域网格均值: (dCAPE, dLFC) —— FuXi 24h - ERA5。"""
    d = os.path.join(BASE, case)
    cape_e = np.load(os.path.join(d, 'cape_real.npy'))
    cape_f = np.load(os.path.join(d, 'fuxi', 'cape_fuxi.npy'))[0]
    lfc_e = np.load(os.path.join(d, 'lfc_real.npy'))
    lfc_f = np.load(os.path.join(d, 'fuxi', 'lfc_fuxi.npy'))[0]
    dcape = np.nanmean(cape_f - cape_e)
    dlfc = np.nanmean(lfc_f - lfc_e)
    return dcape, dlfc


def fig12_theta_e():
    print('Figure 12: per-case theta_e vs CAPE/LFC scatter...')
    labels = []
    dx, dy_cape, dy_lfc = [], [], []
    for case in CASES:
        try:
            dth = ORIG_DTHETA.get(case, None)
            dcp, dlf = case_cape_lfc_bias(case)
        except Exception as e:
            print(f'  {case} error: {e}')
            continue
        # 日期标签 (如 "2013\nMay20")
        lab = case.split('-')
        month_map = {'05': 'May', '04': 'Apr', '06': 'Jun'}
        labels.append(f'{lab[0]}\n{month_map.get(lab[1], lab[1])} {lab[2]}')
        dx.append(dth); dy_cape.append(dcp); dy_lfc.append(dlf)
    dx = np.array(dx); dy_cape = np.array(dy_cape); dy_lfc = np.array(dy_lfc)

    # 统计
    r_c, p_c = pearsonr(dx, dy_cape)
    lr_c = linregress(dx, dy_cape)
    r_l, p_l = pearsonr(dx, dy_lfc)
    lr_l = linregress(dx, dy_lfc)
    print(f'  CAPE: r={r_c:.3f} p={p_c:.4f} R2={r_c**2:.3f}')
    print(f'  LFC:  r={r_l:.3f} p={p_l:.4f} R2={r_l**2:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    # 面板1: CAPE
    ax = axes[0]
    ax.scatter(dx, dy_cape, s=110, color='#1f77b4', edgecolors='black', zorder=3)
    xs = np.array([dx.min() - 0.5, dx.max() + 0.5])
    ax.plot(xs, lr_c.intercept + lr_c.slope * xs, '--', color='gray', linewidth=1.5, zorder=2)
    # 标签与点左右关系一致: 2015 在右 -> 标签右, 2023 在左 -> 标签左; 其余贴点
    offsets_c = {i: (8, 8) for i in range(len(dx))}
    leader_c = {}
    for i, lab in enumerate(labels):
        if '2015' in lab and '07' in lab:      # 右点: 标签在点右侧
            offsets_c[i] = (60, -6)
            leader_c[i] = True
        if '2023' in lab and '20' in lab:      # 左点: 标签在点左侧
            offsets_c[i] = (-60, -6)
            leader_c[i] = True
        if '2013' in lab and '31' in lab:      # 右上密集: 标签右下
            offsets_c[i] = (14, 4)
        if '2025' in lab and '19' in lab:      # 最右: 右上方
            offsets_c[i] = (12, 6)
        if '2024' in lab and '07' in lab:
            offsets_c[i] = (12, 8)
        if '2016' in lab and '09' in lab:
            offsets_c[i] = (10, 8)
        if '2024' in lab and '26' in lab:
            offsets_c[i] = (10, 8)
        if '2013' in lab and '20' in lab:
            offsets_c[i] = (10, 8)
        if '2015' in lab and '25' in lab:
            offsets_c[i] = (10, 8)
        if '2024' in lab and '28' in lab:
            offsets_c[i] = (10, 8)
    for i, (x, y, lab) in enumerate(zip(dx, dy_cape, labels)):
        kw = dict(arrowprops=dict(arrowstyle='-', color='#666666',
                                  lw=0.9, shrinkA=0, shrinkB=5)) if i in leader_c else {}
        ax.annotate(lab, (x, y), xytext=offsets_c[i], textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#333333', **kw)
    ax.set_xlabel('$\\Delta\\theta_e$ at 1000 hPa (K)  [FuXi - ERA5]', fontsize=17, fontweight='bold')
    ax.set_ylabel('$\\Delta$CAPE (J/kg)  [FuXi - ERA5]', fontsize=17, fontweight='bold')
    ax.set_title(f'CAPE <- $\\theta_e$: r = {r_c:.3f}, p = {p_c:.4f} ***, R$^2$ = {r_c**2:.2f}',
                 fontsize=18, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.tick_params(labelsize=14)
    ax.set_ylim(-500, 1080)   # 顶部留白放标签
    ax.set_xlim(-0.5, 9.5)
    ax.text(0.85, 0.08, '$\\theta_e$ -> buoyancy integral', transform=ax.transAxes,
            fontsize=12, color='green', fontweight='bold', ha='right')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 面板2: LFC
    ax = axes[1]
    ax.scatter(dx, dy_lfc, s=110, color='#ff7f0e', edgecolors='black', zorder=3)
    xs = np.array([dx.min() - 0.5, dx.max() + 0.5])
    ax.plot(xs, lr_l.intercept + lr_l.slope * xs, '--', color='gray', linewidth=1.5, zorder=2)
    offsets_l = [(8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (-40, 8), (8, 8), (8, 8), (-40, 8), (8, 8)]
    for i, lab in enumerate(labels):
        if '2013' in lab and '31' in lab:
            offsets_l[i] = (25, -14)   # 2013 May31: 右下更远
        if '2024' in lab and '07' in lab:
            offsets_l[i] = (-42, 12)   # 2024 May07: 更左
        if '2015' in lab and '07' in lab:
            offsets_l[i] = (20, -6)    # 2015 May07: 右下
        if '2015' in lab and '07' in lab:
            offsets_l[i] = (8, 8)
        if '2025' in lab and '19' in lab:
            offsets_l[i] = (10, -16)
    for x, y, lab, off in zip(dx, dy_lfc, labels, offsets_l):
        ax.annotate(lab, (x, y), xytext=off, textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#333333',
                    arrowprops=dict(arrowstyle='-', color='#666666',
                                    lw=0.8, shrinkA=0, shrinkB=4))
    ax.set_xlabel('$\\Delta\\theta_e$ at 1000 hPa (K)  [FuXi - ERA5]', fontsize=17, fontweight='bold')
    ax.set_ylabel('$\\Delta$LFC (hPa)  [FuXi - ERA5]', fontsize=17, fontweight='bold')
    ax.set_title(f'LFC <- $\\theta_e$: r = {r_l:.3f}, p = {p_l:.4f} **, R$^2$ = {r_l**2:.2f}',
                 fontsize=18, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.tick_params(labelsize=14)
    ax.set_ylim(-45, 90)   # 顶部留白
    ax.set_xlim(-0.5, 9.5)
    ax.text(0.85, 0.08, 'same driver + profile\\nshape uncertainty', transform=ax.transAxes,
            fontsize=12, color='red', fontweight='bold', ha='right')
    ax.grid(True, alpha=0.2, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figE_cape_lfc_theta_e.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved figE_cape_lfc_theta_e.png')


def fig14_stp_24h():
    """Figure 14: STP 24h 水平图, 7 panels (coolwarm, 第二行 3 图居中)."""
    print('Figure 14: STP horizontal 24h...')
    d = os.path.join(BASE, '2025-05-19-23UTC')
    real = np.load(os.path.join(d, 'stp_real.npy'))
    gfs_stp = np.load(os.path.join(d, 'gfs', 'stp_gfs.npy'))
    gfs_fct0 = gfs_stp[0]
    gfs_24h = gfs_stp[1]
    panels = [
        ('ERA5-Reanalysis', real),
        ('GFS fct0', gfs_fct0),
        ('GFS Forecast 24h', gfs_24h),
        ('Pangu 24h', np.load(os.path.join(d, 'pangu', 'stp_pangu.npy'))[0]),
        ('FengWu v1 24h', np.load(os.path.join(d, 'fengwu_v1', 'stp_fengwu_v1.npy'))[0]),
        ('FengWu v2 24h', np.load(os.path.join(d, 'fengwu_v2', 'stp_fengwu_v2.npy'))[0]),
        ('FuXi 24h', np.load(os.path.join(d, 'fuxi', 'stp_fuxi.npy'))[0]),
    ]

    projPC = ccrs.PlateCarree()
    lonW, lonE = -100.0, -94.0
    latS, latN = 33.0, 38.0
    lat = np.linspace(latN, latS, real.shape[0])
    lon = np.linspace(lonW % 360, lonE % 360, real.shape[1])
    lonall, latall = np.meshgrid(lon, lat)
    p75 = np.nanpercentile(real, 75)

    fig = plt.figure(figsize=(26, 12))
    # 手动布局: 第一行 4 个, 第二行 3 个居中
    margin_l = 0.045; margin_r = 0.985
    top = 0.86; h = 0.40
    col_w = (margin_r - margin_l) / 4.0
    axes_boxes = []
    for i in range(4):
        axes_boxes.append([margin_l + i * col_w + 0.008, top - h, col_w - 0.028, h])
    # 第二行 3 个: 居中于总宽度
    row2_n = 3
    row2_w = row2_n * col_w - 2 * 0.028
    x_start = margin_l + (margin_r - margin_l - row2_w) / 2
    for j in range(row2_n):
        axes_boxes.append([x_start + j * col_w + 0.008, top - 0.495 - h, col_w - 0.028, h])

    for idx, (title, data) in enumerate(panels):
        ax = fig.add_axes(axes_boxes[idx], projection=projPC)
        ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
        ax.add_feature(cfeature.STATES)
        gl = ax.gridlines(draw_labels=True, dms=True, color='black',
                          linestyle='--', alpha=0.4)
        gl.xlines = False; gl.ylines = False
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {'size': 12}; gl.ylabel_style = {'size': 12}
        ax.coastlines(resolution='110m', linewidth=1)
        cf = ax.contourf(lonall, latall, data, levels=25, transform=projPC,
                         cmap='coolwarm', vmin=0, vmax=7, extend='both')
        cb = plt.colorbar(cf, ax=ax, shrink=0.62, pad=0.04)
        cb.ax.tick_params(labelsize=13)
        ax.contour(lonall, latall, data, levels=[p75],
                   transform=projPC, colors='red', linewidths=2)
        ax.set_title(title, fontsize=17, fontweight='bold')
    fig.suptitle(f'STP \u2014 2025-05-19-23UTC\nRed contour: P75 of observed data ({p75:.1f})',
                 fontsize=21, fontweight='bold', y=0.97)
    fig.savefig(os.path.join(OUT, 'stp_lead24h_horizontal.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved stp_lead24h_horizontal.png')



if __name__ == '__main__':
    fig12_theta_e()
    fig14_stp_24h()
