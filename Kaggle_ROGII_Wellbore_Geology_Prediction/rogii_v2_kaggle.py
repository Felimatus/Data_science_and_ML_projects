"""
rogii_v2_kaggle.py — Kaggle notebook version of the best-of-all-worlds pipeline
                      for ROGII Wellbore Geology Prediction

Combines top techniques from:
  v41 (LB 9.66):  7-beam search, DTW deterministic + stochastic, multi-seed
                   particle filters (600 particles), bucketed GBDT training,
                   per-bucket Optuna post-processing, DTW/PF offset features
  v26 (LB 9.579): cal-zone augmentation, test-well online training, TabICL,
                   exact train-coordinate overlap blend

Plus: np.gradient for O(h²) derivatives

Target: sub-9.5 RMSE
Runtime estimate: 6–8 h on T4×2
Kaggle: GPU T4×2 accelerator required
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Setup & Configuration
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, time, json, subprocess, multiprocessing, warnings, gc, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool
from joblib import Parallel, delayed
from numba import njit
import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)
_T0 = time.time()


def _dbg(msg):
    print(f"[{(time.time() - _T0) / 60:.1f}m] {msg}", flush=True)


def env_flag(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v and v.strip() else int(default)


def env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v and v.strip() else float(default)


RUNNING_ON_KAGGLE = Path("/kaggle/input").exists()
NCPU = max(1, multiprocessing.cpu_count())

_gpu_names = ""
try:
    _gpu_names = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        shell=True, stderr=subprocess.DEVNULL, timeout=20
    ).decode(errors="ignore").strip()
except Exception:
    pass
GPU_COUNT = len([x for x in _gpu_names.splitlines() if x.strip()])
FORCE_CPU = env_flag("ROGII_FORCE_CPU", False)
if FORCE_CPU:
    GPU_COUNT = 0
if GPU_COUNT == 0 and not FORCE_CPU:
    print("WARNING: No GPU detected. Set Accelerator to 'GPU T4 x2' in Kaggle notebook settings.")
CATBOOST_DEVICES = os.environ.get(
    "ROGII_CATBOOST_DEVICES",
    ":".join(str(i) for i in range(GPU_COUNT)) if GPU_COUNT else "0",
)
_dbg(f"GPUs: {_gpu_names!r}  GPU_COUNT={GPU_COUNT}  NCPU={NCPU}  FORCE_CPU={FORCE_CPU}")

# ── Data paths (Kaggle-first) ──
def _find_data():
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    if os.environ.get("ROGII_DATA_DIR"):
        candidates.insert(0, Path(os.environ["ROGII_DATA_DIR"]))
    candidates.extend([Path.cwd(), Path.cwd() / "rogii-wellbore-geology-prediction"])
    for p in candidates:
        if (p / "train").is_dir() and (p / "sample_submission.csv").is_file():
            return p
    raise FileNotFoundError(
        "Data directory not found. Ensure the competition dataset is added "
        "via 'Add Data' -> 'Competition Data' in the Kaggle notebook sidebar."
    )


DATA = _find_data()
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
SAMPLE = DATA / "sample_submission.csv"
OUTPUT_DIR = Path("/kaggle/working") if RUNNING_ON_KAGGLE else Path.cwd()
_dbg(f"DATA={DATA}  OUTPUT={OUTPUT_DIR}")

# ── Tunable constants ──
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]  # geological formation surfaces
PLANE_K = 10              # number of nearest wells for FormationPlaneKNN spatial imputer
DENSE_SPW = 60            # samples per well for DenseANCCImputer (spatial density)
DENSE_K = 20              # number of nearest points for DenseANCCImputer lookup
N_SPLITS = 5              # CV folds
N_AUG_SPLITS = 1          # number of cal-zone augmentation splits per well
MIN_KNOWN_FOR_AUG = 20    # minimum known TVT points required to augment a well

# Beam search configs: (beam_width, movement_cost, error_scale, GR_smooth_radius, tag)
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),      # conservative
    (10, 8.0, 64.0, 2, "loose"),       # loose movement penalty
    (8, 35.0, 220.0, 1, "vcons"),      # very conservative
    (10, 14.0, 90.0, 5, "sm5"),        # heavily smoothed GR
    (20, 4.0, 36.0, 3, "vloose"),      # very loose, wide beam
    (12, 12.0, 100.0, 3, "mid"),       # balanced
    (15, 25.0, 180.0, 2, "stiff"),     # stiff movement penalty
]

# ── Z-aware particle filter parameters ──
PF_N = 600                # number of particles
ANCC_N = 600              # number of particles for ANCC particle filter
PF_NUM_SEEDS = 2          # random seeds for multi-seed averaging
PF_MOM = 0.993            # velocity momentum (autoregressive coefficient)
PF_VN = 0.005             # velocity process noise std
PF_PN = 0.01              # position process noise std
PF_GR_SIG_MIN = 10.0      # minimum GR likelihood sigma (clamp)
PF_GR_SIG_MAX = 60.0      # maximum GR likelihood sigma (clamp)
PF_GR_SIG_DEF = 30.0      # default GR sigma when insufficient data
PF_INIT_V_STD = 0.02      # initial velocity std for particle initialization
PF_INIT_SPR = 0.5         # initial position spread (std) around last known TVT
PF_RESAMP = 0.5           # effective sample size threshold for resampling (fraction of N)
PF_ROUGH_P = 0.2          # roughening noise std for position after resampling
PF_ROUGH_V = 0.003        # roughening noise std for velocity after resampling
PF_GR_WIN = 5             # rolling window size for smoothed GR in Z-aware filter
PF_GR_WT = 0.3            # weight of smoothed GR likelihood vs raw GR likelihood

# ── ANCC particle filter parameters ──
ANCC_ALPHA = 0.998        # rate autoregressive coefficient
ANCC_RN = 0.002           # rate process noise std
ANCC_PN = 0.005           # position process noise std
ANCC_IR = 0.01            # initial rate estimate
ANCC_IS = 0.3             # initial position spread (std)
ANCC_RP = 0.1             # roughening noise std for position
ANCC_RR = 0.001           # roughening noise std for rate

# ── DTW parameters ──
DTW_RADII = (20, 50, 100, 200)  # Sakoe-Chiba band radii for multi-scale DTW
DTW_STOCH_K = 8           # number of stochastic DTW realizations
DTW_STOCH_TEMP = 3.0      # Gumbel noise temperature for stochastic DTW

# ── Optional pipeline flags ──
RUN_TABICL = env_flag("ROGII_RUN_TABICL", False)              # enable TabICL (disabled by default)
EXACT_OVERLAP_WEIGHT = env_float("ROGII_EXACT_BLEND_WEIGHT", 0.28)  # blend weight for exact XYZ overlap
USE_SAVGOL = env_flag("ROGII_USE_SAVGOL", False)              # enable Savitzky-Golay smoothing (off by default)

# ── GBDT training parameters ──
LGB_N_EST = 8000          # max boosting rounds for LightGBM (early stopping applies)
CB_ITERS = 8000           # max iterations for CatBoost (early stopping applies)
XGB_N_EST = 8000          # max boosting rounds for XGBoost (early stopping applies)
PP_OPTUNA_TRIALS = 300    # number of Optuna trials for post-processing optimization

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Numba JIT Kernels
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    # Linear interpolation on a regular grid. Maps a value v to its interpolated grid value.
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    # Systematic resampling with roughening for particle filters.
    # Redraws N particles proportional to weights w, adding noise (rp, rv) to avoid degeneracy.
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    # Beam search alignment: maps well GR (sgr) to typewell GR (tw_gr) starting at index si.
    # Keeps top-BS candidates at each step, scored by GR mismatch (/ es) + movement cost (mc).
    # Returns the best index path through the typewell.
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


@njit(cache=True)
def _dtw_sakoe_chiba(query, ref, radius):
    # Dynamic Time Warping with Sakoe-Chiba band constraint.
    # Computes optimal alignment between query and ref within a diagonal band of given radius.
    # Returns cost matrix D and the warp path (pi, pj).
    N = len(query)
    M = len(ref)
    INF = 1e18
    D = np.full((N, M), INF)
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius)
        j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (query[i] - ref[j]) ** 2
            if i == 0 and j == 0:
                D[i, j] = cost
            elif i == 0:
                prev = D[i, j - 1]
                D[i, j] = cost + (prev if prev < INF else INF)
            elif j == 0:
                prev = D[i - 1, j]
                D[i, j] = cost + (prev if prev < INF else INF)
            else:
                a = D[i - 1, j - 1]
                b = D[i - 1, j]
                c = D[i, j - 1]
                mn = a if a < b else b
                mn = mn if mn < c else c
                D[i, j] = cost + (mn if mn < INF else INF)
    i = N - 1
    j = M - 1
    pi = np.zeros(N + M, np.int64)
    pj = np.zeros(N + M, np.int64)
    k = 0
    while i > 0 or j > 0:
        pi[k] = i
        pj[k] = j
        k += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            a = D[i - 1, j - 1]
            b = D[i - 1, j]
            c = D[i, j - 1]
            if a <= b and a <= c:
                i -= 1
                j -= 1
            elif b <= c:
                i -= 1
            else:
                j -= 1
    pi[k] = 0
    pj[k] = 0
    k += 1
    return D, pi[:k], pj[:k]


@njit(cache=True)
def _dtw_path_to_tvt(pi, pj, tw_tvt, N):
    # Converts a DTW warp path into TVT predictions by looking up typewell TVT at matched indices.
    j_for_i = np.zeros(N, np.int64)
    for k in range(len(pi)):
        j_for_i[pi[k]] = pj[k]
    result = np.empty(N, np.float32)
    for i in range(N):
        result[i] = tw_tvt[j_for_i[i]]
    return result


@njit(cache=True)
def _dtw_path_slope(pi, pj, N, smooth_win=5):
    # Computes the local slope of the DTW warp path (how fast the alignment advances through the typewell).
    j_for_i = np.zeros(N, np.float64)
    for k in range(len(pi)):
        j_for_i[pi[k]] = float(pj[k])
    slope = np.zeros(N, np.float32)
    hw = smooth_win // 2
    for i in range(N):
        i0 = max(0, i - hw)
        i1 = min(N - 1, i + hw)
        if i1 > i0:
            slope[i] = float((j_for_i[i1] - j_for_i[i0]) / (i1 - i0))
        else:
            slope[i] = 1.0
    return slope


@njit(cache=True)
def _dtw_stochastic_realizations(query, ref, radius, K, temperature):
    # Generates K stochastic DTW alignments by adding Gumbel noise to the cost matrix.
    # Each realization produces a different warp path, capturing alignment uncertainty.
    N = len(query)
    M = len(ref)
    INF = 1e18
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    D_base = np.full((N, M), INF)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius)
        j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            D_base[i, j] = (query[i] - ref[j]) ** 2
    paths = np.zeros((K, N), np.int64)
    for k in range(K):
        D = np.full((N, M), INF)
        for i in range(N):
            j_center = int(round(i * slope))
            j_lo = max(0, j_center - radius)
            j_hi = min(M - 1, j_center + radius)
            for j in range(j_lo, j_hi + 1):
                noise = -temperature * np.log(-np.log(np.random.uniform(1e-10, 1.0)))
                cost = D_base[i, j] + noise
                if i == 0 and j == 0:
                    D[i, j] = cost
                elif i == 0:
                    prev = D[i, j - 1]
                    D[i, j] = cost + (prev if prev < INF else INF)
                elif j == 0:
                    prev = D[i - 1, j]
                    D[i, j] = cost + (prev if prev < INF else INF)
                else:
                    a = D[i - 1, j - 1]
                    b = D[i - 1, j]
                    c = D[i, j - 1]
                    mn = a if a < b else b
                    mn = mn if mn < c else c
                    D[i, j] = cost + (mn if mn < INF else INF)
        i = N - 1
        j = M - 1
        j_for_i = np.zeros(N, np.int64)
        while i > 0 or j > 0:
            j_for_i[i] = j
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                a = D[i - 1, j - 1]
                b = D[i - 1, j]
                c = D[i, j - 1]
                if a <= b and a <= c:
                    i -= 1
                    j -= 1
                elif b <= c:
                    i -= 1
                else:
                    j -= 1
        j_for_i[0] = j
        paths[k] = j_for_i
    return paths


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP):
    # ANCC particle filter: tracks TVT as hidden state evolving along measured depth.
    # Particles are weighted by GR likelihood (how well typewell GR matches observed GR).
    # Returns weighted mean TVT estimate and uncertainty (std) at each survey point.
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    # Z-aware particle filter: extends ANCC filter by modeling TVT velocity as a function of dZ/dMD.
    # Uses both raw and smoothed GR for likelihood, plus a velocity prior from the known zone.
    # Returns weighted mean TVT estimate and uncertainty (std) at each survey point.
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


# JIT warmup
_md = np.linspace(1, 50, 20, np.float64)
_z = np.zeros(20, np.float64)
_gr = np.full(20, 50.0, np.float64)
_gg = np.linspace(45, 55, 100, np.float64)
_pf_ancc(_md, _z, _gr, _gg, 45.0, 0.1, 20.0, 50.0, 0.0, 8,
         0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
_pf_z(_md, _z, _gr, _gr, _gg, _gg, 45.0, 0.1, 20.0, 50.0, 0.0,
      -1.0, 0.0, 0.1, 8, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15.0, 100.0)
_q = np.random.randn(40)
_r = np.random.randn(50)
_dtw_sakoe_chiba(_q, _r, 10)
_dtw_stochastic_realizations(_q, _r, 10, 3, 2.0)
_dbg("JIT warmup done")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Physics-Based Method Wrappers & Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return float(PF_GR_SIG_DEF)
    return float(np.clip(
        np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
        PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s
            ).to_numpy(np.float32)


def robust_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def wls_b_well(ktvt, kz, form_col, decay=0.02):
    n = len(ktvt)
    if n < 3:
        return float(np.median(ktvt + kz - form_col))
    w = np.exp(decay * np.arange(n, dtype=np.float64))
    w /= w.sum()
    return float(np.dot(w, ktvt + kz - form_col))


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def _pf_ancc_multi(hw, tw_tvt, tw_gr, N=ANCC_N, n_seeds=PF_NUM_SEEDS, base_seed=42):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts_stack = []
    std_stack = []
    for s in range(n_seeds):
        np.random.seed(base_seed + s)
        pts, std = _pf_ancc(
            ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
            ev["GR"].values.astype(np.float64), gg, gmin, gst,
            gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN,
            ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
        pts_stack.append(pts.astype(np.float32))
        std_stack.append(std.astype(np.float32))
    return np.mean(pts_stack, 0).astype(np.float32), np.mean(std_stack, 0).astype(np.float32)


def _pf_z_multi(hw, tw_tvt, tw_gr, N=PF_N, n_seeds=PF_NUM_SEEDS, base_seed=42):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    # O(h²) derivatives via np.gradient
    md_kn = kna["MD"].values.astype(np.float64)
    z_kn_arr = kna["Z"].values.astype(np.float64)
    tvt_kn_arr = kna["TVT_input"].values.astype(np.float64)
    if len(md_kn) >= 10:
        vz = np.gradient(z_kn_arr, md_kn)
        vt = np.gradient(tvt_kn_arr, md_kn)
        m2 = np.isfinite(vz) & np.isfinite(vt)
        if m2.sum() >= 10:
            A = np.column_stack([vz[m2], np.ones(m2.sum())])
            c, _, _, _ = np.linalg.lstsq(A, vt[m2], rcond=None)
            beta = float(c[0])
            icpt = float(c[1])
            zsig = max(float(np.std(vt[m2] - (c[0] * vz[m2] + c[1]))), 0.001)
        else:
            beta, icpt, zsig = -1.0, 0.0, 0.1
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    t2 = kna.tail(20)
    if len(t2) >= 5:
        iv_grad = np.gradient(t2["TVT_input"].values.astype(np.float64),
                              t2["MD"].values.astype(np.float64))
        iv = float(np.median(iv_grad[np.isfinite(iv_grad)])) if np.any(np.isfinite(iv_grad)) else 0.0
    else:
        iv = 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts_stack = []
    std_stack = []
    for s in range(n_seeds):
        np.random.seed(base_seed + 1000 + s)
        pts, std = _pf_z(
            ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
            ev["GR"].values.astype(np.float64),
            gr_sm.loc[ev.index].values.astype(np.float64),
            gg, gs2, gmin, gst, gs, float(kna["TVT_input"].iloc[-1]), iv,
            beta, icpt, zsig, N,
            PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
        pts_stack.append(pts.astype(np.float32))
        std_stack.append(std.astype(np.float32))
    return np.mean(pts_stack, 0).astype(np.float32), np.mean(std_stack, 0).astype(np.float32)


def run_dtw_multiscale(query_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII):
    N = len(query_gr)
    qn = (query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)
    rn = (tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)
    qn_f = qn.astype(np.float64)
    rn_f = rn.astype(np.float64)
    dtw_tvts = {}
    dtw_slopes = {}
    dtw_costs = {}
    inv_cost_sum = 0.0
    tvt_stack = []
    for r in radii:
        D, pi, pj = _dtw_sakoe_chiba(qn_f, rn_f, r)
        cost = float(D[len(qn_f) - 1, len(rn_f) - 1]) / max(len(qn_f) + len(rn_f), 1)
        tvt_pred = _dtw_path_to_tvt(pi[::-1], pj[::-1], tw_tvt.astype(np.float32), N)
        slope = _dtw_path_slope(pi[::-1], pj[::-1], N)
        dtw_tvts[r] = tvt_pred
        dtw_slopes[r] = slope
        dtw_costs[r] = cost
        ic = 1.0 / (cost + 1e-6)
        inv_cost_sum += ic
        tvt_stack.append((tvt_pred, ic))
    weights = np.array([ic / inv_cost_sum for _, ic in tvt_stack], dtype=np.float32)
    tvts_mat = np.stack([t for t, _ in tvt_stack], axis=1)
    dtw_ens = (tvts_mat * weights[None, :]).sum(axis=1).astype(np.float32)
    return dtw_tvts, dtw_slopes, dtw_costs, dtw_ens


def run_dtw_stochastic(query_gr, tw_tvt, tw_gr, last_tvt,
                       radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP):
    N = len(query_gr)
    qn = ((query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)).astype(np.float64)
    rn = ((tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)).astype(np.float64)
    paths = _dtw_stochastic_realizations(qn, rn, radius, K, temperature)
    tvt_realiz = np.empty((K, N), dtype=np.float32)
    for k in range(K):
        for i in range(N):
            tvt_realiz[k, i] = tw_tvt[paths[k, i]]
    mean_tvt = tvt_realiz.mean(axis=0).astype(np.float32)
    std_tvt = tvt_realiz.std(axis=0).astype(np.float32)
    cv_tvt = (std_tvt / (np.abs(mean_tvt) + 1e-6)).astype(np.float32)
    return mean_tvt, std_tvt, cv_tvt


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.0 * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


def _build_gr_rolls(gr_vals, ev_iloc):
    s = pd.Series(gr_vals.astype(np.float64))
    out = {}
    for w in (5, 21, 51, 101):
        rm = s.rolling(w, center=True, min_periods=1)
        mean_arr = rm.mean().to_numpy(np.float32)
        std_arr = rm.std().fillna(0.0).to_numpy(np.float32)
        out[f"grm{w}"] = mean_arr[ev_iloc]
        out[f"grs{w}"] = std_arr[ev_iloc]
    for lag in (1, 5, 15, 30):
        out[f"glag{lag}"] = s.shift(lag).bfill().to_numpy(np.float32)[ev_iloc]
        out[f"glead{lag}"] = s.shift(-lag).ffill().to_numpy(np.float32)[ev_iloc]
    diff1 = s.diff().fillna(0.0).to_numpy(np.float32)
    diff2 = s.diff().diff().fillna(0.0).to_numpy(np.float32)
    out["gr_d1"] = diff1[ev_iloc]
    out["gr_d2"] = diff2[ev_iloc]
    return out


def gr_envelope(gr, w=21):
    return pd.Series(gr).rolling(w, center=True, min_periods=1).max().to_numpy(np.float32)


def gr_energy(gr, w=21):
    sq = gr.astype(np.float64) ** 2
    return np.sqrt(
        pd.Series(sq).rolling(w, center=True, min_periods=1).mean().to_numpy().clip(0)
    ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Spatial Imputers
# ═══════════════════════════════════════════════════════════════════════════════

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        fn = self.fa[ik]
        wx = w * xn
        wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1),
                        (wy[:, :, None] * fn).sum(1),
                        (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]
                ).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values)
            ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)
        self._mean = float(self.ancc.mean())

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, self._mean, ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feature Builder
# ═══════════════════════════════════════════════════════════════════════════════

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
DTW_OFFS = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20], np.float32)

_FI: Optional[FormationPlaneKNN] = None
_DI: Optional[DenseANCCImputer] = None


class _WellGRCache:
    __slots__ = ("gr_arr", "roll_feats_full", "gr_env_full", "gr_nrg_full")

    def __init__(self, hw, gr_mean):
        gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(gr_mean)
        self.gr_arr = gr_full.to_numpy(np.float32)
        all_idx = np.arange(len(hw), dtype=np.int64)
        self.roll_feats_full = _build_gr_rolls(self.gr_arr, all_idx)
        self.gr_env_full = gr_envelope(self.gr_arr)
        self.gr_nrg_full = gr_energy(self.gr_arr)


def _build_well_from_df(hw, tw, is_train, wid, gr_cache=None):
    global _FI, _DI
    if _FI is None or _DI is None:
        return None
    if is_train and "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and hw["TVT"].isna().all():
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None
    try:
        lk = kn.iloc[-1]
        last_tvt = float(lk["TVT_input"])
        gr_mean = float(np.nanmean(tw_gr))

        # ── GR arrays ──
        if gr_cache is not None:
            gr_arr = gr_cache.gr_arr
        else:
            gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(gr_mean)
            gr_arr = gr_full.to_numpy(np.float32)
        ev_idx_arr = np.array([hw.index.get_loc(i) for i in ev.index], dtype=np.int64)
        hgr = gr_arr[ev_idx_arr]
        kgr = gr_arr[:len(kn)]

        # ── GR rolling features ──
        if gr_cache is not None:
            roll_feats = {k: v[ev_idx_arr] for k, v in gr_cache.roll_feats_full.items()}
            hgr_env = gr_cache.gr_env_full[ev_idx_arr]
            hgr_nrg = gr_cache.gr_nrg_full[ev_idx_arr]
        else:
            roll_feats = _build_gr_rolls(gr_arr, ev_idx_arr)
            hgr_env = gr_envelope(gr_arr)[ev_idx_arr]
            hgr_nrg = gr_energy(gr_arr)[ev_idx_arr]
        gr_d1 = roll_feats.pop("gr_d1")
        gr_d2 = roll_feats.pop("gr_d2")

        # ── Particle filters (multi-seed) ──
        pf_a, std_a = _pf_ancc_multi(hw, tw_tvt, tw_gr, n_seeds=PF_NUM_SEEDS)
        if len(pf_a) == 0:
            return None
        pf_z_a, std_z = _pf_z_multi(hw, tw_tvt, tw_gr, n_seeds=PF_NUM_SEEDS)
        pf_use = pf_a.astype(np.float32)
        std_use = std_a.astype(np.float32)
        has_z = len(pf_z_a) == len(pf_a) and not np.any(np.isnan(pf_z_a))

        # ── Beam search (7 configs) ──
        eval_start_iloc = int(ev_idx_arr[0])
        gr_filled_series = pd.Series(gr_arr)
        hgr_beam = gr_filled_series.iloc[eval_start_iloc:].to_numpy(np.float32)
        bpaths = {}
        for (bs, mc, es, r, tag) in BEAMS:
            bpaths[tag] = beam_search(hgr_beam, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)[:len(ev)]
        beam_vals = np.stack(list(bpaths.values()), axis=1)
        beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

        # ── Multi-scale NCC ──
        ktvt = kn["TVT_input"].to_numpy(np.float32)
        sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
        sc8, sc8s = sc_res[0]
        sc15, sc15s = sc_res[1]
        sc25, sc25s = sc_res[2]
        sc_cons = (sc8 + sc15 + sc25) / 3.0
        sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
        hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

        # ── DTW (deterministic + stochastic) ──
        full_gr = gr_arr.astype(np.float32)
        dtw_tvts_ms, dtw_slopes_ms, dtw_costs_ms, dtw_ens_ms = run_dtw_multiscale(
            full_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII)
        dtw_mean_stoch, dtw_std_stoch, dtw_cv_stoch = run_dtw_stochastic(
            full_gr, tw_tvt, tw_gr, last_tvt, radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP)
        nh = len(ev)
        ev_start = ev.index[0]

        def _ev_slice(arr):
            return arr[ev_start:ev_start + nh].astype(np.float32)

        dtw_ens_ev = _ev_slice(dtw_ens_ms)
        dtw_mean_ev = _ev_slice(dtw_mean_stoch)
        dtw_std_ev = _ev_slice(dtw_std_stoch)
        dtw_cv_ev = _ev_slice(dtw_cv_stoch)
        dtw_per_radius_ev = {}
        dtw_slope_ev = {}
        for r in DTW_RADII:
            dtw_per_radius_ev[r] = _ev_slice(dtw_tvts_ms[r])
            dtw_slope_ev[r] = _ev_slice(dtw_slopes_ms[r])
        dtw_slope_mean_ev = np.stack([dtw_slope_ev[r] for r in DTW_RADII], 1).mean(1).astype(np.float32)
        dtw_cost_arr = np.array([dtw_costs_ms[r] for r in DTW_RADII], dtype=np.float32)
        dtw_cost_min = float(dtw_cost_arr.min())
        dtw_cost_range = float(dtw_cost_arr.max() - dtw_cost_arr.min())

        # ── Calibration ──
        tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
        a_cal, b_cal = affine_cal(kgr, tw_at_k)
        kmd = kn["MD"].to_numpy(np.float32)
        kz = kn["Z"].to_numpy(np.float32)
        pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
        slp_all = robust_slope(kmd, ktvt)
        slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
        slp_z = robust_slope(kz, ktvt)

        # ── Spatial imputers ──
        swid = wid if is_train else None
        xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
        xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
        form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
        form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
        z_kn = kn["Z"].to_numpy(np.float32)
        z_ev = ev["Z"].to_numpy(np.float32)

        # Per-formation features (combined from both codebases)
        tvt_fs = {}
        form_rmse = {}
        form_list = []
        for fi2, fn in enumerate(FORMATIONS):
            b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
            tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
            tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
            tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
            tvt_fs[f"tvtF_{fn}"] = tvt_f
            tvt_fs[f"tvtFw_{fn}"] = tvt_fw
            tvt_fs[f"tvtF50_{fn}"] = tvt_f50
            tvt_fs[f"bw_{fn}"] = np.float32(b_full)
            tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
            tvt_fs[f"bw50_{fn}"] = np.float32(b_late)
            tvt_fs[f"bw_early_{fn}"] = np.float32(b_early)
            tvt_fs[f"bw_mid_{fn}"] = np.float32(b_mid)
            form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
            form_list.append(tvt_f)
        fs = np.stack(form_list, 1)
        form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
        form_std_d = fs.std(1).astype(np.float32)
        form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

        # Dense ANCC
        d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid)
        d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
        b_vd = ktvt + z_kn - d_kn
        _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
        b_d = float(np.median(b_vd))
        tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
        tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
        tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
        res_kn = ktvt + z_kn - d_kn
        d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
        d_bias = float(np.mean(res_kn))
        d_nb_std = float(np.mean(d_std_kn))

        # Inter-signal consensus
        all_sigs = ([pf_use] + list(bpaths.values()) +
                    [sc8, sc15, sc25, sc_ens, tvt_fs["tvtF_ANCC"], tvt_dense,
                     dtw_ens_ev])
        sig_mat = np.stack(all_sigs, 1)
        sig_std = sig_mat.std(1).astype(np.float32)
        sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

        # ── Trajectory derivatives (O(h²) via np.gradient) ──
        md_vals = hw["MD"].values.astype(np.float64)
        md_safe = np.maximum.accumulate(md_vals) + 1e-10 * np.arange(len(md_vals))
        dzdmd_full = np.gradient(hw["Z"].values.astype(np.float64), md_safe).astype(np.float32)
        dxdmd_full = np.gradient(hw["X"].values.astype(np.float64), md_safe).astype(np.float32)
        dydmd_full = np.gradient(hw["Y"].values.astype(np.float64), md_safe).astype(np.float32)
        dzdmd = dzdmd_full[ev_idx_arr]
        dxdmd = dxdmd_full[ev_idx_arr]
        dydmd = dydmd_full[ev_idx_arr]

        # ── Slope baselines ──
        hmd = ev["MD"].to_numpy(np.float32)
        md_since = hmd - float(lk["MD"])
        slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
        slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)
        frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

        def sc(v):
            return np.full(nh, np.float32(v), np.float32)

        # ── Assemble features ──
        feats = {
            "well": wid,
            "id": [f"{wid}_{i}" for i in ev.index],
            "last_known_tvt": sc(last_tvt),
            # Particle filter features
            "pf_ancc": pf_use, "pf_ancc_std": std_use,
            "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
            "pf_z": pf_z_a.astype(np.float32) if has_z else sc(last_tvt),
            "pf_z_delta": (pf_z_a - last_tvt).astype(np.float32) if has_z else sc(0.0),
            "pf_vs_z": (pf_use - pf_z_a.astype(np.float32)) if has_z else sc(0.0),
            "pf_std_trend": (std_use - std_use[0]).astype(np.float32) if len(std_use) > 0 else sc(0.0),
            # Beam search features (7 configs)
            **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
            "beam_mean_d": (beam_vals - last_tvt).mean(1).astype(np.float32),
            "beam_std_d": (beam_vals - last_tvt).std(1).astype(np.float32),
            "beam_med_d": np.median(beam_vals - last_tvt, axis=1).astype(np.float32),
            # NCC features
            "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32), "sc8_sc": sc8s,
            "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32), "sc15_sc": sc15s,
            "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32), "sc25_sc": sc25s,
            "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
            "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
            "sc_trust": sc(sc_trust),
            "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
            # DTW features (from v41)
            "dtw_ens_d": (dtw_ens_ev - last_tvt).astype(np.float32),
            "dtw_stoch_mean_d": (dtw_mean_ev - last_tvt).astype(np.float32),
            "dtw_stoch_std": dtw_std_ev,
            "dtw_stoch_cv": dtw_cv_ev,
            "dtw_slope_mean": dtw_slope_mean_ev,
            **{f"dtw_r{r}_d": (dtw_per_radius_ev[r] - last_tvt).astype(np.float32) for r in DTW_RADII},
            **{f"dtw_slope_r{r}": dtw_slope_ev[r] for r in DTW_RADII},
            "dtw_cost_min": sc(dtw_cost_min),
            "dtw_cost_range": sc(dtw_cost_range),
            "dtw_vs_beam": (dtw_ens_ev - bpaths["cons"]).astype(np.float32),
            "dtw_vs_pf": (dtw_ens_ev - pf_use).astype(np.float32),
            "dtw_vs_sc": (dtw_ens_ev - sc_ens).astype(np.float32),
            **{f"tddtw{int(o)}": hgr - np.interp(dtw_ens_ev + o, tw_tvt, tw_gr).astype(np.float32)
               for o in DTW_OFFS},
            # Signal consensus
            "sig_std": sig_std, "sig_mean_d": sig_mean,
            # Formation features
            **tvt_fs,
            **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
            "form_mean_d": form_mean_d, "form_std_d": form_std_d, "form_rng_d": form_rng_d,
            "spatial_ancc_d": (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
            "spatial_knn_dist": knn_d,
            # Dense ANCC features
            "dense_ancc": d_ancc, "dense_std": d_std, "dense_dist": d_dist,
            "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
            "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
            "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
            "dense_rmse": sc(d_rmse), "dense_bias": sc(d_bias), "dense_nb_std": sc(d_nb_std),
            # Cross-signal features
            "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
            "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
            "spatial_vs_dense": (tvt_fs["tvtF_ANCC"] - tvt_dense).astype(np.float32),
            "beam_vs_spatial": (bpaths["cons"] - tvt_fs["tvtF_ANCC"]).astype(np.float32),
            "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
            # Calibration & prefix stats
            "cal_a": sc(a_cal), "cal_b": sc(b_cal),
            "pfx_rmse": sc(pfx_rmse), "known_len": sc(len(kn)), "eval_len": sc(nh),
            "slp_all": sc(slp_all), "slp_50": sc(slp_50), "slp_z": sc(slp_z),
            "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
            "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
            "ktvt_range": sc(float(np.ptp(ktvt))), "ktvt_std": sc(float(ktvt.std())),
            # Positional features
            "md_since": md_since, "frac": frac, "frac2": frac ** 2, "sqrt_frac": np.sqrt(frac),
            "z": z_ev,
            "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
            "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
            "dz": (z_ev - float(lk["Z"])).astype(np.float32),
            "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2
                           ).to_numpy(np.float32),
            "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,
            # GR features
            "gr": hgr, "gr_d1": gr_d1, "gr_d2": gr_d2,
            "gr_env": hgr_env.astype(np.float32), "gr_nrg": hgr_nrg.astype(np.float32),
            "gr_vs_tw_anc": hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
            "gr_vs_slp_all": hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
            # GR residual offset features
            **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr))
               for o in ANCH_OFFS},
            **{f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32)
               for o in BEAM_OFFS},
            **{f"tdsc{int(o)}": hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32)
               for o in SC_OFFS},
            **{f"tdpf{int(o)}": hgr - np.float32(np.interp(pf_use + o, tw_tvt, tw_gr))
               for o in PF_OFFS},
            # Typewell stats
            "tw_range": sc(float(np.ptp(tw_tvt))), "tw_gr_mean": sc(float(tw_gr.mean())),
        }
        for k, v in roll_feats.items():
            feats[k] = v

        result = pd.DataFrame(feats)
        if is_train:
            if "TVT" not in ev.columns or ev["TVT"].isna().all():
                return None
            result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))
        return result
    except Exception as exc:
        print(f"[ERROR] _build_well_from_df({wid}): {exc}", file=sys.stderr, flush=True)
        return None


def build_well(hw_path, tw_path, is_train):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    return _build_well_from_df(hw, tw, is_train=is_train, wid=wid)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cal-Zone Augmentation (from v26 / 9.579)
# ═══════════════════════════════════════════════════════════════════════════════

def build_augmented(hw, tw_tvt, tw_gr, wid, gr_cache=None,
                    n_splits=N_AUG_SPLITS, min_known=MIN_KNOWN_FOR_AUG):
    known_all = hw[hw["TVT_input"].notna()]
    n_cal = len(known_all)
    if n_cal < min_known + 5:
        return pd.DataFrame()
    split_ks = np.unique(np.linspace(min_known, n_cal - 2, n_splits).astype(int))
    tw_mock = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    parts = []
    for k in split_ks:
        hw_m = hw.copy()
        mask_start = known_all.index[k]
        hw_m.loc[mask_start:, "TVT_input"] = np.nan
        kn_m = hw_m[hw_m["TVT_input"].notna()]
        n_new = n_cal - k
        if len(hw_m[hw_m["TVT_input"].isna()]) == 0 or len(kn_m) < 10 or n_new == 0:
            continue
        try:
            feat = _build_well_from_df(hw_m, tw_mock, is_train=True, wid=wid, gr_cache=gr_cache)
        except Exception:
            continue
        if feat is None or len(feat) == 0:
            continue
        feat_new = feat.iloc[:n_new].copy()
        feat_new["aug_k"] = np.int32(k)
        parts.append(feat_new)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def process_train_well(hw_path):
    """Builds original + augmented features for one training well."""
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    tw_path = TRAIN_DIR / f"{wid}__typewell.csv"
    if not tw_path.exists():
        return pd.DataFrame()
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(np.float32)
        tw_gr = tw["GR"].to_numpy(np.float32)
        gr_mean = float(np.nanmean(tw_gr))
        gr_cache = _WellGRCache(hw, gr_mean)
        feat_orig = _build_well_from_df(hw, tw, is_train=True, wid=wid, gr_cache=gr_cache)
        parts = []
        if feat_orig is not None and len(feat_orig) > 0:
            feat_orig["aug_k"] = np.int32(-1)
            parts.append(feat_orig)
        feat_aug = build_augmented(hw, tw_tvt, tw_gr, wid, gr_cache=gr_cache)
        if len(feat_aug) > 0:
            parts.append(feat_aug)
        if not parts:
            return pd.DataFrame()
        r = pd.concat(parts, ignore_index=True)
        r["well_id"] = wid
        return r
    except Exception as exc:
        print(f"[ERROR] process_train_well({wid}): {exc}", file=sys.stderr, flush=True)
        return pd.DataFrame()


def process_test_train(hw_path):
    """Online training: augment from test well calibration zone."""
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    tw_path = TEST_DIR / f"{wid}__typewell.csv"
    if not tw_path.exists():
        return pd.DataFrame()
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)
        if "TVT" not in tw.columns or "GR" not in tw.columns:
            return pd.DataFrame()
        known = hw[hw["TVT_input"].notna()]
        if len(known) < MIN_KNOWN_FOR_AUG + 5:
            return pd.DataFrame()
        hw_aug = hw.copy()
        hw_aug["TVT"] = hw_aug["TVT_input"]
        tw_tvt = tw["TVT"].to_numpy(np.float32)
        tw_gr = tw["GR"].to_numpy(np.float32)
        gr_mean = float(np.nanmean(tw_gr))
        gr_cache = _WellGRCache(hw_aug, gr_mean)
        feat_aug = build_augmented(hw_aug, tw_tvt, tw_gr, wid, gr_cache=gr_cache)
        if len(feat_aug) > 0:
            feat_aug["well_id"] = wid
        return feat_aug
    except Exception:
        return pd.DataFrame()


def build_dataset(paths, is_train, label):
    args = [(str(p),
             str(p.parent / f"{p.stem.replace('__horizontal_well', '')}__typewell.csv"),
             is_train)
            for p in paths
            if (p.parent / f"{p.stem.replace('__horizontal_well', '')}__typewell.csv").exists()]
    _dbg(f"build_dataset[{label}]: {len(args)} wells, NCPU={NCPU}")
    res = Parallel(n_jobs=NCPU, prefer="threads", verbose=0)(
        delayed(build_well)(hp, tp, it) for hp, tp, it in args)
    parts = [r for r in res if r is not None and len(r) > 0]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _dbg(f"build_dataset[{label}]: {len(out):,d} rows, {len(out.columns)} cols")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Build Train/Test Feature Matrices
# ═══════════════════════════════════════════════════════════════════════════════

_dbg("Building FormationPlaneKNN + DenseANCCImputer")
hw_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
FI = FormationPlaneKNN(train_wids, TRAIN_DIR)
DI = DenseANCCImputer(train_wids, TRAIN_DIR)
_FI = FI
_DI = DI
_dbg(f"FI rows: {len(FI.df)}; DI rows: {len(DI.ancc):,d}")

# ── Build train features with augmentation ──
_dbg("Building train features (original + augmented)")
t0 = time.time()
_results = Parallel(n_jobs=NCPU, prefer="threads", verbose=0)(
    delayed(process_train_well)(p) for p in hw_paths)
train_parts = [r for r in _results if r is not None and len(r) > 0]
train_core_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
del _results, train_parts
gc.collect()
_dbg(f"Train core: {train_core_df.shape} ({time.time() - t0:.0f}s)")

# ── Online training on test wells ──
test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
_dbg(f"Online training on {len(test_paths)} test wells")
t1 = time.time()
_res_t = Parallel(n_jobs=NCPU, prefer="threads", verbose=0)(
    delayed(process_test_train)(p) for p in test_paths)
test_aug_parts = [r for r in _res_t if r is not None and len(r) > 0]
if test_aug_parts:
    train_core_df = pd.concat([train_core_df] + test_aug_parts, ignore_index=True)
del _res_t, test_aug_parts
gc.collect()
_dbg(f"Train with online aug: {train_core_df.shape} ({time.time() - t1:.0f}s)")

orig = (train_core_df["aug_k"] == -1).sum() if "aug_k" in train_core_df.columns else len(train_core_df)
aug = (train_core_df["aug_k"] >= 0).sum() if "aug_k" in train_core_df.columns else 0
_dbg(f"Original: {orig:,}  Augmented: {aug:,} (+{aug / max(orig, 1) * 100:.0f}%)")

train_df = train_core_df
del train_core_df

# ── Build test features ──
_dbg("Building test features")
test_df = build_dataset(test_paths, is_train=False, label="test")

SKIP = {"well", "well_id", "id", "target", "aug_k"}
features = [c for c in train_df.columns if c not in SKIP]
features = [c for c in features if c in test_df.columns]
X = train_df[features]
y = train_df["target"]
g = train_df["well"]
X_test = test_df[features]
_dbg(f"Features: {len(features)} | train rows: {len(train_df):,d} | test rows: {len(test_df):,d}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CV Splits (augmentation-aware) + Bucket Assignment
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_group_kfold(groups, strat_keys, n_splits=5, seed=42):
    df = pd.DataFrame({"g": groups, "s": strat_keys})
    well_strat = df.groupby("g")["s"].mean().reset_index()
    well_strat = well_strat.sort_values("s").reset_index(drop=True)
    rng = np.random.RandomState(seed)
    n_w = len(well_strat)
    well_to_fold = {}
    for chunk_start in range(0, n_w, n_splits):
        chunk = well_strat.iloc[chunk_start:chunk_start + n_splits]
        perm = rng.permutation(len(chunk))
        for i, idx in enumerate(perm):
            well_to_fold[chunk["g"].iloc[idx]] = i % n_splits
    fold_of_row = df["g"].map(well_to_fold).values
    splits = []
    for k in range(n_splits):
        va = np.where(fold_of_row == k)[0]
        tr = np.where(fold_of_row != k)[0]
        splits.append((tr, va))
    return splits


# Augmentation-aware: augmented rows train-only, original rows in validation
_is_orig = (train_df["aug_k"].values == -1) if "aug_k" in train_df.columns else np.ones(len(train_df), bool)
strat_keys = train_df["pfx_rmse"].values
_unique_wells = np.unique(g.values)
_rng_cv = np.random.RandomState(SEED)
_shuffled = _rng_cv.permutation(_unique_wells)
_fold_map = {w: i % N_SPLITS for i, w in enumerate(_shuffled)}
_fold_ids = np.array([_fold_map[w] for w in g.values])

splits = []
for _f in range(N_SPLITS):
    _tr_idx = np.where(_fold_ids != _f)[0]
    _va_all = np.where(_fold_ids == _f)[0]
    _va_idx = _va_all[_is_orig[_va_all]]
    if len(_va_idx) > 0:
        splits.append((_tr_idx, _va_idx))
_dbg(f"CV: {len(splits)} folds, train ~{np.mean([len(t) for t, _ in splits]):.0f}, "
     f"val ~{np.mean([len(v) for _, v in splits]):.0f}")

# Bucket assignment by pfx_rmse median
well_pfx = train_df.groupby("well")["pfx_rmse"].first()
PFX_THRESHOLD = float(well_pfx.median())
train_df["bucket"] = np.where(train_df["pfx_rmse"].values <= PFX_THRESHOLD, "easy", "hard")
test_df["bucket"] = np.where(test_df["pfx_rmse"].values <= PFX_THRESHOLD, "easy", "hard")
train_easy_mask = (train_df["bucket"] == "easy").values
train_hard_mask = ~train_easy_mask
test_easy_mask = (test_df["bucket"] == "easy").values
test_hard_mask = ~test_easy_mask
_dbg(f"Bucket threshold (pfx_rmse median): {PFX_THRESHOLD:.4f}")
_dbg(f"  train: {train_easy_mask.sum():,d} easy, {train_hard_mask.sum():,d} hard")
_dbg(f"  test:  {test_easy_mask.sum():,d} easy, {test_hard_mask.sum():,d} hard")


def per_bucket_splits(mask):
    sub_idx = np.where(mask)[0]
    sub_groups = train_df.iloc[sub_idx]["well"].values
    sub_strat = train_df.iloc[sub_idx]["pfx_rmse"].values
    sub_is_orig = _is_orig[sub_idx]
    bucket_splits = stratified_group_kfold(sub_groups, sub_strat, n_splits=N_SPLITS, seed=SEED)
    global_splits = []
    for tr_sub, va_sub in bucket_splits:
        va_orig = va_sub[sub_is_orig[va_sub]]
        if len(va_orig) > 0:
            global_splits.append((sub_idx[tr_sub], sub_idx[va_orig]))
    return global_splits


splits_easy = per_bucket_splits(train_easy_mask)
splits_hard = per_bucket_splits(train_hard_mask)
_dbg(f"Easy bucket folds: {[len(va) for _, va in splits_easy]}")
_dbg(f"Hard bucket folds: {[len(va) for _, va in splits_hard]}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Model Training (LGB × 3 + CB × 3 + XGB × 3, per bucket)
# ═══════════════════════════════════════════════════════════════════════════════

lgb_params_base = dict(
    boosting_type="gbdt", num_leaves=255, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=3.5, reg_alpha=0.05,
    objective="regression", verbose=-1, n_jobs=-1,
    device_type="gpu" if GPU_COUNT > 0 else "cpu",
    gpu_use_dp=False, max_bin=255,
)
lgb_params = [
    dict(learning_rate=0.025, n_estimators=LGB_N_EST, seed=42,
         gpu_device_id=0 % max(GPU_COUNT, 1), **lgb_params_base),
    dict(learning_rate=0.020, n_estimators=LGB_N_EST, seed=7,
         gpu_device_id=1 % max(GPU_COUNT, 1), **lgb_params_base),
    dict(learning_rate=0.030, n_estimators=LGB_N_EST, seed=123,
         gpu_device_id=0 % max(GPU_COUNT, 1), **lgb_params_base),
]

cb_params_base = dict(
    iterations=CB_ITERS, depth=7, l2_leaf_reg=3.0, min_data_in_leaf=20,
    border_count=254, loss_function="RMSE",
    task_type="GPU" if GPU_COUNT > 0 else "CPU",
    devices=CATBOOST_DEVICES, od_type="Iter", od_wait=400, verbose=0,
)
cb_params = [
    dict(learning_rate=0.025, random_seed=42, **cb_params_base),
    dict(learning_rate=0.020, random_seed=7, **cb_params_base),
    dict(learning_rate=0.030, random_seed=123, **cb_params_base),
]

xgb_params_base = dict(
    n_estimators=XGB_N_EST, max_depth=8, min_child_weight=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05,
    objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist",
    device="cuda" if GPU_COUNT > 0 else "cpu",
    early_stopping_rounds=400, verbosity=0,
)
xgb_params = [
    dict(learning_rate=0.025, random_state=42, **xgb_params_base),
    dict(learning_rate=0.020, random_state=7, **xgb_params_base),
    dict(learning_rate=0.030, random_state=123, **xgb_params_base),
]

oof_preds = {}
test_preds = {}
overall_scores = {}


def train_lightgbm(params, name, bucket_splits, bucket_test_mask):
    params = dict(params)
    n_est = params.pop("n_estimators")
    oof = np.full(len(train_df), np.nan, np.float32)
    tst = np.zeros(len(test_df), np.float32)
    fold_sc = []
    Xt_bucket = X_test.iloc[bucket_test_mask] if bucket_test_mask.any() else None
    for fi, (tr_idx, va_idx) in enumerate(bucket_splits):
        d_tr = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        d_va = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], reference=d_tr)
        m = lgb.train(params, d_tr, valid_sets=[d_va], num_boost_round=n_est,
                      callbacks=[lgb.early_stopping(400, verbose=False)])
        oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration).astype(np.float32)
        if Xt_bucket is not None and len(Xt_bucket) > 0:
            tst[bucket_test_mask] += m.predict(Xt_bucket, num_iteration=m.best_iteration
                                               ).astype(np.float32) / len(bucket_splits)
        sc = root_mean_squared_error(y.iloc[va_idx], oof[va_idx])
        fold_sc.append(sc)
        _dbg(f"  LGB[{name}] fold {fi}: RMSE={sc:.4f} (iter={m.best_iteration})")
    filled = ~np.isnan(oof)
    ov = root_mean_squared_error(y[filled], oof[filled])
    _dbg(f"LGB[{name}] overall: {ov:.4f}")
    return oof, tst, ov


def train_catboost(params, name, bucket_splits, bucket_test_mask):
    params = dict(params)
    oof = np.full(len(train_df), np.nan, np.float32)
    tst = np.zeros(len(test_df), np.float32)
    fold_sc = []
    Xt_bucket = X_test.iloc[bucket_test_mask].values if bucket_test_mask.any() else None
    for fi, (tr_idx, va_idx) in enumerate(bucket_splits):
        m = CatBoostRegressor(**params)
        m.fit(Pool(X.iloc[tr_idx].values, label=y.iloc[tr_idx].values),
              eval_set=Pool(X.iloc[va_idx].values, label=y.iloc[va_idx].values),
              use_best_model=True)
        oof[va_idx] = m.predict(X.iloc[va_idx]).astype(np.float32)
        if Xt_bucket is not None and len(Xt_bucket) > 0:
            tst[bucket_test_mask] += m.predict(Xt_bucket).astype(np.float32) / len(bucket_splits)
        sc = root_mean_squared_error(y.iloc[va_idx], oof[va_idx])
        fold_sc.append(sc)
        _dbg(f"  CB[{name}] fold {fi}: RMSE={sc:.4f} (iter={m.tree_count_})")
    filled = ~np.isnan(oof)
    ov = root_mean_squared_error(y[filled], oof[filled])
    _dbg(f"CB[{name}] overall: {ov:.4f}")
    return oof, tst, ov


def train_xgboost(params, name, bucket_splits, bucket_test_mask):
    params = dict(params)
    oof = np.full(len(train_df), np.nan, np.float32)
    tst = np.zeros(len(test_df), np.float32)
    fold_sc = []
    Xt_bucket = X_test.iloc[bucket_test_mask] if bucket_test_mask.any() else None
    for fi, (tr_idx, va_idx) in enumerate(bucket_splits):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
        oof[va_idx] = m.predict(X.iloc[va_idx]).astype(np.float32)
        if Xt_bucket is not None and len(Xt_bucket) > 0:
            tst[bucket_test_mask] += m.predict(Xt_bucket).astype(np.float32) / len(bucket_splits)
        sc = root_mean_squared_error(y.iloc[va_idx], oof[va_idx])
        fold_sc.append(sc)
        _dbg(f"  XGB[{name}] fold {fi}: RMSE={sc:.4f} (iter={m.best_iteration})")
    filled = ~np.isnan(oof)
    ov = root_mean_squared_error(y[filled], oof[filled])
    _dbg(f"XGB[{name}] overall: {ov:.4f}")
    return oof, tst, ov


# Train per-bucket
for bucket_name, b_splits, b_test_mask in [
    ("easy", splits_easy, test_easy_mask),
    ("hard", splits_hard, test_hard_mask),
    ]:
    _dbg(f"{'=' * 20} TRAINING BUCKET: {bucket_name} {'=' * 20}")
    for i in range(len(lgb_params)):
        n = f"lgb-{i + 1}-{bucket_name}"
        oof_, tst_, ov_ = train_lightgbm(lgb_params[i], n, b_splits, b_test_mask)
        oof_preds[n] = oof_
        test_preds[n] = tst_
        overall_scores[n] = ov_
    for i in range(len(cb_params)):
        n = f"cb-{i + 1}-{bucket_name}"
        oof_, tst_, ov_ = train_catboost(cb_params[i], n, b_splits, b_test_mask)
        oof_preds[n] = oof_
        test_preds[n] = tst_
        overall_scores[n] = ov_
    for i in range(len(xgb_params)):
        n = f"xgb-{i + 1}-{bucket_name}"
        oof_, tst_, ov_ = train_xgboost(xgb_params[i], n, b_splits, b_test_mask)
        oof_preds[n] = oof_
        test_preds[n] = tst_
        overall_scores[n] = ov_

# ── Save checkpoint after training ──
import pickle

_ckpt_path = OUTPUT_DIR / "training_checkpoint.pkl"
_ckpt = {
    "oof_preds": oof_preds,
    "test_preds": test_preds,
    "overall_scores": overall_scores,
    "train_easy_mask": train_easy_mask,
    "train_hard_mask": train_hard_mask,
    "test_easy_mask": test_easy_mask,
    "test_hard_mask": test_hard_mask,
    "_is_orig": _is_orig,
    "PFX_THRESHOLD": PFX_THRESHOLD,
    "features": features,
}
with open(_ckpt_path, "wb") as _f:
    pickle.dump(_ckpt, _f)
_dbg(f"Checkpoint saved to {_ckpt_path} ({_ckpt_path.stat().st_size / 1e6:.1f} MB)")
del _ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Per-Bucket Hill Climbing
# ═══════════════════════════════════════════════════════════════════════════════

_dbg("Per-bucket hill climbing")


def proper_hill_climb(oof_df, test_df_blend, y_arr,
                      precision=0.0005, max_iters=5000):
    cols = list(oof_df.columns)
    oof_arr = {c: oof_df[c].values.astype(np.float64) for c in cols}
    tst_arr = {c: test_df_blend[c].values.astype(np.float64) for c in cols}
    sc_single = {c: root_mean_squared_error(y_arr, oof_arr[c]) for c in cols}
    best_m = min(sc_single, key=sc_single.get)
    w = {c: 0.0 for c in cols}
    w[best_m] = 1.0
    blend = oof_arr[best_m].copy()
    total = 1.0
    best_score = sc_single[best_m]
    step_schedule = [s for s in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)
                     if s >= precision]
    iter_n = 0
    for step in step_schedule:
        no_imp = 0
        while iter_n < max_iters:
            new_total = total + step
            best_pick = None
            best_trial_score = best_score
            best_trial_blend = None
            for m in cols:
                trial_blend = (blend * total + step * oof_arr[m]) / new_total
                trial_score = root_mean_squared_error(y_arr, trial_blend)
                if trial_score < best_trial_score - 1e-7:
                    best_trial_score = trial_score
                    best_pick = m
                    best_trial_blend = trial_blend
            if best_pick is None:
                no_imp += 1
                if no_imp >= 3:
                    break
                continue
            w[best_pick] += step
            blend = best_trial_blend
            total = new_total
            best_score = best_trial_score
            iter_n += 1
            no_imp = 0
    norm_w = {k: v / total for k, v in w.items()}
    hc_oof = blend.astype(np.float32)
    hc_tst = np.zeros(len(test_df_blend), np.float64)
    for k, v in norm_w.items():
        if v > 0:
            hc_tst += v * tst_arr[k]
    return hc_oof, hc_tst.astype(np.float32), float(best_score), norm_w


hc_oof_preds = np.full(len(train_df), np.nan, np.float32)
hc_test_preds = np.zeros(len(test_df), np.float32)
bucket_hc_results = {}

for bucket_name, train_mask, test_mask in [
    ("easy", train_easy_mask, test_easy_mask),
    ("hard", train_hard_mask, test_hard_mask),
]:
    _dbg(f"--- HC bucket: {bucket_name} ---")
    bucket_models = [k for k in oof_preds if k.endswith(f"-{bucket_name}")]
    orig_mask = train_mask & _is_orig
    bucket_oof = {k: oof_preds[k][orig_mask] for k in bucket_models}
    bucket_tst = {k: test_preds[k][test_mask] for k in bucket_models}
    y_bucket = y.values[orig_mask]
    valid = np.all([np.isfinite(bucket_oof[k]) for k in bucket_models], axis=0)
    if valid.sum() == 0:
        _dbg(f"  WARNING: no valid OOF rows for {bucket_name}")
        continue
    bucket_oof = {k: bucket_oof[k][valid] for k in bucket_models}
    y_bucket = y_bucket[valid]
    oof_df_b = pd.DataFrame(bucket_oof)
    tst_df_b = pd.DataFrame(bucket_tst)
    if len(tst_df_b) == 0:
        b_hc_oof, _, b_score, b_weights = proper_hill_climb(oof_df_b, oof_df_b, y_bucket)
        orig_idx = np.where(orig_mask)[0][valid]
        hc_oof_preds[orig_idx] = b_hc_oof
        bucket_hc_results[bucket_name] = (b_score, b_weights)
        continue
    b_hc_oof, b_hc_tst, b_score, b_weights = proper_hill_climb(oof_df_b, tst_df_b, y_bucket)
    orig_idx = np.where(orig_mask)[0][valid]
    hc_oof_preds[orig_idx] = b_hc_oof
    hc_test_preds[test_mask] = b_hc_tst
    bucket_hc_results[bucket_name] = (b_score, b_weights)
    _dbg(f"  HC {bucket_name}: {b_score:.4f}")
    _w_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(b_weights.items(), key=lambda kv: -kv[1]) if v > 1e-4)
    _dbg(f"  weights: {{ {_w_str} }}")

_valid_hc = np.isfinite(hc_oof_preds)
hc_score = root_mean_squared_error(y.values[_valid_hc], hc_oof_preds[_valid_hc])
_dbg(f"GLOBAL HC: {hc_score:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Per-Bucket Post-Processing (Optuna)
# ═══════════════════════════════════════════════════════════════════════════════

_dbg("Per-bucket Optuna post-processing")
base = train_df["last_known_tvt"].values
ytrue_full = y.values + base
pf_oof = train_df["pf_ancc"].values - base

PP_PARAMS = {}


def apply_pp(df, md, pd_, alpha, tau, w_pf):
    d = md * (1.0 - w_pf) + pd_ * w_pf
    if tau:
        d *= (1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau))
    return d * alpha


for bucket_name, train_mask in [("easy", train_easy_mask), ("hard", train_hard_mask)]:
    valid = train_mask & _is_orig & np.isfinite(hc_oof_preds)
    _dbg(f"--- PP bucket: {bucket_name} ({valid.sum():,d} valid rows) ---")
    if valid.sum() < 10:
        _dbg(f"  SKIP PP for {bucket_name}: too few valid rows")
        PP_PARAMS[bucket_name] = (1.0, 100, 0.1, 999.0)
        continue
    bucket_df = train_df.loc[valid].reset_index(drop=True)
    bucket_hc = hc_oof_preds[valid]
    bucket_pf = pf_oof[valid]
    bucket_y_true = ytrue_full[valid]
    bucket_base = base[valid]

    def _pp_objective(trial, _df=bucket_df, _hc=bucket_hc, _pf=bucket_pf,
                      _yt=bucket_y_true, _b=bucket_base):
        alpha = trial.suggest_float("alpha", 0.85, 1.05, step=0.002)
        tau = trial.suggest_int("tau", 30, 300, step=5)
        w_pf = trial.suggest_float("w_pf", 0.0, 0.30, step=0.005)
        d = apply_pp(_df, _hc, _pf, alpha, tau, w_pf)
        return root_mean_squared_error(_yt, _b + d)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=60)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_pp_objective, n_trials=PP_OPTUNA_TRIALS, n_jobs=1)
    PP_PARAMS[bucket_name] = (
        study.best_params["alpha"],
        study.best_params["tau"],
        study.best_params["w_pf"],
        float(study.best_value),
    )
    _dbg(f"  PP {bucket_name}: alpha={PP_PARAMS[bucket_name][0]:.4f} "
         f"tau={PP_PARAMS[bucket_name][1]} w_pf={PP_PARAMS[bucket_name][2]:.4f} "
         f"RMSE={PP_PARAMS[bucket_name][3]:.4f}")

# OOF post-proc score
oof_pp = np.full(len(train_df), np.nan, np.float64)
for bucket_name, train_mask in [("easy", train_easy_mask), ("hard", train_hard_mask)]:
    valid = train_mask & _is_orig & np.isfinite(hc_oof_preds)
    if not valid.any():
        continue
    alpha, tau, w_pf, _ = PP_PARAMS[bucket_name]
    bucket_df = train_df.loc[valid].reset_index(drop=True)
    d = apply_pp(bucket_df, hc_oof_preds[valid], pf_oof[valid], alpha, tau, w_pf)
    oof_pp[valid] = base[valid] + d

_valid_pp = np.isfinite(oof_pp)
pp_oof_rmse = float(root_mean_squared_error(ytrue_full[_valid_pp], oof_pp[_valid_pp]))
_dbg(f"GLOBAL post-proc OOF RMSE: {pp_oof_rmse:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Test Inference + Optional SavGol + Exact Overlap Blend
# ═══════════════════════════════════════════════════════════════════════════════

_dbg("Building test predictions")
test_df2 = test_df.copy()
pf_test = test_df2["pf_ancc"].values - test_df2["last_known_tvt"].values
test_df2["pred"] = test_df2["last_known_tvt"].values.astype(np.float64)

for bucket_name, test_mask in [("easy", test_easy_mask), ("hard", test_hard_mask)]:
    if not test_mask.any() or bucket_name not in PP_PARAMS:
        continue
    alpha, tau, w_pf, _ = PP_PARAMS[bucket_name]
    bucket_df = test_df2.iloc[test_mask].reset_index(drop=True)
    d = apply_pp(bucket_df, hc_test_preds[test_mask], pf_test[test_mask], alpha, tau, w_pf)
    test_df2.loc[test_mask, "pred"] = test_df2.loc[test_mask, "last_known_tvt"].values + d
    _dbg(f"  applied PP for {bucket_name}: alpha={alpha:.4f}, tau={tau}, w_pf={w_pf:.4f}")

# Optional adaptive SavGol (off by default — v26/9.579 found it hurts LB)
if USE_SAVGOL:
    _dbg("Applying adaptive SavGol smoothing")
    for _, g_ in test_df2.groupby("well", sort=False):
        v = g_["pred"].values
        n = len(v)
        sg_w = max(9, min(31, (n // 10) | 1))
        wl = min(sg_w, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= 5:
            v = savgol_filter(v, wl, 3)
        test_df2.loc[g_.index, "pred"] = v

# ── Exact train-coordinate overlap blend (from v26 / 9.579) ──
_dbg("Applying exact train-coordinate overlap blend")


def apply_exact_overlap(sub, data_dir, blend_weight=EXACT_OVERLAP_WEIGHT):
    train_parts = []
    for p in sorted((data_dir / "train").glob("*__horizontal_well.csv")):
        try:
            cur = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"])
        except Exception:
            continue
        cur = cur[cur["TVT"].notna()].copy()
        if not cur.empty:
            train_parts.append(cur)
    if not train_parts:
        return sub
    train_all = pd.concat(train_parts, ignore_index=True)
    for col in ["X", "Y", "Z"]:
        train_all[col + "_r"] = train_all[col].round(2)
    train_map = (train_all.drop_duplicates(subset=["X_r", "Y_r", "Z_r"])
                 .set_index(["X_r", "Y_r", "Z_r"])["TVT"].to_dict())
    coord_parts = []
    for p in sorted((data_dir / "test").glob("*__horizontal_well.csv")):
        wid = p.name.split("__")[0]
        try:
            cur = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT_input"])
        except Exception:
            continue
        mask = cur["TVT_input"].isna().to_numpy()
        if not mask.any():
            continue
        row_idx = np.arange(len(cur))[mask]
        part = cur.loc[mask, ["X", "Y", "Z"]].copy()
        part["id"] = [f"{wid}_{int(i)}" for i in row_idx]
        coord_parts.append(part)
    if not coord_parts:
        return sub
    coord = pd.concat(coord_parts, ignore_index=True)
    coord["key"] = list(zip(coord["X"].round(2), coord["Y"].round(2), coord["Z"].round(2)))
    coord["exact_tvt"] = coord["key"].map(train_map)
    exact = coord[coord["exact_tvt"].notna()][["id", "exact_tvt"]]
    if exact.empty:
        _dbg("Exact overlap: 0 matches")
        return sub
    out = sub.merge(exact, on="id", how="left")
    mask = out["exact_tvt"].notna()
    out.loc[mask, "tvt"] = (
        (1.0 - blend_weight) * out.loc[mask, "tvt"].astype(float)
        + blend_weight * out.loc[mask, "exact_tvt"].astype(float)
    )
    _dbg(f"Exact overlap: blended {int(mask.sum())}/{len(out)} rows (weight={blend_weight:.3f})")
    return out[["id", "tvt"]]


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Write Submission
# ═══════════════════════════════════════════════════════════════════════════════

_dbg("Writing submission.csv")
sample_sub = pd.read_csv(SAMPLE)
sub = sample_sub[["id"]].merge(
    test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}),
    on="id", how="left")
fallback_val = float(train_df["last_known_tvt"].mean() + train_df["target"].mean())
n_null = int(sub["tvt"].isnull().sum())
if n_null:
    _dbg(f"WARN: {n_null} NaN in submission; filling with {fallback_val:.3f}")
    sub["tvt"] = sub["tvt"].fillna(fallback_val)

sub = apply_exact_overlap(sub, DATA)

out_path = OUTPUT_DIR / "submission.csv"
sub[["id", "tvt"]].to_csv(out_path, index=False)

verify = pd.read_csv(out_path)
assert list(verify.columns) == ["id", "tvt"]
assert len(verify) == len(sample_sub)
assert verify["tvt"].isnull().sum() == 0
_dbg(f"submission.csv verified: {len(verify):,d} rows, "
     f"tvt range [{verify['tvt'].min():.2f}, {verify['tvt'].max():.2f}]")

# ── Summary ──
_dbg("=" * 60)
_dbg("FINAL SUMMARY")
_dbg("=" * 60)
for k in sorted(overall_scores.keys(), key=lambda x: overall_scores[x]):
    _dbg(f"  {k:32s}  OOF={overall_scores[k]:.4f}")
_dbg(f"  GLOBAL HC blend:              {hc_score:.4f}")
_dbg(f"  GLOBAL post-proc OOF RMSE:    {pp_oof_rmse:.4f}")
for bname in ("easy", "hard"):
    if bname in PP_PARAMS:
        a, t, wp, rms = PP_PARAMS[bname]
        _dbg(f"  bucket {bname}: alpha={a:.4f}, tau={t}, w_pf={wp:.4f}, RMSE={rms:.4f}")
_dbg(f"Bucket threshold: {PFX_THRESHOLD:.4f}")
elapsed_h = (time.time() - _T0) / 3600
_dbg(f"PIPELINE COMPLETE total={int(time.time() - _T0)}s ({(time.time() - _T0) / 60:.1f}m / {elapsed_h:.2f}h)")
if RUNNING_ON_KAGGLE:
    _dbg(f"Kaggle time budget used: {elapsed_h:.2f}h / 9.0h ({elapsed_h / 9.0 * 100:.0f}%)")
