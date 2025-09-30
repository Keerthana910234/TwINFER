# %% Imports
import os, gc, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations, product
from joblib import Parallel, delayed
from scipy.stats import spearmanr, rankdata
from tqdm_joblib import tqdm_joblib
from numba import njit, prange

warnings.filterwarnings("ignore")

# -----------------------------
# Config knobs (adjust as you like)
# -----------------------------
N_JOBS = 4
BATCH_SIZE = 200
SAVE_INTERVAL = 200
SHUFFLES_GENE_GENE = 10000       # Null A
SHUFFLES_RANDOM_DIFF = 10000     # Null B
SHUFFLES_DIRECTED = 10000        # Cross-time pvals
from numba import set_num_threads
set_num_threads(5)

#%%
# =============================
# Utilities
# =============================
def find_csv_files_fast(folder_path):
    return [f.name for f in Path(folder_path).glob("df*.csv")]

def extract_param_index(filename):
    """Extract 'row id' like '0_1' from df_row_0_1_YYYYMMDD_....csv."""
    try:
        core = filename.split("df_row_")[1]
        parts = core.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 8:  # ddmmyyyy
                return "_".join(parts[:parts.index(part)])
        return "unknown"
    except Exception:
        return "unknown"

def spearman_safe(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan
    r = spearmanr(x, y).correlation
    return r if not np.isnan(r) else np.nan

#%%


def _spearman_matrix_from_ranked(R):
    """
    Compute Spearman correlation matrix from a rank-transformed matrix.
    
    Parameters
    ----------
    R : np.ndarray, shape (n_cells, n_genes)
        Rank matrix (midranks per column).
    
    Returns
    -------
    S : np.ndarray, shape (n_genes, n_genes)
        Spearman correlation matrix.
    """
    n = R.shape[0]
    m = (n + 1) / 2.0
    Rc = R - m
    s2 = np.sum(Rc**2, axis=0)
    denom = np.sqrt(np.outer(s2, s2))
    denom[denom == 0] = np.nan
    N = Rc.T @ Rc
    return N / denom

#%%
# =============================
# Subsampling for a time-pair (exactly your recipe)
# =============================
def subsample_for_timepair(simulation, t1, t2, rng):
    clone_ids = simulation["clone_id"].dropna().unique()
    rng.shuffle(clone_ids)
    if len(clone_ids) < 4:
        # not enough to split; return empties
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    n1 = n2 = len(clone_ids) // 4
    t1_clones = clone_ids[:n1]
    t2_clones = clone_ids[n1:n1+n2]
    across_t_clones = clone_ids[n1+n2:]

    t1_twins = simulation[(simulation["clone_id"].isin(t1_clones)) & (simulation["time_step"] == t1)].reset_index(drop=True)
    t2_twins = simulation[(simulation["clone_id"].isin(t2_clones)) & (simulation["time_step"] == t2)].reset_index(drop=True)
    across_t_twin1 = simulation[(simulation["clone_id"].isin(across_t_clones)) & (simulation["time_step"] == t1) & (simulation["replicate"] == 1)].reset_index(drop=True)
    across_t_twin2 = simulation[(simulation["clone_id"].isin(across_t_clones)) & (simulation["time_step"] == t2) & (simulation["replicate"] == 2)].reset_index(drop=True)

    all_t1_t2 = pd.concat([t1_twins, t2_twins, across_t_twin1, across_t_twin2], ignore_index=True)
    return t1_twins, t2_twins, across_t_twin1, across_t_twin2, all_t1_t2

# =============================
# STEP 1 — Gene–gene (pooled over (t1,t2)) + Null A (independent of twin)
# =============================
# def calculate_pairwise_gene_gene_correlation_matrix(df, gene_list):
#     """Undirected full matrix over pooled cells from t1+t2."""
#     mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
#     X = df[gene_list].values.T  # genes x cells
#     for i, gi in enumerate(gene_list):
#         for j in range(i, len(gene_list)):
#             gj = gene_list[j]
#             r = spearman_safe(X[i], X[j])
#             mat.loc[gi, gj] = mat.loc[gj, gi] = r
#     return mat
#%%


def calculate_pairwise_gene_gene_correlation_matrix(df, gene_list):
    """Undirected full matrix over pooled cells from t1+t2."""
    mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
    X = df[gene_list].values.T  # genes x cells
    for i, gi in enumerate(gene_list):
        for j in range(i, len(gene_list)):
            gj = gene_list[j]
            r = spearman_safe(X[i], X[j])
            mat.loc[gi, gj] = mat.loc[gj, gi] = r
    return mat

# ---- Null A: shuffle **cells** (gene–gene null), independent of replicate-difference null

@njit(parallel=True)
def _compute_gene_gene_null(Rc, denom, seeds, triu_i, triu_j):
    n_shuffles = len(seeds)
    n, p = Rc.shape
    n_pairs = len(triu_i)
    out = np.empty((n_shuffles, n_pairs), dtype=np.float64)

    for k in prange(n_shuffles):
        # Set seed inside Numba
        np.random.seed(seeds[k])
        idx = np.random.permutation(Rc.shape[0])  # Numba-supported
        Rc_perm = Rc[idx, :]

        N = Rc.T @ Rc_perm
        C = N / denom

        for pos in range(n_pairs):
            i, j = triu_i[pos], triu_j[pos]
            out[k, pos] = C[i, j]
    return out


def compute_gene_gene_null_distributions(all_t1_t2, gene_list, n_shuffles, n_jobs=None):
    """
    Null A: Spearman null via shuffle of cells, optimized with Numba.
    Returns: dict[(gi, gj)] = np.ndarray(n_shuffles,)
    """
    X = all_t1_t2[gene_list].to_numpy()
    n, p = X.shape

    # Step 1: Pre-rank and center
    R = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    m = (n + 1) / 2.0
    Rc = R - m

    # Step 2: Precompute denominator matrix
    s2 = np.sum(Rc**2, axis=0)
    denom = np.sqrt(np.outer(s2, s2))
    denom[denom == 0] = np.nan

    # Step 3: Setup shuffle parameters
    triu_i, triu_j = np.triu_indices(p, k=0)
    seeds = np.random.randint(0, 2**31 - 1, size=n_shuffles)

    # Step 4: Compute nulls via Numba
    all_ut = _compute_gene_gene_null(Rc, denom, seeds, triu_i, triu_j)

    # Step 5: Format results
    null = {}
    for pos, (i, j) in enumerate(zip(triu_i, triu_j)):
        gi, gj = gene_list[i], gene_list[j]
        null[(min(gi, gj), max(gi, gj))] = all_ut[:, pos]

    return null

#%%
# =============================
# STEP 2 — Twin correlations at time t, per gene-pair (replicate-differences) + Null B
# =============================
def compute_diff_correlation_vectorized(rep1_tf, rep1_target, rep2_tf, rep2_target):
    """Spearman corr between replicate differences for a gene-pair."""
    try:
        d1 = rep1_tf - rep2_tf
        d2 = rep1_target - rep2_target
        if len(d1) < 3:
            return np.nan
        return spearmanr(d1, d2).correlation
    except Exception:
        return np.nan

def twin_pair_correlation_matrix(df_twins, gene_list):
    """
    Spearman correlations between replicate differences at one timepoint.
    Uses scipy.stats.spearmanr instead of manual ranking.
    """
    mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
    if df_twins.empty:
        return mat

    rep1 = df_twins[df_twins["replicate"] == 1]
    rep2 = df_twins[df_twins["replicate"] == 2]
    n = min(len(rep1), len(rep2))
    if n < 3:
        return mat

    # Build difference matrix
    X1 = rep1[gene_list].to_numpy()[:n]
    X2 = rep2[gene_list].to_numpy()[:n]
    D = X1 - X2

    # Direct Spearman correlation matrix
    S, _ = spearmanr(D, axis=0)

    return pd.DataFrame(S, index=gene_list, columns=gene_list)

def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
    rng = np.random.default_rng(random_state)

    rep_0 = simulation_data[simulation_data["replicate"] == 1].reset_index(drop=True)
    rep_1 = simulation_data[simulation_data["replicate"] == 2].reset_index(drop=True)
    min_cells = min(len(rep_0), len(rep_1))
    p = len(gene_list)

    if min_cells < 3:
        return {(min(gi, gj), max(gi, gj)): np.array([])
                for i, gi in enumerate(gene_list) for j, gj in enumerate(gene_list) if j >= i}

    X1 = rep_0[gene_list].to_numpy()[:min_cells]
    X2 = rep_1[gene_list].to_numpy()[:min_cells]

    triu_i, triu_j = np.triu_indices(p, k=1)
    seeds = rng.integers(0, 2**31 - 1, size=n_shuffles)

    all_ut = _generate_random_shuffle_fast(X1, X2, triu_i, triu_j, seeds)

    correlation_dict = {}
    for pos, (i, j) in enumerate(zip(triu_i, triu_j)):
        gi, gj = gene_list[i], gene_list[j]
        correlation_dict[(min(gi, gj), max(gi, gj))] = all_ut[:, pos]

    # Add diagonals with empty arrays (if needed)
    for g in gene_list:
        key = (g, g)
        if key not in correlation_dict:
            correlation_dict[key] = np.array([])

    return correlation_dict

@njit(parallel=True)
def _generate_random_shuffle_fast(X1, X2, triu_i, triu_j, seeds):
    n_shuffles = len(seeds)
    n, p = X1.shape
    out = np.empty((n_shuffles, len(triu_i)), dtype=np.float64)

    for s in prange(n_shuffles):
        np.random.seed(seeds[s])
        idx = np.random.permutation(n)
        D = X1 - X2[idx, :]

        # Compute Spearman from difference matrix
        S = spearman_matrix_from_diff(D)

        for k in range(len(triu_i)):
            i, j = triu_i[k], triu_j[k]
            out[s, k] = S[i, j]
    return out


from numba import njit
import numpy as np

from numba import njit
import numpy as np

@njit
def rankdata_numba(a):
    n = a.size
    temp = np.argsort(a)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        start = i
        val = a[temp[i]]
        while i + 1 < n and a[temp[i + 1]] == val:
            i += 1
        end = i
        avg_rank = 0.5 * (start + end) + 1
        for j in range(start, end + 1):
            ranks[temp[j]] = avg_rank
        i += 1
    return ranks

@njit
def spearman_matrix_from_diff(D):
    n, p = D.shape
    R = np.empty_like(D)
    for j in range(p):
        R[:, j] = rankdata_numba(D[:, j])

    m = (n + 1) / 2.0
    Rc = R - m

    s2 = np.sum(Rc**2, axis=0)
    denom = np.sqrt(np.outer(s2, s2))

    # ⚠️ Avoid boolean indexing: replace zeros with np.nan manually
    n_rows, n_cols = denom.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if denom[i, j] == 0.0:
                denom[i, j] = np.nan

    N = Rc.T @ Rc
    return N / denom


# def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
#     """
#     Null B: random-pair replicate-difference null distribution.
#     Returns dict {(gi, gj): np.ndarray(n_shuffles,)}.
#     """
#     rng = np.random.default_rng(random_state)

#     rep_0 = simulation_data[simulation_data["replicate"] == 1].reset_index(drop=True)
#     rep_1 = simulation_data[simulation_data["replicate"] == 2].reset_index(drop=True)
#     min_cells = min(len(rep_0), len(rep_1))
#     p = len(gene_list)

#     if min_cells < 3:
#         return {(min(gi, gj), max(gi, gj)): np.array([])
#                 for i, gi in enumerate(gene_list) for j, gj in enumerate(gene_list) if j >= i}

#     X1 = rep_0[gene_list].to_numpy()[:min_cells]
#     X2 = rep_1[gene_list].to_numpy()[:min_cells]
#     n = X1.shape[0]

#     triu_i, triu_j = np.triu_indices(p, k=1)
#     all_ut = np.empty((n_shuffles, len(triu_i)), dtype=np.float64)

#     B = 100
#     for b in range(0, n_shuffles, B):
#         be = min(b + B, n_shuffles)
#         idx_batch = np.vstack([rng.permutation(n) for _ in range(be - b)])

#         for s, idx in enumerate(idx_batch):
#             D = X1 - X2[idx, :]

#             # Handle single-gene case
#             if p == 1:
#                 S = np.array([[1.0]])  # correlation of a variable with itself
#             else:
#                 S, _ = spearmanr(D, axis=0)

#             # If only a scalar was returned, convert it into a 1×1 or 2×2 matrix
#             if np.isscalar(S):
#                 if D.shape[1] == 1:
#                     S = np.array([[1.0]])   # self-correlation
#                 else:
#                     # Just in case scipy collapses two variables into a scalar
#                     S = np.array([[1.0, S], [S, 1.0]])
#             all_ut[b + s] = S[triu_i, triu_j]

#     correlation_dict = {}
#     for pos, (i, j) in enumerate(zip(triu_i, triu_j)):
#         gi, gj = gene_list[i], gene_list[j]
#         correlation_dict[(min(gi, gj), max(gi, gj))] = all_ut[:, pos]

#     for g in gene_list:
#         key = (g, g)
#         if key not in correlation_dict:
#             correlation_dict[key] = np.array([])

#     return correlation_dict



# #%%
# def directed_cross_time_with_pvals(across_twin1, across_twin2, t1, t2, gene_list, n_shuffles):
#     """
#     Compute cross-time Spearman correlations (using scipy.spearmanr)
#     and permutation p-values.

#     Returns dict {(src_gene_time, tgt_gene_time): (corr, pval)}
#     for all non-diagonal cross-time pairs.
#     """
#     out = {}
#     if across_twin1.empty or across_twin2.empty:
#         return out

#     X = across_twin1[gene_list].to_numpy()
#     Y = across_twin2[gene_list].to_numpy()
#     n = min(len(X), len(Y))
#     if n < 3:
#         return out

#     # Align number of cells
#     X = X[:n]
#     Y = Y[:n]
#     p = len(gene_list)   # here p == 2 always

#     # --- Observed correlations ---
#     XY = np.hstack([X, Y])                  # shape (n, 2p)
#     obs_corr, _ = spearmanr(XY, axis=0)     # guaranteed (2p × 2p)
#     if np.ndim(obs_corr) == 0:              # fallback: single float (shouldn't happen now)
#         obs_corr = np.array([[1.0]])        # dummy identity
#     S_obs_XY = obs_corr[:p, p:]             # shape (p × p)

#     # --- Permutations ---
#     counts_XY = np.zeros_like(S_obs_XY, dtype=np.int32)

#     for _ in range(n_shuffles):
#         idx = np.random.permutation(n)
#         Y_perm = Y[idx, :]
#         XY_perm = np.hstack([X, Y_perm])
#         perm_corr, _ = spearmanr(XY_perm, axis=0)
#         if np.ndim(perm_corr) == 0:
#             perm_corr = np.array([[1.0]])
#         S_perm_XY = perm_corr[:p, p:]

#         counts_XY += (np.abs(S_perm_XY) >= np.abs(S_obs_XY))

#     # --- Package results (exclude same-gene diagonals) ---
#     for i, gi in enumerate(gene_list):
#         for j, gj in enumerate(gene_list):
#             if gi == gj:   # skip same-gene diagonals
#                 continue
#             out[(f"{gi}_t{t1}", f"{gj}_t{t2}")] = (
#                 S_obs_XY[i, j],
#                 counts_XY[i, j] / float(n_shuffles)
#             )

#     return out

def directed_cross_time_with_pvals(across_twin1, across_twin2, t1, t2, gene_list, n_shuffles):
    """
    Optimized: Compute cross-time Spearman correlations and p-values.
    Returns dict {(src_gene_time, tgt_gene_time): (corr, pval)}
    """
    out = {}
    if across_twin1.empty or across_twin2.empty:
        return out

    X = across_twin1[gene_list].to_numpy()
    Y = across_twin2[gene_list].to_numpy()
    n = min(len(X), len(Y))
    if n < 3:
        return out

    X = X[:n]
    Y = Y[:n]
    p = len(gene_list)

    # Step 1: Precompute ranks and center
    RX = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    RY = np.apply_along_axis(rankdata, 0, Y).astype(np.float64)

    m = (n + 1) / 2.0
    RXc = RX - m
    RYc = RY - m

    s2x = np.sum(RXc**2, axis=0)
    s2y = np.sum(RYc**2, axis=0)
    denom = np.sqrt(np.outer(s2x, s2y))
    denom[denom == 0] = np.nan

    # Step 2: Generate random seeds
    seeds = np.random.randint(0, 2**31 - 1, size=n_shuffles)

    # Step 3: Run fast permutation test
    S_obs, counts = _permutation_counts_cross_time(RXc, RYc, denom, seeds)

    # Step 4: Package results (excluding same-gene diagonals)
    for i, gi in enumerate(gene_list):
        for j, gj in enumerate(gene_list):
            if gi == gj:
                continue
            key = (f"{gi}_t{t1}", f"{gj}_t{t2}")
            corr = S_obs[i, j]
            pval = counts[i, j] / float(n_shuffles)
            out[key] = (corr, pval)

    return out

@njit(parallel=True)
def _permutation_counts_cross_time(RXc, RYc, denom, seeds):
    n, p = RXc.shape
    counts = np.zeros((p, p), dtype=np.int32)
    n_shuffles = len(seeds)

    # Observed correlation
    N_obs = RXc.T @ RYc
    S_obs = N_obs / denom

    for k in prange(n_shuffles):
        np.random.seed(seeds[k])
        idx = np.random.permutation(n)
        RYc_perm = RYc[idx, :]
        N_perm = RXc.T @ RYc_perm
        S_perm = N_perm / denom
        for i in range(p):
            for j in range(p):
                if np.isnan(S_obs[i, j]) or np.isnan(S_perm[i, j]):
                    continue
                if abs(S_perm[i, j]) >= abs(S_obs[i, j]):
                    counts[i, j] += 1

    return S_obs, counts



#%%
# =============================
# Per-simulation → one wide row
# =============================
def process_simulation(sim_info, time_points, gene_list,
                       n_shuffles_gene_gene=SHUFFLES_GENE_GENE,
                       n_shuffles_random_diff=SHUFFLES_RANDOM_DIFF,
                       n_shuffles_directed=SHUFFLES_DIRECTED,
                       seed=2024):
    sim, folder = sim_info
    path = os.path.join(folder, sim)
    if not os.path.exists(path):
        print(f"[warn] missing {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[error] reading {path}: {e}")
        return None

    param_index = extract_param_index(sim)
    rng = np.random.default_rng(seed)
    row = {"param_index": param_index}

    # build time pairs (t1 < t2). If you also want (t,t), include those too.
    time_pairs = [(time_points[i], time_points[j])
                  for i in range(len(time_points))
                  for j in range(i+1, len(time_points))]

    for (t1, t2) in time_pairs:
        # ---- subsample
        t1_twins, t2_twins, twin1, twin2, all_t1_t2 = subsample_for_timepair(df, t1, t2, rng)

        # ======================
        # STEP 1: gene–gene corr (pooled over t1+t2) + Null A (SEPARATE)
        # ======================
        if all_t1_t2.empty:
            # fill NaNs for all pairs
            for i, gi in enumerate(gene_list):
                for j in range(i, len(gene_list)):
                    gj = gene_list[j]
                    row[f"corr_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = np.nan
                    row[f"pval_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = np.nan
            # still compute directed/self below (may also be NaN if empty)
        else:
            #Calculate gene-gene correlation matrix
            gg_mat = calculate_pairwise_gene_gene_correlation_matrix(all_t1_t2, gene_list)
            # Null A - check the logic
            gg_null = compute_gene_gene_null_distributions(
                all_t1_t2, gene_list,
                n_shuffles=n_shuffles_gene_gene,
                n_jobs=min(4, os.cpu_count() - 2)
            )
            for i, gi in enumerate(gene_list):
                for j in range(i, len(gene_list)):
                    gj = gene_list[j]
                    obs = gg_mat.loc[gi, gj]
                    key = (min(gi, gj), max(gi, gj))
                    null_vals = gg_null.get(key, np.array([]))
                    pval = np.nan if null_vals.size == 0 else np.mean(np.abs(null_vals) >= abs(obs))
                    row[f"corr_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = obs
                    row[f"pval_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = pval

        # ======================
        # STEP 2: twin corr at t1 and at t2; pvals vs Null B (SEPARATE)
        #         Null B is generated once from pooled (t1,t2)
        # ======================
        # Null B (replicate-difference random pairs over pooled (t1,t2))
        rd_null = generate_random_shuffle(all_t1_t2, gene_list,
                                          n_shuffles=n_shuffles_random_diff,
                                          random_state=seed)

        # twin at t1
        t1_twin_mat = twin_pair_correlation_matrix(t1_twins, gene_list)
        for gi in gene_list:
            for gj in gene_list:
                if gi !=gj:
                    obs = t1_twin_mat.loc[gi, gj]
                    key = (min(gi, gj), max(gi, gj))
                    null_vals = rd_null.get(key, np.array([]))
                    if null_vals.size == 0 or np.isnan(obs):
                        pval = np.nan
                        zscore = np.nan
                    else:
                        pval = np.mean(np.abs(null_vals) >= abs(obs))
                        null_mean = np.mean(null_vals)
                        null_std = np.std(null_vals, ddof=1)  # unbiased estimate
                        zscore = (obs - null_mean) / null_std if null_std > 0 else np.nan
                    row[f"zscore_twin_vs_random_{gi}_{gj}_t{t1}"] = zscore
                    row[f"twin_corr_{gi}_{gj}_t{t1}"] = obs
                    row[f"pval_twin_vs_random_{gi}_{gj}_t{t1}"] = pval


        # twin at t2
        t2_twin_mat = twin_pair_correlation_matrix(t2_twins, gene_list)
        for gi in gene_list:
            for gj in gene_list:
                if gi !=gj:
                    obs = t2_twin_mat.loc[gi, gj]
                    key = (min(gi, gj), max(gi, gj))
                    null_vals = rd_null.get(key, np.array([]))
                    if null_vals.size == 0 or np.isnan(obs):
                        pval = np.nan
                        zscore = np.nan
                    else:
                        pval = np.mean(np.abs(null_vals) >= abs(obs))
                        null_mean = np.mean(null_vals)
                        null_std = np.std(null_vals, ddof=1)  # unbiased estimate
                        zscore = (obs - null_mean) / null_std if null_std > 0 else np.nan
                    
                    row[f"twin_corr_{gi}_{gj}_t{t2}"] = obs
                    row[f"pval_twin_vs_random_{gi}_{gj}_t{t2}"] = pval
                    row[f"zscore_twin_vs_random_{gi}_{gj}_t{t2}"] = zscore

        # ======================
        # STEP 3: Directed cross-time (ordered) + pvals; Self-corr
        # ======================
        dc = directed_cross_time_with_pvals(twin1, twin2, t1, t2, gene_list, n_shuffles=n_shuffles_directed)

        for (src, tgt), (corr, pval) in dc.items():
            row[f"directed_corr_{src}__{tgt}"] = corr
            row[f"directed_pval_{src}__{tgt}"] = pval


        # self corr (raw only)
        for g in gene_list:
            x = twin1[g].values
            y = twin2[g].values
            n = min(len(x), len(y))
            row[f"self_corr_{g}_t{t1}_t{t2}"] = np.nan if n < 3 else spearman_safe(x[:n], y[:n])

    del df
    gc.collect()
    return row
#%%
# =============================
# Batch runner
# =============================
def run_pipeline(path_to_simulations, output_folder, genes, time_points,
                 n_jobs=N_JOBS,
                 n_shuffles_gene_gene=SHUFFLES_GENE_GENE,
                 n_shuffles_random_diff=SHUFFLES_RANDOM_DIFF,
                 n_shuffles_directed=SHUFFLES_DIRECTED,
                 batch_size=BATCH_SIZE, save_interval=SAVE_INTERVAL, seed=2024, start_index=0):

    files = find_csv_files_fast(path_to_simulations)

    def sort_key(fname):
        try:
            idx_str = extract_param_index(fname)  # e.g. "0_1_2"
            idx_parts = [int(x) for x in idx_str.split("_")]
            return idx_parts  # Python compares tuples/lists lexicographically
        except Exception:
            return [float("inf")]  # fallback for unexpected filenames

    files = sorted(files, key=sort_key)[start_index:]
    print(f"Found {len(files)} files.")
    os.makedirs(output_folder, exist_ok=True)

    # encourage consistent column names by using gene names as provided
    gene_list = list(genes)

    all_rows, chunk_id = [], 0

    for i in range(0, len(files), batch_size):
        batch = files[i:min(i + batch_size, len(files))]
        print(f"[batch] {i}..{i+len(batch)-1}")

        with tqdm_joblib(desc="Processing simulations", total=len(batch)):

            def safe_process(fname, seed_offset):
                try:
                    return process_simulation(
                        (fname, path_to_simulations),
                        time_points=time_points,
                        gene_list=gene_list,
                        n_shuffles_gene_gene=n_shuffles_gene_gene,
                        n_shuffles_random_diff=n_shuffles_random_diff,
                        n_shuffles_directed=n_shuffles_directed,
                        seed=seed + i + seed_offset
                    )
                except Exception as e:
                    # record the error in the output
                    return {
                        "file": fname,
                        "error": str(e)
                    }

            res = Parallel(n_jobs=n_jobs)(
                delayed(safe_process)(fname, k) for k, fname in enumerate(batch)
            )

        res = [r for r in res if r is not None]
        all_rows.extend(res)

        if len(all_rows) >= save_interval:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(output_folder, f"results_chunk_{chunk_id:03d}.csv"),
                index=False
            )
            print(f"[saved] chunk {chunk_id} with {len(all_rows)} rows")
            all_rows.clear()
            chunk_id += 1

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(output_folder, f"results_chunk_{chunk_id:03d}.csv"),
            index=False
        )
        print(f"[saved] final chunk {chunk_id} with {len(all_rows)} rows")
    
#%%
# =============================
# CLI
# =============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation pipeline with separate nulls for gene-gene and twin-random.")
    parser.add_argument("--path_to_simulations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--genes", nargs="+", required=True,
                        help="Exact column names for gene expressions (e.g., gene_1_mRNA gene_2_mRNA ...)")
    parser.add_argument("--timepoints", nargs="+", type=int, required=True,
                        help="List of time points (e.g., 1 5 10 20). All (t1<t2) pairs will be used.")
    parser.add_argument("--shuffles_gene_gene", type=int, default=SHUFFLES_GENE_GENE)
    parser.add_argument("--shuffles_random_diff", type=int, default=SHUFFLES_RANDOM_DIFF)
    parser.add_argument("--shuffles_directed", type=int, default=SHUFFLES_DIRECTED)
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    path_to_simulations=args.path_to_simulations
    output_folder=args.output
    genes=args.genes
    time_points=args.timepoints
    n_jobs=args.jobs
    n_shuffles_gene_gene=args.shuffles_gene_gene
    n_shuffles_random_diff=args.shuffles_random_diff
    n_shuffles_directed=args.shuffles_directed
    batch_size=args.batch_size
    save_interval=args.save_interval
    start_index=args.start_index
    seed=args.seed
    
    # path_to_simulations="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B/"
    # output_folder="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/temp"
    # genes=["gene_1_mRNA", "gene_2_mRNA"]
    # time_points=[1,10]
    # n_jobs=1
    # n_shuffles_gene_gene=10000
    # n_shuffles_random_diff=10000
    # n_shuffles_directed=10000
    # batch_size=10
    # save_interval=10
    # seed=101010
    # start_index = 0

    run_pipeline(
        path_to_simulations=path_to_simulations,
        output_folder=output_folder,
        genes=genes,
        time_points=time_points,
        n_jobs=n_jobs,
        n_shuffles_gene_gene=n_shuffles_gene_gene,
        n_shuffles_random_diff=n_shuffles_random_diff,
        n_shuffles_directed=n_shuffles_directed,
        batch_size=batch_size,
        save_interval=save_interval,
        seed=seed,
        start_index = start_index
    )
