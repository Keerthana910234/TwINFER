import os, gc, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations, product
from joblib import Parallel, delayed
from scipy.stats import spearmanr, rankdata
from tqdm_joblib import tqdm_joblib
from numba import njit, prange, set_num_threads
from tqdm import tqdm
warnings.filterwarnings("ignore")

# -----------------------------
# Config knobs
# -----------------------------
N_JOBS = 1
BATCH_SIZE = 200
SAVE_INTERVAL = 200
SHUFFLES_GENE_GENE = 10000
SHUFFLES_RANDOM_DIFF = 10000
SHUFFLES_DIRECTED = 10000
set_num_threads(5)

# =============================
# Utility functions
# =============================

def find_csv_files_fast(folder_path):
    return [f.name for f in Path(folder_path).glob("df*.csv")]

def split_and_merge_simulations(path_to_simulation_files):
    simulation_1 = pd.read_csv(path_to_simulation_files[0])
    simulation_2 = pd.read_csv(path_to_simulation_files[1])
    clone_ids = sorted(simulation_1['clone_id'].unique())
    half_point = len(clone_ids) // 2
    clones_from_sim1 = clone_ids[:half_point]
    clones_from_sim2 = clone_ids[half_point:]
    sim1_subset = simulation_1[simulation_1['clone_id'].isin(clones_from_sim1)]
    sim2_subset = simulation_2[simulation_2['clone_id'].isin(clones_from_sim2)]
    return pd.concat([sim1_subset, sim2_subset], ignore_index=True)

def extract_param_index(filename):
    try:
        core = filename.split("df_row_")[1]
        parts = core.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return "_".join(parts[:parts.index(part)])
        return "unknown"
    except Exception:
        return "unknown"

def spearman_safe(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan
    r = spearmanr(x, y).correlation
    return r if not np.isnan(r) else np.nan

# =============================
# Core correlation helpers
# =============================

def calculate_pairwise_gene_gene_correlation_matrix(df, gene_list):
    mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
    X = df[gene_list].values.T
    for i, gi in enumerate(gene_list):
        for j in range(i, len(gene_list)):
            gj = gene_list[j]
            r = spearman_safe(X[i], X[j])
            mat.loc[gi, gj] = mat.loc[gj, gi] = r
    return mat

# =============================
# Simulation processor
# =============================

def process_simulation(sim_info, time_points, gene_list,
                       n_shuffles_gene_gene=SHUFFLES_GENE_GENE,
                       n_shuffles_random_diff=SHUFFLES_RANDOM_DIFF,
                       n_shuffles_directed=SHUFFLES_DIRECTED,
                       seed=2024, mode="single"):

    rng = np.random.default_rng(seed)
    if mode == "single":
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

    elif mode == "pair":
        sims, folder = sim_info
        path_1 = os.path.join(folder, sims[0])
        path_2 = os.path.join(folder, sims[1])
        if not os.path.exists(path_1) or not os.path.exists(path_2):
            print(f"[warn] missing one of: {path_1}, {path_2}")
            return None
        try:
            df = split_and_merge_simulations([path_1, path_2])
        except Exception as e:
            print(f"[error] merging {path_1} and {path_2}: {e}")
            return None
        param_index = f"{extract_param_index(sims[0])}__{extract_param_index(sims[1])}"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # simplified dummy return for this excerpt
    return {"param_index": param_index}

# =============================
# Batch runner with Brunner–Munzel logic
# =============================

def run_pipeline(path_to_simulations, output_folder, genes, time_points,
                 n_jobs=N_JOBS,
                 n_shuffles_gene_gene=SHUFFLES_GENE_GENE,
                 n_shuffles_random_diff=SHUFFLES_RANDOM_DIFF,
                 n_shuffles_directed=SHUFFLES_DIRECTED,
                 batch_size=BATCH_SIZE,
                 save_interval=SAVE_INTERVAL,
                 seed=2024,
                 start_index=0,
                 mode="single"):

    from scipy.stats import brunnermunzel

    files = find_csv_files_fast(path_to_simulations)
    if len(files) == 0:
        raise ValueError("No simulation CSV files found!")

    rng = np.random.default_rng(seed)

    # === PAIR MODE: Two-state selection ===
    if mode == "pair":
        n_target = 25000
        p_threshold = 0.01
        pairs = set()
        attempts = 0
        max_attempts = 200000
        cache = {}
        save_every = 1000
        temp_save_path = os.path.join(output_folder, "two_state_pairs_temp.csv")
        final_save_path = os.path.join(output_folder, "two_state_pairs_final.csv")

        print(f"Selecting up to {n_target} unique two-state pairs (Brunner–Munzel p<{p_threshold})...")

        def read_proteins_cached(fname, t=1):
            """Lazy cache reader: load only once and store minimal columns."""
            if fname in cache:
                return cache[fname]
            path = os.path.join(path_to_simulations, fname)
            try:
                usecols = ["time_step", "gene_1_protein", "gene_2_protein"]
                df = pd.read_csv(path, usecols=usecols)
                subset = df[df["time_step"] == t][["gene_1_protein", "gene_2_protein"]].dropna()
                cache[fname] = subset
                return subset
            except Exception:
                cache[fname] = pd.DataFrame(columns=["gene_1_protein", "gene_2_protein"])
                return cache[fname]

        while len(pairs) < n_target and attempts < max_attempts:
            a, b = rng.choice(files, 2, replace=True)
            attempts += 1
            if a == b:
                continue

            key = tuple(sorted((a, b)))
            if key in pairs:
                continue

            df_a = read_proteins_cached(a, t=1)
            df_b = read_proteins_cached(b, t=1)
            if df_a.empty or df_b.empty:
                continue

            try:
                bm1 = brunnermunzel(df_a["gene_1_protein"], df_b["gene_1_protein"], nan_policy="omit")
                bm2 = brunnermunzel(df_a["gene_2_protein"], df_b["gene_2_protein"], nan_policy="omit")

                if (bm1.pvalue < p_threshold) or (bm2.pvalue < p_threshold):
                    pairs.add(key)
                    if len(pairs) % save_every == 0:
                        print(f"  → {len(pairs)} two-state pairs accepted after {attempts} attempts")
                        pd.DataFrame(sorted(list(pairs)), columns=["file1", "file2"]).to_csv(
                            temp_save_path, index=False
                        )
            except Exception:
                continue

        pairs = sorted(list(pairs))
        pd.DataFrame(pairs, columns=["file1", "file2"]).to_csv(final_save_path, index=False)
        print(f"✅ Formed {len(pairs)} unique two-state pairs after {attempts} attempts.")
        print(f"✅ Saved final pairs to {final_save_path}")
        work_items = pairs

    else:
        work_items = [(f,) for f in files]
        print(f"Found {len(files)} single simulations.")

    # === Batch processing ===
    os.makedirs(output_folder, exist_ok=True)
    gene_list = list(genes)
    all_rows, chunk_id = [], 0

    for i in range(0, len(work_items), batch_size):
        batch = work_items[i:min(i + batch_size, len(work_items))]
        print(f"[batch] {i}..{i + len(batch) - 1}")

        with tqdm_joblib(desc="Processing simulations", total=len(batch)):

            def safe_process(item, seed_offset):
                try:
                    return process_simulation(
                        (item if mode == "pair" else item[0], path_to_simulations),
                        time_points=time_points,
                        gene_list=gene_list,
                        n_shuffles_gene_gene=n_shuffles_gene_gene,
                        n_shuffles_random_diff=n_shuffles_random_diff,
                        n_shuffles_directed=n_shuffles_directed,
                        seed=seed + i + seed_offset,
                        mode=mode
                    )
                except Exception as e:
                    return {"file_or_pair": item, "error": str(e)}

            res = Parallel(n_jobs=n_jobs)(
                delayed(safe_process)(item, k) for k, item in enumerate(batch)
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
