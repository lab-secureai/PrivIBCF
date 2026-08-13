"""Experimental code for PrivIBCF."""

from __future__ import annotations

import os
import re
import gc
import math
import time
import json
import random
import zipfile
import subprocess
import platform
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


BASE_SEED = 2026
PAPER_SEEDS = [2026, 2027, 2028, 2029, 2030]
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

ROOT = Path('/content/privibcf_full_paper') if Path('/content').exists() else Path('/tmp/privibcf_full_paper')
RAW = ROOT / 'raw'
CACHE = ROOT / 'cache'
OUT = ROOT / 'results'
FIG = OUT / 'figures'
for p in (RAW, CACHE, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    'MovieLens-1M': {'n_users': 3000, 'm_values': [500, 1000, 1500, 2000]},
    'Netflix Prize': {'n_users': 5000, 'm_values': [500, 1000, 1500, 2000]},
    'Amazon Book': {'n_users': 10000, 'm_values': [200, 400, 600, 800, 1000]},
}
RUN_DATASETS = list(EXPERIMENTS)

# Evaluation settings
MIN_RATINGS_FOR_RATING_TEST = 5
MIN_RATINGS_FOR_RANK_TEST = 5
RATING_TEST_FRAC = 0.20
RELEVANT_THRESHOLD = 4.0
NEGATIVE_SAMPLES = 99
NEIGHBOR_K = 50
RANK_KS = (5, 10)
FIXEDPOINT_DIGITS = (1, 2, 3, 4)
DEFAULT_FIXEDPOINT_D = 2
SPARSITY_KEEP_RATES = (1.00, 0.75, 0.50, 0.25)
PHASE2_CANDIDATES = (100, 200, 300, 400, 500)
PHASE2_SAMPLE_USERS = 300

# Set these to an integer for a quicker development run.
MAX_RATING_EVAL_USERS: Optional[int] = None
MAX_RANK_EVAL_USERS: Optional[int] = None
MAX_EQUIV_EVAL_USERS: Optional[int] = 2000

# The first Netflix combined file is enough for the default 5k x 2k subset.
NETFLIX_USE_ALL_FILES = False

# Communication accounting
GROUP_ELEMENT_BYTES = 2048 // 8
SIMILARITY_BYTES = 8  # float64 when sharing positive item-item similarities.


def download_with_wget(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f'[cache] {out_path.name}: {out_path.stat().st_size/1e6:.1f} MB')
        return out_path
    print('Downloading:', url)
    subprocess.run(['wget', '-c', '--show-progress', '-O', str(out_path), url], check=True)
    return out_path


def download_movielens_1m() -> Path:
    z = RAW / 'ml-1m.zip'
    download_with_wget('https://files.grouplens.org/datasets/movielens/ml-1m.zip', z)
    ratings = RAW / 'ml-1m' / 'ratings.dat'
    if not ratings.exists():
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(RAW)
    return ratings


def download_amazon_books() -> Path:
    return download_with_wget(
        'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv',
        RAW / 'ratings_Books.csv',
    )


def download_netflix_prize() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError('Install kagglehub: pip install kagglehub') from e

    marker = CACHE / 'netflix_dataset_path.txt'
    if marker.exists():
        saved = Path(marker.read_text().strip())
        if saved.exists():
            return saved

    print('Downloading Netflix Prize through KaggleHub...')
    try:
        path = Path(kagglehub.dataset_download('netflix-inc/netflix-prize-data'))
    except Exception as e:
        raise RuntimeError(
            'KaggleHub could not download netflix-inc/netflix-prize-data. '
            'If authentication is requested, configure Kaggle credentials in Colab and rerun.'
        ) from e
    marker.write_text(str(path))
    return path


def _cache_path(name: str) -> Path:
    return CACHE / f'{name}.parquet'


def _load_cache(name: str) -> Optional[pd.DataFrame]:
    p = _cache_path(name)
    return pd.read_parquet(p) if p.exists() else None


def _save_cache(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_parquet(_cache_path(name), index=False)
    return df


def prepare_movielens_subset(n_users=3000, max_items=2000) -> pd.DataFrame:
    cached = _load_cache('movielens_selected_v2')
    if cached is not None:
        return cached
    path = download_movielens_1m()
    df = pd.read_csv(path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    # Build a deterministic popularity-based subset.
    items = df['item'].value_counts().head(max_items).index
    tmp = df[df['item'].isin(items)]
    users = tmp['user'].value_counts().head(n_users).index
    out = tmp[tmp['user'].isin(users)][['user', 'item', 'rating']].copy()
    out['source'] = 'MovieLens-1M'
    return _save_cache(out, 'movielens_selected_v2')


def _netflix_files(dataset_dir: Path) -> List[Path]:
    files = sorted(dataset_dir.rglob('combined_data_*.txt'))
    if not files:
        for z in dataset_dir.rglob('*.zip'):
            target = dataset_dir / ('unzipped_' + z.stem)
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, 'r') as zf:
                    zf.extractall(target)
        files = sorted(dataset_dir.rglob('combined_data_*.txt'))
    if not files:
        raise FileNotFoundError('No combined_data_*.txt found in Netflix Prize download.')
    return files if NETFLIX_USE_ALL_FILES else files[:1]


def _netflix_item_counts(files: Sequence[Path]) -> Counter:
    counts = Counter()
    for fp in files:
        print('Netflix pass 1/3 — item counts:', fp.name)
        current = None
        c = 0
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(':'):
                    if current is not None:
                        counts[current] += c
                    current = int(line[:-1])
                    c = 0
                else:
                    c += 1
        if current is not None:
            counts[current] += c
    return counts


def _netflix_user_counts(files: Sequence[Path], items: set) -> Counter:
    counts = Counter()
    for fp in files:
        print('Netflix pass 2/3 — user counts:', fp.name)
        current = None
        keep = False
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(':'):
                    current = int(line[:-1])
                    keep = current in items
                elif keep:
                    counts[int(line.split(',', 1)[0])] += 1
    return counts


def _netflix_collect(files: Sequence[Path], items: set, users: set) -> pd.DataFrame:
    rows = []
    for fp in files:
        print('Netflix pass 3/3 — collect:', fp.name)
        current = None
        keep = False
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(':'):
                    current = int(line[:-1])
                    keep = current in items
                elif keep:
                    parts = line.split(',')
                    u = int(parts[0])
                    if u in users:
                        rows.append((u, current, float(parts[1])))
    return pd.DataFrame(rows, columns=['user', 'item', 'rating'])


def prepare_netflix_subset(n_users=5000, max_items=2000) -> pd.DataFrame:
    cached = _load_cache('netflix_selected_v2')
    if cached is not None:
        return cached
    d = download_netflix_prize()
    files = _netflix_files(d)
    item_counts = _netflix_item_counts(files)
    items = set(x for x, _ in item_counts.most_common(max_items))
    user_counts = _netflix_user_counts(files, items)
    users = set(x for x, _ in user_counts.most_common(n_users))
    out = _netflix_collect(files, items, users)
    out['source'] = 'Netflix Prize'
    return _save_cache(out, 'netflix_selected_v2')


def _amazon_top_items(path: Path, max_items=1200, chunksize=2_000_000) -> List[str]:
    # Keep a small buffer so the final subset still contains the requested number of items.
    cp = CACHE / f'amazon_top_items_pool_{max_items}.json'
    if cp.exists():
        return json.loads(cp.read_text())
    counts = Counter()
    print('Amazon pass 1/3 — item counts...')
    for chunk in pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], chunksize=chunksize,
                             dtype={'item': str}, usecols=['item']):
        counts.update(chunk['item'].value_counts().to_dict())
    items = [str(x) for x, _ in counts.most_common(max_items)]
    cp.write_text(json.dumps(items))
    return items


def _amazon_top_users(path: Path, items: Sequence[str], n_users=12000, chunksize=2_000_000) -> List[str]:
    cp = CACHE / f'amazon_top_users_pool_{n_users}_{len(items)}.json'
    if cp.exists():
        return json.loads(cp.read_text())
    item_set = set(map(str, items))
    counts = Counter()
    print('Amazon pass 2/3 — user counts on candidate items...')
    for chunk in pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], chunksize=chunksize,
                             dtype={'user': str, 'item': str}):
        chunk = chunk[chunk['item'].isin(item_set)]
        counts.update(chunk['user'].value_counts().to_dict())
    users = [str(x) for x, _ in counts.most_common(n_users)]
    cp.write_text(json.dumps(users))
    return users


def _amazon_collect(path: Path, items: Sequence[str], users: Sequence[str], chunksize=2_000_000) -> pd.DataFrame:
    item_set, user_set = set(map(str, items)), set(map(str, users))
    pieces = []
    print('Amazon pass 3/3 — collect candidate cross-submatrix...')
    for chunk in pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], chunksize=chunksize,
                             dtype={'user': str, 'item': str}):
        chunk = chunk[chunk['item'].isin(item_set) & chunk['user'].isin(user_set)]
        if len(chunk):
            pieces.append(chunk[['user', 'item', 'rating']].copy())
    if not pieces:
        raise RuntimeError('Amazon subset is empty.')
    return pd.concat(pieces, ignore_index=True)


def prepare_amazon_subset(n_users=10000, max_items=1000) -> pd.DataFrame:
    cached = _load_cache('amazon_selected_v2')
    if cached is not None:
        return cached
    path = download_amazon_books()
    item_pool = _amazon_top_items(path, max_items=max(1200, int(max_items * 1.25)))
    user_pool = _amazon_top_users(path, item_pool, n_users=max(12000, int(n_users * 1.2)))
    candidate = _amazon_collect(path, item_pool, user_pool)

    # Refine users and items a few times to keep the nested subset stable.
    tmp = candidate
    for _ in range(4):
        users = tmp['user'].value_counts().head(n_users).index
        tmp = candidate[candidate['user'].isin(users)]
        items = tmp['item'].value_counts().head(max_items).index
        tmp = candidate[candidate['user'].isin(users) & candidate['item'].isin(items)]
    out = tmp[['user', 'item', 'rating']].copy()
    out['source'] = 'Amazon Book'
    return _save_cache(out, 'amazon_selected_v2')


def prepare_all_selected() -> Dict[str, pd.DataFrame]:
    data = {}
    if 'MovieLens-1M' in RUN_DATASETS:
        data['MovieLens-1M'] = prepare_movielens_subset(3000, 2000)
    if 'Netflix Prize' in RUN_DATASETS:
        data['Netflix Prize'] = prepare_netflix_subset(5000, 2000)
    if 'Amazon Book' in RUN_DATASETS:
        data['Amazon Book'] = prepare_amazon_subset(10000, 1000)
    return data


def nested_setting_df(selected: pd.DataFrame, n_users: int, m_items: int) -> pd.DataFrame:
    # Select items first, then users, and rerank items within the selected users.
    item_order = selected['item'].value_counts().index.tolist()
    tmp = selected[selected['item'].isin(set(item_order[:m_items]))].copy()
    user_order = tmp['user'].value_counts().index.tolist()
    tmp = tmp[tmp['user'].isin(set(user_order[:n_users]))].copy()
    # Rerank after user filtering in case some items disappeared.
    item_order2 = tmp['item'].value_counts().index.tolist()
    tmp = tmp[tmp['item'].isin(set(item_order2[:m_items]))].copy()
    return tmp


def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index, pd.Index]:
    users = pd.Index(df['user'].unique())
    items = pd.Index(df['item'].unique())
    um = {u: i for i, u in enumerate(users)}
    im = {it: j for j, it in enumerate(items)}
    out = df.copy()
    out['u'] = out['user'].map(um).astype(np.int32)
    out['i'] = out['item'].map(im).astype(np.int32)
    return out, users, items


def per_user_random_holdout(df: pd.DataFrame, test_frac=0.2, min_ratings=5, seed=BASE_SEED):
    """Create a per-user random holdout for rating prediction."""
    rng = np.random.default_rng(seed)
    test_indices = []
    for _, g in df.groupby('u', sort=False):
        if len(g) < min_ratings:
            continue
        n_test = max(1, int(round(len(g) * test_frac)))
        n_test = min(n_test, len(g) - 2)  # leave at least two train ratings when possible
        if n_test <= 0:
            continue
        chosen = rng.choice(g.index.to_numpy(), size=n_test, replace=False)
        test_indices.extend(chosen.tolist())
    test = df.loc[test_indices].copy()
    train = df.drop(index=test_indices).copy()
    return train, test


def positive_leave_one_out(df: pd.DataFrame, threshold=4.0, min_ratings=5, seed=BASE_SEED):
    """Hold out one relevant item per eligible user for Top-N evaluation."""
    rng = np.random.default_rng(seed)
    test_indices = []
    for _, g in df.groupby('u', sort=False):
        if len(g) < min_ratings:
            continue
        pos = g[g['rating'] >= threshold]
        if len(pos) == 0:
            continue
        test_indices.append(int(rng.choice(pos.index.to_numpy(), size=1)[0]))
    test = df.loc[test_indices].copy()
    train = df.drop(index=test_indices).copy()
    return train, test


def build_rating_matrix(train: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    return csr_matrix(
        (train['rating'].astype(np.float64).to_numpy(),
         (train['u'].to_numpy(), train['i'].to_numpy())),
        shape=(n_users, n_items), dtype=np.float64,
    )


def _centered_sparse(R: csr_matrix):
    R = R.tocsr().astype(np.float64)
    counts = np.diff(R.indptr)
    sums = np.asarray(R.sum(axis=1)).ravel()
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    X = R.copy()
    X.data = X.data - np.repeat(means, counts)
    B = R.copy()
    B.data = np.ones_like(B.data)
    return R, X, B, means


def adjusted_cosine_from_centered(X: csr_matrix, B: csr_matrix, positive_only=True):
    X2 = X.copy()
    X2.data = X2.data ** 2
    N = (X.T @ X).toarray()
    A = (X2.T @ B).toarray()
    den = np.sqrt(A * A.T)
    S = np.divide(N, den, out=np.zeros_like(N), where=den > 0)
    np.fill_diagonal(S, 0.0)
    if positive_only:
        S[S <= 0] = 0.0
    return S, N, A


def adjusted_cosine_pair_specific(R: csr_matrix, positive_only=True, return_components=False):
    t0 = time.perf_counter()
    R, X, B, means = _centered_sparse(R)
    S, N, A = adjusted_cosine_from_centered(X, B, positive_only=positive_only)
    elapsed = time.perf_counter() - t0
    upper = np.triu_indices(S.shape[0], k=1)
    num = N[upper]
    all_pairs = len(num)
    positive_pairs = int(np.count_nonzero(num > 0))
    nonzero_pairs = int(np.count_nonzero(num != 0))
    diag = {
        'acos_build_seconds': elapsed,
        'all_item_pairs': int(all_pairs),
        'positive_numerator_pairs': positive_pairs,
        'nonzero_numerator_pairs': nonzero_pairs,
        'positive_pair_ratio': positive_pairs / all_pairs if all_pairs else 0.0,
    }
    out = (S.astype(np.float32), means, diag)
    if return_components:
        out += (X, B, N, A)
    return out


def adjusted_cosine_fixedpoint(R: csr_matrix, digits: int, positive_only=True):
    """
    Quantize centered residuals q_ij = round(10^d * (r_ij - mean_i)).
    Each user can then form integer local contributions q_ij*q_it and q_ij^2.
    The common factor 10^(2d) cancels in Adjusted Cosine.
    """
    t0 = time.perf_counter()
    R, X, B, means = _centered_sparse(R)
    scale = 10 ** digits
    Xq = X.copy()
    Xq.data = np.rint(Xq.data * scale).astype(np.int64).astype(np.float64)
    S, Nq, Aq = adjusted_cosine_from_centered(Xq, B, positive_only=positive_only)
    elapsed = time.perf_counter() - t0
    q_abs_max = int(np.max(np.abs(Xq.data))) if Xq.nnz else 0
    product_bound = q_abs_max * q_abs_max
    return S.astype(np.float32), {
        'digits': digits,
        'scale': scale,
        'fixedpoint_build_seconds': elapsed,
        'max_abs_quantized_residual': q_abs_max,
        'signed_product_bound': int(product_bound),
    }


def predict_candidates(R: csr_matrix, S: np.ndarray, u: int, candidates: np.ndarray, neighbor_k=NEIGHBOR_K):
    st, en = R.indptr[u], R.indptr[u + 1]
    rated_items = R.indices[st:en]
    ratings = R.data[st:en]
    if len(rated_items) == 0:
        return np.full(len(candidates), np.nan)
    M = S[np.asarray(candidates, dtype=np.int32)][:, rated_items].astype(np.float64, copy=False)
    if neighbor_k is not None and len(rated_items) > neighbor_k:
        # Row-wise keep top-k positive similarities.
        out = np.full(len(candidates), np.nan, dtype=np.float64)
        for row_idx in range(len(candidates)):
            sims = M[row_idx]
            pos_idx = np.flatnonzero(sims > 0)
            if len(pos_idx) == 0:
                continue
            if len(pos_idx) > neighbor_k:
                local = np.argpartition(sims[pos_idx], -neighbor_k)[-neighbor_k:]
                pos_idx = pos_idx[local]
            denom = np.abs(sims[pos_idx]).sum()
            if denom > 0:
                out[row_idx] = float(np.dot(sims[pos_idx], ratings[pos_idx]) / denom)
        return out
    pos = M > 0
    num = (M * pos) @ ratings
    den = np.abs(M * pos).sum(axis=1)
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float64), where=den > 0)


def _maybe_sample_users(df: pd.DataFrame, max_users: Optional[int], seed: int) -> pd.DataFrame:
    if max_users is None:
        return df
    users = df['u'].unique()
    if len(users) <= max_users:
        return df
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(users, size=max_users, replace=False).tolist())
    return df[df['u'].isin(keep)].copy()


def evaluate_rating_prediction(R, S, test: pd.DataFrame, max_users=None, seed=BASE_SEED):
    test = _maybe_sample_users(test, max_users, seed)
    sqe, ae = [], []
    n_valid = 0
    for u, g in test.groupby('u', sort=False):
        cand = g['i'].to_numpy(dtype=np.int32)
        pred = predict_candidates(R, S, int(u), cand)
        true = g['rating'].to_numpy(dtype=np.float64)
        ok = np.isfinite(pred)
        if np.any(ok):
            err = pred[ok] - true[ok]
            sqe.extend((err ** 2).tolist())
            ae.extend(np.abs(err).tolist())
            n_valid += int(ok.sum())
    return {
        'rating_test_interactions': int(len(test)),
        'valid_rating_predictions': n_valid,
        'RMSE': math.sqrt(float(np.mean(sqe))) if sqe else np.nan,
        'MAE': float(np.mean(ae)) if ae else np.nan,
    }


def _ranking_candidates(R: csr_matrix, test_row, n_items: int, neg_samples: int, rng: np.random.Generator):
    u = int(test_row.u)
    target = int(test_row.i)
    st, en = R.indptr[u], R.indptr[u + 1]
    seen = np.zeros(n_items, dtype=bool)
    seen[R.indices[st:en]] = True
    seen[target] = True
    avail = np.flatnonzero(~seen)
    if len(avail) == 0:
        return None
    kneg = min(neg_samples, len(avail))
    neg = rng.choice(avail, size=kneg, replace=False)
    return np.concatenate(([target], neg.astype(np.int32)))


def evaluate_topn(R, S, test: pd.DataFrame, n_items: int, ks=(5, 10), neg_samples=99,
                  max_users=None, seed=BASE_SEED, return_user_rows=False):
    test = _maybe_sample_users(test, max_users, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for row in test.itertuples(index=False):
        cand = _ranking_candidates(R, row, n_items, neg_samples, rng)
        if cand is None:
            continue
        scores = predict_candidates(R, S, int(row.u), cand)
        scores = np.nan_to_num(scores, nan=-1e30)
        order = np.argsort(-scores, kind='mergesort')
        rank = int(np.where(order == 0)[0][0]) + 1
        rec = {'u': int(row.u), 'rank': rank}
        for k in ks:
            rec[f'HR@{k}'] = 1.0 if rank <= k else 0.0
            rec[f'NDCG@{k}'] = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
        rows.append(rec)
    udf = pd.DataFrame(rows)
    result = {'rank_eval_users': int(len(udf))}
    for k in ks:
        result[f'HR@{k}'] = float(udf[f'HR@{k}'].mean()) if len(udf) else np.nan
        result[f'NDCG@{k}'] = float(udf[f'NDCG@{k}'].mean()) if len(udf) else np.nan
    return (result, udf) if return_user_rows else result


def evaluate_candidate_agreement(R, S_plain, S_fp, test: pd.DataFrame, n_items: int,
                                 k=10, neg_samples=99, max_users=2000, seed=BASE_SEED):
    test = _maybe_sample_users(test, max_users, seed)
    rng = np.random.default_rng(seed)
    agreements, pred_abs = [], []
    for row in test.itertuples(index=False):
        cand = _ranking_candidates(R, row, n_items, neg_samples, rng)
        if cand is None:
            continue
        p1 = predict_candidates(R, S_plain, int(row.u), cand)
        p2 = predict_candidates(R, S_fp, int(row.u), cand)
        both = np.isfinite(p1) & np.isfinite(p2)
        if np.any(both):
            pred_abs.extend(np.abs(p1[both] - p2[both]).tolist())
        s1 = np.nan_to_num(p1, nan=-1e30)
        s2 = np.nan_to_num(p2, nan=-1e30)
        top1 = set(np.argsort(-s1, kind='mergesort')[:k].tolist())
        top2 = set(np.argsort(-s2, kind='mergesort')[:k].tolist())
        agreements.append(len(top1 & top2) / k)
    return {
        f'Agreement@{k}': float(np.mean(agreements)) if agreements else np.nan,
        'mean_abs_prediction_difference': float(np.mean(pred_abs)) if pred_abs else np.nan,
        'max_abs_prediction_difference': float(np.max(pred_abs)) if pred_abs else np.nan,
        'equiv_eval_users': len(agreements),
    }


RFC5114_P_HEX = '''
87A8E61D B4B6663C FFBBD19C 65195999 8CEEF608 660DD0F2
5D2CEED4 435E3B00 E00DF8F1 D61957D4 FAF7DF45 61B2AA30
16C3D911 34096FAA 3BF4296D 830E9A7C 209E0C64 97517ABD
5A8A9D30 6BCF67ED 91F9E672 5B4758C0 22E0B1EF 4275BF7B
6C5BFC11 D45F9088 B941F54E B1E59BB8 BC39A0BF 12307F5C
4FDB70C5 81B23F76 B63ACAE1 CAA6B790 2D525267 35488A0E
F13C6D9A 51BFA4AB 3AD83477 96524D8E F6A167B5 A41825D9
67E144E5 14056425 1CCACB83 E6B486F6 B3CA3F79 71506026
C0B857F6 89962856 DED4010A BD0BE621 C3A3960A 54E710C3
75F26375 D7014103 A4B54330 C198AF12 6116D227 6E11715F
693877FA D7EF09CA DB094AE9 1E1A1597
'''
RFC5114_G_HEX = '''
3FB32C9B 73134D0B 2E775066 60EDBD48 4CA7B18F 21EF2054
07F4793A 1A0BA125 10DBC150 77BE463F FF4FED4A AC0BB555
BE3A6C1B 0C6B47B1 BC3773BF 7E8C6F62 901228F8 C28CBB18
A55AE313 41000A65 0196F931 C77A57F2 DDF463E5 E9EC144B
777DE62A AAB8A862 8AC376D2 82D6ED38 64E67982 428EBC83
1D14348F 6F2F9193 B5045AF2 767164E1 DFC967C1 FB3F2E55
A4BD1BFF E83B9C80 D052B985 D182EA0A DB2A3B73 13D3FE14
C8484B1E 052588B9 B7D2BBD2 DF016199 ECD06E15 57CD0915
B3353BBB 64E0EC37 7FD02837 0DF92B52 C7891428 CDC67EB6
184B523D 1DB246C3 2F630784 90F00EF8 D647D148 D4795451
5E2327CF EF98C582 664B4C0F 6CC41659
'''
RFC5114_Q_HEX = '8CF83642 A709A097 B4479976 40129DA2 99B1A47D 1EB3750B A308B0FE 64F5FBD3'


def _hexint(s: str) -> int:
    return int(re.sub(r'\s+', '', s), 16)


P = _hexint(RFC5114_P_HEX)
Q = _hexint(RFC5114_Q_HEX)
G = _hexint(RFC5114_G_HEX)


def k_for_s(s: int) -> int:
    return 0 if s <= 0 else int(math.ceil(0.5 + math.sqrt(2 * s + 0.25)))


def discrete_log_many(targets: Sequence[int], max_value: int, p=P, g=G) -> np.ndarray:
    """Resolve several small-range discrete-log targets with one scan."""
    target_map = {}
    for idx, y in enumerate(targets):
        target_map.setdefault(int(y), []).append(idx)
    ans = np.full(len(targets), -1, dtype=np.int64)
    remaining = len(targets)
    cur = 1
    for x in range(max_value + 1):
        if cur in target_map:
            for idx in target_map[cur]:
                if ans[idx] < 0:
                    ans[idx] = x
                    remaining -= 1
            if remaining == 0:
                break
        cur = (cur * g) % p
    if np.any(ans < 0):
        raise ValueError('At least one discrete log was not found in configured range.')
    return ans


def secure_multi_sum_nonnegative(V: np.ndarray, p=P, q=Q, g=G) -> np.ndarray:
    V = np.asarray(V, dtype=np.int64)
    if V.ndim != 2 or np.any(V < 0):
        raise ValueError('V must be [n_users, s] non-negative integers.')
    n, s = V.shape
    if s == 0:
        return np.array([], dtype=np.int64)
    k = k_for_s(s)
    sysrng = random.SystemRandom()
    prv = [[sysrng.randrange(1, q) for _ in range(k)] for _ in range(n)]
    pub_i = [[pow(g, prv[i][j], p) for j in range(k)] for i in range(n)]
    pub = []
    for j in range(k):
        acc = 1
        for i in range(n):
            acc = (acc * pub_i[i][j]) % p
        pub.append(acc)

    pairs = []
    for t in range(k - 1):
        for h in range(t + 1, k):
            pairs.append((t, h))
            if len(pairs) == s:
                break
        if len(pairs) == s:
            break

    payload = [[0] * s for _ in range(n)]
    for i in range(n):
        for j, (t, h) in enumerate(pairs):
            a = pow(g, int(V[i, j]), p)
            b = pow(pub[t], prv[i][h], p)
            c = pow(pub[h], q - prv[i][t], p)
            payload[i][j] = (a * b % p) * c % p

    K = []
    for j in range(s):
        acc = 1
        for i in range(n):
            acc = (acc * payload[i][j]) % p
        K.append(acc)
    max_sum = int(V.sum(axis=0).max())
    return discrete_log_many(K, max_sum, p, g)


def secure_multi_sum_signed(V: np.ndarray, public_bound: Optional[int] = None) -> np.ndarray:
    """Signed integer sum using a public additive offset before Algorithm 1."""
    V = np.asarray(V, dtype=np.int64)
    if V.ndim != 2:
        raise ValueError('V must be 2D.')
    B = int(np.max(np.abs(V))) if public_bound is None else int(public_bound)
    if B < int(np.max(np.abs(V))):
        raise ValueError('public_bound is smaller than an input magnitude.')
    encoded = V + B
    secure_encoded_sum = secure_multi_sum_nonnegative(encoded)
    return secure_encoded_sum - V.shape[0] * B


def secure_multisum_sanity_checks() -> Dict[str, bool]:
    rng = np.random.default_rng(BASE_SEED)
    V = rng.integers(0, 6, size=(8, 12), dtype=np.int64)
    ok_nonneg = np.array_equal(V.sum(axis=0), secure_multi_sum_nonnegative(V))
    S = rng.integers(-8, 9, size=(8, 10), dtype=np.int64)
    ok_signed = np.array_equal(S.sum(axis=0), secure_multi_sum_signed(S))
    print('Secure multi-sum nonnegative:', 'PASS' if ok_nonneg else 'FAIL')
    print('Secure multi-sum signed-offset:', 'PASS' if ok_signed else 'FAIL')
    return {'nonnegative': ok_nonneg, 'signed_offset': ok_signed}


def real_rating_secure_tiny_validation(df: pd.DataFrame, digits=1, max_users=8, max_items=10, max_pairs=12):
    """Actual Algorithm-1 run on a tiny slice of real ratings using signed fixed-point contributions."""
    small = nested_setting_df(df, max_users, max_items)
    mapped, users, items = remap_ids(small)
    R = build_rating_matrix(mapped, len(users), len(items))
    _, X, _, _ = _centered_sparse(R)
    scale = 10 ** digits
    Xq = X.toarray()
    Xq = np.rint(Xq * scale).astype(np.int64)

    pairs = []
    for j in range(Xq.shape[1]):
        for t in range(j + 1, Xq.shape[1]):
            pairs.append((j, t))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        return {'status': 'SKIP', 'reason': 'not enough item pairs'}

    num_cols, a_cols, b_cols = [], [], []
    for j, t in pairs:
        num_cols.append(Xq[:, j] * Xq[:, t])
        a_cols.append(Xq[:, j] ** 2)
        b_cols.append(Xq[:, t] ** 2)
    Num = np.column_stack(num_cols)
    A1 = np.column_stack(a_cols)
    A2 = np.column_stack(b_cols)

    secure_num = secure_multi_sum_signed(Num)
    secure_a1 = secure_multi_sum_nonnegative(A1)
    secure_a2 = secure_multi_sum_nonnegative(A2)
    plain_num = Num.sum(axis=0)
    plain_a1 = A1.sum(axis=0)
    plain_a2 = A2.sum(axis=0)
    ok = np.array_equal(secure_num, plain_num) and np.array_equal(secure_a1, plain_a1) and np.array_equal(secure_a2, plain_a2)
    den = np.sqrt(secure_a1.astype(float) * secure_a2.astype(float))
    secure_sim = np.divide(secure_num, den, out=np.zeros_like(den), where=den > 0)
    den2 = np.sqrt(plain_a1.astype(float) * plain_a2.astype(float))
    plain_sim = np.divide(plain_num, den2, out=np.zeros_like(den2), where=den2 > 0)
    max_diff = float(np.max(np.abs(secure_sim - plain_sim))) if len(secure_sim) else 0.0
    return {
        'status': 'PASS' if ok else 'FAIL',
        'digits': digits,
        'users': len(users),
        'items': len(items),
        'pairs_tested': len(pairs),
        'max_similarity_difference': max_diff,
    }


def microbenchmark_group_ops(repeats=50) -> Dict[str, float]:
    rng = random.SystemRandom()
    full_exps = [rng.randrange(1, Q) for _ in range(repeats)]
    small_exps = [rng.randrange(0, 5000) for _ in range(repeats)]
    vals = [pow(G, rng.randrange(1, Q), P) for _ in range(repeats + 1)]

    t0 = time.perf_counter()
    for e in full_exps:
        pow(G, e, P)
    full_pow = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for e in small_exps:
        pow(G, e, P)
    small_pow = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    x = 1
    for i in range(repeats):
        x = (vals[i] * vals[i + 1]) % P
    mul = (time.perf_counter() - t0) / repeats
    return {'full_pow_s': full_pow, 'small_pow_s': small_pow, 'mul_mod_s': mul}


def protocol_one_invocation_model(s: int, n_users: int, ops: Dict[str, float]):
    if s <= 0:
        return {k: 0 for k in ['k', 'user_time_s_est', 'server_time_s_est', 'per_user_upload_B',
                               'per_user_download_B', 'server_in_B', 'server_out_B']}
    k = k_for_s(s)
    user_time = k * ops['full_pow_s'] + s * (2 * ops['full_pow_s'] + ops['small_pow_s'])
    server_time = n_users * (k + s) * ops['mul_mod_s']
    per_user_upload = (k + s) * GROUP_ELEMENT_BYTES
    per_user_download = k * GROUP_ELEMENT_BYTES
    return {
        'k': k,
        'user_time_s_est': user_time,
        'server_time_s_est': server_time,
        'per_user_upload_B': per_user_upload,
        'per_user_download_B': per_user_download,
        'server_in_B': n_users * per_user_upload,
        'server_out_B': n_users * per_user_download,
    }


def workload_model(m_items: int, n_users: int, positive_pairs: int, ops: Dict[str, float]):
    Pairs = m_items * (m_items - 1) // 2
    full_s = 3 * Pairs
    priv_s1 = Pairs
    priv_s2 = 2 * positive_pairs
    full = protocol_one_invocation_model(full_s, n_users, ops)
    p1 = protocol_one_invocation_model(priv_s1, n_users, ops)
    p2 = protocol_one_invocation_model(priv_s2, n_users, ops)

    def add(a, b, key):
        return a[key] + b[key]

    model_broadcast_per_user_B = positive_pairs * SIMILARITY_BYTES
    model_broadcast_server_out_B = n_users * model_broadcast_per_user_B
    priv = {
        'user_time_s_est': add(p1, p2, 'user_time_s_est'),
        'server_time_s_est': add(p1, p2, 'server_time_s_est'),
        'per_user_upload_B': add(p1, p2, 'per_user_upload_B'),
        'per_user_download_crypto_B': add(p1, p2, 'per_user_download_B'),
        'server_in_B': add(p1, p2, 'server_in_B'),
        'server_out_crypto_B': add(p1, p2, 'server_out_B'),
        'k_total': p1['k'] + p2['k'],
    }
    secure_values_priv = priv_s1 + priv_s2
    return {
        'all_pairs': Pairs,
        'secure_values_full': full_s,
        'secure_values_priv': secure_values_priv,
        'secure_value_reduction_pct': 100.0 * (1.0 - secure_values_priv / full_s) if full_s else 0.0,
        'full_user_min_est_excl_dlp': full['user_time_s_est'] / 60,
        'priv_user_min_est_excl_dlp': priv['user_time_s_est'] / 60,
        'full_server_min_est_excl_dlp': full['server_time_s_est'] / 60,
        'priv_server_min_est_excl_dlp': priv['server_time_s_est'] / 60,
        'full_per_user_upload_MB': full['per_user_upload_B'] / 1e6,
        'priv_per_user_upload_MB': priv['per_user_upload_B'] / 1e6,
        'full_per_user_download_crypto_MB': full['per_user_download_B'] / 1e6,
        'priv_per_user_download_crypto_MB': priv['per_user_download_crypto_B'] / 1e6,
        'model_broadcast_per_user_MB': model_broadcast_per_user_B / 1e6,
        'full_server_crypto_traffic_GB': (full['server_in_B'] + full['server_out_B']) / 1e9,
        'priv_server_crypto_traffic_GB': (priv['server_in_B'] + priv['server_out_crypto_B']) / 1e9,
        'model_broadcast_server_out_GB': model_broadcast_server_out_B / 1e9,
        'full_k': full['k'],
        'priv_k_total': priv['k_total'],
    }


def dataset_row(name: str, mapped: pd.DataFrame, n_users: int, n_items: int):
    return {
        'dataset': name,
        'users': n_users,
        'items': n_items,
        'ratings': len(mapped),
        'density_pct': 100.0 * len(mapped) / (n_users * n_items) if n_users * n_items else 0.0,
        'rating_min': float(mapped['rating'].min()),
        'rating_max': float(mapped['rating'].max()),
        'rating_mean': float(mapped['rating'].mean()),
    }


def run_item_scalability(selected_data: Dict[str, pd.DataFrame], ops: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in RUN_DATASETS:
        cfg = EXPERIMENTS[name]
        for m in cfg['m_values']:
            print(f'[item scalability] {name}: target n={cfg["n_users"]}, m={m}')
            df = nested_setting_df(selected_data[name], cfg['n_users'], m)
            mapped, users, items = remap_ids(df)
            R = build_rating_matrix(mapped, len(users), len(items))
            _, _, diag = adjusted_cosine_pair_specific(R, positive_only=True)
            model = workload_model(len(items), len(users), diag['positive_numerator_pairs'], ops)
            row = dataset_row(name, mapped, len(users), len(items))
            row.update(diag)
            row.update(model)
            row['target_users'] = cfg['n_users']
            row['target_items'] = m
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / 'item_scalability_partial.csv', index=False)
            del R, mapped
            gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'item_scalability.csv', index=False)
    return out


def user_grid_for_dataset(name: str, target_n: int) -> List[int]:
    base = [500, 1000, 2000, 3000, 5000, 10000]
    vals = [x for x in base if x <= target_n]
    if target_n not in vals:
        vals.append(target_n)
    return sorted(set(vals))


def fixed_m_for_user_scalability(name: str) -> int:
    return {'MovieLens-1M': 1000, 'Netflix Prize': 1000, 'Amazon Book': 600}[name]


def run_user_scalability(selected_data: Dict[str, pd.DataFrame], ops: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in RUN_DATASETS:
        target_n = EXPERIMENTS[name]['n_users']
        m = fixed_m_for_user_scalability(name)
        for n in user_grid_for_dataset(name, target_n):
            print(f'[user scalability] {name}: target n={n}, m={m}')
            df = nested_setting_df(selected_data[name], n, m)
            mapped, users, items = remap_ids(df)
            R = build_rating_matrix(mapped, len(users), len(items))
            _, _, diag = adjusted_cosine_pair_specific(R, positive_only=True)
            model = workload_model(len(items), len(users), diag['positive_numerator_pairs'], ops)
            row = dataset_row(name, mapped, len(users), len(items))
            row.update(diag)
            row.update(model)
            row.update({'target_users': n, 'target_items': m})
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / 'user_scalability_partial.csv', index=False)
            del R, mapped
            gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'user_scalability.csv', index=False)
    return out


def run_recommendation_quality(selected_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rating_rows, rank_rows = [], []
    for name in RUN_DATASETS:
        cfg = EXPERIMENTS[name]
        max_m = max(cfg['m_values'])
        df = nested_setting_df(selected_data[name], cfg['n_users'], max_m)
        mapped, users, items = remap_ids(df)
        n_users, n_items = len(users), len(items)
        for seed in PAPER_SEEDS:
            print(f'[quality] {name}, seed={seed} — rating prediction')
            tr, te = per_user_random_holdout(mapped, RATING_TEST_FRAC, MIN_RATINGS_FOR_RATING_TEST, seed)
            R = build_rating_matrix(tr, n_users, n_items)
            S, _, _ = adjusted_cosine_pair_specific(R, positive_only=True)
            met = evaluate_rating_prediction(R, S, te, MAX_RATING_EVAL_USERS, seed)
            rating_rows.append({'dataset': name, 'method': 'Plain IBCF-ACOS+WS', 'seed': seed,
                                'users': n_users, 'items': n_items, **met})
            # PrivIBCF numerical implementation with signed fixed-point local statistics.
            Sfp, _ = adjusted_cosine_fixedpoint(R, DEFAULT_FIXEDPOINT_D, positive_only=True)
            met_fp = evaluate_rating_prediction(R, Sfp, te, MAX_RATING_EVAL_USERS, seed)
            rating_rows.append({'dataset': name, 'method': f'PrivIBCF-FP(d={DEFAULT_FIXEDPOINT_D})', 'seed': seed,
                                'users': n_users, 'items': n_items, **met_fp})
            pd.DataFrame(rating_rows).to_csv(OUT / 'rating_prediction_seed_results_partial.csv', index=False)
            del Sfp, S, R, tr, te
            gc.collect()

            print(f'[quality] {name}, seed={seed} — Top-N')
            tr, te = positive_leave_one_out(mapped, RELEVANT_THRESHOLD, MIN_RATINGS_FOR_RANK_TEST, seed)
            R = build_rating_matrix(tr, n_users, n_items)
            S, _, _ = adjusted_cosine_pair_specific(R, positive_only=True)
            met = evaluate_topn(R, S, te, n_items, RANK_KS, NEGATIVE_SAMPLES, MAX_RANK_EVAL_USERS, seed)
            rank_rows.append({'dataset': name, 'method': 'Plain IBCF-ACOS+WS', 'seed': seed,
                              'users': n_users, 'items': n_items, **met})
            Sfp, _ = adjusted_cosine_fixedpoint(R, DEFAULT_FIXEDPOINT_D, positive_only=True)
            met_fp = evaluate_topn(R, Sfp, te, n_items, RANK_KS, NEGATIVE_SAMPLES, MAX_RANK_EVAL_USERS, seed)
            rank_rows.append({'dataset': name, 'method': f'PrivIBCF-FP(d={DEFAULT_FIXEDPOINT_D})', 'seed': seed,
                              'users': n_users, 'items': n_items, **met_fp})
            pd.DataFrame(rank_rows).to_csv(OUT / 'topn_seed_results_partial.csv', index=False)
            del Sfp, S, R, tr, te
            gc.collect()
    rating = pd.DataFrame(rating_rows)
    rank = pd.DataFrame(rank_rows)
    rating.to_csv(OUT / 'rating_prediction_seed_results.csv', index=False)
    rank.to_csv(OUT / 'topn_seed_results.csv', index=False)
    return rating, rank


def summarize_seeds(df: pd.DataFrame, metric_cols: Sequence[str], group_cols=('dataset', 'method')) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row['n_seeds'] = g['seed'].nunique() if 'seed' in g else len(g)
        for c in metric_cols:
            x = pd.to_numeric(g[c], errors='coerce').dropna()
            row[f'{c}_mean'] = float(x.mean()) if len(x) else np.nan
            row[f'{c}_std'] = float(x.std(ddof=1)) if len(x) > 1 else 0.0 if len(x) == 1 else np.nan
            row[f'{c}_ci95'] = 1.96 * row[f'{c}_std'] / math.sqrt(len(x)) if len(x) > 1 else 0.0 if len(x) == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_fixedpoint_and_equivalence(selected_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fp_rows, eq_rows = [], []
    for name in RUN_DATASETS:
        cfg = EXPERIMENTS[name]
        df = nested_setting_df(selected_data[name], cfg['n_users'], max(cfg['m_values']))
        mapped, users, items = remap_ids(df)
        n_users, n_items = len(users), len(items)
        seed = BASE_SEED
        tr, te = positive_leave_one_out(mapped, RELEVANT_THRESHOLD, MIN_RATINGS_FOR_RANK_TEST, seed)
        R = build_rating_matrix(tr, n_users, n_items)
        S_plain, _, _, X, B, N_plain, A_plain = adjusted_cosine_pair_specific(
            R, positive_only=True, return_components=True
        )
        upper = np.triu_indices(n_items, k=1)
        for d in FIXEDPOINT_DIGITS:
            print(f'[fixed point] {name}, d={d}')
            S_fp, fpdiag = adjusted_cosine_fixedpoint(R, d, positive_only=True)
            diff = np.abs(S_plain[upper].astype(np.float64) - S_fp[upper].astype(np.float64))
            nz_mask = (S_plain[upper] != 0) | (S_fp[upper] != 0)
            simdiff = diff[nz_mask] if np.any(nz_mask) else diff
            eq = evaluate_candidate_agreement(
                R, S_plain, S_fp, te, n_items, k=10, neg_samples=NEGATIVE_SAMPLES,
                max_users=MAX_EQUIV_EVAL_USERS, seed=seed,
            )
            # DLP range upper bound for signed numerator after offset: each encoded input <= 2B.
            # B is derived from the actually observed max quantized residual.
            Bprod = int(fpdiag['signed_product_bound'])
            dlp_num_encoded_upper = int(2 * Bprod * n_users)
            dlp_square_upper = int(Bprod * n_users)
            row = {
                'dataset': name,
                'digits': d,
                'users': n_users,
                'items': n_items,
                'mean_abs_similarity_difference': float(np.mean(simdiff)) if len(simdiff) else 0.0,
                'max_abs_similarity_difference': float(np.max(simdiff)) if len(simdiff) else 0.0,
                'dlp_numerator_encoded_upper_bound': dlp_num_encoded_upper,
                'dlp_square_upper_bound': dlp_square_upper,
                **fpdiag,
                **eq,
            }
            fp_rows.append(row)
            pd.DataFrame(fp_rows).to_csv(OUT / 'fixedpoint_sensitivity_partial.csv', index=False)
            if d == DEFAULT_FIXEDPOINT_D:
                eq_rows.append(row.copy())
            del S_fp
            gc.collect()
        del S_plain, R, tr, te, X, B, N_plain, A_plain
        gc.collect()
    fp = pd.DataFrame(fp_rows)
    eq = pd.DataFrame(eq_rows)
    fp.to_csv(OUT / 'fixedpoint_sensitivity.csv', index=False)
    eq.to_csv(OUT / 'equivalence_default_fixedpoint.csv', index=False)
    return fp, eq


def thin_observed_interactions(mapped: pd.DataFrame, keep_rate: float, seed: int) -> pd.DataFrame:
    if keep_rate >= 1.0:
        return mapped.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(mapped)) < keep_rate
    thin = mapped.loc[mask].copy()
    # Keep only users/items with at least one observation; remapping happens afterward.
    return thin


def run_sparsity_sensitivity(selected_data: Dict[str, pd.DataFrame], ops: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in RUN_DATASETS:
        cfg = EXPERIMENTS[name]
        # Use a moderate setting to make controlled thinning inexpensive and comparable.
        m = min(max(cfg['m_values']), 1000)
        df = nested_setting_df(selected_data[name], cfg['n_users'], m)
        base, _, _ = remap_ids(df)
        for keep in SPARSITY_KEEP_RATES:
            print(f'[sparsity] {name}, keep={keep}')
            thin = thin_observed_interactions(base, keep, BASE_SEED + int(keep * 100))
            # Rebuild IDs because thinning can remove all observations for some users/items.
            thin0 = thin[['user', 'item', 'rating']].copy()
            mapped, users, items = remap_ids(thin0)
            R = build_rating_matrix(mapped, len(users), len(items))
            _, _, diag = adjusted_cosine_pair_specific(R, positive_only=True)
            model = workload_model(len(items), len(users), diag['positive_numerator_pairs'], ops)
            row = dataset_row(name, mapped, len(users), len(items))
            row.update(diag)
            row.update(model)
            row['keep_rate'] = keep
            row['removed_observed_pct'] = 100 * (1 - keep)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / 'sparsity_sensitivity_partial.csv', index=False)
            del R, mapped, thin
            gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'sparsity_sensitivity.csv', index=False)
    return out


def run_phase2_local_runtime(selected_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in RUN_DATASETS:
        cfg = EXPERIMENTS[name]
        df = nested_setting_df(selected_data[name], cfg['n_users'], max(cfg['m_values']))
        mapped, users, items = remap_ids(df)
        n_users, n_items = len(users), len(items)
        tr, te = positive_leave_one_out(mapped, RELEVANT_THRESHOLD, MIN_RATINGS_FOR_RANK_TEST, BASE_SEED)
        R = build_rating_matrix(tr, n_users, n_items)
        S, _, _ = adjusted_cosine_pair_specific(R, positive_only=True)
        eligible = te['u'].unique()
        rng = np.random.default_rng(BASE_SEED)
        if len(eligible) > PHASE2_SAMPLE_USERS:
            eligible = rng.choice(eligible, PHASE2_SAMPLE_USERS, replace=False)
        all_items = np.arange(n_items, dtype=np.int32)
        for c in PHASE2_CANDIDATES:
            times = []
            for u in eligible:
                st, en = R.indptr[int(u)], R.indptr[int(u)+1]
                seen = np.zeros(n_items, dtype=bool)
                seen[R.indices[st:en]] = True
                avail = np.flatnonzero(~seen)
                if len(avail) == 0:
                    continue
                cand = rng.choice(avail, size=min(c, len(avail)), replace=False)
                t0 = time.perf_counter()
                _ = predict_candidates(R, S, int(u), cand)
                times.append(time.perf_counter() - t0)
            rows.append({
                'dataset': name,
                'method': 'PrivIBCF local Weighted-Sum prediction',
                'candidate_items': c,
                'sample_users': len(times),
                'mean_user_seconds': float(np.mean(times)) if times else np.nan,
                'std_user_seconds': float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                'median_user_seconds': float(np.median(times)) if times else np.nan,
                'p95_user_seconds': float(np.quantile(times, 0.95)) if times else np.nan,
                'server_seconds': 0.0,
            })
            pd.DataFrame(rows).to_csv(OUT / 'phase2_local_runtime_partial.csv', index=False)
        del S, R, tr, te, mapped
        gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'phase2_local_runtime.csv', index=False)
    return out


def build_paper_tables(item_scal: pd.DataFrame, rating_seed: pd.DataFrame, rank_seed: pd.DataFrame,
                       fp: pd.DataFrame, eq: pd.DataFrame, sparsity: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables = {}
    # Dataset statistics at the largest requested item setting.
    idx = item_scal.groupby('dataset')['target_items'].idxmax()
    tables['dataset_statistics'] = item_scal.loc[idx, [
        'dataset', 'target_users', 'actual_users' if 'actual_users' in item_scal.columns else 'users',
        'target_items', 'items', 'ratings', 'density_pct', 'rating_min', 'rating_max'
    ]].copy() if 'actual_users' in item_scal.columns else item_scal.loc[idx, [
        'dataset', 'target_users', 'users', 'target_items', 'items', 'ratings', 'density_pct', 'rating_min', 'rating_max'
    ]].copy()

    tables['pruning_effectiveness'] = item_scal[[
        'dataset', 'target_items', 'items', 'all_item_pairs', 'positive_numerator_pairs',
        'positive_pair_ratio', 'secure_values_full', 'secure_values_priv', 'secure_value_reduction_pct'
    ]].copy()

    tables['communication_max_setting'] = item_scal.loc[idx, [
        'dataset', 'items', 'full_per_user_upload_MB', 'priv_per_user_upload_MB',
        'full_per_user_download_crypto_MB', 'priv_per_user_download_crypto_MB',
        'model_broadcast_per_user_MB', 'full_server_crypto_traffic_GB',
        'priv_server_crypto_traffic_GB', 'model_broadcast_server_out_GB'
    ]].copy()

    tables['ablation_max_setting'] = item_scal.loc[idx, [
        'dataset', 'items', 'secure_values_full', 'secure_values_priv', 'secure_value_reduction_pct',
        'full_user_min_est_excl_dlp', 'priv_user_min_est_excl_dlp',
        'full_server_min_est_excl_dlp', 'priv_server_min_est_excl_dlp'
    ]].copy()

    rating_summary = summarize_seeds(rating_seed, ['RMSE', 'MAE'])
    rank_summary = summarize_seeds(rank_seed, [f'HR@{k}' for k in RANK_KS] + [f'NDCG@{k}' for k in RANK_KS])
    tables['rating_prediction_summary'] = rating_summary
    tables['topn_summary'] = rank_summary
    tables['fixedpoint_sensitivity'] = fp.copy()
    tables['equivalence'] = eq.copy()
    tables['sparsity'] = sparsity.copy()

    for name, df in tables.items():
        df.to_csv(OUT / f'table_{name}.csv', index=False)
    return tables


def _savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', dpi=240, bbox_inches='tight')
    plt.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.show()


def plot_item_scalability(df: pd.DataFrame):
    for dataset, g in df.groupby('dataset', sort=False):
        g = g.sort_values('items')
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(g['items'], g['secure_value_reduction_pct'], marker='o')
        plt.xlabel('Number of items')
        plt.ylabel('Secure workload reduction (%)')
        plt.title(f'{dataset}: early-pruning effectiveness')
        plt.grid(alpha=.25)
        _savefig(dataset.replace(' ', '_') + '_pruning')

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(g['items'], g['full_user_min_est_excl_dlp'], marker='o', label='Full-ACOS estimate')
        plt.plot(g['items'], g['priv_user_min_est_excl_dlp'], marker='s', label='PrivIBCF estimate')
        plt.xlabel('Number of items')
        plt.ylabel('Estimated user crypto time (min, excl. DLP)')
        plt.title(f'{dataset}: user-side operation model')
        plt.grid(alpha=.25)
        plt.legend()
        _savefig(dataset.replace(' ', '_') + '_user_crypto_est')

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(g['items'], g['full_server_min_est_excl_dlp'], marker='o', label='Full-ACOS estimate')
        plt.plot(g['items'], g['priv_server_min_est_excl_dlp'], marker='s', label='PrivIBCF estimate')
        plt.xlabel('Number of items')
        plt.ylabel('Estimated server crypto time (min, excl. DLP)')
        plt.title(f'{dataset}: server-side operation model')
        plt.grid(alpha=.25)
        plt.legend()
        _savefig(dataset.replace(' ', '_') + '_server_crypto_est')


def plot_user_scalability(df: pd.DataFrame):
    for dataset, g in df.groupby('dataset', sort=False):
        g = g.sort_values('users')
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(g['users'], g['priv_server_min_est_excl_dlp'], marker='o')
        plt.xlabel('Number of users')
        plt.ylabel('Estimated server crypto time (min, excl. DLP)')
        plt.title(f'{dataset}: scalability with users')
        plt.grid(alpha=.25)
        _savefig(dataset.replace(' ', '_') + '_user_scalability')


def plot_fixedpoint(df: pd.DataFrame):
    for dataset, g in df.groupby('dataset', sort=False):
        g = g.sort_values('digits')
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(g['digits'], g['max_abs_similarity_difference'], marker='o', label='Max |Δ ACOS|')
        plt.plot(g['digits'], g['mean_abs_similarity_difference'], marker='s', label='Mean |Δ ACOS|')
        plt.yscale('log')
        plt.xlabel('Fixed-point decimal digits')
        plt.ylabel('Similarity error')
        plt.title(f'{dataset}: fixed-point sensitivity')
        plt.grid(alpha=.25)
        plt.legend()
        _savefig(dataset.replace(' ', '_') + '_fixedpoint')


def plot_sparsity(df: pd.DataFrame):
    plt.figure(figsize=(6.2, 4.2))
    for dataset, g in df.groupby('dataset', sort=False):
        g = g.sort_values('density_pct')
        plt.plot(g['density_pct'], g['secure_value_reduction_pct'], marker='o', label=dataset)
    plt.xlabel('Observed matrix density (%)')
    plt.ylabel('Secure workload reduction (%)')
    plt.title('Impact of sparsity on early pruning')
    plt.grid(alpha=.25)
    plt.legend()
    _savefig('sparsity_vs_pruning')


def plot_phase2(df: pd.DataFrame):
    plt.figure(figsize=(6.2, 4.2))
    for dataset, g in df.groupby('dataset', sort=False):
        plt.plot(g['candidate_items'], 1000 * g['mean_user_seconds'], marker='o', label=dataset)
    plt.xlabel('Candidate items')
    plt.ylabel('Mean local prediction time per user (ms)')
    plt.title('PrivIBCF Phase-2 local prediction cost')
    plt.grid(alpha=.25)
    plt.legend()
    _savefig('phase2_local_runtime')


def fmt_mean_std(mean, std, digits=4):
    if pd.isna(mean):
        return 'TBD'
    return f'{mean:.{digits}f} $\\pm$ {std:.{digits}f}'


def export_latex_tables(tables: Dict[str, pd.DataFrame]) -> Path:
    """Write compact LaTeX tabular blocks with the actual computed numbers."""
    p = OUT / 'latex_tables_generated.tex'
    lines = ['% Auto-generated by privibcf_full_paper_experiments.py', '']

    # Pruning table.
    d = tables['pruning_effectiveness']
    lines += [r'\begin{table}[t]', r'\centering',
              r'\caption{Effectiveness of early elimination of non-positive item pairs.}',
              r'\label{tab:pruning_effectiveness_generated}',
              r'\resizebox{\linewidth}{!}{%',
              r'\begin{tabular}{lrrrrr}', r'\toprule',
              r'Dataset & $m$ & $P$ & $P_{+}$ & $P_{+}/P$ (\%) & Reduction (\%) \\', r'\midrule']
    for r in d.itertuples(index=False):
        lines.append(f'{r.dataset} & {int(r.items)} & {int(r.all_item_pairs):,} & '
                     f'{int(r.positive_numerator_pairs):,} & {100*r.positive_pair_ratio:.2f} & '
                     f'{r.secure_value_reduction_pct:.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']

    # Rating table.
    rsum = tables['rating_prediction_summary']
    lines += [r'\begin{table}[t]', r'\centering', r'\caption{Rating-prediction performance over multiple seeds.}',
              r'\label{tab:rating_generated}', r'\resizebox{\linewidth}{!}{%',
              r'\begin{tabular}{llcc}', r'\toprule',
              r'Dataset & Method & RMSE & MAE \\', r'\midrule']
    for _, r in rsum.iterrows():
        lines.append(f"{r['dataset']} & {r['method']} & "
                     f"{fmt_mean_std(r['RMSE_mean'], r['RMSE_std'])} & "
                     f"{fmt_mean_std(r['MAE_mean'], r['MAE_std'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']

    # Top-N table.
    tsum = tables['topn_summary']
    lines += [r'\begin{table}[t]', r'\centering', r'\caption{Top-$N$ recommendation performance over multiple seeds.}',
              r'\label{tab:topn_generated}', r'\resizebox{\linewidth}{!}{%',
              r'\begin{tabular}{llcccc}', r'\toprule',
              r'Dataset & Method & HR@5 & HR@10 & NDCG@5 & NDCG@10 \\', r'\midrule']
    for _, r in tsum.iterrows():
        lines.append(
            f"{r['dataset']} & {r['method']} & "
            f"{fmt_mean_std(r['HR@5_mean'], r['HR@5_std'])} & "
            f"{fmt_mean_std(r['HR@10_mean'], r['HR@10_std'])} & "
            f"{fmt_mean_std(r['NDCG@5_mean'], r['NDCG@5_std'])} & "
            f"{fmt_mean_std(r['NDCG@10_mean'], r['NDCG@10_std'])} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']

    p.write_text('\n'.join(lines), encoding='utf-8')
    return p


def export_run_manifest(ops: Dict[str, float], secure_checks: Dict[str, bool], tiny_checks: Dict[str, dict]):
    manifest = {
        'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'paper_seeds': PAPER_SEEDS,
        'configuration': {
            'rating_test_frac': RATING_TEST_FRAC,
            'relevant_threshold': RELEVANT_THRESHOLD,
            'negative_samples': NEGATIVE_SAMPLES,
            'neighbor_k': NEIGHBOR_K,
            'rank_ks': RANK_KS,
            'fixedpoint_digits': FIXEDPOINT_DIGITS,
            'default_fixedpoint_d': DEFAULT_FIXEDPOINT_D,
            'netflix_use_all_files': NETFLIX_USE_ALL_FILES,
        },
        'group_microbenchmark': ops,
        'secure_checks': secure_checks,
        'tiny_real_data_secure_checks': tiny_checks,
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def run_full_paper_suite(selected_data: Optional[Dict[str, pd.DataFrame]] = None,
                         run_quality=True, run_fixedpoint=True, run_user_scalability_flag=True,
                         run_sparsity=True, run_phase2=True, make_plots=True):
    print('ROOT =', ROOT)
    if selected_data is None:
        selected_data = prepare_all_selected()

    print('\nSelected data:')
    for name, df in selected_data.items():
        print(f'  {name}: rows={len(df):,}, users={df.user.nunique():,}, items={df.item.nunique():,}')

    print('\nBenchmarking group operations...')
    ops = microbenchmark_group_ops(repeats=40)
    print(ops)

    print('\nChecking secure multi-sum implementation...')
    secure_checks = secure_multisum_sanity_checks()
    tiny_checks = {}
    for name in RUN_DATASETS:
        try:
            tiny_checks[name] = real_rating_secure_tiny_validation(selected_data[name], digits=1)
        except Exception as e:
            tiny_checks[name] = {'status': 'ERROR', 'message': repr(e)}
        print(name, tiny_checks[name])

    print('\nItem scalability and pruning')
    item_scal = run_item_scalability(selected_data, ops)

    if run_user_scalability_flag:
        print('\nUser scalability')
        user_scal = run_user_scalability(selected_data, ops)
    else:
        user_scal = pd.DataFrame()

    if run_quality:
        print('\nRecommendation quality')
        rating_seed, rank_seed = run_recommendation_quality(selected_data)
    else:
        rating_seed = pd.DataFrame()
        rank_seed = pd.DataFrame()

    if run_fixedpoint:
        print('\nFixed-point sensitivity')
        fp, eq = run_fixedpoint_and_equivalence(selected_data)
    else:
        fp, eq = pd.DataFrame(), pd.DataFrame()

    if run_sparsity:
        print('\nSparsity sensitivity')
        sparsity = run_sparsity_sensitivity(selected_data, ops)
    else:
        sparsity = pd.DataFrame()

    if run_phase2:
        print('\nPhase-2 local runtime')
        phase2 = run_phase2_local_runtime(selected_data)
    else:
        phase2 = pd.DataFrame()

    print('\nExporting tables and figures')
    if not rating_seed.empty and not rank_seed.empty and not fp.empty and not eq.empty and not sparsity.empty:
        tables = build_paper_tables(item_scal, rating_seed, rank_seed, fp, eq, sparsity)
        try:
            latex_path = export_latex_tables(tables)
            print('Generated LaTeX:', latex_path)
        except Exception as e:
            print('LaTeX export warning:', repr(e))
            latex_path = None
    else:
        tables, latex_path = {}, None

    if make_plots:
        plot_item_scalability(item_scal)
        if not user_scal.empty:
            plot_user_scalability(user_scal)
        if not fp.empty:
            plot_fixedpoint(fp)
        if not sparsity.empty:
            plot_sparsity(sparsity)
        if not phase2.empty:
            plot_phase2(phase2)

    export_run_manifest(ops, secure_checks, tiny_checks)

    # Save an index of generated CSV files.
    index_rows = []
    for fp_name in sorted(OUT.glob('*.csv')):
        index_rows.append({'file': fp_name.name, 'path': str(fp_name)})
    pd.DataFrame(index_rows).to_csv(OUT / 'result_files_index.csv', index=False)

    print('\nResults directory:', OUT)
    print('Key outputs:')
    for fn in [
        'item_scalability.csv', 'user_scalability.csv', 'rating_prediction_seed_results.csv',
        'topn_seed_results.csv', 'fixedpoint_sensitivity.csv', 'equivalence_default_fixedpoint.csv',
        'sparsity_sensitivity.csv', 'phase2_local_runtime.csv', 'run_manifest.json',
        'latex_tables_generated.tex'
    ]:
        p = OUT / fn
        if p.exists():
            print(' -', p)

    return {
        'selected_data': selected_data,
        'ops': ops,
        'item_scalability': item_scal,
        'user_scalability': user_scal,
        'rating_seed': rating_seed,
        'rank_seed': rank_seed,
        'fixedpoint': fp,
        'equivalence': eq,
        'sparsity': sparsity,
        'phase2': phase2,
        'tables': tables,
        'latex_path': latex_path,
        'out_dir': OUT,
    }


MANUSCRIPT_REPORTED_PHASE1 = {
    'MovieLens-1M': {
        'items': [500, 1000, 1500, 2000],
        'Van_user_min': [6.69, 26.79, 60.30, 107.21],
        'Dung_user_min': [5.08, 20.13, 45.33, 84.02],
        'PrivIBCF_user_min': [3.74, 14.88, 33.23, 62.54],
        'Van_server_min': [9.98, 39.95, 89.91, 156.87],
        'Dung_server_min': [8.35, 31.69, 70.03, 123.26],
        'PrivIBCF_server_min': [6.35, 23.70, 52.05, 91.39],
    },
    'Netflix Prize': {
        'items': [500, 1000, 1500, 2000],
        'Van_user_min': [6.69, 26.79, 60.30, 107.21],
        'Dung_user_min': [5.03, 20.25, 45.33, 80.58],
        'PrivIBCF_user_min': [3.41, 13.45, 30.38, 53.73],
        'Van_server_min': [16.63, 66.59, 149.87, 266.48],
        'Dung_server_min': [13.92, 52.83, 116.73, 205.63],
        'PrivIBCF_server_min': [9.76, 36.18, 79.26, 139.01],
    },
    'Amazon Book': {
        'items': [200, 400, 600, 800, 1000],
        'Van_user_min': [1.07, 4.28, 9.64, 17.14, 26.79],
        'Dung_user_min': [0.81, 3.22, 7.25, 12.89, 20.14],
        'PrivIBCF_user_min': [0.54, 2.15, 4.84, 8.60, 13.43],
        'Van_server_min': [5.31, 21.28, 47.92, 85.22, 133.18],
        'Dung_server_min': [5.14, 18.27, 39.40, 68.53, 105.66],
        'PrivIBCF_server_min': [3.81, 12.95, 27.42, 47.23, 72.37],
    },
}


def export_manuscript_reported_baselines() -> pd.DataFrame:
    """Export the Phase-1 runtime values reported in the manuscript."""
    rows = []
    for dataset, d in MANUSCRIPT_REPORTED_PHASE1.items():
        for idx, m in enumerate(d['items']):
            for method in ['Van', 'Dung', 'PrivIBCF']:
                rows.append({
                    'dataset': dataset,
                    'items': m,
                    'method': method,
                    'user_minutes': d[f'{method}_user_min'][idx],
                    'server_minutes': d[f'{method}_server_min'][idx],
                    'source': 'manuscript-reported; not measured in this implementation',
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'manuscript_reported_phase1_baselines.csv', index=False)
    return out


if __name__ == '__main__':
    selected = prepare_all_selected()
    export_manuscript_reported_baselines()
    run_full_paper_suite(selected)
