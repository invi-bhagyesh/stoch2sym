import numpy as np
import itertools
from pathlib import Path
from tqdm import tqdm
from .gillespie import SIRParams, simulate_ensemble


def generate_dataset(
    betas=np.linspace(0.1, 1.0, 10),
    gammas=np.linspace(0.05, 0.5, 10),
    Ns=(500, 1000, 5000, 10000),
    I0s=(1, 5, 10),
    M=100,
    t_max=100.0,
    dt=0.1,
    extinction_threshold=0.5,
    save_dir="data",
    seed=42,
):
    """Generate the full stochastic SIR dataset.

    Args:
        extinction_threshold: discard combinations where > this fraction
            of realizations show stochastic extinction.

    Returns:
        Dictionary with all mean trajectories, parameters, and metadata.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(betas, gammas, Ns, I0s))
    print(f"Total parameter combinations: {len(combos)}")
    print(f"Realizations per combination: {M}")
    print(f"Total trajectories: {len(combos) * M}")

    all_means_s, all_means_i, all_means_r = [], [], []
    all_stds_s, all_stds_i, all_stds_r = [], [], []
    all_params = []
    t_grid = None
    n_filtered = 0

    rng_base = np.random.default_rng(seed)

    for beta, gamma, N, I0 in tqdm(combos, desc="Simulating"):
        params = SIRParams(beta=beta, gamma=gamma, N=N, I0=I0)
        combo_seed = rng_base.integers(0, 2**31)

        result = simulate_ensemble(params, M=M, t_max=t_max, dt=dt, seed=combo_seed)

        if t_grid is None:
            t_grid = result["t_grid"]

        if result["n_extinct"] / M > extinction_threshold:
            n_filtered += 1
            continue

        all_means_s.append(result["mean_s"])
        all_means_i.append(result["mean_i"])
        all_means_r.append(result["mean_r"])
        all_stds_s.append(result["std_s"])
        all_stds_i.append(result["std_i"])
        all_stds_r.append(result["std_r"])
        all_params.append([beta, gamma, N, I0])

    print(f"\nKept {len(all_params)} / {len(combos)} combinations "
          f"({n_filtered} filtered for extinction)")

    data = {
        "t_grid": t_grid,
        "mean_s": np.array(all_means_s),
        "mean_i": np.array(all_means_i),
        "mean_r": np.array(all_means_r),
        "std_s": np.array(all_stds_s),
        "std_i": np.array(all_stds_i),
        "std_r": np.array(all_stds_r),
        "params": np.array(all_params),
    }

    out_path = save_dir / "sir_dataset.npz"
    np.savez_compressed(out_path, **data)
    print(f"Saved dataset to {out_path} ({len(all_params)} trajectories)")

    return data


def load_dataset(path="data/sir_dataset.npz"):
    data = dict(np.load(path))
    return data


def train_val_test_split(data, seed=42):
    """Split dataset 80/10/10."""
    n = len(data["params"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.9 * n)

    def subset(indices):
        return {
            "t_grid": data["t_grid"],
            "mean_s": data["mean_s"][indices],
            "mean_i": data["mean_i"][indices],
            "mean_r": data["mean_r"][indices],
            "params": data["params"][indices],
        }

    return subset(idx[:n_train]), subset(idx[n_train:n_val]), subset(idx[n_val:])
