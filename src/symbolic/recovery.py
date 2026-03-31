import numpy as np
import torch


def extract_derivatives(model, n_points=10000, s_range=(0.0, 1.0),
                        i_range=(0.0, 0.5), beta_range=(0.1, 1.0),
                        gamma_range=(0.05, 0.5), device="cpu", seed=42):
    """Sample state-parameter points and extract learned derivatives via autograd.

    Returns:
        states: (N, 3) -- [s, i, r]
        params: (N, 2) -- [beta, gamma]
        derivs: (N, 3) -- [ds/dt, di/dt, dr/dt]
    """
    rng = np.random.default_rng(seed)

    s = rng.uniform(*s_range, size=n_points)
    i = rng.uniform(*i_range, size=n_points)
    r = 1.0 - s - i
    r = np.clip(r, 0, 1)
    # filter out invalid points
    valid = (r >= 0) & (s + i <= 1.0)
    s, i, r = s[valid], i[valid], r[valid]

    beta = rng.uniform(*beta_range, size=len(s))
    gamma = rng.uniform(*gamma_range, size=len(s))

    states = torch.tensor(np.stack([s, i, r], axis=1), dtype=torch.float32, device=device)
    params = torch.tensor(np.stack([beta, gamma], axis=1), dtype=torch.float32, device=device)

    model.eval()
    derivs = model.get_derivatives(states, params)

    return (states.cpu().numpy(), params.cpu().numpy(), derivs.cpu().numpy())


def run_sindy(states, params, derivs, threshold=0.1):
    """Apply SINDy to recover symbolic equations.

    Uses PySINDy with a custom library of physics-motivated terms.
    """
    import pysindy as ps

    s, i, r = states[:, 0], states[:, 1], states[:, 2]
    beta, gamma = params[:, 0], params[:, 1]

    # build feature matrix with physics-motivated terms
    features = np.column_stack([
        np.ones(len(s)),        # 1
        s, i, r,                # linear
        s * i,                  # s*i (key SIR term)
        s * r, i * r,           # other quadratic
        s ** 2, i ** 2, r ** 2, # squares
        beta * s * i,           # beta * s * i (target term for ds/dt)
        gamma * i,              # gamma * i (target term for dr/dt)
        beta * s, beta * i,     # beta-coupled
        gamma * s, gamma * r,   # gamma-coupled
    ])

    feature_names = [
        "1", "s", "i", "r",
        "s*i", "s*r", "i*r",
        "s^2", "i^2", "r^2",
        "beta*s*i", "gamma*i",
        "beta*s", "beta*i",
        "gamma*s", "gamma*r",
    ]

    results = {}
    target_names = ["ds/dt", "di/dt", "dr/dt"]

    for col, name in enumerate(target_names):
        y = derivs[:, col]

        # STLS: sequential thresholded least squares
        coeffs = _stls(features, y, threshold=threshold)

        terms = []
        for j, c in enumerate(coeffs):
            if abs(c) > 1e-6:
                terms.append(f"{c:+.4f} * {feature_names[j]}")

        equation = " ".join(terms) if terms else "0"
        results[name] = {
            "equation": equation,
            "coefficients": dict(zip(feature_names, coeffs)),
            "nonzero_terms": len(terms),
        }

    return results


def _stls(X, y, threshold=0.1, max_iter=20):
    """Sequential Thresholded Least Squares."""
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(coeffs) < threshold
        coeffs[small] = 0
        big = ~small
        if big.sum() == 0:
            break
        coeffs[big] = np.linalg.lstsq(X[:, big], y, rcond=None)[0]
    return coeffs


def run_pysr(states, params, derivs, compartment="ds/dt",
             niterations=100, binary_operators=("+", "-", "*", "/"),
             maxsize=15, populations=30):
    """Apply PySR symbolic regression to one compartment."""
    from pysr import PySRRegressor

    s, i, r = states[:, 0], states[:, 1], states[:, 2]
    beta, gamma = params[:, 0], params[:, 1]

    X = np.column_stack([s, i, r, beta, gamma])
    col_map = {"ds/dt": 0, "di/dt": 1, "dr/dt": 2}
    y = derivs[:, col_map[compartment]]

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=list(binary_operators),
        unary_operators=[],
        maxsize=maxsize,
        populations=populations,
        variable_names=["s", "i", "r", "beta", "gamma"],
        progress=True,
        temp_equation_file=True,
    )

    model.fit(X, y)
    return model
