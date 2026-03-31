import numpy as np
from dataclasses import dataclass


@dataclass
class SIRParams:
    beta: float
    gamma: float
    N: int
    I0: int = 1
    S0: int = None

    def __post_init__(self):
        if self.S0 is None:
            self.S0 = self.N - self.I0

    @property
    def R0(self):
        return self.beta / self.gamma


def gillespie_sir(params: SIRParams, t_max: float = 100.0, rng=None):
    """Run one stochastic SIR trajectory using the Gillespie SSA.

    Returns arrays of (times, S, I, R) for each event.
    """
    if rng is None:
        rng = np.random.default_rng()

    S, I, R = params.S0, params.I0, 0
    N = params.N

    times = [0.0]
    S_traj = [S]
    I_traj = [I]
    R_traj = [R]

    t = 0.0
    while t < t_max and I > 0:
        rate_infect = params.beta * S * I / N
        rate_recover = params.gamma * I
        total_rate = rate_infect + rate_recover

        if total_rate == 0:
            break

        dt = rng.exponential(1.0 / total_rate)
        t += dt

        if t > t_max:
            break

        if rng.random() < rate_infect / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1

        times.append(t)
        S_traj.append(S)
        I_traj.append(I)
        R_traj.append(R)

    return np.array(times), np.array(S_traj), np.array(I_traj), np.array(R_traj)


def interpolate_trajectory(times, S, I, R, t_grid):
    """Interpolate a Gillespie trajectory (piecewise constant) onto a regular grid."""
    idx = np.searchsorted(times, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    return S[idx], I[idx], R[idx]


def simulate_ensemble(params: SIRParams, M: int = 100, t_max: float = 100.0,
                      dt: float = 0.1, seed: int = 42):
    """Simulate M stochastic realizations and compute ensemble statistics.

    Returns:
        t_grid: common time grid
        mean_s, mean_i, mean_r: ensemble means (fractions, normalized by N)
        std_s, std_i, std_r: ensemble standard deviations
        n_extinct: number of realizations where epidemic died early
    """
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0, t_max + dt, dt)
    N = params.N

    all_s = np.zeros((M, len(t_grid)))
    all_i = np.zeros((M, len(t_grid)))
    all_r = np.zeros((M, len(t_grid)))

    n_extinct = 0
    for m in range(M):
        times, S, I, R = gillespie_sir(params, t_max=t_max, rng=rng)
        s_interp, i_interp, r_interp = interpolate_trajectory(times, S, I, R, t_grid)
        all_s[m] = s_interp / N
        all_i[m] = i_interp / N
        all_r[m] = r_interp / N

        if np.max(i_interp) < 0.01 * N:
            n_extinct += 1

    return {
        "t_grid": t_grid,
        "mean_s": all_s.mean(axis=0),
        "mean_i": all_i.mean(axis=0),
        "mean_r": all_r.mean(axis=0),
        "std_s": all_s.std(axis=0),
        "std_i": all_i.std(axis=0),
        "std_r": all_r.std(axis=0),
        "n_extinct": n_extinct,
        "all_s": all_s,
        "all_i": all_i,
        "all_r": all_r,
    }


def deterministic_sir(params: SIRParams, t_max: float = 100.0, dt: float = 0.1):
    """Solve the deterministic SIR ODE using scipy for validation."""
    from scipy.integrate import solve_ivp

    N = params.N
    s0, i0, r0 = params.S0 / N, params.I0 / N, 0.0

    def rhs(t, y):
        s, i, r = y
        dsdt = -params.beta * s * i
        didt = params.beta * s * i - params.gamma * i
        drdt = params.gamma * i
        return [dsdt, didt, drdt]

    t_grid = np.arange(0, t_max + dt, dt)
    sol = solve_ivp(rhs, [0, t_max], [s0, i0, r0], t_eval=t_grid, method="RK45")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]
