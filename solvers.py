import numpy as np


def random_fourier_ic(n_samples, n_grid, kmax=8, spectrum_power=2.0, rng=None):
    rng = np.random.default_rng(rng)
    x = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    u = np.zeros((n_samples, n_grid), dtype=np.float64)
    for k in range(1, kmax + 1):
        amp = 1.0 / (k ** spectrum_power)
        a = rng.normal(size=(n_samples, 1))
        b = rng.normal(size=(n_samples, 1))
        u += amp * (a * np.sin(k * x) + b * np.cos(k * x))
    u -= u.mean(axis=1, keepdims=True)
    u /= (np.std(u, axis=1, keepdims=True) + 1e-8)
    return u.astype(np.float32)


def solve_heat(u0, nu, t_final, n_save):
    n, nx = u0.shape
    times = np.linspace(0.0, t_final, n_save, dtype=np.float64)
    k = np.fft.fftfreq(nx, d=1.0 / nx) * 2 * np.pi
    decay = np.exp(-nu * (k ** 2)[None, :] * times[:, None])
    u0_hat = np.fft.fft(u0, axis=1)
    traj = np.fft.ifft(u0_hat[:, None, :] * decay[None, :, :], axis=2).real
    if not np.isfinite(traj).all():
        raise ValueError("Non-finite values in heat solver")
    return traj.astype(np.float32), times.astype(np.float32)


def solve_burgers(u0, nu, t_final, n_save):
    n, nx = u0.shape
    dx = 2 * np.pi / nx
    dt_int = min(0.002, t_final / 200)
    n_steps = int(np.ceil(t_final / dt_int))
    dt_int = t_final / n_steps
    save_idx = np.linspace(0, n_steps, n_save).round().astype(int)
    out = np.empty((n, n_save, nx), dtype=np.float64)
    u = u0.astype(np.float64).copy()
    out[:, 0] = u
    si = 1
    for step in range(1, n_steps + 1):
        ux = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
        uxx = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (dx * dx)
        u = u + dt_int * (-u * ux + nu * uxx)
        if step == save_idx[si]:
            out[:, si] = u
            si += 1
            if si >= n_save:
                break
    if not np.isfinite(out).all():
        raise ValueError("Non-finite values in burgers solver")
    times = np.linspace(0.0, t_final, n_save, dtype=np.float32)
    return out.astype(np.float32), times
