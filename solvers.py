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
    times = np.linspace(0.0, t_final, n_save, dtype=np.float64)
    out = np.empty((n, n_save, nx), dtype=np.float64)
    u = u0.astype(np.float64).copy()
    out[:, 0] = u

    t = 0.0
    for si in range(1, n_save):
        t_target = times[si]
        while t < t_target:
            umax = np.max(np.abs(u))
            adv_dt = np.inf if umax < 1e-12 else 0.4 * dx / umax
            diff_dt = np.inf if nu <= 0 else 0.45 * dx * dx / nu
            dt = min(1e-3, adv_dt, diff_dt, t_target - t)

            f = 0.5 * u * u
            u_r = np.roll(u, -1, axis=1)
            f_r = np.roll(f, -1, axis=1)
            a = np.maximum(np.abs(u), np.abs(u_r))
            flux_iphalf = 0.5 * (f + f_r) - 0.5 * a * (u_r - u)
            flux_imhalf = np.roll(flux_iphalf, 1, axis=1)
            conv = -(flux_iphalf - flux_imhalf) / dx

            uxx = (u_r - 2 * u + np.roll(u, 1, axis=1)) / (dx * dx)
            u = u + dt * (conv + nu * uxx)
            t += dt

            if not np.isfinite(u).all():
                raise ValueError("Non-finite values in burgers solver")

        out[:, si] = u

    return out.astype(np.float32), times.astype(np.float32)
