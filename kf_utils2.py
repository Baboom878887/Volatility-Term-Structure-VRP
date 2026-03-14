"""
kf_utils.py — Kalman Filter utilities for VIX mean-reversion research.

Extracted from research.ipynb to keep the notebook under the 128 KB limit.
Contains: data classes, OU estimation, KF variants (static, adaptive, IMM),
and innovation diagnostic helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — OU Parameter Estimation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OUParams:
    """Ornstein-Uhlenbeck process parameters estimated from discrete data.

    Attributes
    ----------
    theta : float
        Mean-reversion speed  (θ > 0).
    mu : float
        Long-run equilibrium mean.
    sigma : float
        Diffusion coefficient (instantaneous volatility).
    half_life : float
        Time (in days) for a deviation to decay by 50%:  ln(2)/θ.
    a_ols : float
        OLS intercept from  ΔX = a + b·X_{t-1} + ε.
    b_ols : float
        OLS slope coefficient (should be negative for mean reversion).
    r_squared : float
        R² of the OLS regression.
    n_obs : int
        Number of observations used in the fit.
    """
    theta: float
    mu: float
    sigma: float
    half_life: float
    a_ols: float
    b_ols: float
    r_squared: float
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"OUParams(θ={self.theta:.6f}, μ={self.mu:.2f}, σ={self.sigma:.4f}, "
            f"t½={self.half_life:.1f}d, R²={self.r_squared:.4f}, n={self.n_obs})"
        )


def estimate_ou_ols(x: np.ndarray) -> OUParams:
    """Estimate OU parameters from a 1-D price series via OLS on ΔX = a + b·X_{t-1} + ε.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Price / level series (e.g. VIX daily closes), float64 for numerical
        precision in the OLS inversion.

    Returns
    -------
    OUParams
        Fitted parameters with derived half-life.

    Notes
    -----
    Uses the normal equations (X'X)^{-1} X'y directly via numpy for speed.
    Equivalent to statsmodels OLS but avoids the overhead.
    """
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)                       # ΔX_t = X_t - X_{t-1}
    x_lag = x[:-1]                        # X_{t-1}

    # Design matrix  [1, X_{t-1}]
    A = np.column_stack([np.ones_like(x_lag), x_lag])

    # Normal equations: β = (A'A)^{-1} A'y
    AtA = A.T @ A
    Aty = A.T @ dx
    beta = np.linalg.solve(AtA, Aty)      # [a, b]

    a, b = beta
    residuals = dx - A @ beta

    # Map OLS → OU
    theta = -b                             # mean-reversion speed
    mu = a / theta if theta > 1e-12 else np.nan   # long-run mean
    sigma = float(np.std(residuals, ddof=2))       # diffusion (ddof=2 for 2 estimated params)
    half_life = np.log(2) / theta if theta > 1e-12 else np.inf

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((dx - dx.mean()) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return OUParams(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
        a_ols=float(a),
        b_ols=float(b),
        r_squared=float(r_sq),
        n_obs=len(dx),
    )


def rolling_ou_vectorized(x: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Compute rolling-window OU parameter estimates without Python loops.

    Uses the closed-form OLS normal equations expressed as rolling sums so
    that the entire computation is vectorized.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Level series (float64).
    window : int
        Rolling window size (number of observations in each regression).

    Returns
    -------
    dict with keys 'theta', 'mu', 'sigma', 'half_life', 'b_ols', 'a_ols',
    'r_squared', each a 1-D ndarray of length T - window.
    """
    x = np.asarray(x, dtype=np.float64)

    # Dependent & independent variables for entire series
    dx    = np.diff(x)          # length T-1
    x_lag = x[:-1]              # length T-1

    # Products needed for OLS normal equations
    xlag2   = x_lag ** 2
    xlag_dx = x_lag * dx
    dx2     = dx ** 2

    # Rolling sums via cumsum (O(T), no loop) ─────────────────────────────────
    def rolling_sum(arr: np.ndarray, w: int) -> np.ndarray:
        """Rolling sum of *arr* over window *w* using cumsum. Returns len(arr)-w+1."""
        cs = np.concatenate(([0.0], np.cumsum(arr)))
        return cs[w:] - cs[:-w]

    n = window - 1   # each window of x has window points → window-1 diffs

    S_xlag    = rolling_sum(x_lag,   n)   # Σ X_{t-1}       per window
    S_xlag2   = rolling_sum(xlag2,   n)   # Σ X_{t-1}²      per window
    S_dx      = rolling_sum(dx,      n)   # Σ ΔX             per window
    S_xlag_dx = rolling_sum(xlag_dx, n)   # Σ X_{t-1}·ΔX    per window
    S_dx2     = rolling_sum(dx2,     n)   # Σ (ΔX)²          per window

    W = float(n)

    # OLS slope:  b = (W·Σxy - Σx·Σy) / (W·Σx² - (Σx)²)
    denom = W * S_xlag2 - S_xlag ** 2
    denom = np.where(np.abs(denom) < 1e-15, np.nan, denom)

    b = (W * S_xlag_dx - S_xlag * S_dx) / denom
    a = (S_dx - b * S_xlag) / W

    # OU parameters
    theta     = -b
    mu        = np.where(theta > 1e-12, a / theta, np.nan)
    half_life = np.where(theta > 1e-12, np.log(2) / theta, np.nan)

    # Residual variance per window
    ss_res = S_dx2 - a * S_dx - b * S_xlag_dx
    sigma  = np.sqrt(np.maximum(ss_res / (W - 2), 0.0))

    # R² per window
    mean_dx = S_dx / W
    ss_tot  = S_dx2 - W * mean_dx ** 2
    r_sq    = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

    return {
        "theta":     theta,
        "mu":        mu,
        "sigma":     sigma,
        "half_life": half_life,
        "a_ols":     a,
        "b_ols":     b,
        "r_squared": r_sq,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Standard Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KFConfig:
    """Configuration for the 2-state OU Kalman Filter.

    Attributes
    ----------
    F : np.ndarray, shape (2, 2)
        Transition matrix.
    H : np.ndarray, shape (2,)
        Observation vector (1-D observation model).
    Q : np.ndarray, shape (2, 2)
        Process noise covariance.
    R : float
        Measurement noise variance (scalar, since observation is 1-D).
    theta : float
        OU mean-reversion speed used to build F.
    sigma : float
        OU diffusion coefficient.
    """
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: float
    theta: float
    sigma: float


def make_kf_config(
    theta: float,
    mu_sigma: float,
    sigma: float,
    R: float,
    q_mu_ratio: float = 0.05,
) -> KFConfig:
    """Build a KFConfig from OU parameters.

    Parameters
    ----------
    theta : float
        OU mean-reversion speed.
    mu_sigma : float
        OU diffusion coefficient (used for Q[0,0]).
    sigma : float
        Same as mu_sigma in the standard case — kept separate for clarity.
    R : float
        Measurement noise variance.
    q_mu_ratio : float
        Ratio controlling random-walk noise on μ:  q_μ = (sigma * q_mu_ratio)².
        Default 0.05 keeps μ smooth.

    Returns
    -------
    KFConfig
    """
    F = np.array([
        [1.0 - theta, theta],
        [0.0,         1.0  ],
    ], dtype=np.float64)

    H = np.array([1.0, 0.0], dtype=np.float64)

    q_mu = (sigma * q_mu_ratio) ** 2
    Q = np.array([
        [sigma ** 2, 0.0  ],
        [0.0,        q_mu ],
    ], dtype=np.float64)

    return KFConfig(F=F, H=H, Q=Q, R=float(R), theta=theta, sigma=sigma)


def run_kalman_filter_ou(
    y: np.ndarray,
    config: KFConfig,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run a standard Kalman Filter on a 1-D observation sequence.

    Implements the predict → update recursion with pre-allocated output
    arrays and numpy matrix ops inside the (unavoidable) sequential loop.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Observation sequence (VIX levels), float64.
    config : KFConfig
        Filter configuration containing F, H, Q, R.
    x0 : np.ndarray, shape (2,), optional
        Initial state estimate.  Defaults to [y[0], y[0]].
    P0 : np.ndarray, shape (2, 2), optional
        Initial state covariance.  Defaults to a sensible prior.

    Returns
    -------
    dict with keys:
        'x_filt'      : (T, 2)    filtered state estimates [VIX, μ]
        'P_filt'      : (T, 2, 2) filtered state covariances
        'x_pred'      : (T, 2)    predicted (prior) state estimates
        'P_pred'      : (T, 2, 2) predicted state covariances
        'innovations' : (T,)      innovation (prediction error) sequence
        'innov_var'   : (T,)      innovation variance S_t
        'log_lik'     : (T,)      per-step Gaussian log-likelihood
        'kalman_gain' : (T, 2)    Kalman gain at each step
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    n = 2  # state dimension

    F, H, Q, R = config.F, config.H, config.Q, config.R

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    x_filt      = np.empty((T, n),    dtype=np.float64)
    P_filt      = np.empty((T, n, n), dtype=np.float64)
    x_pred      = np.empty((T, n),    dtype=np.float64)
    P_pred      = np.empty((T, n, n), dtype=np.float64)
    innovations = np.empty(T,         dtype=np.float64)
    innov_var   = np.empty(T,         dtype=np.float64)
    log_lik     = np.empty(T,         dtype=np.float64)
    kalman_gain = np.empty((T, n),    dtype=np.float64)

    # ── Initialize state ──────────────────────────────────────────────────────
    if x0 is None:
        x = np.array([y[0], y[0]], dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    if P0 is None:
        ss_var = config.sigma ** 2 / (2.0 * config.theta) if config.theta > 1e-12 else 100.0
        P = np.diag([ss_var, 100.0]).astype(np.float64)
    else:
        P = np.asarray(P0, dtype=np.float64).copy()

    I2 = np.eye(n, dtype=np.float64)
    LOG2PI = np.log(2.0 * np.pi)

    # ── Forward pass (sequential — inherent to KF) ───────────────────────────
    for t in range(T):
        # ── Predict step ─────────────────────────────────────────────────────
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q

        x_pred[t] = x
        P_pred[t] = P

        # ── Update step ──────────────────────────────────────────────────────
        y_hat = H @ x                  # predicted observation (scalar)
        v = y[t] - y_hat               # innovation
        S = H @ P @ H + R              # innovation variance (scalar)

        innovations[t] = v
        innov_var[t]   = S

        # Gaussian log-likelihood for this step
        log_lik[t] = -0.5 * (LOG2PI + np.log(S) + v * v / S)

        # Kalman gain
        K = (P @ H) / S
        kalman_gain[t] = K

        # State update
        x = x + K * v

        # Covariance update (Joseph form for numerical stability)
        IKH = I2 - np.outer(K, H)
        P = IKH @ P @ IKH.T + np.outer(K, K) * R

        # Symmetrize to counter floating-point drift
        P = 0.5 * (P + P.T)

        x_filt[t] = x
        P_filt[t] = P

    return {
        "x_filt":      x_filt,
        "P_filt":      P_filt,
        "x_pred":      x_pred,
        "P_pred":      P_pred,
        "innovations": innovations,
        "innov_var":   innov_var,
        "log_lik":     log_lik,
        "kalman_gain": kalman_gain,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Innovation Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def sample_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute sample autocorrelation function for lags 1..nlags.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Time series (assumed zero-mean or will be demeaned).
    nlags : int
        Maximum lag.

    Returns
    -------
    np.ndarray, shape (nlags,)
        Autocorrelation at lags 1, 2, ..., nlags.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    # Full autocorrelation via FFT (fast)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf_full /= acf_full[0]  # normalize: acf[0] = 1
    return acf_full[1 : nlags + 1]


def ljung_box(x: np.ndarray, lags: list[int]) -> list[tuple[int, float, float]]:
    """Ljung-Box test for autocorrelation at specified lag orders.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Innovation (residual) sequence.
    lags : list of int
        Lag orders to test (e.g. [5, 10, 20]).

    Returns
    -------
    list of (lag, Q_stat, p_value) tuples.
    """
    n = len(x)
    max_lag = max(lags)
    rho = sample_acf(x, max_lag)

    results = []
    for m in lags:
        rho_m = rho[:m]
        k_range = np.arange(1, m + 1)
        Q_stat = n * (n + 2) * np.sum(rho_m ** 2 / (n - k_range))
        p_value = float(sp_stats.chi2.sf(Q_stat, df=m))
        results.append((m, float(Q_stat), p_value))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Adaptive Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

def run_adaptive_kf(
    y: np.ndarray,
    config: KFConfig,
    window: int = 60,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    Q_floor_ratio: float = 0.01,
    Q_ceil_ratio: float = 100.0,
    R_floor: float = 0.01,
    R_ceil_ratio: float = 50.0,
    adapt_R: bool = True,
) -> dict[str, np.ndarray]:
    """Run an Adaptive Kalman Filter with Sage-Husa / Mehra noise estimation.

    Extends the standard KF to estimate Q (process noise) and R (measurement
    noise) online using a windowed innovation covariance method.  Uses
    cumulative sums for O(1) per-step windowed estimation.

    Q estimator (Sage-Husa 1969):
        Δ_t = K_t v_t (K_t v_t)' + P_{t|t} − F P_{t-1|t-1} F'
        Q̂_t = (1/N) Σ Δ_i  (windowed average)

    R estimator (Mehra 1972):
        R̂_t = (1/N) Σ (v_i² − H P_{i|i-1} H')

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Observation sequence (VIX levels), float64.
    config : KFConfig
        Initial filter configuration (F, H, Q_base, R_base).
    window : int
        Rolling window size N for covariance estimation.
    x0 : np.ndarray, shape (2,), optional
        Initial state estimate.  Defaults to [y[0], y[0]].
    P0 : np.ndarray, shape (2, 2), optional
        Initial state covariance.
    Q_floor_ratio : float
        Q[i,i] >= Q_base[i,i] * Q_floor_ratio.
    Q_ceil_ratio : float
        Q[i,i] <= Q_base[i,i] * Q_ceil_ratio.
    R_floor : float
        Minimum allowed R.
    R_ceil_ratio : float
        R <= R_base * R_ceil_ratio.
    adapt_R : bool
        If False, R stays at its baseline value throughout.

    Returns
    -------
    dict with keys:
        'x_filt', 'P_filt', 'x_pred', 'P_pred',
        'innovations', 'innov_var', 'log_lik', 'kalman_gain',
        'Q_adapted' : (T, 2, 2)  adapted Q at each step
        'R_adapted' : (T,)       adapted R at each step
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    n = 2
    F, H = config.F, config.H
    Q_base = config.Q.copy()
    R_base = float(config.R)

    # Safety bounds
    Q_floor_diag = np.diag(Q_base) * Q_floor_ratio
    Q_ceil_diag  = np.diag(Q_base) * Q_ceil_ratio
    R_ceil = R_base * R_ceil_ratio

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    x_filt      = np.empty((T, n),    dtype=np.float64)
    P_filt      = np.empty((T, n, n), dtype=np.float64)
    x_pred      = np.empty((T, n),    dtype=np.float64)
    P_pred      = np.empty((T, n, n), dtype=np.float64)
    innovations = np.empty(T,         dtype=np.float64)
    innov_var   = np.empty(T,         dtype=np.float64)
    log_lik     = np.empty(T,         dtype=np.float64)
    kalman_gain = np.empty((T, n),    dtype=np.float64)
    Q_adapted   = np.empty((T, n, n), dtype=np.float64)
    R_adapted   = np.empty(T,         dtype=np.float64)

    # ── Initialize state ──────────────────────────────────────────────────────
    if x0 is None:
        x = np.array([y[0], y[0]], dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    if P0 is None:
        ss_var = config.sigma ** 2 / (2.0 * config.theta) if config.theta > 1e-12 else 100.0
        P = np.diag([ss_var, 100.0]).astype(np.float64)
    else:
        P = np.asarray(P0, dtype=np.float64).copy()

    P0_saved = P.copy()
    I2 = np.eye(n, dtype=np.float64)
    LOG2PI = np.log(2.0 * np.pi)

    Q_cur = Q_base.copy()
    R_cur = R_base

    # ── Cumulative sums for O(1) windowed estimation ──────────────────────────
    delta_cs = np.zeros((T + 1, n, n), dtype=np.float64)
    r_cs = np.zeros(T + 1, dtype=np.float64)

    # ── Forward pass ──────────────────────────────────────────────────────────
    for t in range(T):
        Q_adapted[t] = Q_cur
        R_adapted[t] = R_cur

        # ── Predict ──────────────────────────────────────────────────────────
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q_cur

        x_pred[t] = x
        P_pred[t] = P

        # ── Innovation ───────────────────────────────────────────────────────
        v = y[t] - H @ x
        S = float(H @ P @ H) + R_cur

        innovations[t] = v
        innov_var[t]   = S
        log_lik[t] = -0.5 * (LOG2PI + np.log(max(S, 1e-12)) + v * v / max(S, 1e-12))

        # ── KF Update (Joseph form) ──────────────────────────────────────────
        K = (P @ H) / S
        kalman_gain[t] = K
        x = x + K * v
        IKH = I2 - np.outer(K, H)
        P = IKH @ P @ IKH.T + np.outer(K, K) * R_cur
        P = 0.5 * (P + P.T)

        x_filt[t] = x
        P_filt[t] = P

        # ── Accumulate Sage-Husa delta for Q estimation ──────────────────────
        w = K * v
        P_prev = P_filt[t - 1] if t > 0 else P0_saved
        delta_t = np.outer(w, w) + P - F @ P_prev @ F.T
        delta_cs[t + 1] = delta_cs[t] + delta_t

        # ── Accumulate Mehra term for R estimation ───────────────────────────
        r_cs[t + 1] = r_cs[t] + (v * v - float(H @ P_pred[t] @ H))

        # ── Adapt Q and R for the NEXT time step ────────────────────────────
        if t >= window - 1:
            start_idx = t + 1 - window

            Q_hat_mat = (delta_cs[t + 1] - delta_cs[start_idx]) / window
            Q_hat_diag = np.diag(Q_hat_mat)
            Q_hat_diag = np.clip(Q_hat_diag, Q_floor_diag, Q_ceil_diag)
            Q_cur = np.diag(Q_hat_diag)

            if adapt_R:
                R_hat = (r_cs[t + 1] - r_cs[start_idx]) / window
                R_cur = float(np.clip(R_hat, R_floor, R_ceil))

    return {
        "x_filt": x_filt, "P_filt": P_filt,
        "x_pred": x_pred, "P_pred": P_pred,
        "innovations": innovations, "innov_var": innov_var,
        "log_lik": log_lik, "kalman_gain": kalman_gain,
        "Q_adapted": Q_adapted, "R_adapted": R_adapted,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Interacting Multiple Model (IMM) Filter
# ═══════════════════════════════════════════════════════════════════════════════

def run_imm_filter(
    y: np.ndarray,
    ou_params_list: list,          # list of OUParams, one per regime
    tpm: np.ndarray,               # (K, K) transition probability matrix
    w0: np.ndarray | None = None,  # (K,) initial regime weights
    q_mu_ratio: float = 0.05,
    R_meas: float = 0.25,
) -> dict[str, np.ndarray]:
    """Run an Interacting Multiple Model (IMM) filter.

    Maintains K parallel OU-based Kalman Filters and blends them using
    Bayesian model-probability updates at each time step.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Observation sequence (VIX levels).
    ou_params_list : list of OUParams
        OU parameters for each regime.
    tpm : np.ndarray, shape (K, K)
        Markov transition probability matrix.  tpm[i,j] = P(regime_t=j | regime_{t-1}=i).
    w0 : np.ndarray, shape (K,), optional
        Initial regime weights.  Defaults to stationary distribution of TPM.
    q_mu_ratio : float
        Random-walk noise ratio for μ (same as Sections 3-4).
    R_meas : float
        Measurement noise variance.

    Returns
    -------
    dict with keys:
        'x_filt'       : (T, 2)    fused (blended) filtered state
        'P_filt'       : (T, 2, 2) fused filtered covariance
        'regime_prob'  : (T, K)    posterior regime probabilities
        'innovations'  : (T, K)    per-regime innovations
        'innov_var'    : (T, K)    per-regime innovation variance S
        'log_lik'      : (T,)      fused per-step log-likelihood
        'x_filt_per_regime' : (T, K, 2)   per-regime filtered states
        'P_filt_per_regime' : (T, K, 2, 2) per-regime filtered covs
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    K = len(ou_params_list)
    n = 2  # state dimension

    # ── Build per-regime KF configurations ────────────────────────────────────
    configs = []
    for ou in ou_params_list:
        cfg = make_kf_config(
            theta=ou.theta, mu_sigma=ou.sigma, sigma=ou.sigma,
            R=R_meas, q_mu_ratio=q_mu_ratio,
        )
        configs.append(cfg)

    # ── Initial regime weights ────────────────────────────────────────────────
    if w0 is None:
        evals, evecs = np.linalg.eig(tpm.T)
        idx_one = np.argmin(np.abs(evals - 1.0))
        w = np.real(evecs[:, idx_one])
        w = np.abs(w) / np.sum(np.abs(w))
    else:
        w = np.asarray(w0, dtype=np.float64).copy()

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    x_filt_all     = np.empty((T, K, n),    dtype=np.float64)
    P_filt_all     = np.empty((T, K, n, n), dtype=np.float64)
    regime_prob    = np.empty((T, K),       dtype=np.float64)
    innovations    = np.empty((T, K),       dtype=np.float64)
    innov_var      = np.empty((T, K),       dtype=np.float64)
    log_lik_fused  = np.empty(T,            dtype=np.float64)
    x_filt_fused   = np.empty((T, n),       dtype=np.float64)
    P_filt_fused   = np.empty((T, n, n),    dtype=np.float64)

    # ── Per-regime state initialization ───────────────────────────────────────
    x_k = np.empty((K, n), dtype=np.float64)
    P_k = np.empty((K, n, n), dtype=np.float64)
    for j in range(K):
        ou_j = ou_params_list[j]
        x_k[j] = np.array([y[0], ou_j.mu], dtype=np.float64)
        ss_var = ou_j.sigma ** 2 / (2.0 * ou_j.theta) if ou_j.theta > 1e-12 else 100.0
        P_k[j] = np.diag([ss_var, 100.0])

    I2 = np.eye(n, dtype=np.float64)
    LOG2PI = np.log(2.0 * np.pi)

    # ── Forward pass ──────────────────────────────────────────────────────────
    for t in range(T):

        # ── Step 1: Interaction / Mixing ──────────────────────────────────────
        if t > 0:
            c_bar = tpm.T @ w
            c_bar = np.maximum(c_bar, 1e-30)
            mu_mix = (tpm * w[:, None]) / c_bar[None, :]

            x_mixed = np.empty_like(x_k)
            P_mixed = np.empty_like(P_k)
            for j in range(K):
                x_bar = np.zeros(n, dtype=np.float64)
                for i in range(K):
                    x_bar += mu_mix[i, j] * x_k[i]
                x_mixed[j] = x_bar

                P_bar = np.zeros((n, n), dtype=np.float64)
                for i in range(K):
                    dx = x_k[i] - x_bar
                    P_bar += mu_mix[i, j] * (P_k[i] + np.outer(dx, dx))
                P_mixed[j] = 0.5 * (P_bar + P_bar.T)

            x_k = x_mixed
            P_k = P_mixed
        else:
            c_bar = w.copy()

        # ── Step 2: Mode-Matched Prediction + Update ──────────────────────────
        likelihoods = np.empty(K, dtype=np.float64)
        for j in range(K):
            F_j = configs[j].F
            H_j = configs[j].H
            Q_j = configs[j].Q
            R_j = configs[j].R

            if t > 0:
                x_k[j] = F_j @ x_k[j]
                P_k[j] = F_j @ P_k[j] @ F_j.T + Q_j

            v = y[t] - H_j @ x_k[j]
            S = float(H_j @ P_k[j] @ H_j) + R_j

            innovations[t, j] = v
            innov_var[t, j]   = S

            likelihoods[j] = np.exp(-0.5 * (LOG2PI + np.log(max(S, 1e-30))
                                             + v * v / max(S, 1e-30)))

            K_gain = (P_k[j] @ H_j) / max(S, 1e-30)
            x_k[j] = x_k[j] + K_gain * v
            IKH = I2 - np.outer(K_gain, H_j)
            P_k[j] = IKH @ P_k[j] @ IKH.T + np.outer(K_gain, K_gain) * R_j
            P_k[j] = 0.5 * (P_k[j] + P_k[j].T)

        x_filt_all[t] = x_k.copy()
        P_filt_all[t] = P_k.copy()

        # ── Step 3: Mode Probability Update ───────────────────────────────────
        w_unnorm = c_bar * likelihoods
        w_sum = np.sum(w_unnorm)
        if w_sum > 1e-300:
            w = w_unnorm / w_sum
        else:
            w = np.ones(K, dtype=np.float64) / K

        regime_prob[t] = w
        log_lik_fused[t] = np.log(max(w_sum, 1e-300))

        # ── Step 4: Estimate Fusion ───────────────────────────────────────────
        x_fused = np.zeros(n, dtype=np.float64)
        for j in range(K):
            x_fused += w[j] * x_k[j]
        x_filt_fused[t] = x_fused

        P_fused = np.zeros((n, n), dtype=np.float64)
        for j in range(K):
            dx = x_k[j] - x_fused
            P_fused += w[j] * (P_k[j] + np.outer(dx, dx))
        P_filt_fused[t] = 0.5 * (P_fused + P_fused.T)

    return {
        "x_filt": x_filt_fused,
        "P_filt": P_filt_fused,
        "regime_prob": regime_prob,
        "innovations": innovations,
        "innov_var": innov_var,
        "log_lik": log_lik_fused,
        "x_filt_per_regime": x_filt_all,
        "P_filt_per_regime": P_filt_all,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Regime Detection Utility
# ═══════════════════════════════════════════════════════════════════════════════

def regime_detection_lag(
    dates: pd.DatetimeIndex,
    stress_prob: np.ndarray,
    event_start: str,
    threshold: float = 0.70,
    lookback_window: int = 30,
) -> tuple[int, str]:
    """Compute how many days after event_start the filter reaches threshold.

    Parameters
    ----------
    dates : pd.DatetimeIndex
    stress_prob : np.ndarray, shape (T,)
    event_start : str or datetime-like
    threshold : float
        Regime probability threshold.
    lookback_window : int
        Max days to search forward.

    Returns
    -------
    (lag_days, detection_date) or (-1, "not detected")
    """
    event_ts = pd.Timestamp(event_start)
    start_idx = np.searchsorted(dates, event_ts)
    end_idx = min(start_idx + lookback_window, len(dates))

    for i in range(start_idx, end_idx):
        if stress_prob[i] >= threshold:
            lag = i - start_idx
            return lag, str(dates[i].date())
    return -1, "not detected"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5B-A — Gaussian HMM (pure numpy/scipy, no hmmlearn dependency)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HMMParams:
    """Container for a trained Gaussian HMM.

    Attributes
    ----------
    K : int
        Number of hidden states.
    means : np.ndarray, shape (K, 1)
        Per-state emission means.
    covars : np.ndarray, shape (K, 1, 1)
        Per-state emission covariances (full, but 1-D obs → scalar).
    transmat : np.ndarray, shape (K, K)
        Transition probability matrix.  transmat[i, j] = P(s_t=j | s_{t-1}=i).
    startprob : np.ndarray, shape (K,)
        Initial state distribution.
    n_iter_used : int
        Number of EM iterations actually performed.
    log_likelihood : float
        Final training log-likelihood.
    """
    K: int
    means: np.ndarray
    covars: np.ndarray
    transmat: np.ndarray
    startprob: np.ndarray
    n_iter_used: int
    log_likelihood: float


def _gaussian_log_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """Vectorized log-pdf of univariate Gaussian. x shape (T,)."""
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


def train_gaussian_hmm(
    obs: np.ndarray,
    K: int = 3,
    n_iter: int = 200,
    n_init: int = 10,
    tol: float = 1e-4,
    random_state: int = 42,
) -> HMMParams:
    """Train a Gaussian HMM via Baum-Welch EM on a 1-D observation sequence.

    Implements the full forward-backward EM algorithm using log-space
    computations for numerical stability.  Multiple random restarts
    (n_init) are used to avoid local optima; each restart initializes
    means via K-Means and perturbs transition/start probabilities.

    Parameters
    ----------
    obs : np.ndarray, shape (T,)
        1-D observation sequence (e.g. VIX log-returns or z-scores).
    K : int
        Number of hidden states.
    n_iter : int
        Maximum EM iterations per restart.
    n_init : int
        Number of random restarts.  Best (highest LL) is kept.
    tol : float
        Convergence threshold on log-likelihood change.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    HMMParams
        Trained model parameters.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    T = len(obs)
    rng = np.random.RandomState(random_state)

    best_ll = -np.inf
    best_params = None

    for init_idx in range(n_init):
        # ── K-Means initialization for means ──────────────────────────────────
        # Simple K-Means: pick K seeds, iterate 20 times
        seeds = rng.choice(obs, size=K, replace=False)
        means = seeds.copy()
        for _ in range(20):
            dists = np.abs(obs[:, None] - means[None, :])  # (T, K)
            labels = np.argmin(dists, axis=1)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    means[k] = obs[mask].mean()
        # Sort by mean so state 0 = lowest
        order = np.argsort(means)
        means = means[order]

        variances = np.empty(K, dtype=np.float64)
        for k in range(K):
            mask = labels == order[k]
            if mask.sum() > 1:
                variances[k] = max(obs[mask].var(), 1e-6)
            else:
                variances[k] = obs.var()

        # Initialize transition matrix with high self-transition
        A = np.full((K, K), 0.02 / max(K - 1, 1), dtype=np.float64)
        np.fill_diagonal(A, 0.98)
        # Small perturbation
        A += rng.dirichlet(np.ones(K) * 5, size=K) * 0.05
        A /= A.sum(axis=1, keepdims=True)

        pi = np.ones(K, dtype=np.float64) / K

        # ── EM (Baum-Welch) ───────────────────────────────────────────────────
        prev_ll = -np.inf
        n_used = 0

        for em_iter in range(n_iter):
            n_used = em_iter + 1

            # ── E-step: forward-backward in log space ─────────────────────────
            # Emission log-probabilities (T, K)
            log_B = np.empty((T, K), dtype=np.float64)
            for k in range(K):
                log_B[:, k] = _gaussian_log_pdf(obs, means[k], variances[k])

            log_A = np.log(np.maximum(A, 1e-300))
            log_pi = np.log(np.maximum(pi, 1e-300))

            # Forward pass  (log-alpha)
            log_alpha = np.empty((T, K), dtype=np.float64)
            log_alpha[0] = log_pi + log_B[0]

            for t in range(1, T):
                # log_alpha[t, j] = log(sum_i alpha[t-1,i]*A[i,j]) + log_B[t,j]
                # Use logsumexp over axis=0 of (log_alpha[t-1,:,None] + log_A)
                temp = log_alpha[t - 1, :, None] + log_A  # (K, K)
                log_alpha[t] = _logsumexp_axis0(temp) + log_B[t]

            # Total log-likelihood
            total_ll = _logsumexp_1d(log_alpha[-1])

            # Check convergence
            if abs(total_ll - prev_ll) < tol and em_iter > 5:
                break
            prev_ll = total_ll

            # Backward pass  (log-beta)
            log_beta = np.empty((T, K), dtype=np.float64)
            log_beta[-1] = 0.0  # log(1)

            for t in range(T - 2, -1, -1):
                # log_beta[t, i] = log(sum_j A[i,j]*B[t+1,j]*beta[t+1,j])
                temp = log_A + log_B[t + 1, None, :] + log_beta[t + 1, None, :]
                log_beta[t] = _logsumexp_axis1(temp)

            # Posterior state probabilities  gamma[t, k] = P(s_t=k | obs)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_axis1_1d(log_gamma)[:, None]
            gamma = np.exp(log_gamma)  # (T, K)

            # Transition posteriors  xi[t, i, j]  — vectorized over t
            # log_xi shape (T-1, K, K)
            log_xi = (log_alpha[:-1, :, None]       # (T-1, K, 1)
                      + log_A[None, :, :]            # (1, K, K)
                      + log_B[1:, None, :]           # (T-1, 1, K)
                      + log_beta[1:, None, :])       # (T-1, 1, K)
            # Normalize each time-step slice
            log_xi_max = log_xi.reshape(T - 1, -1).max(axis=1)[:, None, None]
            log_xi -= log_xi_max
            xi = np.exp(log_xi)
            xi /= xi.reshape(T - 1, -1).sum(axis=1)[:, None, None]
            xi_sum = xi.sum(axis=0)  # (K, K)

            # ── M-step ────────────────────────────────────────────────────────
            # Start probability
            pi = gamma[0] + 1e-10
            pi /= pi.sum()

            # Transition matrix
            A = xi_sum + 1e-10
            A /= A.sum(axis=1, keepdims=True)

            # Emission parameters
            for k in range(K):
                wk = gamma[:, k]
                wk_sum = wk.sum() + 1e-10
                means[k] = np.dot(wk, obs) / wk_sum
                diff = obs - means[k]
                variances[k] = max(np.dot(wk, diff ** 2) / wk_sum, 1e-6)

        if total_ll > best_ll:
            best_ll = total_ll
            best_params = (means.copy(), variances.copy(), A.copy(), pi.copy(), n_used)

    means_f, vars_f, A_f, pi_f, n_used_f = best_params

    # Ensure states are sorted by emission mean (low → high)
    order = np.argsort(means_f)
    means_f = means_f[order]
    vars_f = vars_f[order]
    A_f = A_f[order][:, order]
    pi_f = pi_f[order]

    return HMMParams(
        K=K,
        means=means_f.reshape(K, 1),
        covars=vars_f.reshape(K, 1, 1),
        transmat=A_f,
        startprob=pi_f,
        n_iter_used=n_used_f,
        log_likelihood=best_ll,
    )


def _logsumexp_1d(a: np.ndarray) -> float:
    """Log-sum-exp of a 1-D array."""
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def _logsumexp_axis0(a: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis=0 of a 2-D array. Returns 1-D."""
    a_max = np.max(a, axis=0)
    return a_max + np.log(np.sum(np.exp(a - a_max[None, :]), axis=0))


def _logsumexp_axis1(a: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis=1 of a 2-D array. Returns 1-D."""
    a_max = np.max(a, axis=1)
    return a_max + np.log(np.sum(np.exp(a - a_max[:, None]), axis=1))


def _logsumexp_axis1_1d(a: np.ndarray) -> np.ndarray:
    """Alias for logsumexp along axis=1."""
    return _logsumexp_axis1(a)


def _logsumexp_2d(a: np.ndarray) -> float:
    """Log-sum-exp of all elements in a 2-D array."""
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def viterbi_decode(
    obs: np.ndarray,
    hmm: HMMParams,
) -> np.ndarray:
    """Viterbi algorithm for most-likely state sequence.

    Parameters
    ----------
    obs : np.ndarray, shape (T,)
        Observation sequence.
    hmm : HMMParams
        Trained HMM parameters.

    Returns
    -------
    np.ndarray, shape (T,), dtype int
        Most likely hidden state at each time step.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    T = len(obs)
    K = hmm.K

    means = hmm.means.ravel()
    variances = hmm.covars.ravel()
    log_A = np.log(np.maximum(hmm.transmat, 1e-300))
    log_pi = np.log(np.maximum(hmm.startprob, 1e-300))

    # Emission log-probs
    log_B = np.empty((T, K), dtype=np.float64)
    for k in range(K):
        log_B[:, k] = _gaussian_log_pdf(obs, means[k], variances[k])

    # Viterbi
    delta = np.empty((T, K), dtype=np.float64)
    psi = np.empty((T, K), dtype=np.int32)

    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        trans_score = delta[t - 1, :, None] + log_A  # (K, K)
        psi[t] = np.argmax(trans_score, axis=0)
        delta[t] = np.max(trans_score, axis=0) + log_B[t]

    # Backtrack
    states = np.empty(T, dtype=np.int32)
    states[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def hmm_stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a Markov transition matrix.

    Finds π such that π·A = π  (left eigenvector with eigenvalue 1).

    Parameters
    ----------
    transmat : np.ndarray, shape (K, K)

    Returns
    -------
    np.ndarray, shape (K,)
    """
    evals, evecs = np.linalg.eig(transmat.T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def compute_regime_persistence(labels: np.ndarray) -> dict[int, dict[str, float]]:
    """Compute average consecutive run length per regime from a label sequence.

    Parameters
    ----------
    labels : np.ndarray, shape (T,), dtype int
        Regime label at each time step (e.g. Viterbi output).

    Returns
    -------
    dict  mapping  regime_id →  {'mean_run': float, 'median_run': float,
                                  'max_run': int, 'n_runs': int}
    """
    labels = np.asarray(labels)
    unique_regimes = np.unique(labels)
    result = {}

    for r in unique_regimes:
        # Find contiguous runs of regime r
        is_r = (labels == r).astype(np.int32)
        diff = np.diff(is_r, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        run_lengths = ends - starts

        if len(run_lengths) > 0:
            result[int(r)] = {
                "mean_run": float(np.mean(run_lengths)),
                "median_run": float(np.median(run_lengths)),
                "max_run": int(np.max(run_lengths)),
                "n_runs": len(run_lengths),
            }
        else:
            result[int(r)] = {"mean_run": 0.0, "median_run": 0.0, "max_run": 0, "n_runs": 0}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5B-B — BAYESIAN ONLINE CHANGEPOINT DETECTION (BOCPD)
# ═══════════════════════════════════════════════════════════════════════════════
# Based on Adams & MacKay (2007) "Bayesian Online Changepoint Detection".
# Uses a Student-t predictive distribution with conjugate Normal-Inverse-Gamma
# sufficient statistics, constant hazard function, and run-length capping.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BOCPDResult:
    """Container for BOCPD outputs.

    Parameters
    ----------
    cp_prob : np.ndarray, shape (T,)
        Changepoint signal at each t: P(R_t <= r_short | x_{1:t}).
        This is the cumulative probability of being in the first few days
        of a new segment, which spikes after genuine regime changes.
    cp_prob_raw : np.ndarray, shape (T,)
        Raw P(R_t = 0 | x_{1:t}) — the single-step probability.
    run_length_map : np.ndarray, shape (T, R_max+1)
        Full run-length posterior at each time step (for diagnostics).
    max_run_length : np.ndarray, shape (T,)
        MAP run-length at each t.
    mean_run_length : np.ndarray, shape (T,)
        Posterior mean run-length at each t.
    n_changepoints : int
        Number of time steps where cp_prob > threshold (informational).
    hazard : float
        Hazard rate used (1/lambda).
    r_short : int
        Window used for cumulative changepoint signal.
    """
    cp_prob: np.ndarray
    cp_prob_raw: np.ndarray
    run_length_map: np.ndarray
    max_run_length: np.ndarray
    mean_run_length: np.ndarray
    n_changepoints: int
    hazard: float
    r_short: int


def run_bocpd(
    obs: np.ndarray,
    hazard_lambda: float = 63.0,
    r_max: int = 252,
    r_short: int = 10,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    cp_threshold: float = 0.3,
) -> BOCPDResult:
    """Run Bayesian Online Changepoint Detection on a 1-D observation sequence.

    Implements Adams & MacKay (2007) with:
    - Constant hazard function: H = 1/hazard_lambda
    - Student-t predictive distribution from Normal-Inverse-Gamma conjugate prior
    - Run-length capping at r_max to bound memory
    - Cumulative changepoint signal: P(R_t <= r_short)

    Parameters
    ----------
    obs : np.ndarray, shape (T,)
        Observation sequence (e.g. VIX log-returns).
    hazard_lambda : float
        Expected run length in time steps. H(t) = 1/hazard_lambda.
    r_max : int
        Maximum tracked run length. Hypotheses beyond this are merged.
    r_short : int
        Threshold for cumulative changepoint signal. cp_prob[t] = P(R_t <= r_short).
    mu0, kappa0, alpha0, beta0 : float
        Normal-Inverse-Gamma prior hyperparameters.
    cp_threshold : float
        Threshold for counting changepoints (informational only).

    Returns
    -------
    BOCPDResult
        Changepoint probabilities and diagnostics.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    T = len(obs)
    H = 1.0 / hazard_lambda  # constant hazard

    # Allocate run-length posterior  R[t, r] = P(R_t=r | x_{1:t})
    # At time t, run lengths 0..min(t, r_max) are active.
    R = np.zeros((T + 1, r_max + 1), dtype=np.float64)
    R[0, 0] = 1.0  # initially run length = 0 with certainty

    # Sufficient statistics arrays — one per run-length hypothesis
    # These track the Normal-Inverse-Gamma posterior for each hypothesis
    mu_r = np.full(r_max + 1, mu0, dtype=np.float64)
    kappa_r = np.full(r_max + 1, kappa0, dtype=np.float64)
    alpha_r = np.full(r_max + 1, alpha0, dtype=np.float64)
    beta_r = np.full(r_max + 1, beta0, dtype=np.float64)

    # Output arrays
    cp_prob_raw = np.zeros(T, dtype=np.float64)
    max_rl = np.zeros(T, dtype=np.int32)
    mean_rl = np.zeros(T, dtype=np.float64)

    for t in range(T):
        x = obs[t]

        # Number of active run-length hypotheses at this step
        n_active = min(t + 1, r_max + 1)

        # ── 1. Predictive probability under each run-length hypothesis ────
        # Student-t with df=2*alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa))
        df_t = 2.0 * alpha_r[:n_active]
        scale_t = np.sqrt(
            beta_r[:n_active] * (kappa_r[:n_active] + 1.0)
            / (alpha_r[:n_active] * kappa_r[:n_active])
        )
        # Compute Student-t log-pdf for numerical stability
        z = (x - mu_r[:n_active]) / scale_t
        log_pred = (
            _log_gamma_approx(0.5 * (df_t + 1.0))
            - _log_gamma_approx(0.5 * df_t)
            - 0.5 * np.log(np.pi * df_t)
            - np.log(scale_t)
            - 0.5 * (df_t + 1.0) * np.log(1.0 + z * z / df_t)
        )
        pred = np.exp(log_pred - np.max(log_pred))  # stabilize

        # ── 2. Growth probabilities: R_{t+1}(r+1) ∝ R_t(r) * π(x_t|r) * (1-H)
        growth = R[t, :n_active] * pred * (1.0 - H)

        # ── 3. Changepoint probability: R_{t+1}(0) ∝ sum(R_t(r) * π(x_t|r) * H)
        cp_mass = np.sum(R[t, :n_active] * pred * H)

        # ── 4. Assemble and normalize ───────────────────────────────────────
        # New run-length distribution
        new_n = min(n_active + 1, r_max + 1)
        R[t + 1, :] = 0.0
        R[t + 1, 0] = cp_mass
        if n_active < r_max + 1:
            R[t + 1, 1:n_active + 1] = growth
        else:
            # Cap: merge the longest run-lengths
            R[t + 1, 1:r_max] = growth[:r_max - 1]
            R[t + 1, r_max] = growth[r_max - 1:].sum()

        # Normalize
        total = R[t + 1, :new_n].sum()
        if total > 0:
            R[t + 1, :new_n] /= total

        cp_prob_raw[t] = R[t + 1, 0]
        max_rl[t] = np.argmax(R[t + 1, :new_n])
        r_indices = np.arange(new_n, dtype=np.float64)
        mean_rl[t] = np.dot(R[t + 1, :new_n], r_indices)

        # ── 5. Update sufficient statistics for next step ─────────────────
        # Shift existing stats to account for run-length increment
        # For r=1..n_active: stats come from r-1's previous values
        # For r=0: reset to prior (new segment)

        # Save old values before overwriting
        mu_old = mu_r[:n_active].copy()
        kappa_old = kappa_r[:n_active].copy()
        alpha_old = alpha_r[:n_active].copy()
        beta_old = beta_r[:n_active].copy()

        # Update stats for grown hypotheses (r → r+1)
        kappa_new = kappa_old + 1.0
        mu_new = (kappa_old * mu_old + x) / kappa_new
        alpha_new = alpha_old + 0.5
        beta_new = (beta_old
                    + 0.5 * kappa_old * (x - mu_old) ** 2 / kappa_new)

        # Shift into position
        end_r = min(n_active + 1, r_max + 1)
        mu_r[1:end_r] = mu_new[:end_r - 1]
        kappa_r[1:end_r] = kappa_new[:end_r - 1]
        alpha_r[1:end_r] = alpha_new[:end_r - 1]
        beta_r[1:end_r] = beta_new[:end_r - 1]

        # Reset r=0 to prior (new changepoint hypothesis)
        mu_r[0] = mu0
        kappa_r[0] = kappa0
        alpha_r[0] = alpha0
        beta_r[0] = beta0

    # Cumulative changepoint signal: P(R_t <= r_short)
    R_out = R[1:]  # (T, r_max+1)
    r_short_capped = min(r_short, r_max)
    cp_prob = R_out[:, :r_short_capped + 1].sum(axis=1)

    n_cp = int((cp_prob > cp_threshold).sum())

    return BOCPDResult(
        cp_prob=cp_prob,
        cp_prob_raw=cp_prob_raw,
        run_length_map=R_out,
        max_run_length=max_rl,
        mean_run_length=mean_rl,
        n_changepoints=n_cp,
        hazard=H,
        r_short=r_short,
    )


def _log_gamma_approx(x: np.ndarray) -> np.ndarray:
    """Vectorized log-gamma via scipy for accuracy.

    Falls back to Stirling approximation for very large x to avoid overflow.
    """
    from scipy.special import gammaln
    return gammaln(x)


def bocpd_inflation_events(
    cp_prob: np.ndarray,
    obs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Identify BOCPD-triggered covariance inflation events.

    Parameters
    ----------
    cp_prob : np.ndarray, shape (T,)
        Changepoint probability at each time step.
    obs : np.ndarray, shape (T,)
        Observation sequence (e.g. VIX log-returns).
    threshold : float
        Threshold on cp_prob to trigger inflation.

    Returns
    -------
    dict with keys:
        'triggered' : np.ndarray, bool (T,) — True where inflation fires
        'n_events' : int — total inflation events
        'event_indices' : np.ndarray, int — indices of triggered time steps
        'event_cp_probs' : np.ndarray — cp_prob at each event
        'event_obs_move' : np.ndarray — observation value at each event
        'cluster_starts' : np.ndarray — start index of each cluster
        'cluster_lengths' : np.ndarray — length of each cluster
    """
    triggered = cp_prob > threshold
    event_idx = np.where(triggered)[0]

    # Cluster consecutive triggers
    if len(event_idx) > 0:
        breaks = np.where(np.diff(event_idx) > 1)[0] + 1
        clusters = np.split(event_idx, breaks)
        cluster_starts = np.array([c[0] for c in clusters])
        cluster_lengths = np.array([len(c) for c in clusters])
    else:
        cluster_starts = np.array([], dtype=np.int64)
        cluster_lengths = np.array([], dtype=np.int64)

    return {
        "triggered": triggered,
        "n_events": int(triggered.sum()),
        "event_indices": event_idx,
        "event_cp_probs": cp_prob[event_idx] if len(event_idx) > 0 else np.array([]),
        "event_obs_move": obs[event_idx] if len(event_idx) > 0 else np.array([]),
        "cluster_starts": cluster_starts,
        "cluster_lengths": cluster_lengths,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED SWITCHING KALMAN FILTER  (HMM + BOCPD + IMM)
# ═══════════════════════════════════════════════════════════════════════════════

def run_switching_kf(
    y,
    ou_params_list,
    tpm,
    cp_prob,
    cp_tau=0.3,
    cp_alpha=5.0,
    tpm_flatten=0.0,
    w0=None,
    q_mu_ratio=0.05,
    R_meas=0.25,
):
    """Integrated Switching KF: HMM + BOCPD + IMM.

    Combines HMM-informed regime switching with BOCPD changepoint detection.
    When ``cp_prob[t] > cp_tau``, inflate Q by ``cp_alpha`` and optionally
    flatten the TPM towards uniform to accelerate regime switching.

    Parameters
    ----------
    y : (T,) array — observation series (e.g. VIX levels).
    ou_params_list : list[OUParams] — one per regime.
    tpm : (K, K) array — HMM transition probability matrix.
    cp_prob : (T,) array — BOCPD changepoint probability.
    cp_tau : float — changepoint threshold.
    cp_alpha : float — Q inflation factor during changepoints.
    tpm_flatten : float — blend toward uniform TPM during CP (0 = none).
    w0 : (K,) array or None — initial regime weights (default: stationary).
    q_mu_ratio : float — passed to ``make_kf_config``.
    R_meas : float — measurement noise variance.

    Returns
    -------
    dict with keys: x_filt, P_filt, regime_prob, innovations, innov_var,
    log_lik, x_filt_per_regime, P_filt_per_regime, q_inflated, tpm_active.
    """
    y = np.asarray(y, dtype=np.float64)
    cp_prob = np.asarray(cp_prob, dtype=np.float64)
    T = len(y); K = len(ou_params_list); n = 2
    assert len(cp_prob) == T

    configs = [make_kf_config(theta=ou.theta, mu_sigma=ou.sigma, sigma=ou.sigma,
                              R=R_meas, q_mu_ratio=q_mu_ratio) for ou in ou_params_list]

    tpm_uniform = np.ones((K, K), dtype=np.float64) / K

    if w0 is None:
        evals, evecs = np.linalg.eig(tpm.T)
        idx_one = np.argmin(np.abs(evals - 1.0))
        w = np.real(evecs[:, idx_one])
        w = np.abs(w) / np.sum(np.abs(w))
    else:
        w = np.asarray(w0, dtype=np.float64).copy()

    x_filt_all = np.empty((T, K, n), dtype=np.float64)
    P_filt_all = np.empty((T, K, n, n), dtype=np.float64)
    regime_prob = np.empty((T, K), dtype=np.float64)
    innovations = np.empty((T, K), dtype=np.float64)
    innov_var   = np.empty((T, K), dtype=np.float64)
    log_lik_fused = np.empty(T, dtype=np.float64)
    x_filt_fused  = np.empty((T, n), dtype=np.float64)
    P_filt_fused  = np.empty((T, n, n), dtype=np.float64)
    q_inflated    = np.zeros(T, dtype=np.float64)
    tpm_active    = np.empty((T, K, K), dtype=np.float64)

    x_k = np.empty((K, n), dtype=np.float64)
    P_k = np.empty((K, n, n), dtype=np.float64)
    for j in range(K):
        ou_j = ou_params_list[j]
        x_k[j] = np.array([y[0], ou_j.mu], dtype=np.float64)
        ss_var = ou_j.sigma**2 / (2.0 * ou_j.theta) if ou_j.theta > 1e-12 else 100.0
        P_k[j] = np.diag([ss_var, 100.0])

    I2 = np.eye(n, dtype=np.float64)
    LOG2PI = np.log(2.0 * np.pi)

    for t in range(T):
        is_cp = cp_prob[t] > cp_tau
        q_inflated[t] = float(is_cp)
        if is_cp and tpm_flatten > 0:
            tpm_eff = (1.0 - tpm_flatten) * tpm + tpm_flatten * tpm_uniform
        else:
            tpm_eff = tpm
        tpm_active[t] = tpm_eff

        if t > 0:
            c_bar = tpm_eff.T @ w
            c_bar = np.maximum(c_bar, 1e-30)
            mu_mix = (tpm_eff * w[:, None]) / c_bar[None, :]
            x_mixed = np.empty_like(x_k)
            P_mixed = np.empty_like(P_k)
            for j in range(K):
                x_bar = np.zeros(n, dtype=np.float64)
                for i in range(K):
                    x_bar += mu_mix[i, j] * x_k[i]
                x_mixed[j] = x_bar
                P_bar = np.zeros((n, n), dtype=np.float64)
                for i in range(K):
                    dx = x_k[i] - x_bar
                    P_bar += mu_mix[i, j] * (P_k[i] + np.outer(dx, dx))
                P_mixed[j] = 0.5 * (P_bar + P_bar.T)
            x_k = x_mixed; P_k = P_mixed
        else:
            c_bar = w.copy()

        likelihoods = np.empty(K, dtype=np.float64)
        for j in range(K):
            F_j = configs[j].F; H_j = configs[j].H
            Q_j = configs[j].Q; R_j = configs[j].R
            Q_eff = Q_j * cp_alpha if is_cp else Q_j
            if t > 0:
                x_k[j] = F_j @ x_k[j]
                P_k[j] = F_j @ P_k[j] @ F_j.T + Q_eff
            elif is_cp:
                P_k[j] = P_k[j] * cp_alpha
            v = y[t] - H_j @ x_k[j]
            S = float(H_j @ P_k[j] @ H_j) + R_j
            innovations[t, j] = v; innov_var[t, j] = S
            likelihoods[j] = np.exp(-0.5 * (LOG2PI + np.log(max(S, 1e-30)) + v*v / max(S, 1e-30)))
            K_gain = (P_k[j] @ H_j) / max(S, 1e-30)
            x_k[j] = x_k[j] + K_gain * v
            IKH = I2 - np.outer(K_gain, H_j)
            P_k[j] = IKH @ P_k[j] @ IKH.T + np.outer(K_gain, K_gain) * R_j
            P_k[j] = 0.5 * (P_k[j] + P_k[j].T)

        x_filt_all[t] = x_k.copy(); P_filt_all[t] = P_k.copy()

        w_unnorm = c_bar * likelihoods; w_sum = np.sum(w_unnorm)
        w = w_unnorm / w_sum if w_sum > 1e-300 else np.ones(K, dtype=np.float64) / K
        regime_prob[t] = w; log_lik_fused[t] = np.log(max(w_sum, 1e-300))

        x_fused = np.zeros(n, dtype=np.float64)
        for j in range(K):
            x_fused += w[j] * x_k[j]
        x_filt_fused[t] = x_fused
        P_fused = np.zeros((n, n), dtype=np.float64)
        for j in range(K):
            dx = x_k[j] - x_fused
            P_fused += w[j] * (P_k[j] + np.outer(dx, dx))
        P_filt_fused[t] = 0.5 * (P_fused + P_fused.T)

    return {
        "x_filt": x_filt_fused, "P_filt": P_filt_fused,
        "regime_prob": regime_prob, "innovations": innovations,
        "innov_var": innov_var, "log_lik": log_lik_fused,
        "x_filt_per_regime": x_filt_all, "P_filt_per_regime": P_filt_all,
        "q_inflated": q_inflated, "tpm_active": tpm_active,
    }
