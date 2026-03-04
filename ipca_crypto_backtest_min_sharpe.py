"""
python3 ipca_crypto_backtest.py --csv close_wide_1h.csv --freq H --K 2 3 --tc-bps 15 --min-sharpe 10.0
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# ----------------------------
# Utilities
# ----------------------------

def periods_per_year(freq: str) -> float:
    freq = freq.upper()
    if freq == "D":
        return 365.0
    if freq == "H":
        return 365.0 * 24.0
    if freq in {"T", "MIN"}:
        return 365.0 * 24.0 * 60.0
    raise ValueError(f"Unsupported freq={freq}. Use D or H (or T/MIN).")


def annualization_factor(freq: str) -> float:
    return math.sqrt(periods_per_year(freq))


def safe_solve(A: np.ndarray, b: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    A = np.asarray(A)
    b = np.asarray(b)
    A_reg = A + ridge * np.eye(A.shape[0])
    return np.linalg.solve(A_reg, b)


def standardize_cross_section(df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sig = df.std(axis=1).replace(0.0, np.nan)
    z = df.sub(mu, axis=0).div(sig, axis=0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def sharpe_ratio(r: pd.Series, freq: str) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    mu = r.mean()
    sig = r.std(ddof=1)
    if sig == 0 or np.isnan(sig):
        return np.nan
    return (mu / sig) * annualization_factor(freq)


def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    dd = (wealth / peak) - 1.0
    return float(dd.min())


def cagr_from_wealth(wealth: pd.Series, freq: str) -> float:
    """Annualised geometric return from wealth curve."""
    wealth = wealth.dropna()
    if len(wealth) < 2:
        return np.nan
    total_periods = len(wealth) - 1
    ppy = periods_per_year(freq)
    w0 = float(wealth.iloc[0])
    wT = float(wealth.iloc[-1])
    if w0 <= 0 or wT <= 0:
        return np.nan
    ann_log = (math.log(wT) - math.log(w0)) * (ppy / total_periods)
    return math.expm1(ann_log)


# ----------------------------
# Characteristic construction
# ----------------------------

def build_characteristics(
    base_ret: pd.DataFrame,
    freq: str,
    windows: Dict[str, int] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build price-derived characteristics z_{i,t}.
    base_ret should be "known at t" (e.g., log returns or pct returns).
    """
    if windows is None:
        windows = {
            "mom": 24 * 7 if freq.upper() == "H" else 7,
            "vol": 24 * 7 if freq.upper() == "H" else 30,
            "beta": 24 * 14 if freq.upper() == "H" else 30,
            "tail": 24 * 30 if freq.upper() == "H" else 90,
        }

    mkt = base_ret.mean(axis=1)

    rev_1 = -base_ret
    # Momentum: use rolling sum of log1p returns for stability if base_ret are pct returns
    mom_L = np.log1p(base_ret).rolling(windows["mom"], min_periods=max(5, windows["mom"] // 5)).sum()
    vol_L = base_ret.rolling(windows["vol"], min_periods=max(5, windows["vol"] // 5)).std()

    var_m = mkt.rolling(windows["beta"], min_periods=max(10, windows["beta"] // 5)).var()
    cov_im = base_ret.rolling(windows["beta"], min_periods=max(10, windows["beta"] // 5)).cov(mkt)
    beta_L = cov_im.div(var_m, axis=0)

    tail_w = windows["tail"]
    var_5 = base_ret.rolling(tail_w, min_periods=max(20, tail_w // 5)).quantile(0.05)

    def _es(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        q = np.quantile(x, 0.05)
        tail = x[x <= q]
        return float(np.mean(tail)) if tail.size else np.nan

    es_5 = base_ret.rolling(tail_w, min_periods=max(20, tail_w // 5)).apply(_es, raw=True)

    return {
        "rev_1": rev_1,
        "mom": mom_L,
        "vol": vol_L,
        "beta": beta_L,
        "var5": var_5,
        "es5": es_5,
    }


# ----------------------------
# IPCA core
# ----------------------------

@dataclass
class IPCAResult:
    gamma: np.ndarray          # (L, K)
    factors_in: np.ndarray     # (T, K)
    times_in: pd.DatetimeIndex
    feature_names: List[str]


def fit_ipca_als(
    y_fwd: pd.DataFrame,
    features_std: Dict[str, pd.DataFrame],
    K: int,
    max_iter: int = 30,
    tol: float = 1e-6,
    ridge: float = 1e-6,
    verbose: bool = False,
) -> IPCAResult:
    """Restricted IPCA: r_{i,t+1} = z_{i,t}^T Gamma f_{t+1} + eps (alpha=0)."""
    times = y_fwd.index
    assets = y_fwd.columns

    feat_names = ["const"] + list(features_std.keys())
    feat_panels = {"const": pd.DataFrame(1.0, index=times, columns=assets)}
    for k, df in features_std.items():
        feat_panels[k] = df  # already standardized

    Z_list: List[np.ndarray] = []
    r_list: List[np.ndarray] = []

    for t in times:
        r_t = y_fwd.loc[t]
        mask = r_t.notna()
        for fn in feat_names:
            mask &= feat_panels[fn].loc[t].notna()

        cols = assets[mask.values]
        if len(cols) < max(10, K + 2):
            Z_list.append(None)
            r_list.append(None)
            continue

        Z_t = np.column_stack([feat_panels[fn].loc[t, cols].to_numpy(float) for fn in feat_names])
        r_vec = r_t.loc[cols].to_numpy(float)
        Z_list.append(Z_t)
        r_list.append(r_vec)

    usable_idx = [i for i, Z in enumerate(Z_list) if Z is not None]
    if len(usable_idx) < max(50, 5 * K):
        raise ValueError(f"Not enough usable timestamps to fit IPCA (usable={len(usable_idx)}).")

    times_u = times[usable_idx]
    Z_u = [Z_list[i] for i in usable_idx]
    r_u = [r_list[i] for i in usable_idx]

    T = len(times_u)
    L = Z_u[0].shape[1]

    # init factors with SVD
    R_init = y_fwd.loc[times_u].copy()
    R_init = R_init.sub(R_init.mean(axis=1), axis=0).fillna(0.0).to_numpy(float)
    svd = TruncatedSVD(n_components=K, random_state=0)
    F = svd.fit_transform(R_init)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)

    prev_obj = np.inf
    for it in range(1, max_iter + 1):
        LK = L * K
        S = np.zeros((LK, LK), float)
        b = np.zeros((LK,), float)

        # Gamma update
        for t_i in range(T):
            Zt = Z_u[t_i]
            rt = r_u[t_i]
            ft = F[t_i].reshape(-1, 1)  # (K,1)

            ZTZ = Zt.T @ Zt
            ffT = ft @ ft.T

            S += np.kron(ffT, ZTZ)
            ZTr = Zt.T @ rt
            b += np.kron(ft.ravel(), ZTr)

        vecG = safe_solve(S, b, ridge=ridge)
        Gamma = vecG.reshape((K, L)).T  # (L,K)

        # factor update
        for t_i in range(T):
            Zt = Z_u[t_i]
            rt = r_u[t_i]
            Bt = Zt @ Gamma
            A = Bt.T @ Bt
            c = Bt.T @ rt
            F[t_i] = safe_solve(A, c, ridge=ridge)

        # objective
        obj = 0.0
        for t_i in range(T):
            Zt = Z_u[t_i]
            rt = r_u[t_i]
            ft = F[t_i]
            pred = (Zt @ Gamma) @ ft
            resid = rt - pred
            obj += float(resid @ resid)

        rel_impr = (prev_obj - obj) / (abs(prev_obj) + 1e-12)
        if verbose:
            print(f"[ALS] iter={it:02d} obj={obj:.6e} rel_impr={rel_impr:.3e}")
        if rel_impr < tol:
            break
        prev_obj = obj

    return IPCAResult(gamma=Gamma, factors_in=F, times_in=times_u, feature_names=feat_names)


def realised_factor_and_assets(
    r_fwd_t: pd.Series,
    feats_std_t: Dict[str, pd.Series],
    gamma: np.ndarray,
    feature_names: List[str],
    w_factor: np.ndarray,
    ridge: float = 1e-6,
    leverage: float = 1.0,
) -> tuple[np.ndarray, pd.Series]:
    """
    Given r_{t+1}, Z_t (standardized), Gamma, and factor weights w_factor:
    - compute realised factor return f_hat (OLS)
    - compute tradable asset weights a_t = B (B'B)^(-1) w_factor, then normalize gross exposure.

    Returns:
      f_hat: (K,)
      a_t: Series indexed by assets in the regression cross-section (sums abs to leverage)
    """
    # valid assets
    mask = r_fwd_t.notna()
    for fn in feature_names:
        if fn == "const":
            continue
        mask &= feats_std_t[fn].notna()

    cols = r_fwd_t.index[mask.values]
    K = gamma.shape[1]
    if len(cols) < max(10, K + 2):
        return np.full((K,), np.nan), pd.Series(dtype=float)

    Z = np.column_stack([
        np.ones(len(cols)),
        *[feats_std_t[fn].loc[cols].to_numpy(float) for fn in feature_names if fn != "const"]
    ])  # (n,L)
    B = Z @ gamma  # (n,K)
    r = r_fwd_t.loc[cols].to_numpy(float)

    BtB = B.T @ B
    inv_BtB = np.linalg.inv(BtB + ridge * np.eye(K))
    # realised factor return (OLS)
    f_hat = inv_BtB @ (B.T @ r)  # (K,)

    # tradable asset weights for factor combo:
    # a = B (B'B)^(-1) w
    a = B @ (inv_BtB @ w_factor)  # (n,)
    a = pd.Series(a, index=cols)

    # gross leverage normalization
    gross = float(np.sum(np.abs(a.values)))
    if gross > 0 and np.isfinite(gross):
        a = a * (leverage / gross)
    else:
        a[:] = 0.0

    return f_hat, a


# ----------------------------
# Backtest
# ----------------------------

def tangency_weights(mu: np.ndarray, cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    w = safe_solve(cov, mu, ridge=ridge)
    var = float(w.T @ cov @ w)
    if var > 0 and np.isfinite(var):
        w = w / math.sqrt(var)
    return w


def backtest_ipca_tangency(
    close_csv: str,
    freq: str,
    K_list: List[int],
    train_frac: float,
    refit_every: int,
    max_iter: int,
    out_dir: str,
    verbose: bool,
    ret_type: str,
    leverage: float,
    tc_bps: float = 15.0,
    min_sharpe: float = 0.5,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tc_rate = tc_bps / 10_000  # 15 bps each way -> 0.0015

    prices = pd.read_csv(close_csv, index_col=0, parse_dates=True).sort_index()
    prices.columns = [str(c).strip() for c in prices.columns]
    prices = prices.apply(pd.to_numeric, errors="coerce")

    keep = prices.columns[prices.notna().mean() > 0.85]
    prices = prices[keep]
    if prices.shape[1] < 5:
        raise ValueError("Need at least ~5 assets with sufficient data.")

    # Returns
    ret_type = ret_type.lower()
    if ret_type == "simple":
        ret = prices.pct_change()
        y_fwd = ret.shift(-1)  # forward SIMPLE returns
        base_for_feats = ret
    elif ret_type == "log":
        logret = np.log(prices).diff()
        y_fwd = logret.shift(-1)  # forward LOG returns
        base_for_feats = logret
    else:
        raise ValueError("--ret-type must be 'simple' or 'log'.")

    # Features (standardize once)
    feats_raw = build_characteristics(base_for_feats, freq=freq)
    feats_std = {k: standardize_cross_section(v) for k, v in feats_raw.items()}

    common_idx = y_fwd.index
    for k in feats_std:
        feats_std[k] = feats_std[k].reindex(common_idx)

    T_total = len(common_idx)
    split = int(T_total * train_frac)
    times_all = common_idx[:-1]
    split = min(split, len(times_all) - 50)
    t0 = times_all[split]

    print(f"Loaded prices: {prices.shape[0]} timestamps x {prices.shape[1]} assets")
    print(f"OOS starts at index {split} -> {t0} (train_frac={train_frac})")

    summary_rows = []
    ppy = periods_per_year(freq)

    for K in K_list:
        print(f"\n=== IPCA backtest: K={K} factors ===")

        last_fit_idx = None
        ipca_res: IPCAResult | None = None

        f_hist: List[np.ndarray] = []

        port_r = []
        port_times = []
        all_asset_weights = []
        prev_weights = pd.Series(dtype=float)

        # Per-factor tracking
        factor_port_r: Dict[int, List[float]] = {k: [] for k in range(K)}
        factor_all_weights: Dict[int, list] = {k: [] for k in range(K)}
        factor_prev_weights: Dict[int, pd.Series] = {k: pd.Series(dtype=float) for k in range(K)}

        for j, t in enumerate(times_all[split:]):
            global_idx = split + j

            do_refit = (last_fit_idx is None) or ((global_idx - last_fit_idx) >= refit_every)
            if do_refit:
                train_times = times_all[:global_idx]
                y_train = y_fwd.loc[train_times]
                feats_train = {k: v.loc[train_times] for k, v in feats_std.items()}

                ipca_res = fit_ipca_als(
                    y_fwd=y_train,
                    features_std=feats_train,
                    K=K,
                    max_iter=max_iter,
                    tol=1e-6,
                    ridge=1e-6,
                    verbose=verbose,
                )
                last_fit_idx = global_idx

            assert ipca_res is not None

            # tangency weights on FACTORS from history up to t-1
            if len(f_hist) >= max(50, 5 * K):
                Fh = np.vstack(f_hist)
                mu = np.nanmean(Fh, axis=0)
                cov = np.cov(Fh, rowvar=False)
                # Ex-ante Sharpe gate: only trade if factor opportunity is strong enough
                w_raw = safe_solve(cov, mu, ridge=1e-6)
                sr_sq = float(mu @ w_raw)
                ex_ante_sr_ann = math.sqrt(max(0.0, sr_sq)) * annualization_factor(freq)
                if ex_ante_sr_ann >= min_sharpe:
                    w_factor = tangency_weights(mu, cov, ridge=1e-6)
                else:
                    w_factor = np.zeros((K,), float)
            else:
                w_factor = np.zeros((K,), float)

            # realised factor return + tradable asset weights
            r_t = y_fwd.loc[t]  # r_{t+1}
            feats_t = {k: feats_std[k].loc[t] for k in feats_std.keys()}
            f_hat, a_t = realised_factor_and_assets(
                r_fwd_t=r_t,
                feats_std_t=feats_t,
                gamma=ipca_res.gamma,
                feature_names=ipca_res.feature_names,
                w_factor=w_factor,
                ridge=1e-6,
                leverage=leverage,
            )

            # portfolio return is asset-weighted return (TRADABLE)
            if len(a_t) == 0:
                pr = np.nan
            else:
                r_vec = r_t.loc[a_t.index]
                pr = float(np.nansum(a_t.values * r_vec.values))

                # transaction cost: 15 bps each way on turnover
                all_assets = prev_weights.index.union(a_t.index)
                w_old = prev_weights.reindex(all_assets, fill_value=0.0)
                w_new = a_t.reindex(all_assets, fill_value=0.0)
                turnover = float(np.sum(np.abs((w_new - w_old).values)))
                pr -= turnover * tc_rate

            port_times.append(t)
            port_r.append(pr)
            all_asset_weights.append(a_t)
            prev_weights = a_t.copy() if len(a_t) > 0 else prev_weights

            # Per-factor individual portfolio returns (long each factor independently)
            for fk in range(K):
                e_k = np.zeros(K)
                e_k[fk] = 1.0
                _, a_fk = realised_factor_and_assets(
                    r_fwd_t=r_t, feats_std_t=feats_t,
                    gamma=ipca_res.gamma,
                    feature_names=ipca_res.feature_names,
                    w_factor=e_k, ridge=1e-6, leverage=leverage,
                )
                if len(a_fk) == 0:
                    fk_r = np.nan
                else:
                    fk_r = float(np.nansum(a_fk.values * r_t.loc[a_fk.index].values))
                    # TC for this factor portfolio
                    fk_all = factor_prev_weights[fk].index.union(a_fk.index)
                    fk_old = factor_prev_weights[fk].reindex(fk_all, fill_value=0.0)
                    fk_new = a_fk.reindex(fk_all, fill_value=0.0)
                    fk_to = float(np.sum(np.abs((fk_new - fk_old).values)))
                    fk_r -= fk_to * tc_rate
                factor_port_r[fk].append(fk_r)
                factor_all_weights[fk].append(a_fk)
                factor_prev_weights[fk] = a_fk.copy() if len(a_fk) > 0 else factor_prev_weights[fk]

            # update factor history AFTER using it
            if np.all(np.isfinite(f_hat)):
                f_hist.append(f_hat)

        port_r = pd.Series(port_r, index=pd.DatetimeIndex(port_times), name="portfolio_return")

        # wealth
        if ret_type == "simple":
            wealth = (1.0 + port_r.fillna(0.0)).cumprod()
        else:
            wealth = np.exp(port_r.fillna(0.0).cumsum())
        wealth.iloc[0] = 1.0

        # metrics
        sr = sharpe_ratio(port_r, freq=freq)
        vol_ann = port_r.std(ddof=1) * math.sqrt(ppy)
        cagr = cagr_from_wealth(wealth, freq=freq)
        mdd = max_drawdown(wealth)

        metrics = {
            "K": K,
            "factor": "tangency",
            "ret_type": ret_type,
            "leverage_gross": leverage,
            "tc_bps": tc_bps,
            "min_sharpe": min_sharpe,
            "sharpe_ann": sr,
            "vol_ann": vol_ann,
            "cagr_ann": cagr,
            "max_drawdown": mdd,
            "n_obs": int(port_r.notna().sum()),
            "min_ret": float(np.nanmin(port_r.values)),
            "max_ret": float(np.nanmax(port_r.values)),
        }
        summary_rows.append(metrics)
        pd.DataFrame([metrics]).to_csv(out_path / f"metrics_K{K}.csv", index=False)
        port_r.to_csv(out_path / f"portfolio_returns_K{K}.csv")

        # save weights (ragged -> align to all assets)
        w_df = pd.DataFrame(index=port_r.index, columns=prices.columns, dtype=float)
        for tt, a_t in zip(port_r.index, all_asset_weights):
            if len(a_t) > 0:
                w_df.loc[tt, a_t.index] = a_t.values
        w_df = w_df.fillna(0.0)
        w_df.to_csv(out_path / f"asset_weights_K{K}.csv")

        # charts
        plt.figure()
        wealth.plot()
        plt.title(f"IPCA Tradable Wealth (K={K}, {ret_type} returns, gross={leverage})")
        plt.xlabel("Time")
        plt.ylabel("Wealth Index")
        plt.tight_layout()
        plt.savefig(out_path / f"fig_wealth_K{K}.png", dpi=150)
        plt.close()

        win = 24 * 30 if freq.upper() == "H" else 30
        roll = port_r.rolling(win).apply(lambda x: sharpe_ratio(pd.Series(x), freq=freq), raw=False)
        plt.figure()
        roll.plot()
        plt.title(f"Rolling Sharpe (window={win}) (K={K})")
        plt.xlabel("Time")
        plt.ylabel("Sharpe (annualized)")
        plt.tight_layout()
        plt.savefig(out_path / f"fig_rolling_sharpe_K{K}.png", dpi=150)
        plt.close()

        print(
            f"K={K}  Tangency  Sharpe(ann)={sr:.3f}  vol_ann={vol_ann:.3f}  "
            f"CAGR(ann)={cagr:.3%}  maxDD={mdd:.3%}"
        )

        # --- Save Gamma matrix ---
        gamma_df = pd.DataFrame(
            ipca_res.gamma,
            index=ipca_res.feature_names,
            columns=[f"Factor_{fk+1}" for fk in range(K)],
        )
        gamma_df.to_csv(out_path / f"gamma_K{K}.csv")

        # --- Per-factor individual backtests ---
        for fk in range(K):
            fk_label = f"Factor_{fk+1}"
            fk_ret = pd.Series(
                factor_port_r[fk],
                index=pd.DatetimeIndex(port_times),
                name=f"factor_{fk+1}_return",
            )

            # wealth
            if ret_type == "simple":
                fk_wealth = (1.0 + fk_ret.fillna(0.0)).cumprod()
            else:
                fk_wealth = np.exp(fk_ret.fillna(0.0).cumsum())
            fk_wealth.iloc[0] = 1.0

            # metrics
            fk_sr = sharpe_ratio(fk_ret, freq=freq)
            fk_vol = fk_ret.std(ddof=1) * math.sqrt(ppy)
            fk_cagr = cagr_from_wealth(fk_wealth, freq=freq)
            fk_mdd = max_drawdown(fk_wealth)

            fk_metrics = {
                "K": K,
                "factor": fk + 1,
                "ret_type": ret_type,
                "leverage_gross": leverage,
                "tc_bps": tc_bps,
                "sharpe_ann": fk_sr,
                "vol_ann": fk_vol,
                "cagr_ann": fk_cagr,
                "max_drawdown": fk_mdd,
                "n_obs": int(fk_ret.notna().sum()),
            }
            summary_rows.append(fk_metrics)
            pd.DataFrame([fk_metrics]).to_csv(
                out_path / f"factor_{fk+1}_metrics_K{K}.csv", index=False
            )
            fk_ret.to_csv(out_path / f"factor_{fk+1}_returns_K{K}.csv")

            # save per-factor weights
            fk_w_df = pd.DataFrame(
                index=port_r.index, columns=prices.columns, dtype=float
            )
            for tt, a_fk in zip(port_r.index, factor_all_weights[fk]):
                if len(a_fk) > 0:
                    fk_w_df.loc[tt, a_fk.index] = a_fk.values
            fk_w_df = fk_w_df.fillna(0.0)
            fk_w_df.to_csv(out_path / f"factor_{fk+1}_weights_K{K}.csv")

            # wealth chart
            plt.figure()
            fk_wealth.plot()
            plt.title(f"{fk_label} Wealth (K={K}, {ret_type}, gross={leverage}, tc={tc_bps}bps)")
            plt.xlabel("Time")
            plt.ylabel("Wealth Index")
            plt.tight_layout()
            plt.savefig(out_path / f"fig_factor_{fk+1}_wealth_K{K}.png", dpi=150)
            plt.close()

            print(
                f"K={K}  {fk_label}  Sharpe(ann)={fk_sr:.3f}  vol_ann={fk_vol:.3f}  "
                f"CAGR(ann)={fk_cagr:.3%}  maxDD={fk_mdd:.3%}"
            )

    summary = pd.DataFrame(summary_rows)
    sort_cols = ["K", "factor"] if "factor" in summary.columns else ["K"]
    summary = summary.sort_values(sort_cols)
    summary.to_csv(out_path / "summary_metrics.csv", index=False)

    # --- Strategy summary ---
    summary_text = f"""\
================================================================================
IPCA CRYPTO FACTOR STRATEGY - FULL LOGIC SUMMARY
================================================================================

1. DATA
   - Input:            {close_csv}
   - Frequency:        {freq} ({'hourly' if freq.upper() == 'H' else 'daily'})
   - Return type:      {ret_type}
   - Assets retained:  those with >85% non-missing data
   - Total timestamps: {prices.shape[0]}
   - Assets:           {prices.shape[1]}

2. RETURN CALCULATION
   - Simple returns:   r_t = p_t / p_(t-1) - 1
   - Log returns:      r_t = log(p_t) - log(p_(t-1))
   - Forward returns:  y_t = r_(t+1)  (shift(-1) to avoid look-ahead bias)

3. CHARACTERISTICS (6 features + constant, cross-sectionally z-scored)
   - const:   Intercept = 1 for all assets
   - rev_1:   -r_t (short-term reversal: negated current return)
   - mom:     Rolling sum of log(1+r) over {24*7 if freq.upper()=='H' else 7} periods (momentum)
   - vol:     Rolling std of returns over {24*7 if freq.upper()=='H' else 30} periods (volatility)
   - beta:    Cov(r_i, r_mkt) / Var(r_mkt) over {24*14 if freq.upper()=='H' else 30} periods (market beta)
   - var5:    5th percentile of returns over {24*30 if freq.upper()=='H' else 90} periods (tail risk)
   - es5:     Mean of returns below 5th pct over {24*30 if freq.upper()=='H' else 90} periods (expected shortfall)
   - Each feature is cross-sectionally standardized (z-scored across assets at each t)
     and clipped at +/-5 sigma.

4. IPCA MODEL: r_(i,t+1) = z_(i,t)' * Gamma * f_(t+1) + epsilon
   - Gamma (L x K):    Maps 7 features to K factor loadings (learned via ALS)
   - B_t = Z_t @ Gamma: Each asset's time-varying loading on K factors
   - f_t (K x 1):      Latent factor returns (unobservable, estimated via OLS)
   - Estimation:        Alternating Least Squares (max {max_iter} iterations, tol=1e-6)
   - Initialization:    Truncated SVD on demeaned return matrix
   - Gamma step:        vec(Gamma) = [sum_t (f_t f_t') kron (Z_t'Z_t)]^(-1) * [sum_t kron(f_t, Z_t'r_t)]
   - Factor step:       f_t = (B_t'B_t)^(-1) B_t' r_t   (OLS per timestamp)
   - Ridge:             1e-6 added to diagonal for numerical stability
   - Min assets/period: max(10, K+2)
   - Min timestamps:    max(50, 5*K)

5. TRAIN/TEST SPLIT
   - Train fraction:    {train_frac} ({train_frac*100:.0f}% of data for initial training)
   - Model refit:       Every {refit_every} periods (expanding window: all data up to t)
   - OOS start:         Index {split} -> {t0}

6. TRADING LOGIC (at each OOS period t)
   a) Re-estimate Gamma every {refit_every} periods using all data up to t
   b) Compute tangency weights on historical factor returns:
      - mu_f = mean(f_hist),  Sigma_f = cov(f_hist)
      - w* = Sigma_f^(-1) mu_f,  normalized to unit variance
      - Requires >= max(50, 5*K) factor observations
   c) Ex-ante Sharpe gate (min_sharpe = {min_sharpe}):
      - SR*_ann = sqrt(mu_f' Sigma_f^(-1) mu_f) * sqrt(periods_per_year)
      - If SR*_ann < {min_sharpe}: set w_factor = 0 (no trade, hold flat)
      - If SR*_ann >= {min_sharpe}: proceed with tangency weights
   d) Translate factor weights to asset weights:
      - B_t = Z_t @ Gamma          (N x K factor loadings)
      - a_t = B_t (B_t'B_t)^(-1) w*  (minimum-norm replicating portfolio)
      - Normalize: sum(|a_i|) = {leverage} (gross leverage)
   e) Portfolio return:
      - r_port = sum(a_i * r_(i,t+1))
   f) Transaction costs ({tc_bps} bps each way):
      - turnover = sum(|a_new - a_old|) across all assets
      - cost = turnover * {tc_bps/10000:.4f}
      - r_net = r_port - cost
   g) Update factor history: f_hat = (B_t'B_t)^(-1) B_t' r_t (for next period)

7. INDIVIDUAL FACTOR PORTFOLIOS
   - For each factor k = 1..K independently:
     - Set w_factor = e_k (unit vector: pure exposure to factor k)
     - Compute a_k = B_t (B_t'B_t)^(-1) e_k, normalize to gross leverage = {leverage}
     - Compute return with same TC logic as tangency portfolio
   - Each factor gets its own: returns, weights, metrics, wealth chart

8. WEALTH ACCUMULATION
   - Simple: W_t = prod(1 + r_net_tau) for tau = 1..t
   - Log:    W_t = exp(sum(r_net_tau))

9. PERFORMANCE METRICS
   - Sharpe (annualized):   (mean / std) * sqrt(periods_per_year)
   - Volatility (ann):      std * sqrt(periods_per_year)
   - CAGR:                  exp(log(W_T/W_0) * ppy / T) - 1
   - Max Drawdown:          min(W_t / running_max - 1)

================================================================================
OUTPUT FILES (per K value)
================================================================================
   metrics_K{{K}}.csv              Tangency portfolio performance metrics
   portfolio_returns_K{{K}}.csv    Tangency portfolio hourly returns
   asset_weights_K{{K}}.csv        Tangency portfolio asset weights over time
   gamma_K{{K}}.csv                Gamma matrix (feature -> factor loadings)
   factor_{{k}}_metrics_K{{K}}.csv   Factor k standalone performance metrics
   factor_{{k}}_returns_K{{K}}.csv   Factor k standalone hourly returns
   factor_{{k}}_weights_K{{K}}.csv   Factor k standalone asset weights
   fig_wealth_K{{K}}.png           Tangency wealth curve
   fig_rolling_sharpe_K{{K}}.png   Tangency rolling Sharpe
   fig_factor_{{k}}_wealth_K{{K}}.png  Factor k standalone wealth curve
   summary_metrics.csv            All metrics (tangency + per-factor) combined
   strategy_summary.txt           This file

================================================================================
"""
    (out_path / "strategy_summary.txt").write_text(summary_text)

    print(f"\nSaved outputs to: {out_path.resolve()}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--freq", type=str, default="H")
    p.add_argument("--K", type=int, nargs="+", default=[2, 3])
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--refit-every", type=int, default=168)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--out-dir", type=str, default="ipca_outputs")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--ret-type", type=str, default="simple", choices=["simple", "log"])
    p.add_argument("--leverage", type=float, default=1.0, help="Gross leverage: sum(abs(weights))")
    p.add_argument("--tc-bps", type=float, default=15.0, help="Transaction cost in bps each way (default 15)")
    p.add_argument("--min-sharpe", type=float, default=0.5, help="Min ex-ante annualized factor Sharpe to trade (default 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backtest_ipca_tangency(
        close_csv=args.csv,
        freq=args.freq,
        K_list=args.K,
        train_frac=args.train_frac,
        refit_every=args.refit_every,
        max_iter=args.max_iter,
        out_dir=args.out_dir,
        verbose=args.verbose,
        ret_type=args.ret_type,
        leverage=args.leverage,
        tc_bps=args.tc_bps,
        min_sharpe=args.min_sharpe,
    )