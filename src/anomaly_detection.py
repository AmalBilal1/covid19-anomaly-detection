import pandas as pd
import numpy as np
from collections import defaultdict

def detect_type1_spike_up(r: pd.Series, q_up=0.99):
    """
    Detects sudden upward spikes in a time series based on a quantile threshold.

    Parameters:
        r (pd.Series): Time series of values (e.g., weekly deaths)
        q_up (float): Upper quantile threshold for defining spikes (default 0.99)

    Returns:
        List of timestamps where spikes occur
    """
    r = r.dropna()
    if len(r) < 5:
        return []
    th = r.quantile(q_up)
    idx = r.index[r > th]
    return list(idx)


def detect_type2_turn_patterns(r: pd.Series,
                               w_pre=3, w_post=3,
                               q_h=0.9, q_flat=0.2, q_trend=0.7):
    """
    Detects local turning points (e.g., flat→up, up→flat) in a time series.
    Based on local mean trends before and after each time point.

    Parameters:
        r (pd.Series): Time series of values (e.g., weekly deaths)
        w_pre (int): Window size before a point
        w_post (int): Window size after a point
        q_h (float): Quantile threshold for high derivative changes
        q_flat (float): Quantile threshold defining 'flat' behavior
        q_trend (float): Quantile threshold defining trend strength

    Returns:
        dict: Mapping of pattern types to lists of timestamps
    """
    r = r.astype(float).dropna()
    if len(r) < (w_pre + w_post + 4):
        return defaultdict(list)

    g = r
    h = g.diff()

    th_h = h.abs().quantile(q_h) if h.notna().any() else np.inf
    abs_g = g.abs().dropna()
    th_flat = abs_g.quantile(q_flat) if len(abs_g) else 0.0
    th_trend = abs_g.quantile(q_trend) if len(abs_g) else 0.0
    th_trend = max(th_trend, th_flat * 1.1)

    def trend_label(mu):
        if abs(mu) <= th_flat:
            return 'flat'
        elif mu >= th_trend:
            return 'up'
        elif mu <= -th_trend:
            return 'down'
        else:
            return 'flat'

    labels = defaultdict(list)

    for t_pos in range(1, len(r) - 1):
        t = r.index[t_pos]
        g_prev = g.iloc[t_pos - 1]
        g_curr = g.iloc[t_pos]
        h_curr = h.iloc[t_pos]

        if pd.isna(g_prev) or pd.isna(g_curr) or pd.isna(h_curr):
            continue

        turn_cond = (g_prev * g_curr <= 0) and (abs(h_curr) >= th_h)
        if not turn_cond:
            continue

        pre_mu = g.iloc[max(0, t_pos - w_pre): t_pos].mean()
        post_mu = g.iloc[t_pos + 1: min(len(g), t_pos + 1 + w_post)].mean()
        pre_lbl = trend_label(pre_mu)
        post_lbl = trend_label(post_mu)

        if pre_lbl == 'down' and post_lbl == 'flat':
            labels['down_turn_flat'].append(t)
        elif pre_lbl == 'flat' and post_lbl == 'down':
            labels['flat_turn_down'].append(t)
        elif pre_lbl == 'flat' and post_lbl == 'up':
            labels['flat_turn_up'].append(t)
        elif pre_lbl == 'up' and post_lbl == 'flat':
            labels['up_turn_flat'].append(t)

    return labels