"""Leakage-safe feature engineering.

Every feature is computed from information available *before* the transaction being scored:
 - customer history uses only earlier transactions of the same customer,
 - merchant risk statistics use only labels whose dispute had been raised and resolved before the
   scoring time (label maturity), and
 - the user-vendor affinity model is refitted monthly on data preceding that month.

The same per-customer routine (`person_features`) is used for the batch build and for live scoring,
so a live score is guaranteed to reproduce what the model saw in training.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from . import config as C

DAY = 86400.0
PRIOR_WEIGHT_VENDOR = 20.0      # pseudo-observations pulling a merchant's as-of fraud rate to the population rate
AMOUNT_PRIOR_N = 3.0            # pseudo-visits for the conjugate normal shrinkage of per-vendor amounts

# ----------------------------------------------------------------------------------------------
# Feature lineage: name -> (dictionary fields, dictionary status, group, plain-English definition)
# ----------------------------------------------------------------------------------------------
LINEAGE = {
    "amount_gbp": ("transaction_amt, currency, fx_flag", "Have", "Amount", "Transaction value converted to GBP (debits positive, refunds negative)."),
    "amount_log": ("transaction_amt", "Have", "Amount", "log(1 + amount) - the heavy tail is modelled on the log scale."),
    "amount_is_round": ("transaction_amt", "Have", "Amount", "Amount is a round figure (multiple of 10 / 50 / 100) - typical of cash-out and card-present fraud."),
    "amount_pct_of_balance": ("transaction_amt, balance_or_limit", "Add", "Amount", "Amount as a fraction of available funds / limit at transaction time."),
    "amount_z_vs_user_vendor": ("amount_z_vs_user_vendor", "Derived", "Customer habit", "Standardised log-amount versus this customer's own history at this merchant (conjugate normal shrinkage to the customer's MCC history, then to the population)."),
    "amount_z_vs_user": ("transaction_amt, person_id", "Derived", "Customer habit", "Standardised log-amount versus everything this customer has ever spent."),
    "is_new_vendor_for_user": ("is_new_vendor_for_user", "Derived", "Customer habit", "First approved transaction of this customer at this merchant."),
    "visit_count": ("visit_count", "Derived", "Customer habit", "Prior approved transactions of this customer at this merchant."),
    "days_since_last_visit": ("days_since_last_visit", "Derived", "Customer habit", "Days since the customer last transacted with this merchant (missing when never)."),
    "user_vendor_affinity": ("user_vendor_affinity", "Derived", "Customer habit", "Matrix-factorisation cosine between the customer and the merchant in a 12-dimensional latent space, refitted monthly on prior data (missing = cold start)."),
    "is_recurring": ("is_recurring", "Derived", "Customer habit", "Same merchant, near-identical amount, ~monthly cadence: a subscription rather than discretionary spend."),
    "user_vendor_share": ("visit_count, person_id", "Derived", "Customer habit", "Share of the customer's prior transactions made at this merchant."),
    "user_mcc_share": ("mcc, person_id", "Add", "Customer habit", "Share of the customer's prior transactions in this merchant category."),
    "user_channel_share": ("channel / entry_mode, person_id", "Add", "Customer habit", "Share of the customer's prior transactions through this channel (chip, contactless, e-commerce...)."),
    "user_country_share": ("merchant_country, person_id", "Add", "Customer habit", "Share of the customer's prior transactions in this merchant country."),
    "user_hour_share": ("transaction_timestamp, person_id", "Add", "Customer habit", "Share of the customer's prior transactions in the same 3-hour window of the day."),
    "user_dow_share": ("day_of_week, person_id", "Derived", "Customer habit", "Share of the customer's prior transactions on this weekday."),
    "user_txn_count_prior": ("person_id", "Have", "Customer habit", "Number of prior approved transactions of the customer (confidence weight for all habit features)."),
    "descriptor_deviation": ("descriptor_deviation", "Derived", "Merchant", "1 - similarity between this descriptor (digits masked) and the merchant's usual descriptor pattern."),
    "hour_local": ("transaction_timestamp", "Add", "Time", "Local hour of day at the merchant."),
    "is_night": ("transaction_timestamp", "Add", "Time", "Transaction between 00:00 and 05:59 local time."),
    "day_of_week": ("day_of_week", "Derived", "Time", "Weekday (0 = Monday)."),
    "is_weekend": ("day_of_week", "Derived", "Time", "Saturday or Sunday."),
    "day_of_month": ("day_of_month", "Derived", "Time", "Day of the month (payday and billing-cycle effects)."),
    "log_secs_since_prev_txn": ("transaction_timestamp, person_id", "Add", "Velocity", "log seconds since the customer's previous authorisation."),
    "n_txn_10m": ("transaction_timestamp, person_id", "Add", "Velocity", "Customer authorisations in the previous 10 minutes."),
    "n_txn_1h": ("transaction_timestamp, person_id", "Add", "Velocity", "Customer authorisations in the previous hour."),
    "n_txn_24h": ("transaction_timestamp, person_id", "Add", "Velocity", "Customer authorisations in the previous 24 hours."),
    "n_txn_7d": ("transaction_timestamp, person_id", "Add", "Velocity", "Customer authorisations in the previous 7 days."),
    "amt_sum_24h": ("transaction_timestamp, transaction_amt", "Add", "Velocity", "Approved spend in the previous 24 hours (GBP)."),
    "n_vendors_24h": ("vendor_id, transaction_timestamp", "Add", "Velocity", "Distinct merchants in the previous 24 hours."),
    "n_new_vendors_24h": ("is_new_vendor_for_user, transaction_timestamp", "Derived", "Velocity", "First-time merchants visited in the previous 24 hours."),
    "n_declines_1h": ("auth_result, transaction_timestamp", "Add", "Velocity", "Declined authorisations in the previous hour (card testing signature)."),
    "n_declines_24h": ("auth_result, transaction_timestamp", "Add", "Velocity", "Declined authorisations in the previous 24 hours."),
    "dist_from_home_km": ("merchant_city, cardholder_home_geo", "Add", "Geography", "Great-circle distance from the customer's home to the merchant location."),
    "is_cross_border": ("merchant_country, cardholder_home_geo", "Add", "Geography", "Merchant registered outside the customer's home country."),
    "fx_flag": ("fx_flag", "Add", "Geography", "A foreign-exchange conversion was applied."),
    "travel_speed_kmh": ("merchant_city, transaction_timestamp", "Add", "Geography", "Implied speed between consecutive card-present locations (impossible-travel detection)."),
    "is_online_channel": ("channel / entry_mode", "Add", "Channel", "E-commerce or phone order (card-not-present)."),
    "sca_applied": ("3ds_flag / sca_applied", "Add", "Channel", "Strong customer authentication was performed."),
    "is_new_device": ("device_id", "Add", "Device", "Device fingerprint never seen before for this customer (card-not-present only)."),
    "device_prior_count": ("device_id", "Add", "Device", "Prior transactions of the customer from this device."),
    "days_since_account_event": ("recent_account_events", "Add", "Account", "Days since the last non-monetary event (password reset, new payee, contact change...)."),
    "n_account_events_7d": ("recent_account_events", "Add", "Account", "Non-monetary account events in the previous 7 days (account-takeover precursor)."),
    "card_age_days": ("card_age", "Add", "Account", "Days since the card was issued or re-issued."),
    "account_tenure_days": ("account_tenure", "Add", "Account", "Length of the customer relationship in days."),
    "person_prior_fraud_count": ("fraud_label, label_known_date, person_id", "Add", "Account", "Confirmed fraud events on this customer known before the scoring time."),
    "vendor_fraud_rate_asof": ("merchant_risk_stats, fraud_label, dispute_date", "Add", "Merchant", "Smoothed confirmed-fraud rate of the merchant using only labels matured before the scoring time."),
    "vendor_dispute_rate_asof": ("merchant_risk_stats, dispute_date", "Add", "Merchant", "Smoothed dispute rate (any reason) of the merchant, as of scoring time."),
    "mcc_fraud_rate_asof": ("mcc, fraud_label", "Add", "Merchant", "Smoothed confirmed-fraud rate of the merchant category, as of scoring time (back-off for unseen merchants)."),
    "vendor_txn_count_asof_log": ("vendor_id", "Have", "Merchant", "log(1 + prior transactions at the merchant across all customers)."),
    "vendor_age_days": ("vendor_id, transaction_timestamp", "Have", "Merchant", "Days since the merchant was first seen in the portfolio (left-censored merchants treated as established)."),
    "is_aggregator": ("vendor_id", "Have", "Merchant", "Payment aggregator collapsing many real merchants into one ID."),
    "vendor_is_online": ("vendor_id, channel / entry_mode", "Add", "Merchant", "Merchant primarily trades card-not-present."),
}
for ch in C.CHANNELS:
    LINEAGE[f"channel_{ch}"] = ("channel / entry_mode", "Add", "Channel", f"Entry mode is '{ch}'.")
for m, (desc, *_r) in C.MCC_TABLE.items():
    LINEAGE[f"mcc_{m}"] = ("mcc", "Add", "Merchant category", f"MCC {m}: {desc}.")

MODEL_FEATURES = list(LINEAGE)
LINEAGE_DF = pd.DataFrame([dict(feature=k, source_fields=v[0], dictionary_status=v[1], group=v[2], definition=v[3])
                           for k, v in LINEAGE.items()])


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


_DIGITS = re.compile(r"\d")


def mask_descriptor(s: str) -> str:
    return _DIGITS.sub("#", str(s).upper()).strip()


@lru_cache(maxsize=200_000)
def _desc_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def descriptor_deviation(descriptor: str, pattern: str | None) -> float:
    if pattern is None or pattern == "":
        return np.nan
    return round(1.0 - _desc_ratio(mask_descriptor(descriptor), pattern), 4)


def _window_counts(t: np.ndarray, flags: np.ndarray, window: float) -> np.ndarray:
    """For sorted times t: number of earlier rows (strictly before) within `window` seconds with flag=1."""
    cs = np.concatenate([[0.0], np.cumsum(flags)])
    lo = np.searchsorted(t, t - window, side="right")
    idx = np.arange(len(t))
    return cs[idx] - cs[lo]


# ----------------------------------------------------------------------------------------------
# per-customer routine (batch and live)
# ----------------------------------------------------------------------------------------------
def person_features(g: pd.DataFrame, ev_t: np.ndarray, lk: dict) -> pd.DataFrame:
    """g: one customer's authorisation rows sorted by t_sec. ev_t: sorted event times (seconds).
    lk: population lookups (mcc log-amount mean/sd)."""
    n = len(g)
    t = g["t_sec"].to_numpy(dtype=float)
    amt = g["transaction_amt"].to_numpy(dtype=float)
    approved = (g["auth_result"].to_numpy() == "approved")
    declined = ~approved
    vendor = g["vendor_id"].to_numpy()
    mcc = g["mcc"].to_numpy()
    channel = g["channel"].to_numpy()
    country = g["merchant_country"].to_numpy()
    hour = g["hour_local"].to_numpy(dtype=float)
    dow = g["day_of_week"].to_numpy()
    la = np.log1p(np.clip(amt, 0, None))
    out = pd.DataFrame(index=g.index)

    ok = approved.astype(float)
    prior_ok = np.concatenate([[0.0], np.cumsum(ok)[:-1]])
    out["user_txn_count_prior"] = prior_ok

    # --- per-vendor familiarity -----------------------------------------------------------
    gv = pd.Series(ok, index=g.index).groupby(vendor)
    visit = (gv.cumsum() - ok).to_numpy()
    out["visit_count"] = visit
    out["is_new_vendor_for_user"] = (visit == 0).astype(int)
    t_ok = pd.Series(np.where(approved, t, np.nan), index=g.index)
    prev_ok_t = t_ok.groupby(vendor).shift(1).groupby(vendor).ffill().to_numpy()
    out["days_since_last_visit"] = (t - prev_ok_t) / DAY
    a_ok = pd.Series(np.where(approved, amt, np.nan), index=g.index)
    prev_ok_amt = a_ok.groupby(vendor).shift(1).groupby(vendor).ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        same_amt = np.abs(amt - prev_ok_amt) / np.maximum(np.abs(amt), 0.01) <= 0.02
    dsl = out["days_since_last_visit"].to_numpy()
    out["is_recurring"] = ((visit >= 2) & same_amt & (((dsl >= 26) & (dsl <= 35)) | ((dsl >= 5.5) & (dsl <= 8.5)))).astype(int)
    out["user_vendor_share"] = (visit + 0.5) / (prior_ok + 1.0)

    # --- conjugate normal shrinkage of log-amount: pair -> customer x MCC -> population MCC ---
    la_ok = np.where(approved, la, 0.0)
    la2_ok = np.where(approved, la * la, 0.0)
    s1_pair = (pd.Series(la_ok, index=g.index).groupby(vendor).cumsum() - la_ok).to_numpy()
    s2_pair = (pd.Series(la2_ok, index=g.index).groupby(vendor).cumsum() - la2_ok).to_numpy()
    n_mcc = (pd.Series(ok, index=g.index).groupby(mcc).cumsum() - ok).to_numpy()
    s1_mcc = (pd.Series(la_ok, index=g.index).groupby(mcc).cumsum() - la_ok).to_numpy()
    pop_mu = np.array([lk["mcc_mu"].get(m, lk["pop_mu"]) for m in mcc])
    pop_sd = np.array([lk["mcc_sd"].get(m, lk["pop_sd"]) for m in mcc])
    k0 = AMOUNT_PRIOR_N
    with np.errstate(invalid="ignore", divide="ignore"):
        m_user_mcc = np.where(n_mcc > 0, (k0 * pop_mu + s1_mcc) / (k0 + n_mcc), pop_mu)
        post_mu = (k0 * m_user_mcc + s1_pair) / (k0 + visit)
        var_pair = np.where(visit >= 2, np.maximum(s2_pair / np.maximum(visit, 1) - (s1_pair / np.maximum(visit, 1)) ** 2, 0.0), 0.0)
        var_pooled = (k0 * pop_sd ** 2 + visit * var_pair) / (k0 + visit)
        pred_sd = np.sqrt(np.maximum(var_pooled, 0.03) * (1 + 1 / (k0 + visit)))
        out["amount_z_vs_user_vendor"] = (la - post_mu) / pred_sd
        out["_pair_post_mu"] = post_mu
        out["_pair_pred_sd"] = pred_sd
        # customer-wide
        s1_u = np.concatenate([[0.0], np.cumsum(la_ok)[:-1]])
        s2_u = np.concatenate([[0.0], np.cumsum(la2_ok)[:-1]])
        mu_u = (k0 * lk["pop_mu"] + s1_u) / (k0 + prior_ok)
        var_u = np.where(prior_ok >= 2, np.maximum(s2_u / np.maximum(prior_ok, 1) - (s1_u / np.maximum(prior_ok, 1)) ** 2, 0.0), 0.0)
        var_u = (k0 * lk["pop_sd"] ** 2 + prior_ok * var_u) / (k0 + prior_ok)
        out["amount_z_vs_user"] = (la - mu_u) / np.sqrt(np.maximum(var_u, 0.03) * (1 + 1 / (k0 + prior_ok)))

    # --- profile shares (what does this customer usually do?) --------------------------------
    def share(key):
        cnt = (pd.Series(ok, index=g.index).groupby(key).cumsum() - ok).to_numpy()
        return (cnt + 0.5) / (prior_ok + 1.0)
    out["user_mcc_share"] = share(mcc)
    out["user_channel_share"] = share(channel)
    out["user_country_share"] = share(country)
    out["user_hour_share"] = share((hour // 3).astype(int))
    out["user_dow_share"] = share(dow)

    # --- device ----------------------------------------------------------------------------
    dev = g["device_id"].to_numpy(dtype=object)
    has_dev = pd.notna(dev)
    dev_key = np.where(has_dev, dev, "__none__")
    dev_prior = (pd.Series(np.ones(n), index=g.index).groupby(dev_key).cumsum() - 1).to_numpy()
    out["device_prior_count"] = np.where(has_dev, dev_prior, np.nan)
    out["is_new_device"] = np.where(has_dev, (dev_prior == 0).astype(float), np.nan)

    # --- velocity --------------------------------------------------------------------------
    prev_t = np.concatenate([[np.nan], t[:-1]])
    out["log_secs_since_prev_txn"] = np.log1p(t - prev_t)
    ones = np.ones(n)
    out["n_txn_10m"] = _window_counts(t, ones, 600)
    out["n_txn_1h"] = _window_counts(t, ones, 3600)
    out["n_txn_24h"] = _window_counts(t, ones, DAY)
    out["n_txn_7d"] = _window_counts(t, ones, 7 * DAY)
    out["amt_sum_24h"] = _window_counts(t, np.where(approved, np.clip(amt, 0, None), 0.0), DAY)
    out["n_new_vendors_24h"] = _window_counts(t, (visit == 0).astype(float), DAY)
    out["n_declines_1h"] = _window_counts(t, declined.astype(float), 3600)
    out["n_declines_24h"] = _window_counts(t, declined.astype(float), DAY)
    lo24 = np.searchsorted(t, t - DAY, side="right")
    nv = np.zeros(n)
    for i in range(n):
        if lo24[i] < i:
            nv[i] = len(set(vendor[lo24[i]:i]))
    out["n_vendors_24h"] = nv

    # --- geography -------------------------------------------------------------------------
    mlat = g["merchant_lat"].to_numpy(dtype=float)
    mlon = g["merchant_lon"].to_numpy(dtype=float)
    hlat = g["cardholder_home_lat"].to_numpy(dtype=float)
    hlon = g["cardholder_home_lon"].to_numpy(dtype=float)
    out["dist_from_home_km"] = haversine(hlat, hlon, mlat, mlon)
    out["is_cross_border"] = (country != g["cardholder_home_country"].to_numpy()).astype(int)
    cp = np.isin(channel, list(C.CARD_PRESENT_CHANNELS))
    speed = np.full(n, np.nan)
    cp_idx = np.where(cp)[0]
    if len(cp_idx) > 1:
        i1, i0 = cp_idx[1:], cp_idx[:-1]
        d = haversine(mlat[i0], mlon[i0], mlat[i1], mlon[i1])
        hrs = np.maximum((t[i1] - t[i0]) / 3600.0, 1 / 60)
        speed[i1] = d / hrs
    out["travel_speed_kmh"] = speed

    # --- account events & prior fraud --------------------------------------------------------
    if len(ev_t):
        pos = np.searchsorted(ev_t, t, side="left")           # events strictly before t
        last_ev = np.where(pos > 0, ev_t[np.maximum(pos - 1, 0)], np.nan)
        out["days_since_account_event"] = np.where(pos > 0, (t - last_ev) / DAY, np.nan)
        lo7 = np.searchsorted(ev_t, t - 7 * DAY, side="right")
        out["n_account_events_7d"] = pos - lo7
    else:
        out["days_since_account_event"] = np.nan
        out["n_account_events_7d"] = 0.0
    known = g["label_known_sec"].to_numpy(dtype=float)
    fraud_known = np.sort(known[(g["fraud_label"].to_numpy() == 1) & np.isfinite(known)])
    out["person_prior_fraud_count"] = np.searchsorted(fraud_known, t, side="left") if len(fraud_known) else 0.0
    return out


# ----------------------------------------------------------------------------------------------
# merchant-level as-of statistics (batch and live share `vendor_asof`)
# ----------------------------------------------------------------------------------------------
def vendor_asof(t_sorted: np.ndarray, fraud_known_sorted: np.ndarray, dispute_sorted: np.ndarray,
                t_query: np.ndarray, pop_rate: float, prior_weight: float = PRIOR_WEIGHT_VENDOR):
    """Counts known before t_query: mature denominators use transactions older than the maturity window."""
    n_mature = np.searchsorted(t_sorted, t_query - C.LABEL_MATURITY_DAYS * DAY, side="left").astype(float)
    k_fraud = np.searchsorted(fraud_known_sorted, t_query, side="left").astype(float)
    k_disp = np.searchsorted(dispute_sorted, t_query, side="left").astype(float)
    k_fraud = np.minimum(k_fraud, n_mature)
    k_disp = np.minimum(k_disp, n_mature)
    fraud_rate = (k_fraud + prior_weight * pop_rate) / (n_mature + prior_weight)
    disp_rate = (k_disp + prior_weight * pop_rate * 1.5) / (n_mature + prior_weight)
    n_prior = np.searchsorted(t_sorted, t_query, side="left").astype(float)
    return fraud_rate, disp_rate, n_mature, k_fraud, n_prior


def _asof_tables(tx: pd.DataFrame, key: str) -> dict:
    """Sorted arrays per key needed by `vendor_asof`."""
    tabs = {}
    auth = tx[tx.transaction_type == "authorisation"]
    known = auth["label_known_sec"].to_numpy(dtype=float)
    disp = auth["dispute_sec"].to_numpy(dtype=float)
    fl = auth["fraud_label"].to_numpy()
    for k, idx in auth.groupby(key).indices.items():
        t_sorted = np.sort(auth["t_sec"].to_numpy()[idx])
        fk = known[idx]
        fk = np.sort(fk[(fl[idx] == 1) & np.isfinite(fk)])
        ds = disp[idx]
        ds = np.sort(ds[np.isfinite(ds)])
        tabs[k] = (t_sorted, fk, ds)
    return tabs


def population_asof(tx: pd.DataFrame, t_query: np.ndarray) -> np.ndarray:
    auth = tx[tx.transaction_type == "authorisation"]
    t_sorted = np.sort(auth["t_sec"].to_numpy())
    known = auth["label_known_sec"].to_numpy(dtype=float)
    fk = np.sort(known[(auth["fraud_label"].to_numpy() == 1) & np.isfinite(known)])
    n_mature = np.searchsorted(t_sorted, t_query - C.LABEL_MATURITY_DAYS * DAY, side="left").astype(float)
    k = np.searchsorted(fk, t_query, side="left").astype(float)
    return (k + 5 * 0.008) / (n_mature + 5)


# ----------------------------------------------------------------------------------------------
# user-vendor affinity (matrix factorisation, monthly refit on prior data)
# ----------------------------------------------------------------------------------------------
def fit_affinity(tx_prior: pd.DataFrame, k: int = 12) -> dict | None:
    ok = tx_prior[(tx_prior.transaction_type == "authorisation") & (tx_prior.auth_result == "approved") & (tx_prior.fraud_label == 0)]
    if len(ok) < 2000:
        return None
    cnt = ok.groupby(["person_id", "vendor_id"]).size().reset_index(name="n")
    users = {u: i for i, u in enumerate(cnt.person_id.unique())}
    vends = {v: i for i, v in enumerate(cnt.vendor_id.unique())}
    if len(users) < k + 2 or len(vends) < k + 2:
        return None
    M = csr_matrix((np.log1p(cnt.n.values), (cnt.person_id.map(users).values, cnt.vendor_id.map(vends).values)),
                   shape=(len(users), len(vends)))
    U, S, Vt = svds(M.astype(float), k=k)
    Uf = U * S                                  # user factors scaled by singular values
    Vf = Vt.T
    return dict(users=users, vendors=vends, U=Uf / np.maximum(np.linalg.norm(Uf, axis=1, keepdims=True), 1e-9),
                V=Vf / np.maximum(np.linalg.norm(Vf, axis=1, keepdims=True), 1e-9), k=k)


def affinity_lookup(model: dict | None, person_ids, vendor_ids) -> np.ndarray:
    n = len(person_ids)
    if model is None:
        return np.full(n, np.nan)
    ui = np.array([model["users"].get(p, -1) for p in person_ids])
    vi = np.array([model["vendors"].get(v, -1) for v in vendor_ids])
    valid = (ui >= 0) & (vi >= 0)
    outv = np.full(n, np.nan)
    if valid.any():
        outv[valid] = np.einsum("ij,ij->i", model["U"][ui[valid]], model["V"][vi[valid]])
    return outv


# ----------------------------------------------------------------------------------------------
# batch build
# ----------------------------------------------------------------------------------------------
def prepare_base(tx: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    tx = tx.copy()
    tx["hour_local"] = tx.transaction_timestamp_local.dt.hour + tx.transaction_timestamp_local.dt.minute / 60.0
    tx["day_of_week"] = tx.transaction_timestamp_local.dt.dayofweek
    tx["day_of_month"] = tx.transaction_timestamp_local.dt.day
    tx["label_known_sec"] = (tx.label_known_date - start).dt.total_seconds()
    tx["dispute_sec"] = (tx.dispute_date - start).dt.total_seconds()
    return tx


def population_lookups(tx_train: pd.DataFrame) -> dict:
    ok = tx_train[(tx_train.transaction_type == "authorisation") & (tx_train.auth_result == "approved")]
    la = np.log1p(ok.transaction_amt.clip(lower=0))
    grp = la.groupby(ok.mcc)
    return dict(pop_mu=float(la.mean()), pop_sd=float(la.std()), mcc_mu=grp.mean().to_dict(), mcc_sd=grp.std().fillna(la.std()).to_dict())


def descriptor_patterns(tx: pd.DataFrame, n_first: int = 50) -> dict:
    auth = tx[tx.transaction_type == "authorisation"].sort_values("t_sec")
    pats = {}
    for v, grp in auth.groupby("vendor_id"):
        masked = grp.transaction_detail_char.head(n_first).map(mask_descriptor)
        pats[v] = masked.mode().iloc[0] if len(masked) else ""
    return pats


def vendor_first_seen(tx: pd.DataFrame) -> dict:
    first = tx.groupby("vendor_id").t_sec.min()
    # left-censoring: merchants active in the first 30 days of the extract are treated as established
    return {v: (t - 365 * DAY if t < 30 * DAY else t) for v, t in first.items()}


def build_features(tx: pd.DataFrame, events: pd.DataFrame, start: pd.Timestamp, train_end_day: int, log=print) -> tuple[pd.DataFrame, dict]:
    tx = prepare_base(tx, start)
    auth = tx[tx.transaction_type == "authorisation"].sort_values(["person_id", "t_sec"]).copy()
    train = auth[(auth.t_sec < train_end_day * DAY) & auth.label_mature]
    lk = population_lookups(train)
    lk["descriptor_patterns"] = descriptor_patterns(tx[tx.t_sec < train_end_day * DAY])
    for v in tx.vendor_id.unique():
        lk["descriptor_patterns"].setdefault(v, "")
    lk["vendor_first_seen"] = vendor_first_seen(tx)
    log(f"  population lookups ready ({len(lk['descriptor_patterns'])} merchant descriptor patterns)")

    # per-customer features
    ev = events.copy()
    ev["t_sec"] = (ev.event_ts - start).dt.total_seconds()
    ev_by_p = {p: np.sort(g.t_sec.to_numpy()) for p, g in ev.groupby("person_id")}
    parts = []
    for pid, g in auth.groupby("person_id", sort=False):
        parts.append(person_features(g, ev_by_p.get(pid, np.array([])), lk))
    F = pd.concat(parts).reindex(auth.index)
    log(f"  customer-history features: {F.shape[1]} columns for {len(F):,} authorisations")

    # merchant as-of statistics
    tabs_v = _asof_tables(tx, "vendor_id")
    tabs_m = _asof_tables(tx, "mcc")
    pop_rate = float(train.fraud_label.mean())
    lk["pop_fraud_rate"] = pop_rate
    lk["asof_vendor"] = tabs_v
    lk["asof_mcc"] = tabs_m
    vf = np.full(len(auth), np.nan); vd = vf.copy(); vn = vf.copy(); mf = vf.copy()
    tq = auth.t_sec.to_numpy()
    for v, idx in auth.groupby("vendor_id").indices.items():
        fr, dr, n_mat, kf, n_prior = vendor_asof(*tabs_v[v], tq[idx], pop_rate)
        vf[idx], vd[idx], vn[idx] = fr, dr, n_prior
    for m, idx in auth.groupby("mcc").indices.items():
        fr, dr, n_mat, kf, n_prior = vendor_asof(*tabs_m[m], tq[idx], pop_rate, prior_weight=200.0)
        mf[idx] = fr
    F["vendor_fraud_rate_asof"] = vf
    F["vendor_dispute_rate_asof"] = vd
    F["mcc_fraud_rate_asof"] = mf
    F["vendor_txn_count_asof_log"] = np.log1p(vn)
    F["vendor_age_days"] = (tq - auth.vendor_id.map(lk["vendor_first_seen"]).to_numpy()) / DAY
    F["pop_fraud_rate_asof"] = population_asof(tx, tq)
    log("  merchant as-of risk statistics done")

    # descriptor deviation
    pats = lk["descriptor_patterns"]
    F["descriptor_deviation"] = [descriptor_deviation(d, pats.get(v)) for d, v in zip(auth.transaction_detail_char, auth.vendor_id)]

    # affinity with monthly refits on strictly prior data
    month = (auth.t_sec // (30 * DAY)).astype(int)
    aff = np.full(len(auth), np.nan)
    for m in sorted(month.unique()):
        model = fit_affinity(tx[tx.t_sec < m * 30 * DAY])
        idx = np.where(month.to_numpy() == m)[0]
        aff[idx] = affinity_lookup(model, auth.person_id.to_numpy()[idx], auth.vendor_id.to_numpy()[idx])
    F["user_vendor_affinity"] = aff
    lk["affinity_model"] = fit_affinity(tx)         # as-of-now model for live scoring
    log("  user-vendor affinity (monthly matrix factorisation refits) done")

    # plain transaction attributes
    F["amount_gbp"] = auth.transaction_amt.to_numpy()
    F["amount_log"] = np.log1p(auth.transaction_amt.clip(lower=0).to_numpy())
    a = auth.transaction_amt.to_numpy()
    F["amount_is_round"] = ((np.abs(a) >= 10) & (np.isclose(np.mod(a, 10), 0) | np.isclose(np.mod(a, 10), 10))).astype(int)
    F["amount_pct_of_balance"] = a / np.maximum(auth.balance_or_limit.to_numpy(), 1.0)
    F["hour_local"] = auth.hour_local.to_numpy()
    F["is_night"] = (auth.hour_local.to_numpy() < 6).astype(int)
    F["day_of_week"] = auth.day_of_week.to_numpy()
    F["is_weekend"] = (auth.day_of_week.to_numpy() >= 5).astype(int)
    F["day_of_month"] = auth.day_of_month.to_numpy()
    F["fx_flag"] = auth.fx_flag.astype(int).to_numpy()
    F["is_online_channel"] = auth.channel.isin(["ecommerce", "phone_order"]).astype(int).to_numpy()
    F["sca_applied"] = auth.sca_applied.astype(int).to_numpy()
    F["card_age_days"] = auth.card_age_days.to_numpy()
    F["account_tenure_days"] = auth.account_tenure_days.to_numpy()
    F["is_aggregator"] = auth.is_aggregator.astype(int).to_numpy()
    F["vendor_is_online"] = auth.is_online.astype(int).to_numpy()
    for ch in C.CHANNELS:
        F[f"channel_{ch}"] = (auth.channel.to_numpy() == ch).astype(int)
    for m in C.MCC_TABLE:
        F[f"mcc_{m}"] = (auth.mcc.to_numpy() == m).astype(int)
    F = F.loc[:, ~F.columns.str.startswith("_") | F.columns.isin(["_pair_post_mu", "_pair_pred_sd"])]
    keep = ["transaction_id", "person_id", "vendor_id", "t_sec", "transaction_timestamp", "fraud_label", "fraud_truth",
            "fraud_typology", "label_mature", "label_status", "transaction_amt", "channel", "mcc", "merchant_country", "auth_result"]
    out = pd.concat([auth[keep], F], axis=1).sort_values("t_sec").reset_index(drop=True)
    return out, lk


# ----------------------------------------------------------------------------------------------
# live scoring: same routine on (history + incoming transaction)
# ----------------------------------------------------------------------------------------------
def features_for_new_transaction(new: dict, history: pd.DataFrame, ev_t: np.ndarray, lk: dict, start: pd.Timestamp) -> pd.Series:
    """history: this customer's earlier authorisation rows (prepared with `prepare_base`)."""
    row = pd.DataFrame([new])
    row["transaction_timestamp_local"] = pd.to_datetime(row["transaction_timestamp_local"])
    row["hour_local"] = row.transaction_timestamp_local.dt.hour + row.transaction_timestamp_local.dt.minute / 60.0
    row["day_of_week"] = row.transaction_timestamp_local.dt.dayofweek
    row["day_of_month"] = row.transaction_timestamp_local.dt.day
    row["label_known_sec"] = np.nan
    row["dispute_sec"] = np.nan
    row["fraud_label"] = 0
    cols = ["t_sec", "transaction_amt", "auth_result", "vendor_id", "mcc", "channel", "merchant_country", "hour_local", "day_of_week",
            "day_of_month", "device_id", "merchant_lat", "merchant_lon", "cardholder_home_lat", "cardholder_home_lon",
            "cardholder_home_country", "label_known_sec", "fraud_label"]
    hist = history[history.t_sec < new["t_sec"]].sort_values("t_sec")
    g = pd.concat([hist[cols], row[cols]], ignore_index=True)
    g["auth_result"] = g["auth_result"].astype(object)
    g["device_id"] = g["device_id"].astype(object)
    F = person_features(g, ev_t[ev_t < new["t_sec"]] if len(ev_t) else ev_t, lk).iloc[-1].copy()
    tq = np.array([new["t_sec"]], dtype=float)
    pop_rate = lk["pop_fraud_rate"]
    v = new["vendor_id"]
    if v in lk["asof_vendor"]:
        fr, dr, n_mat, kf, n_prior = vendor_asof(*lk["asof_vendor"][v], tq, pop_rate)
        F["vendor_fraud_rate_asof"], F["vendor_dispute_rate_asof"] = float(fr[0]), float(dr[0])
        F["vendor_txn_count_asof_log"] = float(np.log1p(n_prior[0]))
        F["_vendor_mature_n"], F["_vendor_known_fraud"] = float(n_mat[0]), float(kf[0])
    else:   # unseen merchant: back off to the population prior
        F["vendor_fraud_rate_asof"], F["vendor_dispute_rate_asof"] = pop_rate, pop_rate * 1.5
        F["vendor_txn_count_asof_log"] = 0.0
        F["_vendor_mature_n"], F["_vendor_known_fraud"] = 0.0, 0.0
    m = new["mcc"]
    if m in lk["asof_mcc"]:
        fr, *_ = vendor_asof(*lk["asof_mcc"][m], tq, pop_rate, prior_weight=200.0)
        F["mcc_fraud_rate_asof"] = float(fr[0])
    else:
        F["mcc_fraud_rate_asof"] = pop_rate
    fs = lk["vendor_first_seen"].get(v, new["t_sec"])
    F["vendor_age_days"] = (new["t_sec"] - fs) / DAY
    F["descriptor_deviation"] = descriptor_deviation(new["transaction_detail_char"], lk["descriptor_patterns"].get(v))
    F["user_vendor_affinity"] = float(affinity_lookup(lk["affinity_model"], [new["person_id"]], [v])[0])
    a = float(new["transaction_amt"])
    F["amount_gbp"] = a
    F["amount_log"] = float(np.log1p(max(a, 0)))
    F["amount_is_round"] = int(abs(a) >= 10 and (np.isclose(a % 10, 0) or np.isclose(a % 10, 10)))
    F["amount_pct_of_balance"] = a / max(float(new["balance_or_limit"]), 1.0)
    F["hour_local"] = float(row.hour_local.iloc[0])
    F["is_night"] = int(F["hour_local"] < 6)
    F["day_of_week"] = int(row.day_of_week.iloc[0])
    F["is_weekend"] = int(F["day_of_week"] >= 5)
    F["day_of_month"] = int(row.day_of_month.iloc[0])
    F["fx_flag"] = int(bool(new["fx_flag"]))
    F["is_online_channel"] = int(new["channel"] in ("ecommerce", "phone_order"))
    F["sca_applied"] = int(bool(new["sca_applied"]))
    F["card_age_days"] = float(new["card_age_days"])
    F["account_tenure_days"] = float(new["account_tenure_days"])
    F["is_aggregator"] = int(bool(new["is_aggregator"]))
    F["vendor_is_online"] = int(bool(new["is_online"]))
    for ch in C.CHANNELS:
        F[f"channel_{ch}"] = int(new["channel"] == ch)
    for mm in C.MCC_TABLE:
        F[f"mcc_{mm}"] = int(m == mm)
    return F
