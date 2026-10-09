"""Artifact store and live transaction scoring with a complete, serialisable audit trace."""
from __future__ import annotations

import json
import pickle
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit

from . import config as C, features as FE, models as M

ART = C.ARTIFACT_DIR


class Store:
    """Loads every artifact once; the app wraps this in st.cache_resource."""

    def __init__(self):
        self.start = pd.Timestamp(C.AS_OF_DATE) - pd.Timedelta(days=C.HISTORY_DAYS)
        d = ART / "data"
        self.tx = pd.read_parquet(d / "transactions.parquet")
        self.events = pd.read_parquet(d / "events.parquet")
        self.persons = pd.read_parquet(d / "persons.parquet").set_index("person_id")
        self.cards = pd.read_parquet(d / "cards.parquet")
        self.vendors = pd.read_parquet(d / "vendors.parquet").set_index("vendor_id")
        self.episodes = pd.read_parquet(d / "episodes.parquet")
        self.features = pd.read_parquet(ART / "features.parquet")
        self.scores = pd.read_parquet(ART / "scores.parquet")
        with open(ART / "lookups.pkl", "rb") as fh:
            self.lk = pickle.load(fh)
        with open(ART / "models.pkl", "rb") as fh:
            self.models = pickle.load(fh)
        with open(ART / "bayes.pkl", "rb") as fh:
            self.bayes = pickle.load(fh)
        with open(ART / "evaluation.json") as fh:
            self.evaluation = json.load(fh)
        with open(ART / "registry.json") as fh:
            self.registry = json.load(fh)
        # prepared history for live scoring
        self.txp = FE.prepare_base(self.tx, self.start)
        self.auth = self.txp[self.txp.transaction_type == "authorisation"]
        self.auth_by_person = {p: g for p, g in self.auth.groupby("person_id")}
        ev = self.events.copy()
        ev["t_sec"] = (ev.event_ts - self.start).dt.total_seconds()
        self.ev_by_person = {p: np.sort(g.t_sec.to_numpy()) for p, g in ev.groupby("person_id")}
        self.as_of = pd.Timestamp(C.AS_OF_DATE)
        self._affinity_by_month = {}

    def affinity_model(self, t_sec: float) -> dict | None:
        """The user-vendor factorisation the batch build used for this month (fitted on strictly earlier data), so that
        replaying a historical transaction reproduces its batch score instead of peeking at later behaviour."""
        m = FE.affinity_month(t_sec)
        if m * 30 * FE.DAY >= C.HISTORY_DAYS * FE.DAY:     # at or after the end of history: the as-of-now model
            return self.lk["affinity_model"]
        if m not in self._affinity_by_month:
            self._affinity_by_month[m] = FE.fit_affinity(self.tx[self.tx.t_sec < m * 30 * FE.DAY])
        return self._affinity_by_month[m]

    # convenience --------------------------------------------------------------------------------
    def person_history(self, person_id: str) -> pd.DataFrame:
        return self.auth_by_person.get(person_id, self.auth.iloc[0:0])

    def vendor_pattern(self, vendor_id: str) -> str:
        return self.lk["descriptor_patterns"].get(vendor_id, "")

    def sample_descriptor(self, vendor_id: str) -> str:
        """A representative genuine descriptor: the first one whose digit-masked form matches the merchant's usual pattern."""
        rows = self.tx[self.tx.vendor_id == vendor_id]
        if len(rows) == 0:
            return ""
        pattern = self.lk["descriptor_patterns"].get(vendor_id, "")
        masked = rows.transaction_detail_char.map(FE.mask_descriptor)
        match = rows.transaction_detail_char[masked == pattern]
        return str(match.iloc[0]) if len(match) else str(rows.transaction_detail_char.iloc[0])


# ----------------------------------------------------------------------------------------------
# building the transaction record
# ----------------------------------------------------------------------------------------------
def make_transaction(store: Store, person_id: str, vendor_id: str | None, amount: float, channel: str, ts_utc: pd.Timestamp,
                     sca_applied: bool = False, device: str | None = "known", descriptor: str | None = None,
                     new_vendor: dict | None = None, card_id: str | None = None, balance: float | None = None,
                     ip_address: str | None = None, currency: str | None = None) -> dict:
    """Assemble the incoming authorisation request exactly as the feature pipeline expects it."""
    p = store.persons.loc[person_id]
    if new_vendor is not None:
        v = dict(new_vendor)
        vendor_id = v.get("vendor_id", "NEW-" + uuid.uuid4().hex[:6].upper())
        city = v.get("merchant_city", "London")
        cc = C.CITIES.get(city, ("GB", 51.5, -0.1, 0, 0))
        v.setdefault("merchant_country", cc[0]); v.setdefault("merchant_lat", cc[1]); v.setdefault("merchant_lon", cc[2])
        v.setdefault("is_online", channel in ("ecommerce", "phone_order", "recurring")); v.setdefault("is_aggregator", False)
        v.setdefault("vendor_name", v.get("vendor_name", "Unknown merchant")); v.setdefault("mcc", "5999")
        tz = cc[4]
    else:
        vr = store.vendors.loc[vendor_id]
        v = dict(merchant_city=vr.merchant_city, merchant_country=vr.merchant_country, merchant_lat=vr.merchant_lat, merchant_lon=vr.merchant_lon,
                 is_online=bool(vr.is_online), is_aggregator=bool(vr.is_aggregator), vendor_name=vr.vendor_name, mcc=str(vr.mcc))
        tz = C.CITIES.get(vr.merchant_city, (None, 0, 0, 0, 0))[4]
    cur = currency or C.COUNTRY_CURRENCY.get(v["merchant_country"], "GBP")
    hist = store.person_history(person_id)
    cards = store.cards[store.cards.person_id == person_id]
    if card_id is None:
        card_id = hist.card_id.iloc[-1] if len(hist) else (cards.card_id.iloc[0] if len(cards) else f"C{person_id[1:]}1")
    ts_utc = pd.Timestamp(ts_utc)
    t_sec = (ts_utc - store.start).total_seconds()
    # card age: from the last known issue date on the card (re-issued cards carry an 'R' suffix)
    if len(hist) and (hist.card_id == card_id).any():
        last = hist[hist.card_id == card_id].iloc[-1]
        card_age = float(last.card_age_days + (t_sec - last.t_sec) / 86400.0)
    else:
        issued = cards[cards.card_id == card_id].issued_date
        card_age = float((ts_utc.normalize() - pd.Timestamp(issued.iloc[0])).days) if len(issued) else 0.0
    tenure = float(p.account_tenure_days - (store.as_of - ts_utc).days)
    devices = hist.device_id.dropna().unique().tolist() if len(hist) else []
    if channel in ("ecommerce", "phone_order", "recurring"):
        if device == "new" or (device == "known" and not devices):
            device_id = "D-" + uuid.uuid4().hex[:8]
        elif device == "known":
            device_id = devices[-1]
        else:
            device_id = device
    else:
        device_id = None
    if descriptor is None:
        descriptor = store.sample_descriptor(vendor_id) if new_vendor is None else f"{v['vendor_name'].upper()[:20]} {v['merchant_city'].upper()[:12]}"
    return dict(
        transaction_id="LIVE-" + uuid.uuid4().hex[:8].upper(), person_id=person_id, card_id=card_id, vendor_id=vendor_id,
        vendor_name=v["vendor_name"], transaction_detail_char=descriptor, transaction_amt=float(amount),
        amount_original=float(round(amount / C.FX_TO_GBP.get(cur, 1.0), 2)), currency=cur, fx_flag=bool(cur != C.HOME_CURRENCY),
        transaction_timestamp=ts_utc, transaction_timestamp_local=ts_utc + pd.Timedelta(hours=float(tz)), tz_offset_hours=float(tz),
        t_sec=float(t_sec), mcc=str(v["mcc"]), channel=channel, auth_result="pending", transaction_type="authorisation",
        merchant_country=v["merchant_country"], merchant_city=v["merchant_city"], merchant_lat=float(v["merchant_lat"]), merchant_lon=float(v["merchant_lon"]),
        is_online=bool(v["is_online"]), is_aggregator=bool(v["is_aggregator"]), cardholder_home_country=C.HOME_COUNTRY,
        cardholder_home_city=p.home_city, cardholder_home_lat=float(p.home_lat), cardholder_home_lon=float(p.home_lon),
        sca_applied=bool(sca_applied), balance_or_limit=float(balance if balance is not None else p.credit_limit * 0.6),
        device_id=device_id, ip_address=ip_address or (p.home_ip if device != "new" else None), card_age_days=card_age,
        account_tenure_days=tenure, is_new_vendor_record=new_vendor is not None,
    )


def transaction_from_row(store: Store, row: pd.Series) -> dict:
    """Replay an existing transaction row from the dataset as an incoming authorisation request."""
    d = row.to_dict()
    d["auth_result"] = "pending"
    d["t_sec"] = float(d["t_sec"])
    d["transaction_timestamp_local"] = pd.Timestamp(d["transaction_timestamp_local"])
    d["transaction_timestamp"] = pd.Timestamp(d["transaction_timestamp"])
    d["is_new_vendor_record"] = False
    return d


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------
def _top(contrib: pd.Series, k: int = 10):
    c = contrib.drop(labels=[i for i in contrib.index if i == "bias"], errors="ignore")
    c = c[c.abs().sort_values(ascending=False).index][:k]
    return [dict(feature=f, contribution=float(v)) for f, v in c.items()]


def score_transaction(store: Store, txn: dict, policy: dict | None = None, friction_cost: float = 8.0, extra_history: pd.DataFrame | None = None,
                      extra_event_times: np.ndarray | None = None) -> dict:
    """extra_history / extra_event_times inject hypothetical prior activity (scenario presets) without touching the
    shared Store, which is cached across all sessions."""
    policy = policy or C.DEFAULT_POLICY
    timings = {}
    t0 = time.perf_counter()
    hist = store.person_history(txn["person_id"])
    if extra_history is not None and len(extra_history):
        hist = pd.concat([hist, extra_history], ignore_index=True).sort_values("t_sec")
    ev_t = store.ev_by_person.get(txn["person_id"], np.array([]))
    if extra_event_times is not None and len(extra_event_times):
        ev_t = np.sort(np.concatenate([ev_t, np.asarray(extra_event_times, dtype=float)]))
    f = FE.features_for_new_transaction(txn, hist, ev_t, store.lk, store.start, affinity_model=store.affinity_model(txn["t_sec"]))
    timings["features_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    X = pd.DataFrame([f[FE.MODEL_FEATURES].astype(float)])

    t1 = time.perf_counter()
    xg, tree, sc, stack = store.models["xgb"], store.models["tree"], store.models["scorecard"], store.models["stacker"]
    p_xgb_raw = float(xg.predict_raw(X)[0])
    p_xgb = float(xg.predict_proba(X)[0])
    shap = xg.contributions(X).iloc[0]
    p_tree = float(tree.predict_proba(X)[0])
    tree_expl = tree.explain(X.iloc[0])
    p_sc = float(sc.predict_proba(X)[0])
    sc_contrib = sc.contributions(X).iloc[0]
    timings["models_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    t2 = time.perf_counter()
    vm, pm = store.bayes["vendor"], store.bayes["person"]
    bv = vm.vendor_rate(txn["vendor_id"], txn["mcc"])
    bp = pm.person_rate(txn["person_id"])
    timings["bayes_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    L = pd.Series({"xgb": M.clip_logit(p_xgb), "tree": M.clip_logit(p_tree), "scorecard": M.clip_logit(p_sc),
                   "bayes_vendor": M.clip_logit(bv["mean"]), "bayes_person": M.clip_logit(bp["mean"])})
    blend = stack.decompose(L)
    p = blend["probability"]
    tier = M.tier_for(p, policy)
    amount = float(txn["transaction_amt"])
    expected_loss_if_approved = p * amount
    expected_cost_if_declined = (1 - p) * (friction_cost + 0.02 * amount)
    decision = dict(tier=tier[0], label=tier[1], status=tier[2], probability=p, thresholds=policy,
                    expected_fraud_loss_if_approved=round(expected_loss_if_approved, 2),
                    expected_cost_if_declined=round(expected_cost_if_declined, 2),
                    economically_decline=bool(expected_loss_if_approved > expected_cost_if_declined),
                    friction_cost=friction_cost)

    feats = {k: (None if pd.isna(f[k]) else float(f[k])) for k in FE.MODEL_FEATURES}
    lineage = FE.LINEAGE
    trace = dict(
        trace_id="TR-" + uuid.uuid4().hex[:10].upper(),
        scored_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        model_version=store.registry["model_version"], app_version=store.registry["app_version"],
        input={k: (v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v) for k, v in txn.items()},
        history=dict(prior_authorisations=int(len(hist[hist.t_sec < txn["t_sec"]])), prior_events=int((ev_t < txn["t_sec"]).sum()),
                     pair_posterior_log_amount=dict(mean=float(f["_pair_post_mu"]), sd=float(f["_pair_pred_sd"])),
                     vendor_mature_n=float(f.get("_vendor_mature_n", np.nan)), vendor_known_fraud=float(f.get("_vendor_known_fraud", np.nan))),
        features=feats,
        feature_lineage={k: dict(source=lineage[k][0], status=lineage[k][1], group=lineage[k][2]) for k in FE.MODEL_FEATURES},
        components=dict(
            xgb=dict(prob_raw=p_xgb_raw, prob=p_xgb, logit=float(L["xgb"]), bias=float(shap["bias"]), top_contributions=_top(shap, 12),
                     all_contributions={k: float(v) for k, v in shap.items()}),
            tree=dict(prob=p_tree, logit=float(L["tree"]), **tree_expl),
            scorecard=dict(prob=p_sc, logit=float(L["scorecard"]), intercept=sc.intercept, top_contributions=_top(sc_contrib, 12)),
            bayes_vendor=dict(prob=bv["mean"], logit=float(L["bayes_vendor"]), **{k: v for k, v in bv.items() if k != "mean"}),
            bayes_person=dict(prob=bp["mean"], logit=float(L["bayes_person"]), **{k: v for k, v in bp.items() if k != "mean"}),
        ),
        blend=dict(intercept=blend["intercept"], weights=stack.weights.to_dict(), inputs=L.to_dict(), contributions=blend["contributions"],
                   logit=blend["logit"], probability=p),
        decision=decision, timings_ms=timings,
    )
    trace["timings_ms"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return trace


def trace_to_json(trace: dict) -> str:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(trace, indent=2, default=default)
