"""Command centre: portfolio health, model health and how a transaction is scored."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta

from fraudlab import config as C, ui
from fraudlab.ui import P, STATUS


def weekly_rates(tx: pd.DataFrame) -> pd.DataFrame:
    auth = tx[(tx.transaction_type == "authorisation") & tx.label_mature]
    w = auth.groupby(auth.transaction_timestamp.dt.to_period("W").dt.start_time).agg(n=("fraud_label", "size"), y=("fraud_label", "sum"),
                                                                                      loss=("transaction_amt", lambda s: 0.0))
    fr = auth[(auth.fraud_label == 1) & (auth.auth_result == "approved")]
    loss = fr.groupby(fr.transaction_timestamp.dt.to_period("W").dt.start_time).transaction_amt.sum()
    w["loss"] = loss.reindex(w.index).fillna(0.0)
    w["rate"] = w.y / w.n
    w["lo"] = beta.ppf(0.025, w.y + 0.5, w.n - w.y + 0.5)
    w["hi"] = beta.ppf(0.975, w.y + 0.5, w.n - w.y + 0.5)
    return w.reset_index(names="week")


def render():
    store = ui.get_store()
    tx, sc, ev, reg = store.tx, store.scores, store.evaluation, store.registry
    policy = ui.get_policy()
    ui.page_header(f"{C.APP_NAME} command centre",
                   f"{C.BANK_NAME} · real-time card fraud detection built on the fraud model data dictionary · as of {reg['as_of_date']} · synthetic demonstration data")

    # ------------------------------------------------------------------ KPIs
    mature = tx[(tx.transaction_type == "authorisation") & tx.label_mature]
    recent = mature[mature.transaction_timestamp >= store.as_of - pd.Timedelta(days=150)]
    fraud_rate = recent.fraud_label.mean()
    n, y = len(recent), int(recent.fraud_label.sum())
    lo, hi = beta.ppf([0.025, 0.975], y + 0.5, n - y + 0.5)
    losses = recent[(recent.fraud_label == 1) & (recent.auth_result == "approved")].transaction_amt.sum()
    live = sc[sc.split == "live"].copy()
    live["tier"] = ui.tiers_now(live.p_blend)          # under the policy currently set in the sidebar
    alerts = live[live.tier.isin(ui.ALERT_TIERS)]
    m = ev["metrics"]["blend"]
    rhat_max = max(h["rhat"] for h in ev["bayes_vendor_hyper"] + ev["bayes_person_hyper"])
    ui.kpi_row([
        dict(label="Confirmed fraud rate", value=ui.fmt_pct(fraud_rate, 2), sub=f"mature window · 95% interval {ui.fmt_pct(lo, 2)}–{ui.fmt_pct(hi, 2)}", status="warning"),
        dict(label="Confirmed fraud losses", value=ui.fmt_gbp(losses), sub=f"{y:,} fraudulent authorisations in the last 90 mature days"),
        dict(label="Blend AUC (test)", value=f"{m['auc_roc']:.3f}", sub=f"PR-AUC {m['auc_pr']:.3f} · KS {m['ks']:.2f}", status="good"),
        dict(label="Fraud caught @ 1% alerts", value=ui.fmt_pct(m['recall@1.0%'], 0), sub=f"{ui.fmt_pct(m.get('value_recall@1.0%', np.nan), 0)} of fraud value · test window"),
        dict(label="Alert rate (live stream)", value=ui.fmt_pct(len(alerts) / max(len(live), 1), 2), sub=f"{len(alerts):,} step-up or decline of {len(live):,} authorisations", status="warning" if len(alerts) / max(len(live), 1) > 0.02 else "good"),
        dict(label="MCMC convergence", value=f"R̂ max {rhat_max:.3f}", sub="4 chains · vendor-in-MCC hierarchy & customer propensity", status="good" if rhat_max < 1.05 else "serious"),
    ])

    # ------------------------------------------------------------------ pipeline
    ui.section("How every authorisation is scored", "One request in, one probability and a complete audit trace out - every step is inspectable on the other pages.")
    ui.flow([
        ("1 · Input", "Authorisation request", "Customer, card, merchant, amount, channel, timestamp, device, 3DS - the dictionary's Have + Add fields."),
        ("2 · Features", f"{reg['n_features']} leakage-safe features", "Customer habits, velocity, geography, device, merchant risk as-of scoring time. Every Derived field in the dictionary."),
        ("3 · Models", "Tree · XGBoost · Scorecard", "Rules for analysts, gradient boosting for patterns, a logistic scorecard for points; XGBoost explained with exact SHAP values."),
        ("4 · Bayesian priors", "MCMC posteriors", "Merchant risk pooled vendor → MCC → population, and customer propensity, with credible intervals for cold starts."),
        ("5 · Blend & decide", "Stacker → policy tiers", "Transparent weights combine log-odds; thresholds map probability to approve / monitor / step-up / decline."),
    ])

    # ------------------------------------------------------------------ charts
    c1, c2 = st.columns([3, 2])
    with c1:
        w = weekly_rates(tx)
        fig = ui.line_ci_fig(w.week, w.rate, w.lo, w.hi, name="confirmed fraud rate", title="Confirmed fraud rate per week (mature labels, 95% Beta interval)")
        fig.add_vline(x=(store.start + pd.Timedelta(days=C.SPLIT["train_end"])).timestamp() * 1000, line=dict(color=P["axis"], width=1, dash="dot"))
        fig.add_annotation(x=store.start + pd.Timedelta(days=C.SPLIT["train_end"]), y=w.hi.max(), text="train | validation", showarrow=False, font=dict(size=10, color=P["muted"]), xanchor="left")
        ui.plot(fig, height=300)
    with c2:
        typ = tx[tx.fraud_truth == 1].groupby("fraud_typology").size().sort_values()
        fig = ui.bar_fig(typ.values, [C.TYPOLOGY_SHORT.get(t, t) for t in typ.index], orientation="h", title="Fraud attempts by typology (simulation truth)",
                         text=[f"{v:,}" for v in typ.values], hover="%{y}<br>%{x:,} transactions<extra></extra>", height=300)
        fig.update_layout(margin=dict(l=10, r=40, t=36, b=40), yaxis=dict(tickfont=dict(size=11)))
        ui.plot(fig)

    c3, c4 = st.columns([3, 2])
    with c3:
        d = live.copy()
        d["day"] = d.transaction_timestamp.dt.normalize()
        mix = d.groupby(["day", "tier"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for tier, _, status in C.TIERS:
            if tier in mix.columns and tier != "APPROVE":
                fig.add_trace(go.Bar(x=mix.index, y=mix[tier], name=ui.TIER_LABEL[tier], marker=dict(color=STATUS[status], line=dict(color=P["surface"], width=1)),
                                     hovertemplate="%{x}<br>" + ui.TIER_LABEL[tier] + ": %{y}<extra></extra>"))
        fig.update_layout(barmode="stack", title="Daily alert volume in the live stream under the current policy", height=300, yaxis_title="authorisations", bargap=0.2)
        ui.plot(fig)
    with c4:
        ch = mature.groupby("channel").agg(n=("fraud_label", "size"), y=("fraud_label", "sum"))
        ch["rate"] = ch.y / ch.n
        ch["lo"] = beta.ppf(0.025, ch.y + 0.5, ch.n - ch.y + 0.5)
        ch["hi"] = beta.ppf(0.975, ch.y + 0.5, ch.n - ch.y + 0.5)
        ch = ch.sort_values("rate", ascending=False)
        ui.plot(ui.bar_ci_fig(ch.index, ch.rate, ch.lo, ch.hi, title="Fraud rate by channel (95% interval)", height=300))

    # ------------------------------------------------------------------ quick links
    ui.section("Explore", "")
    cols = st.columns(4)
    links = [("views/live.py", "Score a transaction live", ":material/bolt:", "live"), ("views/customer.py", "Customer 360", ":material/person_search:", "customer"),
             ("views/models_lab.py", "Model lab & evaluation", ":material/science:", "models"), ("views/bayes_lab.py", "Bayesian & MCMC diagnostics", ":material/functions:", "bayes")]
    pages = st.session_state.get("pages", {})
    for col, (_, label, icon, path) in zip(cols, links):
        with col:
            if path in pages:
                st.page_link(pages[path], label=label, icon=icon)
    ui.note("<b>About the data.</b> All customers, merchants and transactions are synthetic: 1,500 customers with habits (usual merchants, hours, "
            "subscriptions, travel), 800+ merchants across 24 MCCs and six injected fraud typologies. Fraud is enriched to ≈1% of authorisations "
            "so the demo has signal; UK card fraud runs closer to 0.1%. Disputes lag transactions by weeks, so the last 60 days are treated as unlabelled.")
