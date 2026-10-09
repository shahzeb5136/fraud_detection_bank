"""Portfolio analytics: fraud on an overall basis - rates, losses, typologies, merchants, geography and label maturity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta

from fraudlab import config as C, ui
from fraudlab.ui import P, STATUS


def _rate_ci(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["rate"] = g.y / g.n
    g["lo"] = beta.ppf(0.025, g.y + 0.5, g.n - g.y + 0.5)
    g["hi"] = beta.ppf(0.975, g.y + 0.5, g.n - g.y + 0.5)
    return g


def render():
    store = ui.get_store()
    tx, sc = store.tx, store.scores
    ui.page_header("Portfolio analytics", "Fraud on an overall basis: where it happens, what it costs, which merchants carry it, and how quickly labels mature.")

    # ------------------------------------------------------------------ filters (one row, scoping everything below)
    auth = tx[tx.transaction_type == "authorisation"]
    f1, f2, f3 = st.columns([1.2, 1.5, 1])
    with f1:
        window = st.selectbox("Window", ["Mature labels only (recommended)", "Last 90 days", "Last 180 days", "Full history"], key="pf_window")
    with f2:
        chans = st.multiselect("Channels", C.CHANNELS, default=[], placeholder="all channels", key="pf_chan")
    with f3:
        only_approved = st.toggle("Approved authorisations only", value=False, key="pf_appr")
    # scope = channel / approval filters; d = scope within the time window. Every chart below uses one of the two.
    scope = auth
    if chans:
        scope = scope[scope.channel.isin(chans)]
    if only_approved:
        scope = scope[scope.auth_result == "approved"]
    since = {"Last 90 days": 90, "Last 180 days": 180}.get(window)
    if window == "Mature labels only (recommended)":
        d = scope[scope.label_mature]
    elif since:
        d = scope[scope.transaction_timestamp >= store.as_of - pd.Timedelta(days=since)]
    else:
        d = scope
    imm = scope[~scope.label_mature] if window.startswith("Mature") else d[~d.label_mature]
    if len(d) == 0:
        st.info("No authorisations match these filters.")
        return
    if not window.startswith("Mature") and len(imm):
        st.caption(f"{len(imm):,} of the {len(d):,} authorisations in this window are younger than {C.LABEL_MATURITY_DAYS} days: their disputes have mostly not "
                   "arrived yet, so confirmed fraud rates and losses in the shaded weeks are understated. That is why the mature window is recommended.")
    n, y = len(d), int(d.fraud_label.sum())
    lo, hi = beta.ppf([0.025, 0.975], y + 0.5, n - y + 0.5)
    loss = d[(d.fraud_label == 1) & (d.auth_result == "approved")].transaction_amt.sum()
    spend = d[d.auth_result == "approved"].transaction_amt.sum()
    ui.kpi_row([
        dict(label="Authorisations", value=f"{n:,}", sub=f"{d.person_id.nunique():,} customers · {d.vendor_id.nunique():,} merchants"),
        dict(label="Confirmed fraud rate", value=ui.fmt_pct(y / max(n, 1), 2), sub=f"{y:,} confirmed · 95% interval {ui.fmt_pct(lo, 2)}–{ui.fmt_pct(hi, 2)}", status="warning"),
        dict(label="Confirmed fraud losses", value=ui.fmt_gbp(loss), sub=f"{ui.fmt_pct(loss / max(spend, 1), 3)} of approved spend (basis points: {1e4 * loss / max(spend, 1):.1f})", status="critical"),
        dict(label="Dispute rate (any reason)", value=ui.fmt_pct(d.is_disputed.mean(), 2), sub=f"{int(d.is_disputed.sum()):,} disputes · {int((d.dispute_reason_family == 'fraud').sum()):,} fraud-coded"),
        dict(label="Decline rate", value=ui.fmt_pct((d.auth_result == 'declined').mean(), 2), sub="declines are signal: card testing shows as bursts of small declines"),
        dict(label="Unlabelled (immature)", value=f"{len(imm):,}", sub=f"authorisations younger than {C.LABEL_MATURITY_DAYS} days" + (" - excluded from this view and from training" if window.startswith("Mature") else " - included, labels incomplete")),
    ])

    # ------------------------------------------------------------------ time & channel
    a, b = st.columns([3, 2])
    with a:
        w = d.groupby(d.transaction_timestamp.dt.to_period("W").dt.start_time).agg(n=("fraud_label", "size"), y=("fraud_label", "sum"))
        w = _rate_ci(w).reset_index(names="week")
        fig = ui.line_ci_fig(w.week, w.rate, w.lo, w.hi, name="confirmed fraud rate", title="Confirmed fraud rate per week with 95% Beta interval")
        cut = store.as_of - pd.Timedelta(days=C.LABEL_MATURITY_DAYS)
        if d.transaction_timestamp.max() > cut:
            fig.add_vrect(x0=cut.timestamp() * 1000, x1=store.as_of.timestamp() * 1000, fillcolor=P["muted"], opacity=0.12, line_width=0, annotation_text="labels immature",
                          annotation_position="top left", annotation_font=dict(size=10, color=P["muted"]))
        ui.plot(fig, height=300)
    with b:
        ch = _rate_ci(d.groupby("channel").agg(n=("fraud_label", "size"), y=("fraud_label", "sum"))).sort_values("rate", ascending=False)
        ui.plot(ui.bar_ci_fig(ch.index, ch.rate, ch.lo, ch.hi, title="Fraud rate by entry mode", height=300))

    a, b = st.columns([3, 2])
    with a:
        m = _rate_ci(d.groupby("mcc").agg(n=("fraud_label", "size"), y=("fraud_label", "sum"))).sort_values("rate", ascending=False).head(14)
        labels = [f"{i} · {C.MCC_TABLE[i][0][:26]}" for i in m.index]
        ui.plot(ui.bar_ci_fig(labels, m.rate, m.lo, m.hi, title="Riskiest merchant categories (MCC) with 95% interval", height=380))
    with b:
        co = _rate_ci(d.groupby("merchant_country").agg(n=("fraud_label", "size"), y=("fraud_label", "sum")))
        co = co[co.n >= 30].sort_values("rate", ascending=False).head(12)
        ui.plot(ui.bar_ci_fig([C.COUNTRY_NAMES.get(c, c) for c in co.index], co.rate, co.lo, co.hi, title="Fraud rate by merchant country (≥ 30 authorisations)", height=380))

    # ------------------------------------------------------------------ hour x dow, typology
    a, b = st.columns([1, 1])
    with a:
        hl = d.transaction_timestamp_local
        z = pd.crosstab(hl.dt.dayofweek, (hl.dt.hour // 3), values=d.fraud_label, aggfunc="mean").reindex(index=range(7), columns=range(8)).fillna(0)
        fig = ui.heatmap_fig(z.values, [f"{3 * i:02d}–{3 * i + 2:02d}h" for i in range(8)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], title="Fraud rate by local hour band and weekday", zfmt=".1%", height=300)
        fig.update_yaxes(autorange="reversed")
        ui.plot(fig)
    with b:
        fr = d[d.fraud_truth == 1].copy()
        fr["month"] = fr.transaction_timestamp.dt.to_period("M").dt.start_time
        mix = fr.groupby(["month", "fraud_typology"]).transaction_amt.sum().unstack(fill_value=0)
        fig = go.Figure()
        for i, t in enumerate(list(C.TYPOLOGIES)):
            if t in mix.columns:
                fig.add_trace(go.Bar(x=mix.index, y=mix[t], name=C.TYPOLOGY_SHORT[t], marker=dict(color=P["series"][i], line=dict(color=P["surface"], width=1)),
                                     hovertemplate="%{x|%b %Y}<br>" + C.TYPOLOGY_SHORT[t] + ": £%{y:,.0f}<extra></extra>"))
        fig.update_layout(barmode="stack", title="Attempted fraud value per month by typology (simulation truth)", height=300, yaxis_tickprefix="£", bargap=0.25, legend=dict(font=dict(size=10)))
        ui.plot(fig)

    # ------------------------------------------------------------------ merchants (Bayesian league table)
    ui.section("Merchant risk league", "Posterior fraud rates from the hierarchical MCMC model - shrinkage keeps a merchant with 2 frauds in 20 transactions from looking worse than one with 40 in 1,000.")
    vm = store.bayes["vendor"]
    vs = pd.DataFrame(vm.vendor_summary)
    vs = vs[vs.vendor_id.isin(d.vendor_id.unique())]           # merchants active in the current selection
    vs["merchant"] = vs.vendor_id.map(store.vendors.vendor_name)
    vs["city"] = vs.vendor_id.map(store.vendors.merchant_city)
    vs["country"] = vs.vendor_id.map(store.vendors.merchant_country)
    vs["mcc_desc"] = vs.mcc.map(lambda m: C.MCC_TABLE[m][0])
    vs["flag"] = np.where(vs.vendor_id.map(store.vendors.is_fraudulent), "fraudulent shop (truth)", np.where(vs.vendor_id.map(store.vendors.is_aggregator), "aggregator", ""))
    rank_opts = {"Lower 95% bound": "q2_5", "Posterior mean": "post_mean", "Raw rate": "raw_rate"}
    rank = st.segmented_control("Rank merchants by", list(rank_opts), default="Lower 95% bound", key="pf_rank",
                                help="The lower credible bound ranks merchants we are confident are risky; ranking by the raw rate (or even the posterior mean) "
                                     "rewards a handful of unlucky transactions at tiny merchants.") or "Lower 95% bound"
    top = vs.sort_values([rank_opts[rank], "n"], ascending=False).head(20)
    a, b = st.columns([1.3, 1])
    with a:
        st.dataframe(top[["merchant", "mcc", "mcc_desc", "city", "country", "n", "y", "raw_rate", "post_mean", "q2_5", "q97_5", "prob_gt_3x_pop", "flag"]], width="stretch", hide_index=True, height=420,
                     column_config={"raw_rate": st.column_config.NumberColumn("raw rate", format="percent"), "post_mean": st.column_config.NumberColumn("posterior", format="percent"),
                                    "q2_5": st.column_config.NumberColumn("2.5%", format="percent"), "q97_5": st.column_config.NumberColumn("97.5%", format="percent"),
                                    "prob_gt_3x_pop": st.column_config.ProgressColumn("P(> 3× population)", format="%.2f", min_value=0, max_value=1),
                                    "n": st.column_config.NumberColumn("txns (train)"), "y": st.column_config.NumberColumn("fraud")})
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vs.raw_rate, y=vs.post_mean, mode="markers", name="merchant",
                                 marker=dict(size=np.clip(4 + 2.2 * np.sqrt(vs.n / 20), 4, 26), color=P["series"][0], opacity=0.55, line=dict(color=P["surface"], width=1)),
                                 text=vs.merchant + " · n=" + vs.n.astype(int).astype(str), hovertemplate="%{text}<br>raw %{x:.2%} → posterior %{y:.2%}<extra></extra>"))
        mx = float(max(vs.raw_rate.max(), vs.post_mean.max()))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=P["axis"], width=1, dash="dot"), name="no shrinkage", hoverinfo="skip"))
        fig.update_layout(title="Shrinkage: raw merchant fraud rate vs posterior mean (marker size ∝ √transactions)", xaxis=dict(tickformat=".0%", title="raw rate"), yaxis=dict(tickformat=".0%", title="posterior mean"), height=420)
        ui.plot(fig)

    # ------------------------------------------------------------------ label maturity & alerts
    ui.section("Label maturity and policy load", "Disputes arrive weeks after the transaction; the maturity window decides which rows can be trusted as labelled.")
    a, b = st.columns([1, 1])
    with a:
        disp = scope[scope.is_disputed & scope.dispute_date.notna()]
        lag = (disp.dispute_date - disp.transaction_timestamp).dt.total_seconds() / 86400
        xs = np.arange(0, 181)
        cdf = [(lag <= x).mean() for x in xs]
        fig = go.Figure(go.Scatter(x=xs, y=cdf, mode="lines", line=dict(color=P["series"][0], width=2), name="share of disputes raised", hovertemplate="day %{x}<br>%{y:.0%} of disputes raised<extra></extra>"))
        fig.add_vline(x=C.LABEL_MATURITY_DAYS, line=dict(color=P["diverging_pos"], width=1.5, dash="dot"), annotation_text=f"maturity window {C.LABEL_MATURITY_DAYS} d ({(lag <= C.LABEL_MATURITY_DAYS).mean():.0%} raised)", annotation_font=dict(size=11))
        fig.update_layout(title="Dispute lag: cumulative share of disputes raised within N days of the transaction", xaxis_title="days after transaction", yaxis=dict(tickformat=".0%"), height=300)
        ui.plot(fig)
    with b:
        te = sc[sc.split.isin(["test", "live"]) & sc.transaction_id.isin(scope.transaction_id)]
        if since:
            te = te[te.transaction_timestamp >= store.as_of - pd.Timedelta(days=since)]
        tiers = pd.Series(ui.tiers_now(te.p_blend), index=te.index, name="tier")
        g = te.groupby([te.transaction_timestamp.dt.to_period("W").dt.start_time, tiers]).size().unstack(fill_value=0)
        fig = go.Figure()
        for t, _, s in C.TIERS[1:]:
            if t in g.columns:
                fig.add_trace(go.Bar(x=g.index, y=g[t], name=ui.TIER_LABEL[t], marker=dict(color=STATUS[s], line=dict(color=P["surface"], width=1)), hovertemplate="%{x|%d %b}<br>" + ui.TIER_LABEL[t] + ": %{y}<extra></extra>"))
        fig.update_layout(barmode="stack", title="Weekly operational load under the current policy (test + live windows)", height=300, yaxis_title="authorisations", bargap=0.25)
        ui.plot(fig)
