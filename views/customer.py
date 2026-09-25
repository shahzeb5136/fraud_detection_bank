"""Customer 360: one cardholder's habits, geography, Bayesian propensity and scored history."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudlab import config as C, ui
from fraudlab.ui import P, STATUS


def render():
    store = ui.get_store()
    persons, sc = store.persons, store.scores
    ui.page_header("Customer 360", "Per-customer behavioural baseline: what 'normal' looks like for this cardholder, and how every one of their transactions was scored.")

    c1, c2, c3 = st.columns([2, 1, 1])
    options = persons.index.tolist()
    with c1:
        pid = st.selectbox("Customer", options, index=options.index(st.session_state.get("person_sel", options[0])),
                           format_func=lambda x: f"{x} · {persons.loc[x].segment.replace('_', ' ')} · {persons.loc[x].home_city}", key="c360_person")
        st.session_state["person_sel"] = pid
    with c2:
        if st.button("Random customer", width="stretch"):
            st.session_state["person_sel"] = str(np.random.default_rng().choice(options)); st.rerun()
    with c3:
        if st.button("Random fraud victim", width="stretch"):
            st.session_state["person_sel"] = str(np.random.default_rng().choice(store.episodes.person_id.unique())); st.rerun()

    p = persons.loc[pid]
    hist = store.person_history(pid)
    ok = hist[hist.auth_result == "approved"]
    my = sc[sc.person_id == pid].sort_values("t_sec")
    pm = store.bayes["person"]
    prop = pm.person_rate(pid)
    n_fraud = int(hist.fraud_label.sum())
    eps = store.episodes[store.episodes.person_id == pid]
    online_share = float(hist.channel.isin(["ecommerce", "phone_order"]).mean()) if len(hist) else 0.0
    foreign_share = float((hist.merchant_country != "GB").mean()) if len(hist) else 0.0
    cards = store.cards[store.cards.person_id == pid]

    ui.kpi_row([
        dict(label="Segment · home", value=f"{p.segment.replace('_', ' ').title()}", sub=f"{p.home_city} · joined {p.joined_date} · tenure {int(p.account_tenure_days):,} days"),
        dict(label="Authorisations", value=f"{len(hist):,}", sub=f"{ui.fmt_gbp(ok.transaction_amt.sum())} approved spend · {len(ok.vendor_id.unique())} merchants"),
        dict(label="Typical amount", value=ui.fmt_gbp(ok.transaction_amt.median(), 2) if len(ok) else "–", sub=f"p90 {ui.fmt_gbp(ok.transaction_amt.quantile(0.9), 0) if len(ok) else '–'} · limit {ui.fmt_gbp(p.credit_limit)}"),
        dict(label="Channel footprint", value=ui.fmt_pct(online_share, 0) + " online", sub=f"{ui.fmt_pct(foreign_share, 0)} outside the UK · {int(p.n_devices)} devices · {int(p.n_subscriptions)} subscriptions"),
        dict(label="Bayesian propensity", value=ui.fmt_pct(prop["mean"], 2), sub=f"95% interval {ui.fmt_pct(prop['lo'], 2)}–{ui.fmt_pct(prop['hi'], 2)} · {prop['level']}", status="serious" if prop["mean"] > 2 * pm.pop_mean else "good"),
        dict(label="Confirmed fraud history", value=f"{n_fraud}", sub=f"{len(eps)} fraud episode(s) in simulation · {len(cards)} card(s)", status="critical" if n_fraud else "good"),
    ])

    # ------------------------------------------------------------------ habits
    ui.section("Behavioural baseline", "The features compare every new authorisation against these habits, computed strictly from earlier transactions.")
    a, b, c = st.columns([1.2, 1, 1])
    with a:
        if len(ok):
            m = ok.groupby(ok.transaction_timestamp.dt.to_period("M").dt.start_time).transaction_amt.sum()
            fig = go.Figure(go.Bar(x=m.index, y=m.values, marker=dict(color=P["series"][0], line_width=0), hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>", name="spend"))
            fr = hist[hist.fraud_truth == 1]
            if len(fr):
                fm = fr.groupby(fr.transaction_timestamp.dt.to_period("M").dt.start_time).transaction_amt.sum()
                fig.add_trace(go.Bar(x=fm.index, y=fm.values, marker=dict(color=STATUS["critical"], line_width=0), name="fraud attempts", hovertemplate="%{x|%b %Y}<br>fraud £%{y:,.0f}<extra></extra>"))
            fig.update_layout(title="Monthly approved spend (fraud attempts in red)", height=280, barmode="overlay", yaxis_tickprefix="£")
            ui.plot(fig)
    with b:
        if len(ok):
            top = ok.groupby("vendor_id").size().sort_values(ascending=False).head(10)
            names = [f"{store.vendors.loc[v].vendor_name[:22]} ({store.vendors.loc[v].mcc})" for v in top.index]
            fig = ui.bar_fig(top.values[::-1], names[::-1], orientation="h", title="Usual merchants (visits)", hover="%{y}<br>%{x} visits<extra></extra>", height=280)
            fig.update_layout(margin=dict(l=10, r=10, t=36, b=30), yaxis=dict(tickfont=dict(size=10.5)))
            ui.plot(fig)
    with c:
        if len(ok):
            z = np.zeros((7, 8))
            for dow, h in zip(ok.day_of_week, (ok.hour_local // 3).astype(int)):
                z[dow, h] += 1
            z = z / z.sum()
            fig = ui.heatmap_fig(z, [f"{3 * i:02d}–{3 * i + 2:02d}h" for i in range(8)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], title="When they transact (share)", zfmt=".1%", height=280)
            fig.update_yaxes(autorange="reversed")
            ui.plot(fig)

    a, b = st.columns([1, 1])
    with a:
        if len(ok):
            top5 = ok.groupby("vendor_id").size().sort_values(ascending=False).head(6).index
            fig = go.Figure()
            for i, v in enumerate(top5):
                s = ok[ok.vendor_id == v].transaction_amt
                fig.add_trace(go.Box(y=s, name=store.vendors.loc[v].vendor_name[:18], marker=dict(color=P["series"][i % 8]), line=dict(width=1.5), boxpoints="outliers", hovertemplate="£%{y:,.2f}<extra></extra>"))
            fig.update_layout(title="Amount distribution at the customer's top merchants (the per-vendor baseline is far tighter than the customer-wide one)", height=300, showlegend=False, yaxis_tickprefix="£")
            ui.plot(fig)
    with b:
        fig = go.Figure()
        cp = hist[hist.channel.isin(list(C.CARD_PRESENT_CHANNELS))]
        if len(cp):
            g = cp.groupby(["merchant_city", "merchant_lat", "merchant_lon"]).agg(n=("transaction_id", "size"), fraud=("fraud_truth", "sum")).reset_index()
            fig.add_trace(go.Scattergeo(lat=g.merchant_lat, lon=g.merchant_lon, mode="markers", name="card-present merchants",
                                        marker=dict(size=np.clip(6 + 2 * np.sqrt(g.n), 6, 24), color=[STATUS["critical"] if f else P["series"][0] for f in g.fraud], line=dict(color=P["surface"], width=1.5), opacity=0.85),
                                        text=g.merchant_city + " · " + g.n.astype(str) + " visits", hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scattergeo(lat=[p.home_lat], lon=[p.home_lon], mode="markers", name="home", marker=dict(size=14, symbol="star", color=P["ink"], line=dict(color=P["surface"], width=1.5)), hovertemplate="home · " + p.home_city + "<extra></extra>"))
        fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor=P["grid"], landcolor="#f1f0ec", showland=True, oceancolor="#ffffff", showocean=True, lakecolor="#ffffff",
                        fitbounds="locations", showframe=False, coastlinecolor=P["axis"])
        fig.update_layout(title="Card-present footprint (red = a fraud attempt happened there)", height=300, margin=dict(l=0, r=0, t=36, b=0))
        ui.plot(fig)

    # ------------------------------------------------------------------ scored history
    ui.section("Scored history", "Every authorisation of this customer through the blended model. Hover for merchant and decision; simulation truth marks real fraud attempts.")
    if len(my):
        fig = go.Figure()
        colors = [STATUS[ui.TIER_STATUS[t]] for t in my.tier]
        symbols = np.where(my.fraud_truth == 1, "x", "circle")
        fig.add_trace(go.Scatter(x=my.transaction_timestamp, y=my.p_blend, mode="markers",
                                 marker=dict(size=np.where(my.fraud_truth == 1, 12, 8), color=colors, symbol=symbols, line=dict(color=P["surface"], width=1)),
                                 text=[f"{store.vendors.loc[v].vendor_name} · £{a:,.2f} · {c} · {ui.TIER_LABEL[t]}" + (" · FRAUD (" + str(ty) + ")" if f == 1 else "") for v, a, c, t, f, ty in zip(my.vendor_id, my.transaction_amt, my.channel, my.tier, my.fraud_truth, my.fraud_typology)],
                                 hovertemplate="%{x|%d %b %Y %H:%M}<br>%{text}<br>p = %{y:.3f}<extra></extra>", name="authorisation"))
        pol = ui.get_policy()
        for thr, lab in [(pol["approve_max"], "monitor"), (pol["monitor_max"], "step-up"), (pol["stepup_max"], "decline")]:
            fig.add_hline(y=thr, line=dict(color=P["axis"], width=1, dash="dot"), annotation_text=lab, annotation_position="right", annotation_font=dict(size=10, color=P["muted"]))
        fig.update_layout(title="Fraud probability per authorisation (× = fraud attempt in simulation truth; colour = policy tier)", height=320, yaxis=dict(type="log", title="fraud probability", tickformat=".3f"), showlegend=False)
        ui.plot(fig)
        tab = my[["transaction_id", "transaction_timestamp", "vendor_id", "transaction_amt", "channel", "merchant_country", "p_blend", "p_xgb", "p_tree", "p_bayes_vendor", "tier", "label_status", "fraud_truth", "fraud_typology"]].copy()
        tab["merchant"] = tab.vendor_id.map(store.vendors.vendor_name)
        tab["decision"] = tab.tier.map(ui.TIER_LABEL)
        tab = tab.sort_values("transaction_timestamp", ascending=False)
        st.dataframe(tab[["transaction_id", "transaction_timestamp", "merchant", "transaction_amt", "channel", "merchant_country", "p_blend", "p_xgb", "p_tree", "p_bayes_vendor", "decision", "label_status", "fraud_typology"]],
                     width="stretch", hide_index=True, height=360,
                     column_config={"p_blend": st.column_config.ProgressColumn("blend", format="%.3f", min_value=0, max_value=1),
                                    "p_xgb": st.column_config.NumberColumn("xgb", format="%.3f"), "p_tree": st.column_config.NumberColumn("tree", format="%.3f"),
                                    "p_bayes_vendor": st.column_config.NumberColumn("merchant prior", format="%.4f"),
                                    "transaction_amt": st.column_config.NumberColumn("amount", format="£%.2f"),
                                    "transaction_timestamp": st.column_config.DatetimeColumn("time (UTC)", format="D MMM YYYY HH:mm")})
    if len(eps):
        st.markdown("**Fraud episodes in the simulation** (a real bank would only see the disputes)")
        e = eps.copy()
        e["start"] = (store.start + pd.to_timedelta(e.start_day, unit="D")).dt.date
        e["typology"] = e.typology.map(lambda t: C.TYPOLOGIES.get(t, t))
        st.dataframe(e[["episode_id", "start", "typology"]], width="stretch", hide_index=True)
    pages = st.session_state.get("pages", {})
    if "live" in pages:
        st.page_link(pages["live"], label="Score a hypothetical transaction for this customer →", icon=":material/bolt:")
