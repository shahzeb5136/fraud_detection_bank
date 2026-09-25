"""Live scoring: score a hand-built transaction (with scenario presets) or replay the live stream, with a full audit trace."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudlab import config as C, features as FE, ui
from fraudlab.models import STACK_LABELS
from fraudlab.scoring import make_transaction, score_transaction, trace_to_json, transaction_from_row
from fraudlab.ui import P, STATUS

PRESETS = {
    "custom": "Custom",
    "typical": "Typical purchase (usual merchant, usual amount)",
    "overseas_cnp": "Overseas online spree (new merchant, night, new device, no 3DS)",
    "card_testing": "Card-testing burst (tiny declined attempts, then a purchase)",
    "ato": "Account takeover (password reset yesterday, big electronics order)",
    "lost_card": "Lost card cash-out (ATM + contactless far from home)",
    "fake_shop": "Fake online shop (fraudulent merchant, own device)",
}


# ----------------------------------------------------------------------------------------------
# preset construction
# ----------------------------------------------------------------------------------------------
def _person_label(store, pid):
    p = store.persons.loc[pid]
    return f"{pid} · {p.segment.replace('_', ' ')} · {p.home_city}"


def build_preset(store, pid: str, preset: str, rng: np.random.Generator) -> dict:
    hist = store.person_history(pid)
    ok = hist[hist.auth_result == "approved"]
    vendors = store.vendors
    now = store.as_of + pd.Timedelta(hours=13, minutes=12)
    d = dict(vendor_id=None, new_vendor=None, amount=45.0, channel="chip", ts=now, sca=False, device="known", extra=None, note="")
    if preset == "typical" or preset == "custom":
        if len(ok):
            top = ok.vendor_id.value_counts().index[0]
            amt = float(ok[ok.vendor_id == top].transaction_amt.median())
            v = vendors.loc[top]
            d.update(vendor_id=top, amount=round(amt, 2), channel="ecommerce" if v.is_online else "contactless",
                     ts=now.normalize() + pd.Timedelta(hours=float(ok.hour_local.median())), sca=bool(v.is_online))
        d["note"] = "The customer's most frequent merchant at their median amount and usual hour."
    elif preset == "overseas_cnp":
        pool = vendors[vendors.is_online & (vendors.merchant_country != "GB") & vendors.mcc.isin(["4816", "5964", "5732", "7995"]) & (~vendors.is_fraudulent)]
        pool = pool[~pool.index.isin(hist.vendor_id.unique())]
        vid = pool.sample(1, random_state=int(rng.integers(0, 1 << 30))).index[0]
        scale = float(ok.transaction_amt.median()) if len(ok) else 40.0
        d.update(vendor_id=vid, amount=round(max(scale * 4.5, 180.0), 2), channel="ecommerce", ts=now.normalize() + pd.Timedelta(hours=2, minutes=41), sca=False, device="new",
                 note="Card-not-present at a foreign high-risk merchant the customer has never used, 02:41 local, unknown device, no strong authentication.")
    elif preset == "card_testing":
        pool = vendors[vendors.is_online & vendors.mcc.isin(["4816", "5968", "7372"]) & (~vendors.is_fraudulent)]
        pool = pool[~pool.index.isin(hist.vendor_id.unique())]
        vids = pool.sample(2, random_state=int(rng.integers(0, 1 << 30))).index.tolist()
        cash = vendors[vendors.is_online & vendors.mcc.isin(["5732", "5964"]) & (~vendors.is_fraudulent) & (~vendors.index.isin(hist.vendor_id.unique()))]
        cvid = cash.sample(1, random_state=int(rng.integers(0, 1 << 30))).index[0]
        ts = now.normalize() + pd.Timedelta(hours=1, minutes=5)
        extra = []
        for k in range(9):
            v = vendors.loc[vids[k % 2]]
            t = ts - pd.Timedelta(minutes=14 - k * 1.4)
            extra.append(dict(t_sec=(t - store.start).total_seconds(), transaction_amt=float(rng.choice([0.99, 1.0, 1.5, 2.0, 2.99])),
                              auth_result="declined" if k % 2 == 0 else "approved", vendor_id=vids[k % 2], mcc=str(v.mcc), channel="ecommerce",
                              merchant_country=v.merchant_country, hour_local=float(t.hour + t.minute / 60), day_of_week=int(t.dayofweek), day_of_month=int(t.day),
                              device_id="D-testburst", merchant_lat=v.merchant_lat, merchant_lon=v.merchant_lon,
                              cardholder_home_lat=float(store.persons.loc[pid].home_lat), cardholder_home_lon=float(store.persons.loc[pid].home_lon),
                              cardholder_home_country="GB", label_known_sec=np.nan, fraud_label=0, card_id=hist.card_id.iloc[-1] if len(hist) else None,
                              card_age_days=float(hist.card_age_days.iloc[-1]) if len(hist) else 0.0))
        d.update(vendor_id=cvid, amount=249.0, channel="ecommerce", ts=ts, sca=False, device="D-testburst", extra=pd.DataFrame(extra),
                 note="Nine £1–£3 attempts in the previous 14 minutes at two digital-goods merchants (half declined), then a £249 electronics order from the same new device.")
    elif preset == "ato":
        pool = vendors[vendors.is_online & vendors.mcc.isin(["5732", "5944"]) & (~vendors.is_fraudulent) & (~vendors.index.isin(hist.vendor_id.unique()))]
        vid = pool.sample(1, random_state=int(rng.integers(0, 1 << 30))).index[0]
        d.update(vendor_id=vid, amount=1289.0, channel="ecommerce", ts=now.normalize() + pd.Timedelta(hours=11, minutes=20), sca=True, device="new",
                 extra="events", note="A password reset and a new-payee event 1.3 days earlier (injected), then a £1,289 electronics order from a new device; 3DS passed because the fraudster controls the phone.")
    elif preset == "lost_card":
        home = store.persons.loc[pid].home_city
        cities = [c for c, v in C.CITIES.items() if v[0] == "GB" and c != home]
        city = str(rng.choice(cities))
        atm = vendors[(vendors.merchant_city == city) & (vendors.mcc == "6011")]
        vid = atm.index[0] if len(atm) else vendors[vendors.mcc == "6011"].index[0]
        d.update(vendor_id=vid, amount=300.0, channel="atm", ts=now.normalize() + pd.Timedelta(hours=23, minutes=48), sca=False, device=None,
                 note=f"£300 ATM withdrawal in {city} at 23:48 - the customer lives in {home} and has no card-present history there.")
    elif preset == "fake_shop":
        shops = vendors[vendors.is_fraudulent]
        vid = shops.sample(1, random_state=int(rng.integers(0, 1 << 30))).index[0]
        d.update(vendor_id=vid, amount=139.0, channel="ecommerce", ts=now.normalize() + pd.Timedelta(hours=20, minutes=5), sca=True, device="known",
                 note="The customer's own device and normal evening hour, but the merchant is a fake shop whose disputes have matured - the Bayesian merchant prior should carry this one.")
    return d


def inject_ato_events(store, pid, ts):
    ev = store.ev_by_person.get(pid, np.array([]))
    t = (ts - store.start).total_seconds()
    return np.sort(np.concatenate([ev, [t - 1.3 * 86400, t - 1.1 * 86400]]))


# ----------------------------------------------------------------------------------------------
# result rendering
# ----------------------------------------------------------------------------------------------
def render_trace(store, trace: dict, key: str = "t"):
    policy = ui.get_policy()
    ui.decision_banner(trace)
    c1, c2, c3 = st.columns([1, 1.5, 1.9])
    with c1:
        ui.plot(ui.gauge_fig(trace["decision"]["probability"], policy), key=f"{key}_gauge")
        st.html('<div class="fl-small" style="margin-top:-8px">Policy bands: approve &lt; %s · monitor &lt; %s · step-up &lt; %s · decline ≥ %s</div>' % (
            ui.fmt_pct(policy["approve_max"], 0), ui.fmt_pct(policy["monitor_max"], 0), ui.fmt_pct(policy["stepup_max"], 0), ui.fmt_pct(policy["stepup_max"], 0)))
    with c2:
        st.html('<div class="fl-small"><b>Each model on its own</b> - calibrated fraud probability and its weight in the blend</div>')
        ui.plot(ui.component_fig(trace["components"], trace["blend"]["weights"]), key=f"{key}_comp")
    with c3:
        ui.plot(ui.waterfall_fig(trace["blend"]), key=f"{key}_wf")

    tabs = st.tabs(["Why · XGBoost SHAP", "Rules · decision tree", "Scorecard points", "Bayesian priors", "Customer context", "Features & lineage", "Audit trace"])
    feats = trace["features"]
    with tabs[0]:
        x = trace["components"]["xgb"]
        a, b = st.columns([3, 2])
        with a:
            ui.plot(ui.contrib_fig(x["top_contributions"], feats, title=f"Top drivers of the XGBoost log-odds (calibrated p = {ui.fmt_pct(x['prob'], 2)})"), key=f"{key}_shap")
        with b:
            st.markdown("**Reading the chart**")
            st.markdown("Each bar is an exact SHAP value from the tree-path algorithm: how much that feature moved the log-odds away from the "
                        f"training baseline (bias {x['bias']:+.2f}). Red pushes towards fraud, blue towards genuine. Values after the dot are the feature values used.")
            st.markdown(f"Raw model probability {ui.fmt_pct(x['prob_raw'], 2)} → isotonic-calibrated {ui.fmt_pct(x['prob'], 2)}; "
                        f"calibration was fitted on the validation window so probabilities can be read as frequencies.")
    with tabs[1]:
        t = trace["components"]["tree"]
        a, b = st.columns([3, 2])
        with a:
            st.markdown(f"**Decision path** ({len(t['steps'])} splits → leaf {t['leaf']})")
            for i, s in enumerate(t["steps"]):
                st.html(f'<div class="fl-rule">{i + 1}. {s["rule"]}   <span style="color:{P["muted"]}">(value {ui.fmt_num(s["value"]) if s["value"] is not None else "missing"})</span></div>')
        with b:
            ui.kpi_row([dict(label="Leaf fraud rate", value=ui.fmt_pct(t["rate"], 2), sub="Laplace-smoothed, unweighted training counts"),
                        dict(label="Training transactions in leaf", value=f"{t['n_train']:,}", sub=f"{t['n_fraud']:,} confirmed fraud")])
            st.markdown("The tree is grown with balanced class weights so rare fraud shapes the splits, but the probability reported is the honest "
                        "unweighted fraud rate observed in that leaf during training - the number an analyst can verify.")
    with tabs[2]:
        s = trace["components"]["scorecard"]
        a, b = st.columns([3, 2])
        with a:
            ui.plot(ui.contrib_fig(s["top_contributions"], feats, title=f"Scorecard points (standardised value × coefficient), p = {ui.fmt_pct(s['prob'], 2)}"), key=f"{key}_sc")
        with b:
            st.markdown("**Logistic scorecard**")
            st.markdown(f"Intercept {s['intercept']:+.2f} log-odds. Each feature is median-imputed (with a missing-indicator), standardised and multiplied by its "
                        "coefficient; the sum is the log-odds. Simple enough to publish as a points table for model-risk review.")
    with tabs[3]:
        bv, bp = trace["components"]["bayes_vendor"], trace["components"]["bayes_person"]
        a, b = st.columns(2)
        vm, pm = store.bayes["vendor"], store.bayes["person"]
        with a:
            vid = trace["input"]["vendor_id"]
            if vid in vm.vendor_pos:
                draws = vm.vendor_draws(vid)
            else:
                rng = np.random.default_rng(0)
                D = vm.draws
                sv = D["sigma_v"].reshape(-1)
                aa = D["a"][:, :, vm.mcc_index[trace["input"]["mcc"]]].reshape(-1) if trace["input"]["mcc"] in vm.mcc_index else D["mu"].reshape(-1)
                from scipy.special import expit
                draws = expit(aa + sv * rng.normal(size=sv.shape))
            pop = np.random.default_rng(1).choice(vm.draws["mu"].reshape(-1), 2000)
            from scipy.special import expit as _e
            fig = ui.density_fig(draws, "this merchant", C.MODEL_COLORS["bayes_vendor"], title=f"Merchant fraud-rate posterior · level: {bv['level']}",
                                 ref=vm.pop_rate, ref_label="population", extra=[dict(draws=_e(pop), name="population", color=P["muted"])])
            ui.plot(fig, key=f"{key}_bv")
            st.markdown(f"Posterior mean **{ui.fmt_pct(bv['prob'], 2)}** (95% interval {ui.fmt_pct(bv['lo'], 2)}–{ui.fmt_pct(bv['hi'], 2)}); "
                        f"{bv['y']} confirmed fraud in {bv['n']:,} training transactions (raw {ui.fmt_pct(bv['raw'], 2) if bv['raw'] == bv['raw'] else 'n/a'}); "
                        f"P(rate > 3× population) = {ui.fmt_pct(bv['prob_gt_3x'], 0)}.")
        with b:
            pid = trace["input"]["person_id"]
            al, be_ = pm.alpha_beta
            rng = np.random.default_rng(2)
            th = rng.beta(al + bp["y"], be_ + bp["n"] - bp["y"], 4000)
            fig = ui.density_fig(th, "this customer", C.MODEL_COLORS["bayes_person"], title=f"Customer propensity posterior · level: {bp['level']}",
                                 ref=pm.pop_mean, ref_label="population", extra=[dict(draws=rng.beta(al, be_, 4000), name="new customer prior", color=P["muted"])])
            ui.plot(fig, key=f"{key}_bp")
            st.markdown(f"Posterior mean **{ui.fmt_pct(bp['prob'], 2)}** (95% interval {ui.fmt_pct(bp['lo'], 2)}–{ui.fmt_pct(bp['hi'], 2)}); "
                        f"{bp['y']} confirmed fraud in {bp['n']:,} training transactions, shrunk towards the population with Beta({al:.1f}, {be_:.1f}).")
    with tabs[4]:
        _customer_context(store, trace, key)
    with tabs[5]:
        rows = []
        for f in FE.MODEL_FEATURES:
            v = feats[f]
            L = FE.LINEAGE[f]
            rows.append(dict(feature=f, value=v, group=L[2], status=L[1], source_fields=L[0], definition=L[3]))
        df = pd.DataFrame(rows)
        grp = st.multiselect("Feature group", sorted(df.group.unique()), default=[], key=f"{key}_grp", placeholder="all groups")
        if grp:
            df = df[df.group.isin(grp)]
        st.dataframe(df, width="stretch", hide_index=True, height=420,
                     column_config={"value": st.column_config.NumberColumn(format="%.4g"), "definition": st.column_config.TextColumn(width="large")})
        st.caption("Status follows the data dictionary: Have = in the current extract, Add = requested from source systems, Derived = computed from fields already held.")
    with tabs[6]:
        st.markdown(f"Trace **{trace['trace_id']}** · scored {trace['scored_at']} · model {trace['model_version']} · "
                    f"features {trace['timings_ms']['features_ms']} ms · models {trace['timings_ms']['models_ms']} ms · priors {trace['timings_ms']['bayes_ms']} ms")
        ui.download_json_button(trace, "Download audit trace (JSON)", f"{trace['trace_id']}.json", key=f"{key}_dl")
        st.json({k: v for k, v in trace.items() if k not in ("feature_lineage",)}, expanded=False)


def _customer_context(store, trace, key):
    pid, vid = trace["input"]["person_id"], trace["input"]["vendor_id"]
    t_new = trace["input"]["t_sec"]
    hist = store.person_history(pid)
    hist = hist[hist.t_sec < t_new]
    a, b = st.columns(2)
    with a:
        pair = hist[(hist.vendor_id == vid) & (hist.auth_result == "approved")]
        amt = trace["input"]["transaction_amt"]
        pp = trace["history"]["pair_posterior_log_amount"]
        if len(pair) >= 3:
            fig = ui.hist_fig(pair.transaction_amt, title=f"This customer's {len(pair)} prior amounts at this merchant", xtitle="amount (GBP)", vline=amt, vline_label=f"now {ui.fmt_gbp(amt, 2)}", nbins=25)
        else:
            base = hist[hist.auth_result == "approved"]
            fig = ui.hist_fig(base.transaction_amt, title=f"No merchant history - all {len(base)} prior amounts of the customer", xtitle="amount (GBP)", vline=amt, vline_label=f"now {ui.fmt_gbp(amt, 2)}", nbins=30, log_x=True)
        ui.plot(fig, key=f"{key}_amt")
        mu, sd = pp["mean"], pp["sd"]
        st.markdown(f"Conjugate-normal expectation for log-amount at this merchant: mean {mu:.2f} (≈ {ui.fmt_gbp(np.expm1(mu), 0)}), predictive sd {sd:.2f} → "
                    f"z = **{trace['features']['amount_z_vs_user_vendor']:+.2f}**. Fewer prior visits ⇒ wider interval (shrinkage to the customer's MCC habit, then the population).")
    with b:
        win = hist[hist.t_sec > t_new - 7 * 86400]
        fig = go.Figure()
        if len(win):
            fig.add_trace(go.Scatter(x=store.start + pd.to_timedelta(win.t_sec, unit="s"), y=win.transaction_amt, mode="markers",
                                     marker=dict(size=9, color=[P["series"][0] if r == "approved" else STATUS["critical"] for r in win.auth_result], line=dict(color=P["surface"], width=2)),
                                     name="previous 7 days", text=win.vendor_name + " · " + win.channel + " · " + win.auth_result, hovertemplate="%{x}<br>%{text}<br>£%{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[store.start + pd.Timedelta(seconds=t_new)], y=[trace["input"]["transaction_amt"]], mode="markers",
                                 marker=dict(size=14, color=STATUS["serious"], symbol="diamond", line=dict(color=P["surface"], width=2)), name="incoming", hovertemplate="incoming<br>£%{y:,.2f}<extra></extra>"))
        fig.update_layout(title="Activity in the 7 days before this authorisation (red = declined)", yaxis_title="amount (GBP)", height=300)
        ui.plot(fig, key=f"{key}_win")
        f = trace["features"]
        st.markdown(f"Velocity: **{int(f['n_txn_1h'])}** in the last hour, **{int(f['n_txn_24h'])}** in 24 h, **{int(f['n_declines_24h'])}** declines in 24 h; "
                    f"{int(f['n_new_vendors_24h'])} first-time merchants in 24 h; last non-monetary account event "
                    f"{('%.1f days ago' % f['days_since_account_event']) if f['days_since_account_event'] is not None else 'none on record'}.")


# ----------------------------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------------------------
def render():
    store = ui.get_store()
    policy = ui.get_policy()
    ui.page_header("Live scoring", "Score an incoming authorisation and inspect every step: features, each model's view, Bayesian priors, the blend and the decision.")
    with st.sidebar:
        st.markdown("**Decision policy**")
        a = st.slider("Approve below", 0.005, 0.20, float(policy["approve_max"]), 0.005, format="%.3f", key="pol_a")
        m = st.slider("Monitor below", 0.02, 0.40, float(max(policy["monitor_max"], a + 0.005)), 0.005, format="%.3f", key="pol_m")
        s = st.slider("Step-up below (decline at or above)", 0.05, 0.90, float(max(policy["stepup_max"], m + 0.005)), 0.01, format="%.2f", key="pol_s")
        policy.update(approve_max=a, monitor_max=max(m, a + 0.001), stepup_max=max(s, m + 0.001))
        friction = st.number_input("Friction cost of a false decline (£)", 0.0, 100.0, 8.0, 1.0, key="pol_f")

    tab_score, tab_stream = st.tabs(["Score a transaction", "Stream replay (live window)"])

    # ------------------------------------------------------------------ manual scoring
    with tab_score:
        persons = store.persons
        c1, c2 = st.columns([1, 2.2])
        with c1:
            st.markdown("**1 · Customer**")
            pick = st.radio("Quick pick", ["choose", "random customer", "random fraud victim"], horizontal=True, label_visibility="collapsed", key="pick")
            if pick != "choose" and st.session_state.get("_pick_last") != pick + str(st.session_state.get("_pick_n", 0)):
                rng = np.random.default_rng()
                if pick == "random customer":
                    st.session_state["person_sel"] = str(rng.choice(persons.index))
                else:
                    victims = store.episodes.person_id.unique()
                    st.session_state["person_sel"] = str(rng.choice(victims))
                st.session_state["_pick_last"] = pick + str(st.session_state.get("_pick_n", 0))
            options = persons.index.tolist()
            pid = st.selectbox("Customer", options, index=options.index(st.session_state.get("person_sel", options[0])), format_func=lambda x: _person_label(store, x), key="person_box")
            st.session_state["person_sel"] = pid
            if st.button("Shuffle quick pick", key="shuffle"):
                st.session_state["_pick_n"] = st.session_state.get("_pick_n", 0) + 1
                st.session_state["_pick_last"] = None
                st.rerun()
            hist = store.person_history(pid)
            pr = persons.loc[pid]
            st.html(f'<div class="fl-note">{pr.segment.replace("_", " ").title()} · {pr.home_city} · tenure {int(pr.account_tenure_days)} days · '
                    f'{len(hist):,} authorisations · {len(hist.vendor_id.unique())} merchants · limit {ui.fmt_gbp(pr.credit_limit)}</div>')
            st.markdown("**2 · Scenario**")
            preset = st.selectbox("Preset", list(PRESETS), format_func=lambda k: PRESETS[k], key="preset")
            seed_n = st.session_state.get("_pick_n", 0)
            d = build_preset(store, pid, preset, np.random.default_rng(seed_n + hash(pid) % 1000))
            if d["note"]:
                st.caption(d["note"])
        with c2:
            st.markdown("**3 · Authorisation request** (edit anything)")
            k = f"{preset}_{pid}"
            vend_opts = ["NEW"] + store.vendors.index.tolist()
            vlabel = lambda v: "＋ New / unseen merchant" if v == "NEW" else f"{store.vendors.loc[v].vendor_name} · MCC {store.vendors.loc[v].mcc} · {store.vendors.loc[v].merchant_city}" + (" · aggregator" if store.vendors.loc[v].is_aggregator else "")
            f1, f2, f3 = st.columns(3)
            with f1:
                vid = st.selectbox("Merchant", vend_opts, index=vend_opts.index(d["vendor_id"]) if d["vendor_id"] in vend_opts else 1, format_func=vlabel, key=f"v_{k}")
                amount = st.number_input("Amount (GBP)", 0.01, 50000.0, float(d["amount"]), 1.0, key=f"a_{k}")
                channel = st.selectbox("Channel / entry mode", C.CHANNELS, index=C.CHANNELS.index(d["channel"]), key=f"c_{k}")
            with f2:
                date = st.date_input("Date (UTC)", d["ts"].date(), key=f"d_{k}")
                tm = st.time_input("Time (UTC)", d["ts"].time(), key=f"t_{k}", step=300)
                sca = st.toggle("3DS / strong customer authentication performed", value=bool(d["sca"]), key=f"s_{k}")
            with f3:
                dev_choice = st.selectbox("Device (card-not-present)", ["known device", "new device"], index=1 if d["device"] == "new" else 0, key=f"dv_{k}")
                bal = st.number_input("Available balance / limit (GBP)", 0.0, 100000.0, float(round(pr.credit_limit * 0.6, -1)), 50.0, key=f"b_{k}")
                new_mcc = st.selectbox("MCC (new merchant only)", list(C.MCC_TABLE), format_func=lambda m: f"{m} · {C.MCC_TABLE[m][0]}", index=list(C.MCC_TABLE).index("5999"), key=f"m_{k}") if vid == "NEW" else None
                new_city = st.selectbox("Merchant city (new merchant only)", list(C.CITIES), key=f"ci_{k}") if vid == "NEW" else None
            default_desc = store.sample_descriptor(vid) if vid != "NEW" else ""
            desc = st.text_input("Transaction descriptor (free text)", value=default_desc, key=f"desc_{k}", help="Deviation from the merchant's usual descriptor pattern is itself a signal.")
            go_btn = st.button("Score this authorisation", type="primary", key=f"go_{k}", width="stretch")
        if go_btn:
            ts = pd.Timestamp.combine(date, tm)
            device = "new" if dev_choice == "new device" else ("known" if d["device"] in (None, "known", "new") else d["device"])
            new_vendor = dict(vendor_name="Unseen merchant", mcc=new_mcc, merchant_city=new_city) if vid == "NEW" else None
            txn = make_transaction(store, pid, None if vid == "NEW" else vid, amount, channel, ts, sca_applied=sca, device=device,
                                   descriptor=desc or None, new_vendor=new_vendor, balance=bal)
            extra = d["extra"] if isinstance(d["extra"], pd.DataFrame) else None
            if isinstance(d["extra"], str) and d["extra"] == "events":
                store_ev = store.ev_by_person.get(pid, np.array([]))
                store.ev_by_person[pid] = inject_ato_events(store, pid, ts)
                try:
                    trace = score_transaction(store, txn, policy, friction, extra_history=extra)
                finally:
                    store.ev_by_person[pid] = store_ev
            else:
                trace = score_transaction(store, txn, policy, friction, extra_history=extra)
            trace["scenario"] = PRESETS[preset]
            ui.audit_log().append(trace)
            st.session_state["last_trace"] = trace
        if st.session_state.get("last_trace"):
            st.divider()
            render_trace(store, st.session_state["last_trace"], key="manual")

    # ------------------------------------------------------------------ stream replay
    with tab_stream:
        sc = store.scores
        live = sc[sc.split == "live"].sort_values("t_sec").reset_index(drop=True)
        st.markdown(f"The live window holds the last {C.HISTORY_DAYS - C.SPLIT['test_end']} days ({len(live):,} authorisations). Disputes for this period have not matured, "
                    "so these transactions were never used for training - the scores below are true out-of-time predictions. Simulation truth is shown only to judge the model.")
        pos = st.session_state.setdefault("stream_pos", 0)
        b1, b2, b3, b4, b5 = st.columns([1.1, 1.1, 1.1, 0.8, 2.4])
        if b1.button("Next", key="next1", type="primary", icon=":material/skip_next:", width="stretch"):
            pos += 1
        if b2.button("Next alert", key="nexta", icon=":material/warning:", width="stretch"):
            nxt = live.index[(live.index > pos) & live.tier.isin(["STEP_UP", "DECLINE"])]
            pos = int(nxt[0]) if len(nxt) else pos
        if b3.button("Next fraud (truth)", key="nextf", icon=":material/gpp_bad:", width="stretch"):
            nxt = live.index[(live.index > pos) & (live.fraud_truth == 1)]
            pos = int(nxt[0]) if len(nxt) else pos
        if b4.button("Reset", key="reset", width="stretch"):
            pos = 0
        pos = int(min(max(pos, 0), len(live) - 1))
        st.session_state["stream_pos"] = pos
        row = live.iloc[pos]
        seen = live.iloc[: pos + 1]
        n_alert = int(seen.tier.isin(["STEP_UP", "DECLINE"]).sum())
        n_fraud = int(seen.fraud_truth.sum())
        caught = int(((seen.fraud_truth == 1) & seen.tier.isin(["STEP_UP", "DECLINE"])).sum())
        fp = int(((seen.fraud_truth == 0) & seen.tier.isin(["STEP_UP", "DECLINE"])).sum())
        ui.kpi_row([dict(label="Processed", value=f"{pos + 1:,}", sub=f"of {len(live):,} in the live window"),
                    dict(label="Alerts raised", value=f"{n_alert:,}", sub=f"{ui.fmt_pct(n_alert / (pos + 1), 2)} alert rate", status="warning"),
                    dict(label="Fraud attempts so far", value=f"{n_fraud:,}", sub="simulation truth"),
                    dict(label="Caught (step-up or decline)", value=f"{caught:,}", sub=f"{ui.fmt_pct(caught / max(n_fraud, 1), 0)} of attempts · {fp:,} false alerts", status="good"),
                    dict(label="Fraud value blocked", value=ui.fmt_gbp(seen[(seen.fraud_truth == 1) & seen.tier.isin(["STEP_UP", "DECLINE"])].transaction_amt.sum()), sub="if the policy had been enforced")])
        full = store.auth[store.auth.transaction_id == row.transaction_id].iloc[0]
        txn = transaction_from_row(store, full)
        trace = score_transaction(store, txn, policy, friction)
        trace["simulation_truth"] = dict(fraud_truth=int(row.fraud_truth), typology=row.fraud_typology if isinstance(row.fraud_typology, str) else None, label_status=row.label_status)
        truth = trace["simulation_truth"]
        st.html(f'<div class="fl-note">Authorisation <b>{row.transaction_id}</b> · {pd.Timestamp(row.transaction_timestamp):%Y-%m-%d %H:%M} UTC · {full.vendor_name} ({full.merchant_city}, MCC {full.mcc}) · '
                f'{ui.fmt_gbp(row.transaction_amt, 2)} · {row.channel} · customer {row.person_id} &nbsp;&nbsp; Simulation truth: '
                + (ui.badge("fraud · " + C.TYPOLOGY_SHORT.get(truth["typology"], str(truth["typology"])), "critical") if truth["fraud_truth"] else ui.badge("genuine", "good")) + "</div>")
        render_trace(store, trace, key=f"stream{pos}")
        st.markdown("**Recent feed**")
        feed = live.iloc[max(0, pos - 14): pos + 1][::-1][["transaction_id", "transaction_timestamp", "person_id", "vendor_id", "transaction_amt", "channel", "p_blend", "tier", "fraud_truth", "fraud_typology"]].copy()
        feed["vendor"] = feed.vendor_id.map(store.vendors.vendor_name)
        feed["decision"] = feed.tier.map(ui.TIER_LABEL)
        feed["truth"] = np.where(feed.fraud_truth == 1, "fraud: " + feed.fraud_typology.map(C.TYPOLOGY_SHORT).fillna("").astype(str), "genuine")
        st.dataframe(feed[["transaction_id", "transaction_timestamp", "person_id", "vendor", "transaction_amt", "channel", "p_blend", "decision", "truth"]], width="stretch", hide_index=True,
                     column_config={"p_blend": st.column_config.ProgressColumn("fraud probability", format="%.3f", min_value=0, max_value=1),
                                    "transaction_amt": st.column_config.NumberColumn("amount", format="£%.2f"),
                                    "transaction_timestamp": st.column_config.DatetimeColumn("time (UTC)", format="D MMM HH:mm")})
