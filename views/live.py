"""Live scoring: score a hand-built transaction (with scenario presets) or replay the live stream, with a full audit trace."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudlab import config as C, features as FE, ui
from fraudlab.scoring import make_transaction, score_transaction, transaction_from_row
from fraudlab.ui import ALERT_TIERS, P, STATUS

PRESETS = {
    "custom": "Custom",
    "typical": "Typical purchase (usual merchant, usual amount)",
    "overseas_cnp": "Overseas online spree (new merchant, night, new device, no 3DS)",
    "card_testing": "Card-testing burst (tiny declined attempts, then a purchase)",
    "ato": "Account takeover (password reset yesterday, big electronics order)",
    "lost_card": "Lost card cash-out (ATM + contactless far from home)",
    "fake_shop": "Fake online shop (fraudulent merchant, own device)",
}
RANDOM_PRESETS = {"overseas_cnp", "card_testing", "ato", "lost_card", "fake_shop"}   # presets that draw a merchant at random
MODES = ["Score a transaction", "Stream replay (live window)"]
CASE_WINDOW_H = 24          # alerts on the same customer within this many hours belong to one case


# ----------------------------------------------------------------------------------------------
# preset construction
# ----------------------------------------------------------------------------------------------
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


def ato_event_times(store, ts) -> np.ndarray:
    """Injected account-takeover precursors: a password reset and a new payee shortly before the order."""
    t = (ts - store.start).total_seconds()
    return np.array([t - 1.3 * 86400, t - 1.1 * 86400])


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
            ui.fmt_pct(policy["approve_max"], 1), ui.fmt_pct(policy["monitor_max"], 1), ui.fmt_pct(policy["stepup_max"], 0), ui.fmt_pct(policy["stepup_max"], 0)))
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
        from scipy.special import expit
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
                draws = expit(aa + sv * rng.normal(size=sv.shape))
            pop = np.random.default_rng(1).choice(vm.draws["mu"].reshape(-1), 2000)
            fig = ui.density_fig(draws, "this merchant", C.MODEL_COLORS["bayes_vendor"], title=f"Merchant fraud-rate posterior · level: {bv['level']}",
                                 ref=vm.pop_rate, ref_label="population", extra=[dict(draws=expit(pop), name="population", color=P["muted"])])
            ui.plot(fig, key=f"{key}_bv")
            st.markdown(f"Posterior mean **{ui.fmt_pct(bv['prob'], 2)}** (95% interval {ui.fmt_pct(bv['lo'], 2)}–{ui.fmt_pct(bv['hi'], 2)}); "
                        f"{bv['y']} confirmed fraud in {bv['n']:,} training transactions (raw {ui.fmt_pct(bv['raw'], 2) if bv['raw'] == bv['raw'] else 'n/a'}); "
                        f"P(rate > 3× population) = {ui.fmt_pct(bv['prob_gt_3x'], 0)}.")
        with b:
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
            fig = ui.hist_fig(base.transaction_amt, title=f"No merchant history - all {len(base)} prior amounts of the customer", xtitle="amount (GBP, log scale)", vline=amt, vline_label=f"now {ui.fmt_gbp(amt, 2)}", nbins=30, log_x=True)
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
def _keep_mode():
    """A segmented control can be clicked off; fall back to the previous mode instead of showing nothing."""
    if st.session_state.get("live_mode") is None:
        st.session_state["live_mode"] = st.session_state.get("_live_mode_last", MODES[0])
    st.session_state["_live_mode_last"] = st.session_state["live_mode"]


def render():
    store = ui.get_store()
    ui.page_header("Live scoring", "Score an incoming authorisation and inspect every step: features, each model's view, Bayesian priors, the blend and the decision. "
                                   "Thresholds and friction cost live under Decision policy in the sidebar.")
    if "live_mode" not in st.session_state:       # widget state is dropped while on other pages; restore the last mode
        st.session_state["live_mode"] = st.session_state.get("_live_mode_last", MODES[0])
    mode = st.segmented_control("Mode", MODES, key="live_mode", on_change=_keep_mode, label_visibility="collapsed")
    if mode == MODES[1]:
        render_stream(store)
    else:
        render_manual(store)


# ------------------------------------------------------------------ manual scoring
def _redraw():
    st.session_state["_draw"] = st.session_state.get("_draw", 0) + 1


def render_manual(store):
    policy, friction = ui.get_policy(), ui.get_friction()
    persons = store.persons
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("**1 · Customer**")
        pid = ui.customer_picker(store, key="live_person")
        ui.random_customer_buttons(store, key="live", compact=True)
        hist = store.person_history(pid)
        pr = persons.loc[pid]
        st.html(f'<div class="fl-note">{pr.segment.replace("_", " ").title()} · {pr.home_city} · tenure {int(pr.account_tenure_days)} days · '
                f'{len(hist):,} authorisations · {len(hist.vendor_id.unique())} merchants · limit {ui.fmt_gbp(pr.credit_limit)}</div>')
        st.markdown("**2 · Scenario**")
        preset = st.selectbox("Preset", list(PRESETS), format_func=lambda k: PRESETS[k], key="preset")
        draw = st.session_state.get("_draw", 0)
        # deterministic per (customer, draw) so the request does not jump on every rerun
        d = build_preset(store, pid, preset, np.random.default_rng([draw, zlib.crc32(pid.encode())]))
        if d["note"]:
            st.caption(d["note"])
        if preset in RANDOM_PRESETS:
            st.button("Draw a different merchant", icon=":material/casino:", on_click=_redraw, key="redraw")
    k = f"{preset}_{pid}_{draw}"       # new widget keys whenever the customer, preset or draw changes -> fresh defaults
    with c2:
        st.markdown("**3 · Authorisation request** (edit anything)")
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
    signature = (k, vid, amount, channel, str(date), str(tm), sca, dev_choice, bal, new_mcc, new_city, desc, tuple(policy.values()), friction)
    if go_btn:
        ts = pd.Timestamp.combine(date, tm)
        device = "new" if dev_choice == "new device" else ("known" if d["device"] in (None, "known", "new") else d["device"])
        new_vendor = dict(vendor_name="Unseen merchant", mcc=new_mcc, merchant_city=new_city) if vid == "NEW" else None
        txn = make_transaction(store, pid, None if vid == "NEW" else vid, amount, channel, ts, sca_applied=sca, device=device,
                               descriptor=desc or None, new_vendor=new_vendor, balance=bal)
        extra = d["extra"] if isinstance(d["extra"], pd.DataFrame) else None
        events = ato_event_times(store, ts) if isinstance(d["extra"], str) and d["extra"] == "events" else None
        trace = score_transaction(store, txn, policy, friction, extra_history=extra, extra_event_times=events)
        trace["scenario"] = PRESETS[preset]
        ui.audit_log().append(trace)
        st.session_state["last_trace"] = trace
        st.session_state["last_trace_sig"] = signature
    last = st.session_state.get("last_trace")
    if last:
        st.divider()
        last_sig = st.session_state.get("last_trace_sig") or ()
        if last["input"]["person_id"] != pid or (last_sig and last_sig[0] != k):
            st.info(f"The request above is new - press **Score this authorisation** to see its decision. "
                    f"(The previous result, {last['trace_id']} for customer {last['input']['person_id']}, is kept in the Audit log.)", icon=":material/bolt:")
            return
        if last_sig != signature:
            st.warning("The request or policy has been edited since this result was produced - press **Score this authorisation** to refresh it.", icon=":material/edit:")
        render_trace(store, last, key="manual")


# ------------------------------------------------------------------ stream replay
@st.cache_data(show_spinner=False)
def _live_window(_store, model_version: str) -> pd.DataFrame:
    """The unlabelled live window in arrival order, with the simulation's episode ids and display fields attached."""
    sc = _store.scores
    live = sc[sc.split == "live"].sort_values("t_sec", kind="stable").reset_index(drop=True)
    tx = _store.tx.set_index("transaction_id")
    live["episode"] = live.transaction_id.map(tx.fraud_episode_id).where(live.fraud_truth == 1)
    live["episode_open"] = live.episode.notna() & ~live.episode.duplicated()
    live["vendor"] = live.vendor_id.map(_store.vendors.vendor_name)
    return live


def _cases(live: pd.DataFrame, tiers: np.ndarray) -> pd.DataFrame:
    """Group alerts into cases: an alert on a customer who already had one in the last CASE_WINDOW_H hours joins that
    case instead of opening a new one - a fraud team works one case per compromised card, not one per attempt."""
    out = pd.DataFrame(index=live.index)
    out["tier"] = tiers
    out["alert"] = np.isin(tiers, ALERT_TIERS)
    a = live.loc[out.alert, ["person_id", "t_sec"]].sort_values(["person_id", "t_sec"])
    gap = a.t_sec - a.groupby("person_id").t_sec.shift()
    new = gap.isna() | (gap > CASE_WINDOW_H * 3600)
    seq = new.cumsum()
    first_t = a.t_sec.groupby(seq).transform("min")
    order = first_t.groupby(seq).first().rank(method="first").astype(int)       # number cases chronologically
    out["case"] = seq.map(order).map(lambda i: f"C-{i:03d}").reindex(out.index)
    out["case_open"] = new.reindex(out.index, fill_value=False).astype(bool)
    return out


def _set_pos(pos: int):
    st.session_state["stream_pos"] = int(pos)


def _jump(targets: np.ndarray, pos: int):
    nxt = targets[targets > pos]
    if len(nxt):
        _set_pos(nxt[0])
    else:
        st.toast("Nothing further in the live window.", icon=":material/info:")


def _truth_badge(row) -> str:
    if row.fraud_truth == 1:
        return ui.badge("fraud · " + C.TYPOLOGY_SHORT.get(row.fraud_typology, str(row.fraud_typology)), "critical")
    if row.fraud_typology == "friendly_fraud":
        return ui.badge("genuine · later disputed (friendly fraud)", "warning")
    return ui.badge("genuine", "good")


def render_stream(store):
    policy, friction = ui.get_policy(), ui.get_friction()
    live = _live_window(store, store.registry["model_version"])
    cs = _cases(live, ui.tiers_now(live.p_blend))
    n = len(live)
    st.caption(f"The live window holds the last {C.HISTORY_DAYS - C.SPLIT['test_end']} days ({n:,} authorisations in arrival order). Disputes for this period have not "
               "matured, so these transactions were never used for training - every score is a true out-of-time prediction. Simulation truth is shown only to judge the model.")

    pos = int(min(max(st.session_state.get("stream_pos", 0), 0), n - 1))
    case_opens = np.flatnonzero(cs.case_open.to_numpy())
    episode_opens = np.flatnonzero(live.episode_open.to_numpy())
    b1, b2, b3, b4, b5 = st.columns([1, 1.1, 1.3, 0.8, 1.6])
    b1.button("Next", key="next1", type="primary", icon=":material/skip_next:", width="stretch", on_click=_set_pos, args=(min(pos + 1, n - 1),),
              help="The next authorisation to arrive.")
    b2.button("Next alert", key="nexta", icon=":material/warning:", width="stretch", on_click=_jump, args=(case_opens, pos),
              help=f"The next authorisation that opens a new case (step-up or decline under the current policy). Further attempts on a customer already "
                   f"under an open case - e.g. the rest of a card-testing burst - join that case instead of raising a new alert ({CASE_WINDOW_H} h window).")
    b3.button("Next fraud episode (truth)", key="nextf", icon=":material/gpp_bad:", width="stretch", on_click=_jump, args=(episode_opens, pos),
              help="The first transaction of the next fraud episode in the simulation truth, whether or not the model alerted on it.")
    b4.button("Reset", key="reset", width="stretch", on_click=_set_pos, args=(0,))
    show_trace = b5.toggle("Show full decision trace", value=True, key="stream_trace", help="Turn off to step through the feed faster.")
    st.session_state["stream_pos"] = pos

    # ---- running totals up to the current position
    seen, seen_cs = live.iloc[: pos + 1], cs.iloc[: pos + 1]
    alert = seen_cs.alert.to_numpy()
    fraud = (seen.fraud_truth == 1).to_numpy()
    eps_seen = seen.episode.dropna().unique()
    eps_caught = seen.episode[alert & fraud].dropna().unique()
    cases_seen = seen_cs.case.dropna().unique()
    cases_with_fraud = seen_cs.case[fraud].dropna().unique()
    ui.kpi_row([
        dict(label="Processed", value=f"{pos + 1:,}", sub=f"of {n:,} · up to {pd.Timestamp(live.transaction_timestamp.iloc[pos]):%d %b %H:%M} UTC"),
        dict(label="Alerts raised", value=f"{int(alert.sum()):,}", sub=f"{len(cases_seen):,} cases · {ui.fmt_pct(alert.mean(), 2)} alert rate", status="warning"),
        dict(label="Fraud episodes so far", value=f"{len(eps_seen):,}", sub=f"{int(fraud.sum()):,} fraudulent authorisations (simulation truth)"),
        dict(label="Episodes caught", value=f"{len(eps_caught):,}", sub=f"{ui.fmt_pct(len(eps_caught) / max(len(eps_seen), 1), 0)} of episodes · "
                                                                       f"{len(set(cases_seen) - set(cases_with_fraud)):,} false-positive cases", status="good"),
        dict(label="Fraud value stopped", value=ui.fmt_gbp(seen.transaction_amt[alert & fraud].sum()), sub=f"of {ui.fmt_gbp(seen.transaction_amt[fraud].sum())} attempted, if the policy were enforced"),
    ])

    # ---- the authorisation at the current position
    row, crow = live.iloc[pos], cs.iloc[pos]
    full = store.auth[store.auth.transaction_id == row.transaction_id].iloc[0]
    person = store.persons.loc[row.person_id]
    cells = [
        ("Authorisation", row.transaction_id, f"{pd.Timestamp(row.transaction_timestamp):%a %d %b %Y %H:%M} UTC"),
        ("Customer", row.person_id, f"{person.segment.replace('_', ' ')} · {person.home_city}"),
        ("Merchant", full.vendor_name, f"{full.merchant_city} · MCC {full.mcc} {C.MCC_TABLE[str(full.mcc)][0]}"),
        ("Amount · channel", ui.fmt_gbp(row.transaction_amt, 2), f"{row.channel} · {full.auth_result} by the issuer"),
        ("Model decision", ui.tier_badge(crow.tier), f"fraud probability {ui.fmt_pct(row.p_blend, 1) if row.p_blend >= 0.001 else '< 0.1%'}"),
        ("Simulation truth", _truth_badge(row), f"episode {row.episode}" if isinstance(row.episode, str) else "&nbsp;"),
    ]
    st.html('<div class="fl-txn">' + "".join(f'<div><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for l, v, s in cells) + "</div>")

    # ---- the case this authorisation belongs to (if any)
    if isinstance(crow.case, str):
        members = cs.index[cs.case == crow.case]
        so_far = members[members <= pos]
        k_in, k_all = len(so_far), len(members)
        span_min = (live.t_sec[members].max() - live.t_sec[members].min()) / 60
        opened = pd.Timestamp(live.transaction_timestamp[members[0]])
        lead = "This authorisation opened" if crow.case_open else "This authorisation joins"
        rest = (f" The case will group <b>{k_all - k_in}</b> more alert{'s' if k_all - k_in != 1 else ''} on this customer over the next "
                f"{span_min:.0f} min - <b>Next alert</b> skips those and moves to the next new case.") if k_all > k_in else ""
        st.html(f'<div class="fl-case">{lead} case <b>{crow.case}</b> (opened {opened:%d %b %H:%M} UTC) - alert {k_in} of {k_all} on customer {row.person_id}.{rest}</div>')
        ctx = live[(live.person_id == row.person_id) & (live.t_sec >= live.t_sec[members[0]] - 6 * 3600) & (live.index <= pos)]
        tab = ctx[["transaction_timestamp", "vendor", "transaction_amt", "channel", "p_blend"]].copy()
        tab["decision"] = cs.tier[ctx.index].map(ui.TIER_LABEL)
        tab["case"] = cs.case[ctx.index].fillna("")
        tab = tab.iloc[::-1]
        with st.expander(f"Customer {row.person_id}'s activity in this case so far ({len(tab)} authorisations)", expanded=False):
            st.dataframe(tab, width="stretch", hide_index=True,
                         column_config={"p_blend": st.column_config.ProgressColumn("fraud probability", format="%.3f", min_value=0, max_value=1),
                                        "transaction_amt": st.column_config.NumberColumn("amount", format="£%.2f"),
                                        "transaction_timestamp": st.column_config.DatetimeColumn("time (UTC)", format="D MMM HH:mm:ss")})

    # ---- recent feed (newest first; the top row is the authorisation above)
    feed = live.iloc[max(0, pos - 14): pos + 1][::-1].copy()
    feed.insert(0, "now", np.where(feed.index == pos, "▶", ""))
    feed["decision"] = cs.tier[feed.index].map(ui.TIER_LABEL)
    feed["case"] = cs.case[feed.index].fillna("")
    feed["truth"] = np.where(feed.fraud_truth == 1, "fraud: " + feed.fraud_typology.map(C.TYPOLOGY_SHORT).fillna("").astype(str), "genuine")
    st.markdown("**Recent feed** (newest first)")
    st.dataframe(feed[["now", "transaction_timestamp", "person_id", "vendor", "transaction_amt", "channel", "p_blend", "decision", "case", "truth"]], width="stretch", hide_index=True,
                 column_config={"now": st.column_config.TextColumn("", width="small"),
                                "p_blend": st.column_config.ProgressColumn("fraud probability", format="%.3f", min_value=0, max_value=1),
                                "transaction_amt": st.column_config.NumberColumn("amount", format="£%.2f"),
                                "transaction_timestamp": st.column_config.DatetimeColumn("time (UTC)", format="D MMM HH:mm")})

    if show_trace:
        ui.section("Decision trace for this authorisation", "Re-scored live from the raw request with the as-of feature pipeline - it reproduces the stored batch score.")
        trace = score_transaction(store, transaction_from_row(store, full), policy, friction)
        trace["simulation_truth"] = dict(fraud_truth=int(row.fraud_truth), typology=row.fraud_typology if isinstance(row.fraud_typology, str) else None, label_status=row.label_status)
        render_trace(store, trace, key=f"stream{pos}")
