"""Model lab: data split, evaluation on the out-of-time test window, explanations, tree rules, stacker and threshold economics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudlab import config as C, features as FE, ui
from fraudlab.models import STACK_LABELS, confusion_at
from fraudlab.ui import P, STATUS

MODEL_ORDER = ["blend", "xgb", "tree", "scorecard", "bayes_vendor", "bayes_person"]
NAMES = dict(blend="Blend (stacker)", **STACK_LABELS)


def render():
    store = ui.get_store()
    ev, reg, sc = store.evaluation, store.registry, store.scores
    ui.page_header("Model lab", "How the models were trained and how well they work on an out-of-time test window they never saw - plus the economics of the decision thresholds.")

    # ------------------------------------------------------------------ split timeline
    ui.section("Time-based split with label maturity", "Training on the past, validating and testing strictly on later weeks - the only honest way to evaluate a fraud model.")
    s = C.SPLIT
    segs = [("Train", "train", 0, s["train_end"], P["series"][0]), ("Validation (early stopping, calibration, stacker)", "valid", s["train_end"], s["valid_end"], P["series"][2]),
            ("Test (all metrics below)", "test", s["valid_end"], s["test_end"], P["series"][1]), ("Live stream (labels immature)", "live", s["test_end"], C.HISTORY_DAYS, P["muted"])]
    fig = go.Figure()
    for name, key, a, b, col in segs:
        fig.add_trace(go.Bar(x=[b - a], y=["window"], base=[a], orientation="h", name=name, marker=dict(color=col, line=dict(color=P["surface"], width=2)),
                             text=f"{name}<br>{ev['split_rows'][key]:,} rows", textposition="inside", insidetextanchor="middle",
                             textfont=dict(color="#ffffff", size=11), hovertemplate=f"{name}<br>day {a}–{b}<extra></extra>"))
    fig.update_layout(barmode="stack", height=130, margin=dict(l=10, r=10, t=10, b=30), xaxis=dict(title="days since start of history", range=[0, C.HISTORY_DAYS]),
                      yaxis=dict(showticklabels=False), showlegend=False, bargap=0.1)
    ui.plot(fig)
    ui.kpi_row([dict(label=f"{k.title()} rows", value=f"{ev['split_rows'][k]:,}", sub=f"{ev['split_fraud'][k]:,} confirmed fraud ({ev['split_fraud'][k] / ev['split_rows'][k]:.2%})") for k in ["train", "valid", "test", "live"]])

    # ------------------------------------------------------------------ metrics table
    ui.section("Out-of-time test performance", "All numbers are on the test window; the stacker and calibration only ever saw the validation window.")
    rows = []
    for k in MODEL_ORDER:
        m = ev["metrics"][k]
        rows.append(dict(model=NAMES[k], **{"AUC-ROC": m["auc_roc"], "PR-AUC": m["auc_pr"], "KS": m["ks"], "Brier": m["brier"], "log-loss": m["log_loss"],
                                            "recall @0.5%": m["recall@0.5%"], "recall @1%": m["recall@1.0%"], "recall @2%": m["recall@2.0%"],
                                            "precision @1%": m["precision@1.0%"], "value recall @1%": m.get("value_recall@1.0%", np.nan)}))
    mt = pd.DataFrame(rows).set_index("model")
    st.dataframe(mt.style.format("{:.3f}").background_gradient(cmap="Blues", subset=["AUC-ROC", "PR-AUC", "KS", "recall @1%", "precision @1%", "value recall @1%"], axis=0), width="stretch", height=260)
    st.caption("recall @x% = share of confirmed fraud caught when the x% highest-scoring authorisations are alerted; value recall weights by amount. "
               "The Bayesian priors are deliberately weak on their own - they carry the merchant/customer context into the blend and the cold-start intervals.")

    a, b, c = st.columns(3)
    with a:
        fig = go.Figure()
        for k in ["blend", "xgb", "tree", "scorecard", "bayes_vendor"]:
            r = ev["curves"][k]["roc"]
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=NAMES[k], line=dict(color=C.MODEL_COLORS[k], width=2 if k == "blend" else 1.5), hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra>" + NAMES[k] + "</extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=P["axis"], width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
        fig.update_layout(title="ROC curves (test)", xaxis=dict(title="false positive rate", range=[0, 0.2]), yaxis=dict(title="true positive rate"), height=320, legend=dict(font=dict(size=10)))
        ui.plot(fig)
    with b:
        fig = go.Figure()
        for k in ["blend", "xgb", "tree", "scorecard", "bayes_vendor"]:
            r = ev["curves"][k]["pr"]
            fig.add_trace(go.Scatter(x=r["recall"], y=r["precision"], mode="lines", name=NAMES[k], line=dict(color=C.MODEL_COLORS[k], width=2 if k == "blend" else 1.5), hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra>" + NAMES[k] + "</extra>"))
        fig.update_layout(title="Precision–recall curves (test)", xaxis_title="recall", yaxis_title="precision", height=320, legend=dict(font=dict(size=10)))
        ui.plot(fig)
    with c:
        cal = ev["curves"]["blend"]["calibration"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, max(cal["pred"]) * 1.05], y=[0, max(cal["pred"]) * 1.05], mode="lines", line=dict(color=P["axis"], width=1, dash="dot"), name="perfect", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers", name="blend", line=dict(color=C.MODEL_COLORS["blend"], width=2), marker=dict(size=9, line=dict(color=P["surface"], width=2)),
                                 text=[f"n={n:,}" for n in cal["n"]], hovertemplate="predicted %{x:.2%}<br>observed %{y:.2%}<br>%{text}<extra></extra>"))
        fig.update_layout(title="Calibration (score deciles, test)", xaxis=dict(title="mean predicted probability", tickformat=".0%"), yaxis=dict(title="observed fraud rate", tickformat=".0%"), height=320)
        ui.plot(fig)

    # ------------------------------------------------------------------ policy economics
    ui.section("Decision thresholds and economics", "Move the sliders in the sidebar of Live scoring to change the policy; this section shows what the policy does on the test window.")
    pol = ui.get_policy()
    te = sc[sc.split == "test"]
    y, p, amt = te.fraud_label.to_numpy(), te.p_blend.to_numpy(), te.transaction_amt.abs().to_numpy()
    a, b = st.columns([1, 1.4])
    with a:
        rows = []
        for name, thr in [("Decline", pol["stepup_max"]), ("Step-up or decline", pol["monitor_max"]), ("Any alert (monitor+)", pol["approve_max"])]:
            cm = confusion_at(y, p, thr)
            rows.append(dict(action=name, threshold=thr, alerts=cm["tp"] + cm["fp"], alert_rate=cm["alert_rate"], fraud_caught=cm["tp"], fraud_missed=cm["fn"], false_alerts=cm["fp"], precision=cm["precision"], recall=cm["recall"]))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"threshold": st.column_config.NumberColumn(format="%.3f"), "alert_rate": st.column_config.NumberColumn("alert rate", format="%.2%"),
                                    "precision": st.column_config.NumberColumn(format="%.1%"), "recall": st.column_config.NumberColumn(format="%.1%")})
        r = ev["curves"]["blend"]["alert"]
        fig = go.Figure(go.Scatter(x=r["rate"], y=r["recall"], mode="lines", line=dict(color=C.MODEL_COLORS["blend"], width=2), name="blend", hovertemplate="alert %{x:.1%} of authorisations<br>catches %{y:.0%} of fraud<extra></extra>"))
        fig.update_layout(title="Fraud caught vs share of authorisations alerted", xaxis=dict(title="alert rate", tickformat=".0%"), yaxis=dict(title="fraud recall", tickformat=".0%"), height=260)
        ui.plot(fig)
    with b:
        c1, c2 = st.columns(2)
        friction = c1.slider("Cost of a false decline (£, friction + attrition)", 1.0, 60.0, 8.0, 1.0, key="ml_friction")
        recov = c2.slider("Share of fraud value recovered anyway (chargeback)", 0.0, 0.9, 0.3, 0.05, key="ml_recov")
        thr = np.linspace(0.005, 0.9, 120)
        cost = []
        for t in thr:
            dec = p >= t
            loss = ((~dec) & (y == 1)) * amt * (1 - recov)
            fric = (dec & (y == 0)) * (friction + 0.02 * amt)
            cost.append(float(loss.sum() + fric.sum()))
        cost = np.array(cost)
        best = thr[int(cost.argmin())]
        fig = go.Figure(go.Scatter(x=thr, y=cost, mode="lines", line=dict(color=P["series"][0], width=2), name="expected cost", hovertemplate="threshold %{x:.3f}<br>cost £%{y:,.0f}<extra></extra>"))
        fig.add_vline(x=best, line=dict(color=P["diverging_pos"], width=1.5, dash="dot"), annotation_text=f"minimum at {best:.3f}", annotation_font=dict(size=11))
        fig.add_vline(x=pol["stepup_max"], line=dict(color=P["ink"], width=1, dash="dot"), annotation_text=f"current decline threshold {pol['stepup_max']:.2f}", annotation_position="top left", annotation_font=dict(size=11))
        fig.update_layout(title="Total cost on the test window = unrecovered fraud losses + false-decline friction, by decline threshold", xaxis_title="decline threshold (probability)", yaxis_tickprefix="£", height=320)
        ui.plot(fig)
        st.caption(f"With these assumptions the cost-minimising decline threshold is {best:.3f} (cost {ui.fmt_gbp(cost.min())} vs {ui.fmt_gbp(cost[np.abs(thr - pol['stepup_max']).argmin()])} at the current policy). "
                   "Step-up authentication sits below it because a 3DS challenge costs far less friction than a decline.")

    # ------------------------------------------------------------------ recall by typology
    a, b = st.columns([1.2, 1])
    with a:
        ty = pd.DataFrame(ev["typology"])
        ty = ty[ty.typology != "genuine"]
        fig = go.Figure()
        for t, lab, s in C.TIERS:
            fig.add_trace(go.Bar(x=[C.TYPOLOGY_SHORT.get(x, x) for x in ty.typology], y=ty[t], name=lab, marker=dict(color=STATUS[s], line=dict(color=P["surface"], width=1)),
                                 hovertemplate="%{x}<br>" + lab + ": %{y:.0%}<extra></extra>"))
        fig.update_layout(barmode="stack", title="Which fraud types does the policy catch? Tier mix per typology (test window, simulation truth)", yaxis=dict(tickformat=".0%"), height=340, xaxis=dict(tickfont=dict(size=10.5)))
        ui.plot(fig)
        st.caption("Friendly fraud is genuine spend later disputed - no transaction-time model can catch it, which is exactly why the dictionary asks for dispute_resolution.")
    with b:
        w = ev["stacker"]["weights"]
        fig = ui.bar_fig([w[k] for k in STACK_LABELS], [STACK_LABELS[k] for k in STACK_LABELS], orientation="h", title=f"Stacker weights on component log-odds (intercept {ev['stacker']['intercept']:+.2f})",
                         text=[f"{w[k]:+.2f}" for k in STACK_LABELS], colors=[C.MODEL_COLORS[k] for k in STACK_LABELS], hover="%{y}<br>weight %{x:+.3f}<extra></extra>", height=340)
        fig.update_layout(margin=dict(l=10, r=50, t=36, b=40), yaxis=dict(autorange="reversed"))
        ui.plot(fig)
        st.caption("A logistic regression fitted on the validation window: final log-odds = intercept + Σ weight × component log-odds. Weights near zero mean the component adds little beyond XGBoost on this data - the blend keeps them for interpretability and cold-start intervals.")

    # ------------------------------------------------------------------ XGBoost internals
    ui.section("XGBoost", f"{ev['xgb_best_iteration'] + 1} boosting rounds selected by early stopping on validation log-loss; probabilities isotonic-calibrated on validation.")
    a, b, c = st.columns([1.2, 1.2, 1])
    with a:
        g = pd.Series(ev["shap_global"]).sort_values(ascending=False).head(18)
        fig = ui.bar_fig(g.values[::-1], g.index[::-1], orientation="h", title="Global importance: mean |SHAP| on the test window (log-odds)", hover="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>", height=460)
        fig.update_layout(margin=dict(l=10, r=10, t=36, b=40), yaxis=dict(tickfont=dict(size=10.5)))
        ui.plot(fig)
    with b:
        g2 = pd.Series(ev["gain_importance"]).sort_values(ascending=False).head(18)
        fig = ui.bar_fig(g2.values[::-1], g2.index[::-1], orientation="h", title="Split gain importance (training)", color=P["series"][2], hover="%{y}<br>gain %{x:.1f}<extra></extra>", height=460)
        fig.update_layout(margin=dict(l=10, r=10, t=36, b=40), yaxis=dict(tickfont=dict(size=10.5)))
        ui.plot(fig)
    with c:
        cur = ev["xgb_eval_curve"]
        fig = go.Figure(go.Scatter(y=cur, mode="lines", line=dict(color=P["series"][0], width=2), name="validation PR-AUC", hovertemplate="round %{x}<br>PR-AUC %{y:.3f}<extra></extra>"))
        fig.add_vline(x=ev["xgb_best_iteration"], line=dict(color=P["diverging_pos"], width=1.5, dash="dot"), annotation_text="selected", annotation_font=dict(size=10))
        fig.update_layout(title="Validation PR-AUC per boosting round", xaxis_title="round", height=230)
        ui.plot(fig)
        prm = {k: v for k, v in ev["xgb_params"].items() if k in ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "gamma", "scale_pos_weight")}
        st.dataframe(pd.DataFrame({"parameter": list(prm), "value": list(prm.values())}), width="stretch", hide_index=True, height=230)

    # ------------------------------------------------------------------ decision tree
    ui.section("Decision tree", f"{ev['tree_n_leaves']} leaves, depth ≤ 6, balanced class weights, ≥ 150 training transactions per leaf. Every leaf reports its honest fraud rate.")
    a, b = st.columns([1.4, 1])
    with a:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        tree = store.models["tree"]
        fig_m, ax = plt.subplots(figsize=(13, 5.2), dpi=110)
        plot_tree(tree.clf, feature_names=FE.MODEL_FEATURES, class_names=["genuine", "fraud"], filled=True, rounded=True, max_depth=3, fontsize=7, impurity=False, proportion=True, ax=ax)
        ax.set_title("Top of the tree (depth 3 shown; colour = majority class after balancing)", fontsize=9, color=P["ink2"], loc="left")
        fig_m.patch.set_alpha(0)
        st.pyplot(fig_m, width="stretch")
        plt.close(fig_m)
    with b:
        lt = tree.leaf_table.sort_values("rate", ascending=False).head(12).reset_index()
        st.dataframe(lt, width="stretch", hide_index=True, height=330, column_config={"leaf": "leaf id", "n_train": "training rows", "n_fraud": "fraud", "rate": st.column_config.NumberColumn("fraud rate", format="%.2%")})
        with st.expander("Full rule set (text export)"):
            st.code(ev["tree_rules"], language="text")

    # ------------------------------------------------------------------ scorecard
    ui.section("Logistic scorecard", "Standardised features × coefficients: the classic points-based scorecard that model-risk teams can audit line by line.")
    coef = pd.Series(ev["scorecard_coef"])
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(20)
    fig = go.Figure(go.Bar(x=top.values[::-1], y=top.index[::-1], orientation="h", marker=dict(color=[P["diverging_pos"] if v > 0 else P["diverging_neg"] for v in top.values[::-1]], line_width=0),
                           hovertemplate="%{y}<br>coefficient %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=f"Largest coefficients (log-odds per standard deviation; intercept {ev['scorecard_intercept']:+.2f})", height=460, margin=dict(l=10, r=10, t=36, b=40), yaxis=dict(tickfont=dict(size=10.5)))
    fig.add_vline(x=0, line_color=P["axis"], line_width=1)
    ui.plot(fig)

    # ------------------------------------------------------------------ model card
    with st.expander("Model card", expanded=False):
        st.markdown(f"""
**Purpose.** Score every card authorisation of {C.BANK_NAME} for the probability that it is fraudulent, at authorisation time, and map it to an action.

**Data.** Synthetic 18-month history ({reg['world']['n_legit_rows_generated']:,} genuine rows generated, {reg['world']['n_episodes']} injected fraud episodes across {reg['world']['n_victims']} customers).
Label = confirmed fraud (scheme reason codes 10.x, dispute not reversed or withdrawn). Labels younger than {C.LABEL_MATURITY_DAYS} days are treated as unknown.

**Features.** {reg['n_features']} features, all computed from information available before the scoring time (see Data & dictionary → lineage).

**Models.** Decision tree (depth 6), XGBoost ({ev['xgb_best_iteration'] + 1} rounds, isotonic calibration), logistic scorecard, hierarchical Bayesian merchant prior and Beta-Binomial customer prior (MCMC, 4 chains), blended by a logistic stacker fitted on the validation window.

**Performance (test, out-of-time).** AUC {ev['metrics']['blend']['auc_roc']:.3f}, PR-AUC {ev['metrics']['blend']['auc_pr']:.3f}, {ev['metrics']['blend']['recall@1.0%']:.0%} of confirmed fraud caught at a 1% alert rate.

**Known limitations.** Synthetic data; fraud enriched ≈10× reality; friendly fraud is undetectable at authorisation time; merchant country and city proxy for protected characteristics and need a fairness review before production.

**Reproducibility.** Seed {reg['seed']}, model version `{reg['model_version']}`, built {reg['built_at']} with pandas {reg['libraries']['pandas']}, scikit-learn {reg['libraries']['sklearn']}, XGBoost {reg['libraries']['xgboost']}.
""")
