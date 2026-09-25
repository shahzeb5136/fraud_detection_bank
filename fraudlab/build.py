"""One-shot build: synthetic world -> features -> MCMC priors -> models -> stacker -> evaluation -> artifacts.

Run:  python -m fraudlab.build
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import xgboost

from . import __version__, bayes, config as C, features as FE, models as M, synth

ART = C.ARTIFACT_DIR


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bayes_components(F: pd.DataFrame, vm: bayes.VendorHierarchyMCMC, pm: bayes.PersonPropensityMCMC) -> pd.DataFrame:
    """Posterior-mean fraud rates per row (unseen merchants back off to MCC; unseen customers to the population)."""
    vend_mean = pd.Series(vm.vendor_summary["post_mean"], index=vm.vendor_summary["vendor_id"])
    mcc_new = {m: vm.new_vendor_rate(m)["mean"] for m in C.MCC_TABLE}
    pv = F.vendor_id.map(vend_mean)
    pv = pv.fillna(F.mcc.map(mcc_new)).fillna(vm.pop_rate).to_numpy()
    pers_mean = pd.Series(pm.person_summary["post_mean"], index=pm.person_summary["person_id"])
    pp = F.person_id.map(pers_mean).fillna(pm.pop_mean).to_numpy()
    return pd.DataFrame({"bayes_vendor_p": pv, "bayes_person_p": pp}, index=F.index)


def main(seed: int = C.SEED):
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    (ART / "data").mkdir(exist_ok=True)
    start = pd.Timestamp(C.AS_OF_DATE) - pd.Timedelta(days=C.HISTORY_DAYS)

    # 1. synthetic world -------------------------------------------------------------------------
    _log("1/6 generating synthetic world")
    world = synth.generate(seed)
    tx, events = world["transactions"], world["events"]
    for name in ["transactions", "events", "persons", "cards", "vendors", "episodes"]:
        world[name].to_parquet(ART / "data" / f"{name}.parquet", index=False)
    _log(f"    {len(tx):,} transactions, {tx.fraud_label.mean():.2%} labelled fraud, {tx.fraud_truth.mean():.2%} true fraud")

    # 2. features ----------------------------------------------------------------------------------
    _log("2/6 engineering leakage-safe features")
    F, lk = FE.build_features(tx, events, start, C.SPLIT["train_end"], log=_log)
    splits = M.time_split(F)
    for k, v in splits.items():
        _log(f"    {k:5s}: {len(v):7,} rows, {int(v.fraud_label.sum()):5d} fraud ({v.fraud_label.mean():.2%})")

    # 3. Bayesian priors by MCMC -----------------------------------------------------------------
    _log("3/6 MCMC: merchant hierarchy (vendor within MCC within population)")
    tr = splits["train"]
    agg_v = tr.groupby("vendor_id").agg(n=("fraud_label", "size"), y=("fraud_label", "sum"), mcc=("mcc", "first")).reset_index()
    vm = bayes.VendorHierarchyMCMC(agg_v.vendor_id.values, agg_v.n.values, agg_v.y.values, agg_v.mcc.values, list(C.MCC_TABLE)).run(log=_log)
    _log("    MCMC: customer propensity (Beta-Binomial)")
    agg_p = tr.groupby("person_id").agg(n=("fraud_label", "size"), y=("fraud_label", "sum")).reset_index()
    pm = bayes.PersonPropensityMCMC(agg_p.person_id.values, agg_p.n.values, agg_p.y.values).run(log=_log)
    chan = bayes.channel_posteriors(tr)
    B = bayes_components(F, vm, pm)

    # 4. supervised models -------------------------------------------------------------------------
    _log("4/6 training decision tree, XGBoost and logistic scorecard")
    Xtr, ytr = splits["train"], splits["train"].fraud_label.to_numpy()
    Xva, yva = splits["valid"], splits["valid"].fraud_label.to_numpy()
    tree = M.TreeModel().fit(Xtr, ytr)
    _log(f"    tree: {tree.n_leaves} leaves")
    xg = M.XGBModel().fit(Xtr, ytr, Xva, yva)
    _log(f"    xgboost: best iteration {xg.best_iteration}")
    sc = M.ScorecardModel().fit(Xtr, ytr)
    _log("    scorecard fitted")

    # 5. components + stacker ---------------------------------------------------------------------------
    _log("5/6 blending and evaluating")
    comp = pd.DataFrame(index=F.index)
    comp["p_xgb_raw"] = xg.predict_raw(F)
    comp["p_xgb"] = xg.predict_proba(F)
    comp["p_tree"] = tree.predict_proba(F)
    comp["p_scorecard"] = sc.predict_proba(F)
    comp["p_bayes_vendor"] = B.bayes_vendor_p
    comp["p_bayes_person"] = B.bayes_person_p
    L = pd.DataFrame({k: M.clip_logit(comp[f"p_{k}"]) for k in M.STACK_INPUTS}, index=F.index)
    stack = M.Stacker().fit(L.loc[Xva.index], yva)
    comp["p_blend"] = stack.predict_proba(L)
    _log("    stacker weights: " + ", ".join(f"{k}={v:.2f}" for k, v in stack.weights.items()))

    evaluation = dict(split_rows={k: int(len(v)) for k, v in splits.items()},
                      split_fraud={k: int(v.fraud_label.sum()) for k, v in splits.items()},
                      split_days=C.SPLIT, stacker=dict(intercept=stack.intercept, weights=stack.weights.to_dict()),
                      metrics={}, curves={}, typology={}, policy_eval={})
    te = splits["test"]
    yte = te.fraud_label.to_numpy()
    amt = te.transaction_amt.to_numpy()
    for k in ["blend", "xgb", "tree", "scorecard", "bayes_vendor", "bayes_person"]:
        p = comp.loc[te.index, f"p_{k}"].to_numpy()
        evaluation["metrics"][k] = M.metrics(yte, p, amt)
        evaluation["curves"][k] = M.curves(yte, p)
        _log(f"    test {k:12s} AUC={evaluation['metrics'][k]['auc_roc']:.3f} PR-AUC={evaluation['metrics'][k]['auc_pr']:.3f} recall@1%={evaluation['metrics'][k]['recall@1.0%']:.2f}")
    # validation metrics too (for the model card)
    for k in ["blend", "xgb", "tree", "scorecard"]:
        evaluation["metrics"][f"{k}_valid"] = M.metrics(yva, comp.loc[Xva.index, f"p_{k}"].to_numpy(), Xva.transaction_amt.to_numpy())
    # recall by typology (test) for the blend at each policy tier
    pb = comp.loc[te.index, "p_blend"].to_numpy()
    tiers = M.tiers_vec(pb)
    typ = te.fraud_typology.fillna("genuine").to_numpy()
    rows = []
    for tname in list(C.TYPOLOGIES) + ["genuine"]:
        mask = typ == tname
        if mask.sum() == 0:
            continue
        rows.append(dict(typology=tname, n=int(mask.sum()), n_fraud_label=int(yte[mask].sum()),
                         **{t[0]: float((tiers[mask] == t[0]).mean()) for t in C.TIERS}, mean_p=float(pb[mask].mean())))
    evaluation["typology"] = rows
    # policy evaluation on test
    for name, thr in [("decline", C.DEFAULT_POLICY["stepup_max"]), ("stepup", C.DEFAULT_POLICY["monitor_max"]), ("monitor", C.DEFAULT_POLICY["approve_max"])]:
        evaluation["policy_eval"][name] = M.confusion_at(yte, pb, thr)
    # global explanations
    samp = te.sample(n=min(4000, len(te)), random_state=seed)
    shap_te = xg.contributions(samp)
    evaluation["shap_global"] = shap_te.drop(columns=["bias"]).abs().mean().sort_values(ascending=False).round(5).to_dict()
    evaluation["shap_signed_mean"] = shap_te.drop(columns=["bias"]).mean().round(5).to_dict()
    evaluation["gain_importance"] = xg.gain_importance.round(4).to_dict()
    evaluation["xgb_eval_curve"] = [float(v) for v in xg.eval_curve]
    evaluation["xgb_params"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in xg.params.items()}
    evaluation["xgb_best_iteration"] = xg.best_iteration
    evaluation["tree_rules"] = tree.rules_text
    evaluation["tree_n_leaves"] = tree.n_leaves
    evaluation["scorecard_coef"] = sc.coef.round(4).to_dict()
    evaluation["scorecard_intercept"] = sc.intercept
    evaluation["channel_posteriors"] = chan
    evaluation["bayes_vendor_hyper"] = vm.hyper_summary
    evaluation["bayes_person_hyper"] = pm.hyper_summary
    evaluation["bayes_acceptance"] = dict(vendor=vm.acceptance, person=pm.acceptance)
    evaluation["fraud_typologies"] = C.TYPOLOGIES

    # 6. persist ----------------------------------------------------------------------------------------
    _log("6/6 writing artifacts")
    scores = pd.concat([F[["transaction_id", "person_id", "vendor_id", "t_sec", "transaction_timestamp", "transaction_amt", "channel",
                           "mcc", "merchant_country", "fraud_label", "fraud_truth", "fraud_typology", "label_status", "label_mature"]], comp], axis=1)
    d = F.t_sec / 86400.0
    scores["split"] = np.select([d < C.SPLIT["train_end"], d < C.SPLIT["valid_end"], d < C.SPLIT["test_end"]], ["train", "valid", "test"], default="live")
    scores["tier"] = M.tiers_vec(scores.p_blend.to_numpy())
    scores.to_parquet(ART / "scores.parquet", index=False)
    F.to_parquet(ART / "features.parquet", index=False)
    with open(ART / "lookups.pkl", "wb") as fh:
        pickle.dump(lk, fh)
    with open(ART / "models.pkl", "wb") as fh:
        pickle.dump(dict(tree=tree, xgb=xg, scorecard=sc, stacker=stack), fh)
    with open(ART / "bayes.pkl", "wb") as fh:
        pickle.dump(dict(vendor=vm, person=pm), fh)
    with open(ART / "evaluation.json", "w") as fh:
        json.dump(evaluation, fh, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    hashes = {}
    for f in ["models.pkl", "bayes.pkl", "lookups.pkl", "features.parquet", "scores.parquet", "data/transactions.parquet"]:
        hashes[f] = hashlib.sha256((ART / f).read_bytes()).hexdigest()[:16]
    registry = dict(app_version=__version__, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), seed=seed,
                    as_of_date=C.AS_OF_DATE, history_days=C.HISTORY_DAYS, split=C.SPLIT, label_maturity_days=C.LABEL_MATURITY_DAYS,
                    policy=C.DEFAULT_POLICY, n_features=len(FE.MODEL_FEATURES), hashes=hashes,
                    model_version="fl-" + hashlib.sha256("".join(hashes.values()).encode()).hexdigest()[:10],
                    libraries=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__),
                    world=world["meta"], build_seconds=round(time.time() - t0, 1))
    with open(ART / "registry.json", "w") as fh:
        json.dump(registry, fh, indent=2)
    _log(f"done in {time.time() - t0:.0f}s -> {ART}")
    return registry


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else C.SEED)
