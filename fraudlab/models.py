"""Supervised models: decision tree (rules), XGBoost (patterns), logistic scorecard (points) and the
transparent stacker that blends them with the Bayesian priors. Also metrics, curves and explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config as C
from .features import MODEL_FEATURES

EPS = 1e-5


def clip_logit(p):
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


# ----------------------------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------------------------
def time_split(F: pd.DataFrame) -> dict:
    d = F.t_sec / 86400.0
    s = C.SPLIT
    ok = F.label_mature.astype(bool)
    return dict(
        train=F[(d < s["train_end"]) & ok],
        valid=F[(d >= s["train_end"]) & (d < s["valid_end"]) & ok],
        test=F[(d >= s["valid_end"]) & (d < s["test_end"]) & ok],
        live=F[d >= s["test_end"]],
    )


# ----------------------------------------------------------------------------------------------
# decision tree with honest leaf statistics
# ----------------------------------------------------------------------------------------------
class TreeModel:
    def __init__(self, max_depth=6, min_samples_leaf=150, seed=C.SEED):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight="balanced",
                                          random_state=seed, ccp_alpha=2e-5)
        self.features = MODEL_FEATURES

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.clf.fit(X[self.features], y)
        leaves = self.clf.apply(X[self.features])
        tab = pd.DataFrame({"leaf": leaves, "y": y}).groupby("leaf").y.agg(["size", "sum"])
        tab.columns = ["n_train", "n_fraud"]
        p0 = float(y.mean())
        tab["rate"] = (tab.n_fraud + 2 * p0) / (tab.n_train + 2)      # Laplace-style smoothing to the base rate
        self.leaf_table = tab
        self.base_rate = p0
        self.n_leaves = int(self.clf.get_n_leaves())
        self.rules_text = export_text(self.clf, feature_names=self.features, max_depth=6, decimals=3, show_weights=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        leaves = self.clf.apply(X[self.features])
        return self.leaf_table.rate.reindex(leaves).fillna(self.base_rate).to_numpy()

    def explain(self, x: pd.Series) -> dict:
        """Decision path as human-readable rules plus the leaf's training statistics."""
        tree = self.clf.tree_
        xv = x[self.features].to_numpy(dtype=float).reshape(1, -1)
        node_ind = self.clf.decision_path(xv)
        nodes = node_ind.indices
        steps = []
        for nid in nodes:
            if tree.children_left[nid] == tree.children_right[nid]:
                leaf = nid
                break
            f = self.features[tree.feature[nid]]
            thr = float(tree.threshold[nid])
            val = float(xv[0, tree.feature[nid]])
            go_left = (val <= thr) if not np.isnan(val) else bool(tree.missing_go_to_left[nid])
            steps.append(dict(feature=f, threshold=round(thr, 4), value=None if np.isnan(val) else round(val, 4),
                              rule=f"{f} {'<=' if go_left else '>'} {thr:.4g}" + (" (missing)" if np.isnan(val) else "")))
        stats = self.leaf_table.loc[leaf] if leaf in self.leaf_table.index else None
        return dict(leaf=int(leaf), steps=steps,
                    n_train=int(stats.n_train) if stats is not None else 0, n_fraud=int(stats.n_fraud) if stats is not None else 0,
                    rate=float(stats.rate) if stats is not None else self.base_rate)


# ----------------------------------------------------------------------------------------------
# XGBoost with isotonic calibration and native SHAP contributions
# ----------------------------------------------------------------------------------------------
class XGBModel:
    def __init__(self, seed=C.SEED):
        self.features = MODEL_FEATURES
        self.params = dict(n_estimators=700, learning_rate=0.04, max_depth=5, subsample=0.8, colsample_bytree=0.7,
                           min_child_weight=6, reg_lambda=2.0, gamma=0.1, tree_method="hist", eval_metric=["aucpr", "logloss"],
                           early_stopping_rounds=100, random_state=seed, n_jobs=4)

    def fit(self, Xtr, ytr, Xva, yva):
        self.params["scale_pos_weight"] = 1.0        # unweighted objective calibrates better; validated vs sqrt(neg/pos)
        self.clf = xgb.XGBClassifier(**self.params)
        self.clf.fit(Xtr[self.features], ytr, eval_set=[(Xva[self.features], yva)], verbose=False)
        self.best_iteration = int(self.clf.best_iteration)
        raw_va = self.predict_raw(Xva)
        self.calibrator = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(raw_va, yva)
        ev = self.clf.evals_result()["validation_0"]
        self.eval_curve = ev["aucpr"]
        self.eval_logloss = ev["logloss"]
        imp = self.clf.get_booster().get_score(importance_type="gain")
        self.gain_importance = pd.Series({f: imp.get(f, 0.0) for f in self.features}).sort_values(ascending=False)
        return self

    def predict_raw(self, X) -> np.ndarray:
        return self.clf.predict_proba(X[self.features], iteration_range=(0, self.best_iteration + 1))[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(X))

    def contributions(self, X) -> pd.DataFrame:
        """Exact SHAP values (log-odds) from XGBoost's tree path algorithm; last column is the bias."""
        dm = xgb.DMatrix(X[self.features], feature_names=self.features)
        contrib = self.clf.get_booster().predict(dm, pred_contribs=True, iteration_range=(0, self.best_iteration + 1))
        return pd.DataFrame(contrib, columns=self.features + ["bias"], index=X.index)


# ----------------------------------------------------------------------------------------------
# logistic scorecard
# ----------------------------------------------------------------------------------------------
class ScorecardModel:
    def __init__(self, seed=C.SEED):
        self.features = MODEL_FEATURES
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=0.2, max_iter=3000, random_state=seed)),
        ])

    def fit(self, X, y):
        self.pipe.fit(X[self.features], y)
        imp = self.pipe.named_steps["impute"]
        ind = [self.features[i] + "__missing" for i in imp.indicator_.features_] if imp.indicator_ is not None else []
        self.columns = self.features + ind
        self.coef = pd.Series(self.pipe.named_steps["lr"].coef_[0], index=self.columns)
        self.intercept = float(self.pipe.named_steps["lr"].intercept_[0])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X[self.features])[:, 1]

    def contributions(self, X) -> pd.DataFrame:
        Z = self.pipe.named_steps["scale"].transform(self.pipe.named_steps["impute"].transform(X[self.features]))
        return pd.DataFrame(Z * self.coef.to_numpy()[None, :], columns=self.columns, index=X.index)


# ----------------------------------------------------------------------------------------------
# stacker: blends model log-odds + Bayesian priors, fitted on the validation window
# ----------------------------------------------------------------------------------------------
STACK_INPUTS = ["xgb", "tree", "scorecard", "bayes_vendor", "bayes_person"]
STACK_LABELS = {"xgb": "XGBoost (patterns)", "tree": "Decision tree (rules)", "scorecard": "Logistic scorecard (points)",
                "bayes_vendor": "Bayesian merchant prior (MCMC)", "bayes_person": "Bayesian customer prior (MCMC)"}


class Stacker:
    def fit(self, L: pd.DataFrame, y):
        self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(L[STACK_INPUTS], y)
        self.weights = pd.Series(self.lr.coef_[0], index=STACK_INPUTS)
        self.intercept = float(self.lr.intercept_[0])
        self.means = L[STACK_INPUTS].mean()
        return self

    def predict_proba(self, L: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(L[STACK_INPUTS])[:, 1]

    def decompose(self, l: pd.Series) -> dict:
        contrib = {k: float(self.weights[k] * l[k]) for k in STACK_INPUTS}
        z = self.intercept + sum(contrib.values())
        return dict(intercept=self.intercept, contributions=contrib, logit=float(z), probability=float(expit(z)))


# ----------------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------------
def tier_for(p: float, policy: dict | None = None) -> tuple[str, str, str]:
    policy = policy or C.DEFAULT_POLICY
    if p < policy["approve_max"]:
        return C.TIERS[0]
    if p < policy["monitor_max"]:
        return C.TIERS[1]
    if p < policy["stepup_max"]:
        return C.TIERS[2]
    return C.TIERS[3]


def tiers_vec(p: np.ndarray, policy: dict | None = None) -> np.ndarray:
    policy = policy or C.DEFAULT_POLICY
    return np.select([p < policy["approve_max"], p < policy["monitor_max"], p < policy["stepup_max"]],
                     [C.TIERS[0][0], C.TIERS[1][0], C.TIERS[2][0]], default=C.TIERS[3][0])


# ----------------------------------------------------------------------------------------------
# metrics & curves
# ----------------------------------------------------------------------------------------------
def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def recall_at_alert_rate(y, p, amounts=None, rate=0.01):
    n_alert = max(int(np.ceil(len(p) * rate)), 1)
    order = np.argsort(-p)[:n_alert]
    caught = y[order].sum()
    out = dict(alert_rate=rate, n_alert=int(n_alert), recall=float(caught / max(y.sum(), 1)), precision=float(caught / n_alert))
    if amounts is not None:
        a = np.abs(amounts)
        out["value_recall"] = float(a[order][y[order] == 1].sum() / max(a[y == 1].sum(), 1e-9))
    return out


def metrics(y, p, amounts=None) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    m = dict(n=int(len(y)), n_fraud=int(y.sum()), auc_roc=float(roc_auc_score(y, p)), auc_pr=float(average_precision_score(y, p)),
             brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, np.clip(p, EPS, 1 - EPS))), ks=ks_stat(y, p))
    for r in (0.005, 0.01, 0.02, 0.05):
        d = recall_at_alert_rate(y, p, amounts, r)
        m[f"recall@{r:.1%}"] = d["recall"]
        m[f"precision@{r:.1%}"] = d["precision"]
        if amounts is not None:
            m[f"value_recall@{r:.1%}"] = d["value_recall"]
    return m


def curves(y, p, n_bins=10) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    fpr, tpr, thr = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    # thin the curves for storage
    def thin(a, k=400):
        idx = np.unique(np.linspace(0, len(a) - 1, min(k, len(a))).astype(int))
        return a[idx]
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-9
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
    cal = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    # alert-rate curve
    order = np.argsort(-p)
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    rates = np.linspace(0.001, 0.1, 60)
    idx = np.clip((rates * len(p)).astype(int) - 1, 0, len(p) - 1)
    return dict(roc=dict(fpr=thin(fpr).tolist(), tpr=thin(tpr).tolist()), pr=dict(recall=thin(rec).tolist(), precision=thin(prec).tolist()),
                calibration=cal.to_dict("list"), alert=dict(rate=rates.tolist(), recall=cum[idx].tolist()))


def confusion_at(y, p, threshold):
    y = np.asarray(y); pred = np.asarray(p) >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), alert_rate=(tp + fp) / len(y))
