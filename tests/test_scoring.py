"""Regression tests on the shipped artifacts.  Run:  python -m pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlab import config as C, features as FE, models as M
from fraudlab.scoring import Store, score_transaction, transaction_from_row


@pytest.fixture(scope="module")
def store():
    return Store()


def test_live_rescore_reproduces_batch_features_and_score(store):
    """Replaying a stored authorisation through the live pipeline must give the features and probability the batch
    build produced - including the monthly user-vendor affinity refit, which must not see later data."""
    F = store.features.set_index("transaction_id")
    sc = store.scores.set_index("transaction_id")
    live = store.scores[store.scores.split.isin(["test", "live"])]
    ids = pd.concat([live[live.tier != "APPROVE"].transaction_id.sample(15, random_state=1),
                     live.transaction_id.sample(15, random_state=2)])
    for tid in ids:
        row = store.auth[store.auth.transaction_id == tid].iloc[0]
        trace = score_transaction(store, transaction_from_row(store, row), C.DEFAULT_POLICY)
        for f in FE.MODEL_FEATURES:
            got, want = trace["features"][f], F.loc[tid, f]
            assert (got is None) == pd.isna(want), (tid, f)
            if got is not None:
                assert np.isclose(got, want, rtol=1e-6, atol=1e-8), (tid, f, got, want)
        assert abs(trace["decision"]["probability"] - sc.loc[tid, "p_blend"]) < 1e-6, tid


def test_alerts_group_into_cases(store):
    """'Next alert' walks cases: follow-on alerts on the same customer within the case window join the open case."""
    from views.live import CASE_WINDOW_H, _cases, _live_window
    lw = _live_window(store, store.registry["model_version"])
    cs = _cases(lw, M.tiers_vec(lw.p_blend.to_numpy(), C.DEFAULT_POLICY))
    assert cs.case.notna().equals(cs.alert)
    assert cs.case_open.sum() == cs.case.nunique() < cs.alert.sum()
    for case, idx in cs[cs.alert].groupby("case").groups.items():
        g = lw.loc[idx]
        assert g.person_id.nunique() == 1, case
        assert (g.t_sec.diff().dropna() <= CASE_WINDOW_H * 3600).all(), case
        assert cs.case_open[idx].sum() == 1 and cs.case_open[idx[0]], case
