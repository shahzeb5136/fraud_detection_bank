"""Audit log: every decision made in this session, the model registry and reproducibility information."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from fraudlab import config as C, ui
from fraudlab.scoring import trace_to_json


def render():
    store = ui.get_store()
    reg = store.registry
    log = ui.audit_log()
    ui.page_header("Audit log", "Every score produced in this session is retained with its full trace: inputs, features, each model's output, blend weights, policy and timings.")

    ui.kpi_row([dict(label="Decisions this session", value=f"{len(log)}", sub="manual scores (stream replays are re-computed on demand)"),
                dict(label="Model version", value=reg["model_version"], sub=f"built {reg['built_at'][:19].replace('T', ' ')} UTC"),
                dict(label="Seed", value=f"{reg['seed']}", sub="synthetic world, MCMC chains and models are fully reproducible"),
                dict(label="Median latency", value=f"{pd.Series([t['timings_ms']['total_ms'] for t in log]).median():.0f} ms" if log else "–", sub="features + 3 models + 2 posteriors + blend")])

    ui.section("Decision log", "")
    if not log:
        st.info("No manual decisions yet. Score a transaction on the Live scoring page and it will appear here with its trace.")
    else:
        rows = []
        for t in log:
            i, d = t["input"], t["decision"]
            rows.append(dict(trace_id=t["trace_id"], scored_at=t["scored_at"][:19].replace("T", " "), scenario=t.get("scenario", ""), customer=i["person_id"], merchant=i.get("vendor_name", i["vendor_id"]),
                             amount=i["transaction_amt"], channel=i["channel"], probability=d["probability"], decision=ui.TIER_LABEL[d["tier"]],
                             xgb=t["components"]["xgb"]["prob"], tree=t["components"]["tree"]["prob"], scorecard=t["components"]["scorecard"]["prob"],
                             merchant_prior=t["components"]["bayes_vendor"]["prob"], customer_prior=t["components"]["bayes_person"]["prob"], model=t["model_version"], ms=t["timings_ms"]["total_ms"]))
        df = pd.DataFrame(rows)[::-1]
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1), "amount": st.column_config.NumberColumn(format="£%.2f"),
                                    **{k: st.column_config.NumberColumn(format="%.3f") for k in ["xgb", "tree", "scorecard", "merchant_prior", "customer_prior"]}})
        c1, c2 = st.columns([1, 3])
        with c1:
            st.download_button("Download full session log (JSON)", data=json.dumps([json.loads(trace_to_json(t)) for t in log], indent=2), file_name="fraudlens_session_audit.json", mime="application/json")
            if st.button("Clear session log"):
                log.clear(); st.rerun()
        with c2:
            sel = st.selectbox("Inspect a trace", [t["trace_id"] for t in log][::-1], key="audit_sel")
            tr = next(t for t in log if t["trace_id"] == sel)
            st.json({k: v for k, v in tr.items() if k != "feature_lineage"}, expanded=False)

    ui.section("Model registry", "What was built, from what, with which libraries - the provenance a validation team needs.")
    a, b = st.columns(2)
    with a:
        st.dataframe(pd.DataFrame({"artifact": list(reg["hashes"]), "sha256 (prefix)": list(reg["hashes"].values())}), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame({"library": list(reg["libraries"]), "version": list(reg["libraries"].values())}), width="stretch", hide_index=True)
    with b:
        st.markdown(f"""
- **As-of date** {reg['as_of_date']} · **history** {reg['history_days']} days · **label maturity** {reg['label_maturity_days']} days
- **Split (days from start)** train < {reg['split']['train_end']} · validation < {reg['split']['valid_end']} · test < {reg['split']['test_end']} · live thereafter
- **Default policy** approve < {reg['policy']['approve_max']:.0%} · monitor < {reg['policy']['monitor_max']:.0%} · step-up < {reg['policy']['stepup_max']:.0%} · decline otherwise
- **Features** {reg['n_features']} · **build time** {reg['build_seconds']} s
- **World** {reg['world']['n_legit_rows_generated']:,} genuine rows generated · {reg['world']['n_episodes']} fraud episodes · {reg['world']['n_victims']} victims

**Reproduce:** `python -m fraudlab.build` regenerates the world, features, MCMC posteriors and models from seed {reg['seed']}; artifact hashes above will match.
""")
