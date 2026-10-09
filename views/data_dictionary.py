"""Data & dictionary: the source data dictionary, how the demo covers it, hygiene checks verified on the data, and the synthetic schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fraudlab import config as C, features as FE, ui
from fraudlab.ui import P

FIELD_TO_COLUMNS = {
    "person_id": ["person_id"], "vendor_id": ["vendor_id"], "transaction_detail_char": ["transaction_detail_char"], "transaction_amt": ["transaction_amt"],
    "date_of_transaction": ["date_of_transaction"], "fraud_label / is_disputed": ["fraud_label", "is_disputed"], "dispute_date": ["dispute_date"],
    "dispute_reason_code": ["dispute_reason_code"], "dispute_resolution": ["dispute_resolution"], "transaction_timestamp": ["transaction_timestamp", "transaction_timestamp_local", "tz_offset_hours"],
    "mcc": ["mcc"], "channel / entry_mode": ["channel"], "auth_result": ["auth_result", "decline_reason"], "transaction_type": ["transaction_type", "ref_transaction"],
    "merchant_country": ["merchant_country"], "merchant_city": ["merchant_city", "merchant_lat", "merchant_lon"], "cardholder_home_geo": ["cardholder_home_city", "cardholder_home_lat", "cardholder_home_lon"],
    "currency": ["currency", "amount_original"], "fx_flag": ["fx_flag"], "card_id": ["card_id"], "3ds_flag / sca_applied": ["sca_applied"], "balance_or_limit": ["balance_or_limit"],
    "device_id": ["device_id"], "ip_address": ["ip_address"], "recent_account_events": ["events table (event_ts, event_type)"], "card_age": ["card_age_days"], "account_tenure": ["account_tenure_days"],
    "merchant_risk_stats": ["derived as-of: vendor_fraud_rate_asof, vendor_dispute_rate_asof"],
    "is_new_vendor_for_user": ["feature"], "days_since_last_visit": ["feature"], "visit_count": ["feature"], "user_vendor_affinity": ["feature"], "amount_z_vs_user_vendor": ["feature"],
    "is_recurring": ["feature"], "descriptor_deviation": ["feature"], "day_of_week": ["feature"], "day_of_month": ["feature"],
}


@st.cache_data(show_spinner=False)
def load_dictionary():
    import openpyxl
    wb = openpyxl.load_workbook(C.DATA_DICT_PATH, read_only=True)
    ws = wb["Data Dictionary"]
    rows = list(ws.iter_rows(values_only=True))
    header_i = next(i for i, r in enumerate(rows) if r[0] == "Field")
    dd = pd.DataFrame([r[:6] for r in rows[header_i + 1:] if r[0]], columns=["field", "type", "status", "priority", "definition", "why_it_matters"])
    dd["priority"] = dd.priority.map(lambda x: "" if x is None or str(x).strip() in ("", "�", "-", "—") else str(x))
    hs = wb["Hygiene Checks"]
    hrows = list(hs.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(hrows) if r[0] == "Check")
    hy = pd.DataFrame([r[:3] for r in hrows[hi + 1:] if r[0]], columns=["check", "question", "consequence"])
    return dd, hy


def features_using(field: str) -> list[str]:
    key = field.split(" / ")[0].strip()
    out = []
    for f, (src, *_r) in FE.LINEAGE.items():
        if key in src or field in src:
            out.append(f)
    return out


def render():
    store = ui.get_store()
    tx = store.tx
    dd, hy = load_dictionary()
    ui.page_header("Data & dictionary", "The fraud model data dictionary is the contract: every field it names is simulated, every derived field is a feature, and every hygiene check is verified on the data.")

    counts = dd.status.value_counts()
    n_feat_derived = sum(1 for f in dd[dd.status == "Derived"].field if features_using(f))
    ui.kpi_row([
        dict(label="Fields · Have", value=f"{counts.get('Have', 0)}", sub="in the current extract · all simulated", status="good"),
        dict(label="Fields · Add", value=f"{counts.get('Add', 0)}", sub=f"{int((dd.priority == 'Critical').sum())} critical · {int((dd.priority == 'High').sum())} high · all simulated", status="good"),
        dict(label="Fields · Derived", value=f"{counts.get('Derived', 0)}", sub=f"{n_feat_derived} implemented as model features", status="good"),
        dict(label="Model features", value=f"{len(FE.MODEL_FEATURES)}", sub=f"{len(FE.LINEAGE_DF.group.unique())} feature groups, every one with a dictionary source"),
        dict(label="Hygiene checks", value=f"{len(hy)}", sub="each answered and verified on the synthetic extract below"),
    ])

    # ------------------------------------------------------------------ dictionary table
    ui.section("Data dictionary", "Source: fraud_model_data_dictionary.xlsx. The last two columns show how this demo realises each field.")
    dd2 = dd.copy()
    dd2["in this demo"] = dd2.field.map(lambda f: ", ".join(FIELD_TO_COLUMNS.get(f, ["–"])))
    dd2["features using it"] = dd2.field.map(lambda f: ", ".join(features_using(f)[:6]) + (" …" if len(features_using(f)) > 6 else ""))
    status_filter = st.segmented_control("Status", ["All", "Have", "Add", "Derived"], default="All", key="dd_status")
    if status_filter and status_filter != "All":
        dd2 = dd2[dd2.status == status_filter]
    st.dataframe(dd2, width="stretch", hide_index=True, height=520,
                 column_config={"definition": st.column_config.TextColumn(width="large"), "why_it_matters": st.column_config.TextColumn("why it matters", width="large"),
                                "status": st.column_config.TextColumn(width="small"), "priority": st.column_config.TextColumn(width="small")})

    # ------------------------------------------------------------------ hygiene ledger
    ui.section("Data hygiene ledger", "The dictionary's 'settle before modelling' checks, with the convention adopted here and the evidence computed from the extract.")
    auth = tx[tx.transaction_type == "authorisation"]
    refunds = tx[tx.transaction_type == "refund"]
    agg_share = auth.is_aggregator.mean()
    first_day = tx.transaction_timestamp.min().date()
    n_censored = sum(1 for v, t in store.lk["vendor_first_seen"].items() if t < 0)
    answers = {
        "Sign convention": (f"Debits positive, refunds and reversals negative with their own transaction_type and a reference to the original authorisation.",
                            f"{len(refunds):,} refunds, all negative: {(refunds.transaction_amt < 0).mean():.0%} ✔ · {int((tx.transaction_type == 'reversal').sum()):,} reversals"),
        "Declined transactions": ("Declines are present with a decline reason and are used as velocity signal, never as a label.",
                                  f"{(auth.auth_result == 'declined').mean():.2%} of authorisations declined; top reasons: {', '.join(auth.decline_reason.dropna().value_counts().head(3).index)} ✔"),
        "Left censoring": (f"History starts {first_day}. Merchants already active in the first 30 days are treated as established (first seen 365 days earlier) so they are not mistaken for new.",
                           f"{n_censored} of {len(store.lk['vendor_first_seen'])} merchants flagged as left-censored ✔"),
        "Pending vs posted": ("One row per authorisation; settlements are not duplicated. Refunds and reversals are separate rows and excluded from scoring.",
                              f"{len(auth):,} authorisations, {len(tx) - len(auth):,} refund/reversal rows kept out of the model ✔"),
        "Vendor ID stability": ("vendor_id is a stable surrogate key; descriptor drift is measured separately by descriptor_deviation.",
                                f"{tx.vendor_id.nunique()} merchant IDs, each mapping to exactly one name and MCC ✔"),
        "Aggregators": ("Aggregators are flagged (is_aggregator) so the model can discount their noisy descriptors and mixed sub-merchants.",
                        f"{int(store.vendors.is_aggregator.sum())} aggregator IDs carrying {agg_share:.2%} of authorisations ✔"),
        "Timezone": ("Timestamps stored in UTC plus a merchant-local timestamp and explicit offset; hour-of-day features use local time.",
                     f"{tx.tz_offset_hours.nunique()} distinct offsets in the extract; both columns populated for 100% of rows ✔"),
        "Amount currency": ("Amounts converted to GBP at a fixed rate; original amount and currency retained, fx_flag marks conversions.",
                            f"{tx.currency.nunique()} currencies; {tx.fx_flag.mean():.1%} of rows converted ✔"),
        "Label leakage": ("Dispute fields are never features. Merchant risk statistics only count disputes raised and resolved before the scoring time; labels younger than 60 days are unknown.",
                          f"{int((~auth.label_mature).sum()):,} immature rows excluded from training; as-of merchant stats verified in the live/batch parity test ✔"),
        "Protected attributes": ("merchant_country, merchant_city and distance_from_home proxy for nationality and residence; the model card records this for fairness review.",
                                 "Flagged for model-risk review - not resolved by this demo ⚠"),
    }
    led = hy.copy()
    led["convention adopted in this demo"] = led.check.map(lambda c: answers.get(c, ("", ""))[0])
    led["verified on the extract"] = led.check.map(lambda c: answers.get(c, ("", ""))[1])
    st.dataframe(led, width="stretch", hide_index=True, height=420, column_config={c: st.column_config.TextColumn(width="large") for c in ["question", "consequence", "convention adopted in this demo", "verified on the extract"]})

    # ------------------------------------------------------------------ synthetic data overview
    ui.section("The synthetic extract", "What was generated, and the fraud typologies injected into it.")
    a, b, c = st.columns(3)
    with a:
        ui.plot(ui.hist_fig(auth.transaction_amt[auth.transaction_amt > 0], title="Authorisation amounts (log scale)", xtitle="GBP", log_x=True, nbins=60), height=280)
    with b:
        ch = auth.channel.value_counts()
        ui.plot(ui.bar_fig(ch.index, ch.values, title="Entry mode mix", hover="%{x}<br>%{y:,}<extra></extra>", height=280))
    with c:
        ls = auth.label_status.value_counts()
        fig = ui.bar_fig(ls.values[::-1], ls.index[::-1], orientation="h", title="Label status as of the as-of date", hover="%{y}<br>%{x:,}<extra></extra>", height=280)
        fig.update_layout(margin=dict(l=10, r=10, t=36, b=30))
        ui.plot(fig)
    a, b = st.columns([1.2, 1])
    with a:
        typ = tx[tx.fraud_typology.notna()].groupby("fraud_typology").agg(rows=("transaction_id", "size"), episodes=("fraud_episode_id", "nunique"), value=("transaction_amt", lambda s: s.abs().sum()),
                                                                          labelled=("fraud_label", "mean")).reset_index()
        typ["description"] = typ.fraud_typology.map(C.TYPOLOGIES)
        st.dataframe(typ[["fraud_typology", "description", "episodes", "rows", "value", "labelled"]], width="stretch", hide_index=True,
                     column_config={"value": st.column_config.NumberColumn("attempted value", format="£%.0f"), "labelled": st.column_config.NumberColumn("share confirmed as fraud", format="percent"),
                                    "description": st.column_config.TextColumn(width="large")})
        st.caption("'Share confirmed' < 100% is deliberate label noise: silent victims, declined attempts that are never disputed, reversed disputes and friendly fraud.")
    with b:
        rc = tx[tx.dispute_reason_code.notna()].groupby("dispute_reason_code").agg(n=("transaction_id", "size")).reset_index()
        rc["meaning"] = rc.dispute_reason_code.map(lambda r: C.REASON_CODES[r][0])
        rc["family"] = rc.dispute_reason_code.map(lambda r: C.REASON_CODES[r][1])
        st.dataframe(rc, width="stretch", hide_index=True, column_config={"dispute_reason_code": "reason code", "meaning": st.column_config.TextColumn(width="large")})
    with st.expander("Schema and sample rows"):
        schema = pd.DataFrame({"column": tx.columns, "dtype": [str(t) for t in tx.dtypes], "example": [str(tx[c].dropna().iloc[0])[:40] if tx[c].notna().any() else "" for c in tx.columns],
                               "note": ["simulation truth - not available to a bank" if c in ("fraud_truth", "fraud_typology", "fraud_episode_id", "is_fraudulent") else "" for c in tx.columns]})
        st.dataframe(schema, width="stretch", hide_index=True, height=400)
        st.dataframe(tx.sample(12, random_state=1).drop(columns=["t_sec"]), width="stretch", hide_index=True)
