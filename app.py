"""FraudLens - transaction fraud detection demo (Streamlit entry point).

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from fraudlab import config as C, ui

st.set_page_config(page_title=f"{C.APP_NAME} · {C.BANK_NAME}", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
ui.setup_plotly()
ui.inject_css()
ui.sidebar_logo()

st.sidebar.html(f'<div class="fl-brand">🛡️ {C.APP_NAME}</div><div class="fl-brand-sub">{C.BANK_NAME} · Transaction fraud detection</div>')

if not ui.artifacts_ready():
    ui.page_header("First-time setup", "The synthetic world, features, MCMC posteriors and models have not been built yet.")
    st.info("Building takes about 90 seconds: it simulates 18 months of card transactions for 1,500 customers, engineers "
            "leakage-safe features, runs the MCMC samplers and trains the decision tree, XGBoost and scorecard models.")
    if st.button("Build the demo now", type="primary"):
        from fraudlab import build
        log_box = st.empty()
        lines = []

        def _log(msg):
            lines.append(msg)
            log_box.code("\n".join(lines[-14:]))
        with st.spinner("Building…"):
            import builtins
            orig = builtins.print
            build._log = lambda m: _log(m)
            build.main()
            builtins.print = orig
        st.success("Build complete. Reloading…")
        st.rerun()
    st.stop()

from views import audit, bayes_lab, customer, data_dictionary, live, models_lab, overview, portfolio  # noqa: E402

pages = {
    "Overview": [st.Page(overview.render, title="Command centre", icon=":material/dashboard:", url_path="overview", default=True)],
    "Detection": [
        st.Page(live.render, title="Live scoring", icon=":material/bolt:", url_path="live"),
        st.Page(customer.render, title="Customer 360", icon=":material/person_search:", url_path="customer"),
    ],
    "Analytics": [
        st.Page(portfolio.render, title="Portfolio analytics", icon=":material/monitoring:", url_path="portfolio"),
        st.Page(models_lab.render, title="Model lab", icon=":material/science:", url_path="models"),
        st.Page(bayes_lab.render, title="Bayesian & MCMC", icon=":material/functions:", url_path="bayes"),
    ],
    "Governance": [
        st.Page(data_dictionary.render, title="Data & dictionary", icon=":material/menu_book:", url_path="data"),
        st.Page(audit.render, title="Audit log", icon=":material/verified:", url_path="audit"),
    ],
}
st.session_state["pages"] = {pg.url_path: pg for group in pages.values() for pg in group}
nav = st.navigation(pages, position="sidebar")
store = ui.get_store()
ui.sidebar_footer(store)
nav.run()
