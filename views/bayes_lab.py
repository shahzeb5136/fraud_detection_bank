"""Bayesian & MCMC: model specification, chain diagnostics, posteriors, shrinkage, cold-start predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import expit

from fraudlab import config as C, ui
from fraudlab.ui import P, STATUS


def render():
    store = ui.get_store()
    vm, pm, ev = store.bayes["vendor"], store.bayes["person"], store.evaluation
    ui.page_header("Bayesian & MCMC", "Hierarchical fraud-rate models sampled with hand-written Metropolis-within-Gibbs: uncertainty, shrinkage and cold-start back-off, fully traceable.")

    # ------------------------------------------------------------------ specification
    a, b = st.columns([1.15, 1])
    with a:
        ui.section("Merchant hierarchy: vendor within MCC within population", "The pooling structure the data dictionary asks for under `mcc`: back-off for unseen merchants.")
        st.latex(r"""\begin{aligned}
y_v &\sim \mathrm{Binomial}(n_v,\; \sigma(b_v))\\
b_v &\sim \mathcal{N}(a_{m(v)},\; \sigma_v^2)\\
a_m &\sim \mathcal{N}(\mu,\; \sigma_m^2)\\
\mu &\sim \mathcal{N}(\mathrm{logit}(0.01),\,1.5^2),\quad \sigma_m,\sigma_v \sim \mathrm{HalfNormal}(1)
\end{aligned}""")
        st.caption("y_v = confirmed fraud out of n_v training authorisations at merchant v · b_v = merchant effect pooled within its MCC · a_m = MCC effect pooled within the population.")
        st.markdown("**Sampler.** 4 chains × 7,000 iterations (2,500 warm-up with step-size adaptation). Merchant effects use a vectorised random-walk Metropolis step "
                    "(conditionally independent given the MCC effects); MCC effects and μ are exact Gibbs draws from their Normal conditionals; the two scales use "
                    "adaptive Metropolis on the log scale. No compiled back-end: the sampler is 120 lines of NumPy you can read in `fraudlab/bayes.py`.")
    with b:
        ui.section("Customer propensity: Beta-Binomial", "How likely is *this* customer's next authorisation to be fraud, given their own history?")
        st.latex(r"""\begin{aligned}
y_p &\sim \mathrm{Binomial}(n_p,\; \theta_p), \qquad \theta_p \sim \mathrm{Beta}(\kappa\phi,\; \kappa(1-\phi))\\
\phi &\sim \mathrm{Beta}(2, 200), \qquad \log\kappa \sim \mathcal{N}(\log 50,\; 1)\\
\theta_p \mid y_p, \phi, \kappa &\sim \mathrm{Beta}(\kappa\phi + y_p,\; \kappa(1-\phi) + n_p - y_p)
\end{aligned}""")
        st.markdown("θ is integrated out analytically (Beta-Binomial likelihood); (logit φ, log κ) are sampled by adaptive random-walk Metropolis, then each "
                    "customer's posterior is conjugate. A new customer inherits the population prior; a customer with prior confirmed fraud is shrunk upward in proportion to κ.")

    # ------------------------------------------------------------------ diagnostics
    ui.section("Convergence diagnostics", "Split-R̂ close to 1 and healthy effective sample sizes across all four chains.")
    hy = pd.DataFrame(ev["bayes_vendor_hyper"] + ev["bayes_person_hyper"])
    hy_show = hy[~hy.parameter.str.startswith("a[")].copy()
    acc = ev["bayes_acceptance"]
    acc_v = acc["vendor"]["vendor"] if isinstance(acc["vendor"], dict) else acc["vendor"]
    acc_p = acc["person"]
    ui.kpi_row([dict(label="Max split-R̂", value=f"{hy.rhat.max():.3f}", sub="all parameters incl. 24 MCC effects", status="good" if hy.rhat.max() < 1.05 else "serious"),
                dict(label="Min ESS", value=f"{hy.ess.min():,.0f}", sub="effective draws (Geyer initial positive sequence)", status="good" if hy.ess.min() > 200 else "warning"),
                dict(label="Acceptance · merchant effects", value=f"{acc_v:.2f}", sub="target ≈ 0.44 for scalar random-walk Metropolis"),
                dict(label="Acceptance · customer hyper-parameters", value=f"{acc_p:.2f}", sub="target ≈ 0.23–0.35 for a 2-D block"),
                dict(label="Typical merchant rate σ(μ)", value=ui.fmt_pct(vm.pop_rate, 2), sub="posterior mean; the transaction-weighted rate is higher because fraud concentrates in a few merchants"),
                dict(label="Customer propensity (φ)", value=ui.fmt_pct(pm.pop_mean, 2), sub=f"κ = {float(pm.draws['kappa'].mean()):.1f} pseudo-transactions")])
    st.dataframe(hy_show.set_index("parameter").style.format({"mean": "{:.4f}", "sd": "{:.4f}", "q2_5": "{:.4f}", "q50": "{:.4f}", "q97_5": "{:.4f}", "rhat": "{:.4f}", "ess": "{:,.0f}"}), width="stretch")
    D = vm.draws
    a, b, c = st.columns(3)
    with a:
        ui.plot(ui.trace_fig(D["mu"], "μ (population log-odds)"))
        ui.plot(ui.density_fig(expit(D["mu"].reshape(-1)), "σ(μ)", P["series"][0], title="Posterior of the population fraud rate σ(μ)", xtitle="fraud rate"), height=230)
    with b:
        ui.plot(ui.trace_fig(D["sigma_m"], "σ_mcc"))
        ui.plot(ui.density_fig(D["sigma_m"].reshape(-1), "σ_mcc", P["series"][1], title="Posterior of σ_mcc (spread between categories)", xtitle="σ_mcc"), height=230)
    with c:
        ui.plot(ui.trace_fig(D["sigma_v"], "σ_vendor"))
        ui.plot(ui.density_fig(D["sigma_v"].reshape(-1), "σ_vendor", P["series"][2], title="Posterior of σ_vendor (spread within a category)", xtitle="σ_vendor"), height=230)

    # ------------------------------------------------------------------ MCC forest + shrinkage
    ui.section("What the hierarchy learned", "Category effects with credible intervals, and how individual merchants are pulled towards their category.")
    a, b = st.columns([1, 1.2])
    with a:
        ms = pd.DataFrame(vm.mcc_summary).sort_values("post_mean")
        labels = [f"{m} · {C.MCC_TABLE[m][0][:24]}" for m in ms.mcc]
        fig = go.Figure(go.Scatter(x=ms.post_mean, y=labels, mode="markers", marker=dict(size=9, color=P["series"][6], line=dict(color=P["surface"], width=2)),
                                   error_x=dict(type="data", symmetric=False, array=ms.q97_5 - ms.post_mean, arrayminus=ms.post_mean - ms.q2_5, color=P["series"][6], thickness=1.5, width=0),
                                   text=[f"{int(n)} merchants · {int(y)} fraud / {int(nn):,}" for n, y, nn in zip(ms.n_vendors, ms.y, ms.n)], hovertemplate="%{y}<br>%{x:.2%} %{text}<extra></extra>", name="MCC"))
        fig.add_vline(x=vm.pop_rate, line=dict(color=P["ink"], width=1, dash="dot"), annotation_text="population", annotation_font=dict(size=10))
        fig.update_layout(title="MCC-level fraud rate σ(a_m) with 95% credible intervals", xaxis=dict(tickformat=".1%"), height=560, margin=dict(l=10, r=10, t=36, b=40), yaxis=dict(tickfont=dict(size=10.5)))
        ui.plot(fig)
    with b:
        vs = pd.DataFrame(vm.vendor_summary)
        vs["merchant"] = vs.vendor_id.map(store.vendors.vendor_name)
        vs["lo_err"] = vs.post_mean - vs.q2_5
        vs["hi_err"] = vs.q97_5 - vs.post_mean
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vs.n, y=vs.raw_rate, mode="markers", name="raw rate y/n", marker=dict(size=6, color=P["muted"], opacity=0.5), text=vs.merchant, hovertemplate="%{text}<br>n=%{x} raw %{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=vs.n, y=vs.post_mean, mode="markers", name="posterior mean", marker=dict(size=7, color=P["series"][6], opacity=0.85, line=dict(color=P["surface"], width=1)),
                                 error_y=dict(type="data", symmetric=False, array=vs.hi_err, arrayminus=vs.lo_err, color=P["series"][6] + "55", thickness=1, width=0),
                                 text=vs.merchant, hovertemplate="%{text}<br>n=%{x} posterior %{y:.2%}<extra></extra>"))
        fig.add_hline(y=vm.pop_rate, line=dict(color=P["ink"], width=1, dash="dot"))
        fig.update_layout(title="Shrinkage by evidence: with few transactions the posterior stays near the category; with many it follows the data", xaxis=dict(type="log", title="training authorisations at merchant (log scale)"),
                          yaxis=dict(tickformat=".0%", title="fraud rate", range=[0, min(1.0, float(vs.raw_rate.max()) * 1.1)]), height=560)
        ui.plot(fig)

    # ------------------------------------------------------------------ merchant lookup & cold start
    ui.section("Merchant posterior and cold start", "Pick any merchant to see its full posterior; pick an MCC to see what the model believes about a merchant it has never seen.")
    a, b = st.columns(2)
    with a:
        opts = vs.sort_values("post_mean", ascending=False).vendor_id.tolist()
        vid = st.selectbox("Merchant", opts, format_func=lambda v: f"{store.vendors.loc[v].vendor_name} · MCC {store.vendors.loc[v].mcc} · {store.vendors.loc[v].merchant_city}", key="bl_vendor")
        info = vm.vendor_rate(vid)
        draws = vm.vendor_draws(vid)
        mcc = store.vendors.loc[vid].mcc
        mdraws = expit(D["a"][:, :, vm.mcc_index[mcc]].reshape(-1))
        fig = ui.density_fig(draws, "merchant", P["series"][6], title=f"{store.vendors.loc[vid].vendor_name}: posterior fraud rate", ref=vm.pop_rate, ref_label="population",
                             extra=[dict(draws=np.random.default_rng(0).choice(mdraws, min(len(mdraws), len(draws))), name=f"its MCC ({mcc})", color=P["series"][3])])
        ui.plot(fig)
        st.markdown(f"{info['y']} confirmed fraud in {info['n']:,} training authorisations (raw {ui.fmt_pct(info['raw'], 2)}) → posterior mean **{ui.fmt_pct(info['mean'], 2)}**, "
                    f"95% interval {ui.fmt_pct(info['lo'], 2)}–{ui.fmt_pct(info['hi'], 2)}, P(rate > 3× population) = {ui.fmt_pct(info['prob_gt_3x'], 0)}.")
    with b:
        mcc2 = st.selectbox("MCC of an unseen merchant", list(C.MCC_TABLE), format_func=lambda m: f"{m} · {C.MCC_TABLE[m][0]}", key="bl_mcc")
        nv = vm.new_vendor_rate(mcc2)
        rng = np.random.default_rng(3)
        sv = D["sigma_v"].reshape(-1)
        aa = D["a"][:, :, vm.mcc_index[mcc2]].reshape(-1)
        pred = expit(aa + sv * rng.normal(size=sv.shape))
        fig = ui.density_fig(pred, "unseen merchant in this MCC", P["series"][3], title=f"Predictive fraud rate for a brand-new merchant in MCC {mcc2}", ref=vm.pop_rate, ref_label="population")
        ui.plot(fig)
        st.markdown(f"Posterior predictive mean **{ui.fmt_pct(nv['mean'], 2)}**, 95% interval {ui.fmt_pct(nv['lo'], 2)}–{ui.fmt_pct(nv['hi'], 2)}. "
                    "The interval is wide on purpose: an unseen merchant carries the full within-category spread σ_vendor - this is the dictionary's 'weak cold start' made explicit.")

    # ------------------------------------------------------------------ customer propensity
    ui.section("Customer propensity", "Per-customer posterior after shrinkage. Most customers sit at the population rate; those with prior confirmed fraud move up, but only as far as κ allows.")
    a, b, c = st.columns([1, 1, 1.2])
    with a:
        ui.plot(ui.trace_fig(pm.draws["phi"], "φ"), height=200)
        ui.plot(ui.trace_fig(pm.draws["kappa"], "κ"), height=200)
    with b:
        ps = pd.DataFrame(pm.person_summary)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ps.n, y=ps.raw_rate, mode="markers", name="raw rate", marker=dict(size=5, color=P["muted"], opacity=0.45), text=ps.person_id, hovertemplate="%{text}<br>n=%{x} raw %{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=ps.n, y=ps.post_mean, mode="markers", name="posterior mean", marker=dict(size=6, color=P["series"][4], opacity=0.8), text=ps.person_id, hovertemplate="%{text}<br>n=%{x} posterior %{y:.2%}<extra></extra>"))
        fig.update_layout(title="Customer shrinkage (training window)", xaxis=dict(type="log", title="training authorisations (log)"), yaxis=dict(tickformat=".0%", title="fraud rate", range=[0, 0.4]), height=410)
        ui.plot(fig)
    with c:
        chan = pd.DataFrame(ev["channel_posteriors"]).sort_values("post_mean", ascending=False)
        ui.plot(ui.bar_ci_fig(chan.channel, chan.post_mean, chan.q2_5, chan.q97_5, title="Fraud rate by channel: conjugate Beta(1,1) posteriors (analytic)", color=P["series"][2], height=410))

    # ------------------------------------------------------------------ conjugate amount model
    with st.expander("Also Bayesian: the per-customer amount baseline used by `amount_z_vs_user_vendor`"):
        st.latex(r"""\mu_{u,v}^{post} = \frac{k_0\,\mu_{u,\mathrm{mcc}} + \sum_{i=1}^{n_{u,v}} \log(1+a_i)}{k_0 + n_{u,v}}, \qquad
\sigma_{pred}^2 = \frac{k_0\,\sigma_{\mathrm{mcc}}^2 + n_{u,v}\,s_{u,v}^2}{k_0 + n_{u,v}}\left(1 + \frac{1}{k_0 + n_{u,v}}\right), \qquad
z = \frac{\log(1+a) - \mu_{u,v}^{post}}{\sigma_{pred}}""")
        st.markdown("A conjugate normal update with k₀ = 3 pseudo-visits: the customer-merchant mean is shrunk towards the customer's own mean in that merchant category, "
                    "which is itself shrunk towards the population mean for the category. The predictive standard deviation widens automatically when history is thin, "
                    "so a first visit is judged loosely and a fiftieth visit tightly - the dictionary's observation that the per-vendor distribution is 'far tighter than a global per-user amount distribution', made quantitative.")
