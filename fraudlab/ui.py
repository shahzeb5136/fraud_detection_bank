"""Streamlit UI helpers: theme, CSS, chart factory and shared state."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import config as C
from .scoring import Store

P = C.PALETTE
STATUS = P["status"]
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
STATUS_ICON = {"good": "✔", "warning": "◐", "serious": "▲", "critical": "■"}
TIER_STATUS = {t[0]: t[2] for t in C.TIERS}
TIER_LABEL = {t[0]: t[1] for t in C.TIERS}


# ----------------------------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models, MCMC posteriors and transaction history…")
def get_store() -> Store:
    return Store()


def artifacts_ready() -> bool:
    return all((C.ARTIFACT_DIR / f).exists() for f in ["models.pkl", "bayes.pkl", "lookups.pkl", "scores.parquet", "registry.json"])


def get_policy() -> dict:
    if "policy" not in st.session_state:
        st.session_state["policy"] = dict(C.DEFAULT_POLICY)
    return st.session_state["policy"]


def get_friction() -> float:
    return float(st.session_state.get("pol_f", 8.0))


def tiers_now(p) -> np.ndarray:
    """Policy tier of each probability under the policy currently set in the sidebar (the stored `tier` column
    reflects the default policy at build time)."""
    from .models import tiers_vec
    return tiers_vec(np.asarray(p, dtype=float), get_policy())


ALERT_TIERS = ("STEP_UP", "DECLINE")
_POLICY_KEYS = {"pol_a": "approve_max", "pol_m": "monitor_max", "pol_s": "stepup_max"}


def _apply_policy():
    ss = st.session_state
    get_policy().update(approve_max=ss["pol_a"], monitor_max=ss["pol_m"], stepup_max=ss["pol_s"])


def _on_policy_change(changed: str):
    """Keep approve < monitor < step-up by nudging the neighbouring sliders (callbacks run before the widgets)."""
    ss = st.session_state
    a, m, s = ss["pol_a"], ss["pol_m"], ss["pol_s"]
    if changed == "pol_a":
        m = max(m, round(a + 0.005, 3))
        s = max(s, round(m + 0.01, 2))
    elif changed == "pol_m":
        a = min(a, round(m - 0.005, 3))
        s = max(s, round(m + 0.01, 2))
    else:
        m = min(m, round(s - 0.005, 3))
        a = min(a, round(m - 0.005, 3))
    ss["pol_a"], ss["pol_m"], ss["pol_s"] = a, m, s


def _reset_policy():
    for k, field in _POLICY_KEYS.items():
        st.session_state[k] = C.DEFAULT_POLICY[field]
    st.session_state["pol_f"] = 8.0


def policy_sidebar():
    """App-wide decision policy. It lives in the sidebar of every page because Command centre, Live scoring,
    Customer 360, Portfolio analytics and Model lab all evaluate the same thresholds."""
    pol = get_policy()
    for k, field in _POLICY_KEYS.items():
        st.session_state.setdefault(k, float(pol[field]))
    st.session_state.setdefault("pol_f", 8.0)
    with st.sidebar.expander("Decision policy", icon=":material/tune:"):
        st.slider("Approve below", 0.005, 0.20, step=0.005, format="%.3f", key="pol_a", on_change=_on_policy_change, args=("pol_a",))
        st.slider("Monitor below", 0.01, 0.40, step=0.005, format="%.3f", key="pol_m", on_change=_on_policy_change, args=("pol_m",))
        st.slider("Step-up below (decline at or above)", 0.02, 0.90, step=0.01, format="%.2f", key="pol_s", on_change=_on_policy_change, args=("pol_s",))
        st.number_input("Friction cost of a false decline (£)", 0.0, 100.0, step=1.0, key="pol_f")
        st.button("Reset to default policy", on_click=_reset_policy, width="stretch")
    _apply_policy()


def audit_log() -> list:
    return st.session_state.setdefault("audit", [])


# ----------------------------------------------------------------------------------------------
# customer selection shared by Live scoring and Customer 360
# ----------------------------------------------------------------------------------------------
PERSON_KEY = "person_sel"


def selected_person(store: Store) -> str:
    pid = st.session_state.get(PERSON_KEY)
    if pid not in store.persons.index:
        pid = store.persons.index[0]
        st.session_state[PERSON_KEY] = pid
    return pid


def _pick_random_person(options):
    st.session_state[PERSON_KEY] = str(np.random.default_rng().choice(options))


def customer_picker(store: Store, key: str, label: str = "Customer") -> str:
    """Selectbox bound to the app-wide selected customer.

    A keyed Streamlit widget ignores `index=` once it holds state, so the selection is written into the widget's
    own key before it renders; edits flow back through `on_change`. This is what makes the random buttons work."""
    persons = store.persons
    st.session_state[key] = selected_person(store)
    st.selectbox(label, persons.index.tolist(), key=key,
                 format_func=lambda x: f"{x} · {persons.loc[x].segment.replace('_', ' ')} · {persons.loc[x].home_city}",
                 on_change=lambda: st.session_state.update({PERSON_KEY: st.session_state[key]}))
    return st.session_state[PERSON_KEY]


def random_customer_buttons(store: Store, key: str, compact: bool = False):
    a, b = st.columns(2)
    a.button("Random" if compact else "Random customer", key=f"{key}_rand", icon=":material/shuffle:", width="stretch",
             on_click=_pick_random_person, args=(store.persons.index.to_numpy(),), help="Any customer, picked at random.")
    b.button("Fraud victim" if compact else "Random fraud victim", key=f"{key}_victim", icon=":material/gpp_maybe:", width="stretch",
             on_click=_pick_random_person, args=(store.episodes.person_id.unique(),),
             help="A customer who suffered at least one fraud episode in the simulation.")


# ----------------------------------------------------------------------------------------------
# theme
# ----------------------------------------------------------------------------------------------
def setup_plotly():
    axis = dict(gridcolor=P["grid"], linecolor=P["axis"], zerolinecolor=P["grid"], tickfont=dict(color=P["muted"], size=11),
                title=dict(font=dict(color=P["ink2"], size=12)), showline=False, ticks="")
    pio.templates["fraudlens"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12, color=P["ink2"]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["surface"],
        colorway=P["series"], margin=dict(l=48, r=16, t=36, b=40), xaxis=axis, yaxis=axis, bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=P["axis"], font=dict(family=FONT, color=P["ink"], size=12)),
        title=dict(font=dict(size=13, color=P["ink"]), x=0, xanchor="left"),
    ))
    pio.templates.default = "fraudlens"


CSS = f"""
<style>
:root {{ --ink:{P['ink']}; --ink2:{P['ink2']}; --muted:{P['muted']}; --grid:{P['grid']}; --axis:{P['axis']}; --navy:{P['navy']}; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.fl-title {{ font-size: 26px; font-weight: 650; color: var(--ink); margin: 0; line-height: 1.15; }}
.fl-sub {{ font-size: 13px; color: var(--muted); margin: 4px 0 0 0; }}
.fl-section {{ font-size: 16px; font-weight: 650; color: var(--ink); margin: 18px 0 2px 0; }}
.fl-section-sub {{ font-size: 12.5px; color: var(--muted); margin: 0 0 10px 0; }}
.fl-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 12px 0; }}
.fl-kpi {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 12px 14px 10px 14px; min-height: 84px; }}
.fl-kpi .l {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.fl-kpi .v {{ font-size: 24px; font-weight: 650; color: var(--ink); margin-top: 4px; line-height: 1.1; }}
.fl-kpi .s {{ font-size: 12px; color: var(--ink2); margin-top: 5px; }}
.fl-badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; line-height: 1.4; white-space: nowrap; }}
.fl-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; }}
.fl-chip {{ display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; border: 1px solid rgba(11,11,11,0.12); background: #f3f3f0; color: var(--ink2); margin-right: 4px; }}
.fl-chip.have {{ background: #e3f6ea; color: #0a5a2a; border-color: #b7e4c7; }}
.fl-chip.add {{ background: #e6f0fb; color: #164a86; border-color: #b7d3f6; }}
.fl-chip.derived {{ background: #f3ecfb; color: #4a3aa7; border-color: #d9cdf5; }}
.fl-banner {{ border-radius: 12px; padding: 16px 20px; display: flex; gap: 22px; align-items: center; border: 1px solid rgba(11,11,11,0.10); background: #ffffff; }}
.fl-banner .p {{ font-size: 40px; font-weight: 700; line-height: 1; color: var(--ink); }}
.fl-banner .t {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.fl-banner .a {{ font-size: 15px; font-weight: 650; color: var(--ink); margin-top: 2px; }}
.fl-banner .d {{ font-size: 12.5px; color: var(--ink2); margin-top: 3px; }}
.fl-flow {{ display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap; margin: 6px 0 10px 0; }}
.fl-step {{ flex: 1 1 150px; background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 10px 12px; position: relative; }}
.fl-step .n {{ font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
.fl-step .h {{ font-size: 13.5px; font-weight: 650; color: var(--ink); margin: 2px 0; }}
.fl-step .b {{ font-size: 12px; color: var(--ink2); line-height: 1.35; }}
.fl-step + .fl-step::before {{ content: "›"; position: absolute; left: -9px; top: 38%; color: var(--axis); font-size: 18px; }}
.fl-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; background: #f3f3f0; border: 1px solid rgba(11,11,11,0.08); border-radius: 8px; padding: 10px 12px; color: var(--ink); white-space: pre-wrap; }}
.fl-rule {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12.5px; padding: 6px 10px; border-left: 3px solid {P['series'][1]}; background: #fff7f3; margin: 4px 0; border-radius: 0 6px 6px 0; color: var(--ink); }}
.fl-note {{ font-size: 12.5px; color: var(--ink2); background: #f3f3f0; border-radius: 8px; padding: 10px 12px; border: 1px solid rgba(11,11,11,0.08); }}
.fl-small {{ font-size: 12px; color: var(--muted); }}
.fl-chart-title {{ font-size: 13px; font-weight: 650; color: var(--ink); line-height: 1.3; margin: 6px 0 -6px 0; }}
.fl-txn {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px; padding: 12px 16px; display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px 18px; }}
.fl-txn .l {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.fl-txn .v {{ font-size: 13.5px; font-weight: 600; color: var(--ink); margin-top: 2px; }}
.fl-txn .s {{ font-size: 11.5px; color: var(--ink2); }}
.fl-case {{ border-radius: 10px; padding: 10px 14px; border: 1px solid {P['status']['serious']}55; background: {P['status']['serious']}12; font-size: 13px; color: var(--ink); }}
section[data-testid="stSidebar"] .fl-brand {{ font-size: 20px; font-weight: 700; color: #ffffff; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] .fl-brand-sub {{ font-size: 11.5px; color: #b9c8de; margin-bottom: 8px; }}
section[data-testid="stSidebar"] .fl-side-kv {{ font-size: 11.5px; color: #b9c8de; line-height: 1.5; }}
section[data-testid="stSidebar"] .fl-side-kv b {{ color: #ffffff; font-weight: 600; }}
div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
</style>
"""


def inject_css():
    st.html(CSS)


# ----------------------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------------------
def fmt_gbp(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"£{x:,.{d}f}"


def fmt_pct(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.{d}f}%"


def fmt_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else f"{x}"


def status_hex(status: str) -> str:
    return STATUS.get(status, P["muted"])


# ----------------------------------------------------------------------------------------------
# HTML components
# ----------------------------------------------------------------------------------------------
def page_header(title: str, subtitle: str = ""):
    st.html(f'<p class="fl-title">{title}</p><p class="fl-sub">{subtitle}</p>')


def section(title: str, subtitle: str = ""):
    st.html(f'<p class="fl-section">{title}</p>' + (f'<p class="fl-section-sub">{subtitle}</p>' if subtitle else ""))


def badge(text: str, status: str = "good") -> str:
    col = status_hex(status)
    return (f'<span class="fl-badge" style="background:{col}1f;color:{col};border:1px solid {col}55">'
            f'<span class="fl-dot" style="background:{col}"></span>{STATUS_ICON.get(status, "")} {text}</span>')


def tier_badge(tier: str) -> str:
    return badge(TIER_LABEL.get(tier, tier), TIER_STATUS.get(tier, "good"))


def chip(status: str) -> str:
    return f'<span class="fl-chip {status.lower()}">{status}</span>'


def kpi_html(label: str, value: str, sub: str = "", status: str | None = None) -> str:
    dot = f'<span class="fl-dot" style="background:{status_hex(status)};margin-right:6px"></span>' if status else ""
    return f'<div class="fl-kpi"><div class="l">{label}</div><div class="v">{dot}{value}</div><div class="s">{sub}</div></div>'


def kpi_row(items: list[dict]):
    st.html('<div class="fl-kpis">' + "".join(kpi_html(**i) for i in items) + "</div>")


def note(text: str):
    st.html(f'<div class="fl-note">{text}</div>')


def flow(steps: list[tuple[str, str, str]]):
    st.html('<div class="fl-flow">' + "".join(f'<div class="fl-step"><div class="n">{n}</div><div class="h">{h}</div><div class="b">{b}</div></div>' for n, h, b in steps) + "</div>")


def decision_banner(trace: dict):
    d = trace["decision"]
    col = status_hex(d["status"])
    p = d["probability"]
    econ = "Expected fraud loss exceeds decline friction" if d["economically_decline"] else "Decline friction exceeds expected fraud loss"
    st.html(
        f'<div class="fl-banner" style="border-left:6px solid {col}">'
        f'<div><div class="t">Fraud probability</div><div class="p">{100 * p:.1f}%</div></div>'
        f'<div style="flex:1"><div class="t">Recommended action</div><div class="a">{tier_badge(d["tier"])}</div>'
        f'<div class="d">{econ}: expected loss if approved {fmt_gbp(d["expected_fraud_loss_if_approved"], 2)} vs expected cost if declined {fmt_gbp(d["expected_cost_if_declined"], 2)}.</div></div>'
        f'<div><div class="t">Trace</div><div class="d"><b>{trace["trace_id"]}</b><br>model {trace["model_version"]}<br>{trace["timings_ms"]["total_ms"]} ms end-to-end</div></div>'
        f'</div>')


# ----------------------------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------------------------
def _legend_visible(fig: go.Figure) -> bool:
    if fig.layout.showlegend is not None:
        return bool(fig.layout.showlegend)
    return sum(1 for t in fig.data if t.showlegend is not False) > 1


def plot(fig: go.Figure, height: int | None = None, key: str | None = None):
    """Render a figure. Its title is lifted into HTML above the chart so long titles wrap instead of being clipped
    at the column edge or drawn underneath a top legend."""
    title = fig.layout.title.text
    if title:
        st.html(f'<div class="fl-chart-title">{html.escape(title)}</div>')
        # keep room for a top legend, or for vline annotations / outside bar labels
        fig.update_layout(title_text="", margin=dict(t=40 if _legend_visible(fig) else 28))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=key, config=dict(displayModeBar=False, responsive=True))


def gauge_fig(p: float, policy: dict) -> go.Figure:
    steps = [
        dict(range=[0, 100 * policy["approve_max"]], color=STATUS["good"] + "33"),
        dict(range=[100 * policy["approve_max"], 100 * policy["monitor_max"]], color=STATUS["warning"] + "44"),
        dict(range=[100 * policy["monitor_max"], 100 * policy["stepup_max"]], color=STATUS["serious"] + "44"),
        dict(range=[100 * policy["stepup_max"], 100], color=STATUS["critical"] + "33"),
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number=dict(suffix="%", valueformat=".1f", font=dict(size=30, color=P["ink"], family=FONT)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%", tickfont=dict(size=10, color=P["muted"])),
                   bar=dict(color=P["ink"], thickness=0.22), bgcolor=P["surface"], borderwidth=0, steps=steps,
                   threshold=dict(line=dict(color=P["ink"], width=2), thickness=0.8, value=100 * p)),
    ))
    fig.update_layout(height=210, margin=dict(l=28, r=34, t=20, b=6), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def contrib_fig(items: list[dict], values: dict | None = None, title: str = "", unit: str = "log-odds", top: int = 12) -> go.Figure:
    """Signed horizontal bars: positive pushes towards fraud (red), negative towards genuine (blue)."""
    items = items[:top][::-1]
    names = [i["feature"] for i in items]
    vals = [i["contribution"] for i in items]
    labels = []
    for i in items:
        v = values.get(i["feature"]) if values else None
        labels.append(f"{i['feature']}  ·  {fmt_num(v) if v is not None else 'missing'}")
    colors = [P["diverging_pos"] if v > 0 else P["diverging_neg"] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=colors, line_width=0),
                           hovertemplate="%{y}<br>contribution %{x:+.3f} " + unit + "<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=f"contribution ({unit})", yaxis=dict(tickfont=dict(size=11)), bargap=0.3,
                      height=max(260, 26 * len(items) + 70), margin=dict(l=10, r=16, t=36, b=40))
    fig.add_vline(x=0, line_color=P["axis"], line_width=1)
    return fig


def component_fig(components: dict, weights: dict) -> go.Figure:
    order = ["xgb", "tree", "scorecard", "bayes_vendor", "bayes_person"]
    from .models import STACK_LABELS
    probs = [components[k]["prob"] for k in order]
    fig = go.Figure(go.Bar(x=[100 * p for p in probs], y=[STACK_LABELS[k] for k in order], orientation="h",
                           marker=dict(color=[C.MODEL_COLORS[k] for k in order], line_width=0),
                           text=[f"{100 * p:.1f}%  (weight {weights[k]:+.2f})" for p, k in zip(probs, order)], textposition="outside",
                           textfont=dict(size=11, color=P["ink2"]), cliponaxis=False,
                           hovertemplate="%{y}<br>fraud probability %{x:.2f}%<extra></extra>"))
    fig.update_layout(xaxis=dict(title="fraud probability (%)", range=[0, max(100 * max(probs) * 1.6, 5)]), yaxis=dict(autorange="reversed"),
                      height=250, margin=dict(l=10, r=80, t=10, b=40), bargap=0.4)
    return fig


def waterfall_fig(blend: dict) -> go.Figure:
    from .models import STACK_LABELS
    order = ["xgb", "tree", "scorecard", "bayes_vendor", "bayes_person"]
    short = {"xgb": "XGBoost", "tree": "Decision tree", "scorecard": "Scorecard", "bayes_vendor": "Merchant prior", "bayes_person": "Customer prior"}
    x = ["intercept"] + [short[k] for k in order] + ["final log-odds"]
    y = [blend["intercept"]] + [blend["contributions"][k] for k in order] + [blend["logit"]]
    measure = ["absolute"] + ["relative"] * len(order) + ["total"]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=measure, connector=dict(line=dict(color=P["axis"], width=1)),
                                 increasing=dict(marker=dict(color=P["diverging_pos"], line_width=0)),
                                 decreasing=dict(marker=dict(color=P["diverging_neg"], line_width=0)),
                                 totals=dict(marker=dict(color=P["ink"], line_width=0)),
                                 text=[f"{v:+.2f}" for v in y], textposition="outside", textfont=dict(size=11),
                                 hovertemplate="%{x}<br>%{y:+.3f}<extra></extra>"))
    fig.update_layout(title="How the blend reaches its log-odds: intercept + Σ weight × component log-odds", yaxis_title="log-odds", height=280,
                      margin=dict(l=48, r=16, t=40, b=40), xaxis=dict(tickfont=dict(size=11)), showlegend=False)
    return fig


def line_ci_fig(x, y, lo=None, hi=None, name="rate", color=None, title="", yfmt=".2%", ytitle="") -> go.Figure:
    color = color or P["series"][0]
    fig = go.Figure()
    if lo is not None and hi is not None:
        fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=color + "22",
                                 line=dict(width=0), hoverinfo="skip", showlegend=True, name="95% interval"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2),
                             hovertemplate="%{x}<br>" + name + " %{y:" + yfmt + "}<extra></extra>"))
    fig.update_layout(title=title, yaxis=dict(tickformat=yfmt.replace("2", "1"), title=ytitle), hovermode="x unified", height=300)
    return fig


def bar_fig(x, y, orientation="v", color=None, title="", text=None, yfmt=None, xfmt=None, height=300, hover=None, colors=None) -> go.Figure:
    color = color or P["series"][0]
    fig = go.Figure(go.Bar(x=x, y=y, orientation=orientation, marker=dict(color=colors or color, line_width=0), text=text,
                           textposition="outside" if text is not None else None, textfont=dict(size=11, color=P["ink2"]), cliponaxis=False,
                           hovertemplate=hover or "%{x}<br>%{y}<extra></extra>"))
    fig.update_layout(title=title, height=height)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    if xfmt:
        fig.update_xaxes(tickformat=xfmt)
    return fig


def bar_ci_fig(labels, means, lo, hi, title="", color=None, height=320, orientation="h", fmt=".2%") -> go.Figure:
    color = color or P["series"][0]
    err_plus = np.array(hi) - np.array(means)
    err_minus = np.array(means) - np.array(lo)
    if orientation == "h":
        fig = go.Figure(go.Bar(x=means, y=labels, orientation="h", marker=dict(color=color, line_width=0),
                               error_x=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, color=P["ink2"], thickness=1.2, width=3),
                               hovertemplate="%{y}<br>%{x:" + fmt + "}<extra></extra>"))
        fig.update_layout(xaxis=dict(tickformat=fmt), yaxis=dict(autorange="reversed"))
    else:
        fig = go.Figure(go.Bar(x=labels, y=means, marker=dict(color=color, line_width=0),
                               error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, color=P["ink2"], thickness=1.2, width=3),
                               hovertemplate="%{x}<br>%{y:" + fmt + "}<extra></extra>"))
        fig.update_layout(yaxis=dict(tickformat=fmt))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=16, t=36, b=40))
    return fig


def heatmap_fig(z, x, y, title="", zfmt=".2%", colorscale=None, height=320) -> go.Figure:
    colorscale = colorscale or [[0, P["sequential"][0]], [0.5, P["sequential"][6]], [1, P["sequential"][12]]]
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, xgap=2, ygap=2, colorbar=dict(tickformat=zfmt, thickness=10, len=0.8),
                               hovertemplate="%{x} · %{y}<br>%{z:" + zfmt + "}<extra></extra>"))
    fig.update_layout(title=title, height=height, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def hist_fig(values, name="", color=None, title="", nbins=40, xtitle="", log_x=False, vline=None, vline_label=None, height=280) -> go.Figure:
    color = color or P["series"][0]
    marker = dict(color=color, line=dict(color=P["surface"], width=1))
    if log_x:
        # Plotly cannot bin a histogram on a log axis (it renders empty), so bin log10(x) and label the ticks in x.
        v = np.asarray(values, dtype=float)
        lv = np.log10(v[v > 0])
        counts, edges = np.histogram(lv, bins=nbins) if len(lv) else (np.array([]), np.array([0.0, 1.0]))
        fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), marker=marker, name=name,
                               customdata=np.stack([10 ** edges[:-1], 10 ** edges[1:]], axis=1),
                               hovertemplate="%{customdata[0]:,.2f} – %{customdata[1]:,.2f}<br>count %{y}<extra></extra>"))
        lo, hi = (edges[0], edges[-1]) if len(lv) else (0.0, 1.0)
        if vline is not None and vline > 0:
            lo, hi = min(lo, np.log10(vline)), max(hi, np.log10(vline))
        steps = (1, 2, 5) if hi - lo <= 2.5 else (1,)          # one tick per decade when the range is wide
        ticks = [m * 10.0 ** k for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for m in steps]
        ticks = [t for t in ticks if lo - 0.05 <= np.log10(t) <= hi + 0.05]
        fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[fmt_num(float(t)) for t in ticks])
        vline = np.log10(vline) if vline is not None and vline > 0 else None
    else:
        fig = go.Figure(go.Histogram(x=values, nbinsx=nbins, marker=marker, name=name, hovertemplate="%{x}<br>count %{y}<extra></extra>"))
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color=P["diverging_pos"], width=2), annotation_text=vline_label or "", annotation_position="top right",
                      annotation_font=dict(size=11, color=P["diverging_pos"]))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="count", height=height, bargap=0.05)
    return fig


def density_fig(draws: np.ndarray, name: str, color: str, title="", ref: float | None = None, ref_label="", xtitle="fraud rate", extra: list | None = None,
                xfmt: str = ".1%") -> go.Figure:
    """Posterior histogram. xfmt is the d3 tick format: '.1%' for rates, e.g. '.2f' for scale parameters."""
    hover = f"{xtitle} %{{x:{'.2%' if xfmt.endswith('%') else xfmt}}}<br>density %{{y:.1f}}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=draws, histnorm="probability density", nbinsx=60, marker=dict(color=color, line=dict(color=P["surface"], width=1)),
                               name=name, opacity=0.9, hovertemplate=hover))
    for e in extra or []:
        fig.add_trace(go.Histogram(x=e["draws"], histnorm="probability density", nbinsx=60, marker=dict(color=e["color"], line=dict(color=P["surface"], width=1)),
                                   name=e["name"], opacity=0.55, hovertemplate=hover))
    if ref is not None:
        fig.add_vline(x=ref, line=dict(color=P["ink"], width=1.5, dash="dot"), annotation_text=ref_label, annotation_position="top right", annotation_font=dict(size=11))
    fig.update_layout(title=title, xaxis=dict(title=xtitle, tickformat=xfmt), yaxis_title="posterior density", barmode="overlay", height=280)
    return fig


def trace_fig(draws: np.ndarray, name: str, height=220) -> go.Figure:
    """draws: (chains, iters). Fixed hue per chain (categorical slots 1-4)."""
    fig = go.Figure()
    for c in range(draws.shape[0]):
        fig.add_trace(go.Scatter(y=draws[c], mode="lines", name=f"chain {c + 1}", line=dict(width=1, color=P["series"][c]), opacity=0.85,
                                 hovertemplate="iter %{x}<br>" + name + " %{y:.4f}<extra></extra>"))
    fig.update_layout(title=f"Trace · {name}", height=height, xaxis_title="post-warm-up iteration", margin=dict(l=48, r=10, t=36, b=36))
    return fig


def download_json_button(obj, label, filename, key=None):
    from .scoring import trace_to_json
    st.download_button(label, data=trace_to_json(obj) if isinstance(obj, dict) else json.dumps(obj, indent=2, default=str),
                       file_name=filename, mime="application/json", key=key)


def sidebar_logo():
    logo = C.ROOT / "assets" / "logo.png"
    if logo.exists():
        st.logo(str(logo), size="large")


def sidebar_footer(store: Store):
    r = store.registry
    st.sidebar.html(
        f'<div class="fl-side-kv">Model <b>{r["model_version"]}</b><br>As-of <b>{r["as_of_date"]}</b> · seed {r["seed"]}<br>'
        f'Built {r["built_at"][:16].replace("T", " ")} UTC<br>{r["n_features"]} features · XGBoost {r["libraries"]["xgboost"]}<br>'
        f'<span style="color:#7fb2f0">Synthetic data — demonstration only</span></div>')
