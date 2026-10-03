"""The one chart per case, plus the live tab's generic plots. Pure plotly, no Streamlit."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# validated categorical slots; the discovered equation is always BLUE, baselines orange/aqua, reference grey
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY = "rgba(120,120,120,0.85)"


def _layout(fig, h=360, legend_top=False, **kw):
    legend = dict(orientation="h", y=1.1, x=0) if legend_top else dict(orientation="h", y=-0.2, yanchor="top", x=0)
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), legend=legend, **kw)
    return fig


# ----------------------------------------------------------------------------- LAGEOS
LAGEOS_STYLE = {"Kepler": (ORANGE, "dash"), "Kepler + J2": (BLUE, "solid"), "neural step model (MLP)": (AQUA, "dot")}
LAGEOS_LABEL = {"Kepler": "Kepler (−μr/r³)", "Kepler + J2": "discovered law (Kepler + J2)",
                "neural step model (MLP)": "neural net (MLP), same data"}


def lageos_errors(days, errs, agent_pts=None):
    fig = go.Figure()
    for name, e in errs.items():
        c, dash = LAGEOS_STYLE.get(name, (GREY, "solid"))
        fig.add_trace(go.Scatter(x=days, y=np.maximum(e, 1e-3), mode="lines", name=LAGEOS_LABEL.get(name, name),
                                 line=dict(color=c, width=2.5, dash=dash),
                                 hovertemplate="%{y:,.3g} km<extra>" + LAGEOS_LABEL.get(name, name) + "</extra>"))
    if agent_pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in agent_pts], y=[p[1] for p in agent_pts], mode="markers",
                                 name="eqdisc agent's own law", marker=dict(size=11, color=VIOLET, symbol="diamond")))
    fig.update_layout(xaxis_title="days into the unseen month (after the 2017 training year)",
                      yaxis=dict(type="log", title="position error (km)", exponentformat="power"))
    return _layout(fig, 340)


# ----------------------------------------------------------------------------- KS
KS_STYLE = {"true PDE from noisy state": (GREY, "dash"), "weak SINDy (no LLM)": (AQUA, "dot"),
            "eqdisc agent (refit)": (BLUE, "solid"), "FNO (same noisy data)": (ORANGE, "solid")}


def ks_errors(t_lyap, errs, thr=0.5):
    fig = go.Figure()
    for name, e in errs.items():
        c, dash = KS_STYLE.get(name, (GREY, "solid"))
        fig.add_trace(go.Scatter(x=t_lyap, y=e, mode="lines", name=name, line=dict(color=c, width=2.5, dash=dash),
                                 hovertemplate="%{y:.2f}<extra>" + name + "</extra>"))
    fig.add_hline(y=thr, line=dict(color="rgba(120,120,120,.6)", width=1, dash="dash"),
                  annotation_text="valid-forecast threshold", annotation_position="bottom right",
                  annotation_font_size=10)
    fig.update_layout(xaxis_title="Lyapunov times into the unseen future",
                      yaxis=dict(title="relative error vs truth", range=[0, 1.6]))
    return _layout(fig, 340)


# ----------------------------------------------------------------------------- Gray-Scott
WELL_REF = {"FNO": (0.89, ">10"), "U-net": (0.57, ">10"), "CNextU-net": (0.29, 7.62)}  # rollout windows 6-12 / 13-30
GS_STYLE = {"true PDE from noisy frame": GREY, "weak SINDy (no LLM)": AQUA, "eqdisc agent (refit)": BLUE,
            "FNO (same noisy data)": ORANGE}


def gs_vrmse(vrmse):
    """vrmse: {method: {"6-12": v, "13-30": v}} -> grouped bars (log) + The Well paper's surrogates as lines."""
    wins = ["6-12", "13-30"]
    fig = go.Figure()
    for name, v in vrmse.items():
        ys = [min(float(v.get(w, np.nan)), 50) if v.get(w) is not None else None for w in wins]
        fig.add_trace(go.Bar(x=[f"steps {w}" for w in wins], y=ys, name=name,
                             marker=dict(color=GS_STYLE.get(name, VIOLET), cornerradius=4),
                             hovertemplate="%{y:.3g}<extra>" + name + "</extra>"))
    groups = {}
    for nm, vals in WELL_REF.items():
        for k, val in enumerate(vals):
            groups.setdefault((k, val), []).append(nm)
    for j, ((k, val), nms) in enumerate(groups.items()):
        nm = ", ".join(nms)
        yv = 10.0 if val == ">10" else float(val)
        fig.add_trace(go.Scatter(x=[k - 0.45, k + 0.45], y=[yv, yv], xaxis="x2", mode="lines+text",
                                 text=["", f"{nm} {val}"], textposition="top left", textfont=dict(size=10, color="gray"),
                                 line=dict(color="rgba(120,120,120,.7)", dash="dash", width=1),
                                 name="The Well paper (neural, 100s of trajectories)", legendgroup="well",
                                 showlegend=(j == 0), hoverinfo="skip"))
    fig.update_layout(barmode="group", bargap=0.3,
                      xaxis2=dict(overlaying="x", range=[-0.5, 1.5], visible=False),
                      yaxis=dict(type="log", title="VRMSE on the held-out trajectory (lower is better)",
                                 exponentformat="power"))
    return _layout(fig, 360, hovermode="closest")


def gs_animation(t, D, M, label="true PDE (reference)"):
    lo, hi = float(np.nanmin(D)), float(np.nanmax(D))
    fig = make_subplots(1, 2, subplot_titles=("held-out data", label), horizontal_spacing=0.02)
    hm = lambda A: go.Heatmap(z=A.T, colorscale="Magma", zmin=lo, zmax=hi, showscale=False, hoverinfo="skip")
    fig.add_trace(hm(D[0]), 1, 1)
    fig.add_trace(hm(M[0]), 1, 2)
    fig.frames = [go.Frame(data=[hm(D[i]), hm(np.nan_to_num(M[i], nan=lo))], traces=[0, 1], name=str(i))
                  for i in range(len(t))]
    for c in (1, 2):
        fig.update_xaxes(visible=False, constrain="domain", row=1, col=c)
        fig.update_yaxes(visible=False, scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)
    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.02, xanchor="left", yanchor="top",
                                        buttons=[dict(label="▶ Play", method="animate",
                                                      args=[None, dict(frame=dict(duration=150, redraw=True),
                                                                       fromcurrent=True, transition=dict(duration=0))])])])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=30))
    return fig


def gs_gallery(gal, regimes):
    names = [r for r in regimes if r in gal]
    fig = make_subplots(1, len(names), subplot_titles=names, horizontal_spacing=0.01)
    for c, r in enumerate(names, 1):
        fig.add_trace(go.Heatmap(z=gal[r].T, colorscale="Magma", showscale=False, hoverinfo="skip"), 1, c)
        fig.update_xaxes(visible=False, constrain="domain", row=1, col=c)
        fig.update_yaxes(visible=False, scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)
    fig.update_layout(height=210, margin=dict(l=0, r=0, t=30, b=0))
    return fig


# ----------------------------------------------------------------------------- live (static mode)
def pred_vs_true(y, yhat):
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 0.05 * (hi - lo + 1e-12)
    fig = go.Figure([go.Scattergl(x=y, y=np.clip(yhat, lo - (hi - lo), hi + (hi - lo)), mode="markers", name="rows",
                                  marker=dict(size=4, color=BLUE, opacity=0.5)),
                     go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines", name="perfect",
                                line=dict(color=GREY, dash="dash", width=1))])
    fig.update_layout(xaxis_title="measured", yaxis_title="predicted by the discovered law", showlegend=False)
    return _layout(fig, 360, hovermode="closest")
