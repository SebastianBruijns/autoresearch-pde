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
LAGEOS_STYLE = {"Kepler": (ORANGE, "dash"), "Kepler + J2": (GREY, "dot"), "neural step model (MLP)": (VIOLET, "solid"),
                "agent": (AQUA, "solid")}
LAGEOS_LABEL = {"Kepler": "Kepler (−μr/r³)", "Kepler + J2": "physics reference (Kepler + J2, hand-built)",
                "neural step model (MLP)": "neural net (MLP), same data", "agent": "data-only agent (refit)"}


def lageos_errors(days, errs, agent_pts=None):
    fig = go.Figure()
    for name, e in errs.items():
        c, dash = LAGEOS_STYLE.get(name, (GREY, "solid"))
        fig.add_trace(go.Scatter(x=days, y=np.maximum(e, 1e-3), mode="lines", name=LAGEOS_LABEL.get(name, name),
                                 line=dict(color=c, width=2.5, dash=dash),
                                 hovertemplate="%{y:,.3g} km<extra>" + LAGEOS_LABEL.get(name, name) + "</extra>"))
    if agent_pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in agent_pts], y=[p[1] for p in agent_pts], mode="markers",
                                 name="data-only agent", marker=dict(size=11, color=AQUA, symbol="diamond")))
    fig.update_layout(xaxis_title="days into the unseen month (after the 2017 training year)",
                      yaxis=dict(type="log", title="position error (km)", exponentformat="power"))
    return _layout(fig, 340)


# ----------------------------------------------------------------------------- KS
KS_STYLE = {"true PDE from noisy state": (GREY, "dash"), "weak SINDy (no LLM)": (AQUA, "dot"),
            "eqdisc agent (refit)": (BLUE, "solid"), "FNO (same noisy data)": (ORANGE, "solid")}


def _earth_mesh(n_lat=36, n_lon=72):
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    LON, LAT = np.meshgrid(lon, lat)
    return go.Surface(x=np.cos(LAT) * np.cos(LON), y=np.cos(LAT) * np.sin(LON), z=np.sin(LAT),
                      surfacecolor=np.sin(LAT), colorscale=[[0, "#1e4f9a"], [1, "#3f8fd8"]], showscale=False,
                      hoverinfo="skip", name="Earth", opacity=0.95,
                      lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2))


def lageos_training(orbits, dates, h=None):
    """The training data in 3-D: one orbit every two weeks of 2017 (real, hourly samples densified for display),
    animated through the year so the orbit plane's slow turn about Earth's axis is visible."""
    lim = 2.2
    ax = dict(range=[-lim, lim], showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="",
              showspikes=False)
    n = len(orbits)
    cols = [f"rgba(42,120,214,{0.15 + 0.6 * i / max(n - 1, 1):.2f})" for i in range(n)]
    fig = go.Figure()
    fig.add_trace(_earth_mesh())
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.7, 1.7], mode="lines", name="Earth's spin axis",
                               line=dict(color="rgba(150,150,150,0.9)", width=4, dash="dash"), hoverinfo="skip"))
    for i, O in enumerate(orbits):          # faint history of all fortnightly orbits
        fig.add_trace(go.Scatter3d(x=O[:, 0], y=O[:, 1], z=O[:, 2], mode="lines", showlegend=i == 0,
                                   name="2017 training orbits (one every 2 weeks)",
                                   line=dict(color=cols[i], width=2), hoverinfo="skip"))
    cur = lambda i: go.Scatter3d(x=orbits[i][:, 0], y=orbits[i][:, 1], z=orbits[i][:, 2], mode="lines",
                                 name="orbit on this date", line=dict(color="#eb6834", width=7),
                                 hovertemplate=f"{dates[i]}<extra></extra>")
    fig.add_trace(cur(0))
    k = len(fig.data) - 1
    fig.frames = [go.Frame(data=[cur(i)], traces=[k], name=str(i)) for i in range(n)]
    fig.update_layout(
        scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=1.1, y=1.1, z=0.7))),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=0.02, xanchor="left", yanchor="top",
                          direction="left", pad=dict(t=8, r=8),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0), mode="immediate")]),
                                   dict(label="⏸", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, y=0.02, x=0.14, len=0.86, xanchor="left", yanchor="top", pad=dict(t=8),
                      currentvalue=dict(prefix="", visible=True), ticklen=0,
                      steps=[dict(method="animate", label=d, args=[[str(i)], dict(mode="immediate",
                             frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                             for i, d in enumerate(dates)])])
    return _layout(fig, 520, legend_top=True)


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
