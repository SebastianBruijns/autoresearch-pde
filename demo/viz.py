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
LAGEOS_LABEL = {"Kepler": "Simple gravity", "Kepler + J2": "Textbook law",
                "neural step model (MLP)": "Neural network", "agent": "Discovered law"}


def lageos_errors(days, errs, agent_pts=None, xlabel="days into the unseen month (after the 2017 training year)"):
    fig = go.Figure()
    for name, e in errs.items():
        c, dash = LAGEOS_STYLE.get(name, (GREY, "solid"))
        fig.add_trace(go.Scatter(x=days, y=np.maximum(e, 1e-3), mode="lines", name=LAGEOS_LABEL.get(name, name),
                                 line=dict(color=c, width=2.5, dash=dash),
                                 hovertemplate="%{y:,.3g} km<extra>" + LAGEOS_LABEL.get(name, name) + "</extra>"))
    if agent_pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in agent_pts], y=[p[1] for p in agent_pts], mode="markers",
                                 name="data-only agent", marker=dict(size=11, color=AQUA, symbol="diamond")))
    fig.update_layout(xaxis_title=xlabel,
                      yaxis=dict(type="log", title="position error (km)", exponentformat="power"))
    return _layout(fig, 340)


# ----------------------------------------------------------------------------- synthetic orbit (restored)
def _play_buttons(duration=60, y=0, x=0.0):
    return [dict(type="buttons", showactive=False, x=x, y=y, xanchor="left", yanchor="top", direction="left",
                 pad=dict(t=8, r=8),
                 buttons=[dict(label="▶ Play", method="animate",
                               args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True,
                                                transition=dict(duration=0), mode="immediate")]),
                          dict(label="⏸ Pause", method="animate",
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]


def _slider(labels, prefix="t = ", y=0):
    return [dict(active=0, y=y, x=0.12, len=0.88, xanchor="left", yanchor="top", pad=dict(t=8), ticklen=0,
                 minorticklen=0, font=dict(size=1, color="rgba(0,0,0,0)"),
                 currentvalue=dict(prefix=prefix, visible=True),
                 steps=[dict(method="animate", label=lb,
                             args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0))]) for i, lb in enumerate(labels)])]


# ----------------------------------------------------------------------------- orbit
def _earth(n_lat=48, n_lon=96, R=1.0):
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    LON, LAT = np.meshgrid(lon, lat)
    x, y, z = R * np.cos(LAT) * np.cos(LON), R * np.cos(LAT) * np.sin(LON), R * np.sin(LAT)
    # smooth pseudo-continents: a few low-order waves, thresholded by the colourscale
    rng = np.random.default_rng(3)
    f = np.zeros_like(LAT)
    for _ in range(9):
        a, b, ph = rng.integers(1, 5), rng.integers(1, 4), rng.uniform(0, 2 * np.pi)
        f += rng.uniform(0.4, 1.0) * np.cos(a * LON + ph) * np.cos(b * LAT + rng.uniform(0, np.pi))
    f = (f - f.min()) / (f.max() - f.min())
    f = np.where(np.abs(LAT) > 1.25, 1.2, f)            # polar ice
    cs = [[0, "#0b3d91"], [0.45, "#1e6fd9"], [0.52, "#3fa0e8"], [0.53, "#2f8f46"], [0.7, "#6aa84f"],
          [0.82, "#b9a76b"], [0.83, "#f2f6fa"], [1, "#ffffff"]]
    return go.Surface(x=x, y=y, z=z, surfacecolor=np.clip(f / 1.2, 0, 1), colorscale=cs, cmin=0, cmax=1,
                      showscale=False, hoverinfo="skip", name="Earth",
                      lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.6),
                      lightposition=dict(x=1e4, y=5e3, z=5e3))


def _scene(lim=2.7):
    ax = dict(range=[-lim, lim], showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="",
              showspikes=False)
    return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=0.95, y=0.95, z=0.5)))


def orbit_animation(t, U, n_orbits=4.0, n_frames=90, full_label="3 noisy days"):
    """Earth + measured orbit (faint, full) + animated satellite with a trail over the first few orbits."""
    period = 18.4
    k_end = int(np.searchsorted(t, t[0] + n_orbits * period))
    idx = np.linspace(0, k_end - 1, n_frames).astype(int)
    full = U[:: max(1, len(U) // 2500)]
    trail_len = int(0.6 * period / (t[1] - t[0]))
    fig = go.Figure()
    fig.add_trace(_earth())
    fig.add_trace(go.Scatter3d(x=full[:, 0], y=full[:, 1], z=full[:, 2], mode="lines", name=full_label,
                               line=dict(color="rgba(100,116,139,0.6)", width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.6, 1.6], mode="lines", name="spin axis",
                               line=dict(color="rgba(200,200,200,0.8)", width=4, dash="dash"), hoverinfo="skip"))

    def trail(i):
        a = max(0, i - trail_len)
        return go.Scatter3d(x=U[a:i + 1, 0], y=U[a:i + 1, 1], z=U[a:i + 1, 2], mode="lines", name="recent track",
                            line=dict(color=ORANGE, width=6), hoverinfo="skip")

    def sat(i):
        return go.Scatter3d(x=[U[i, 0]], y=[U[i, 1]], z=[U[i, 2]], mode="markers", name="satellite",
                            marker=dict(size=7, color="#ffd400", line=dict(color="#222", width=1)),
                            hovertemplate=f"t = {t[i]:.1f}<br>r = {np.linalg.norm(U[i]):.2f} Re<extra></extra>")
    fig.add_trace(trail(idx[0]))
    fig.add_trace(sat(idx[0]))
    fig.frames = [go.Frame(data=[trail(i), sat(i)], traces=[3, 4], name=str(j)) for j, i in enumerate(idx)]
    fig.update_layout(scene=_scene(), updatemenus=_play_buttons(70, y=0.02),
                      sliders=_slider([f"{t[i]:.0f}" for i in idx], y=0.02))
    return _layout(fig, 560, legend_top=True, showlegend=True)


def orbit_kepler_vs_j2(tt, S_disc, S_kep, horizon=110.0):
    """Same initial state, two laws: Kepler's ellipse stays in one plane; the J2 orbit's plane precesses."""
    k = int(np.searchsorted(tt, tt[0] + horizon))
    fig = go.Figure()
    fig.add_trace(_earth(32, 64))
    fig.add_trace(go.Scatter3d(x=S_kep[:k, 0], y=S_kep[:k, 1], z=S_kep[:k, 2], mode="lines", name="Newton's gravity, perfectly round Earth",
                               line=dict(color=ORANGE, width=6), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=S_disc[:k, 0], y=S_disc[:k, 1], z=S_disc[:k, 2], mode="lines",
                               name="true law with the bulge (light → dark = time)",
                               line=dict(color=tt[:k], colorscale="Blues", width=4, cmin=-horizon * 0.4, cmax=horizon),
                               hovertemplate="t = %{line.color:.0f}<extra></extra>"))
    fig.update_layout(scene=_scene(2.6))
    return _layout(fig, 520)


def orbit_elements(td_s, raan_data, argp_data, tt, raan_disc, argp_disc, raan_kep, argp_kep):
    fig = make_subplots(1, 2, subplot_titles=("where the orbit plane crosses the equator, Ω (°)", "where the orbit is lowest, ω (°)"),
                        horizontal_spacing=0.08)
    for col, (d, m, k) in enumerate([(raan_data, raan_disc, raan_kep), (argp_data, argp_disc, argp_kep)], 1):
        fig.add_trace(go.Scatter(x=td_s, y=d, mode="lines", name="measured (averaged over one orbit)",
                                 line=dict(color=GREY, width=6), showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=tt, y=m, mode="lines", name="eqdisc forecast (unseen half)", line=dict(color=BLUE, width=2),
                                 showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=tt, y=k, mode="lines", name="round-Earth gravity forecast", line=dict(color=ORANGE, width=2, dash="dash"),
                                 showlegend=col == 1), 1, col)
    fig.update_xaxes(title_text="hours")
    for col, d in enumerate((raan_data, argp_data), 1):          # keep the measured data readable
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
        pad = 0.25 * (hi - lo + 1e-9)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=col)
    _layout(fig, 400)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=60))
    return fig



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
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.7, 1.7], mode="lines", name="spin axis",
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


PLAIN = {"true PDE from noisy state": "True equation", "true PDE from noisy frame": "True equation",
         "weak SINDy (no LLM)": "Sparse regression", "eqdisc agent (refit)": "Discovered law",
         "FNO (same noisy data)": "Neural network"}


def ks_errors(t_lyap, errs, thr=0.5):
    fig = go.Figure()
    for name, e in errs.items():
        c, dash = KS_STYLE.get(name, (GREY, "solid"))
        lab = PLAIN.get(name, name)
        fig.add_trace(go.Scatter(x=t_lyap, y=e, mode="lines", name=lab, line=dict(color=c, width=2.5, dash=dash),
                                 hovertemplate="%{y:.2f}<extra>" + lab + "</extra>"))
    fig.add_hline(y=thr, line=dict(color="rgba(120,120,120,.6)", width=1, dash="dash"),
                  annotation_text="useless above here", annotation_position="bottom right",
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
        fig.add_trace(go.Bar(x=[f"steps {w}" for w in wins], y=ys, name=PLAIN.get(name, name),
                             marker=dict(color=GS_STYLE.get(name, VIOLET), cornerradius=4),
                             hovertemplate="%{y:.3g}<extra>" + PLAIN.get(name, name) + "</extra>"))
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
                                 name="published neural nets", legendgroup="well",
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


def gs_simple(vrmse):
    """One clean horizontal bar per method (forecast error, steps 6-12), plus the best published neural net."""
    order = [("eqdisc agent (refit)", "Discovered law", BLUE), ("true PDE from noisy frame", "True equation", GREY),
             ("FNO (same noisy data)", "Neural network", ORANGE), ("weak SINDy (no LLM)", "Sparse regression", AQUA)]
    lab, val, col = [], [], []
    for k, name, c in order:
        if k in vrmse:
            lab.append(name)
            val.append(float(vrmse[k]["6-12"]))
            col.append(c)
    lab.append("Best published neural net")
    val.append(0.29)
    col.append("rgba(120,120,120,.45)")
    fig = go.Figure(go.Bar(x=val, y=lab, orientation="h", marker_color=col, text=[f"{v:.2f}" for v in val],
                           textposition="outside", textfont=dict(size=15)))
    fig.update_layout(xaxis=dict(title="error (lower is better)", range=[0, max(val) * 1.25]),
                      yaxis=dict(autorange="reversed"), showlegend=False)
    return _layout(fig, 360, hovermode="closest")
