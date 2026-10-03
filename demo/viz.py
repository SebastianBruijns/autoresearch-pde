"""Plotly figures for the Showcase cases. Pure functions of the arrays in demo/showcase/<case>/ (no Streamlit)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# categorical slots (validated reference palette, first three slots are safe all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, MAGENTA = "#4a3aa7", "#e87ba4"
DATA = "rgba(128,128,128,0.75)"
METHOD_COLOR = {"agent": BLUE, "pysr": ORANGE, "sparse": AQUA, "sindy": ORANGE, "weak_sindy": AQUA, "truth": VIOLET}
METHOD_LABEL = {"agent": "eqdisc agent", "pysr": "PySR", "sparse": "sparse regression (SINDy-style)",
                "sindy": "SINDy", "weak_sindy": "weak SINDy", "truth": "true PDE (re-simulated)"}


def _layout(fig, h=420, legend_top=False, **kw):
    legend = (dict(orientation="h", y=1.08, x=0) if legend_top else
              dict(orientation="h", y=-0.16, yanchor="top", x=0))
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=legend, hovermode="closest", **kw)
    return fig


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


def orbit_animation(t, U, n_orbits=4.0, n_frames=90):
    """Earth + measured orbit (faint, full) + animated satellite with a trail over the first few orbits."""
    period = 18.4
    k_end = int(np.searchsorted(t, t[0] + n_orbits * period))
    idx = np.linspace(0, k_end - 1, n_frames).astype(int)
    full = U[:: max(1, len(U) // 2500)]
    trail_len = int(0.6 * period / (t[1] - t[0]))
    fig = go.Figure()
    fig.add_trace(_earth())
    fig.add_trace(go.Scatter3d(x=full[:, 0], y=full[:, 1], z=full[:, 2], mode="lines", name="measured orbit (all 6 days)",
                               line=dict(color="rgba(100,116,139,0.6)", width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.6, 1.6], mode="lines", name="Earth's spin axis",
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
    fig.add_trace(go.Scatter3d(x=S_kep[:k, 0], y=S_kep[:k, 1], z=S_kep[:k, 2], mode="lines", name="pure Kepler (−r/|r|³)",
                               line=dict(color=ORANGE, width=6), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=S_disc[:k, 0], y=S_disc[:k, 1], z=S_disc[:k, 2], mode="lines",
                               name="discovered law (Kepler + J2)",
                               line=dict(color=tt[:k], colorscale="Blues", width=4, cmin=-horizon * 0.4, cmax=horizon),
                               hovertemplate="t = %{line.color:.0f}<extra></extra>"))
    fig.update_layout(scene=_scene(2.6))
    return _layout(fig, 520)


def orbit_elements(td_s, raan_data, argp_data, tt, raan_disc, argp_disc, raan_kep, argp_kep):
    fig = make_subplots(1, 2, subplot_titles=("node angle Ω (°)", "argument of perigee ω (°)"),
                        horizontal_spacing=0.08)
    for col, (d, m, k) in enumerate([(raan_data, raan_disc, raan_kep), (argp_data, argp_disc, argp_kep)], 1):
        fig.add_trace(go.Scatter(x=td_s, y=d - d[0] + m[np.searchsorted(tt, td_s[0])], mode="lines", name="data (1-orbit average)",
                                 line=dict(color=DATA, width=6), showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=tt, y=m, mode="lines", name="discovered model", line=dict(color=BLUE, width=2),
                                 showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=tt, y=k, mode="lines", name="pure Kepler", line=dict(color=ORANGE, width=2, dash="dash"),
                                 showlegend=col == 1), 1, col)
    fig.update_xaxes(title_text="time (units of 806.8 s)")
    _layout(fig, 400)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=60))
    return fig


# ----------------------------------------------------------------------------- KS
def ks_heatmaps(t, x, U, Y):
    lim = float(np.abs(U).max())
    D = Y - U
    dl = float(np.nanmax(np.abs(D))) or 1.0
    fig = make_subplots(1, 3, subplot_titles=("data u(x,t)", "discovered PDE, rolled out from t=0",
                                              f"model − data (max |Δ| = {dl:.1e})"),
                        shared_yaxes=True, horizontal_spacing=0.03)
    for c, (A, lo, hi) in enumerate([(U, -lim, lim), (Y, -lim, lim), (D, -dl, dl)], 1):
        fig.add_trace(go.Heatmap(z=A, x=x, y=t, colorscale="RdBu_r", zmin=lo, zmax=hi, showscale=c == 2,
                                 hovertemplate="x=%{x:.1f}<br>t=%{y:.1f}<br>%{z:.3g}<extra></extra>"), 1, c)
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="t", col=1)
    return _layout(fig, 430)


def ks_animation(t, x, U, Y, step=2):
    idx = list(range(0, len(t), step))
    lim = float(np.abs(U).max()) * 1.1
    fig = go.Figure([go.Scatter(x=x, y=U[0], mode="lines", name="data", line=dict(color=DATA, width=6)),
                     go.Scatter(x=x, y=Y[0], mode="lines", name="discovered PDE", line=dict(color=BLUE, width=2))])
    fig.frames = [go.Frame(data=[go.Scatter(y=U[i]), go.Scatter(y=Y[i])], traces=[0, 1], name=str(j))
                  for j, i in enumerate(idx)]
    fig.update_layout(yaxis=dict(range=[-lim, lim], title="u"), xaxis=dict(title="x"),
                      updatemenus=_play_buttons(50, y=-0.12), sliders=_slider([f"{t[i]:.1f}" for i in idx], y=-0.12))
    return _layout(fig, 430, legend_top=True)


def ks_error(t, err):
    fig = go.Figure(go.Scatter(x=t, y=np.maximum(err, 1e-12), mode="lines", line=dict(color=BLUE, width=2),
                               name="relative error"))
    fig.update_layout(yaxis=dict(type="log", exponentformat="power", title="rollout error / std(u)"), xaxis=dict(title="t"))
    return _layout(fig, 260)


# ----------------------------------------------------------------------------- Gray-Scott
def gs_animation(t, data, model, field=1, label="discovered PDE"):
    D, M = data[..., field], model[..., field]
    lo, hi = float(np.nanmin(D)), float(np.nanmax(D))
    fig = make_subplots(1, 2, subplot_titles=("held-out data (The Well)", f"{label}, rolled out from frame 0"),
                        horizontal_spacing=0.03)
    hm = lambda A, show: go.Heatmap(z=A.T, colorscale="Viridis", zmin=lo, zmax=hi, showscale=show,
                                    hovertemplate="B = %{z:.3f}<extra></extra>")
    fig.add_trace(hm(D[0], False), 1, 1)
    fig.add_trace(hm(M[0], True), 1, 2)
    fig.frames = [go.Frame(data=[hm(D[i], False), hm(np.nan_to_num(M[i], nan=lo), True)], traces=[0, 1], name=str(i))
                  for i in range(len(t))]
    for c in (1, 2):
        fig.update_xaxes(showticklabels=False, constrain="domain", row=1, col=c)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)
    fig.update_layout(updatemenus=_play_buttons(150, y=-0.02), sliders=_slider([f"{v:g}" for v in t], "t = ", y=-0.02))
    return _layout(fig, 520, legend_top=True)


WELL_REF = {  # The Well (Ohana et al. 2024), gray_scott_reaction_diffusion: one-step VRMSE, rollout 6-12, 13-30
    "FNO": (0.1365, 0.89, ">10"), "TFNO": (0.3633, 1.54, ">10"), "U-net": (0.2252, 0.57, ">10"),
    "CNextU-net": (0.1761, 0.29, 7.62)}


def gs_vrmse_chart(results, regime, own_curve=None):
    """VRMSE vs noise (log y) for each arm, two windows; neural surrogates as horizontal reference lines."""
    fig = make_subplots(1, 2, subplot_titles=("rollout steps 6–12", "rollout steps 13–30"), horizontal_spacing=0.06)
    x0, x1 = -0.004, 0.125
    for c, key in enumerate(("vrmse_6-12", "vrmse_13-30"), 1):
        groups = {}
        for nm, ref in WELL_REF.items():
            groups.setdefault(ref[1] if c == 1 else ref[2], []).append(nm)
        for j, (v, nms) in enumerate(groups.items()):
            nm = ", ".join(nms)
            yv = 10.0 if v == ">10" else float(v)
            fig.add_trace(go.Scatter(x=[x0, x1], y=[yv, yv], mode="lines+text", text=["", f"{nm} {v}"],
                                     textposition="middle right", textfont=dict(size=10, color="gray"),
                                     line=dict(color="rgba(140,140,140,.75)", width=1, dash="dash"),
                                     name="neural surrogates (The Well)", legendgroup="nn", showlegend=(c == 1 and j == 0),
                                     hovertemplate=f"{nm}: VRMSE {v}<extra></extra>"), 1, c)
        if results:
            rows = sorted([(float(k.split("|")[1]), v) for k, v in results.items() if k.split("|")[0] == regime])
            for arm in ("sindy", "weak_sindy", "agent", "truth"):
                xs, ys = [], []
                for n, row in rows:
                    v = (row.get(arm) or {}).get(key)
                    if v is not None:
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        xs.append(n)
                        ys.append(min(fv, 15.0) if np.isfinite(fv) else 15.0)   # blow-ups pinned to the top
                if xs:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=METHOD_LABEL[arm], legendgroup=arm,
                                             line=dict(color=METHOD_COLOR[arm], width=2,
                                                       dash="dot" if arm == "truth" else "solid"),
                                             marker=dict(size=9), showlegend=c == 1), 1, c)
        elif own_curve is not None:
            fig.add_trace(go.Scatter(x=[0.05], y=[own_curve[key]], mode="markers", name=METHOD_LABEL["truth"],
                                     marker=dict(size=11, color=VIOLET), showlegend=c == 1, legendgroup="truth"), 1, c)
    fig.update_yaxes(type="log", range=[-2.2, 1.3])
    fig.update_yaxes(title_text="VRMSE (log, lower is better)", col=1)
    fig.update_xaxes(title_text="noise level (fraction of signal std)", range=[x0, x1 + 0.045],
                     tickvals=[0, 0.01, 0.05, 0.1], ticktext=["0", "1%", "5%", "10%"])
    _layout(fig, 430)
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=70))
    return fig


def gs_gallery(gal, regimes):
    names = [r for r in regimes if r in gal]
    fig = make_subplots(1, len(names), subplot_titles=[f"{r}<br><sub>F={regimes[r][0]}, k={regimes[r][1]}</sub>" for r in names],
                        horizontal_spacing=0.01)
    for c, r in enumerate(names, 1):
        fig.add_trace(go.Heatmap(z=gal[r].T, colorscale="Viridis", showscale=False, hoverinfo="skip"), 1, c)
        fig.update_xaxes(showticklabels=False, constrain="domain", row=1, col=c)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)
    return _layout(fig, 250)


# ----------------------------------------------------------------------------- pendulum
def pendulum_phase(U, ics, fans, pick=0):
    fig = make_subplots(1, 2, column_widths=[0.55, 0.45], subplot_titles=(
        "What you measured (grey) and where to measure next (★)", f"Plausible models from start ★{pick + 1}: they disagree"))
    for j in range(U.shape[0]):
        fig.add_trace(go.Scatter(x=U[j, :, 0], y=U[j, :, 1], mode="lines", line=dict(color=DATA, width=1.2),
                                 name="measured trajectories", showlegend=j == 0, hoverinfo="skip"), 1, 1)
    F = fans[pick]
    for k in range(F.shape[0]):
        fig.add_trace(go.Scatter(x=F[k, :, 0], y=F[k, :, 1], mode="lines", line=dict(color="rgba(42,120,214,0.18)", width=1),
                                 name="plausible models (coeffs drawn from 90% CIs)", showlegend=k == 0, hoverinfo="skip"), 1, 1)
    for i, ic in enumerate(ics[:5]):
        fig.add_trace(go.Scatter(x=[ic[0]], y=[ic[1]], mode="markers+text", text=[f"#{i + 1}"], textposition="top right",
                                 marker=dict(symbol="star", size=22 if i == pick else 15,
                                             color=ORANGE if i == pick else "#eda100", line=dict(color="#222", width=1)),
                                 name="recommended starts", showlegend=i == 0,
                                 hovertemplate=f"#{i + 1}: θ={ic[0]:.2f}, ω={ic[1]:.2f}<extra></extra>"), 1, 1)
    return fig


def pendulum_fan(U, ics, fans, tt, pick=0):
    fig = pendulum_phase(U, ics, fans, pick)
    F = fans[pick]
    for k in range(F.shape[0]):
        fig.add_trace(go.Scatter(x=tt, y=F[k, :, 0], mode="lines", line=dict(color="rgba(42,120,214,0.25)", width=1),
                                 showlegend=False, hoverinfo="skip"), 1, 2)
    lo, hi = np.percentile(F[:, :, 0], 5, axis=0), np.percentile(F[:, :, 0], 95, axis=0)
    fig.add_trace(go.Scatter(x=np.r_[tt, tt[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="none",
                             fillcolor="rgba(235,104,52,0.18)", name="90% band of plausible predictions",
                             hoverinfo="skip"), 1, 2)
    fig.update_xaxes(title_text="θ (rad)", row=1, col=1)
    fig.update_yaxes(title_text="ω (rad/s)", row=1, col=1)
    fig.update_xaxes(title_text="t (s)", row=1, col=2)
    fig.update_yaxes(title_text="θ (rad)", row=1, col=2)
    return _layout(fig, 470)


# ----------------------------------------------------------------------------- static SR
def pred_vs_true(sets, title_fmt="{split}"):
    """sets: {split: (y, {method: yhat})}"""
    splits = list(sets)
    fig = make_subplots(1, len(splits), subplot_titles=[s for s in splits], horizontal_spacing=0.07)
    for c, s in enumerate(splits, 1):
        y, preds = sets[s]
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 0.1 * (hi - lo)
        for m, yh in preds.items():
            yh = np.clip(yh, lo - 3 * (hi - lo), hi + 3 * (hi - lo))
            fig.add_trace(go.Scattergl(x=y, y=yh, mode="markers", name=METHOD_LABEL.get(m, m),
                                       marker=dict(size=5, color=METHOD_COLOR.get(m, BLUE), opacity=0.55),
                                       showlegend=c == 1, legendgroup=m), 1, c)
        fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines", showlegend=False,
                                 line=dict(color="rgba(128,128,128,.8)", width=1, dash="dash"), hoverinfo="skip"), 1, c)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=c)
        fig.update_xaxes(range=[lo - pad, hi + pad], title_text="true", row=1, col=c)
    fig.update_yaxes(title_text="predicted", col=1)
    return _layout(fig, 420)


def nmse_bars(rows, methods=("agent", "pysr", "sparse"), key="OOD"):
    """rows: {problem: {method: {ID, OOD}}}"""
    fig = go.Figure()
    probs = list(rows)
    for m in methods:
        fig.add_trace(go.Bar(x=probs, y=[rows[p][m][key] for p in probs], name=METHOD_LABEL.get(m, m),
                             marker=dict(color=METHOD_COLOR.get(m), cornerradius=4),
                             hovertemplate="%{x}: %{y:.3g}<extra>" + METHOD_LABEL.get(m, m) + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                      yaxis=dict(type="log", exponentformat="power", title=f"{key} NMSE (log, lower is better)"))
    return _layout(fig, 400)


def restoring_force(xs, curves, train_range, ood_range):
    fig = go.Figure()
    fig.add_vrect(x0=ood_range[0], x1=ood_range[1], fillcolor="rgba(235,104,52,.07)", line_width=0,
                  annotation_text="test (OOD) range", annotation_position="top left")
    fig.add_vrect(x0=train_range[0], x1=train_range[1], fillcolor="rgba(42,120,214,.12)", line_width=0,
                  annotation_text="training range", annotation_position="bottom left")
    for name, (y, color, dash) in curves.items():
        fig.add_trace(go.Scatter(x=xs, y=y, mode="lines", name=name, line=dict(color=color, width=2.5, dash=dash)))
    fig.update_layout(xaxis_title="position x", yaxis_title="restoring acceleration (v = 0, no forcing)")
    return _layout(fig, 380)


def ecoli_collapse(x, y_collapsed, xs, curve, xlabel, nbins=30):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    b = np.digitize(x, edges) - 1
    med = np.array([np.nanmedian(y_collapsed[b == i]) if np.any(b == i) else np.nan for i in range(nbins)])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y_collapsed, mode="markers", name="data ÷ other factors",
                               marker=dict(size=4, color=DATA, opacity=0.35)))
    fig.add_trace(go.Scatter(x=mid, y=med, mode="markers", name="binned median", marker=dict(size=9, color=ORANGE)))
    fig.add_trace(go.Scatter(x=xs, y=curve, mode="lines", name="discovered factor", line=dict(color=BLUE, width=3)))
    fig.update_layout(xaxis_title=xlabel, yaxis_title="factor")
    return _layout(fig, 330)


def bars_simple(labels, values, colors, title):
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{x}: %{y:.3g}<extra></extra>"))
    fig.update_layout(yaxis=dict(type="log", exponentformat="power", title=title))
    return _layout(fig, 340)
