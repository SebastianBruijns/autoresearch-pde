"""Tests for eqdisc.ingest: synthetic round-trips + the real files in the repo (skipped if absent).

    python -m pytest eqdisc/tests/test_ingest.py      (or)      python -m eqdisc.tests.test_ingest
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from eqdisc.ingest import ingest, inspect

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
DS = ROOT / "datasets"


def _tmp():
    return Path(tempfile.mkdtemp(prefix="ingest_test_"))


def _load(d):
    meta = json.loads((Path(d) / "meta.json").read_text())
    data = dict(np.load(Path(d) / "data.npz"))
    return meta, data


def _src(name):
    meta = json.loads((DS / name / "meta.json").read_text())
    return meta, dict(np.load(DS / name / "data.npz"))


def _skip(cond, msg):
    if cond:
        try:
            import pytest
            pytest.skip(msg)
        except ImportError:
            raise _Skip(msg)


class _Skip(Exception):
    pass


def _lorenz_long_df():
    import pandas as pd
    meta, data = _src("lorenz_n0.01_dt1_s0")
    U, t = data["U"], data["t"]
    rows = []
    for j in range(U.shape[0]):
        rows.append(pd.DataFrame({"run": j, "t": t, **{v: U[j, :, i] for i, v in enumerate(meta["variables"])}}))
    return pd.concat(rows), U, t


# ----------------------------------------------------------------------------- ODE round trips
def test_lorenz_csv_long_with_traj_col():
    _skip(not (DS / "lorenz_n0.01_dt1_s0").exists(), "lorenz dataset missing")
    df, U, t = _lorenz_long_df()
    tmp = _tmp()
    df.sample(frac=1, random_state=0).to_csv(tmp / "lorenz.csv", index=False)   # shuffled rows
    d, card = ingest(tmp / "lorenz.csv", out_dir=tmp / "out")
    meta, data = _load(d)
    assert meta["kind"] == "ode" and meta["variables"] == ["x", "y", "z"]
    assert meta["allowed_symbols"] == ["x", "y", "z", "t"]
    np.testing.assert_allclose(data["U"], U, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(data["t"], t)
    assert abs(meta["dt"] - 0.01) < 1e-12
    assert any(a["what"] == "trajectory column" and a["value"] == "run" for a in card["assumptions"])


def test_lorenz_csv_no_traj_col_time_resets():
    _skip(not (DS / "lorenz_n0.01_dt1_s0").exists(), "lorenz dataset missing")
    df, U, t = _lorenz_long_df()
    tmp = _tmp()
    df.drop(columns="run").to_csv(tmp / "lorenz.tsv", index=False, sep="\t")
    d, card = ingest(tmp / "lorenz.tsv", out_dir=tmp / "out")
    _, data = _load(d)
    np.testing.assert_allclose(data["U"], U, rtol=1e-9, atol=1e-9)


def test_lorenz_mat_vars_first_and_hints():
    _skip(not (DS / "lorenz_n0.01_dt1_s0").exists(), "lorenz dataset missing")
    import scipy.io as sio
    _, data0 = _src("lorenz_n0.01_dt1_s0")
    U, t = data0["U"], data0["t"]
    tmp = _tmp()
    # (n_vars, nt, n_traj) layout: needs a hint
    sio.savemat(tmp / "l.mat", {"time": t[None, :], "Y": np.transpose(U, (2, 1, 0))})
    d, card = ingest(tmp / "l.mat", hints={"layout": "var,t,run", "rename": "y0:x,y1:y,y2:z"},
                     out_dir=tmp / "out")
    meta, data = _load(d)
    np.testing.assert_allclose(data["U"], U)
    assert meta["variables"] == ["x", "y", "z"]


def test_nonuniform_time_and_nans_and_params():
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []
    for j, a in enumerate([0.5, 1.0, 2.0]):
        t = np.sort(np.concatenate([[0, 10], rng.uniform(0, 10, 400)]))
        rows.append(pd.DataFrame({"case": j, "time": t, "x": np.sin(a * t), "v": a * np.cos(a * t),
                                  "omega": a, "const": 3.0}))
    df = pd.concat(rows, ignore_index=True)
    df.loc[[5, 6, 7, 300], "x"] = np.nan
    tmp = _tmp()
    df.to_csv(tmp / "osc.csv", index=False)
    d, card = ingest(tmp / "osc.csv", out_dir=tmp / "out")
    meta, data = _load(d)
    assert meta["variables"] == ["x", "v"] and meta["parameters"] == ["omega"]
    np.testing.assert_allclose(data["params"][:, 0], [0.5, 1.0, 2.0])
    t = data["t"]
    assert np.allclose(np.diff(t), meta["dt"])
    truth = np.stack([np.sin(a * t) for a in (0.5, 1.0, 2.0)])
    assert np.max(np.abs(data["U"][..., 0] - truth)) < 2e-2
    whats = [a["what"] for a in card["assumptions"]]
    assert any("time resampling" in w for w in whats)
    assert any("constant columns" in w for w in whats)
    assert "missing_values" in card["checks"] or any("missing" in w for w in card["warnings"])


def test_name_sanitising():
    import pandas as pd
    t = np.linspace(0, 1, 50)
    df = pd.DataFrame({"t": t, "I": t, "lambda": t ** 2, "S(t) [mol]": np.cos(t), "sin": np.sin(t)})
    tmp = _tmp()
    df.to_csv(tmp / "n.csv", index=False)
    d, card = ingest(tmp / "n.csv", out_dir=tmp / "out")
    meta, _ = _load(d)
    assert meta["variables"] == ["I", "lambda_", "S_t_mol", "sin_"]
    assert all(v.isidentifier() for v in meta["variables"])
    from eqdisc.solvers import parse
    parse("lambda_*I + S_t_mol*sin_", meta["allowed_symbols"])


def test_single_row_table_rejected():
    import pandas as pd
    tmp = _tmp()
    pd.DataFrame({"G": [1.0], "R": [2.0]}).to_csv(tmp / "p.csv", index=False)
    try:
        ingest(tmp / "p.csv", out_dir=tmp / "out")
    except ValueError as e:
        assert "constants" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ----------------------------------------------------------------------------- PDE round trips
def test_kdv_csv_long_pivot():
    _skip(not (DS / "kdv_n0.01_dt1_s0").exists(), "kdv dataset missing")
    import pandas as pd
    meta0, data0 = _src("kdv_n0.01_dt1_s0")
    U, t, x = data0["U"], data0["t"], data0["x"]
    T, Xg = np.meshgrid(t, x, indexing="ij")
    df = pd.concat([pd.DataFrame({"traj": j, "t": T.ravel(), "x": Xg.ravel(), "u": U[j, ..., 0].ravel()})
                    for j in range(U.shape[0])])
    tmp = _tmp()
    df.to_csv(tmp / "kdv_long.csv", index=False)
    d, card = ingest(tmp / "kdv_long.csv", out_dir=tmp / "out")
    meta, data = _load(d)
    assert meta["kind"] == "pde" and meta["boundary"] == "periodic"
    np.testing.assert_allclose(data["U"], U, rtol=1e-9, atol=1e-9)
    assert abs(meta["L"] - meta0["L"]) < 1e-9 and meta["nx"] == meta0["nx"]
    assert meta["allowed_symbols"] == meta0["allowed_symbols"]
    assert meta["grid"]["x"]["n"] == 256 and meta["spatial_dims"] == ["x"]


def test_kdv_mat_permuted_axes():
    _skip(not (DS / "kdv_n0.01_dt1_s0").exists(), "kdv dataset missing")
    import scipy.io as sio
    meta0, data0 = _src("kdv_n0.01_dt1_s0")
    U, t, x = data0["U"], data0["t"], data0["x"]
    tmp = _tmp()
    sio.savemat(tmp / "k.mat", {"x": x[:, None], "t": t[None, :], "u": np.transpose(U[..., 0], (2, 1, 0))})
    d, card = ingest(tmp / "k.mat", out_dir=tmp / "out")
    meta, data = _load(d)
    np.testing.assert_allclose(data["U"], U)
    assert meta["boundary"] == "periodic" and abs(meta["L"] - meta0["L"]) < 1e-9
    assert meta["n_traj"] == 3


def test_duplicated_endpoint_and_nonperiodic():
    _skip(not (DS / "kdv_n0.01_dt1_s0").exists(), "kdv dataset missing")
    meta0, data0 = _src("kdv_n0.01_dt1_s0")
    U, t, x = data0["U"][..., 0], data0["t"], data0["x"]
    tmp = _tmp()
    Ud = np.concatenate([U, U[..., :1]], -1)                       # endpoint included
    np.savez(tmp / "dup.npz", t=t, x=np.append(x, meta0["L"]), u=Ud)
    d, card = ingest(tmp / "dup.npz", out_dir=tmp / "out")
    meta, data = _load(d)
    assert meta["boundary"] == "periodic" and meta["nx"] == 256 and abs(meta["L"] - meta0["L"]) < 1e-9
    np.testing.assert_allclose(data["U"][..., 0], U)
    # a window cut out of the middle of the domain is not periodic
    np.savez(tmp / "cut.npz", t=t, x=x[40:160], u=U[..., 40:160])
    d, card = ingest(tmp / "cut.npz", out_dir=tmp / "out")
    meta, _ = _load(d)
    assert meta["boundary"] == "unknown"
    assert any("boundary=unknown" in w for w in card["warnings"])


def test_2d_pde_meta():
    nx, ny, nt = 32, 24, 12
    x = np.arange(nx) * 2 * np.pi / nx
    y = np.arange(ny) * 4 * np.pi / ny
    t = np.linspace(0, 1, nt)
    u = np.sin(x[None, :, None] - t[:, None, None]) * np.cos(0.5 * y[None, None, :])
    v = np.cos(x[None, :, None] + t[:, None, None]) * np.sin(0.5 * y[None, None, :])
    tmp = _tmp()
    np.savez(tmp / "w.npz", t=t, x=x, y=y, u=u, v=v)
    d, card = ingest(tmp / "w.npz", out_dir=tmp / "out")
    meta, data = _load(d)
    assert data["U"].shape == (1, nt, nx, ny, 2)
    assert meta["spatial_dims"] == ["x", "y"] and meta["variables"] == ["u", "v"]
    assert meta["boundary"] == "periodic"
    assert abs(meta["grid"]["y"]["L"] - 4 * np.pi) < 1e-9 and meta["grid"]["x"]["n"] == nx
    assert "u_yy" in meta["allowed_symbols"] and "y" in meta["allowed_symbols"]


def test_eqdisc_dir_and_json_npy():
    _skip(not (DS / "lorenz_n0.01_dt1_s0").exists(), "lorenz dataset missing")
    tmp = _tmp()
    d, _ = ingest(DS / "lorenz_n0.01_dt1_s0", out_dir=tmp / "out", name="lz")
    meta, data = _load(d)
    assert meta["variables"] == ["x", "y", "z"]
    _, data0 = _src("lorenz_n0.01_dt1_s0")
    t, U = data0["t"], data0["U"]
    (tmp / "l.json").write_text(json.dumps({"t": t.tolist(), "a": U[0, :, 0].tolist(), "b": U[0, :, 1].tolist()}))
    d, _ = ingest(tmp / "l.json", out_dir=tmp / "out")
    _, data = _load(d)
    np.testing.assert_allclose(data["U"][0], U[0, :, :2])
    np.save(tmp / "arr.npy", U[0])
    d, card = ingest(tmp / "arr.npy", hints={"dt": 0.01}, out_dir=tmp / "out")
    meta, data = _load(d)
    assert data["U"].shape == (1, U.shape[1], 3) and abs(meta["dt"] - 0.01) < 1e-12


# ----------------------------------------------------------------------------- real files
DATA = ROOT / "examples" / "data"
KDV1 = DATA / "kdv_data.mat"
KDVW = DATA / "kdv_data_for_workshop.mat"
KS = DATA / "KS_data.mat"
ORBIT = DATA


def test_real_kdv_mat():
    _skip(not KDV1.exists(), "kdv_data.mat missing")
    d, card = ingest(KDV1, out_dir=_tmp())
    meta, data = _load(d)
    assert data["U"].shape == (1, 51, 401, 1) and meta["boundary"] == "periodic"
    pre = card["checks"]["precomputed_derivatives"]
    assert pre["u_x_1"]["best_match"] == "u_x" and pre["u_x_3"]["best_match"] == "u_xxx"
    assert abs(pre["_implied_L"]["L_implied_by_tables"] - 2.0) < 1e-3


def test_real_kdv_workshop_mat():
    _skip(not KDVW.exists(), "kdv_data_for_workshop.mat missing")
    d, card = ingest(KDVW, out_dir=_tmp())
    meta, data = _load(d)
    assert data["U"].shape == (8, 10, 400, 1) and meta["n_traj"] == 8
    raw = __import__("scipy.io", fromlist=["loadmat"]).loadmat(KDVW)["u"]
    np.testing.assert_allclose(data["U"][..., 0], raw)
    assert any("under-resolved" in w for w in card["warnings"])


def test_real_ks_mat():
    _skip(not KS.exists(), "KS_data.mat missing")
    d, card = ingest(KS, out_dir=_tmp(), hints={"rename": "uu:u"})
    meta, data = _load(d)
    assert data["U"].shape == (1, 251, 1024, 1) and meta["variables"] == ["u"]
    assert meta["boundary"] == "periodic" and abs(meta["L"] - 32 * np.pi) < 1e-6


def test_real_orbit_csv():
    _skip(not (ORBIT / "Challenge1.csv").exists(), "orbit data missing")
    d, card = ingest(ORBIT / "Challenge1.csv", out_dir=_tmp())
    meta, data = _load(d)
    assert meta["variables"] == ["rx", "ry", "rz", "vx", "vy", "vz"] and meta["dt"] == 30.0
    if (ORBIT / "lageos1.csv").exists():          # 18 MB, not bundled: SymbolicModel/orbit_discover
        inv = inspect(ORBIT / "lageos1.csv")
        assert len(inv["columns"]) == 13


if __name__ == "__main__":
    import sys
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", k)
            except _Skip as e:
                print("SKIP", k, e)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", k, repr(e))
    sys.exit(1 if fails else 0)
