"""Structural priors and checks (eqdisc/audit/priors.py, structure.py) on small generated PDE data."""
import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def burgers(tmp_path_factory):
    from eqdisc.datagen import generate
    from eqdisc.evaluate import load
    clean = load(generate("burgers", out_root=tmp_path_factory.mktemp("d"), noise=0.0, n_traj=2, n_test=1, t_end=2, plot=False))
    noisy = load(generate("burgers", out_root=tmp_path_factory.mktemp("d"), noise=0.05, n_traj=2, n_test=1, t_end=2, plot=False))
    return clean, noisy


def _fired(findings, fid):
    return next(f for f in findings if f["id"] == fid)["fired"]


def test_priors_contract_and_implications(burgers):
    from eqdisc.audit import priors
    (m, D), _ = burgers
    out = priors.audit(m, D)
    assert out and all({"id", "stage", "fired", "severity", "fix", "details"} <= set(f) for f in out)
    card = priors.card(m, D)
    prog = card["programme"]["args"]
    assert prog.get("width_factor") == 12                     # clean data -> wide test functions
    assert {"u", "u**2"} <= set(prog["exclude_terms"])        # conserved mean -> no source monomials
    assert not any("burgers" in s.lower() for s in card["summary"])   # no equation names


def test_true_model_passes(burgers):
    from eqdisc.audit import structure
    for m, D in burgers:
        out = structure.audit(m, D, {"u": "-u*u_x + 0.05*u_xx"})
        assert not [f["id"] for f in out if f["fired"] and f["severity"] == "critical"]


def test_spurious_term_caught_and_polished(burgers):
    from eqdisc.audit import structure
    for m, D in burgers:
        bad = {"u": "-u*u_x + 0.05*u_xx + 0.01*u"}
        out = structure.audit(m, D, bad)
        assert _fired(out, "structure_width_stable") or _fired(out, "structure_necessary")
        assert _fired(out, "structure_prior_conflict")
        pol = structure.polish(m, D, bad)
        assert set(structure.structure(m, pol["rhs"])["u"]) == {"u*u_x", "u_xx"}


def test_ill_posed_flagged(burgers):
    from eqdisc.audit import structure
    _, (m, D) = burgers
    out = structure.audit(m, D, {"u": "-u*u_x - 0.05*u_xx"})
    assert _fired(out, "structure_well_posed")
