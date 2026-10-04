"""Symbolic-equivalence judge (eqdisc.judge.sympy_check): coefficients within 5% count as equivalent, matched term
by term. Regression test for integer coefficients (exactly 1) and coefficients shared between terms.

    python -m eqdisc.tests.test_judge
"""
from eqdisc.judge import sympy_check
from eqdisc.solvers import derivative_symbols

NAMES = derivative_symbols(["u", "v", "omega", "s"], ["x", "y"], 2)


def eq(t, c):
    return sympy_check(t, c, NAMES)[0] is True


def test_equivalent_up_to_5pct():
    truth = "-u*omega_x - v*omega_y + 0.0001*(omega_xx + omega_yy)"
    assert eq(truth, truth)
    assert eq(truth, "-1.004*u*omega_x - 0.998*v*omega_y + 0.000102*omega_xx + 0.0000995*omega_yy")
    assert eq("-u*s_x - v*s_y", "-1.03*u*s_x - 0.99*v*s_y")


def test_not_equivalent():
    truth = "-u*omega_x - v*omega_y + 0.0001*(omega_xx + omega_yy)"
    assert not eq(truth, "-1.2*u*omega_x - v*omega_y + 0.0001*(omega_xx + omega_yy)")    # coefficient 20% off
    assert not eq(truth, "-u*omega_x - v*omega_y")                                        # missing terms
    assert not eq(truth, "-u*omega_x - v*omega_y + 0.0001*(omega_xx + omega_yy) + 0.3*u")  # extra term
    assert not eq("-u*s_x", "-1.004*v*s_x")                                                # wrong variable


def test_rational_and_inner_constants():
    assert eq("2*u/(1 + 3*v)", "2.02*u/(1 + 2.98*v)")
    assert not eq("2*u/(1 + 3*v)", "2*u/(1 + 4*v)")
    assert eq("exp(-0.5*u)", "exp(-0.51*u)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("JUDGE TESTS PASSED")
