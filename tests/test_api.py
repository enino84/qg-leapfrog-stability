import numpy as np
from qgpython import QGModel, api, presets


def test_propagate_shapes_and_determinism():
    prm = presets.sakov_oke_stationary(mrefin=4)
    model = QGModel(prm); model.set_ic("zero")
    x = api.pack(model.q, model.psi)
    assert x.size == api.n_state(prm)
    y1 = api.propagate(model, x, [0.0, 50.0]); y2 = api.propagate(model, x, [0.0, 50.0])
    assert np.allclose(y1, y2) and np.isfinite(y1).all()


def test_psi_is_diagnostic_of_q():
    prm = presets.sakov_oke_stationary(mrefin=4)
    model = QGModel(prm); model.set_ic("zero")
    x = api.propagate(model, api.pack(model.q, model.psi), [0.0, 200.0])
    q, psi = api.unpack(model, x)
    assert np.allclose(psi, model._calc_psi(q))


def test_interior_mask():
    prm = presets.sakov_oke_stationary(mrefin=4)
    msk = api.interior_mask(prm)
    assert msk.sum() == (prm.m - 2) * (prm.n - 2)


def test_stationary_preset_does_not_grow_unboundedly():
    """Short proxy of the measured behaviour: at mrefin 4 the q std after 1500 units is
    within a factor of two of its value after 750 (no runaway intensification)."""
    prm = presets.sakov_oke_stationary(mrefin=4)
    model = QGModel(prm); model.set_ic("zero")
    x = api.propagate(model, api.pack(model.q, model.psi), [0.0, 750.0]); s1 = api.unpack(model, x)[0].std()
    x = api.propagate(model, x, [750.0, 1500.0]); s2 = api.unpack(model, x)[0].std()
    assert s2 < 2.0 * s1 + 1e-9
