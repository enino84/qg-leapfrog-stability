"""qgpython: a 1.5-layer quasi-geostrophic model (Python port of QG-C).

    from qgpython import stationary_model, api
    model = stationary_model(mrefin=5)                 # rkh2 = 1e-11, RK4, dt = 0.3125
    x = api.propagate(model, api.pack(model.q, model.psi), [0.0, 20000.0])

The PyPI distribution is ``qg-python``; ``import qg_model`` still works.
"""
from .model import QGModel, QGParams, HelmholtzSolver, arakawa, laplacian, main, __version__
from . import api, presets
from .presets import stationary_model, sakov_oke_stationary

__all__ = ["QGModel", "QGParams", "HelmholtzSolver", "arakawa", "laplacian", "main", "api", "presets",
           "stationary_model", "sakov_oke_stationary", "__version__"]
