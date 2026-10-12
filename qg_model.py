"""Backward-compatible module: ``from qg_model import QGModel, QGParams`` keeps working.

The implementation lives in the ``qgpython`` package since 1.1.0:

    from qgpython import QGModel, QGParams, stationary_model, api
"""
from qgpython.model import *                      # noqa: F401,F403
from qgpython.model import QGModel, QGParams, HelmholtzSolver, arakawa, laplacian, main, __version__  # noqa: F401

if __name__ == "__main__":
    main()
