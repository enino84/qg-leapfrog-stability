# -*- coding: utf-8 -*-
"""Named configurations whose behaviour is documented and tested.

``sakov_oke_stationary``
    The regime used for data-assimilation testbeds: dissipation ten times
    the QG-C default (``rkh2 = 1e-11``), RK4, ``dt = 0.3125``. With the
    default ``rkh2 = 1e-12`` the flow keeps intensifying (the std of q is
    still rising at t = 8000 and beyond) and there is no stationary regime
    to draw a climatology from; at ``1e-11`` the std of q settles (1.107e4
    at t = 20000 against 1.222e4 at t = 60000). Sakov & Oke (2008) make the
    same adjustment. Measured on this implementation (Sept 2026): no
    forecast blow-up in more than 500 cycled runs of 60-120 cycles at
    ``mrefin = 5``.

``qgc_default``
    The parameters of the original QG-C model, for reproducing the article.
"""
from __future__ import annotations

from .model import QGParams


def sakov_oke_stationary(mrefin: int = 5, dt: float = 0.3125, scheme: str = "rk4", **overrides) -> QGParams:
    kw = dict(mrefin=mrefin, rkh2=1.0e-11, scheme=scheme, dt=dt, verbose=False, save_q=False)
    kw.update(overrides)
    return QGParams(**kw)


def qgc_default(mrefin: int = 7, **overrides) -> QGParams:
    kw = dict(mrefin=mrefin, verbose=False)
    kw.update(overrides)
    return QGParams(**kw)


def spinup_time(prm: QGParams) -> float:
    """Spin-up after which the stationary preset is statistically settled (measured at mrefin 5)."""
    return 20000.0


def stationary_model(mrefin: int = 5, **overrides):
    """The recommended entry point for data assimilation: a ``QGModel`` in the
    stationary regime, initialised at rest (call ``api.propagate`` for the spin-up)."""
    from .model import QGModel
    model = QGModel(sakov_oke_stationary(mrefin=mrefin, **overrides))
    model.set_ic("zero")
    return model
