# -*- coding: utf-8 -*-
"""State-vector API for data assimilation: pack/unpack, propagate, masks.

The model class in :mod:`qgpython.model` is run-oriented (``set_ic`` + ``run``
with output to netCDF). Data-assimilation frameworks need a stateless
"propagate this vector over this interval" call. This module provides it
without touching the model code.

State layout: ``x = [q.ravel(), psi.ravel()]`` with ``q`` and ``psi`` on the
``(m, n)`` grid (row-major). ``psi`` is a diagnostic of ``q`` (Helmholtz
inverse); it is carried in the vector for convenience and recomputed from
``q`` at the start of every propagation.
"""
from __future__ import annotations

import numpy as np

from .model import QGModel, QGParams


def n_state(prm: QGParams) -> int:
    return 2 * prm.m * prm.n


def field_size(prm: QGParams) -> int:
    return prm.m * prm.n


def pack(q: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(q).ravel(), np.asarray(psi).ravel()])


def unpack(model: QGModel, x: np.ndarray):
    m, n = model.prm.m, model.prm.n
    fs = m * n
    return x[:fs].reshape(m, n), x[fs:2 * fs].reshape(m, n)


def psi_from_q(model: QGModel, x: np.ndarray) -> np.ndarray:
    """Return a copy of ``x`` whose psi block is the Helmholtz inverse of its q block."""
    q, _ = unpack(model, x)
    return pack(q, model._calc_psi(q))


def interior_mask(prm: QGParams) -> np.ndarray:
    """Boolean mask over one field: True off the Dirichlet boundary (the estimated points)."""
    m, n = prm.m, prm.n
    msk = np.ones((m, n), dtype=bool)
    msk[0, :] = msk[-1, :] = False
    msk[:, 0] = msk[:, -1] = False
    return msk.ravel()


def propagate(model: QGModel, x0: np.ndarray, T, just_final_state: bool = True) -> np.ndarray:
    """Propagate ``x0 = [q; psi]`` from ``T[0]`` to ``T[-1]`` with the model's scheme and ``dt``.

    Only the end points of ``T`` are used; the number of steps is
    ``round((T[-1] - T[0]) / dt)``. Multistep memory (leapfrog) is reset, so
    consecutive calls are independent: this is what an ensemble forecast needs.
    """
    if not just_final_state:
        raise NotImplementedError("trajectory output: use QGModel.run() instead")
    q, _ = unpack(model, x0)
    model.q = q.copy()
    model.psi = model._calc_psi(model.q)
    model.t = float(T[0])
    model._q_prev = None
    n_steps = max(1, int(round((float(T[-1]) - float(T[0])) / model.prm.dt)))
    for _ in range(n_steps):
        model._do_step()
    return pack(model.q, model.psi)
