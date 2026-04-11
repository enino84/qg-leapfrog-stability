"""
QG-Python: 1.5-layer Quasi-Geostrophic Model
=============================================
Faithful Python port of the QG-C model (Pavel Sakov, Bureau of Meteorology).

PHYSICS:
  Potential vorticity equation:
    dq/dt = -r·J(psi,q) - rkb·zeta + rkh·nabla²zeta - rkh2·nabla⁴zeta + curl(tau) - dpsi/dy
    q = nabla²psi - F·psi   (potential vorticity)
  with Dirichlet boundary conditions: psi = 0

TIME-STEPPING SCHEMES:
  order1   - explicit Euler (1st order)
  order2   - midpoint Runge-Kutta (2nd order)
  rk4      - classical Runge-Kutta (4th order)  [default]
  dp5      - Dormand-Prince (5th order)
  leapfrog - Leapfrog with Robert-Asselin (RA) or Robert-Asselin-Williams (RAW) filter

LEAPFROG FILTER OPTIONS:
  --ra-alpha FLOAT   Robert-Asselin filter parameter (default: 0.1)
  --raw              Use RAW (Robert-Asselin-Williams) filter instead of RA
                     RAW is 3rd-order accurate in the filter correction vs
                     1st-order for RA, and reduces spurious amplitude damping.

OUTPUT:
  If netCDF4 is installed  -> .nc  (compatible with the original QG-C)
  Otherwise                -> .npz (NumPy compressed, readable with np.load)

USAGE:
  python qg_model.py                                        # default parameters
  python qg_model.py my_config.prm                          # parameter file
  python qg_model.py --tend 1000 --scheme dp5               # command-line options
  python qg_model.py --scheme leapfrog --ra-alpha 0.1       # Leapfrog + RA filter
  python qg_model.py --scheme leapfrog --raw --ra-alpha 0.1 # Leapfrog + RAW filter

  From Python:
    from qg_model import QGModel, QGParams
    prm = QGParams(tend=5000, dtout=50, scheme='leapfrog',
                   ra_alpha=0.1, raw_filter=False)
    m = QGModel(prm)
    psi, q = m.run('restart', fname='qg_fourier.npz')

DEPENDENCIES:
  numpy, scipy   (required)
  netCDF4        (optional - for .nc output)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from dataclasses import dataclass, field
from typing import Optional
import time
import os

try:
    import netCDF4 as _nc
    HAS_NC = True
except ImportError:
    HAS_NC = False


# ============================================================================
# Parameters
# ============================================================================

@dataclass
class QGParams:
    # --- Grid ---
    mrefin: int   = 7
    nx1:    int   = 3
    ny1:    int   = 3
    lx:     float = 1.0

    # --- Physics ---
    rkb:    float = 3.0e-4
    rkh:    float = -5.0e-8
    rkh2:   float = 1.0e-12
    f:      float = 1600.0
    r:      float = 1.0e-5
    a:      float = 6.29
    k:      float = 1.0

    # --- Time integration ---
    scheme:     str   = 'rk4'
    dt:         float = 1.0
    tend:       float = 50000.0
    dtout:      float = 20.0

    # --- Leapfrog filter options ---
    ra_alpha:   float = 0.1    # Robert-Asselin filter strength (0 = off)
    raw_filter: bool  = False  # True -> use RAW (Williams 2009) instead of RA

    # --- I/O ---
    outfname: str  = 'qg.nc'
    save_q:   bool = False
    verbose:  bool = True

    m: int = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        ksq   = 2 ** (self.mrefin - 1)
        self.m = self.ny1 * ksq + 1
        self.n = self.nx1 * ksq + 1

    @classmethod
    def from_file(cls, fname: str) -> 'QGParams':
        prm = cls()
        _conv = {
            'mrefin':     ('mrefin',     int),
            'nx1':        ('nx1',        int),
            'ny1':        ('ny1',        int),
            'lx':         ('lx',         float),
            'rkb':        ('rkb',        float),
            'rkh':        ('rkh',        float),
            'rkh2':       ('rkh2',       float),
            'f':          ('f',          float),
            'r':          ('r',          float),
            'a':          ('a',          float),
            'k':          ('k',          float),
            'scheme':     ('scheme',     str),
            'dt':         ('dt',         float),
            'tend':       ('tend',       float),
            'dtout':      ('dtout',      float),
            'ra_alpha':   ('ra_alpha',   float),
            'raw_filter': ('raw_filter', lambda x: x.strip() in ('1','yes','true','Yes')),
            'outfname':   ('outfname',   str),
            'save_q':     ('save_q',     lambda x: x.strip() in ('1','yes','true','Yes')),
        }
        with open(fname) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip().lower()
                val = val.strip()
                if key in _conv:
                    attr, conv = _conv[key]
                    setattr(prm, attr, conv(val))
        prm.__post_init__()
        return prm

    def describe(self):
        print(f"  grid        : {self.m} x {self.n}  (mrefin={self.mrefin}, nx1={self.nx1}, ny1={self.ny1})")
        print(f"  lx          : {self.lx}")
        print(f"  rkb/rkh/rkh2: {self.rkb} / {self.rkh} / {self.rkh2}")
        print(f"  F, r, A, k  : {self.f}, {self.r}, {self.a}, {self.k}")
        print(f"  scheme      : {self.scheme}")
        if self.scheme == 'leapfrog':
            filt = 'RAW' if self.raw_filter else 'RA'
            print(f"  LF filter   : {filt}  alpha={self.ra_alpha}")
        print(f"  dt / tend   : {self.dt} / {self.tend}")
        print(f"  dtout       : {self.dtout}")
        print(f"  output      : {self.outfname}  (save_q={self.save_q})")


# ============================================================================
# Numerical operators
# ============================================================================

def arakawa(h: float, psi: np.ndarray, q: np.ndarray) -> np.ndarray:
    c = 12.0 * h * h
    J = np.zeros_like(psi)

    pjm = psi[:-2,  1:-1]; pjp = psi[2:,  1:-1]
    pim = psi[1:-1, :-2];  pip = psi[1:-1, 2:]
    pmm = psi[:-2,  :-2];  pmp = psi[:-2,  2:]
    ppm = psi[2:,   :-2];  ppp = psi[2:,   2:]

    qjm = q[:-2,  1:-1]; qjp = q[2:,  1:-1]
    qim = q[1:-1, :-2];  qip = q[1:-1, 2:]
    qmm = q[:-2,  :-2];  qmp = q[:-2,  2:]
    qpm = q[2:,   :-2];  qpp = q[2:,   2:]

    J[1:-1, 1:-1] = (
          (pjm   - pim  ) * qmm
        + (pmm   + pjm   - ppm   - pjp  ) * qim
        + (pim   - pjp  ) * qpm
        + (pmp   + pip   - pmm   - pim  ) * qjm
        + (pim   + ppm   - pip   - ppp  ) * qjp
        + (pip   - pjm  ) * qmp
        + (pjp   + ppp   - pjm   - pmp  ) * qip
        + (pjp   - pip  ) * qpp
    ) / c
    return J


def laplacian(h: float, A: np.ndarray) -> np.ndarray:
    h2 = 1.0 / (h * h)
    L  = np.zeros_like(A)
    L[1:-1, 1:-1] = (
        A[:-2, 1:-1] + A[2:, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:]
        - 4.0 * A[1:-1, 1:-1]
    ) * h2
    return L


# ============================================================================
# Helmholtz solver
# ============================================================================

def _build_helmholtz_matrix(m: int, n: int, h: float, F: float):
    mn  = m * n
    h2  = 1.0 / (h * h)

    diag = np.full(mn, -(4.0 * h2 + F))
    off  = np.full(mn,  h2)

    for j in range(m):
        for i in range(n):
            if j == 0 or j == m-1 or i == 0 or i == n-1:
                diag[j*n + i] = 1.0

    off_e = off.copy()
    off_w = off.copy()
    for j in range(m):
        k = j*n + (n-1)
        off_e[k] = 0.0
        off_w[k] = 0.0

    border = np.zeros(mn, dtype=bool)
    border[:n]     = True
    border[-n:]    = True
    border[::n]    = True
    border[n-1::n] = True

    off_e[border] = 0.0
    off_w[border] = 0.0
    off_n = off.copy(); off_n[border] = 0.0
    off_s = off.copy(); off_s[border] = 0.0

    A = diags(
        [off_s[:mn-n], off_w[:mn-1], diag, off_e[:mn-1], off_n[:mn-n]],
        [-n, -1, 0, 1, n],
        shape=(mn, mn),
        format='csr'
    )
    return A


class HelmholtzSolver:
    def __init__(self, m: int, n: int, h: float, F: float):
        self.m = m
        self.n = n
        A = _build_helmholtz_matrix(m, n, h, F)
        self._lu = factorized(A)

    def solve(self, q: np.ndarray) -> np.ndarray:
        rhs = q.ravel().copy()
        m, n = self.m, self.n
        mn   = m * n
        border_idx = (
            list(range(n))
          + list(range(mn-n, mn))
          + list(range(0, mn, n))
          + list(range(n-1, mn, n))
        )
        rhs[border_idx] = 0.0
        return self._lu(rhs).reshape(m, n)


# ============================================================================
# QG Model
# ============================================================================

class QGModel:

    def __init__(self, prm: QGParams):
        self.prm  = prm
        self.m    = prm.m
        self.n    = prm.n
        self.h    = prm.lx / (prm.n - 1)
        self.t    = 0.0

        self.psi  = np.zeros((prm.m, prm.n))
        self.q    = np.zeros((prm.m, prm.n))

        # Leapfrog requires q at the previous time level
        self._q_prev = None   # set to None; initialised at first LF step

        i_idx      = np.arange(prm.n)
        tmp        = np.sin(prm.k * 2.0 * np.pi * i_idx / prm.n)
        self.curlt = -prm.a * tmp * np.abs(tmp)

        if prm.verbose:
            print(f"  Building Helmholtz solver ({prm.m}x{prm.n})...", end=' ', flush=True)
        t0 = time.time()
        self.helm = HelmholtzSolver(prm.m, prm.n, self.h, prm.f)
        if prm.verbose:
            print(f"done ({time.time()-t0:.2f}s)")

        self._out_t   = []
        self._out_psi = []
        self._out_q   = []

    def _calc_psi(self, q):
        return self.helm.solve(q)

    def _rhs(self, q, psi):
        prm = self.prm
        h   = self.h

        J      = arakawa(h, psi, q)
        zeta   = laplacian(h, psi)
        zeta2  = laplacian(h, zeta)
        zeta4  = laplacian(h, zeta2)

        dpsi_dy             = np.zeros_like(psi)
        dpsi_dy[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)

        F = np.zeros_like(q)
        F[1:-1, 1:-1] = (
            - prm.r    * J[1:-1, 1:-1]
            - prm.rkb  * zeta[1:-1, 1:-1]
            + prm.rkh  * zeta2[1:-1, 1:-1]
            - prm.rkh2 * zeta4[1:-1, 1:-1]
            + self.curlt[1:-1]
            - dpsi_dy[1:-1, 1:-1]
        )
        return F

    # ── existing schemes ─────────────────────────────────────────────────────

    def _step_order1(self):
        dt   = self.prm.dt
        psi1 = self._calc_psi(self.q)
        k1   = self._rhs(self.q, psi1)
        self.q  += dt * k1
        self.psi = self._calc_psi(self.q)
        self.t  += dt

    def _step_order2(self):
        dt   = self.prm.dt
        q0   = self.q.copy()
        psi0 = self._calc_psi(q0)
        k1   = self._rhs(q0, psi0)
        qm   = q0 + 0.5 * dt * k1
        psim = self._calc_psi(qm)
        k2   = self._rhs(qm, psim)
        self.q   = q0 + dt * k2
        self.psi = self._calc_psi(self.q)
        self.t  += dt

    def _step_rk4(self):
        dt = self.prm.dt
        q0 = self.q.copy()

        psi1 = self._calc_psi(q0);               k1 = self._rhs(q0,            psi1)
        psi2 = self._calc_psi(q0 + 0.5*dt*k1);  k2 = self._rhs(q0+0.5*dt*k1, psi2)
        psi3 = self._calc_psi(q0 + 0.5*dt*k2);  k3 = self._rhs(q0+0.5*dt*k2, psi3)
        psi4 = self._calc_psi(q0 + dt*k3);       k4 = self._rhs(q0+dt*k3,     psi4)

        self.q   = q0 + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        self.psi = self._calc_psi(self.q)
        self.t  += dt

    def _step_dp5(self):
        dt = self.prm.dt
        q0 = self.q.copy()

        def f(q): return self._rhs(q, self._calc_psi(q))

        k1 = f(q0)
        k2 = f(q0 + dt * 0.2 * k1)
        k3 = f(q0 + dt * (3/40*k1   + 9/40*k2))
        k4 = f(q0 + dt * (44/45*k1  - 56/15*k2  + 32/9*k3))
        k5 = f(q0 + dt * (19372/6561*k1 - 25360/2187*k2
                          + 64448/6561*k3 - 212/729*k4))
        k6 = f(q0 + dt * (9017/3168*k1  - 355/33*k2
                          + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))

        self.q   = q0 + dt * (35/384*k1  + 500/1113*k3 + 125/192*k4
                               - 2187/6784*k5 + 11/84*k6)
        self.psi = self._calc_psi(self.q)
        self.t  += dt

    # ── Leapfrog + Robert-Asselin / RAW filter ────────────────────────────────
    #
    # Standard Leapfrog (2nd-order):
    #   q^{n+1} = q^{n-1} + 2*dt * F(q^n, psi^n)
    #
    # Robert-Asselin (RA) filter applied to q^n after each step:
    #   q^n <- q^n + alpha/2 * (q^{n-1} - 2*q^n + q^{n+1})
    #
    # Robert-Asselin-Williams (RAW) filter (Williams 2009):
    #   Splits the RA correction between q^n and q^{n+1} using parameter beta,
    #   preserving 2nd-order accuracy of the physical mode while still
    #   suppressing the computational mode.  With beta=0.5 (default):
    #   q^n    <- q^n    + alpha*(1-beta)/2 * (q^{n-1} - 2*q^n + q^{n+1})
    #   q^{n+1}<- q^{n+1} - alpha*beta/2   * (q^{n-1} - 2*q^n + q^{n+1})
    #
    # First leapfrog step (n=0) uses a single RK4 step to obtain q^1,
    # ensuring a clean start without the computational mode.

    def _step_leapfrog(self):
        dt    = self.prm.dt
        alpha = self.prm.ra_alpha
        raw   = self.prm.raw_filter
        beta  = 0.5  # RAW splitting parameter (Williams 2009 recommends 0.5)

        # ── bootstrap: first step uses RK4 to avoid exciting computational mode
        if self._q_prev is None:
            self._q_prev = self.q.copy()
            self._step_rk4()   # advances self.q and self.t by dt
            return

        # ── standard leapfrog leap
        q_nm1 = self._q_prev          # q^{n-1}
        q_n   = self.q.copy()         # q^n
        psi_n = self._calc_psi(q_n)   # psi^n  (one Helmholtz solve)

        F_n   = self._rhs(q_n, psi_n)
        q_np1 = q_nm1 + 2.0 * dt * F_n   # leapfrog step: q^{n+1}

        # ── filter correction
        delta = q_nm1 - 2.0 * q_n + q_np1   # 2nd-difference (centred on n)

        if raw:
            # RAW filter (Williams 2009) — less dissipative than plain RA
            q_n_filtered   = q_n   + 0.5 * alpha * (1.0 - beta) * delta
            q_np1_filtered = q_np1 - 0.5 * alpha * beta          * delta
        else:
            # Classic Robert-Asselin filter
            q_n_filtered   = q_n   + 0.5 * alpha * delta
            q_np1_filtered = q_np1   # RA does not modify q^{n+1}

        # ── advance state
        self._q_prev = q_n_filtered        # filtered q^n becomes q^{n-1}
        self.q       = q_np1_filtered      # filtered q^{n+1} is new state
        self.psi     = self._calc_psi(self.q)
        self.t      += dt

    # ── dispatcher ───────────────────────────────────────────────────────────

    def _do_step(self):
        s = self.prm.scheme
        if   s == 'order1':   self._step_order1()
        elif s == 'order2':   self._step_order2()
        elif s == 'rk4':      self._step_rk4()
        elif s == 'dp5':      self._step_dp5()
        elif s == 'leapfrog': self._step_leapfrog()
        else: raise ValueError(f"Unknown scheme: '{s}'")

        if not np.isfinite(self.psi).all():
            raise RuntimeError(
                f"\n\n  *** INSTABILITY (NaN/Inf) at t={self.t:.1f} ***\n"
                f"  Suggestions:\n"
                f"    1. Reduce dt          (current={self.prm.dt})  -> try dt <= {self.prm.dt/4:.4g}\n"
                f"    2. Increase RA filter: --ra-alpha 0.2\n"
                f"    3. Try RAW filter:     --raw\n"
                f"    4. Disable neg. fric.: --rkh 0\n"
                f"    5. More dissipation:   --rkh2 1e-11\n"
            )

    def _q_from_psi(self):
        self.q = laplacian(self.h, self.psi) - self.prm.f * self.psi

    # ── initial conditions ───────────────────────────────────────────────────

    def set_ic(self, kind: str = 'zero', **kwargs):
        m, n = self.m, self.n
        self.psi     = np.zeros((m, n))
        self.q       = np.zeros((m, n))
        self.t       = 0.0
        self._q_prev = None   # reset leapfrog memory

        if kind == 'zero':
            pass

        elif kind == 'fourier':
            amplitude = kwargs.get('amplitude', 1.0)
            kmax      = kwargs.get('kmax', 5)
            seed      = kwargs.get('seed', 42)
            rng       = np.random.default_rng(seed)

            j_idx = np.arange(m)
            i_idx = np.arange(n)
            psi   = np.zeros((m, n))
            for kj in range(1, kmax + 1):
                for ki in range(1, kmax + 1):
                    amp  = rng.uniform(-1, 1) * amplitude / (kj**2 + ki**2)
                    psi += amp * np.outer(
                        np.sin(kj * np.pi * j_idx / (m - 1)),
                        np.sin(ki * np.pi * i_idx / (n - 1))
                    )
            self.psi = psi
            self._q_from_psi()

        elif kind == 'vortex':
            vortices = kwargs.get('vortices', [
                {'x': 0.5, 'y': 0.5, 'r': 0.1, 'amp': 1.0}
            ])
            lx = self.prm.lx
            x  = np.linspace(0, lx, n)
            y  = np.linspace(0, lx, m)
            X, Y = np.meshgrid(x, y)

            psi = np.zeros((m, n))
            for v in vortices:
                x0  = v.get('x', 0.5) * lx
                y0  = v.get('y', 0.5) * lx
                r   = v.get('r', 0.1) * lx
                amp = v.get('amp', 1.0)
                psi += amp * np.exp(-((X-x0)**2 + (Y-y0)**2) / (2.0 * r**2))

            psi[0, :]  = 0.0
            psi[-1, :] = 0.0
            psi[:, 0]  = 0.0
            psi[:, -1] = 0.0
            self.psi = psi
            self._q_from_psi()

        elif kind == 'restart':
            fname  = kwargs.get('fname')
            record = kwargs.get('record', -1)
            if fname is None:
                raise ValueError("set_ic('restart', ...) requires fname=<path>")

            if fname.endswith('.npz'):
                data     = np.load(fname)
                self.psi = data['psi'][record].astype(float)
                if 'q' in data:
                    self.q = data['q'][record].astype(float)
                else:
                    self._q_from_psi()
                if 't' in data:
                    self.t = float(data['t'][record])

            elif fname.endswith('.nc') or fname.endswith('.nc4'):
                if not HAS_NC:
                    raise ImportError("netCDF4 is not installed; use a .npz file instead")
                ds = _nc.Dataset(fname, 'r')
                self.psi = np.array(ds.variables['psi'][record], dtype=float)
                if 'q' in ds.variables:
                    self.q = np.array(ds.variables['q'][record], dtype=float)
                else:
                    self._q_from_psi()
                if 't' in ds.variables:
                    self.t = float(ds.variables['t'][record])
                ds.close()
            else:
                raise ValueError(f"Unrecognised file format: {fname!r}  (use .npz or .nc)")

            if self.prm.verbose:
                print(f"  restart from '{fname}', record={record}, t={self.t:.1f}")

        else:
            raise ValueError(
                f"Unknown IC kind='{kind}'. "
                "Valid: 'zero', 'fourier', 'vortex', 'restart'"
            )

    # ── output helpers ───────────────────────────────────────────────────────

    def _save_record(self):
        self._out_t.append(float(self.t))
        self._out_psi.append(self.psi.copy().astype(np.float32))
        if self.prm.save_q:
            self._out_q.append(self.q.copy().astype(np.float32))

    def _write_output(self):
        prm   = self.prm
        fname = prm.outfname
        t_arr = np.array(self._out_t)
        p_arr = np.stack(self._out_psi)

        if HAS_NC:
            if not fname.endswith('.nc'):
                fname += '.nc'
            ds = _nc.Dataset(fname, 'w', format='NETCDF4')
            ds.setncattr('dt',         prm.dt);       ds.setncattr('lx',    prm.lx)
            ds.setncattr('rkb',        prm.rkb);      ds.setncattr('rkh',   prm.rkh)
            ds.setncattr('rkh2',       prm.rkh2);     ds.setncattr('F',     prm.f)
            ds.setncattr('r',          prm.r);        ds.setncattr('A',     prm.a)
            ds.setncattr('k',          prm.k);        ds.setncattr('scheme',prm.scheme)
            ds.setncattr('ra_alpha',   prm.ra_alpha)
            ds.setncattr('raw_filter', int(prm.raw_filter))
            ds.setncattr('comment',    'qg-python output')
            ds.createDimension('record', None)
            ds.createDimension('j', prm.m)
            ds.createDimension('i', prm.n)
            v = ds.createVariable('t',   'f8', ('record',));        v[:] = t_arr
            v = ds.createVariable('psi', 'f4', ('record','j','i')); v[:] = p_arr
            if prm.save_q:
                v = ds.createVariable('q', 'f4', ('record','j','i'))
                v[:] = np.stack(self._out_q)
            ds.close()
        else:
            if not fname.endswith('.npz'):
                fname = os.path.splitext(fname)[0] + '.npz'
            kw = dict(t=t_arr, psi=p_arr,
                      lx=prm.lx, dt=prm.dt, rkb=prm.rkb,
                      rkh=prm.rkh, rkh2=prm.rkh2, F=prm.f,
                      r=prm.r, A=prm.a, k=prm.k,
                      ra_alpha=prm.ra_alpha,
                      raw_filter=int(prm.raw_filter))
            if prm.save_q:
                kw['q'] = np.stack(self._out_q)
            np.savez_compressed(fname, **kw)

        if prm.verbose:
            print(f"\n  Output saved: {fname}  ({len(t_arr)} records)")

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self, ic: str = 'zero', **ic_kwargs):
        prm = self.prm
        self.set_ic(ic, **ic_kwargs)
        self._save_record()

        nstep = int((prm.tend - self.t) / prm.dt)
        dnout = max(1, int(prm.dtout / prm.dt))

        if prm.verbose:
            print(f"  nstep = {nstep}, nrecord ~= {nstep // dnout}")
            print(f"  integrating [{prm.scheme}]:", end='', flush=True)

        t0 = time.time()
        for step in range(1, nstep + 1):
            self._do_step()
            if step % dnout == 0:
                self._save_record()
                if prm.verbose:
                    print(':', end='', flush=True)

        elapsed = time.time() - t0
        if prm.verbose:
            print(f"\n  done in {elapsed:.1f}s  ({nstep/elapsed:.0f} steps/s)")

        self._write_output()
        return self.psi, self.q


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='QG-Python: 1.5-layer quasi-geostrophic model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples — restart from equilibrium state (recommended for analysis):
  python qg_model.py --scheme rk4      --dt 1 --tend 110000 --dtout 40 --ic restart --restart-file qg_fourier.npz --out qg_rk4_ref.npz
  python qg_model.py --scheme leapfrog --dt 1 --tend 110000 --dtout 40 --ic restart --restart-file qg_fourier.npz --ra-alpha 0.1 --out qg_lf_ra.npz
  python qg_model.py --scheme leapfrog --dt 1 --tend 110000 --dtout 40 --ic restart --restart-file qg_fourier.npz --ra-alpha 0.1 --raw --out qg_lf_raw.npz

Stability sweep (vary dt):
  python qg_model.py --scheme leapfrog --dt 0.5 --tend 110000 --dtout 40 --ic restart --restart-file qg_fourier.npz --ra-alpha 0.1 --out qg_lf_dt05.npz
  python qg_model.py --scheme leapfrog --dt 2.0 --tend 110000 --dtout 40 --ic restart --restart-file qg_fourier.npz --ra-alpha 0.1 --out qg_lf_dt2.npz
""")
    parser.add_argument('prmfile', nargs='?', help='parameter file (.prm)')

    # Time
    parser.add_argument('--scheme', choices=['order1','order2','rk4','dp5','leapfrog'])
    parser.add_argument('--tend',   type=float)
    parser.add_argument('--dt',     type=float)
    parser.add_argument('--dtout',  type=float)
    parser.add_argument('--out',    metavar='FILE')
    parser.add_argument('--save-q', action='store_true')
    parser.add_argument('--quiet',  action='store_true')

    # Grid
    parser.add_argument('--mrefin', type=int)
    parser.add_argument('--nx1',    type=int)
    parser.add_argument('--ny1',    type=int)

    # Physics
    parser.add_argument('--rkb',  type=float)
    parser.add_argument('--rkh',  type=float)
    parser.add_argument('--rkh2', type=float)
    parser.add_argument('--a',    type=float)
    parser.add_argument('--f',    type=float)
    parser.add_argument('--r',    type=float)

    # Leapfrog filter
    parser.add_argument('--ra-alpha', type=float, default=None,
        metavar='ALPHA',
        help='Robert-Asselin filter strength (default: 0.1). '
             'Used with --scheme leapfrog.')
    parser.add_argument('--raw', action='store_true',
        help='Use RAW (Robert-Asselin-Williams) filter instead of classic RA. '
             'Less dissipative; recommended for long runs.')

    # IC
    parser.add_argument('--ic', default='zero',
        choices=['zero','fourier','vortex','restart'])
    parser.add_argument('--amplitude',      type=float, default=1.0)
    parser.add_argument('--kmax',           type=int,   default=5)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--restart-file',   metavar='FILE')
    parser.add_argument('--restart-record', type=int,   default=-1)

    args = parser.parse_args()

    prm = QGParams.from_file(args.prmfile) if args.prmfile else QGParams()

    # Time
    if args.scheme:  prm.scheme   = args.scheme
    if args.tend:    prm.tend     = args.tend
    if args.dt:      prm.dt       = args.dt
    if args.dtout:   prm.dtout    = args.dtout
    if args.out:     prm.outfname = args.out
    if args.save_q:  prm.save_q   = True
    if args.quiet:   prm.verbose  = False

    # Grid
    if args.mrefin or args.nx1 or args.ny1:
        if args.mrefin: prm.mrefin = args.mrefin
        if args.nx1:    prm.nx1    = args.nx1
        if args.ny1:    prm.ny1    = args.ny1
        prm.__post_init__()

    # Physics
    if args.rkb  is not None: prm.rkb  = args.rkb
    if args.rkh  is not None: prm.rkh  = args.rkh
    if args.rkh2 is not None: prm.rkh2 = args.rkh2
    if args.a    is not None: prm.a    = args.a
    if args.f    is not None: prm.f    = args.f
    if args.r    is not None: prm.r    = args.r

    # Leapfrog filter
    if args.ra_alpha is not None: prm.ra_alpha   = args.ra_alpha
    if args.raw:                  prm.raw_filter = True

    print("  QG-Python")
    prm.describe()
    print(f"  netCDF4 available  : {HAS_NC}")
    print(f"  initial condition  : {args.ic}")

    ic_kwargs: dict = {}
    if args.ic == 'fourier':
        ic_kwargs = dict(amplitude=args.amplitude, kmax=args.kmax, seed=args.seed)
    elif args.ic == 'restart':
        if not args.restart_file:
            parser.error("--ic restart requires --restart-file FILE")
        ic_kwargs = dict(fname=args.restart_file, record=args.restart_record)

    model = QGModel(prm)
    model.run(ic=args.ic, **ic_kwargs)