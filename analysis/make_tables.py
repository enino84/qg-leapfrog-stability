"""
make_tables.py, regenerate the LaTeX rows of Tables 5 and 6 from the .npz runs.

Single source of truth: the diagnostics are imported from uncertainty.py, so the
statistical summary and the alpha sweep can no longer drift apart.

Run from the ROOT of the repository (where the .npz files live):

    python analysis/make_tables.py

Add --compare to also print the values obtained when the wall ring is included
in the average, which is how the alpha-sweep figures were produced before this
script existed.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty import kinetic_energy, enstrophy


def enstrophy_with_walls(psi, h):
    """Domain average of (nabla^2 psi)^2 over ALL nodes, walls included.

    Kept only for provenance: at a wall node the 5-point stencil reduces to
    psi_adjacent / h^2, which is O(1e5) here, so including that ring inflates
    the mean and its variance substantially. Not used for published numbers.
    """
    lap = (np.roll(psi, -1, 1) + np.roll(psi, 1, 1) +
           np.roll(psi, -1, 2) + np.roll(psi, 1, 2) - 4 * psi) / h ** 2
    return np.mean(lap ** 2, axis=(1, 2))


def load(path):
    d = np.load(path)
    return d['psi'].astype(float)


def stats(psi, h, walls=False):
    ke = kinetic_energy(psi)
    en = enstrophy_with_walls(psi, h) if walls else enstrophy(psi, h)
    return ke.mean(), ke.std(), en.mean(), en.std(), len(ke)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rk4',    default='qg_rk4_ref.npz')
    p.add_argument('--lf-raw', default='qg_lf_raw.npz')
    p.add_argument('--a005',   default='qg_lf_ra_a005.npz')
    p.add_argument('--a010',   default='qg_lf_ra.npz')
    p.add_argument('--a020',   default='qg_lf_ra_a020.npz')
    p.add_argument('--a040',   default='qg_lf_ra_a040.npz')
    p.add_argument('--compare', action='store_true',
                   help='also report the wall-inclusive average')
    a = p.parse_args()

    missing = [f for f in (a.rk4, a.lf_raw, a.a005, a.a010, a.a020, a.a040)
               if not os.path.exists(f)]
    if missing:
        sys.exit('Missing run files: ' + ', '.join(missing))

    psi_rk4 = load(a.rk4)
    h = 1.0 / (psi_rk4.shape[2] - 1)
    print(f"grid {psi_rk4.shape[1]}x{psi_rk4.shape[2]}, h = {h:.5f}, "
          f"{psi_rk4.shape[0]} records\n")

    for walls in ([False, True] if a.compare else [False]):
        tag = 'walls included (NOT for publication)' if walls else 'interior nodes'
        print(f"{'=' * 70}\n  Enstrophy averaged over {tag}\n{'=' * 70}")

        ke0, ske0, en0, sen0, n = stats(psi_rk4, h, walls)
        rows = {'RK4': (ke0, ske0, en0, sen0)}
        for label, f in (('LF+RAW', a.lf_raw), ('0.05', a.a005),
                         ('0.10', a.a010), ('0.20', a.a020), ('0.40', a.a040)):
            rows[label] = stats(load(f), h, walls)[:4]

        print("\n--- Table 5 rows ---")
        ra, raw = rows['0.10'], rows['LF+RAW']
        print(r"Mean KE $\psibar$ &")
        print(f"  ${ke0:.2f}$ & ${ra[0]:.2f}$ & ${(ra[0]-ke0)/ke0*100:+.0f}\\%$ "
              f"& ${raw[0]:.2f}$ & ${(raw[0]-ke0)/ke0*100:+.0f}\\%$ \\\\")
        print(r"Std KE &")
        print(f"  ${ske0:.3f}$ & ${ra[1]:.3f}$ & -- & ${raw[1]:.3f}$ & -- \\\\")
        print(r"Mean Enst.\ ($\times10^9$) &")
        print(f"  ${en0/1e9:.3f}$ & ${ra[2]/1e9:.3f}$ "
              f"& ${(ra[2]-en0)/en0*100:+.0f}\\%$ & ${raw[2]/1e9:.3f}$ "
              f"& ${(raw[2]-en0)/en0*100:+.0f}\\%$ \\\\")
        print(r"Std Enst.\ ($\times10^8$) &")
        print(f"  ${sen0/1e8:.3f}$ & ${ra[3]/1e8:.3f}$ & -- "
              f"& ${raw[3]/1e8:.3f}$ & -- \\\\")

        print("\n--- Table 6 rows ---")
        print(f"RK4 & ${ke0:.2f}$ & -- & ${en0/1e9:.3f}$ & -- "
              f"& ${ske0:.3f}$ & ${sen0/1e8:.2f}$ \\\\")
        for alpha in ('0.05', '0.10', '0.20', '0.40'):
            k, sk, e, se = rows[alpha]
            print(f"${alpha}$ & ${k:.2f}$ & ${(k-ke0)/ke0*100:+.1f}\\%$ "
                  f"& ${e/1e9:.3f}$ & ${(e-en0)/en0*100:+.1f}\\%$ "
                  f"& ${sk:.3f}$ & ${se/1e8:.2f}$ \\\\")
        print(f"\ncaption: Enst$_{{\\rm RK4}}={en0/1e9:.3f}\\times10^9$, "
              f"KE$_{{\\rm RK4}}={ke0:.2f}$\n")


if __name__ == '__main__':
    main()
