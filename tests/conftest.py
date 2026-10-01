import numpy as np
import pytest
from pymatgen.core import Composition

MU = {'Li': -1.9, 'S': -4.1, 'Cl': -1.8}  # eV/atom of the pure elements

# Formation energies (eV/atom). Li3SCl lies below the Li2S-LiCl tie line, so it is the one stable
# ternary; Li6S2Cl2 is the same point in a larger cell, 30 meV/atom higher.
EXPLORED = {'Li3 S1 Cl1': -0.96, 'Li6 S2 Cl2': -0.93, 'Li4 S1 Cl2': -0.93, 'Li5 S2 Cl1': -0.90,
            'Li5 S1 Cl3': -0.95, 'Li7 S2 Cl3': -0.94, 'Li8 S3 Cl2': -0.92, 'Li7 S1 Cl5': -0.95}
REFERENCES = {'Li1': 0.0, 'S1': 0.0, 'Cl1': 0.0, 'Li2 S1': -0.90, 'Li1 Cl1': -1.00}
IONS = {'Li': 1, 'S': -2, 'Cl': -1}


def total_energy(formula, formation_energy):
    comp = Composition(formula)
    return sum(MU[str(el)] * n for el, n in comp.items()) + comp.num_atoms * formation_energy


@pytest.fixture
def li_s_cl():
    """A small Li-S-Cl phase field: compositions with total energies, the reference rows, the ions."""
    data = np.array([[f, total_energy(f, e)] for f, e in {**EXPLORED, **REFERENCES}.items()], dtype=object)
    return data, data[len(EXPLORED):], IONS
