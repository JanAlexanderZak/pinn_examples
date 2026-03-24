""" Generate dataset for thick-walled cylinder (Lame problem).

Governing ODE (Euler-Cauchy equation for radial displacement):
    d2u/dr2 + (1/r) du/dr - u/r2 = 0

where u(r) is the radial displacement.

Stress from displacement (plane stress):
    sigma_r     = E/(1-nu^2) * (du/dr + nu * u/r)
    sigma_theta = E/(1-nu^2) * (u/r + nu * du/dr)

Load cases:
    1. internal_pressure:  sigma_r(a) = -p_i, sigma_r(b) = 0
    2. external_pressure:  sigma_r(a) = 0,    sigma_r(b) = -p_o
    3. combined_pressure:  sigma_r(a) = -p_i, sigma_r(b) = -p_o

References:
    Lame, G. (1852).
        "Lecons sur la theorie mathematique de l'elasticite des corps solides."
        Bachelier, Paris.

    Timoshenko, S.P. & Goodier, J.N. (1970).
        "Theory of Elasticity." McGraw-Hill, 3rd edition.

    Boresi, A.P. & Schmidt, R.J. (2003).
        "Advanced Mechanics of Materials." Wiley, 6th edition.
"""
import os

import numpy as np
from pyDOE import lhs


def exact_solution(r, load_case, a=1.0, b=2.0, E=1.0, nu=0.3, p_i=1.0, p_o=0.5):
    """Compute exact displacement and stresses for a given load case.

    Args:
        r: Radial coordinates, shape (n,) or (n, 1).
        load_case: One of "internal_pressure", "external_pressure",
            "combined_pressure".
        a: Inner radius.
        b: Outer radius.
        E: Young's modulus.
        nu: Poisson's ratio.
        p_i: Internal pressure.
        p_o: External pressure.

    Returns:
        (u, sigma_r, sigma_theta, sigma_vm): Displacement and stress arrays.
    """
    r = np.asarray(r).flatten()
    a2, b2 = a**2, b**2
    diff = b2 - a2

    if load_case == "internal_pressure":
        sigma_r = (p_i * a2 / diff) * (1.0 - b2 / r**2)
        sigma_theta = (p_i * a2 / diff) * (1.0 + b2 / r**2)
        u = (p_i * a2 / (E * diff)) * ((1.0 - nu) * r + (1.0 + nu) * b2 / r)

    elif load_case == "external_pressure":
        sigma_r = (-p_o * b2 / diff) * (1.0 - a2 / r**2)
        sigma_theta = (-p_o * b2 / diff) * (1.0 + a2 / r**2)
        u = (-p_o * b2 / (E * diff)) * ((1.0 - nu) * r + (1.0 + nu) * a2 / r)

    elif load_case == "combined_pressure":
        # Superposition of internal and external
        sr_int = (p_i * a2 / diff) * (1.0 - b2 / r**2)
        st_int = (p_i * a2 / diff) * (1.0 + b2 / r**2)
        u_int = (p_i * a2 / (E * diff)) * ((1.0 - nu) * r + (1.0 + nu) * b2 / r)

        sr_ext = (-p_o * b2 / diff) * (1.0 - a2 / r**2)
        st_ext = (-p_o * b2 / diff) * (1.0 + a2 / r**2)
        u_ext = (-p_o * b2 / (E * diff)) * ((1.0 - nu) * r + (1.0 + nu) * a2 / r)

        sigma_r = sr_int + sr_ext
        sigma_theta = st_int + st_ext
        u = u_int + u_ext

    else:
        raise ValueError(
            f"Unknown load_case: '{load_case}'. "
            "Choose from: internal_pressure, external_pressure, combined_pressure"
        )

    # Von Mises equivalent stress (plane stress)
    sigma_vm = np.sqrt(
        sigma_r**2 - sigma_r * sigma_theta + sigma_theta**2
    )

    return u, sigma_r, sigma_theta, sigma_vm


def _build_bc_arrays(load_case, a=1.0, b=2.0, p_i=1.0, p_o=0.5):
    """Build boundary condition arrays for stress BCs.

    BCs are on sigma_r at inner and outer surfaces. The BC type indicator
    (column 1) is 0 for sigma_r boundary condition.

    Returns:
        x_train_BC: shape (n_bc, 2) -- col 0: radial position, col 1: BC type (0=sigma_r)
        y_train_BC: shape (n_bc, 1) -- target sigma_r value
    """
    if load_case == "internal_pressure":
        bc_x = np.array([
            [a, 0],  # sigma_r(a) = -p_i
            [b, 0],  # sigma_r(b) = 0
        ])
        bc_y = np.array([
            [-p_i],
            [0.0],
        ])

    elif load_case == "external_pressure":
        bc_x = np.array([
            [a, 0],  # sigma_r(a) = 0
            [b, 0],  # sigma_r(b) = -p_o
        ])
        bc_y = np.array([
            [0.0],
            [-p_o],
        ])

    elif load_case == "combined_pressure":
        bc_x = np.array([
            [a, 0],  # sigma_r(a) = -p_i
            [b, 0],  # sigma_r(b) = -p_o
        ])
        bc_y = np.array([
            [-p_i],
            [-p_o],
        ])

    else:
        raise ValueError(f"Unknown load_case: '{load_case}'.")

    return bc_x, bc_y


def generate_dataset(
    load_case="internal_pressure",
    path="src/thick_walled_cylinder/data",
    a=1.0,
    b=2.0,
    E=1.0,
    nu=0.3,
    p_i=1.0,
    p_o=0.5,
):
    """Generate and save training/evaluation data for the thick-walled cylinder.

    Args:
        load_case: One of "internal_pressure", "external_pressure",
            "combined_pressure".
        path: Directory to save data files.
        a, b: Inner and outer radii.
        E, nu: Material properties.
        p_i, p_o: Internal and external pressures.
    """
    # Fine evaluation grid
    r_star = np.linspace(a, b, 201).reshape(-1, 1)
    u_exact, sigma_r_exact, sigma_theta_exact, sigma_vm_exact = exact_solution(
        r_star, load_case, a=a, b=b, E=E, nu=nu, p_i=p_i, p_o=p_o,
    )

    # Collocation points via Latin Hypercube Sampling on [a, b]
    n_collocation = 400
    r_train = a + lhs(1, n_collocation) * (b - a)

    # Boundary conditions with type indicator
    bc_x, bc_y = _build_bc_arrays(load_case, a=a, b=b, p_i=p_i, p_o=p_o)

    # Replicate BC points for stable batching (only 2 unique BCs)
    n_replicate = 100
    x_train_BC = np.tile(bc_x, (n_replicate, 1))
    y_train_BC = np.tile(bc_y, (n_replicate, 1))

    # Save
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "r_train"), r_train)
    np.save(os.path.join(path, "x_train_BC"), x_train_BC)
    np.save(os.path.join(path, "y_train_BC"), y_train_BC)
    np.save(os.path.join(path, "r_star"), r_star)
    np.save(os.path.join(path, "u_exact"), u_exact)
    np.save(os.path.join(path, "sigma_r_exact"), sigma_r_exact)
    np.save(os.path.join(path, "sigma_theta_exact"), sigma_theta_exact)
    np.save(os.path.join(path, "sigma_vm_exact"), sigma_vm_exact)


if __name__ == "__main__":
    for case in ["internal_pressure", "external_pressure", "combined_pressure"]:
        generate_dataset(
            load_case=case,
            path=f"src/thick_walled_cylinder/data/{case}",
        )
