
"""
Local tangent-plane TE/TM transfer-matrix solver for a curved graded radome.

Model
-----
- Lower surface is a paraboloid: z = z_b + alpha * (x^2 + y^2)
- Radome has constant normal thickness t
- Relative permittivity varies radially:
      eps_r(r) = a_eps * r^2 + b_eps * r + c_eps
- Source is a Hertzian electric dipole at the origin, polarized along +y
- Upper and lower exterior media are free space by default
- The shell is modeled patch-by-patch using a local planar slab TE/TM TMM
- The transmitted field on the outer surface is converted to equivalent currents
  and propagated to an observation plane using a Huygens-type surface integral

This is an adiabatic / local-slab solver:
    * surface curvature must be slow on the wavelength scale
    * the radial permittivity profile must vary slowly over one wavelength
    * lateral coupling / guided propagation inside the shell is neglected
"""

import numpy as np
from dataclasses import dataclass


# ----------------------------
# Physical constants
# ----------------------------
c0 = 299792458.0
mu0 = 4e-7 * np.pi
eps0 = 1.0 / (mu0 * c0**2)
eta0 = np.sqrt(mu0 / eps0)


@dataclass
class RadomeParams:
    frequency_hz: float
    z_b: float               # vertex z of the lower paraboloid
    alpha: float             # paraboloid coefficient, z = z_b + alpha r^2
    thickness_n: float       # constant normal thickness
    a_eps: float             # eps_r(r) = a_eps r^2 + b_eps r + c_eps
    b_eps: float
    c_eps: float
    dipole_moment: complex = 1.0   # p_y in C*m
    n1: float = 1.0          # lower exterior refractive index
    n3: float = 1.0          # upper exterior refractive index
    aperture_radius: float = 0.10  # truncate the radome to this radius (m)


def complex_sqrt_pos_imag(x):
    """Branch-consistent complex sqrt."""
    y = np.sqrt(x + 0j)
    if np.imag(y) < 0:
        y = -y
    return y


def lower_surface_z(x, y, p: RadomeParams):
    return p.z_b + p.alpha * (x*x + y*y)


def lower_surface_dzdx(x, y, p: RadomeParams):
    return 2.0 * p.alpha * x


def lower_surface_dzdy(x, y, p: RadomeParams):
    return 2.0 * p.alpha * y


def lower_surface_normal(x, y, p: RadomeParams):
    """
    Upward unit normal of the lower surface.
    Surface: f(x,y,z) = z - z_b - alpha (x^2+y^2) = 0
    Upward normal is proportional to (-df/dx, -df/dy, 1).
    """
    n = np.array([
        -lower_surface_dzdx(x, y, p),
        -lower_surface_dzdy(x, y, p),
        1.0
    ], dtype=float)
    return n / np.linalg.norm(n)


def lower_surface_point(x, y, p: RadomeParams):
    return np.array([x, y, lower_surface_z(x, y, p)], dtype=float)


def upper_surface_point(x, y, p: RadomeParams):
    """
    Exact normal-offset outer surface for constant normal thickness.
    """
    r0 = lower_surface_point(x, y, p)
    n = lower_surface_normal(x, y, p)
    return r0 + p.thickness_n * n


def surface_jacobian_lower(x, y, p: RadomeParams):
    dzdx = lower_surface_dzdx(x, y, p)
    dzdy = lower_surface_dzdy(x, y, p)
    return np.sqrt(1.0 + dzdx*dzdx + dzdy*dzdy)


def eps_r_profile(x, y, p: RadomeParams):
    r = np.sqrt(x*x + y*y)
    return p.a_eps * r*r + p.b_eps * r + p.c_eps


def layer_matrix(delta, Z):
    cd = np.cos(delta)
    sd = np.sin(delta)
    return np.array([
        [cd, 1j * Z * sd],
        [1j * sd / Z, cd]
    ], dtype=complex)


def hertzian_dipole_fields(r_obs, freq_hz, p_vec, eps=eps0, mu=mu0):
    """
    Exact electric field of an electric Hertzian dipole in homogeneous medium.
    Magnetic field is computed from vector potential curl result.

    Time convention: exp(-i omega t)

    E(r) = e^{i k R}/(4 pi eps) * [ k^2 ((Rhat x p) x Rhat)/R
                                   + (1/R^3 - i k / R^2)*(3 Rhat(Rhat·p)-p) ]

    H(r) = omega * e^{i k R}/(4 pi R) * (k + i/R) * (Rhat x p)

    where p is the dipole moment vector in C·m.
    """
    omega = 2.0 * np.pi * freq_hz
    k = omega * np.sqrt(mu * eps)

    R = np.linalg.norm(r_obs)
    if R == 0:
        raise ValueError("Observation point cannot be at the dipole location.")

    Rhat = r_obs / R
    expfac = np.exp(1j * k * R) / (4.0 * np.pi)

    term1 = (k**2 / R) * np.cross(np.cross(Rhat, p_vec), Rhat)
    term2 = (1.0 / R**3 - 1j * k / R**2) * (3.0 * Rhat * np.dot(Rhat, p_vec) - p_vec)
    E = expfac * (term1 + term2) / eps

    H = expfac * omega * (k + 1j / R) * np.cross(Rhat, p_vec) / R
    return E, H


def local_te_tm_basis(s_hat, n_hat, fallback_axis=np.array([0.0, 1.0, 0.0])):
    """
    Build local s/p basis from incident direction s_hat and surface normal n_hat.

    e_s is perpendicular to the local plane of incidence.
    e_p is tangential and lies in the plane of incidence.

    If incidence is nearly normal, fall back to a stable tangential direction.
    """
    v = np.cross(n_hat, s_hat)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        # Pick a stable tangential basis
        t = fallback_axis - np.dot(fallback_axis, n_hat) * n_hat
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            fallback_axis = np.array([1.0, 0.0, 0.0])
            t = fallback_axis - np.dot(fallback_axis, n_hat) * n_hat
            nt = np.linalg.norm(t)
        e_s = t / nt
    else:
        e_s = v / nv

    e_p = np.cross(e_s, n_hat)
    e_p /= np.linalg.norm(e_p)
    return e_s, e_p


def local_patch_transmission(x, y, p: RadomeParams):
    """
    Compute transmitted tangential fields on the outer surface of the shell
    for one patch, using local TE/TM transfer matrices.
    """
    # Geometry
    rin = lower_surface_point(x, y, p)
    n_hat = lower_surface_normal(x, y, p)
    rout = upper_surface_point(x, y, p)

    # Exact incident dipole field on the lower surface
    p_vec = np.array([0.0, p.dipole_moment, 0.0], dtype=complex)
    E_inc, H_inc = hertzian_dipole_fields(rin, p.frequency_hz, p_vec, eps0 * p.n1**2, mu0)

    # Local ray direction from source to patch
    s_hat = rin / np.linalg.norm(rin)

    # Local s/p basis
    e_s, e_p = local_te_tm_basis(s_hat, n_hat)

    # Tangential incident components
    Esi = np.dot(E_inc, e_s)
    Epi = np.dot(E_inc, e_p)

    # Local incidence angle
    cos_t1 = np.clip(np.dot(s_hat, n_hat), -1.0, 1.0)
    # We want the acute incidence angle magnitude
    cos_t1 = abs(cos_t1)
    sin_t1_sq = max(0.0, 1.0 - cos_t1*cos_t1)

    # Local dielectric profile
    epsr = eps_r_profile(x, y, p)
    if np.real(epsr) <= 0 and abs(np.imag(epsr)) < 1e-15:
        raise ValueError(f"eps_r <= 0 at x={x}, y={y}: {epsr}")

    n2 = complex_sqrt_pos_imag(epsr)
    eta1 = eta0 / p.n1
    eta2 = eta0 / n2
    eta3 = eta0 / p.n3

    # Snell and impedances
    sin_t2_sq = (p.n1**2 / (n2**2)) * sin_t1_sq
    cos_t2 = complex_sqrt_pos_imag(1.0 - sin_t2_sq)
    sin_t3_sq = (p.n1**2 / (p.n3**2)) * sin_t1_sq
    cos_t3 = complex_sqrt_pos_imag(1.0 - sin_t3_sq)

    Z1te = eta1 / cos_t1
    Z1tm = eta1 * cos_t1

    Z2te = eta2 / cos_t2
    Z2tm = eta2 * cos_t2

    Z3te = eta3 / cos_t3
    Z3tm = eta3 * cos_t3

    k0 = 2.0 * np.pi * p.frequency_hz / c0
    delta = k0 * n2 * p.thickness_n * cos_t2

    Mte = layer_matrix(delta, Z2te)
    Mtm = layer_matrix(delta, Z2tm)

    Ate, Bte, Cte, Dte = Mte[0, 0], Mte[0, 1], Mte[1, 0], Mte[1, 1]
    Atm, Btm, Ctm, Dtm = Mtm[0, 0], Mtm[0, 1], Mtm[1, 0], Mtm[1, 1]

    # Local transmitted TE and TM electric amplitudes
    Es_out = 2.0 * Esi / (Ate + Bte / Z3te + Z1te * (Cte + Dte / Z3te))
    Ep_out = 2.0 * Epi / (Atm + Btm / Z3tm + Z1tm * (Ctm + Dtm / Z3tm))

    Hp_out = Es_out / Z3te
    Hs_out = -Ep_out / Z3tm

    # Reconstruct tangential output fields on outer surface
    E_t_out = Es_out * e_s + Ep_out * e_p
    H_t_out = Hs_out * e_s + Hp_out * e_p

    return {
        "rin": rin,
        "rout": rout,
        "n_hat": n_hat,
        "e_s": e_s,
        "e_p": e_p,
        "E_inc": E_inc,
        "H_inc": H_inc,
        "E_t_out": E_t_out,
        "H_t_out": H_t_out,
        "epsr": epsr,
        "n2": n2,
        "delta": delta,
    }


def build_surface_currents(x_grid, y_grid, p: RadomeParams):
    """
    Sample the outer surface currents over the truncated radome.
    Returns arrays for rout, J_s, M_s, area element.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    nx = len(x_grid)
    ny = len(y_grid)

    if nx < 2 or ny < 2:
        raise ValueError("Need at least 2 grid points in x and y.")

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    patches = []

    for iy, y in enumerate(y_grid):
        for ix, x in enumerate(x_grid):
            r = np.sqrt(x*x + y*y)
            if r > p.aperture_radius:
                continue

            loc = local_patch_transmission(x, y, p)
            n_hat = loc["n_hat"]

            # Tangential output fields on the outer surface
            E_t = loc["E_t_out"]
            H_t = loc["H_t_out"]

            # Equivalence principle
            J_s = np.cross(n_hat, H_t)
            M_s = -np.cross(n_hat, E_t)

            # Use the lower-surface metric as a patch area weight.
            # Since the upper surface is a normal offset, this is first-order accurate
            # in the slow-curvature regime.
            dS = surface_jacobian_lower(x, y, p) * dx * dy

            patches.append({
                "rout": loc["rout"],
                "n_hat": n_hat,
                "J_s": J_s,
                "M_s": M_s,
                "dS": dS,
                "meta": loc
            })

    return patches


def huygens_field_at_point(r_obs, patches, freq_hz, n_medium=1.0):
    """
    Field above the radome from equivalent electric and magnetic surface currents.

    Approximate free-space Huygens radiation integrals in homogeneous medium:

      E(r) ≈ i ω μ ∫ G J_s dS - ∫ [grad G × M_s] dS
      H(r) ≈ i ω ε ∫ G M_s dS + ∫ [grad G × J_s] dS

    with
      G = exp(i k R)/(4 pi R)
      grad G = (exp(i k R)/(4 pi)) * (i k R - 1) * Rvec / R^3
    """
    omega = 2.0 * np.pi * freq_hz
    eps = eps0 * n_medium**2
    mu = mu0
    k = omega * np.sqrt(mu * eps)

    E = np.zeros(3, dtype=complex)
    H = np.zeros(3, dtype=complex)

    for patch in patches:
        rs = patch["rout"]
        Rvec = r_obs - rs
        R = np.linalg.norm(Rvec)
        if R == 0:
            continue

        G = np.exp(1j * k * R) / (4.0 * np.pi * R)
        gradG = np.exp(1j * k * R) * (1j * k * R - 1.0) * Rvec / (4.0 * np.pi * R**3)

        dS = patch["dS"]
        J = patch["J_s"]
        M = patch["M_s"]

        E += (1j * omega * mu * G * J - np.cross(gradG, M)) * dS
        H += (1j * omega * eps * G * M + np.cross(gradG, J)) * dS

    return E, H


def compute_observation_plane(z_obs, x_obs, y_obs, patches, p: RadomeParams):
    """
    Compute E and H on a horizontal observation plane z = z_obs.
    Returns arrays of shape (ny, nx, 3).
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    E_map = np.zeros((len(y_obs), len(x_obs), 3), dtype=complex)
    H_map = np.zeros((len(y_obs), len(x_obs), 3), dtype=complex)

    for iy, y in enumerate(y_obs):
        for ix, x in enumerate(x_obs):
            r_obs = np.array([x, y, z_obs], dtype=float)
            E, H = huygens_field_at_point(r_obs, patches, p.frequency_hz, p.n3)
            E_map[iy, ix, :] = E
            H_map[iy, ix, :] = H

    return E_map, H_map


def solve_radome(
    params: RadomeParams,
    x_surface,
    y_surface,
    z_obs,
    x_obs,
    y_obs,
):
    """
    Full solver:
      1) sample transmitted fields on the outer radome surface
      2) build equivalent surface currents
      3) propagate to observation plane z = z_obs
    """
    patches = build_surface_currents(x_surface, y_surface, params)
    E_map, H_map = compute_observation_plane(z_obs, x_obs, y_obs, patches, params)
    return {
        "patches": patches,
        "E_map": E_map,
        "H_map": H_map,
        "x_obs": np.asarray(x_obs),
        "y_obs": np.asarray(y_obs),
        "z_obs": z_obs,
        "x_surface": np.asarray(x_surface),
        "y_surface": np.asarray(y_surface),
        "params": params,
    }


def plane_magnitude(E_map):
    return np.linalg.norm(E_map, axis=-1)


def example():
    # Example parameters
    params = RadomeParams(
        frequency_hz=10e9,       # 10 GHz
        z_b=0.12,                # lower-surface vertex 12 cm above origin
        alpha=6.0,               # paraboloid coefficient, m^-1
        thickness_n=0.004,       # 4 mm normal thickness
        a_eps=-8.0,              # example graded profile
        b_eps=1.5,
        c_eps=2.8,
        dipole_moment=1e-9,      # arbitrary example magnitude
        aperture_radius=0.08,
    )

    # Surface sampling grid
    Ls = params.aperture_radius
    x_surface = np.linspace(-Ls, Ls, 61)
    y_surface = np.linspace(-Ls, Ls, 61)

    # Observation plane
    z_obs = 0.20
    Lo = 0.06
    x_obs = np.linspace(-Lo, Lo, 41)
    y_obs = np.linspace(-Lo, Lo, 41)

    result = solve_radome(params, x_surface, y_surface, z_obs, x_obs, y_obs)

    Emag = plane_magnitude(result["E_map"])
    center = Emag[len(y_obs)//2, len(x_obs)//2]

    print("Solved radome field on observation plane.")
    print(f"Number of active source patches: {len(result['patches'])}")
    print(f"|E| at plane center: {center:.6e} V/m")

    # A few sample complex field values at the center
    E_center = result["E_map"][len(y_obs)//2, len(x_obs)//2]
    H_center = result["H_map"][len(y_obs)//2, len(x_obs)//2]
    print("E(center) =", E_center)
    print("H(center) =", H_center)


if __name__ == "__main__":
    example()
