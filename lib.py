import numpy as np

def define_concave_lwa(radius, arc_length, num_points):
    theta_arc = arc_length / radius
    theta = np.linspace(-theta_arc / 2, theta_arc / 2, num_points)
    xk = radius * np.sin(theta)
    yk = -radius * (1 - np.cos(theta))
    uk = np.array([np.sin(theta), np.cos(theta)])
    return xk, yk, uk, theta_arc
    

def compute_aperture_amplitude(xk, yk, χk, σk, sk, nk, xi_k, alpha, εr):
    A0 = 1.0
    A_LWA = A0 * np.sqrt(alpha) * np.exp(-alpha * xi_k)
    delta_L = np.linalg.norm(np.gradient(np.vstack((xk, yk)), axis=1), axis=0)
    delta_L_prime = np.linalg.norm(np.gradient(χk, axis=1), axis=0)
    dot_sigma_uk = np.sum(σk * np.vstack((np.gradient(xk), np.gradient(yk))), axis=0)
    dot_sk_nk = np.sum(sk * nk, axis=0)
    cos_phi_i_k = np.sum(σk * nk, axis=0)
    phi_r_k = np.arcsin(np.clip(np.sqrt(εr) * (σk[0, :] * nk[1, :] - σk[1, :] * nk[0, :]), -1.0, 1.0))
    Tk = 2 * cos_phi_i_k / (cos_phi_i_k + np.sqrt(1 / εr) * np.cos(phi_r_k))
    sigma_k = np.linalg.norm(χk - np.vstack((xk, yk)), axis=0)
    A_aperture = A_LWA * np.sqrt((delta_L * dot_sigma_uk) / (delta_L_prime * dot_sk_nk)) * Tk
    return A_aperture, Tk, delta_L_prime, sigma_k
    
def generate_ray_directions(theta_LWA, beta, kd):
    theta_k = np.arcsin(beta / kd)
    ϕd_k = theta_LWA + theta_k
    σk = np.array([np.sin(ϕd_k), np.cos(ϕd_k)])
    return σk, ϕd_k

def intersect_rays_with_lens(xk, yk, σk, lens_radius):
    χk = []
    for i in range(len(xk)):
        dx, dy = σk[0, i], σk[1, i]
        x0, y0 = xk[i], yk[i]
        a = dx**2 + dy**2
        b = 2 * (x0 * dx + y0 * dy)
        c = x0**2 + y0**2 - lens_radius**2
        t = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        χk.append([x0 + t * dx, y0 + t * dy])
    χk = np.array(χk).T
    nk = χk / np.linalg.norm(χk, axis=0)
    return χk, nk

def compute_exit_directions(σk, nk, εr):
    cos_phi_i_k = np.sum(σk * nk, axis=0)
    sin_phi_i_k = σk[0, :] * nk[1, :] - σk[1, :] * nk[0, :]
    sin_phi_r_k = np.sqrt(εr) * sin_phi_i_k
    phi_r_k = np.arcsin(np.clip(sin_phi_r_k, -1.0, 1.0))
    phi_n_k = np.arctan2(nk[0, :], nk[1, :])
    phi_0_k = phi_n_k + phi_r_k
    sk = np.array([np.sin(phi_0_k), np.cos(phi_0_k)])
    return sk, phi_0_k

def farfield_Krchhoff(R_obs, angles_rad, xk, yk, χk, A_aperture, Tk, delta_L_prime, sk, nk, xi_k, beta, kd, k0):
    # Length through lens (σ_k)
    sigma_k = np.linalg.norm(χk - np.vstack((xk, yk)), axis=0)  # shape: (num_rays,)
    
    # Observation points on the arc
    obs_points = R_obs * np.array([np.sin(angles_rad), np.cos(angles_rad)])  # shape (2, N_angles)

    # Initialize E(θ)
    E_theta = np.zeros(len(angles_rad), dtype=complex)

    # Loop over rays
    for k in range(xk.shape[0]):
        chi_k = χk[:, k]          # aperture point
        A_k = A_aperture[k]
        Tk_k = Tk[k]
        dL_k = delta_L_prime[k]
        s_k = sk[:, k]
        n_k = nk[:, k]
        xi = xi_k[k]
        sigma_k_val = sigma_k[k]
        base_phase = beta * xi + kd * sigma_k_val

        # Vector from aperture point to each observation point
        r_vec = obs_points - chi_k[:, np.newaxis]  # shape (2, N_angles)
        r_norm = np.linalg.norm(r_vec, axis=0)     # distance |r_k,i|
        r_hat = r_vec / r_norm                     # unit vectors (2, N_angles)

        # Phase delay from lens to observation point
        phase = base_phase + k0 * r_norm           # total phase

        # Projection term: n · s + n · r_hat (vector projection)
        proj = np.dot(n_k, s_k) + np.sum(n_k[:, np.newaxis] * r_hat, axis=0)  # (N_angles,)

        # Field contribution from this ray
        E_theta += A_k * np.exp(-1j * phase) * proj * Tk_k * dL_k
        
    return E_theta
