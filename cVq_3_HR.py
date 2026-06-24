
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
import os

# =============================================================================
# Global configuration
# =============================================================================

OUTPUT_DIR = "."
DEBUG_TXT_FILE = os.path.join(OUTPUT_DIR, "debug_mid_results_2D_QvC.txt")

# Physical parameters
m_eff = 1.0      # effective mass
g_int = 0.1      # interaction strength
Delta = 0.0      # detuning
kappa = 0.0      # external potential strength (set to 0 here)

# Pump profile type: "ring" or "gaussian_disc" (we use "ring" to match vortex geometry)
PUMP_PROFILE = "ring"

# Grid globals (will be set by setup_grid)
Nx = None
Ny = None
Lx = None
Ly = None
x = None
y = None
dx = None
dy = None
X = None
Y = None
R = None
Theta = None
N_sites = None
L2D = None
H0_matrix = None


def laplacian_1d_matrix(N, d):
    r"""
    Finite-difference Laplacian in 1D with periodic boundary conditions.
    Returns an (N x N) matrix approximating d^2/dx^2.
    """
    lap = -2.0 * np.eye(N)
    for i in range(N):
        lap[i, (i - 1) % N] = 1.0
        lap[i, (i + 1) % N] = 1.0
    return lap / d**2


def laplacian_2d_matrix(Nx_in, Ny_in, dx_in, dy_in):
    r"""
    Construct the 2D Laplacian on a periodic Nx_in x Ny_in grid
    via Kronecker sums of 1D Laplacians in x and y.
    """
    lap_x = laplacian_1d_matrix(Nx_in, dx_in)
    lap_y = laplacian_1d_matrix(Ny_in, dy_in)

    Ix = np.eye(Nx_in)
    Iy = np.eye(Ny_in)

    L2D_local = np.kron(Iy, lap_x) + np.kron(lap_y, Ix)
    return L2D_local


def setup_grid(Nx_in, Ny_in, Lx_in=40.0, Ly_in=40.0):
    r"""
    Initialize the global grid variables and the single-particle H0 matrix
    for a given Nx, Ny, Lx, Ly.
    """
    global Nx, Ny, Lx, Ly
    global x, y, dx, dy, X, Y, R, Theta
    global N_sites, L2D, H0_matrix

    Nx = Nx_in
    Ny = Ny_in
    Lx = Lx_in
    Ly = Ly_in

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    N_sites = Nx * Ny

    # 2D Laplacian and single-particle H0
    L2D = laplacian_2d_matrix(Nx, Ny, dx, dy)
    H0_matrix = build_H0_matrix(L2D)


def build_H0_matrix(L2D_local):
    r"""
    Build the single-particle Hamiltonian H0:

        H0 = - (1 / (2 m_eff)) ∇^2 - Delta + U(r)

    Here we take U(r) = kappa * r^2, but set kappa = 0 by default.
    """
    global R
    H_kin = -(1.0 / (2.0 * m_eff)) * L2D_local

    U = kappa * (R**2)
    U_diag = np.diag(U.reshape(-1))

    H0 = H_kin - Delta * np.eye(U_diag.shape[0]) + U_diag
    return H0.astype(np.complex128)


def pump_profile_gaussian_disc(Fp, l):
    r"""
    Gaussian disc pump with vortex phase:

        F_p(r, θ) = Fp * exp(-r^2 / (2 σ^2)) * exp(i l θ).
    """
    global R, Theta, Lx
    sigma = Lx / 8.0
    envelope = np.exp(-R**2 / (2.0 * sigma**2))
    return Fp * envelope * np.exp(1j * l * Theta)


def pump_profile_ring(Fp, l):
    r"""
    Ring-shaped pump with vortex phase:

        F_p(r, θ) = Fp * exp(-(r - r0)^2 / (2 σ_r^2)) * exp(i l θ).
    """
    global R, Theta, Lx
    r0 = Lx / 4.0
    sigma_r = Lx / 10.0
    envelope = np.exp(-(R - r0)**2 / (2.0 * sigma_r**2))
    return Fp * envelope * np.exp(1j * l * Theta)


def pump_profile(Fp, l):
    r"""
    Select pump profile according to PUMP_PROFILE.
    Returns a complex array of shape (Nx, Ny).
    """
    global PUMP_PROFILE
    if PUMP_PROFILE == "gaussian_disc":
        return pump_profile_gaussian_disc(Fp, l)
    else:
        return pump_profile_ring(Fp, l)


def mean_field_rhs(psi_vec, Fp_vec, gamma):
    r"""
    Right-hand side of the driven-dissipative GPE:

        i ∂ψ/∂t = [H0 + g |ψ|^2 - i γ/2] ψ + F_p.

    We rewrite:

        ∂ψ/∂t = -i H0 ψ - i g |ψ|^2 ψ - i F_p - (γ/2) ψ.

    psi_vec, Fp_vec are flattened (N_sites,).
    """
    global H0_matrix

    psi_vec = psi_vec.reshape(-1)
    abs2 = np.abs(psi_vec)**2
    diag_nl = g_int * abs2

    H0_psi = H0_matrix @ psi_vec
    H_nl_psi = diag_nl * psi_vec

    rhs = -1j * H0_psi - 1j * H_nl_psi - 1j * Fp_vec - (gamma / 2.0) * psi_vec
    return rhs


def mean_field_residual(psi_vec, Fp_grid, gamma):
    r"""
    Compute max absolute residual of the stationary equation:

        0 = [H0 + g |ψ|^2 - i γ/2] ψ + F_p(r).
    """
    global H0_matrix, Nx, Ny

    psi_vec = psi_vec.reshape(-1)
    psi_grid = psi_vec.reshape(Nx, Ny)
    abs2 = np.abs(psi_grid)**2
    diag_nl = g_int * abs2.reshape(-1)

    H0_psi = H0_matrix @ psi_vec
    H_nl_psi = diag_nl * psi_vec

    res_vec = H0_psi + H_nl_psi - 1j * (gamma / 2.0) * psi_vec + Fp_grid.reshape(-1)
    return float(np.max(np.abs(res_vec)))


def solve_mean_field_2d_time(Fp, gamma, l,
                             dt=1e-2,
                             max_steps=20000,
                             tol=1e-8,
                             residual_tol=1e-4,
                             residual_check_every=100):
    global Nx, Ny

    Fp_grid = pump_profile(Fp, l)
    Fp_vec = Fp_grid.reshape(-1)

    # Initial guess: crude local balance
    psi0_grid = Fp_grid / (gamma / 2.0 + 1.0)
    psi_vec = psi0_grid.reshape(-1).astype(np.complex128)

    for step in range(max_steps):
        psi_old = psi_vec.copy()

        k1 = mean_field_rhs(psi_vec, Fp_vec, gamma)
        psi_half = psi_vec + 0.5 * dt * k1
        k2 = mean_field_rhs(psi_half, Fp_vec, gamma)
        psi_vec = psi_vec + dt * k2

        max_diff = np.max(np.abs(psi_vec - psi_old))
        if max_diff < tol and step > 100:
            break

        if (step + 1) % residual_check_every == 0:
            res = mean_field_residual(psi_vec, Fp_grid, gamma)
            if res < residual_tol and step > 100:
                break

    final_res = mean_field_residual(psi_vec, Fp_grid, gamma)
    return psi_vec.reshape(Nx, Ny), final_res, step + 1


# =============================================================================
# Bogoliubov, drift, diffusion
# =============================================================================

def build_bogoliubov_matrix_2d(psi0, gamma):
    global H0_matrix, N_sites

    psi_vec = psi0.reshape(-1)
    abs2 = np.abs(psi_vec)**2
    diag_nl = 2.0 * g_int * abs2
    diag_pair = g_int * psi_vec**2

    A_block = H0_matrix + np.diag(diag_nl) - 1j * (gamma / 2.0) * np.eye(N_sites)
    D_block = -np.conjugate(H0_matrix) - np.diag(diag_nl) - 1j * (gamma / 2.0) * np.eye(N_sites)

    B_block = np.diag(diag_pair)
    C_block = -np.diag(np.conjugate(diag_pair))

    M = np.block([[A_block, B_block],
                  [C_block, D_block]])
    return M


def nambu_to_quadrature_transform(N):
    S = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    for j in range(N):
        a = j
        ad = j + N
        x_idx = 2 * j
        p_idx = 2 * j + 1

        S[x_idx, a] = 1.0 / np.sqrt(2.0)
        S[x_idx, ad] = 1.0 / np.sqrt(2.0)
        S[p_idx, a] = 1j / np.sqrt(2.0)
        S[p_idx, ad] = -1j / np.sqrt(2.0)

    return S


def build_diffusion_matrix_from_noise_2d(N, gamma):
    N_Xi = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    # Block basis: Top-left N x N block is <psi psi^dag>
    np.fill_diagonal(N_Xi[0:N, 0:N], gamma)
    
    S = nambu_to_quadrature_transform(N)
    D_raw = S @ N_Xi @ S.conj().T
    D = 0.5 * (D_raw + D_raw.conj().T)
    return np.real(D)
    

def build_drift_and_diffusion_2d(M, gamma):
    N2 = M.shape[0]
    N = N2 // 2

    S = nambu_to_quadrature_transform(N)
    Sinv = la.inv(S)

    L = -1j * M
    A = np.real(S @ L @ Sinv)
    D = build_diffusion_matrix_from_noise_2d(N, gamma)

    return A, D


# =============================================================================
# Lyapunov solver and entanglement
# =============================================================================

def solve_steady_covariance(A, D):
    V = la.solve_continuous_lyapunov(A, -D)
    return V


def symplectic_form(Nmodes):
    r"""
    Build the symplectic form Omega for Nmodes (each with x,p).
    """
    Omega = np.zeros((2 * Nmodes, 2 * Nmodes))
    for j in range(Nmodes):
        Omega[2*j, 2*j+1] = 1.0
        Omega[2*j+1, 2*j] = -1.0
    return Omega


def log_negativity_and_min_symplectic(V, region_A, region_B):
    idx_A = []
    for j in region_A:
        idx_A.extend([2*j, 2*j+1])
    idx_B = []
    for j in region_B:
        idx_B.extend([2*j, 2*j+1])

    idx_AB = idx_A + idx_B
    V_AB = V[np.ix_(idx_AB, idx_AB)]

    Nm_A = len(region_A)
    Nm_B = len(region_B)
    Nm_AB = Nm_A + Nm_B

    # Partial transpose w.r.t. B: flip sign of p_B entries
    Lambda_B = np.eye(2 * Nm_AB)
    for k in range(Nm_B):
        p_index = 2 * Nm_A + 2 * k + 1
        Lambda_B[p_index, p_index] = -1.0





    V_PT = Lambda_B @ V_AB @ Lambda_B
    Omega_AB = symplectic_form(Nm_AB)

    eigvals = la.eigvals(1j * Omega_AB @ V_PT)

    # Symplectic eigenvalues come in ± pairs; work with sorted absolute values
    eig_abs_sorted = np.sort(np.abs(eigvals))

    # Take one eigenvalue from each ± pair (indices 0, 2, 4, ...)
    nu_tilde = eig_abs_sorted[::2][:Nm_AB]
    nu_tilde = np.real_if_close(nu_tilde)

    nu_min_PT = float(np.min(nu_tilde))
    
    
    
    
    

    EN = 0.0
    for nu in nu_tilde:
        nu_safe = max(float(np.real(nu)), 1e-12)
        val = max(0.0, -np.log(2.0 * nu_safe))
        EN += val

    return float(EN), nu_min_PT


# =============================================================================
# Debug logging
# =============================================================================

def append_debug_info_2d(Fp, gamma, l, psi0, M, A, EN, nu_min_PT,
                         mf_residual_val, core_radius, pump_profile,
                         Nx_local, Ny_local, n_steps):
    r"""
    Append mid-results to the 2D QvC debug file.
    """
    psi_vec = psi0.reshape(-1)
    eig_M = la.eigvals(M)
    eig_A = la.eigvals(A)

    min_Re_M = np.min(eig_M.real)
    max_Re_M = np.max(eig_M.real)
    min_Re_A = np.min(eig_A.real)
    max_Re_A = np.max(eig_A.real)

    psi_norm = np.linalg.norm(psi_vec)
    psi_max_abs = np.max(np.abs(psi_vec))

    with open(DEBUG_TXT_FILE, "a") as f:
        f.write(f"--- 2D QvC: Fp={Fp:.6g}, gamma={gamma:.6g}, l={l:d} ---\n")
        f.write(
            f"Nx={Nx_local}, Ny={Ny_local}, Lx={Lx:.3f}, Ly={Ly:.3f}, "
            f"core_radius={core_radius:.3f}, pump_profile={pump_profile}\n"
        )
        f.write(f"time_steps = {n_steps:d}\n")
        f.write(f"psi0_norm = {psi_norm:.6e}, psi0_max_abs = {psi_max_abs:.6e}\n")
        f.write(f"mean_field_residual = {mf_residual_val:.6e}\n")
        f.write(
            f"M eigenvalues Re[min,max] = ({min_Re_M:.6e}, {max_Re_M:.6e})\n"
        )
        f.write(
            f"A eigenvalues Re[min,max] = ({min_Re_A:.6e}, {max_Re_A:.6e})\n"
        )
        f.write(f"E_N_quantum = {EN:.6e}, nu_min_PT = {nu_min_PT:.6e}\n\n")


# =============================================================================
# Wrapper: compute EN(Fp) in 2D for given parameters
# =============================================================================

def compute_EN_2d_for_params(Fp, gamma, l, core_radius,
                             log_debug=True):
    global R, N_sites, Nx, Ny, PUMP_PROFILE

    psi0, mf_res, n_steps = solve_mean_field_2d_time(Fp, gamma, l)
    M = build_bogoliubov_matrix_2d(psi0, gamma)
    A, D = build_drift_and_diffusion_2d(M, gamma)
    V_ss = solve_steady_covariance(A, D)

    r_flat = R.reshape(-1)
    site_indices = np.arange(N_sites)

    core_mask = (r_flat < core_radius)
    core_indices = site_indices[core_mask].tolist()
    outer_indices = site_indices[~core_mask].tolist()

    EN, nu_min_PT = log_negativity_and_min_symplectic(
        V_ss, core_indices, outer_indices
    )

    if log_debug:
        append_debug_info_2d(
            Fp, gamma, l, psi0, M, A, EN, nu_min_PT,
            mf_res, core_radius, PUMP_PROFILE, Nx, Ny, n_steps
        )

    return EN


# =============================================================================
# Scan and plot EN(Fp): quantum vs classical baseline
# =============================================================================

def scan_and_plot_EN_vs_Fp_2d_QvC(gamma=0.5, l=1, core_radius=5.0):
    global PUMP_PROFILE, Nx, Ny

    # Pump range (same as before for consistency)
    Fp_values = np.linspace(0.1, 2.0, 8)
    EN_quantum = []

    for Fp in Fp_values:
        print(
            f"[2D QvC, profile={PUMP_PROFILE}, Nx={Nx}, Ny={Ny}] "
            f"E_N_quantum for Fp={Fp:.3f}, gamma={gamma:.3f}, "
            f"l={l:d}, core_radius={core_radius:.3f}"
        )
        EN = compute_EN_2d_for_params(Fp, gamma, l, core_radius,
                                      log_debug=True)
        EN_quantum.append(EN)

    Fp_values = np.array(Fp_values)
    EN_quantum = np.array(EN_quantum)

    # Classical Gaussian baseline: identically zero
    EN_classical = np.zeros_like(EN_quantum)

    # --- Publication-style figure: quantum vs classical ---
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(Fp_values, EN_quantum, marker="o", linewidth=1.5,
             label=r"quantum QLE")
    plt.plot(Fp_values, EN_classical, linestyle="--", linewidth=1.0,
             label=r"classical Gaussian baseline")
    plt.xlabel(r"$F_p$")
    plt.ylabel(r"$E_{\mathcal{N}}$")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    base_name = (
        f"EN_vs_Fp_2D_QvC_gamma_{gamma:.3f}_l_{l:d}_coreR_{core_radius:.2f}"
        f"_profile_{PUMP_PROFILE}_Nx{Nx}_Ny{Ny}"
    )

    pdf_name = os.path.join(OUTPUT_DIR, base_name + ".pdf")
    png_name = os.path.join(OUTPUT_DIR, base_name + ".png")

    plt.savefig(pdf_name)
    plt.savefig(png_name, dpi=300)
    plt.close()
    print(f"Saved 2D QvC EN(Fp) PDF: {pdf_name}")
    print(f"Saved 2D QvC EN(Fp) PNG: {png_name}")

    # --- JSON with data ---
    json_name = os.path.join(OUTPUT_DIR, base_name + ".json")
    data = {
        "gamma": float(gamma),
        "vortex_charge_l": int(l),
        "core_radius": float(core_radius),
        "pump_profile": PUMP_PROFILE,
        "Fp_values": Fp_values.tolist(),
        "EN_quantum_values": EN_quantum.tolist(),
        "EN_classical_values": EN_classical.tolist(),
        "Nx": int(Nx),
        "Ny": int(Ny),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "g_int": float(g_int),
        "Delta": float(Delta),
        "kappa": float(kappa),
        "classical_baseline_comment": (
            "EN_classical_values ≡ 0 represent the Gaussian classical "
            "baseline (no entanglement) for any positive-P classical "
            "field model with the same drift A."
        ),
    }
    with open(json_name, "w") as jf:
        json.dump(data, jf, indent=2)
    print(f"Saved 2D QvC EN(Fp) JSON: {json_name}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Initialize / overwrite debug file with a header
    with open(DEBUG_TXT_FILE, "w") as f:
        f.write("Debug mid-results for 2D giant-vortex quantum vs classical benchmark\n")
        f.write(
            f"g_int={g_int}, Delta={Delta}, kappa={kappa}, "
            f"pump_profile={PUMP_PROFILE}\n\n"
        )
    print(f"2D QvC debug log will be written to: {DEBUG_TXT_FILE}")

    gamma = 0.5
    l = 1
    core_radius = 5.0

    # Choose the grid for the benchmark (30x30 is a good flagship choice)
    Nx_in, Ny_in = 30, 30
    print("\n========================================")
    print(f"Running 2D QvC EN(Fp) scan for grid {Nx_in} x {Ny_in}")
    print("========================================\n")

    setup_grid(Nx_in, Ny_in, Lx_in=40.0, Ly_in=40.0)
    scan_and_plot_EN_vs_Fp_2d_QvC(gamma=gamma, l=l, core_radius=core_radius)
