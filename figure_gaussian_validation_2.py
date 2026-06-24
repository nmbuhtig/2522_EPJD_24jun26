
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
import csv

# ============================================================================
# MANUSCRIPT PARAMETERS (EXACT)
# ============================================================================

m_eff = 1.0       # Effective mass
g_int = 0.1       # Interaction strength
Delta = 0.0       # Detuning
gamma = 0.5       # Loss rate
ell = 1           # Vortex charge

# Grid
Nx = 30
Ny = 30
Lx = 40.0
Ly = 40.0

# Pump
Fp = 0.65
R0 = Lx / 4.0
sigma_R = Lx / 10.0

# Validation
r_ergo_fixed = 6.5
delta_r_init = 1.0
delta_r_max = 2.0
min_annulus_points = 80
delta_r_growth = 0.2


# ============================================================================
# Grid setup
# ============================================================================

def setup_grid():
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    N_sites = Nx * Ny
    
    lap_x = laplacian_1d(Nx, dx)
    lap_y = laplacian_1d(Ny, dy)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    L2D = np.kron(Iy, lap_x) + np.kron(lap_y, Ix)
    
    H0 = -(1.0 / (2.0 * m_eff)) * L2D - Delta * np.eye(N_sites)
    H0 = H0.astype(np.complex128)
    
    return X, Y, R, Theta, dx, dy, H0, N_sites


def laplacian_1d(N, d):
    lap = -2.0 * np.eye(N)
    for i in range(N):
        lap[i, (i - 1) % N] = 1.0
        lap[i, (i + 1) % N] = 1.0
    return lap / d**2


# ============================================================================
# Pump profile
# ============================================================================

def ring_pump(X, Y, Theta):
    R = np.sqrt(X**2 + Y**2)
    envelope = np.exp(-(R - R0)**2 / (2.0 * sigma_R**2))
    phase = np.exp(1j * ell * Theta)
    return Fp * envelope * phase


# ============================================================================
# Mean-field solver
# ============================================================================

def mean_field_rhs(psi_vec, Fp_vec, H0):
    psi_vec = psi_vec.reshape(-1)
    abs2 = np.abs(psi_vec)**2
    diag_nl = g_int * abs2
    
    rhs = (-1j * (H0 @ psi_vec) - 
           1j * diag_nl * psi_vec - 
           1j * Fp_vec - 
           (gamma / 2.0) * psi_vec)
    return rhs


def solve_mean_field(Fp_vec, H0, dt=1e-2, max_steps=20000, tol=1e-8):
    print("Solving mean-field GPE (real-time RK2)...")
    
    psi_vec = Fp_vec / (gamma / 2.0 + 1.0)
    psi_vec = psi_vec.astype(np.complex128)
    
    for step in range(max_steps):
        psi_old = psi_vec.copy()
        
        k1 = mean_field_rhs(psi_vec, Fp_vec, H0)
        psi_half = psi_vec + 0.5 * dt * k1
        k2 = mean_field_rhs(psi_half, Fp_vec, H0)
        psi_vec = psi_vec + dt * k2
        
        max_diff = np.max(np.abs(psi_vec - psi_old))
        
        if step % 500 == 0:
            max_n = np.max(np.abs(psi_vec)**2)
            print(f"  Step {step:5d}: max|Δψ|={max_diff:.3e}, max|ψ|²={max_n:.4f}")
        
        if max_diff < tol and step > 100:
            print(f"✓ Converged at step {step}")
            break
    
    psi0 = psi_vec.reshape(Nx, Ny)
    return psi0, max_diff, step + 1


# ============================================================================
# Bogoliubov matrix
# ============================================================================

def build_bogoliubov_matrix(psi0, H0, N_sites):
    psi_vec = psi0.reshape(-1)
    abs2 = np.abs(psi_vec)**2
    
    A = H0 + np.diag(2.0 * g_int * abs2) - 1j * (gamma / 2.0) * np.eye(N_sites)
    B = np.diag(g_int * psi_vec**2)
    C = -np.diag(g_int * np.conjugate(psi_vec)**2)
    D_block = -np.conjugate(H0) - np.diag(2.0 * g_int * abs2) - 1j * (gamma / 2.0) * np.eye(N_sites)
    
    M = np.block([[A, B], [C, D_block]])
    return M


# ============================================================================
# Drift and diffusion
# ============================================================================

def nambu_to_quadrature_transform(N):
    S = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    for j in range(N):
        x_idx = 2 * j
        p_idx = 2 * j + 1
        S[x_idx, j] = 1.0 / np.sqrt(2.0)
        S[x_idx, j + N] = 1.0 / np.sqrt(2.0)
        S[p_idx, j] = 1j / np.sqrt(2.0)
        S[p_idx, j + N] = -1j / np.sqrt(2.0)
    return S


def build_diffusion_matrix(N):
    N_Xi = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    np.fill_diagonal(N_Xi[0:N, 0:N], gamma)
    
    S = nambu_to_quadrature_transform(N)
    D_raw = S @ N_Xi @ S.conj().T
    D = 0.5 * (D_raw + D_raw.conj().T)
    return np.real(D)


def build_drift_and_diffusion(M, N_sites):
    S = nambu_to_quadrature_transform(N_sites)
    Sinv = la.inv(S)
    
    L = -1j * M
    A = np.real(S @ L @ Sinv)
    D = build_diffusion_matrix(N_sites)
    
    return A, D


# ============================================================================
# Lyapunov solver
# ============================================================================

def solve_steady_covariance(A, D):
    print("Solving Lyapunov equation...")
    V = la.solve_continuous_lyapunov(A, -D)
    print("✓ Lyapunov solved")
    return V


def extract_fluctuation_density(V, N_sites):
    n_fl = np.zeros(N_sites)
    for j in range(N_sites):
        V_xx = V[2*j, 2*j]
        V_pp = V[2*j+1, 2*j+1]
        n_fl[j] = (V_xx + V_pp - 1.0) / 2.0
    return n_fl


# ============================================================================
# Adaptive annulus
# ============================================================================

def get_annulus_adaptive(R, r_ergo):
    delta_r = delta_r_init
    
    while delta_r <= delta_r_max:
        mask = np.abs(R - r_ergo) <= delta_r
        n_points = np.sum(mask)
        
        if n_points >= min_annulus_points:
            return delta_r, mask, n_points
        
        delta_r += delta_r_growth
    
    mask = np.abs(R - r_ergo) <= delta_r_max
    n_points = np.sum(mask)
    print(f"⚠ Warning: Only {n_points} points at max Δr={delta_r_max:.2f}")
    return delta_r_max, mask, n_points


def annulus_stats(field, mask):
    values = field[mask]
    return {
        'n_points': len(values),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
    }


def export_data_and_create_figure(psi0, n_fl, R, delta_r_eff, stats_n0, stats_nfl, r_centers, n0_r, nfl_r):
    print("\nExporting data and creating APS-style figure (no validation box)...")
    
    n0 = np.abs(psi0)**2
    
    # CSV: radial profiles
    csv_filename = "validation_radial_profile_r6p5.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['r_um', 'n0_mean', 'n_fl_mean'])
        for r, n0_val, nfl_val in zip(r_centers, n0_r, nfl_r):
            writer.writerow([f"{r:.3f}", f"{n0_val:.6e}", f"{nfl_val:.6e}"])
    print(f"✓ Saved radial data: {csv_filename}")
    
    # JSON: parameters and stats
    json_data = {
        "parameters": {
            "r_ergo_fixed_um": r_ergo_fixed,
            "delta_r_effective_um": float(delta_r_eff),
            "annulus_points": stats_n0["n_points"],
            "Fp": Fp,
            "gamma": gamma,
            "g_int": g_int,
            "grid": f"{Nx}x{Ny}"
        },
        "annulus_statistics": {
            "n0": {k: float(v) for k, v in stats_n0.items()},
            "n_fl": {k: float(v) for k, v in stats_nfl.items()},
            "ratio_max_nfl_over_median_n0": float(stats_nfl["max"] / stats_n0["median"])
        }
    }
    json_filename = "validation_figure_data_r6p5.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved figure metadata: {json_filename}")
    
    # APS-style figure
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'lines.linewidth': 3.0,
    })
    
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    
    pos_n0  = n0_r[n0_r > 0]
    pos_nfl = nfl_r[nfl_r > 0]
    use_log = len(pos_n0) > 0 and len(pos_nfl) > 0
    
    if use_log:
        ax.semilogy(r_centers, n0_r, 'b-', label=r'$|\psi_0|^2(r)$')
        ax.semilogy(r_centers, nfl_r, 'r-', label=r'$n_{\mathrm{fl}}(r)$')
        ymin = max(1e-12, pos_nfl.min() * 0.4 if len(pos_nfl) > 0 else 1e-12)
        ymax = pos_n0.max() * 4.0
        ax.set_ylim(ymin, ymax)
    else:
        ax.plot(r_centers, n0_r, 'b-', label=r'$|\psi_0|^2(r)$')
        ax.plot(r_centers, nfl_r, 'r-', label=r'$n_{\mathrm{fl}}(r)$')
    
    ax.axvline(r_ergo_fixed, color='gray', linestyle='--', linewidth=3.0,
               label=f'$r = {r_ergo_fixed:.1f}$ μm')
    
    ax.axvspan(r_ergo_fixed - delta_r_eff, r_ergo_fixed + delta_r_eff,
               alpha=0.15, color='gold',
               label=f'$\Delta r = {delta_r_eff:.2f}$ μm')
    
    ax.set_xlabel(r'Radius $r$ ($\mu$m)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, r_centers.max())
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.legend(
        loc='upper right',
        frameon=True,
        edgecolor='gray',
        framealpha=0.9,
        fontsize=10,                # 30% smaller than original 14
        handlelength=1.8,           # shorter legend handles
        handletextpad=0.6,          # less space between symbol and text
        borderpad=0.4,              # smaller padding inside the box
        labelspacing=0.5            # less vertical spacing between entries
    )    
    
    
    
    pdf_name = "validation_for_publication_r6p5.pdf"
    png_name = "validation_for_publication_r6p5.png"
    plt.savefig(pdf_name, dpi=400, bbox_inches='tight')
    plt.savefig(png_name, dpi=400, bbox_inches='tight')
    print(f"✓ Saved: {pdf_name}")
    print(f"✓ Saved: {png_name}")


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PUBLICATION VALIDATION FIGURE (APS style – no text box)")
    print("="*70)
    
    X, Y, R, Theta, dx, dy, H0, N_sites = setup_grid()
    Fp_grid = ring_pump(X, Y, Theta)
    Fp_vec = Fp_grid.reshape(-1)
    
    psi0, residual, n_steps = solve_mean_field(Fp_vec, H0)
    
    M = build_bogoliubov_matrix(psi0, H0, N_sites)
    A, D = build_drift_and_diffusion(M, N_sites)
    V = solve_steady_covariance(A, D)
    
    n_fl_flat = extract_fluctuation_density(V, N_sites)
    n_fl = n_fl_flat.reshape(Nx, Ny)
    
    delta_r_eff, annulus_mask, n_points = get_annulus_adaptive(R, r_ergo_fixed)
    
    n0 = np.abs(psi0)**2
    stats_n0 = annulus_stats(n0, annulus_mask)
    stats_nfl = annulus_stats(n_fl, annulus_mask)
    
    # Radial profiles
    r_bins = np.linspace(0, R.max(), 100)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    n0_r = np.zeros_like(r_centers)
    nfl_r = np.zeros_like(r_centers)
    
    for i in range(len(r_centers)):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.any(mask):
            n0_r[i] = np.mean(n0[mask])
            nfl_r[i] = np.mean(n_fl[mask])
    
    export_data_and_create_figure(psi0, n_fl, R, delta_r_eff,
                                 stats_n0, stats_nfl, r_centers, n0_r, nfl_r)
    
    ratio = stats_nfl['max'] / stats_n0['median']
    print("\n" + "="*70)
    print(f"FINAL VALIDATION RATIO: n_fl^max / n₀^med = {ratio:.4e}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()