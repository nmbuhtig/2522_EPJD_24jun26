
import argparse
import json
import logging
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------

@dataclass
class ModelParams:

    L: float = 100.0          # system length
    N: int = 256              # number of grid points
    m_eff: float = 1.0        # effective mass (scaled)
    g: float = 0.01           # interaction strength
    gamma: float = 0.1        # loss rate
    Delta0: float = 0.0       # uniform detuning offset

    # Uniform test-case parameters
    psi0_amp: float = 1.0     # amplitude for uniform psi0
    psi0_phase_gradient: float = 0.0  # phase gradient for uniform psi0

    # Toy analog-horizon profile parameters
    horizon_core_amp: float = 1.0     # amplitude in core region
    horizon_bg_amp: float = 0.1       # amplitude outside core
    horizon_radius: float = 20.0      # radius of high-density core
    horizon_smoothing: float = 5.0    # smoothing length for tanh transition
    horizon_phase_gradient: float = 0.0  # flow-like phase gradient


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

def setup_logging(log_path: str, log_level=logging.DEBUG) -> None:
    """Configure logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(log_level)
    fh_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler (info level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    logging.debug("Logging initialized. Log file: %s", log_path)



def build_grid(params: ModelParams) -> np.ndarray:
    """Construct 1D spatial grid x with N points on a ring of length L."""
    L = params.L
    N = params.N
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    logging.debug("Grid built: L=%g, N=%d, dx=%g", L, N, dx)
    return x


def build_uniform_psi0(params: ModelParams, x: np.ndarray) -> np.ndarray:
    """Construct uniform psi0(x) = psi0_amp * exp(i k x)."""
    amp = params.psi0_amp
    k = params.psi0_phase_gradient
    psi0 = amp * np.exp(1j * k * x)
    logging.debug("Uniform psi0 built: amp=%g, phase_gradient=%g", amp, k)
    return psi0


def build_toy_horizon_psi0(params: ModelParams, x: np.ndarray) -> np.ndarray:
    """
    Construct a toy analog-horizon psi0(x):
      - Larger amplitude near |x| < horizon_radius
      - Smaller amplitude outside
      - Smooth tanh transition
      - Optional phase gradient to mimic flow
    """
    core_amp = params.horizon_core_amp
    bg_amp = params.horizon_bg_amp
    R = params.horizon_radius
    w = params.horizon_smoothing
    k = params.horizon_phase_gradient

    # Smooth step ~1 in core, ~0 outside
    step = 0.5 * (1.0 - np.tanh((np.abs(x) - R) / w))
    amp_profile = bg_amp + (core_amp - bg_amp) * step

    psi0 = amp_profile * np.exp(1j * k * x)

    logging.debug(
        "Toy horizon psi0: core_amp=%g, bg_amp=%g, R=%g, w=%g, k=%g",
        core_amp, bg_amp, R, w, k,
    )
    logging.debug(
        "Toy psi0: min|psi0|^2=%g, max|psi0|^2=%g",
        float(np.min(np.abs(psi0) ** 2)),
        float(np.max(np.abs(psi0) ** 2)),
    )
    return psi0


def build_laplacian(N: int, dx: float) -> np.ndarray:
    """Build 1D Laplacian with periodic BCs."""
    lap = np.zeros((N, N), dtype=complex)
    for j in range(N):
        lap[j, j] = -2.0
        lap[j, (j + 1) % N] = 1.0
        lap[j, (j - 1) % N] = 1.0
    lap /= dx ** 2
    logging.debug("Laplacian built: N=%d, dx=%g", N, dx)
    return lap


def build_drift_matrix(params: ModelParams,
                       x: np.ndarray,
                       psi0: np.ndarray,
                       Delta_x: np.ndarray,
                       Vbg_x: np.ndarray) -> np.ndarray:
    N = params.N
    L = params.L
    m_eff = params.m_eff
    g = params.g
    gamma = params.gamma

    dx = L / N
    lap = build_laplacian(N, dx)

    # Kinetic term: - (1 / 2m) d^2/dx^2 (hbar=1)
    T = -(1.0 / (2.0 * m_eff)) * lap

    abs2 = np.abs(psi0) ** 2

    # H_BdG = T - Delta(x) + V_bg(x) + 2 g |psi0|^2
    H_diag = -Delta_x + Vbg_x + 2.0 * g * abs2
    H_BdG = T + np.diag(H_diag)

    # Pairing: Delta_pair = g psi0^2
    Delta_pair = np.diag(g * psi0 ** 2)
    Delta_pair_conj = np.diag(g * np.conj(psi0) ** 2)

    # Homogeneous damping -gamma/2 on both sectors
    damping = -gamma / 2.0
    damp_mat = damping * np.eye(N, dtype=complex)

    # Drift blocks
    A11 = -1j * H_BdG + damp_mat           # particles
    A22 =  1j * H_BdG.conj() + damp_mat    # holes
    A12 = -1j * Delta_pair
    A21 =  1j * Delta_pair_conj

    A_top = np.hstack([A11, A12])
    A_bot = np.hstack([A21, A22])
    A = np.vstack([A_top, A_bot])

    logging.debug("Drift matrix A built: shape=%s", A.shape)
    return A


def compute_spectrum(A: np.ndarray) -> dict:
    logging.info("Diagonalizing drift matrix A (size %d x %d)...", A.shape[0], A.shape[1])
    eigvals, eigvecs = np.linalg.eig(A)
    logging.info("Eigen-decomposition completed.")

    lambda_vals = eigvals
    omega_vals = -1j * lambda_vals

    growth_rates = lambda_vals.real
    freqs = omega_vals.real

    idx_sort = np.argsort(freqs)
    freqs_sorted = freqs[idx_sort]
    growth_sorted = growth_rates[idx_sort]
    lambda_sorted = lambda_vals[idx_sort]

    logging.debug(
        "Spectrum: min(Re lambda)=%g, max(Re lambda)=%g",
        float(growth_sorted.min()),
        float(growth_sorted.max()),
    )
    logging.debug(
        "Spectrum: min(Re omega)=%g, max(Re omega)=%g",
        float(freqs_sorted.min()),
        float(freqs_sorted.max()),
    )

    diagnostics = {
        "lambda_min_real": float(growth_sorted.min()),
        "lambda_max_real": float(growth_sorted.max()),
        "omega_min_real": float(freqs_sorted.min()),
        "omega_max_real": float(freqs_sorted.max()),
    }

    return {
        "lambda_sorted": lambda_sorted,
        "freqs_sorted": freqs_sorted,
        "growth_sorted": growth_sorted,
        "diagnostics": diagnostics,
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_spectrum_json(output_prefix: str,
                       params: ModelParams,
                       mode: str,
                       x: np.ndarray,
                       Delta_x: np.ndarray,
                       Vbg_x: np.ndarray,
                       spectrum: dict) -> str:
    """Save spectrum data and metadata to a JSON file."""
    out_path = output_prefix + "_spectrum.json"

    data = {
        "params": asdict(params),
        "mode": mode,
        "x": x.tolist(),
        "Delta_x": Delta_x.tolist(),
        "Vbg_x": Vbg_x.tolist(),
        "lambda_sorted_real": spectrum["lambda_sorted"].real.tolist(),
        "lambda_sorted_imag": spectrum["lambda_sorted"].imag.tolist(),
        "freqs_sorted": spectrum["freqs_sorted"].tolist(),
        "growth_sorted": spectrum["growth_sorted"].tolist(),
        "diagnostics": spectrum["diagnostics"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logging.info("Spectrum JSON saved to %s", out_path)
    return out_path


def make_publication_quality_figure(output_prefix: str,
                                    spectrum: dict,
                                    params: ModelParams) -> str:
    freqs = spectrum["freqs_sorted"]
    growth = spectrum["growth_sorted"]
    num_modes = len(freqs)
    n = np.arange(num_modes)

    # Typographical settings (rough AIP style)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "dejavuserif",
    })

    # Wider and less tall for two side-by-side panels
    fig_width = 6.8   # inches (approx. 2 columns or rescaled to 1 column)
    fig_height = 2.4  # inches
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Left panel: Re(omega_n)
    ax1 = axes[0]
    ax1.plot(n, freqs, marker="o", linestyle="none", markersize=2)
    ax1.set_xlabel(r"Mode index $n$")
    ax1.set_ylabel(r"$\mathrm{Re}\,\omega_n$")
    ax1.text(0.03, 0.9, "(a)", transform=ax1.transAxes)

    # Right panel: Re(lambda_n) with reference line at -gamma/2
    ax2 = axes[1]
    gamma_half = -params.gamma / 2.0

    ax2.axhline(gamma_half, linestyle="--", linewidth=0.7)
    ax2.plot(n, growth, marker="o", linestyle="none", markersize=2)
    ax2.set_xlabel(r"Mode index $n$")
    ax2.set_ylabel(r"$\mathrm{Re}\,\lambda_n$")

    ax2.text(
        0.03,
        0.1,
        r"$\mathrm{Re}\,\lambda_n = -\gamma/2$",
        transform=ax2.transAxes,
        fontsize=8,
    )
    ax2.text(0.03, 0.9, "(b)", transform=ax2.transAxes)

    # Centre y-limits for lambda around -gamma/2 with a margin
    margin = 0.2 * abs(gamma_half) if gamma_half != 0.0 else 0.01
    ax2.set_ylim(gamma_half - margin, gamma_half + margin)

    # X-ticks: a few evenly spaced indices (same on both panels)
    xticks = np.linspace(0, num_modes - 1, 6, dtype=int)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)

    fig.tight_layout()
    out_path = output_prefix + "_spectrum.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info("Spectrum figure saved to %s", out_path)
    return out_path


def save_profiles_uniform(psi0: np.ndarray, Delta_x: np.ndarray, Vbg_x: np.ndarray) -> None:
    """Save uniform test-case profiles."""
    np.save("psi0_uniform.npy", psi0)
    np.save("Delta_x_uniform.npy", Delta_x)
    np.save("Vbg_x_uniform.npy", Vbg_x)
    logging.info("Saved psi0_uniform.npy, Delta_x_uniform.npy, Vbg_x_uniform.npy.")


def save_profiles_horizon(psi0: np.ndarray, Delta_x: np.ndarray, Vbg_x: np.ndarray) -> None:
    """Save toy horizon profiles."""
    np.save("psi0_horizon.npy", psi0)
    np.save("Delta_x.npy", Delta_x)
    np.save("Vbg_x.npy", Vbg_x)
    logging.info("Saved psi0_horizon.npy, Delta_x.npy, Vbg_x.npy.")


# -----------------------------------------------------------------------------
# Argument parsing and main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BdG / drift spectrum for 1D driven–dissipative polariton flow "
            "(self-contained: creates any needed .npy profiles)."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["uniformTestCase", "analogHorizonSpectrum"],
        default="uniformTestCase",
        help="Run mode: 'uniformTestCase' or 'analogHorizonSpectrum'.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="bdg_1d",
        help="Prefix for output files (PNG, JSON, log).",
    )

    # Model parameters
    parser.add_argument("--L", type=float, default=100.0, help="System length (dimensionless).")
    parser.add_argument("--N", type=int, default=256, help="Number of grid points.")
    parser.add_argument("--m-eff", type=float, default=1.0, help="Effective mass (scaled units).")
    parser.add_argument("--g", type=float, default=0.01, help="Interaction strength g.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Loss rate gamma.")
    parser.add_argument("--Delta0", type=float, default=0.0, help="Uniform detuning Delta0.")

    # Uniform test-case parameters
    parser.add_argument(
        "--psi0-amp",
        type=float,
        default=1.0,
        help="Amplitude for uniform psi0 in uniformTestCase mode.",
    )
    parser.add_argument(
        "--psi0-phase-gradient",
        type=float,
        default=0.0,
        help="Phase gradient k for uniform psi0 ~ amp * exp(i k x).",
    )

    # Toy horizon parameters
    parser.add_argument(
        "--horizon-core-amp",
        type=float,
        default=1.0,
        help="Core amplitude for toy analog-horizon psi0.",
    )
    parser.add_argument(
        "--horizon-bg-amp",
        type=float,
        default=0.1,
        help="Background amplitude for toy analog-horizon psi0.",
    )
    parser.add_argument(
        "--horizon-radius",
        type=float,
        default=20.0,
        help="Core radius (|x| < R) for toy analog-horizon psi0.",
    )
    parser.add_argument(
        "--horizon-smoothing",
        type=float,
        default=5.0,
        help="Smoothing length for tanh transition in toy analog-horizon psi0.",
    )
    parser.add_argument(
        "--horizon-phase-gradient",
        type=float,
        default=0.0,
        help="Phase gradient k for toy analog-horizon psi0.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_path = args.output_prefix + ".log"
    setup_logging(log_path)

    logging.info("Starting BdG 1D spectrum computation.")
    logging.info("Mode: %s", args.mode)
    logging.info("Output prefix: %s", args.output_prefix)

    params = ModelParams(
        L=args.L,
        N=args.N,
        m_eff=args.m_eff,
        g=args.g,
        gamma=args.gamma,
        Delta0=args.Delta0,
        psi0_amp=args.psi0_amp,
        psi0_phase_gradient=args.psi0_phase_gradient,
        horizon_core_amp=args.horizon_core_amp,
        horizon_bg_amp=args.horizon_bg_amp,
        horizon_radius=args.horizon_radius,
        horizon_smoothing=args.horizon_smoothing,
        horizon_phase_gradient=args.horizon_phase_gradient,
    )

    logging.debug("Model parameters: %s", params)

    # Grid
    x = build_grid(params)

    # Build psi0, Delta_x, V_bg depending on mode
    if args.mode == "uniformTestCase":
        logging.info("Building uniform test-case profiles.")
        psi0 = build_uniform_psi0(params, x)
        Delta_x = params.Delta0 * np.ones_like(x)
        Vbg_x = np.zeros_like(x)
        save_profiles_uniform(psi0, Delta_x, Vbg_x)
    else:  # analogHorizonSpectrum
        logging.info("Building toy analog-horizon profiles.")
        psi0 = build_toy_horizon_psi0(params, x)
        Delta_x = params.Delta0 * np.ones_like(x)
        Vbg_x = np.zeros_like(x)
        save_profiles_horizon(psi0, Delta_x, Vbg_x)

    # Build drift matrix and compute spectrum
    A = build_drift_matrix(params, x, psi0, Delta_x, Vbg_x)
    spectrum = compute_spectrum(A)

    # Save JSON + figure
    json_path = save_spectrum_json(args.output_prefix, params, args.mode, x, Delta_x, Vbg_x, spectrum)
    fig_path = make_publication_quality_figure(args.output_prefix, spectrum, params)

    logging.info("Computation finished successfully.")
    logging.info("JSON: %s", json_path)
    logging.info("Figure: %s", fig_path)
    logging.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
