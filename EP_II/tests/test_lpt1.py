import sys
from pathlib import Path
import numpy as np
import pytest


from numcosmo_II import lpt1


@pytest.fixture
def lpt():
    simulation = lpt1.LPT(
        L=100,
        Ng=32,
        aref=1,
        h=0.67,
        Omega_m=0.3,
        Omega_lambda=0.7,
        Omega_r=0.0,
        w0_fld=-1.0,
        wa_fld=0.0,
        a_ini=0.01,
        ai=1.0,
    )

    simulation.run()

    return simulation

def calculate_divergence(lpt):
    """Calculate divergence of the displacement field in Fourier space."""

    div_Psi_k = (
        1j * lpt.kx * lpt.Psi_x_k
        + 1j * lpt.ky * lpt.Psi_y_k
        + 1j * lpt.kz * lpt.Psi_z_k
    )

    factor = lpt.GF.Ng**3 / lpt.GF.V
    div_Psi = np.fft.ifftn(factor * div_Psi_k).real

    return div_Psi


def calculate_curl(lpt):
    """Calculate curl of the displacement field in Fourier space."""

    curl_x_k = (
        1j * lpt.ky * lpt.Psi_z_k
        - 1j * lpt.kz * lpt.Psi_y_k
    )

    curl_y_k = (
        1j * lpt.kz * lpt.Psi_x_k
        - 1j * lpt.kx * lpt.Psi_z_k
    )

    curl_z_k = (
        1j * lpt.kx * lpt.Psi_y_k
        - 1j * lpt.ky * lpt.Psi_x_k
    )

    factor = lpt.GF.Ng**3 / lpt.GF.V

    curl_x = np.fft.ifftn(factor * curl_x_k).real
    curl_y = np.fft.ifftn(factor * curl_y_k).real
    curl_z = np.fft.ifftn(factor * curl_z_k).real

    return curl_x, curl_y, curl_z


def test_mean_density(lpt):
    """Test the mean of the Gaussian density contrast."""

    mean_density = np.mean(lpt.delta_x)

    print("\nMean density contrast:")
    print("Mean =", mean_density)


def test_hermitian_symmetry(lpt):
    """Test Hermitian symmetry of the Fourier density field."""

    delta_k = lpt.delta_ref_k
    Ng = lpt.Ng

    hermitian_error = 0.0

    for i in range(Ng):
        for j in range(Ng):
            for k in range(Ng):

                im = (-i) % Ng
                jm = (-j) % Ng
                km = (-k) % Ng

                error = np.abs(
                    delta_k[i, j, k]
                    - np.conj(delta_k[im, jm, km])
                )

                hermitian_error = max(
                    hermitian_error, error
                )

    print("\nHermitian symmetry:")
    print("Maximum error =", hermitian_error)


def test_imaginary_fft_residual(lpt):
    """Test the imaginary residual of the FFT of a real field."""

    delta_fft = np.fft.fftn(lpt.delta_x)

    residual_imaginary = np.max(
        np.abs(delta_fft.imag)
    )

    real_amplitude = np.max(
        np.abs(delta_fft.real)
    )

    relative_residual = (
        residual_imaginary / real_amplitude
        if real_amplitude > 0
        else 0.0
    )

    print("\nImaginary FFT residual:")
    print("Relative residual =", relative_residual)


def test_displacement(lpt):
    """Test the divergence and curl of the 1LPT displacement."""

    div_Psi = calculate_divergence(lpt)

    curl_x, curl_y, curl_z = calculate_curl(lpt)

    divergence_rms = np.sqrt(np.mean(div_Psi**2))

    curl_rms = np.sqrt(
        np.mean(
            curl_x**2 + curl_y**2 + curl_z**2
        )
    )

    print("\nDisplacement field:")
    print("Divergence RMS =", divergence_rms)
    print("Curl RMS =", curl_rms)

    # First-order LPT relation:
    # -div(Psi) = D1 * delta_ref

    lhs = -div_Psi
    rhs = lpt.D1_ai * lpt.delta_x

    error = lhs - rhs

    rhs_rms = np.sqrt(np.mean(rhs**2))
    error_rms = np.sqrt(np.mean(error**2))

    relative_error = (
        error_rms / rhs_rms
        if rhs_rms > 0
        else error_rms
    )

    print("\n1LPT relation:")
    print("-div(Psi) = D1 * delta_ref")
    print("Relative RMS error =", relative_error)


def test_total_mass(lpt):
    """Test mass conservation for the particle load."""

    total_mass = len(lpt.q)
    expected_mass = lpt.Ng**3

    mass_error = abs(total_mass - expected_mass)

    print("\nTotal particle mass:")
    print("Total mass =", total_mass)
    print("Expected mass =", expected_mass)
    print("Error =", mass_error)


def test_velocity_normalization(lpt):
    """Test the 1LPT particle velocity normalization."""

    H_i = lpt.cosmo.Ha(a=lpt.ai)

    v_expected = (
        lpt.f1_i * H_i * lpt.Psi_q
    )

    velocity_error = lpt.v_p - v_expected

    expected_rms = np.sqrt(
        np.mean(v_expected**2)
    )

    error_rms = np.sqrt(
        np.mean(velocity_error**2)
    )

    relative_error = (
        error_rms / expected_rms
        if expected_rms > 0
        else error_rms
    )

    print("\nVelocity normalization:")
    print("Relative RMS error =", relative_error)


def main():
    # Initialize and run the 1LPT simulation.
    lpt = lpt1.LPT(
        L=100,
        Ng=32,
        aref=1,
        h=0.67,
        Omega_m=0.3,
        Omega_lambda=0.7,
        Omega_r=0.0,
        w0_fld=-1.0,
        wa_fld=0.0,
        a_ini=0.01,
        ai=1.0,
    )

    lpt.run()

    # Run all tests.
    test_mean_density(lpt)
    test_hermitian_symmetry(lpt)
    test_imaginary_fft_residual(lpt)
    test_displacement(lpt)
    test_total_mass(lpt)
    test_velocity_normalization(lpt)


if __name__ == "__main__":
    main()