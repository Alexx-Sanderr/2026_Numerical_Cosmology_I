import numpy as np
import pytest
import numcosmo_II as nc

from numcosmo_II import lpt1, lpt2

@pytest.fixture
def simulation_results():
    L = 100.0
    Ng = 16
    aref = 1.0
    a_ini = 0.05
    ai = 1.0

    cosmo_params = {
        "h": 0.67,
        "Omega_m": 0.315,
        "Omega_lambda": 0.685,}

    lpt_sim = nc.LPT(
        L=L,
        Ng=Ng,
        aref=aref,
        h=cosmo_params["h"],
        Omega_m=cosmo_params["Omega_m"],
        Omega_lambda=cosmo_params["Omega_lambda"],
        a_ini=a_ini,
        ai=ai,)

    lpt_sim.run(seed=42)

    gf = lpt_sim.GF
    delta_1_x = lpt_sim.delta_x
    delta_1_k = lpt_sim.delta_ref_k

    kx, ky, kz, k = gf.normal_modes(gf.L, gf.V, gf.Ng)
    k2 = k**2
    mask = k2 > 0

    phi1_k = np.zeros_like(delta_1_k, dtype=complex)
    phi1_k[mask] = -delta_1_k[mask] / k2[mask]

    lpt2_sim = lpt2.LPT2(
        L=L,
        Ng=Ng,
        aref=aref,
        h=cosmo_params["h"],
        Omega_m=cosmo_params["Omega_m"],
        Omega_lambda=cosmo_params["Omega_lambda"],
        a_ini=a_ini,
        ai=ai,
        GF=gf,)

    source2_x, source2_k, phi2_k, phi2_x = (lpt2_sim.Second_Order_Potential(phi1_k=phi1_k))
    D1 = lpt_sim.D1_ai


    D2 = -(3.0 / 7.0) * D1**2

    x_p_2lpt = lpt2_sim.particle_positions_2lpt(
        q=lpt_sim.q,
        Psi1_q=lpt_sim.Psi_q,
        D2=D2,)

    x_p_za = lpt_sim.x_p.copy()
    delta_za_x = lpt1.particles_to_mesh_ngp(lpt_sim)
    delta_za_k = (gf.V / Ng**3) * np.fft.fftn(delta_za_x)

    try:
        lpt_sim.x_p = x_p_2lpt
        delta_2lpt_x = lpt1.particles_to_mesh_ngp(lpt_sim)
        delta_2lpt_k = (gf.V / Ng**3) * np.fft.fftn(delta_2lpt_x)
    finally:
        lpt_sim.x_p = x_p_za

    return {
        "lpt_sim": lpt_sim,
        "lpt2_sim": lpt2_sim,
        "delta_1_x": delta_1_x,
        "delta_1_k": delta_1_k,
        "source2_x": source2_x,
        "source2_k": source2_k,
        "phi2_k": phi2_k,
        "phi2_x": phi2_x,
        "x_p_2lpt": x_p_2lpt,
        "x_p_za": x_p_za,
        "delta_za_x": delta_za_x,
        "delta_za_k": delta_za_k,
        "delta_2lpt_x": delta_2lpt_x,
        "delta_2lpt_k": delta_2lpt_k,
    }


def test_second_order_potential_is_finite(simulation_results):
    results = simulation_results

    assert np.all(np.isfinite(results["source2_x"]))
    assert np.all(np.isfinite(results["source2_k"]))
    assert np.all(np.isfinite(results["phi2_k"]))
    assert np.all(np.isfinite(results["phi2_x"]))


def test_second_order_potential_has_zero_mean_mode(simulation_results):
    phi2_k = simulation_results["phi2_k"]

    assert phi2_k[0, 0, 0] == 0.0


def test_2lpt_positions_are_periodic_and_finite(simulation_results):
    results = simulation_results
    L = results["lpt_sim"].L
    positions = results["x_p_2lpt"]

    assert positions.shape == results["lpt_sim"].q.shape
    assert np.all(np.isfinite(positions))
    assert np.all(positions >= 0.0)
    assert np.all(positions < L)


def test_density_meshes_are_finite_and_have_zero_mean(simulation_results,):
    results = simulation_results

    for name in ("delta_za_x", "delta_2lpt_x"):
        delta = results[name]

        assert delta.shape == (16, 16, 16)
        assert np.all(np.isfinite(delta))
        assert np.mean(delta) == pytest.approx(0.0, abs=1e-12)


def test_fourier_density_fields_are_finite(simulation_results):
    results = simulation_results

    assert np.all(np.isfinite(results["delta_1_k"]))
    assert np.all(np.isfinite(results["delta_za_k"]))
    assert np.all(np.isfinite(results["delta_2lpt_k"]))


def test_za_positions_are_restored(simulation_results):
    results = simulation_results

    np.testing.assert_array_equal(
        results["lpt_sim"].x_p,
        results["x_p_za"],)


def test_2lpt_differs_from_za(simulation_results):
    results = simulation_results

    assert not np.allclose(
        results["x_p_2lpt"],
        results["x_p_za"],)


def test_statistics_can_be_measured(simulation_results):
    results = simulation_results
    spt = nc.SPTStatistics(results["lpt_sim"].GF)

    R_array = np.linspace(4.0, 12.0, 5)

    fields = [
        (
            results["delta_1_x"],
            results["delta_1_k"],
        ),
        (
            results["delta_za_x"],
            results["delta_za_k"],
        ),
        (
            results["delta_2lpt_x"],
            results["delta_2lpt_k"],
        ),
    ]

    for delta_x, delta_k in fields:
        k_b, B_meas, B_tree = spt.spt_equilateral_bispectrum(delta_k,num_bins=4,)
        _, S3 = spt.spt_S3_profile(delta_x, R_array)

        assert len(k_b) == len(B_meas)
        assert len(k_b) == len(B_tree)
        assert len(S3) == len(R_array)

        assert np.all(np.isfinite(k_b))
        assert np.all(np.isfinite(B_meas))
        assert np.all(np.isfinite(S3))
