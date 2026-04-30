"""Tests für das erweiterte Resektionsmodul (JAG3D-konformes Modell)."""

import math
import numpy as np
import pytest

from resection_extended import resection_extended


def _make_synthetic_dataset(
    Xp: float = 1000.0,
    Yp: float = 2000.0,
    Zp: float = 100.0,
    o_rad: float = 1.234,
    ih: float = 1.50,
    th: float = 1.80,
    n_points: int = 5,
    seed: int = 42,
):
    """Erzeugt einen synthetischen Datensatz mit bekannten Parametern."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.1, 2 * math.pi - 0.1, n_points)
    distances = rng.uniform(50.0, 200.0, n_points)
    dz = rng.uniform(-10.0, 10.0, n_points)

    pts = np.zeros((n_points, 3))
    sd_meas = np.zeros(n_points)
    hz_meas = np.zeros(n_points)
    za_meas = np.zeros(n_points)
    th_arr = np.full(n_points, th)

    for i in range(n_points):
        # Punkt liegt mit horizontalem Azimut "angles[i]" und horizontaler Distanz
        # "distances[i]" sowie vertikaler Differenz "dz[i]" vom Standpunkt entfernt.
        u = distances[i] * math.sin(angles[i])  # dx (Rechtswert)
        v = distances[i] * math.cos(angles[i])  # dy (Hochwert)
        w_geom = dz[i]                          # dz_geometrisch (Punkt - Standpunkt)
        pts[i] = (Xp + u, Yp + v, Zp + w_geom)

        #  ih/th: w = (Pz - Zs) + th - ih
        w = w_geom + th - ih
        sd_meas[i] = math.sqrt(u * u + v * v + w * w)
        # Hz_obs = atan2(u, v) - o → hz_obs = bearing - o, also hz_obs + o = bearing
        # Tachymeterablesung: bearing - o
        hz_meas[i] = math.atan2(u, v) - o_rad
        s2D = math.sqrt(u * u + v * v)
        za_meas[i] = math.atan2(s2D, w)

    # Normieren auf [0, 2π)
    hz_meas = np.mod(hz_meas, 2 * math.pi)
    return pts, sd_meas, hz_meas, za_meas, th_arr, ih


def test_perfect_reconstruction_sd_hz_za():
    """Bei rauschfreien Daten muss die Position exakt rekonstruiert werden."""
    Xp, Yp, Zp, o_rad, ih, th = 1000.0, 2000.0, 100.0, 1.234, 1.5, 1.8
    pts, sd, hz, za, th_arr, ih_v = _make_synthetic_dataset(
        Xp=Xp, Yp=Yp, Zp=Zp, o_rad=o_rad, ih=ih, th=th)

    res = resection_extended(
        pts,
        measured_slant_distances=sd,
        measured_hz_angles=hz,
        measured_zenith_angles=za,
        instrument_height=ih_v,
        target_heights=th_arr,
        apply_earth_curvature=False,  # Synthetik berücksichtigt sie nicht
        refraction_coefficient=0.0,
    )

    np.testing.assert_allclose(res.position, [Xp, Yp, Zp], atol=1e-6)
    # Orientierung modulo 2π
    diff = (res.orientation - o_rad) % (2 * math.pi)
    diff = min(diff, 2 * math.pi - diff)
    assert diff < 1e-7
    assert res.sigma0 < 1e-3


def test_target_height_changes_z():
    """Ohne th-Berücksichtigung müsste Z um (th - ih) abweichen."""
    Xp, Yp, Zp, o_rad, ih, th = 1000.0, 2000.0, 100.0, 0.5, 1.5, 1.8
    pts, sd, hz, za, th_arr, ih_v = _make_synthetic_dataset(
        Xp=Xp, Yp=Yp, Zp=Zp, o_rad=o_rad, ih=ih, th=th)

    # Mit korrektem ih/th
    res_ok = resection_extended(
        pts, measured_slant_distances=sd, measured_hz_angles=hz,
        measured_zenith_angles=za, instrument_height=ih_v,
        target_heights=th_arr, apply_earth_curvature=False,
        refraction_coefficient=0.0,
    )
    assert abs(res_ok.position[2] - Zp) < 1e-6

    # Mit ih=th=0 (klassisches Modell): Z weicht ab
    res_naive = resection_extended(
        pts, measured_slant_distances=sd, measured_hz_angles=hz,
        measured_zenith_angles=za, instrument_height=0.0,
        target_heights=np.zeros_like(th_arr), apply_earth_curvature=False,
        refraction_coefficient=0.0,
    )
    # Bei ZA + SD wird der Versatz primär in Z abgebildet
    assert abs(res_naive.position[2] - Zp) > 0.1


def test_scale_estimation():
    """Maßstab als Unbekannte schätzen."""
    Xp, Yp, Zp, o_rad, ih, th = 1000.0, 2000.0, 100.0, 0.0, 0.0, 0.0
    true_scale = 1.0002
    pts, sd, hz, za, th_arr, ih_v = _make_synthetic_dataset(
        Xp=Xp, Yp=Yp, Zp=Zp, o_rad=o_rad, ih=ih, th=th, n_points=8)
    sd_scaled = sd / true_scale  # gemessen = wahr / scale (vgl. JAG3D)

    res = resection_extended(
        pts, measured_slant_distances=sd_scaled,
        measured_hz_angles=hz, measured_zenith_angles=za,
        instrument_height=ih_v, target_heights=th_arr,
        apply_earth_curvature=False, refraction_coefficient=0.0,
        estimate_scale_sd=True,
    )
    assert res.scale_sd is not None
    assert abs(res.scale_sd - true_scale) < 1e-6
    np.testing.assert_allclose(res.position, [Xp, Yp, Zp], atol=1e-5)


def test_minimum_observations_raises():
    pts = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        resection_extended(pts, measured_slant_distances=np.array([10.0]))


def test_redundancy_components_in_unit_interval():
    pts, sd, hz, za, th_arr, ih_v = _make_synthetic_dataset(n_points=6)
    res = resection_extended(
        pts, measured_slant_distances=sd,
        measured_hz_angles=hz, measured_zenith_angles=za,
        instrument_height=ih_v, target_heights=th_arr,
        apply_earth_curvature=False, refraction_coefficient=0.0,
    )
    assert res.redundancy_components.shape[0] == res.num_obs
    assert np.all(res.redundancy_components >= -1e-9)
    assert np.all(res.redundancy_components <= 1.0 + 1e-9)
    assert abs(np.sum(res.redundancy_components) - res.dof) < 1e-6
