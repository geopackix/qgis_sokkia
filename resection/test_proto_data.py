"""Test mit den echten Protokoll-Daten aus Stationierung #1.

Vergleicht verschiedene Gewichtungsstrategien.
"""
import numpy as np
import math
import sys
sys.path.insert(0, '.')
from resection import resection


def gon_to_rad(g):
    return g * math.pi / 200.0

def rad_to_gon(r):
    return r * 200.0 / math.pi


# Messdaten aus Stationierung #1
obs = [
    {'name': '103.1', 'X': 567554.0122, 'Y': 5300105.1033, 'Z': 670.8545,
     'hz_gon': 522.6550, 'za_gon': 98.4186, 'sd_m': 10.0830},
    {'name': '103.2', 'X': 567554.1549, 'Y': 5300089.5454, 'Z': 671.1308,
     'hz_gon': 570.3311, 'za_gon': 98.4291, 'sd_m': 21.3531},
    {'name': '103.3', 'X': 567531.2033, 'Y': 5300075.3058, 'Z': 672.0350,
     'hz_gon': 624.2826, 'za_gon': 97.4629, 'sd_m': 35.9174},
    {'name': '103.6', 'X': 567548.6628, 'Y': 5300097.0415, 'Z': 668.8560,
     'hz_gon': 578.3264, 'za_gon': 106.9613, 'sd_m': 12.3525},
]

P = np.array([[o['X'], o['Y'], o['Z']] for o in obs])
sd = np.array([o['sd_m'] for o in obs])
v_ang = np.array([gon_to_rad(100.0 - o['za_gon']) for o in obs])
hz_ang = np.array([gon_to_rad(o['hz_gon'] % 400.0) for o in obs])
n = len(obs)


def calc_residuals(position, z0_gon):
    """Berechne Hz/SD/ZA Residuen."""
    results = []
    for o in obs:
        diff = np.array([o['X'], o['Y'], o['Z']]) - position
        sd_calc = float(np.linalg.norm(diff))
        dh = math.sqrt(diff[0]**2 + diff[1]**2)
        if dh < 1e-10:
            dh = 1e-10
        za_calc_gon = 100.0 - rad_to_gon(math.atan2(diff[2], dh))
        sd_res_mm = (o['sd_m'] - sd_calc) * 1000.0
        za_res_mgon = (o['za_gon'] - za_calc_gon) * 1000.0
        t_rad = math.atan2(diff[0], diff[1])
        t_gon = rad_to_gon(t_rad) % 400.0
        hz_res_gon = (t_gon - z0_gon - o['hz_gon']) % 400.0
        if hz_res_gon > 200.0:
            hz_res_gon -= 400.0
        hz_res_mgon = hz_res_gon * 1000.0
        results.append((o['name'], hz_res_mgon, sd_res_mm, za_res_mgon))
    return results


def print_result(label, result, z0_gon):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    X_P, Y_P, Z_P = result.position
    print(f"  Position: X={X_P:.4f}  Y={Y_P:.4f}  Z={Z_P:.4f}")
    print(f"  z0 = {z0_gon:.4f} gon")
    print(f"  sigma0 = {result.sigma0:.4f}   DOF = {result.dof}")
    print(f"  std: sX={result.std_dev[0]:.4f} sY={result.std_dev[1]:.4f} sZ={result.std_dev[2]:.4f}")
    print(f"  {'Punkt':<8} {'vHz[mgon]':>10} {'vSD[mm]':>10} {'vZA[mgon]':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for name, hz, sd_r, za in calc_residuals(result.position, z0_gon):
        print(f"  {name:<8} {hz:>+10.1f} {sd_r:>+10.2f} {za:>+10.1f}")


def z0_posthoc(position):
    """z0 post-hoc berechnen (Kreismittel)."""
    z0_list = []
    for o in obs:
        diff = np.array([o['X'], o['Y'], o['Z']]) - position
        t_rad = math.atan2(diff[0], diff[1])
        t_gon = rad_to_gon(t_rad) % 400.0
        z0_i = (t_gon - o['hz_gon']) % 400.0
        z0_list.append(z0_i)
    sins = [math.sin(gon_to_rad(a)) for a in z0_list]
    coss = [math.cos(gon_to_rad(a)) for a in z0_list]
    return (rad_to_gon(math.atan2(sum(sins)/len(sins), sum(coss)/len(coss)))) % 400.0


# ── TEST D: Nur SD + ZA (bisheriges Verhalten, OHNE Hz) ──────────────────────
result_d = resection(P, measured_slant_distances=sd, measured_v_angles=v_ang)
z0_d = z0_posthoc(result_d.position)
print_result("TEST D: Nur SD+ZA, gleiche Gewichte (bisheriges Verhalten)", result_d, z0_d)

# ── TEST B: SD + Hz + ZA, gleiche Gewichte (1.0) ─────────────────────────────
result_b = resection(P, measured_slant_distances=sd, measured_v_angles=v_ang,
                     measured_hz_angles=hz_ang)
z0_b = rad_to_gon(result_b.orientation) % 400.0
print_result("TEST B: SD+Hz+ZA, gleiche Gewichte", result_b, z0_b)

# ── TEST C: SD + Hz + ZA, moderate Feldgewichte ──────────────────────────────
sigma_sd_c = 0.005  # 5 mm
sigma_hz_c = gon_to_rad(0.010)  # 10 mgon
sigma_za_c = gon_to_rad(0.050)  # 50 mgon (ZA ist schlechter)
w_sd_c = np.full(n, 1.0 / sigma_sd_c**2)
w_hz_c = np.full(n, 1.0 / sigma_hz_c**2)
w_za_c = np.full(n, 1.0 / sigma_za_c**2)
result_c = resection(P, measured_slant_distances=sd, measured_v_angles=v_ang,
                     measured_hz_angles=hz_ang, weights_slant_distances=w_sd_c,
                     weights_v_angles=w_za_c, weights_hz_angles=w_hz_c)
z0_c = rad_to_gon(result_c.orientation) % 400.0
print_result("TEST C: SD+Hz+ZA, Feldgewichte (SD=5mm, Hz=10mgon, ZA=50mgon)", result_c, z0_c)

# ── TEST A: SD + Hz + ZA, strenge geodaetische Gewichte ──────────────────────
sigma_sd_a = 0.003  # 3 mm
sigma_angle_a = gon_to_rad(0.001)  # 1 mgon
w_sd_a = np.full(n, 1.0 / sigma_sd_a**2)
w_angle_a = np.full(n, 1.0 / sigma_angle_a**2)
result_a = resection(P, measured_slant_distances=sd, measured_v_angles=v_ang,
                     measured_hz_angles=hz_ang, weights_slant_distances=w_sd_a,
                     weights_v_angles=w_angle_a, weights_hz_angles=w_angle_a)
z0_a = rad_to_gon(result_a.orientation) % 400.0
print_result("TEST A: SD+Hz+ZA, strenge Gewichte (SD=3mm, angle=1mgon)", result_a, z0_a)
