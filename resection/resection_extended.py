"""
Erweiterte Resektion (Freie Stationierung, 3D) 

Im Gegensatz zum bestehenden ``resection``-Modul implementiert dieses Modul die
Beobachtungsgleichungen im konformen Modell direkt, ohne die Umwege über Lotabweichungen und:

* Berücksichtigung von Instrumenten- (``ih``) und Reflektorhöhe (``th``).
* Schräg- und Horizontalstrecke unterstützen optional einen Maßstab
  (``scale``) und eine Additionskonstante (``add``) als Zusatzunbekannte.
* Zenitwinkel mit Refraktions- und Erdkrümmungskorrektur
  ``geoCorr = (1 - k) * s_2D / (2 R)``.
* Pro Beobachtungstyp können A-priori-Sigmen vorgegeben werden, woraus die
  Gewichte ``w_i = 1 / sigma_i^2`` gebildet werden.
* Zusätzlich werden Redundanzanteile ``r_i`` und normierte Verbesserungen
  ``NV_i`` zurückgegeben.

Die Funktion bleibt vollständig parallel zum klassischen ``resection``;
der bestehende Berechnungsweg wird *nicht* verändert.

Geometrie-Konvention (kompatibel zum bestehenden Plugin):
* ``X`` = Rechtswert, ``Y`` = Hochwert, ``Z`` = Höhe.
* Horizontalrichtungen ``Hz`` werden geodätisch gemessen
  (``hz = atan2(dx, dy)`` von der Y-Achse im Uhrzeigersinn).
* Zenitwinkel ``z = atan2(dh, dz)`` mit ``dh = sqrt(dx² + dy²)``.

Vereinfachung der Gleichungen (Lotabweichungen = 0, keine Projektion):
``u = xe - xs``, ``v = ye - ys``, ``w = ze - zs + th - ih``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

DEFAULT_EARTH_RADIUS = 6_378_137.0
DEFAULT_REFRACTION = 0.13


@dataclass
class ResectionExtendedResult:
    """Ergebnis einer erweiterten Resektion."""

    position: np.ndarray              # (3,) Standpunkt X, Y, Z
    std_dev: np.ndarray               # (3,) Standardabweichungen X, Y, Z
    covariance: np.ndarray            # (3,3) Kovarianzmatrix der Position
    sigma0: float                      # Varianzfaktor (a posteriori)
    rms_residual: float                # RMS aller Residuen
    num_obs: int                       # Anzahl Beobachtungen
    num_unknowns: int                  # Anzahl Unbekannte
    dof: int                           # Freiheitsgrade
    redundancy: float                  # Gesamtredundanzgrad dof / num_obs
    orientation: Optional[float] = None     # Orientierungsunbekannte o (rad)
    scale_sd: Optional[float] = None        # Maßstabsfaktor Schrägstrecken
    add_sd: Optional[float] = None          # Additionskonstante Schrägstrecken (m)
    scale_hd: Optional[float] = None        # Maßstabsfaktor Horizontalstrecken
    add_hd: Optional[float] = None          # Additionskonstante Horizontalstrecken (m)
    refraction_coefficient: float = DEFAULT_REFRACTION
    earth_radius: float = DEFAULT_EARTH_RADIUS
    instrument_height: float = 0.0
    target_heights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Per-Beobachtung
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    redundancy_components: np.ndarray = field(default_factory=lambda: np.zeros(0))
    normalized_residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obs_labels: List[str] = field(default_factory=list)


def _build_w(diff_z: np.ndarray, th: np.ndarray, ih: float) -> np.ndarray:
    """w_i = (ze - zs) + th_i - ih ."""
    return diff_z + th - ih


def resection_extended(
    observed_points: np.ndarray,
    *,
    measured_slant_distances: Optional[np.ndarray] = None,
    measured_distances: Optional[np.ndarray] = None,
    measured_hz_angles: Optional[np.ndarray] = None,
    measured_zenith_angles: Optional[np.ndarray] = None,
    instrument_height: float = 0.0,
    target_heights: Optional[np.ndarray] = None,
    refraction_coefficient: float = DEFAULT_REFRACTION,
    apply_earth_curvature: bool = True,
    earth_radius: float = DEFAULT_EARTH_RADIUS,
    sigma_sd: float = 0.005,
    sigma_hd: float = 0.005,
    sigma_hz: float = 0.0015,    # rad (≈ 1 mgon)
    sigma_za: float = 0.0015,    # rad
    estimate_scale_sd: bool = False,
    estimate_add_sd: bool = False,
    estimate_scale_hd: bool = False,
    estimate_add_hd: bool = False,
    initial_position: Optional[np.ndarray] = None,
    initial_orientation: Optional[float] = None,
    max_iterations: int = 30,
    tolerance: float = 1e-9,
    obs_labels: Optional[List[str]] = None,
) -> ResectionExtendedResult:
    """Berechnet die freie Stationierung im konformen Modell.

    Parameters
    ----------
    observed_points : (N, 3) array_like
        Bekannte Anschlusspunktkoordinaten (X, Y, Z).
    measured_slant_distances, measured_distances : (N,) array_like, optional
        Schräg- bzw. Horizontalstrecken in Metern (jeweils auf die N
        ``observed_points`` bezogen).
    measured_hz_angles : (N,) array_like, optional
        Horizontalrichtungen in Radiant. Geodätische Konvention
        ``hz = atan2(dx, dy)``. Eine globale Orientierungsunbekannte ``o``
        wird automatisch mitgeschätzt.
    measured_zenith_angles : (N,) array_like, optional
        Zenitwinkel in Radiant. ``z = 0`` zenith, ``z = π/2`` horizontal.
    instrument_height : float
        Instrumentenhöhe ``ih`` über dem Standpunkt in Metern.
    target_heights : (N,) array_like, optional
        Reflektorhöhen pro Anschlusspunkt in Metern (Default: 0).
    refraction_coefficient : float
        Refraktionskoeffizient ``k`` für die Zenitwinkelkorrektur.
    apply_earth_curvature : bool
        Wenn ``True`` wird die Erdkrümmungskorrektur addiert.
    earth_radius : float
        Erdradius in Metern für die Refraktions-/Erdkrümmungskorrektur.
    sigma_* : float
        A-priori Standardabweichungen pro Beobachtungstyp.
    estimate_scale_*, estimate_add_* : bool
        Aktiviert die Schätzung eines Maßstabsfaktors bzw. einer
        Additionskonstante für Schräg- (``_sd``) bzw. Horizontalstrecken
        (``_hd``).
    initial_position : (3,) array_like, optional
        Näherungsposition. Default: Schwerpunkt der Anschlusspunkte mit
        kurzer Distanz-Vor-Iteration.
    initial_orientation : float, optional
        Näherungswert für ``o`` in Radiant.
    max_iterations, tolerance : int, float
        Iterationssteuerung.
    obs_labels : list[str], optional
        Optionale Beschriftung pro Anschlusspunkt für die Ausgabe.

    Returns
    -------
    ResectionExtendedResult
    """

    P = np.asarray(observed_points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("observed_points muss ein (N,3)-Array sein.")
    N = P.shape[0]

    if target_heights is None:
        th = np.zeros(N)
    else:
        th = np.asarray(target_heights, dtype=float).flatten()
        if th.shape[0] != N:
            raise ValueError("target_heights muss N Elemente haben.")

    ih = float(instrument_height)

    # --- Beobachtungen -----------------------------------------------------
    has_sd = measured_slant_distances is not None
    has_hd = measured_distances is not None
    has_hz = measured_hz_angles is not None
    has_za = measured_zenith_angles is not None

    if not (has_sd or has_hd or has_hz or has_za):
        raise ValueError("Mindestens eine Messungsart erforderlich.")

    sd = np.asarray(measured_slant_distances, dtype=float).flatten() if has_sd else None
    hd = np.asarray(measured_distances, dtype=float).flatten() if has_hd else None
    hz = np.asarray(measured_hz_angles, dtype=float).flatten() if has_hz else None
    za = np.asarray(measured_zenith_angles, dtype=float).flatten() if has_za else None

    for arr, name in ((sd, "measured_slant_distances"),
                      (hd, "measured_distances"),
                      (hz, "measured_hz_angles"),
                      (za, "measured_zenith_angles")):
        if arr is not None and arr.shape[0] != N:
            raise ValueError(f"{name} muss N={N} Elemente haben.")

    # Maßstab/Add nur freigeben, wenn entsprechende Messung vorhanden
    estimate_scale_sd = bool(estimate_scale_sd) and has_sd
    estimate_add_sd = bool(estimate_add_sd) and has_sd
    estimate_scale_hd = bool(estimate_scale_hd) and has_hd
    estimate_add_hd = bool(estimate_add_hd) and has_hd

    # --- Unbekannte: [X, Y, Z, (o), (m_sd), (a_sd), (m_hd), (a_hd)] -------
    idx = {"X": 0, "Y": 1, "Z": 2}
    n_u = 3
    if has_hz:
        idx["o"] = n_u
        n_u += 1
    if estimate_scale_sd:
        idx["m_sd"] = n_u
        n_u += 1
    if estimate_add_sd:
        idx["a_sd"] = n_u
        n_u += 1
    if estimate_scale_hd:
        idx["m_hd"] = n_u
        n_u += 1
    if estimate_add_hd:
        idx["a_hd"] = n_u
        n_u += 1

    # --- Initialisierung ---------------------------------------------------
    if initial_position is not None:
        X = np.asarray(initial_position, dtype=float).flatten().copy()
        if X.shape[0] != 3:
            raise ValueError("initial_position muss 3 Elemente haben.")
    else:
        X = np.mean(P, axis=0)
        # Robuste Initialisierung: bevorzugt Hz+SD/HD für Polaraufnahme
        if has_hz and (has_sd or has_hd):
            _dists_init = sd if has_sd else hd

            # Iterative Bootstrap: o schätzen → Polar → o verfeinern → Polar → ...
            for _bootstrap in range(10):
                _o_estimates = []
                for i in range(N):
                    u = P[i, 0] - X[0]
                    v = P[i, 1] - X[1]
                    _o_estimates.append(hz[i] - np.arctan2(u, v))
                _o_init = float(np.arctan2(
                    np.mean(np.sin(_o_estimates)),
                    np.mean(np.cos(_o_estimates))
                ))

                # Polaraufnahme: Standpunkt aus jedem Festpunkt rückrechnen
                _positions = []
                for i in range(N):
                    bearing = hz[i] + _o_init
                    d = _dists_init[i]
                    _sx = P[i, 0] - d * np.sin(bearing)
                    _sy = P[i, 1] - d * np.cos(bearing)
                    _sz = P[i, 2]
                    _positions.append([_sx, _sy, _sz])
                X_new = np.mean(_positions, axis=0)

                # Konvergenzcheck
                if np.max(np.abs(X_new - X)) < tolerance:
                    X = X_new
                    break
                X = X_new

            # Z-Komponente aus ZA verfeinern
            if has_za:
                _z_ests = []
                for i in range(N):
                    u = P[i, 0] - X[0]
                    v = P[i, 1] - X[1]
                    s2D = max(np.sqrt(u * u + v * v), 1e-6)
                    # za = atan2(s2D, w) => w = s2D / tan(za)
                    if abs(za[i]) > 1e-6 and abs(za[i] - np.pi) > 1e-6:
                        w_est = s2D / np.tan(za[i])
                        # w = (P_z - X_z) + th - ih => X_z = P_z + th - ih - w
                        _z_ests.append(P[i, 2] + th[i] - ih - w_est)
                if _z_ests:
                    X[2] = np.mean(_z_ests)
        elif has_sd or has_hd:
            # Nur Distanzen: Gedämpfte Vor-Iteration
            for _ in range(max_iterations):
                rows_l: List[float] = []
                rows_A: List[List[float]] = []
                if has_sd:
                    for i in range(N):
                        u = P[i, 0] - X[0]
                        v = P[i, 1] - X[1]
                        w = P[i, 2] - X[2] + th[i] - ih
                        d = max(np.sqrt(u * u + v * v + w * w), 1e-10)
                        rows_l.append(sd[i] - d)
                        rows_A.append([-u / d, -v / d, -w / d])
                if has_hd:
                    for i in range(N):
                        u = P[i, 0] - X[0]
                        v = P[i, 1] - X[1]
                        d = max(np.sqrt(u * u + v * v), 1e-10)
                        rows_l.append(hd[i] - d)
                        rows_A.append([-u / d, -v / d, 0.0])
                A0 = np.asarray(rows_A)
                l0 = np.asarray(rows_l)
                try:
                    dX = np.linalg.lstsq(A0, l0, rcond=None)[0]
                except np.linalg.LinAlgError:
                    break
                # Schritt-Dämpfung
                _max_step = np.max(np.abs(dX))
                _mean_d = np.mean(sd) if has_sd else np.mean(hd)
                if _max_step > _mean_d:
                    dX = dX * (_mean_d / _max_step)
                X = X + dX
                if np.max(np.abs(dX)) < tolerance:
                    break

    o = 0.0
    if has_hz:
        if initial_orientation is not None:
            o = float(initial_orientation)
        else:
            angles = []
            for i in range(N):
                u = P[i, 0] - X[0]
                v = P[i, 1] - X[1]
                angles.append(hz[i] - np.arctan2(u, v))
            o = float(np.arctan2(np.mean(np.sin(angles)),
                                 np.mean(np.cos(angles))))

    m_sd = 1.0
    a_sd = 0.0
    m_hd = 1.0
    a_hd = 0.0

    # --- Gewichte (1 / sigma^2) -------------------------------------------
    w_sd = 1.0 / (sigma_sd ** 2) if (has_sd and sigma_sd > 0) else 1.0
    w_hd = 1.0 / (sigma_hd ** 2) if (has_hd and sigma_hd > 0) else 1.0
    w_hz = 1.0 / (sigma_hz ** 2) if (has_hz and sigma_hz > 0) else 1.0
    w_za = 1.0 / (sigma_za ** 2) if (has_za and sigma_za > 0) else 1.0

    # --- Hilfsfunktionen für Beobachtungsgleichungen -----------------------
    def _build_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        rows_A: List[np.ndarray] = []
        rows_l: List[float] = []
        rows_w: List[float] = []
        labels: List[str] = []

        u_arr = P[:, 0] - X[0]
        v_arr = P[:, 1] - X[1]
        w_arr = (P[:, 2] - X[2]) + th - ih  # vereinfachtes w

        # Schrägstrecke: s_calc = (sqrt(u²+v²+w²) - a_sd) / m_sd
        if has_sd:
            for i in range(N):
                u, v, w = u_arr[i], v_arr[i], w_arr[i]
                d3 = max(np.sqrt(u * u + v * v + w * w), 1e-12)
                s_calc = (d3 - a_sd) / m_sd
                row = np.zeros(n_u)
                # ∂s_calc/∂Xs = (1/m_sd) · ∂d3/∂Xs ; ∂d3/∂Xs = -u/d3
                row[idx["X"]] = -u / (m_sd * d3)
                row[idx["Y"]] = -v / (m_sd * d3)
                row[idx["Z"]] = -w / (m_sd * d3)
                if estimate_scale_sd:
                    row[idx["m_sd"]] = -(d3 - a_sd) / (m_sd * m_sd)
                if estimate_add_sd:
                    row[idx["a_sd"]] = -1.0 / m_sd
                rows_A.append(row)
                rows_l.append(sd[i] - s_calc)
                rows_w.append(w_sd)
                labels.append(f"SD #{i}")

        # Horizontalstrecke: d_calc = (sqrt(u²+v²) - a_hd) / m_hd
        if has_hd:
            for i in range(N):
                u, v = u_arr[i], v_arr[i]
                d2 = max(np.sqrt(u * u + v * v), 1e-12)
                d_calc = (d2 - a_hd) / m_hd
                row = np.zeros(n_u)
                row[idx["X"]] = -u / (m_hd * d2)
                row[idx["Y"]] = -v / (m_hd * d2)
                # Z-Komponente entfällt
                if estimate_scale_hd:
                    row[idx["m_hd"]] = -(d2 - a_hd) / (m_hd * m_hd)
                if estimate_add_hd:
                    row[idx["a_hd"]] = -1.0 / m_hd
                rows_A.append(row)
                rows_l.append(hd[i] - d_calc)
                rows_w.append(w_hd)
                labels.append(f"HD #{i}")

        # Horizontalrichtung: hz_calc = atan2(u, v) - o  (geodätisch)
        if has_hz:
            for i in range(N):
                u, v = u_arr[i], v_arr[i]
                denom = max(u * u + v * v, 1e-20)
                hz_calc = np.arctan2(u, v) - o
                res = hz[i] - hz_calc
                # auf [-π, π] normieren
                res = np.arctan2(np.sin(res), np.cos(res))
                row = np.zeros(n_u)
                row[idx["X"]] = -v / denom    # ∂atan2(u,v)/∂xs = -v/(u²+v²) (mit dxs = -1)
                row[idx["Y"]] = u / denom
                row[idx["o"]] = -1.0
                rows_A.append(row)
                rows_l.append(res)
                rows_w.append(w_hz)
                labels.append(f"Hz #{i}")

        # Zenitwinkel: z_calc = atan2(s2D, w) + (1 - k_eff) * s2D / (2 R)
        # k_eff: ohne Erdkrümmung wirkt nur -k * s / (2R)
        if has_za:
            for i in range(N):
                u, v, w = u_arr[i], v_arr[i], w_arr[i]
                s2D = max(np.sqrt(u * u + v * v), 1e-12)
                d3sq = s2D * s2D + w * w
                z_geom = np.arctan2(s2D, w)
                corr_factor = (1.0 - refraction_coefficient) if apply_earth_curvature \
                    else (-refraction_coefficient)
                geo_corr = corr_factor * s2D / (2.0 * earth_radius)
                z_calc = z_geom + geo_corr
                res = za[i] - z_calc
                # ∂atan2(s2D, w) /∂(u,v,w):  d/dxe = w*u/(s2D*d3²),  d/dye = w*v/(s2D*d3²),  d/dze = -s2D/d3²
                # mit ∂(u,v,w)/∂(Xs,Ys,Zs) = -I → Vorzeichen drehen.
                d_z_xs = -(w * u / (s2D * d3sq))
                d_z_ys = -(w * v / (s2D * d3sq))
                d_z_zs = -(-s2D / d3sq)
                # Geodätische Korrektur: ∂(corr * s2D/(2R)) /∂Xs = corr * (-u/s2D)/(2R)
                d_corr_xs = corr_factor * (-u / s2D) / (2.0 * earth_radius)
                d_corr_ys = corr_factor * (-v / s2D) / (2.0 * earth_radius)
                row = np.zeros(n_u)
                row[idx["X"]] = d_z_xs + d_corr_xs
                row[idx["Y"]] = d_z_ys + d_corr_ys
                row[idx["Z"]] = d_z_zs
                rows_A.append(row)
                rows_l.append(res)
                rows_w.append(w_za)
                labels.append(f"ZA #{i}")

        A = np.vstack(rows_A)
        l = np.asarray(rows_l)
        w_vec = np.asarray(rows_w)
        return A, l, w_vec, labels

    # --- Gauß-Newton Iteration --------------------------------------------
    A = l = w_vec = None
    labels: List[str] = []
    for _it in range(max_iterations):
        A, l, w_vec, labels = _build_system()
        WA = A * w_vec[:, None]
        N_mat = A.T @ WA
        n_vec = A.T @ (w_vec * l)
        try:
            dx = np.linalg.solve(N_mat, n_vec)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Singuläre Normalgleichungsmatrix – Konfiguration nicht "
                "lösbar."
            ) from exc

        X = X + dx[:3]
        if has_hz:
            o = o + dx[idx["o"]]
        if estimate_scale_sd:
            m_sd = m_sd + dx[idx["m_sd"]]
        if estimate_add_sd:
            a_sd = a_sd + dx[idx["a_sd"]]
        if estimate_scale_hd:
            m_hd = m_hd + dx[idx["m_hd"]]
        if estimate_add_hd:
            a_hd = a_hd + dx[idx["a_hd"]]

        if np.max(np.abs(dx)) < tolerance:
            break

    # --- Endgültige Residuen ----------------------------------------------
    A, l, w_vec, labels = _build_system()
    if obs_labels is not None:
        # Ersetze "<typ> #i" durch Punktbezeichnung
        new_labels = []
        for lab in labels:
            try:
                typ, idx_part = lab.split("#")
                i = int(idx_part)
                new_labels.append(f"{typ.strip()} {obs_labels[i]}")
            except (ValueError, IndexError):
                new_labels.append(lab)
        labels = new_labels

    n_obs = A.shape[0]
    dof = n_obs - n_u
    if dof <= 0:
        raise ValueError(
            f"Zu wenige Freiheitsgrade ({dof}) – mehr Beobachtungen oder "
            "weniger Unbekannte erforderlich."
        )

    P_diag = w_vec
    WA = A * P_diag[:, None]
    N_mat = A.T @ WA
    try:
        Qxx_full = np.linalg.inv(N_mat)
    except np.linalg.LinAlgError:
        Qxx_full = np.eye(n_u) * np.nan

    # vᵀ P v
    ssr = float(np.sum(P_diag * l * l))
    sigma0_sq = ssr / dof
    sigma0 = float(np.sqrt(sigma0_sq))

    Qxx = sigma0_sq * Qxx_full
    cov_pos = Qxx[:3, :3]
    std_dev = np.sqrt(np.abs(np.diag(cov_pos)))

    # Redundanzanteile r_i = (Qvv * P)_ii  mit Qvv = Q_ll - A Qxx_full Aᵀ
    # Q_ll = diag(1/P)
    try:
        Qvv = np.diag(1.0 / P_diag) - A @ Qxx_full @ A.T
        r_i = np.clip(np.diag(Qvv) * P_diag, 0.0, 1.0)
    except Exception:
        r_i = np.full(n_obs, np.nan)

    # Normierte Verbesserung NV_i = |v_i| / (sigma0 * sqrt(q_vv,ii))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_v = sigma0 * np.sqrt(np.maximum(np.diag(Qvv) if 'Qvv' in locals() else np.zeros(n_obs), 0.0))
        nv = np.where(sigma_v > 0, np.abs(l) / sigma_v, np.nan)

    rms = float(np.sqrt(np.mean(l * l)))

    return ResectionExtendedResult(
        position=X.copy(),
        std_dev=std_dev,
        covariance=cov_pos,
        sigma0=sigma0,
        rms_residual=rms,
        num_obs=n_obs,
        num_unknowns=n_u,
        dof=dof,
        redundancy=dof / n_obs,
        orientation=(o if has_hz else None),
        scale_sd=(m_sd if estimate_scale_sd else None),
        add_sd=(a_sd if estimate_add_sd else None),
        scale_hd=(m_hd if estimate_scale_hd else None),
        add_hd=(a_hd if estimate_add_hd else None),
        refraction_coefficient=refraction_coefficient,
        earth_radius=earth_radius,
        instrument_height=ih,
        target_heights=th.copy(),
        residuals=l.copy(),
        redundancy_components=r_i,
        normalized_residuals=nv,
        obs_labels=labels,
    )
