"""
Resection Module (Freie Stationierung, 3D) - Advanced Version mit gemischten Messungen

Dieses Modul berechnet die freie Stationierung (3D-Resektion) eines Standpunktes
aus beliebig vielen bekannten Anschlusspunkten mit verschiedenen Messungstypen:
- Vollständige 3D-Vektoren (Strecke + Richtung)
- Reine Streckenmessungen (Distanzen)
- Reine Winkelmessungen (Richtungen/Unit-Vektoren)

Die Berechnung erfolgt mittels nichtlinearer Ausgleichung nach kleinsten Quadraten:
- Gauss-Newton-Verfahren für nichtlineare Beobachtungsgleichungen
- Vollständige statistische Auswertung:
  * Standpunktkoordinaten (X, Y, Z)
  * Standardabweichungen der Koordinaten
  * Fehlermetriken
  * Residuenanalyse
  * Kovarianzmatrix

Modelle:
- Vollständige Vektoren: P_i = X + V_i + e_i
- Strecken: ||P_i - X|| = d_i + e_i
- Richtungen: (P_i - X) / ||P_i - X|| = r_i + e_i

Dabei:
- P_i: bekannte Anschlusspunktkoordinate
- X: zu bestimmender Standpunkt
- V_i: gemessener 3D-Vektor
- d_i: gemessene Distanz
- r_i: gemessene Richtung (Unit-Vektor)
- e_i: Messfehler

Autor: GitHub Copilot
Datum: 19.02.2026
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ResectionResult:
    """Ergebnisobjekt einer Resektion"""
    position: np.ndarray           # (3,) - berechneter Standpunkt
    std_dev: np.ndarray            # (3,) - Standardabweichungen
    covariance: np.ndarray         # (3,3) - Kovarianzmatrix
    residuals: np.ndarray          # (N,3) - Residuen
    sigma0: float                  # Varianzfaktor
    rms_residual: float            # RMS der Residuen
    max_residual_index: int        # Index des größten Residuums
    num_points: int                # Anzahl der Anschlusspunkte
    dof: int                       # Freiheitsgrade
    redundancy: float              # Redundanzgrad

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "RESEKTION-ERGEBNIS (3D)",
            "=" * 60,
            f"Standpunkt (X, Y, Z):        {self.position}",
            f"Standardabweichungen:         {self.std_dev}",
            f"Sigma0 (Varianzfaktor):       {self.sigma0:.6g}",
            f"RMS-Residuum:                 {self.rms_residual:.6g}",
            f"Max. Residuum (Index {self.max_residual_index}): {np.max(np.abs(self.residuals)):.6g}",
            f"Anzahl Punkte:                {self.num_points}",
            f"Freiheitsgrade:               {self.dof}",
            f"Redundanzgrad:                {self.redundancy:.4f}",
            "",
            "Kovarianzmatrix (m²):",
            str(self.covariance),
            "",
            "Erste 5 Residuen (m):",
            str(self.residuals[:5]),
            "=" * 60,
        ]
        return "\n".join(lines)

    def get_ellipsoid_params(self) -> Tuple[float, float, float]:
        """
        Berechnet die Halbachsen des Fehlerellipsoids (3D).
        Rückgabe: Tuple (a, b, c) in aufsteigender Reihenfolge.
        """
        eigenvalues = np.linalg.eigvals(self.covariance)
        eigenvalues = np.sort(np.abs(eigenvalues))
        return tuple(np.sqrt(eigenvalues))


def resection(
    observed_points: np.ndarray,
    measured_vectors: Optional[np.ndarray] = None,
    measured_distances: Optional[np.ndarray] = None,
    measured_slant_distances: Optional[np.ndarray] = None,
    measured_directions: Optional[np.ndarray] = None,
    measured_hz_angles: Optional[np.ndarray] = None,
    measured_v_angles: Optional[np.ndarray] = None,
    max_iterations: int = 20,
    tolerance: float = 1e-8,
    weights_vectors: Optional[np.ndarray] = None,
    weights_distances: Optional[np.ndarray] = None,
    weights_slant_distances: Optional[np.ndarray] = None,
    weights_directions: Optional[np.ndarray] = None,
    weights_hz_angles: Optional[np.ndarray] = None,
    weights_v_angles: Optional[np.ndarray] = None,
) -> ResectionResult:
    """
    Berechnet die freie Stationierung (3D) mit gemischten Messungstypen mittels
    nichtlinearer Ausgleichung (Gauss-Newton-Verfahren).

    Args:
        observed_points: (N,3) Array der bekannten Anschlusspunkte [X, Y, Z]
        measured_vectors: (N_v,3) Array der gemessenen Vektoren (optional)
                          Wenn gegeben, werden Vektor-Beobachtungsgleichungen aufgebaut
        measured_distances: (N_d,) Array der gemessenen Horizontalstrecken (optional)
                           ||P_i - X|| in Metern (Horizontalkomponente)
        measured_slant_distances: (N_s,) Array der gemessenen Schrägstrecken (optional)
                                 Direkte 3D-Distanzen: ||P_i - X||
                                 (z.B. von Tachymeter gemessen)
        measured_directions: (N_r,3) Array der gemessenen Unit-Vektoren (optional)
                            Richtungen sollten normalisiert sein
        measured_hz_angles: (N_hz,) Array der gemessenen Horizontalwinkel (Azimute) in Radiant (optional)
                           Azimut wird von X-Achse gegen Y-Achse gemessen: atan2(dy, dx)
        measured_v_angles: (N_v,) Array der gemessenen Vertikalwinkel (Höhenwinkel) in Radiant (optional)
                          Höhenwinkel: atan(dz / sqrt(dx² + dy²)), wobei 0 = horizontal, π/2 = oben
        max_iterations: Maximale Anzahl Newton-Gauss Iterationen (default: 20)
        tolerance: Konvergenz-Toleranz für Parameteränderungen (default: 1e-8)
        weights_vectors: (N_v,3) oder (N_v,) Gewichte für Vektor-Beobachtungen
        weights_distances: (N_d,) Gewichte für Horizontalstrecken-Beobachtungen
        weights_slant_distances: (N_s,) Gewichte für Schrägstrecken-Beobachtungen
        weights_directions: (N_r,3) oder (N_r,) Gewichte für Richtungs-Beobachtungen
        weights_hz_angles: (N_hz,) Gewichte für Horizontalwinkel-Beobachtungen
        weights_v_angles: (N_v,) Gewichte für Vertikalwinkel-Beobachtungen

    Returns:
        ResectionResult mit Standpunkt, Fehlerstatistik und Residuen

    Raises:
        ValueError: wenn Eingaben ungültig sind oder zu wenige Messungen gegeben
    """
    P = np.asarray(observed_points, dtype=float)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("observed_points muss ein (N,3)-Array sein.")

    n_total = P.shape[0]

    # Sammle Information über Verfügbarkeit von Messungen
    has_vectors = measured_vectors is not None
    has_distances = measured_distances is not None
    has_slant_distances = measured_slant_distances is not None
    has_directions = measured_directions is not None
    has_hz_angles = measured_hz_angles is not None
    has_v_angles = measured_v_angles is not None

    if not (has_vectors or has_distances or has_slant_distances or has_directions or has_hz_angles or has_v_angles):
        raise ValueError("Mindestens eine Messungsart erforderlich (Vektoren, Distanzen, Schrägstrecken, Richtungen, Horizontal- oder Vertikalwinkel).")

    # Validierung und Vorbereitung Vektor-Messungen
    V_vec = None
    n_v = 0
    W_vec = None
    indices_vec = None
    if has_vectors:
        V_vec = np.asarray(measured_vectors, dtype=float)
        if V_vec.ndim != 2 or V_vec.shape[1] != 3:
            raise ValueError("measured_vectors muss ein (N,3)-Array sein.")
        n_v = V_vec.shape[0]
        indices_vec = np.arange(n_v)
        if weights_vectors is not None:
            W_vec = np.asarray(weights_vectors, dtype=float)
            if W_vec.shape[0] != n_v:
                raise ValueError("Länge von weights_vectors muss mit measured_vectors übereinstimmen.")
            if W_vec.ndim == 1:
                W_vec = np.tile(W_vec[:, np.newaxis], (1, 3))
        else:
            W_vec = np.ones((n_v, 3))

    # Validierung und Vorbereitung Distanz-Messungen
    D_meas = None
    n_d = 0
    W_dist = None
    indices_dist = None
    if has_distances:
        D_meas = np.asarray(measured_distances, dtype=float).flatten()
        if D_meas.ndim != 1:
            raise ValueError("measured_distances muss ein 1D-Array sein.")
        n_d = len(D_meas)
        indices_dist = np.arange(n_d)
        if weights_distances is not None:
            W_dist = np.asarray(weights_distances, dtype=float).flatten()
            if len(W_dist) != n_d:
                raise ValueError("Länge von weights_distances muss mit measured_distances übereinstimmen.")
        else:
            W_dist = np.ones(n_d)

    # Validierung und Vorbereitung Schrägstrecken-Messungen
    S_meas = None
    n_s = 0
    W_slant = None
    if has_slant_distances:
        S_meas = np.asarray(measured_slant_distances, dtype=float).flatten()
        if S_meas.ndim != 1:
            raise ValueError("measured_slant_distances muss ein 1D-Array sein.")
        n_s = len(S_meas)
        if weights_slant_distances is not None:
            W_slant = np.asarray(weights_slant_distances, dtype=float).flatten()
            if len(W_slant) != n_s:
                raise ValueError("Länge von weights_slant_distances muss mit measured_slant_distances übereinstimmen.")
        else:
            W_slant = np.ones(n_s)

    # Validierung und Vorbereitung Richtungs-Messungen
    R_meas = None
    n_r = 0
    W_dir = None
    indices_dir = None
    if has_directions:
        R_meas = np.asarray(measured_directions, dtype=float)
        if R_meas.ndim != 2 or R_meas.shape[1] != 3:
            raise ValueError("measured_directions muss ein (N,3)-Array sein.")
        n_r = R_meas.shape[0]
        indices_dir = np.arange(n_r)
        if weights_directions is not None:
            W_dir = np.asarray(weights_directions, dtype=float)
            if W_dir.shape[0] != n_r:
                raise ValueError("Länge von weights_directions muss mit measured_directions übereinstimmen.")
            if W_dir.ndim == 1:
                W_dir = np.tile(W_dir[:, np.newaxis], (1, 3))
        else:
            W_dir = np.ones((n_r, 3))

    # Validierung und Vorbereitung Horizontalwinkel-Messungen
    HZ_meas = None
    n_hz = 0
    W_hz = None
    if has_hz_angles:
        HZ_meas = np.asarray(measured_hz_angles, dtype=float).flatten()
        if HZ_meas.ndim != 1:
            raise ValueError("measured_hz_angles muss ein 1D-Array sein.")
        n_hz = len(HZ_meas)
        if weights_hz_angles is not None:
            W_hz = np.asarray(weights_hz_angles, dtype=float).flatten()
            if len(W_hz) != n_hz:
                raise ValueError("Länge von weights_hz_angles muss mit measured_hz_angles übereinstimmen.")
        else:
            W_hz = np.ones(n_hz)

    # Validierung und Vorbereitung Vertikalwinkel-Messungen
    V_meas = None
    n_v_angles = 0
    W_v = None
    if has_v_angles:
        V_meas = np.asarray(measured_v_angles, dtype=float).flatten()
        if V_meas.ndim != 1:
            raise ValueError("measured_v_angles muss ein 1D-Array sein.")
        n_v_angles = len(V_meas)
        if weights_v_angles is not None:
            W_v = np.asarray(weights_v_angles, dtype=float).flatten()
            if len(W_v) != n_v_angles:
                raise ValueError("Länge von weights_v_angles muss mit measured_v_angles übereinstimmen.")
        else:
            W_v = np.ones(n_v_angles)

    # Gesamtanzahl Messungen
    n_meas = 3 * n_v + n_d + n_s + 3 * n_r + n_hz + n_v_angles

    if n_meas < 3:
        raise ValueError(f"Zu wenige Messungen an Standpunkt({n_meas}), mindestens 3 erforderlich.")

    # Initialisierung: Näherungskoordinaten bestimmen
    if has_vectors:
        # Mit Vektoren kann linear initialisiert werden
        X0 = np.mean(P[:n_v] - V_vec, axis=0)
    elif has_distances:
        # Mit Distanzen: Zentralpunkt der Anschlusspunkte als Näherung
        X0 = np.mean(P[:n_d], axis=0)
    elif has_directions:
        # Mit Richtungen: Zentralpunkt nehmen
        X0 = np.mean(P[:n_r], axis=0)
    elif has_hz_angles or has_v_angles:
        # Mit Winkeln: Zentralpunkt nehmen
        X0 = np.mean(P, axis=0)
    else:
        X0 = np.mean(P, axis=0)

    X = X0.copy()

    # Gauss-Newton-Iteration
    for iteration in range(max_iterations):
        # Aufbau der Beobachtungsgleichungen und Designmatrix
        l = []  # Beobachtungsvektoren (Rückstände)
        A = []  # Designmatrix (Jacobi)
        P_weights = []  # Gewichtsmatrix

        # Vektor-Beobachtungen
        if has_vectors:
            for i in range(n_v):
                # Beobachtungsgleichung: P_i = X + V_i
                # Residuum: l_i = V_i^gemessen - (P_i - X)
                l_i = V_vec[i] - (P[i] - X)
                l.append(l_i)

                # Jacobi: d(P_i - X)/dX = -I_3
                A.append(-np.eye(3))
                P_weights.append(W_vec[i])

        # Distanz-Beobachtungen
        if has_distances:
            for i in range(n_d):
                diff = P[i] - X
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    dist = 1e-10
                
                # Beobachtungsgleichung: ||P_i - X|| = d_i
                # Residuum: l_i = d_i^gemessen - ||P_i - X||
                l_i = np.array([D_meas[i] - dist])
                l.append(l_i)

                # Jacobi: d(||P_i - X||)/dX = -(P_i - X)^T / ||P_i - X||
                # Shape: (1, 3)
                grad_dist = -diff / dist
                A.append(grad_dist.reshape(1, 3))
                P_weights.append(np.array([W_dist[i]]))

        # Schrägstrecken-Beobachtungen
        if has_slant_distances:
            for i in range(n_s):
                diff = P[i] - X
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    dist = 1e-10
                
                # Beobachtungsgleichung: ||P_i - X|| = s_i (3D-Distanz)
                # Residuum: l_i = s_i^gemessen - ||P_i - X||
                l_i = np.array([S_meas[i] - dist])
                l.append(l_i)

                # Jacobi: d(||P_i - X||)/dX = -(P_i - X)^T / ||P_i - X||
                # Shape: (1, 3)
                grad_dist = -diff / dist
                A.append(grad_dist.reshape(1, 3))
                P_weights.append(np.array([W_slant[i]]))

        # Richtungs-Beobachtungen
        if has_directions:
            for i in range(n_r):
                diff = P[i] - X
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    dist = 1e-10
                
                dir_calc = diff / dist
                
                # Beobachtungsgleichung: (P_i - X) / ||P_i - X|| = r_i
                # Residuum: l_i = r_i^gemessen - (P_i - X) / ||P_i - X||
                l_i = R_meas[i] - dir_calc
                l.append(l_i)

                # Jacobi: d((P_i - X)/||P_i - X||)/dX
                # = (I/||d|| - d*d^T/||d||^3) * (-I)
                dist_sq = dist * dist
                A_i = (np.eye(3) - np.outer(dir_calc, dir_calc)) / dist
                A.append(-A_i)
                P_weights.append(W_dir[i])

        # Horizontalwinkel-Beobachtungen (Azimut)
        if has_hz_angles:
            for i in range(n_hz):
                diff = P[i] - X
                dx = diff[0]
                dy = diff[1]
                
                # Beobachtungsgleichung: hz = atan2(dy, dx)
                hz_calc = np.arctan2(dy, dx)
                # Residuum: Winkeldifferenz (mit Periodizität beachten)
                hz_res = HZ_meas[i] - hz_calc
                # Normalize to [-π, π]
                hz_res = np.arctan2(np.sin(hz_res), np.cos(hz_res))
                l_i = np.array([hz_res])
                l.append(l_i)
                
                # Jacobi: d(atan2(dy, dx))/dX = [-dy/(dx² + dy²), dx/(dx² + dy²), 0]
                denom = dx*dx + dy*dy
                if denom < 1e-10:
                    denom = 1e-10
                A_i = np.array([[-dy/denom, dx/denom, 0.0]])
                A.append(A_i)
                P_weights.append(np.array([W_hz[i]]))

        # Vertikalwinkel-Beobachtungen (Höhenwinkel)
        if has_v_angles:
            for i in range(n_v_angles):
                diff = P[i] - X
                dx = diff[0]
                dy = diff[1]
                dz = diff[2]
                
                # Horizontale Distanz
                dh = np.sqrt(dx*dx + dy*dy)
                if dh < 1e-10:
                    dh = 1e-10
                
                # Beobachtungsgleichung: v = atan(dz / dh)
                v_calc = np.arctan2(dz, dh)
                # Residuum
                v_res = V_meas[i] - v_calc
                l_i = np.array([v_res])
                l.append(l_i)
                
                # Jacobi: d(atan2(dz, dh))/dX
                # dh/dX = [dx/dh, dy/dh, 0]
                # v = atan(dz/dh) => dv/dX = [-dz*dh/(dh² + dz²) * d(dh)/dX + 1/(dh² + dz²) * [0, 0, dh]]
                dist_sq = dh*dh + dz*dz
                if dist_sq < 1e-10:
                    dist_sq = 1e-10
                
                # dv/d(dh) = -dz / dist_sq
                # dv/d(dz) = dh / dist_sq
                A_i = np.array([
                    [-dz*dx / (dh * dist_sq), -dz*dy / (dh * dist_sq), dh / dist_sq]
                ])
                A.append(A_i)
                P_weights.append(np.array([W_v[i]]))

        # Zusammenfassen
        l_vec = np.concatenate(l)
        A_mat = np.vstack(A)
        P_weights_vec = np.concatenate(P_weights)

        # Gewichtete LS-Lösung: (A^T W A)^{-1} A^T W l
        W_diag = np.diag(P_weights_vec)
        AtWA = A_mat.T @ W_diag @ A_mat
        AtWl = A_mat.T @ W_diag @ l_vec

        try:
            dX = np.linalg.solve(AtWA, AtWl)
        except np.linalg.LinAlgError:
            raise ValueError("Singulare Designmatrix - Messungen nicht ausreichend unabhängig.")

        # Konvergenz prüfen
        if np.max(np.abs(dX)) < tolerance:
            break

        X = X + dX

    # Abschließende Residuen berechnen
    residuals_list = []

    if has_vectors:
        for i in range(n_v):
            res_i = V_vec[i] - (P[i] - X)
            residuals_list.append(res_i)

    if has_distances:
        for i in range(n_d):
            diff = P[i] - X
            dist = np.linalg.norm(diff)
            res_i = D_meas[i] - dist
            residuals_list.append(np.array([res_i, 0, 0]))  # Für Kompatibilität als 3D-Vektor

    if has_slant_distances:
        for i in range(n_s):
            diff = P[i] - X
            dist = np.linalg.norm(diff)
            res_i = S_meas[i] - dist
            residuals_list.append(np.array([res_i, 0, 0]))  # Für Kompatibilität als 3D-Vektor

    if has_directions:
        for i in range(n_r):
            diff = P[i] - X
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                dist = 1e-10
            dir_calc = diff / dist
            res_i = R_meas[i] - dir_calc
            residuals_list.append(res_i)

    if has_hz_angles:
        for i in range(n_hz):
            diff = P[i] - X
            dx = diff[0]
            dy = diff[1]
            hz_calc = np.arctan2(dy, dx)
            hz_res = HZ_meas[i] - hz_calc
            hz_res = np.arctan2(np.sin(hz_res), np.cos(hz_res))
            residuals_list.append(np.array([hz_res, 0, 0]))

    if has_v_angles:
        for i in range(n_v_angles):
            diff = P[i] - X
            dz = diff[2]
            dh = np.sqrt(diff[0]**2 + diff[1]**2)
            if dh < 1e-10:
                dh = 1e-10
            v_calc = np.arctan2(dz, dh)
            v_res = V_meas[i] - v_calc
            residuals_list.append(np.array([v_res, 0, 0]))

    residuals = np.array(residuals_list)

    # Fehlerstatistik
    # Für nichtlineare Probleme: Verwendung aller Komponenten
    dof = n_meas - 3

    if dof <= 0:
        raise ValueError("Zu wenige Freiheitsgrade für Fehlerschätzung.")

    # Finale Residuen mit Gewichten
    l_final = []
    if has_vectors:
        for i in range(n_v):
            l_final.append(V_vec[i] - (P[i] - X))
    if has_distances:
        for i in range(n_d):
            diff = P[i] - X
            dist = np.linalg.norm(diff)
            l_final.append([D_meas[i] - dist])
    if has_slant_distances:
        for i in range(n_s):
            diff = P[i] - X
            dist = np.linalg.norm(diff)
            l_final.append([S_meas[i] - dist])
    if has_directions:
        for i in range(n_r):
            diff = P[i] - X
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                dist = 1e-10
            dir_calc = diff / dist
            l_final.append(R_meas[i] - dir_calc)

    if has_hz_angles:
        for i in range(n_hz):
            diff = P[i] - X
            dx = diff[0]
            dy = diff[1]
            hz_calc = np.arctan2(dy, dx)
            hz_res = HZ_meas[i] - hz_calc
            hz_res = np.arctan2(np.sin(hz_res), np.cos(hz_res))
            l_final.append([hz_res])

    if has_v_angles:
        for i in range(n_v_angles):
            diff = P[i] - X
            dz = diff[2]
            dh = np.sqrt(diff[0]**2 + diff[1]**2)
            if dh < 1e-10:
                dh = 1e-10
            v_calc = np.arctan2(dz, dh)
            v_res = V_meas[i] - v_calc
            l_final.append([v_res])

    l_final_vec = np.concatenate(l_final)
    ssq = np.sum(l_final_vec**2)
    sigma0_sq = ssq / dof
    sigma0 = np.sqrt(sigma0_sq)

    # Kovarianzmatrix der Parameter: Qxx = sigma0^2 * (A^T W A)^{-1}
    try:
        Qxx = sigma0_sq * np.linalg.inv(AtWA)
    except np.linalg.LinAlgError:
        Qxx = np.eye(3) * sigma0_sq  # Fallback

    std_dev = np.sqrt(np.abs(np.diag(Qxx)))

    # Weitere Metriken
    rms_residual = np.sqrt(ssq / len(l_final_vec))
    max_res_idx = np.argmax(np.linalg.norm(l_final_vec.reshape(-1, 1), axis=1))
    redundancy = dof / n_meas

    # Gesamtanzahl der Anschlusspunkte, die verwendet wurden
    num_points_used = (n_v if has_vectors else 0) + (n_d if has_distances else 0) + (n_s if has_slant_distances else 0) + (n_r if has_directions else 0) + (n_hz if has_hz_angles else 0) + (n_v_angles if has_v_angles else 0)

    return ResectionResult(
        position=X,
        std_dev=std_dev,
        covariance=Qxx,
        residuals=residuals,
        sigma0=sigma0,
        rms_residual=rms_residual,
        max_residual_index=max_res_idx,
        num_points=num_points_used,
        dof=dof,
        redundancy=redundancy,
    )


