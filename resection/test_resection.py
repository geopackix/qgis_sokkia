"""
Testscript für das Resection-Modul.

Führt verschiedene Tests durch:
1. Basis-Test mit 4 Punkten
2. Erweiterter Test mit 8 Punkten
3. Genauigkeitsvergleich
... und weitere 8 Tests mit verschiedenen Messungstypen

Visualisierungen werden mit matplotlib erstellt.
"""

import numpy as np
from resection import resection

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warnung: matplotlib nicht installiert. Visualisierungen werden übersprungen.")


def plot_resection_3d(
    observed_points: np.ndarray,
    true_station: np.ndarray,
    result,
    title: str = "3D-Resektion",
    show_ellipsoid: bool = True
):
    """Visualisiert die Resektion-Ergebnisse im 3D-Raum"""
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Anschlusspunkte
    ax.scatter(
        observed_points[:, 0],
        observed_points[:, 1],
        observed_points[:, 2],
        c='blue',
        s=100,
        marker='s',
        label='Anschlusspunkte',
        edgecolors='darkblue',
        linewidths=2
    )
    
    # Wahrer Standpunkt
    ax.scatter(
        true_station[0],
        true_station[1],
        true_station[2],
        c='green',
        s=200,
        marker='*',
        label='Wahrer Standpunkt',
        edgecolors='darkgreen',
        linewidths=2
    )
    
    # Berechneter Standpunkt
    ax.scatter(
        result.position[0],
        result.position[1],
        result.position[2],
        c='red',
        s=150,
        marker='o',
        label='Berechneter Standpunkt',
        edgecolors='darkred',
        linewidths=2
    )
    
    # Verbindungslinie (Fehler)
    ax.plot(
        [true_station[0], result.position[0]],
        [true_station[1], result.position[1]],
        [true_station[2], result.position[2]],
        'r--',
        linewidth=2,
        label=f'Fehler: {np.linalg.norm(result.position - true_station):.4f} m'
    )
    
    # Fehlerellipsoid (vereinfacht als Sphäre mit Radius = mittlere Standardabweichung)
    if show_ellipsoid:
        ellipsoid_params = result.get_ellipsoid_params()
        mean_std = np.mean(ellipsoid_params)
        
        # Kugel um berechneten Punkt
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = mean_std * np.outer(np.cos(u), np.sin(v)) + result.position[0]
        y_sphere = mean_std * np.outer(np.sin(u), np.sin(v)) + result.position[1]
        z_sphere = mean_std * np.outer(np.ones(np.size(u)), np.cos(v)) + result.position[2]
        
        ax.plot_surface(
            x_sphere, y_sphere, z_sphere,
            alpha=0.2,
            color='red',
            label=f'Fehlerellipsoid (σ={mean_std:.4f}m)'
        )
    
    # Beschriftungen für Punkte
    for i, point in enumerate(observed_points):
        ax.text(point[0], point[1], point[2], f'  P{i+1}', fontsize=9)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(result, title: str = "Residuenanalyse"):
    """Visualisiert die Residuen"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuenbetrag pro Punkt
    residual_magnitudes = np.linalg.norm(result.residuals, axis=1)
    ax = axes[0]
    bars = ax.bar(range(len(residual_magnitudes)), residual_magnitudes, color='steelblue', edgecolor='navy', linewidth=1.5)
    ax.set_xlabel('Punkt-Index')
    ax.set_ylabel('Residuenbetrag (m)')
    ax.set_title('Residuenbetrag pro Punkt')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Markiere größtes Residuum
    max_idx = result.max_residual_index
    if max_idx < len(bars):
        bars[max_idx].set_color('red')
    
    # Residuenverteilung
    ax = axes[1]
    ax.hist(residual_magnitudes, bins=max(5, len(residual_magnitudes)//2), 
            color='steelblue', edgecolor='navy', alpha=0.7)
    ax.axvline(result.rms_residual, color='red', linestyle='--', linewidth=2, label=f'RMS: {result.rms_residual:.6f} m')
    ax.set_xlabel('Residuenbetrag (m)')
    ax.set_ylabel('Häufigkeit')
    ax.set_title('Residuenverteilung')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_statistics(result, title: str = "Fehlerstatistik"):
    """Visualisiert die Fehlerstatistik"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Standardabweichungen der Koordinaten
    ax = axes[0, 0]
    coords = ['X', 'Y', 'Z']
    bars = ax.bar(coords, result.std_dev, color=['red', 'green', 'blue'], edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Standardabweichung (m)')
    ax.set_title('Standardabweichungen der Koordinaten')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, result.std_dev):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    # Kovarianzmatrix Heatmap
    ax = axes[0, 1]
    im = ax.imshow(result.covariance, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(coords)
    ax.set_yticklabels(coords)
    ax.set_title('Kovarianzmatrix (m²)')
    plt.colorbar(im, ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{result.covariance[i, j]:.2e}', ha='center', va='center', fontsize=8, color='black')
    
    # Qualitätsmaße
    ax = axes[1, 0]
    ax.axis('off')
    stats_text = f"""
QUALITÄTSMASSE:
━━━━━━━━━━━━━━━━━━━━━━━━━
σ₀ (Varianzfaktor):        {result.sigma0:.6g}
RMS Residuum:              {result.rms_residual:.6g} m
Max. Residuum:             {np.max(np.abs(result.residuals)):.6g} m
Freiheitsgrade:            {result.dof}
Redundanzgrad:             {result.redundancy:.4f}
Anzahl Punkte:             {result.num_points}

FEHLERELLIPSOID:
━━━━━━━━━━━━━━━━━━━━━━━━━
Halbachse a:               {result.get_ellipsoid_params()[2]:.6g} m
Halbachse b:               {result.get_ellipsoid_params()[1]:.6g} m
Halbachse c:               {result.get_ellipsoid_params()[0]:.6g} m
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
    
    # Korrelationsmatrix
    ax = axes[1, 1]
    corr = result.covariance / np.outer(result.std_dev, result.std_dev)
    im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(coords)
    ax.set_yticklabels(coords)
    ax.set_title('Korrelationsmatrix')
    plt.colorbar(im, ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=9, color='black')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_basic():
    """Basis-Test mit 4 Anschlusspunkten"""
    print("\n" + "=" * 60)
    print("TEST 1: Basis-Resektion mit 4 Punkten")
    print("=" * 60)

    # Bekannte Punkte
    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
    ])

    # Wahrer Standpunkt
    true_station = np.array([1002.0, 2003.0, 100.0])

    # Ideale Vektoren mit Rauschen
    np.random.seed(123)
    measured_vectors = observed_points - true_station
    measured_vectors += np.random.normal(scale=0.015, size=measured_vectors.shape)

    # Berechnung
    result = resection(observed_points, measured_vectors=measured_vectors)
    print(result)

    # Vergleich
    error = result.position - true_station
    print(f"\nWahrer Standpunkt: {true_station}")
    print(f"Fehler (m):        {error}")
    print(f"Abstand (m):       {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:   {result.get_ellipsoid_params()}")
    
    # Visualisierungen
    plot_resection_3d(observed_points, true_station, result, "TEST 1: Basis-Resektion mit 4 Punkten")
    plot_residuals(result, "TEST 1: Residuenanalyse")
    plot_statistics(result, "TEST 1: Fehlerstatistik")


def test_extended():
    """Erweiterter Test mit 8 Punkten"""
    print("\n" + "=" * 60)
    print("TEST 2: Resektion mit 8 Anschlusspunkten")
    print("=" * 60)

    # Mehr Punkte => bessere Genauigkeit
    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
        [998.0, 2002.0, 101.5],
        [1015.0, 1990.0, 97.5],
        [985.0, 2010.0, 103.0],
        [1008.0, 1998.0, 99.5],
    ])

    true_station = np.array([1001.5, 2002.3, 100.2])
    np.random.seed(456)
    measured_vectors = observed_points - true_station
    measured_vectors += np.random.normal(scale=0.012, size=measured_vectors.shape)

    result = resection(observed_points, measured_vectors=measured_vectors)
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt: {true_station}")
    print(f"Fehler (m):        {error}")
    print(f"Abstand (m):       {np.linalg.norm(error):.6g}")
    
    # Visualisierungen
    plot_resection_3d(observed_points, true_station, result, "TEST 2: Resektion mit 8 Punkten")
    plot_statistics(result, "TEST 2: Fehlerstatistik")


def test_accuracy_improvement():
    """Zeigt, wie sich die Genauigkeit mit mehr Punkten verbessert"""
    print("\n" + "=" * 60)
    print("TEST 3: Genauigkeitsverbesserung mit mehr Punkten")
    print("=" * 60)

    # Generiere Punkte auf einer Sphäre
    n_tests = [3, 5, 8, 12]
    true_station = np.array([500.0, 500.0, 100.0])

    for n in n_tests:
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        phi = np.linspace(0.3, 0.7*np.pi, n)
        radius = 1000.0

        observed_points = np.array([
            true_station + radius * np.array([
                np.sin(phi[i]) * np.cos(theta[i]),
                np.sin(phi[i]) * np.sin(theta[i]),
                np.cos(phi[i])
            ])
            for i in range(n)
        ])

        np.random.seed(789)
        measured_vectors = observed_points - true_station
        measured_vectors += np.random.normal(scale=0.02, size=measured_vectors.shape)

        result = resection(observed_points, measured_vectors=measured_vectors)
        error = np.linalg.norm(result.position - true_station)
        print(f"n={n:2d}: Abstand={error:.6g} m, sigma0={result.sigma0:.6g}, std_dev={result.std_dev}")


def test_distances_only():
    """Test mit nur Streckenmessungen (Distanzen)"""
    print("\n" + "=" * 60)
    print("TEST 4: Resektion mit reinen Streckenmessungen")
    print("=" * 60)

    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
    ])

    true_station = np.array([1002.0, 2003.0, 100.0])
    np.random.seed(42)
    distances = np.linalg.norm(observed_points - true_station, axis=1)
    distances += np.random.normal(scale=0.01, size=len(distances))

    result = resection(observed_points, measured_distances=distances)
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")


def test_directions_only():
    """Test mit nur Richtungsmessungen (Winkeln/Azimutwinkeln)"""
    print("\n" + "=" * 60)
    print("TEST 5: Resektion mit reinen Richtungsmessungen")
    print("=" * 60)

    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
        [1000.0, 2000.0, 105.0],
    ])

    true_station = np.array([1002.0, 2003.0, 100.0])
    np.random.seed(42)
    
    dirs = (observed_points - true_station) / np.linalg.norm(observed_points - true_station, axis=1, keepdims=True)
    perturbation = np.random.normal(scale=0.01, size=dirs.shape)
    dirs_measured = (dirs + perturbation) / np.linalg.norm(dirs + perturbation, axis=1, keepdims=True)

    result = resection(observed_points, measured_directions=dirs_measured)
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")


def test_mixed_measurements():
    """Test mit gemischten Messungen (Vektoren + Distanzen + Richtungen)"""
    print("\n" + "=" * 60)
    print("TEST 6: Resektion mit gemischten Messungstypen")
    print("=" * 60)

    P_all = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
        [1000.0, 2000.0, 105.0],
    ])
    true_station = np.array([1002.0, 2003.0, 100.0])

    np.random.seed(42)
    
    # Punkte 0-1: Vollständige Vektoren
    P_vec = P_all[:2]
    V_vec = P_vec - true_station + np.random.normal(scale=0.01, size=(2, 3))

    # Punkte 2-3: Nur Distanzen
    P_dist = P_all[2:4]
    distances = np.linalg.norm(P_dist - true_station, axis=1) + np.random.normal(scale=0.01, size=2)

    # Punkt 4: Nur Richtung
    P_dir = P_all[4:5]
    dirs = (P_dir - true_station) / np.linalg.norm(P_dir - true_station, axis=1, keepdims=True)
    perturbation = np.random.normal(scale=0.01, size=dirs.shape)
    dirs_measured = (dirs + perturbation) / np.linalg.norm(dirs + perturbation, axis=1, keepdims=True)

    # Alle Punkte zusammenfassen
    P_combined = np.vstack([P_vec, P_dist, P_dir])

    result = resection(
        P_combined,
        measured_vectors=V_vec,
        measured_distances=distances,
        measured_directions=dirs_measured
    )
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")
    
    # Visualisierungen
    plot_resection_3d(P_combined, true_station, result, "TEST 6: Gemischte Messungen (Vektoren + Distanzen + Richtungen)")
    plot_statistics(result, "TEST 6: Fehlerstatistik")


def test_slant_distances():
    """Test mit Schrägstrecken-Messungen (z.B. von Tachymeter)"""
    print("\n" + "=" * 60)
    print("TEST 7: Resektion mit Schrägstrecken-Messungen")
    print("=" * 60)

    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
    ])

    true_station = np.array([1002.0, 2003.0, 100.0])
    np.random.seed(42)
    
    # Die Schrägstrecken sind die direkten 3D-Distanzen
    slant_distances = np.linalg.norm(observed_points - true_station, axis=1)
    # Addiere Messfehler (z.B. 0.5cm Messunsicherheit)
    slant_distances += np.random.normal(scale=0.005, size=len(slant_distances))

    result = resection(observed_points, measured_slant_distances=slant_distances)
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")


def test_horizontal_angles():
    """Test mit Richtungsvektoren und Schrägstrecken (kombiniert)"""
    print("\n" + "=" * 60)
    print("TEST 8: Resektion mit Richtungsvektoren + Schrägstrecken")
    print("=" * 60)

    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
    ])

    true_station = np.array([1002.0, 2003.0, 100.0])
    np.random.seed(42)
    
    # Berechne Richtungsvektoren und Schrägstrecken
    diff = observed_points - true_station
    
    # Richtungsvektoren (normalisiert)
    directions = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    directions += np.random.normal(scale=0.01, size=directions.shape)
    # Re-normalisiere nach Rauschen
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Schrägstrecken (3D-Distanzen)
    slant_distances = np.linalg.norm(diff, axis=1)
    slant_distances += np.random.normal(scale=0.005, size=len(slant_distances))

    result = resection(
        observed_points,
        measured_directions=directions,
        measured_slant_distances=slant_distances
    )
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")


def test_vertical_angles():
    """Test mit Horizontal- und Schrägstrecken (Mixed distance measurements)"""
    print("\n" + "=" * 60)
    print("TEST 9: Resektion mit Horizontal- + Schrägstrecken")
    print("=" * 60)

    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
        [1000.0, 2000.0, 105.0],
    ])

    true_station = np.array([1002.0, 2003.0, 100.0])
    np.random.seed(42)
    
    diff = observed_points - true_station
    
    # Horizontalstrecken (2D distances)
    horizontal_distances = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    horizontal_distances += np.random.normal(scale=0.005, size=len(horizontal_distances))
    
    # Schrägstrecken (3D distances)
    slant_distances = np.linalg.norm(diff, axis=1)
    slant_distances += np.random.normal(scale=0.005, size=len(slant_distances))

    result = resection(
        observed_points,
        measured_distances=horizontal_distances,
        measured_slant_distances=slant_distances
    )
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")


def test_tachymeter_data():
    """Test mit realistischen Tachymeter-Daten (Hz, V, Schrägstrecke)"""
    print("\n" + "=" * 60)
    print("TEST 10: Resektion mit realistischen Tachymeter-Messungen")
    print("         (Horizontal- & Vertikalwinkel + Schrägstrecken)")
    print("=" * 60)

    observed_points = np.array([
        [1000.0, 2000.0, 100.0],
        [1010.0, 2005.0, 102.0],
        [995.0, 1995.0, 98.0],
        [1005.0, 2010.0, 99.0],
        [1000.0, 2000.0, 105.0],
    ])

    true_station = np.array([1002.0, 2003.0, 100.0])
    np.random.seed(42)
    
    # Berechne Horizontalwinkel (Azimutwinkeln)
    hz_angles = np.arctan2(observed_points[:, 1] - true_station[1], 
                           observed_points[:, 0] - true_station[0])
    hz_angles += np.random.normal(scale=0.002, size=len(hz_angles))  # ±0.002 rad error
    
    # Berechne Vertikalwinkel (Höhenwinkel)
    diff = observed_points - true_station
    dh = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    v_angles = np.arctan2(diff[:, 2], dh)
    v_angles += np.random.normal(scale=0.002, size=len(v_angles))  # ±0.002 rad error
    
    # Berechne Schrägstrecken (3D-Distanzen vom Tachymeter)
    slant_distances = np.linalg.norm(diff, axis=1)
    slant_distances += np.random.normal(scale=0.01, size=len(slant_distances))  # ±1cm error

    result = resection(
        observed_points,
        measured_hz_angles=hz_angles,
        measured_v_angles=v_angles,
        measured_slant_distances=slant_distances
    )
    print(result)

    error = result.position - true_station
    print(f"\nWahrer Standpunkt:   {true_station}")
    print(f"Berechneter Punkt:   {result.position}")
    print(f"Fehler (m):          {error}")
    print(f"Abstand (m):         {np.linalg.norm(error):.6g}")
    print(f"Fehlerellipsoid:     {result.get_ellipsoid_params()}")
    
    # Visualisierungen
    plot_resection_3d(observed_points, true_station, result, "TEST 11: Tachymeter-Messungen (Hz + V + Schrägstrecke)")
    plot_residuals(result, "TEST 11: Residuenanalyse")
    plot_statistics(result, "TEST 11: Fehlerstatistik")


if __name__ == "__main__":
    if HAS_MATPLOTLIB:
        print("\n" + "=" * 70)
        print("RESECTION-TESTSUITES MIT VISUALISIERUNGEN")
        print("=" * 70)
        print("\nMatplotlib ist verfügbar. Grafische Ausgaben werden angezeigt.")
        print("Schließen Sie die Grafikfenster, um zum nächsten Test zu gehen.\n")
    else:
        print("\nWarnung: Matplotlib nicht installiert!")
        print("Bitte ausführen: pip install matplotlib")
        print("Nur numerische Ausgaben werden angezeigt.\n")
    
    test_basic()
    test_extended()
    test_accuracy_improvement()
    test_distances_only()
    test_directions_only()
    test_mixed_measurements()
    test_slant_distances()
    test_horizontal_angles()
    test_vertical_angles()
    test_tachymeter_data()
    print("\n" + "=" * 60)
    print("ALLE TESTS ABGESCHLOSSEN")
    print("=" * 60)
