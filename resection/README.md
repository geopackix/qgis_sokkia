# Freie Stationierung (Resection) - Modul `haiku`

Ein Python-Modul zur Berechnung der freie Stationierung (3D-Resektion) aus bekannten Anschlusspunkten mit **gemischten Messungstypen** und vollständiger statistischer Auswertung.

## Überblick

Die freie Stationierung ist ein wichtiges Verfahren in der Vermessung, um die unbekannten Koordinaten eines Standpunktes zu bestimmen. Das Modul unterstützt verschiedene Messungstypen:

- **3D-Vektoren**: vollständige Strecke und Richtung
- **Horizontalstrecken**: nur die horizontale Distanz
- **Schrägstrecken**: 3D-Distanzen (z.B. vom Tachymeter gemessen)
- **Richtungen**: Unit-Vektoren ohne Distanzinformation
- **Horizontalwinkel**: Azimutwinkeln (Hz)
- **Vertikalwinkel**: Höhenwinkel/Zenitwinkel (V)

Die Lösung erfolgt mittels **nichtlinearer Ausgleichung** (Gauss-Newton-Verfahren) mit vollständiger Fehlerstatistik.

**Beobachtungsgleichungen:**

| Messungstyp | Gleichung |
|------------|-----------|
| 3D-Vektor | $P_i = X + V_i + e_i$ |
| Horizontalstrecke | $\sqrt{dx_i^2 + dy_i^2} = d_i + e_i$ |
| Schrägstrecke | $\|\|P_i - X\|\| = s_i + e_i$ |
| Richtung | $\frac{P_i - X}{\|\|P_i - X\|\|} = r_i + e_i$ |
| Horizontalwinkel | $\arctan_2(dy_i, dx_i) = hz_i + e_i$ |
| Vertikalwinkel | $\arctan(dz_i / \sqrt{dx_i^2 + dy_i^2}) = v_i + e_i$ |

## Dateien

- **resection.py**: Hauptmodul mit Funktion `resection()` und Datentyp `ResectionResult`
- **test_resection.py**: Umfassendes Testskript mit 10 verschiedenen Tests + Visualisierungen
- **requirements.txt**: Python-Abhängigkeiten

## Installation

```bash
pip install -r requirements.txt
```

Für Visualisierungen ist matplotlib erforderlich:
```bash
pip install matplotlib
```

## Verwendung

### Beispiel 1: Vollständige 3D-Vektoren (klassisch)

```python
import numpy as np
from resection import resection

# Bekannte Anschlusspunkte (N x 3)
observed_points = np.array([
    [1000.0, 2000.0, 100.0],
    [1010.0, 2005.0, 102.0],
    [995.0, 1995.0, 98.0],
    [1005.0, 2010.0, 99.0],
])

# Gemessene Vektoren
measured_vectors = np.array([...])

# Berechnung
result = resection(observed_points, measured_vectors=measured_vectors)
```

### Beispiel 2: Schrägstrecke + Winkel (Tachymeter-Messungen)

```python
# Tachymeter-Messungen
hz_angles = np.array([...])         # Horizontalwinkel
v_angles = np.array([...])          # Vertikalwinkel
slant_distances = np.array([...])   # Schrägstrecken

result = resection(
    observed_points,
    measured_hz_angles=hz_angles,
    measured_v_angles=v_angles,
    measured_slant_distances=slant_distances
)
```

### Beispiel 3: Gemischte Messungen

```python
result = resection(
    observed_points,
    measured_vectors=full_vectors,        # Einige komplette Messungen
    measured_distances=horizontal_dists,  # Horizontalstrecken
    measured_slant_distances=slant_dists, # Schrägstrecken
    measured_hz_angles=hz_angles,         # Horizontalwinkel
    measured_v_angles=v_angles            # Vertikalwinkel
)
```

## ⚠️ Wichtige Hinweise zu Überbestimmtheit

### Messungstypen und ihre Unabhängigkeit

Nicht alle Kombinationen von Messungstypen führen zu einer lösbaren Aufgabe. Das Gleichungssystem muss **überbestimmt** sein (mehr Gleichungen als Unbekannte):

| Messungstyp | Anzahl Gleichungen | Anmerkung |
|------------|-----------------|----------|
| 3D-Vektor | 3 pro Punkt | Vollständige Information |
| Horizontalstrecke | 1 pro Punkt | Nur XY-Distanz |
| Schrägstrecke | 1 pro Punkt | 3D-Distanz (besser: immer lösbar) |
| Richtung | 3 pro Punkt | Nur Richtung, keine Distanz |
| Horizontalwinkel (Hz) | 1 pro Punkt | Nur XY-Richtung! |
| Vertikalwinkel (V) | 1 pro Punkt | Nur Z-Information! |

### ❌ Nicht funktioniert

- **Nur Horizontalwinkel**: Z-Koordinate kann nicht bestimmt werden
- **Nur Vertikalwinkel**: XY-Position kann nicht bestimmt werden  
- **Nur Horizontalstrecken**: Z-Koordinate nicht bestimmbar

### ✓ Funktioniert

- **Hz + V + Schrägstrecke** ← Standardfall (Tachymeter)
- **Hz + V** (2+ Punkte) ← Vollständige Winkelmessung
- **3D-Vektoren** (3+ Punkte) ← Klassischer Fall
- **Schrägstrecken** (3+ Punkte) ← Trilateration
- **Gemischte Messungen** ← Beliebige Kombinationen, solange überbestimmt

### Mindestens verforderliche Messungen

Für eine eindeutige 3D-Position (3 Unbekannte) benötigt man:
- **Mindestens 3 unabhängige Gleichungen**
- Mit Redundanz (Fehlerstatistik) ideal: **4+ Punkte / Messungen**

## Ausgabe und Ergebnisse

Das `ResectionResult`-Objekt liefert:

| Attribut | Beschreibung |
|----------|-------------|
| `position` | Bestimmter Standpunkt (X, Y, Z) |
| `std_dev` | Standardabweichungen der Koordinaten |
| `covariance` | Kovarianzmatrix (3x3) |
| `residuals` | Residuen der Beobachtungen (N, 3) |
| `sigma0` | Varianzfaktor / a-posteriori Standardabweichung |
| `rms_residual` | RMS der Residuen |
| `max_residual_index` | Index des größten Residuums |
| `num_points` | Anzahl der verwendeten Punkte |
| `dof` | Freiheitsgrade |
| `redundancy` | Redundanzgrad |

## Testscript ausführen

```bash
cd haiku
python test_resection.py
```

Das Testscript führt 10 verschiedene Tests aus mit Visualisierungen:

1. **TEST 1**: Basis-Resektion mit 4 Punkten (3D-Vektoren)
2. **TEST 2**: Resektion mit 8 Punkten (bessere Genauigkeit)
3. **TEST 3**: Genauigkeitsverbesserung mit wachsender Punktanzahl
4. **TEST 4**: Resektion mit reinen Horizontalstrecken
5. **TEST 5**: Resektion mit reinen Richtungsmessungen
6. **TEST 6**: Gemischte Messungen (Vektoren + Distanzen + Richtungen)
7. **TEST 7**: Resektion mit Schrägstrecken
8. **TEST 8**: Resektion mit Horizontalwinkeln + eine Schrägstrecke
9. **TEST 9**: Resektion mit Horizontal- und Vertikalwinkeln (als kombinierte Winkelmessung)
10. **TEST 10**: Realistische Tachymeter-Messungen (Hz + V + Schrägstrecke)

Alle Tests produzieren grafische Visualisierungen mit matplotlib, die zeigen:
- Lage der Anschlusspunkte, Standpunkt und Messfehler
- Fehlerellipsoid im 3D-Raum
- Residuenverteilung

## Mathematische Details

### Nichtlineare Ausgleichung (Gauss-Newton)

Für nichtlineare Beobachtungsgleichungen $f(X) \approx l_i + e_i$ wird iterativ gelöst:

$$\Delta X^{(k)} = (A^T W A)^{-1} A^T W l^{(k)}$$

wobei:
- $A$ = Jacobi-Matrix (Designmatrix) 
- $W$ = Gewichtsmatrix
- $l^{(k)}$ = Residuen in Iteration k

### Jacobi-Matrizen (Ableitungen)

Für verschiedene Messungstypen:

| Messungstyp | Jacobi-Zeile |
|------------|---------------|
| Schrägstrecke | $-\frac{P_i - X}{\|\|P_i - X\|\|}$ |
| Horizontalwinkel | $\frac{1}{dx^2 + dy^2} [-dy, dx, 0]$ |
| Vertikalwinkel | $\frac{1}{\rho^2} [-\frac{dz \cdot dx}{dh}, -\frac{dz \cdot dy}{dh}, dh]$ |

wobei $\rho^2 = dh^2 + dz^2$ und $dh = \sqrt{dx^2 + dy^2}$

### Fehlerstatistik

Nach Konvergenz:

$$\sigma_0 = \sqrt{\frac{\sum_i \|l_i\|^2}{dof}}$$

$$\Sigma_X = \sigma_0^2 (A^T W A)^{-1}$$

Die Standardabweichungen der Koordinaten ergibt sich dann aus der Diagonale von $\Sigma_X$.

## Gewichtung von Messungen

Unterschiedliche Messgenauigkeiten können durch Gewichte berücksichtigt werden:

```python
result = resection(
    observed_points,
    measured_vectors=vectors,
    measured_hz_angles=hz_angles,
    weights_vectors=w_vec,           # Gewichte für Vektoren
    weights_hz_angles=w_hz           # Gewichte für Hz-Winkel
)
```

## Features

- ✓ Unterstützung von 6 verschiedenen Messungstypen
- ✓ Gemischte Messungen in einer Ausgleichung
- ✓ Nichtlineare Ausgleichung mit Gauss-Newton
- ✓ Gewichtete Ausgleichung
- ✓ Vollständige Fehlerstatistik (Kovarianzmatrix, Fehlerellipsoid)
- ✓ Residuenanalyse
- ✓ Graphische Visualisierung der Ergebnisse (3D-Plot, Fehlerellipsoid, Residuen)
- ✓ 10 umfangreiche Tests mit verschiedenen Szenarien
- ✓ Optional: Matplotlib-basierte Visualisierungen

## Visualisierungen

Die erweiterten Testskripte erstellen automatisch folgende Visualisierungen:

### 1. 3D-Resektion Plot
Zeigt an:
- **Anschlusspunkte** (blaue Quadrate)
- **Wahrer Standpunkt** (grüner Stern)
- **Berechneter Standpunkt** (roter Kreis)
- **Fehlervektor** (rote gestrichelte Linie) zwischen wahrem und berechnetem Punkt
- **Fehlerellipsoid** (rote transparente Sphäre) um den berechneten Standpunkt

```python
plot_resection_3d(observed_points, true_station, result, 
                  title="3D-Resektion", show_ellipsoid=True)
```

### 2. Residuenanalyse Plot
Zeigt:
- **Residuenbetrag pro Punkt** (Balkendiagramm, rote Hervorhebung des größten Fehlers)
- **Residuenverteilung** (Histogramm mit RMS-Wert)

```python
plot_residuals(result, title="Residuenanalyse")
```

### 3. Fehlerstatistik Plot
Vier Subplots mit:
- **Standardabweichungen** der X, Y, Z Koordinaten
- **Kovarianzmatrix** (Heatmap mit numerischen Werten)
- **Qualitätsmasse** (numerische Zusammenfassung)
- **Korrelationsmatrix** (Heatmap zwischen -1 und +1)

```python
plot_statistics(result, title="Fehlerstatistik")
```

## Testszenarien

Das Testskript enthält 10 verschiedene Szenarien, um unterschiedliche Messungstypen und Kombinationen zu demonstrieren:

| Test | Beschreibung | Messungstyp |
|------|-------------|-----------|
| TEST 1 | Basis-Resektion mit 4 Anschlusspunkten | 3D-Vektoren |
| TEST 2 | Resektion mit 8 Punkten für höhere Genauigkeit | 3D-Vektoren |
| TEST 3 | Genauigkeitsverbesserung durch Redundanz | 3D-Vektoren (12 Punkte) |
| TEST 4 | Resektion mit reinen Streckenmessungen | 2D-Horizontalstrecken |
| TEST 5 | Resektion mit reinen Richtungsmessungen | Richtungsvektoren |
| TEST 6 | Gemischte Messungstypen kombiniert | Vektoren + Strecken |
| TEST 7 | Trilateration mit Schrägstrecken | 3D-Distanzen |
| TEST 8 | Kombination: Richtungen + Schrägstrecken | Richtungen + 3D-Distanzen |
| TEST 9 | Kombination: Horizontal- + Schrägstrecken | 2D + 3D-Distanzen |
| TEST 10 | Realistische Tachymeter-Messungen | Hz + V + Schrägstrecken |

## Tests mit Visualisierungen

Folgende Tests produzieren Visualisierungen:
- **TEST 1**: Basis-Resektion (3D-Plot, Residuen, Statistik)
- **TEST 2**: Resektion mit 8 Punkten (3D-Plot, Statistik)
- **TEST 6**: Gemischte Messungen (3D-Plot, Statistik)
- **TEST 10**: Tachymeter-Messungen (3D-Plot, Residuen, Statistik)

Die anderen Tests zeigen nur numerische Ausgaben.

Die anderen Tests zeigen nur numerische Ausgaben.

## Anforderungen

- Python 3.7+
- NumPy >= 1.19
- Matplotlib >= 3.3 (optional, für Visualisierungen)

## Lizenz

MIT License

## Autor

GitHub Copilot, 19.02.2026
