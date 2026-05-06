# -*- coding: utf-8 -*-
"""
JAG3D Export fuer Resektionsergebnisse.

Exportiert Messungen und Punkte im offiziellen JAG3D ASCII-Format gemäß:
https://software.applied-geodesy.org/wiki/user-interface/import

Format-Zusammenfassung:
  - Kommentare beginnen mit '#'
  - Trennzeichen: Whitespace (Leerzeichen/Tabulator)
  - Winkel: Gon (400 gon = Vollkreis)
  - Strecken: Meter
  - Letzte Spalte (sigma): wenn < 1 --> Standardunsicherheit [m oder gon],
                            wenn >= 1 --> Entfernung fuer stoch. Modell

Terrestrische Beobachtungen:
  <Standpunkt>  <Zielpunkt>  [<ih>  <th>]  <Wert>  [<sigma>]

Punkte und Koordinaten (3D):
  <Punktname>  <East(Rechtswert)>  <North(Hochwert)>  <Hoehe>  [<sigma>]
"""

import os
from datetime import datetime
from typing import List, Dict, Optional


def _normalize_gon(gon: float) -> float:
    """Normiert einen Winkel auf [0, 400) gon. 400 gon = Vollkreis."""
    gon = gon % 400.0
    if gon < 0:
        gon += 400.0
    return gon


class JAG3DExporter:
    """Exportiert Resektionsdaten im offiziellen JAG3D ASCII-Format."""

    DEFAULT_SIGMA_SD_M   = 0.005
    DEFAULT_SIGMA_HZ_GON = 0.0003
    DEFAULT_SIGMA_ZA_GON = 0.0003

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_complete(
        self,
        station_id: str,
        station_coord: tuple,
        fixed_points: List[Dict],
        observations: List[Dict],
        instrument_height: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        files = {}
        files['Festpunkte']      = self._export_festpunkte(fixed_points)
        files['Neupunkte']       = self._export_neupunkte(station_id, station_coord)
        files['Schraegstrecken'] = self._export_schraegstrecken(station_id, observations, instrument_height)
        files['Richtungen']      = self._export_richtungen(station_id, observations)
        files['Zenitwinkel']     = self._export_zenitwinkel(station_id, observations, instrument_height)
        files['README']          = self._export_readme(station_id, station_coord, fixed_points, observations, instrument_height, metadata)
        return files

    def _export_festpunkte(self, fixed_points: List[Dict]) -> str:
        filepath = os.path.join(self.output_dir, 'Festpunkte.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# JAG3D - Anschlusspunkte (3D)\n")
            f.write("# Format: Punktname  East(Rechtswert)  North(Hochwert)  Hoehe  (Einheit: m)\n")
            f.write("# Import in JAG3D: Import -> Punkte und Koordinaten -> Raumpunkte -> Anschlusspunkte\n")
            f.write("#\n")
            for pt in fixed_points:
                name = str(pt.get('name', pt.get('id', '?')))
                x = float(pt.get('X', 0.0))
                y = float(pt.get('Y', 0.0))
                z = float(pt.get('Z', 0.0))
                f.write(f"{name}\t{x:.4f}\t{y:.4f}\t{z:.4f}\n")
        return filepath

    def _export_neupunkte(self, station_id: str, station_coord: tuple) -> str:
        filepath = os.path.join(self.output_dir, 'Neupunkte.txt')
        x, y, z = station_coord
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# JAG3D - Neupunkte (3D) - Naherungskoordinaten aus Freier Stationierung\n")
            f.write("# Format: Punktname  East(Rechtswert)  North(Hochwert)  Hoehe  (Einheit: m)\n")
            f.write("# Import in JAG3D: Import -> Punkte und Koordinaten -> Raumpunkte -> Neupunkte\n")
            f.write("#\n")
            f.write(f"{station_id}\t{float(x):.4f}\t{float(y):.4f}\t{float(z):.4f}\n")
        return filepath

    def _export_schraegstrecken(self, station_id: str, observations: List[Dict], ih: float) -> str:
        filepath = os.path.join(self.output_dir, 'Schraegstrecken.txt')
        obs_with_sd = [o for o in observations if o.get('sd_m') is not None]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# JAG3D - Schraegstrecken\n")
            f.write("# Format: Standpunkt  Zielpunkt  ih[m]  th[m]  Strecke[m]  sigma[m]\n")
            f.write("# Import in JAG3D: Import -> Terrestrische Beobachtungen -> Schraegstrecken\n")
            f.write("# Hinweis: sigma < 1 wird als Standardunsicherheit [m] interpretiert\n")
            f.write("#\n")
            if not obs_with_sd:
                f.write("# Keine Schraegstreckenmessungen vorhanden\n")
            for obs in obs_with_sd:
                target = str(obs.get('name', obs.get('id', '?')))
                sd  = float(obs['sd_m'])
                th  = float(obs.get('th_m', 0.0))
                f.write(f"{station_id}\t{target}\t{ih:.4f}\t{th:.4f}\t{sd:.4f}\t{self.DEFAULT_SIGMA_SD_M:.4f}\n")
        return filepath

    def _export_richtungen(self, station_id: str, observations: List[Dict]) -> str:
        filepath = os.path.join(self.output_dir, 'Richtungen.txt')
        obs_with_hz = [o for o in observations if o.get('hz_gon') is not None]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# JAG3D - Richtungen (Horizontalwinkel)\n")
            f.write("# Format: Standpunkt  Zielpunkt  Richtung[gon]  sigma[gon]\n")
            f.write("# Import in JAG3D: Import -> Terrestrische Beobachtungen -> Richtungen\n")
            f.write("# Hinweis: Winkel in Gon (400 gon = Vollkreis)\n")
            f.write("#\n")
            if not obs_with_hz:
                f.write("# Keine Richtungsmessungen vorhanden\n")
            for obs in obs_with_hz:
                target = str(obs.get('name', obs.get('id', '?')))
                hz = _normalize_gon(float(obs['hz_gon']))
                f.write(f"{station_id}\t{target}\t{hz:.5f}\t{self.DEFAULT_SIGMA_HZ_GON:.6f}\n")
        return filepath

    def _export_zenitwinkel(self, station_id: str, observations: List[Dict], ih: float) -> str:
        filepath = os.path.join(self.output_dir, 'Zenitwinkel.txt')
        obs_with_za = [o for o in observations if o.get('za_gon') is not None]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# JAG3D - Zenitwinkel\n")
            f.write("# Format: Standpunkt  Zielpunkt  ih[m]  th[m]  ZA[gon]  sigma[gon]\n")
            f.write("# Import in JAG3D: Import -> Terrestrische Beobachtungen -> Zenitwinkel\n")
            f.write("# Hinweis: Zenitwinkel in Gon; 100 gon = Horizontale\n")
            f.write("#\n")
            if not obs_with_za:
                f.write("# Keine Zenitwinkel-Messungen vorhanden\n")
            for obs in obs_with_za:
                target = str(obs.get('name', obs.get('id', '?')))
                za = _normalize_gon(float(obs['za_gon']))
                th = float(obs.get('th_m', 0.0))
                f.write(f"{station_id}\t{target}\t{ih:.4f}\t{th:.4f}\t{za:.5f}\t{self.DEFAULT_SIGMA_ZA_GON:.6f}\n")
        return filepath

    def _export_readme(self, station_id, station_coord, fixed_points, observations, ih, metadata):
        filepath = os.path.join(self.output_dir, 'README.txt')
        n_sd = sum(1 for o in observations if o.get('sd_m') is not None)
        n_hz = sum(1 for o in observations if o.get('hz_gon') is not None)
        n_za = sum(1 for o in observations if o.get('za_gon') is not None)
        x, y, z = station_coord
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("JAG3D EXPORT - FREIE STATIONIERUNG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Exportiert am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            if metadata:
                f.write("METADATEN\n")
                f.write("-" * 60 + "\n")
                for k, v in metadata.items():
                    if v is not None:
                        f.write(f"  {k}: {v}\n")
                f.write("\n")
            f.write("ZUSAMMENFASSUNG\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Standpunkt-ID:       {station_id}\n")
            f.write(f"  Koordinaten (X/Y/Z): {float(x):.4f} / {float(y):.4f} / {float(z):.4f} m\n")
            f.write(f"  Instrumentenhoehe:   {ih:.4f} m\n")
            f.write(f"  Festpunkte:          {len(fixed_points)}\n")
            f.write(f"  Schraegstrecken:     {n_sd}\n")
            f.write(f"  Richtungen:          {n_hz}\n")
            f.write(f"  Zenitwinkel:         {n_za}\n\n")
            f.write("DATEIEN UND IMPORT IN JAG3D\n")
            f.write("-" * 60 + "\n")
            f.write("  Festpunkte.txt\n    Import: Import -> Punkte und Koordinaten -> Raumpunkte -> Anschlusspunkte\n\n")
            f.write("  Neupunkte.txt\n    Import: Import -> Punkte und Koordinaten -> Raumpunkte -> Neupunkte\n\n")
            f.write("  Schraegstrecken.txt\n    Import: Import -> Terrestrische Beobachtungen -> Schraegstrecken\n\n")
            f.write("  Richtungen.txt\n    Import: Import -> Terrestrische Beobachtungen -> Richtungen\n\n")
            f.write("  Zenitwinkel.txt\n    Import: Import -> Terrestrische Beobachtungen -> Zenitwinkel\n\n")
            f.write("EINHEITEN\n")
            f.write("-" * 60 + "\n")
            f.write("  Koordinaten/Strecken: Meter [m]\n")
            f.write("  Winkel:               Gon [gon] (400 gon = Vollkreis)\n")
            f.write("  --> Im JAG3D-Projekt muss die Einheit auf GON eingestellt sein!\n\n")
            f.write("FORMAT-REFERENZ\n")
            f.write("-" * 60 + "\n")
            f.write("  https://software.applied-geodesy.org/wiki/user-interface/import\n")
        return filepath
