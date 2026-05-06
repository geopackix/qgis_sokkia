# -*- coding: utf-8 -*-
"""
Dialog zur Anzeige der Resektionsergebnisse mit Qualitätseinordnung.

Zeigt alle relevanten Metriken und ordnet die Stationierung
als "Ausgezeichnet", "Gut", "Akzeptabel" oder "Schlecht" ein.
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QFrame, QScrollArea,
    QFileDialog, QMessageBox
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont, QBrush

from .jag3d_exporter import JAG3DExporter


def _normalize_gon(gon: float) -> float:
    """Normiert einen Winkel auf [0, 400) gon. 400 gon = Vollkreis."""
    gon = gon % 400.0
    if gon < 0:
        gon += 400.0
    return gon


@dataclass
class QualityThresholds:
    """Qualitätsgrenzwerte"""
    name: str
    color: str
    std_dev_xy_max_m: float
    std_dev_z_max_m: float
    sigma0_min: float
    sigma0_max: float
    rms_residual_max_m: float
    max_residual_max_m: float
    redundancy_min: float


class ResectionResultDialog(QDialog):
    """Dialog zur Anzeige von Resektionsergebnissen mit Qualitätseinordnung"""

    result_accepted = pyqtSignal(float, float, float, float, dict)

    def __init__(self, parent, resection_result, measured_obs, fixed_points=None, station_id="SP", instrument_height=0.0, quality_config_path=None):
        """
        Args:
            parent: Parent widget
            resection_result: ResectionResult object mit allen Metriken
            measured_obs: Liste der gemessenen Beobachtungen
            fixed_points: Liste der Festpunkte (optional, für JAG3D-Export)
            station_id: ID des Standpunktes (optional, für JAG3D-Export)
            quality_config_path: Pfad zur JSON-Konfigurationsdatei
        """
        super().__init__(parent)
        self.resection_result = resection_result
        self.measured_obs = measured_obs
        self.fixed_points = fixed_points or []
        self.station_id = station_id
        self.instrument_height = instrument_height

        # Konfiguration laden
        if quality_config_path is None:
            quality_config_path = os.path.join(
                os.path.dirname(__file__), 'resection_quality.json'
            )
        self.quality_thresholds = self._load_quality_config(quality_config_path)
        self.quality_class = self._determine_quality()

        self.setWindowTitle("Resektions-Ergebnis")
        self.setGeometry(100, 100, 900, 700)
        self.init_ui()

    def _load_quality_config(self, config_path: str) -> dict:
        """Lädt die Qualitätsgrenzwerte aus JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('quality_thresholds', {})
        except FileNotFoundError:
            print(f"⚠ Qualitätskonfiguration nicht gefunden: {config_path}")
            return {}

    @staticmethod
    def _rms_and_max(result) -> tuple:
        """
        Berechnet RMS und Max-Residuum in Metern, unabhängig vom Ergebnis-Typ.

        ResectionResult:         residuals ist (N,3) – 3D-Residuen in Metern pro Punkt.
        ResectionExtendedResult: residuals ist 1D – Beobachtungsresiduen in gemischten
                                  Einheiten (m + rad). Nur SD-Residuen werden für den
                                  physikalisch interpretierbaren Vergleich genutzt;
                                  als Fallback dient std_dev.
        """
        res_raw = np.atleast_1d(result.residuals)

        if res_raw.ndim == 2:
            # Standard-Methode: (N, 3) – 3D-Residuen in Metern
            norms = np.linalg.norm(res_raw, axis=1)  # Betrag je Punkt
            rms = float(np.sqrt(np.mean(norms**2)))
            max_res = float(np.max(norms))
        else:
            # Erweiterte Methode: 1D-Vektor mit gemischten Einheiten.
            # Nutze nur Werte im plausiblen Meter-Bereich (|v| < 1 m).
            meter_vals = res_raw[np.abs(res_raw) < 1.0]
            if meter_vals.size > 0:
                rms = float(np.sqrt(np.mean(meter_vals**2)))
                max_res = float(np.max(np.abs(meter_vals)))
            else:
                # Fallback: RMS der Lagestandabweichungen
                rms = float(np.sqrt(
                    result.std_dev[0]**2 + result.std_dev[1]**2
                ))
                max_res = float(np.max(np.abs(result.std_dev)))
        return rms, max_res

    def _determine_quality(self) -> str:
        """Bestimmt die Qualitätsklasse basierend auf Grenzwerten"""
        if not self.quality_thresholds:
            return 'unknown'

        rms_residual, max_residual = self._rms_and_max(self.resection_result)

        # Berechne max Std Dev
        std_dev_xy = np.sqrt(
            self.resection_result.std_dev[0]**2 + 
            self.resection_result.std_dev[1]**2
        )
        std_dev_z = self.resection_result.std_dev[2]

        # Überprüfe gegen Grenzwerte (von oben nach unten)
        for quality_key in ['excellent', 'good', 'acceptable', 'poor']:
            if quality_key not in self.quality_thresholds:
                continue

            thresh = self.quality_thresholds[quality_key]

            # Alle Kriterien müssen erfüllt sein (sigma0 bewusst ausgelassen –
            # nicht einheitlich vergleichbar bei gemischten Beobachtungstypen)
            checks = [
                std_dev_xy <= thresh['std_dev_xy_max_m'],
                std_dev_z <= thresh['std_dev_z_max_m'],
                rms_residual <= thresh['rms_residual_max_m'],
                max_residual <= thresh['max_residual_max_m'],
                self.resection_result.redundancy >= thresh['redundancy_min']
            ]

            if all(checks):
                return quality_key

        return 'poor'

    def init_ui(self):
        """Erstellt die Benutzeroberfläche"""
        layout = QVBoxLayout()

        # Qualitäts-Box oben
        quality_box = self._create_quality_box()
        layout.addWidget(quality_box)

        # Ergebnisse (scrollbar für lange Listen)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = self._create_results_widget()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("💾 JAG3D Export")
        btn_export.setToolTip("Exportiere Messungen und Punkte für JAG3D Netzausgleichung")
        btn_export.clicked.connect(self.export_jag3d)
        btn_accept = QPushButton("✓ Übernehmen")
        btn_accept.clicked.connect(self.accept_result)
        btn_cancel = QPushButton("✗ Ablehnen")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_accept)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _create_quality_box(self) -> QGroupBox:
        """Erstellt die Qualitäts-Einordnungs-Box"""
        quality_class = self.quality_class
        thresholds = self.quality_thresholds.get(
            quality_class, self.quality_thresholds.get('poor', {})
        )

        name = thresholds.get('name', 'Unbekannt')
        color_name = thresholds.get('color', 'gray')
        color_map = {
            'green': QColor(76, 175, 80),
            'lightgreen': QColor(156, 204, 101),
            'yellow': QColor(255, 193, 7),
            'red': QColor(244, 67, 54),
        }
        color = color_map.get(color_name, QColor(128, 128, 128))

        config_path = os.path.join(
            os.path.dirname(__file__), 'resection_quality.json'
        )
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                description = config.get('description', {}).get(quality_class, '')
        except:
            description = ''

        box = QGroupBox("Qualitätseinordnung")
        layout = QVBoxLayout()

        # Titel mit Farbe
        title_label = QLabel(f"Status: {name}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)

        # Farbbalken
        color_bar = QFrame()
        color_bar.setStyleSheet(f"background-color: {color.name()};")
        color_bar.setFixedHeight(30)

        # Beschreibung
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)

        layout.addWidget(color_bar)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        box.setLayout(layout)

        return box

    @staticmethod
    def _colored_label(text: str, good: bool) -> QLabel:
        """Erstellt ein QLabel mit grüner oder roter Vordergrundfarbe."""
        lbl = QLabel(text)
        color = "#2e7d32" if good else "#c62828"   # dunkelgrün / dunkelrot
        lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        return lbl

    def _create_results_widget(self):
        """Erstellt das Widget mit allen Ergebnissen"""
        widget = QVBoxLayout()

        # Grenzwerte für Einzelwert-Einfärbung (Schwelle: "gut")
        thresh_gd = self.quality_thresholds.get('good', {})

        def std_color(val, key):
            return val <= thresh_gd.get(key, 999.0)

        # Position
        pos_group = QGroupBox("Standpunkt (X, Y, Z)")
        pos_layout = QVBoxLayout()
        x, y, z = self.resection_result.position
        sd = self.resection_result.std_dev
        pos_layout.addWidget(self._colored_label(
            f"X (Rechts):  {x:.4f} m  ± {sd[0]:.6f} m", std_color(sd[0], 'std_dev_xy_max_m')))
        pos_layout.addWidget(self._colored_label(
            f"Y (Hoch):    {y:.4f} m  ± {sd[1]:.6f} m", std_color(sd[1], 'std_dev_xy_max_m')))
        pos_layout.addWidget(self._colored_label(
            f"Z (Höhe):    {z:.4f} m  ± {sd[2]:.6f} m", std_color(sd[2], 'std_dev_z_max_m')))
        if self.resection_result.orientation is not None:
            z0_gon = _normalize_gon(self._rad_to_gon(self.resection_result.orientation))
            pos_layout.addWidget(QLabel(f"z₀ (Orientierung): {z0_gon:.4f} gon"))
        pos_group.setLayout(pos_layout)
        widget.addWidget(pos_group)

        # Genauigkeitsmetriken
        accuracy_group = QGroupBox("Genauigkeitsmetriken")
        accuracy_layout = QVBoxLayout()

        rms_residual, max_residual = self._rms_and_max(self.resection_result)
        num_points = getattr(self.resection_result, 'num_points',
                     getattr(self.resection_result, 'num_obs', len(self.measured_obs)))
        redundancy = self.resection_result.redundancy
        dof = self.resection_result.dof

        # Sigma0 – informativ, keine Einfärbung (einheitenabhängig)
        accuracy_layout.addWidget(QLabel(
            f"Sigma₀ (Varianzfaktor):     {self.resection_result.sigma0:.6g}"))

        # RMS-Residuum
        accuracy_layout.addWidget(self._colored_label(
            f"RMS-Residuum:                {rms_residual*1000:.2f} mm",
            rms_residual <= thresh_gd.get('rms_residual_max_m', 999.0)))

        # Max Residuum
        accuracy_layout.addWidget(self._colored_label(
            f"Max. Residuum:               {max_residual*1000:.2f} mm",
            max_residual <= thresh_gd.get('max_residual_max_m', 999.0)))

        # Beobachtungen (neutral)
        accuracy_layout.addWidget(QLabel(
            f"Anzahl Beobachtungen:        {num_points}"))

        # Freiheitsgrade (gut wenn > 0)
        accuracy_layout.addWidget(self._colored_label(
            f"Freiheitsgrade:              {dof}", dof > 0))

        # Redundanzgrad
        accuracy_layout.addWidget(self._colored_label(
            f"Redundanzgrad:               {redundancy:.4f}",
            redundancy >= thresh_gd.get('redundancy_min', 0.0)))

        accuracy_group.setLayout(accuracy_layout)
        widget.addWidget(accuracy_group)

        # Residuen-Tabelle
        residuals_group = QGroupBox("Residuen nach Anschlusspunkt")
        residuals_layout = QVBoxLayout()

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Punkt-ID", "ΔX (m)", "ΔY (m)", "ΔZ (m)", "3D (m)"])
        table.setMaximumHeight(250)

        residuals_m = np.atleast_1d(self.resection_result.residuals)
        max_res_good = thresh_gd.get('max_residual_max_m', 999.0)

        for i, obs in enumerate(self.measured_obs):
            # Stelle sicher, dass wir das richtige Residuum-Format haben
            if residuals_m.ndim == 1:
                # Wenn 1D und nicht genug Elemente, skip
                if len(residuals_m) < 3 or i > 0:
                    break
                res = residuals_m[:3]  # Erste 3 Komponenten
            else:
                # Wenn 2D
                if i >= len(residuals_m):
                    break
                res = residuals_m[i]

            # Berechne 3D-Norm
            if len(res) >= 3:
                res_3d = np.linalg.norm(res[:3])
            else:
                res_3d = np.abs(res[0]) if len(res) > 0 else 0.0

            table.insertRow(i)
            table.setItem(i, 0, QTableWidgetItem(str(obs.get('name', f'P{i+1}'))))
            table.setItem(i, 1, QTableWidgetItem(f"{res[0]:.6f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{res[1]:.6f}" if len(res) > 1 else "n/a"))
            table.setItem(i, 3, QTableWidgetItem(f"{res[2]:.6f}" if len(res) > 2 else "n/a"))
            table.setItem(i, 4, QTableWidgetItem(f"{res_3d:.6f}"))

            # Zeile einfärben je nach 3D-Residuum
            row_color = QColor("#c8e6c9") if res_3d <= max_res_good else QColor("#ffcdd2")
            for col in range(5):
                item = table.item(i, col)
                if item:
                    item.setBackground(QBrush(row_color))

        table.resizeColumnsToContents()
        residuals_layout.addWidget(table)
        residuals_group.setLayout(residuals_layout)
        widget.addWidget(residuals_group)

        container = QFrame()
        container.setLayout(widget)
        return container

    def accept_result(self):
        """Akzeptiert das Ergebnis und emittiert Signal"""
        x, y, z = self.resection_result.position
        z0_rad = self.resection_result.orientation or 0.0

        rms_res, _ = self._rms_and_max(self.resection_result)
        num_points = getattr(self.resection_result, 'num_points',
                     getattr(self.resection_result, 'num_obs', len(self.measured_obs)))

        details = {
            'quality_class': self.quality_class,
            'sigma0': self.resection_result.sigma0,
            'rms_residual': rms_res,
            'redundancy': self.resection_result.redundancy,
            'num_observations': num_points,
        }

        self.result_accepted.emit(x, y, z, z0_rad, details)
        self.accept()

    def export_jag3d(self):
        """Exportiert die Resektionsergebnisse für JAG3D"""
        if not self.fixed_points:
            QMessageBox.warning(
                self,
                "Export nicht möglich",
                "Keine Festpunkte verfügbar. Export kann nicht durchgeführt werden."
            )
            return

        # Wähle Verzeichnis
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Wähle Verzeichnis für JAG3D-Export",
            os.path.expanduser("~")
        )

        if not output_dir:
            return

        try:
            # Erstelle Exporter
            exporter = JAG3DExporter(output_dir)

            # Bereite Metadaten vor
            metadata = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quality_class': self.quality_class,
                'sigma0': self.resection_result.sigma0,
                'redundancy': self.resection_result.redundancy,
                'comment': f'Freie Stationierung - Qualität: {self.quality_thresholds.get(self.quality_class, {}).get("name", "Unbekannt")}'
            }

            # Exportiere
            files = exporter.export_complete(
                station_id=self.station_id,
                station_coord=tuple(self.resection_result.position),
                fixed_points=self.fixed_points,
                observations=self.measured_obs,
                instrument_height=self.instrument_height,
                metadata=metadata
            )

            # Zeige Erfolg
            file_list = "\n".join([f"  • {os.path.basename(f)}" for f in files.values()])
            QMessageBox.information(
                self,
                "JAG3D-Export erfolgreich",
                f"Dateien exportiert nach:\n\n{output_dir}\n\n"
                f"Erstellte Dateien:\n{file_list}\n\n"
                f"Diese können jetzt in JAG3D importiert werden."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export-Fehler",
                f"Fehler beim Exportieren:\n\n{str(e)}"
            )

    @staticmethod
    def _rad_to_gon(rad: float) -> float:
        """Konvertiert Radiant zu Gon"""
        return rad * 200.0 / math.pi
