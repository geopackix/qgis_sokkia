# -*- coding: utf-8 -*-
"""
Dialog für fiktive Test-Messungen zur Entwicklung und zum Testen.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox, QComboBox
)
from qgis.PyQt.QtCore import Qt
import math
import json
import os
import random


class TestMeasurementDialog(QDialog):
    """Dialog zur Erstellung fiktiver Test-Messungen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test-Messung erstellen")
        self.setGeometry(100, 100, 500, 400)
        self.test_data = None
        self._point_types = {}
        self._load_point_types()
        self._build_ui()

    def _load_point_types(self):
        """Lädt die Punkttypen aus pointTypes.json."""
        json_path = os.path.join(os.path.dirname(__file__), 'pointTypes.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            point_prefix_data = data.get('PointPrefixNumbers', {})
            self._point_types = {}
            for label, value in point_prefix_data.items():
                if isinstance(value, dict):
                    self._point_types[label] = value
                else:
                    self._point_types[label] = {"prefix": value}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[TestMeasurementDialog] pointTypes.json nicht geladen: {e}")
            self._point_types = {}

    def _build_ui(self):
        """Baut die Benutzeroberfläche auf."""
        layout = QVBoxLayout(self)

        # Info
        info = QLabel(
            "Erstelle fiktive Messwerte für Entwicklung und Tests.\n"
            "Die Messung wird wie eine normale Messung vom Gerät behandelt."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-style: italic; padding: 4px;")
        layout.addWidget(info)

        # Messwerte
        grp_meas = QGroupBox("Messwerte")
        grid = QGridLayout(grp_meas)

        # Hz (Horizontal angle)
        grid.addWidget(QLabel("Hz (Richtung) [gon]:"), 0, 0)
        self.input_hz = QDoubleSpinBox()
        self.input_hz.setRange(0.0, 400.0)
        self.input_hz.setValue(100.5)
        self.input_hz.setDecimals(4)
        grid.addWidget(self.input_hz, 0, 1)

        # ZA (Zenith angle)
        grid.addWidget(QLabel("ZA (Zenitwinkel) [gon]:"), 1, 0)
        self.input_za = QDoubleSpinBox()
        self.input_za.setRange(70.0, 130.0)
        self.input_za.setValue(100.0)
        self.input_za.setDecimals(4)
        grid.addWidget(self.input_za, 1, 1)

        # SD (Slope distance)
        grid.addWidget(QLabel("SD (Schrägstrecke) [m]:"), 2, 0)
        self.input_sd = QDoubleSpinBox()
        self.input_sd.setRange(0.1, 10000.0)
        self.input_sd.setValue(50.5)
        self.input_sd.setDecimals(4)
        grid.addWidget(self.input_sd, 2, 1)

        layout.addWidget(grp_meas)

        # Punkt-Information
        grp_point = QGroupBox("Punkt-Information")
        grid2 = QGridLayout(grp_point)

        # Punkttyp-Auswahl
        grid2.addWidget(QLabel("Punkt-Typ:"), 0, 0)
        self.combo_point_type = QComboBox()
        self._populate_point_type_combo()
        grid2.addWidget(self.combo_point_type, 0, 1)

        # Punkt-Nummer
        grid2.addWidget(QLabel("Punkt-Nummer:"), 1, 0)
        self.input_point_id = QLineEdit()
        self.input_point_id.setText("TEST.001")
        self.input_point_id.setToolTip("Punkt-Nummer (wird mit ausgewähltem Präfix befüllt)")
        grid2.addWidget(self.input_point_id, 1, 1)

        # Reflektorhöhe th
        grid2.addWidget(QLabel("Reflektorhöhe th [m]:"), 2, 0)
        self.input_th = QDoubleSpinBox()
        self.input_th.setRange(0.0, 5.0)
        self.input_th.setValue(0.0)
        self.input_th.setDecimals(3)
        grid2.addWidget(self.input_th, 2, 1)

        layout.addWidget(grp_point)

        # Vordefinierte Test-Szenarien
        grp_presets = QGroupBox("Vordefinierte Test-Szenarien")
        preset_layout = QVBoxLayout(grp_presets)

        btn_layout = QHBoxLayout()
        
        btn_close = QPushButton("Nahbereich (10m)")
        btn_close.setToolTip("Typische Nahbereichsmessung")
        btn_close.clicked.connect(self._preset_close_range)
        btn_layout.addWidget(btn_close)

        btn_mid = QPushButton("Mittelbereich (50m)")
        btn_mid.setToolTip("Typische Mitteldistanz-Messung")
        btn_mid.clicked.connect(self._preset_mid_range)
        btn_layout.addWidget(btn_mid)

        btn_far = QPushButton("Fernbereich (200m)")
        btn_far.setToolTip("Typische Fernmessung")
        btn_far.clicked.connect(self._preset_far_range)
        btn_layout.addWidget(btn_far)

        preset_layout.addLayout(btn_layout)

        # Random Button
        btn_layout2 = QHBoxLayout()
        btn_random = QPushButton("🎲 Random Messwerte")
        btn_random.setToolTip("Zufällige Messwerte im realistischen Bereich generieren")
        btn_random.setStyleSheet("QPushButton { background-color: #ff9800; color: white; font-weight: bold; padding: 4px; }")
        btn_random.clicked.connect(self._random_measurements)
        btn_layout2.addStretch()
        btn_layout2.addWidget(btn_random)
        btn_layout2.addStretch()
        preset_layout.addLayout(btn_layout2)

        layout.addWidget(grp_presets)

        # Button-Zeile
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_ok = QPushButton("✓  Messung durchführen")
        btn_ok.setStyleSheet("QPushButton { background-color: #2e7d32; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }")
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)

        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

        # Verbinde Signal NACH dem Erstellen aller Widgets
        self.combo_point_type.currentIndexChanged.connect(self._on_point_type_changed)

    def _populate_point_type_combo(self):
        """Befüllt die ComboBox mit Punkttypen aus pointTypes.json."""
        self.combo_point_type.clear()
        for label, data in self._point_types.items():
            prefix = data.get('prefix', '') if isinstance(data, dict) else data
            description = data.get('description', '') if isinstance(data, dict) else ''
            display_text = f"{label} ({prefix})"
            if description:
                display_text += f" - {description}"
            self.combo_point_type.addItem(display_text, prefix)

    def _on_point_type_changed(self, index):
        """Aktualisiert das Prefix in der Punktnummer wenn der Typ wechselt."""
        prefix = self.combo_point_type.currentData()
        if prefix is None:
            return
        current_id = self.input_point_id.text()
        
        # Versuche, einen bekannten Prefix zu erkennen und zu ersetzen
        for label, data in self._point_types.items():
            old_prefix = data.get('prefix') if isinstance(data, dict) else data
            if current_id.startswith(old_prefix):
                # Entferne den alten Prefix und setze den neuen
                suffix = current_id[len(old_prefix):]
                self.input_point_id.setText(prefix + suffix)
                return
        
        # Kein bekannter Prefix gefunden – setze neuen mit "001" als Startnummer
        self.input_point_id.setText(prefix + "001")

    def _preset_close_range(self):
        """Setzt Werte für Nahbereichsmessung."""
        self.input_hz.setValue(95.2500)
        self.input_za.setValue(100.0000)
        self.input_sd.setValue(10.0250)
        self.input_point_id.setText(self.combo_point_type.currentData() + "001")

    def _preset_mid_range(self):
        """Setzt Werte für Mitteldistanz-Messung."""
        self.input_hz.setValue(120.5000)
        self.input_za.setValue(99.8500)
        self.input_sd.setValue(50.3500)
        self.input_point_id.setText(self.combo_point_type.currentData() + "002")

    def _preset_far_range(self):
        """Setzt Werte für Fernmessung."""
        self.input_hz.setValue(45.7500)
        self.input_za.setValue(98.5000)
        self.input_sd.setValue(200.5000)
        self.input_point_id.setText(self.combo_point_type.currentData() + "003")

    def _random_measurements(self):
        """Generiert zufällige realistische Messwerte."""
        # Zufällige Messwerte im realistischen Bereich
        hz = random.uniform(0.0, 400.0)
        za = random.uniform(70.0, 130.0)
        sd = random.uniform(5.0, 500.0)
        
        self.input_hz.setValue(hz)
        self.input_za.setValue(za)
        self.input_sd.setValue(sd)
        
        # Zufällige Punkt-Nummer mit Präfix
        prefix = self.combo_point_type.currentData()
        random_suffix = str(random.randint(1, 999)).zfill(3)
        self.input_point_id.setText(prefix + random_suffix)

    def get_test_measurement(self):
        """Gibt die Test-Messwerte als Dictionary zurück."""
        return {
            'hz_gon': self.input_hz.value(),
            'za_gon': self.input_za.value(),
            'sd_m': self.input_sd.value(),
            'point_id': self.input_point_id.text() or "TEST.001",
            'th': self.input_th.value(),
        }

