# -*- coding: utf-8 -*-
"""
Dialog für die Freie Stationierung (Rückwärtsschnitt).

Ablauf:
  1. Benutzer wählt einen QGIS-Layer mit bekannten Anschlusspunkten.
  2. Benutzer ordnet jeder Messung (aus dem aktiven Messlayer) einen
     Anschlusspunkt zu und füllt so die Zuordnungstabelle.
  3. Das resection-Modul berechnet den Standpunkt (X, Y, Z).
  4. Die Orientierung z0 wird aus den Hz-Messungen abgeleitet.
  5. Ergebnisse werden angezeigt; auf Wunsch werden Standpunkt und
     Orientierung in das Plugin übernommen.
"""

import math
import os
import sys
import json

import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QComboBox, QHeaderView, QMessageBox,
    QSizePolicy, QFrame, QCheckBox, QLineEdit,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont

from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox
from qgis.core import (
    QgsMapLayerProxyModel, QgsProject, QgsWkbTypes,
)

# Sicherstellen, dass das resection-Modul importierbar ist
sys.path.insert(0, os.path.dirname(__file__))
from resection import resection  # noqa: E402
from resection_extended import resection_extended  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Winkel-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _gon_to_rad(gon: float) -> float:
    return gon * math.pi / 200.0


def _rad_to_gon(rad: float) -> float:
    return rad * 200.0 / math.pi


def _parse_float(value) -> float:
    """Konvertiert einen String/Wert zu Float. Ersetzt Kommas durch Punkte (Lokalisierung)."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None or value == '':
        return 0.0
    # Kommas durch Punkte ersetzen (deutsche Dezimal-Trennzeichen)
    s = str(value).strip().replace(',', '.')
    return float(s)


def _normalize_gon(gon: float) -> float:
    """Normiert einen Winkel auf [0, 400) gon."""
    gon = gon % 400.0
    if gon < 0:
        gon += 400.0
    return gon


def _mean_angle_gon(angles_gon: list) -> float:
    """Mittelt eine Liste von Winkeln in gon korrekt (Umlauf beachten)."""
    sins = [math.sin(_gon_to_rad(a)) for a in angles_gon]
    coss = [math.cos(_gon_to_rad(a)) for a in angles_gon]
    mean_sin = sum(sins) / len(sins)
    mean_cos = sum(coss) / len(coss)
    return _normalize_gon(_rad_to_gon(math.atan2(mean_sin, mean_cos)))


def _load_apriori_sigmas():
    """Lädt die A-priori Standardabweichungen aus resectionConfig.json."""
    plugin_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(plugin_dir, 'resectionConfig.json')
    
    defaults = {
        'sigma_sd_m': 0.005,
        'sigma_hz_mgon': 1.0,
        'sigma_za_mgon': 1.0,
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        apriori = data.get('APrioriStandardDeviations', {})
        if apriori:
            return {
                'sigma_sd_m': apriori.get('sigma_sd_m', defaults['sigma_sd_m']),
                'sigma_hz_mgon': apriori.get('sigma_hz_mgon', defaults['sigma_hz_mgon']),
                'sigma_za_mgon': apriori.get('sigma_za_mgon', defaults['sigma_za_mgon']),
            }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ResectionDialog] resectionConfig.json nicht geladen: {e}")
    
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Hauptdialog
# ─────────────────────────────────────────────────────────────────────────────

class ResectionDialog(QDialog):
    """
    Dialog zur Bestimmung des Standpunkts aus Messungen (Freie Stationierung).

    Signals:
        result_accepted(x, y, z, z0_rad):  Emittiert, wenn der Benutzer das
            Ergebnis übernimmt ('Standpunkt & Orientierung übernehmen').
    """

    result_accepted = pyqtSignal(float, float, float, float, object)  # x, y, z, z0_rad, details_dict

    def __init__(self, iface, mlayer, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.mlayer = mlayer
        self._default_mlayer = mlayer
        self._result = None          # ResectionResult-Objekt nach Berechnung
        self._result_z0_rad = None   # berechnete Orientierung in Radiant
        self._observations = []      # aktuelle Beobachtungsliste
        self._standort_dlg = None    # Referenz zum standort_dialog, um ihn zu schließen

        self.setWindowTitle("Freie Stationierung – Rückwärtsschnitt")
        self.setMinimumWidth(920)
        self.setMinimumHeight(660)
        self._measurements = []
        self._load_measurements()
        self._build_ui()

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(8)
        # ── 0. Messlayer-Auswahl ───────────────────────────────────────────────────────────────────
        grp_mlayer = QGroupBox("Messungen (Layer)")
        grp_mlayer.setToolTip("Layer, aus dem die Messungen (Hz, ZA, SD) gelesen werden")
        ml = QHBoxLayout(grp_mlayer)
        ml.addWidget(QLabel("Messlayer:"))
        self._measure_layer_combo = QgsMapLayerComboBox()
        self._measure_layer_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        if self._default_mlayer is not None:
            self._measure_layer_combo.blockSignals(True)
            self._measure_layer_combo.setLayer(self._default_mlayer)
            self._measure_layer_combo.blockSignals(False)
        self._measure_layer_combo.layerChanged.connect(self._on_measure_layer_changed)
        ml.addWidget(self._measure_layer_combo, 2)
        btn_reload_m = QPushButton("↺ Neu laden")
        btn_reload_m.setToolTip("Messungen aus dem gewählten Layer neu einlesen")
        btn_reload_m.clicked.connect(self._reload_measurements)
        ml.addWidget(btn_reload_m)
        main.addWidget(grp_mlayer)
        # ── 1. Layer-Auswahl für bekannte Punkte ──────────────────────────────
        grp_layer = QGroupBox("Anschlusspunkte – bekannte Punkte (Layer)")
        ll = QHBoxLayout(grp_layer)

        ll.addWidget(QLabel("Layer:"))
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        ll.addWidget(self.layer_combo, 2)

        ll.addWidget(QLabel("ID-Feld:"))
        self.field_id = QgsFieldComboBox()
        self.field_id.setAllowEmptyFieldName(True)
        ll.addWidget(self.field_id, 1)

        ll.addWidget(QLabel("X / Rechts:"))
        self.field_x = QgsFieldComboBox()
        self.field_x.setAllowEmptyFieldName(True)
        ll.addWidget(self.field_x, 1)

        self.chk_geom_x = QCheckBox("$x")
        self.chk_geom_x.setToolTip("X-Koordinate aus der Geometrie des Features verwenden")
        self.chk_geom_x.toggled.connect(lambda on: self.field_x.setDisabled(on))
        self.chk_geom_x.toggled.connect(self._refresh_coord_display)
        ll.addWidget(self.chk_geom_x)
        ll.addWidget(QLabel("Y / Hoch:"))
        self.field_y = QgsFieldComboBox()
        self.field_y.setAllowEmptyFieldName(True)
        ll.addWidget(self.field_y, 1)

        self.chk_geom_y = QCheckBox("$y")
        self.chk_geom_y.setToolTip("Y-Koordinate aus der Geometrie des Features verwenden")
        self.chk_geom_y.toggled.connect(lambda on: self.field_y.setDisabled(on))
        self.chk_geom_y.toggled.connect(self._refresh_coord_display)
        ll.addWidget(self.chk_geom_y)

        ll.addWidget(QLabel("Z / Höhe:"))
        self.field_z = QgsFieldComboBox()
        self.field_z.setAllowEmptyFieldName(True)
        ll.addWidget(self.field_z, 1)

        self.layer_combo.layerChanged.connect(self._on_layer_changed)

        main.addWidget(grp_layer)

        # ── 2. Zuordnungstabelle ──────────────────────────────────────────────
        grp_assign = QGroupBox("Zuordnung: Messung  →  Anschlusspunkt")
        al = QVBoxLayout(grp_assign)

        n_meas = len(self._measurements)
        self.lbl_measure_info = QLabel(
            f"Wählen Sie für jede Messung den zugehörigen bekannten Anschlusspunkt. "
            f"Mindestens 2 Zuordnungen mit SD und ZA sind erforderlich (≥ 3 empfohlen). "
            f"— {n_meas} Messung(en) verfügbar."
        )
        self.lbl_measure_info.setWordWrap(True)
        self.lbl_measure_info.setStyleSheet("color: #555; padding: 2px 0 4px 0;")
        al.addWidget(self.lbl_measure_info)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Zeile hinzufügen")
        btn_add.setToolTip("Neue Zuordnungszeile hinzufügen")
        btn_remove = QPushButton("– Zeile entfernen")
        btn_remove.setToolTip("Markierte Zeile entfernen")
        btn_refresh = QPushButton("↺ Messungen neu laden")
        btn_refresh.setToolTip("Messungen aus dem Messlayer neu einlesen und Auswahllisten aktualisieren")
        btn_add.clicked.connect(self._add_row)
        btn_remove.clicked.connect(self._remove_row)
        btn_refresh.clicked.connect(self._reload_measurements)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(btn_refresh)
        al.addLayout(btn_row)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Messung (Pkt.Nr.)",
            "Hz [gon]",
            "ZA [gon]",
            "SD [m]",
            "Anschlusspunkt",
            "Bekannte Koordinaten",
            "th [m]",
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.Stretch)
        for col in (1, 2, 3, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumHeight(150)
        # Spalte 'th [m]' nur im Modus 'Erweitert' sichtbar
        self.table.setColumnHidden(6, True)
        al.addWidget(self.table)

        main.addWidget(grp_assign)

        # ── 2b. Standpunktnummer & Instrumentenhöhe ───────────────────────────
        ih_row = QHBoxLayout()
        ih_row.addStretch()
        ih_row.addWidget(QLabel("Standpunktnummer (SP-ID):"))
        self.input_sp_id = QLineEdit()
        self.input_sp_id.setText("SP")
        self.input_sp_id.setMaximumWidth(80)
        self.input_sp_id.setToolTip("Bezeichnung des Standpunkts")
        ih_row.addWidget(self.input_sp_id)
        ih_row.addSpacing(20)
        ih_row.addWidget(QLabel("Instrumentenhöhe (ih):"))
        self.input_ih = QLineEdit()
        self.input_ih.setText("0.0")
        self.input_ih.setMaximumWidth(100)
        self.input_ih.setToolTip("Instrumentenhöhe über Standpunkt in Metern")
        ih_row.addWidget(self.input_ih)
        ih_row.addWidget(QLabel("m"))
        ih_row.addStretch()
        main.addLayout(ih_row)

        # ── 2c. Berechnungsmodus (Standard / Erweitert) ───────────────────────
        mode_row = QHBoxLayout()
        mode_row.addStretch()
        mode_row.addWidget(QLabel("Berechnungsmodus:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Standard (klassisch)", "standard")
        self.mode_combo.addItem("Erweitert (konform)", "extended")
        self.mode_combo.setToolTip(
            "Standard: bisheriger Algorithmus ohne ih/th, ohne Refraktion.\n"
            "Erweitert: konformes   Modell mit Instrumenten-/Reflektor-"
            "höhe, Refraktion + Erdkrümmung, optional Maßstab/Add für Strecken."
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        main.addLayout(mode_row)

        # ── 2d. Erweiterte Parameter (nur im Modus 'Erweitert' sichtbar) ──────
        self.grp_extended = QGroupBox("Erweiterte Parameter (Stufen 1–3)")
        ex_layout = QVBoxLayout(self.grp_extended)

        # Refraktion / Erdradius
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Refraktionskoeff. k:"))
        self.input_k = QLineEdit("0.13")
        self.input_k.setMaximumWidth(70)
        ref_row.addWidget(self.input_k)
        self.chk_earth_curv = QCheckBox("Erdkrümmung berücksichtigen")
        self.chk_earth_curv.setChecked(True)
        ref_row.addWidget(self.chk_earth_curv)
        ref_row.addSpacing(20)
        ref_row.addWidget(QLabel("Erdradius R [m]:"))
        self.input_R = QLineEdit("6378137")
        self.input_R.setMaximumWidth(110)
        ref_row.addWidget(self.input_R)
        ref_row.addStretch()
        ex_layout.addLayout(ref_row)

        # A-priori-Sigmen
        sig_row = QHBoxLayout()
        # Lade A-priori-Werte aus Konfiguration
        apriori_config = _load_apriori_sigmas()
        
        sig_row.addWidget(QLabel("σ SD [m]:"))
        self.input_sigma_sd = QLineEdit(str(apriori_config['sigma_sd_m']))
        self.input_sigma_sd.setMaximumWidth(70)
        sig_row.addWidget(self.input_sigma_sd)
        sig_row.addSpacing(10)
        sig_row.addWidget(QLabel("σ Hz [mgon]:"))
        self.input_sigma_hz = QLineEdit(str(apriori_config['sigma_hz_mgon']))
        self.input_sigma_hz.setMaximumWidth(70)
        sig_row.addWidget(self.input_sigma_hz)
        sig_row.addSpacing(10)
        sig_row.addWidget(QLabel("σ ZA [mgon]:"))
        self.input_sigma_za = QLineEdit(str(apriori_config['sigma_za_mgon']))
        self.input_sigma_za.setMaximumWidth(70)
        sig_row.addWidget(self.input_sigma_za)
        sig_row.addStretch()
        ex_layout.addLayout(sig_row)

        # Maßstab / Additionskonstante schätzen
        sa_row = QHBoxLayout()
        self.chk_estimate_scale = QCheckBox("Maßstab (SD) mitschätzen")
        self.chk_estimate_scale.setToolTip(
            "Schätzt einen gemeinsamen Maßstabsfaktor für alle "
            "Schrägstrecken als zusätzliche Unbekannte.")
        self.chk_estimate_add = QCheckBox("Additionskonstante (SD) mitschätzen")
        self.chk_estimate_add.setToolTip(
            "Schätzt eine Additionskonstante für alle Schrägstrecken als "
            "zusätzliche Unbekannte.")
        sa_row.addWidget(self.chk_estimate_scale)
        sa_row.addWidget(self.chk_estimate_add)
        sa_row.addStretch()
        ex_layout.addLayout(sa_row)

        info = QLabel(
            "Hinweis: Reflektorhöhen th werden je Zeile in der Spalte "
            "'th [m]' der Zuordnungstabelle eingetragen."
        )
        info.setStyleSheet("color: #555; font-style: italic;")
        info.setWordWrap(True)
        ex_layout.addWidget(info)

        self.grp_extended.setVisible(False)
        main.addWidget(self.grp_extended)

        # ── 3. Berechnen-Button ───────────────────────────────────────────────
        self.btn_calc = QPushButton("  Rückwärtsschnitt berechnen  ")
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        self.btn_calc.setFont(f)
        self.btn_calc.setMinimumHeight(38)
        self.btn_calc.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; padding: 6px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.btn_calc.clicked.connect(self._calculate)
        main.addWidget(self.btn_calc)

        # ── 4. Ergebnisbereich ────────────────────────────────────────────────
        grp_result = QGroupBox("Ergebnis")
        rl = QVBoxLayout(grp_result)

        # Koordinatenzeile
        coord_row = QHBoxLayout()
        self.lbl_x = QLabel("X (Rechts): —")
        self.lbl_y = QLabel("Y (Hoch):   —")
        self.lbl_z = QLabel("Z (Höhe):   —")
        self.lbl_z0 = QLabel("z₀: —")
        for lbl in (self.lbl_x, self.lbl_y, self.lbl_z, self.lbl_z0):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "font-weight: bold; font-size: 13px; padding: 6px; "
                "border: 1px solid #ccc; border-radius: 4px; background: #f5f5f5;"
            )
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            coord_row.addWidget(lbl)
        rl.addLayout(coord_row)

        # Genauigkeitszeile
        acc_row = QHBoxLayout()
        self.lbl_sx = QLabel("σX: —")
        self.lbl_sy = QLabel("σY: —")
        self.lbl_sz = QLabel("σZ: —")
        self.lbl_sigma0 = QLabel("σ₀: —")
        self.lbl_dof = QLabel("f: —")
        for lbl in (self.lbl_sx, self.lbl_sy, self.lbl_sz,
                    self.lbl_sigma0, self.lbl_dof):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #444; padding: 3px;")
            acc_row.addWidget(lbl)
        rl.addLayout(acc_row)

        # Trennlinie
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        rl.addWidget(sep)

        # Residualtabelle
        res_lbl = QLabel("Residuen (Verbesserungen) je Anschlusspunkt:")
        res_lbl.setStyleSheet("font-weight: bold; padding: 2px 0;")
        rl.addWidget(res_lbl)

        self.res_table = QTableWidget(0, 6)
        self.res_table.setHorizontalHeaderLabels([
            "Anschlusspunkt",
            "SD ber. [m]",
            "vSD [m]",
            "ZA ber. [gon]",
            "vZA [gon]",
            "Hz t° [gon]",
        ])
        res_hdr = self.res_table.horizontalHeader()
        res_hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            res_hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.res_table.setMaximumHeight(180)
        self.res_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.res_table.setAlternatingRowColors(True)
        rl.addWidget(self.res_table)

        # Übernehmen / Schließen
        take_row = QHBoxLayout()
        self.btn_use = QPushButton("✓  Standpunkt && Orientierung übernehmen")
        self.btn_use.setEnabled(False)
        self.btn_use.setMinimumHeight(34)
        self.btn_use.setStyleSheet(
            "QPushButton:enabled { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #bdbdbd; color: #757575; "
            "padding: 6px; border-radius: 4px; }"
        )
        self.btn_use.clicked.connect(self._use_result)

        btn_close = QPushButton("Schließen")
        btn_close.setMinimumHeight(34)
        btn_close.clicked.connect(self.close)

        take_row.addWidget(self.btn_use)
        take_row.addStretch()
        take_row.addWidget(btn_close)
        rl.addLayout(take_row)

        main.addWidget(grp_result)

        # Initiale Feldbefüllung – erst hier, da self.table jetzt existiert
        self._on_layer_changed(self.layer_combo.currentLayer())

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _on_layer_changed(self, layer):
        for fld_combo in (self.field_id, self.field_x, self.field_y, self.field_z):
            fld_combo.setLayer(layer)
        self._auto_select_fields(layer)
        self._refresh_ap_combos()

    def _auto_select_fields(self, layer):
        """Versucht, bekannte Plugin-Feldnamen automatisch vorzubelegen.

        Erkannte Muster (AP-Layer / Mess-Layer des Plugins):
          - ID: 'Punktnummer' oder 'Punktnumme'
          - X:  'x' oder 'calc_x'
          - Y:  'y' oder 'calc_y'
          - Z:  'z' oder 'calc_z'
        Wenn X/Y aus der Geometrie kommen sollen ($x/$y), werden die
        Checkboxen aktiviert.
        """
        if layer is None:
            return
        field_names = [f.name() for f in layer.fields()]

        # --- ID-Feld ---
        for candidate in ('Punktnummer', 'Punktnumme'):
            if candidate in field_names:
                idx = self.field_id.findText(candidate)
                if idx >= 0:
                    self.field_id.setCurrentIndex(idx)
                break

        # --- X-Feld ---
        x_set = False
        for candidate in ('x', 'calc_x'):
            if candidate in field_names:
                idx = self.field_x.findText(candidate)
                if idx >= 0:
                    self.field_x.setCurrentIndex(idx)
                    self.chk_geom_x.setChecked(False)
                    x_set = True
                break
        if not x_set and layer.geometryType() == QgsWkbTypes.PointGeometry:
            self.chk_geom_x.setChecked(True)

        # --- Y-Feld ---
        y_set = False
        for candidate in ('y', 'calc_y'):
            if candidate in field_names:
                idx = self.field_y.findText(candidate)
                if idx >= 0:
                    self.field_y.setCurrentIndex(idx)
                    self.chk_geom_y.setChecked(False)
                    y_set = True
                break
        if not y_set and layer.geometryType() == QgsWkbTypes.PointGeometry:
            self.chk_geom_y.setChecked(True)

        # --- Z-Feld ---
        for candidate in ('z', 'calc_z'):
            if candidate in field_names:
                idx = self.field_z.findText(candidate)
                if idx >= 0:
                    self.field_z.setCurrentIndex(idx)
                break

    def _load_measurements(self, layer=None):
        """Lädt alle Messungen aus dem angegebenen bzw. Standard-Messlayer."""
        self._measurements = []
        active_layer = layer if layer is not None else self.mlayer
        if active_layer is None:
            return
        # Feld-Indizes einmalig ermitteln, um KeyErrors zu vermeiden
        flds = active_layer.fields()
        idx_pnr = flds.indexFromName('Punktnummer')
        idx_pnr_alt = flds.indexFromName('Punktnumme')
        idx_hz = flds.indexFromName('mess_ha')
        idx_za = flds.indexFromName('mess_za')
        idx_sd = flds.indexFromName('mess_sd')

        for feat in active_layer.getFeatures():
            # Punktnummer mit Fallbacks
            pnr_attr = None
            if idx_pnr >= 0:
                pnr_attr = feat.attribute(idx_pnr)
            elif idx_pnr_alt >= 0:
                pnr_attr = feat.attribute(idx_pnr_alt)
            pnr = str(pnr_attr) if pnr_attr not in (None, '') else str(feat.id())

            hz = feat.attribute(idx_hz) if idx_hz >= 0 else None
            za = feat.attribute(idx_za) if idx_za >= 0 else None
            sd = feat.attribute(idx_sd) if idx_sd >= 0 else None
            if hz is None or za is None or sd is None:
                continue
            try:
                self._measurements.append({
                    "label": f"{pnr}  |  Hz={_parse_float(hz):.4f}  ZA={_parse_float(za):.4f}  SD={_parse_float(sd):.4f}",
                    "pnr": pnr,
                    "hz": _parse_float(hz),
                    "za": _parse_float(za),
                    "sd": _parse_float(sd),
                })
            except (TypeError, ValueError):
                continue

    def _get_ap_features(self):
        """Gibt [(display_label, feature_id), ...] aller Features im AP-Layer zurück."""
        layer = self.layer_combo.currentLayer()
        if layer is None:
            return []
        id_field = self.field_id.currentField()
        items = []
        for feat in layer.getFeatures():
            if id_field:
                label = str(feat[id_field])
            else:
                label = f"FID {feat.id()}"
            items.append((label, feat.id()))
        return items

    def _get_ap_coords(self, feature_id):
        """
        Gibt (X, Y, Z) eines bekannten-Punkt-Features zurück.
        X/Y werden aus der Geometrie gelesen wenn $x/$y-Checkbox aktiv ist,
        sonst aus den gewählten Attributfeldern.
        """
        layer = self.layer_combo.currentLayer()
        if layer is None:
            return None
        feat = layer.getFeature(feature_id)
        x_field = self.field_x.currentField()
        y_field = self.field_y.currentField()
        z_field = self.field_z.currentField()
        use_geom_x = self.chk_geom_x.isChecked()
        use_geom_y = self.chk_geom_y.isChecked()
        try:
            pt = feat.geometry().asPoint()
            X = pt.x() if use_geom_x else (float(feat[x_field]) if x_field else pt.x())
            Y = pt.y() if use_geom_y else (float(feat[y_field]) if y_field else pt.y())
            Z = float(feat[z_field]) if z_field else 0.0
        except Exception:
            return None
        return X, Y, Z

    # ── Tabellenoperationen ───────────────────────────────────────────────────

    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Spalte 0: Messung (ComboBox)
        combo_m = QComboBox()
        combo_m.addItem("— Messung wählen —", None)
        for m in self._measurements:
            combo_m.addItem(m["label"], m)
        combo_m.currentIndexChanged.connect(
            lambda _idx, r=row: self._on_measurement_changed(r))
        self.table.setCellWidget(row, 0, combo_m)

        # Spalten 1–3: automatisch befüllt, schreibgeschützt
        for col in range(1, 4):
            item = QTableWidgetItem("—")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        # Spalte 4: Anschlusspunkt (ComboBox)
        combo_ap = QComboBox()
        combo_ap.addItem("— Anschlusspunkt wählen —", None)
        for label, fid in self._get_ap_features():
            combo_ap.addItem(label, fid)
        combo_ap.currentIndexChanged.connect(
            lambda _idx, r=row: self._on_ap_changed(r))
        self.table.setCellWidget(row, 4, combo_ap)

        # Spalte 5: bekannte Koordinaten (schreibgeschützt)
        coord_item = QTableWidgetItem("—")
        coord_item.setFlags(coord_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 5, coord_item)

        # Spalte 6: Reflektorhöhe th [m] (editierbar, nur im Modus 'Erweitert')
        th_item = QTableWidgetItem("0.000")
        th_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 6, th_item)

    def _remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _auto_match_anschlusspoint(self, row, measurement_pnr):
        """
        Sucht automatisch einen passenden Anschlusspunkt basierend auf der Punktnummer.
        Wenn der Wert des ID-Feldes eines AP-Features mit der Messungs-Punktnummer übereinstimmt,
        wird dieser AP automatisch in der ComboBox (Spalte 4) ausgewählt.
        """
        layer = self.layer_combo.currentLayer()
        id_field_name = self.field_id.currentField()
        
        if layer is None or not id_field_name:
            return
        
        # Durchsuche alle AP-Features nach einem Match
        for feat in layer.getFeatures():
            ap_id_value = str(feat[id_field_name]).strip() if id_field_name else None
            measurement_pnr_str = str(measurement_pnr).strip()
            
            if ap_id_value and ap_id_value == measurement_pnr_str:
                # Passender AP gefunden – setze die ComboBox
                combo_ap = self.table.cellWidget(row, 4)
                if combo_ap is not None:
                    combo_ap.blockSignals(True)
                    # Suche den Index basierend auf feature_id
                    for i in range(combo_ap.count()):
                        if combo_ap.itemData(i) == feat.id():
                            combo_ap.setCurrentIndex(i)
                            break
                    combo_ap.blockSignals(False)
                    # Trigger die ap_changed Logik
                    self._on_ap_changed(row)
                break

    def _on_measurement_changed(self, row):
        combo = self.table.cellWidget(row, 0)
        if combo is None:
            return
        m = combo.currentData()
        labels = {1: "hz", 2: "za", 3: "sd"}
        for col, key in labels.items():
            item = self.table.item(row, col)
            if item:
                item.setText(f"{m[key]:.4f}" if m else "—")
        
        # Automatische Zuordnung: Wenn eine Messung ausgewählt wurde,
        # versuche den passenden Anschlusspunkt basierend auf der Punktnummer zu finden
        if m is not None:
            self._auto_match_anschlusspoint(row, m["pnr"])

    def _on_ap_changed(self, row):
        combo = self.table.cellWidget(row, 4)
        item = self.table.item(row, 5)
        if combo is None or item is None:
            return
        fid = combo.currentData()
        if fid is None:
            item.setText("—")
            return
        coords = self._get_ap_coords(fid)
        if coords:
            item.setText(f"X={coords[0]:.3f}  Y={coords[1]:.3f}  Z={coords[2]:.3f}")
        else:
            item.setText("Fehler beim Lesen")

    def _refresh_ap_combos(self):
        """Aktualisiert alle AP-ComboBoxen nach Layerwechsel."""
        ap_items = self._get_ap_features()
        for row in range(self.table.rowCount()):
            combo_ap = self.table.cellWidget(row, 4)
            if combo_ap is None:
                continue
            current_fid = combo_ap.currentData()
            combo_ap.blockSignals(True)
            combo_ap.clear()
            combo_ap.addItem("— Anschlusspunkt wählen —", None)
            for label, fid in ap_items:
                combo_ap.addItem(label, fid)
            idx = combo_ap.findData(current_fid)
            combo_ap.setCurrentIndex(max(idx, 0))
            combo_ap.blockSignals(False)
            self._on_ap_changed(row)

    def _refresh_coord_display(self, *args):
        """Aktualisiert die Koordinatenspalte aller Tabellenzeilen (nach Checkbox-Änderung)."""
        if not hasattr(self, 'table'):
            return
        for row in range(self.table.rowCount()):
            self._on_ap_changed(row)
    def _on_measure_layer_changed(self, layer):
        """Wird aufgerufen, wenn der Messlayer im Combo gewechselt wird."""
        self._load_measurements(layer)
        n = len(self._measurements)
        if hasattr(self, 'lbl_measure_info'):
            self.lbl_measure_info.setText(
                f"Wählen Sie für jede Messung den zugehörigen bekannten Anschlusspunkt. "
                f"Mindestens 2 Zuordnungen mit SD und ZA sind erforderlich (≥ 3 empfohlen). "
                f"— {n} Messung(en) verfügbar."
            )
        if not hasattr(self, 'table'):
            return
        for row in range(self.table.rowCount()):
            combo_m = self.table.cellWidget(row, 0)
            if combo_m is None:
                continue
            current_data = combo_m.currentData()
            current_pnr = current_data.get("pnr") if current_data else None
            combo_m.blockSignals(True)
            combo_m.clear()
            combo_m.addItem("— Messung wählen —", None)
            for m in self._measurements:
                combo_m.addItem(m["label"], m)
            if current_pnr is not None:
                for i in range(1, combo_m.count()):
                    d = combo_m.itemData(i)
                    if d and d.get("pnr") == current_pnr:
                        combo_m.setCurrentIndex(i)
                        break
            combo_m.blockSignals(False)
            self._on_measurement_changed(row)

    def _reload_measurements(self):
        """Lädt Messungen aus dem Messlayer neu und aktualisiert alle Messung-ComboBoxen."""
        self._on_measure_layer_changed(self._measure_layer_combo.currentLayer())
        n = len(self._measurements)
        msg = f"{n} Messung(en) geladen." if n > 0 else "Keine Messungen im Layer gefunden."
        QMessageBox.information(self, "Messungen neu geladen", msg)
    # ── Datenerfassung ────────────────────────────────────────────────────────

    def _collect_observations(self):
        """
        Liest alle Tabellenzeilen aus und gibt eine Liste von Beobachtungs-
        dicts zurück: {name, X, Y, Z, hz_gon, za_gon, sd_m}.
        Wirft ValueError bei fehlenden/ungültigen Angaben.
        """
        observations = []
        for row in range(self.table.rowCount()):
            combo_m = self.table.cellWidget(row, 0)
            combo_ap = self.table.cellWidget(row, 4)
            if combo_m is None or combo_ap is None:
                raise ValueError(f"Zeile {row + 1}: Widgets fehlen.")

            m = combo_m.currentData()
            fid = combo_ap.currentData()
            if m is None:
                raise ValueError(f"Zeile {row + 1}: Keine Messung ausgewählt.")
            if fid is None:
                raise ValueError(f"Zeile {row + 1}: Kein Anschlusspunkt ausgewählt.")

            coords = self._get_ap_coords(fid)
            if coords is None:
                raise ValueError(
                    f"Zeile {row + 1}: Koordinaten des Anschlusspunkts nicht lesbar.")

            # AP-Name aus ID-Feld
            ap_name = f"FID {fid}"
            id_field = self.field_id.currentField()
            layer = self.layer_combo.currentLayer()
            if layer and id_field:
                feat = layer.getFeature(fid)
                ap_name = str(feat[id_field])

            observations.append({
                "name": ap_name,
                "X": coords[0],
                "Y": coords[1],
                "Z": coords[2],
                "hz_gon": m["hz"],
                "za_gon": m["za"],
                "sd_m": m["sd"],
                "th_m": self._read_th_for_row(row),
            })
        return observations

    def _read_th_for_row(self, row):
        """Liest die Reflektorhöhe (th) aus Spalte 6. 0.0 wenn leer/ungültig."""
        item = self.table.item(row, 6)
        if item is None:
            return 0.0
        try:
            return _parse_float(item.text())
        except (ValueError, TypeError):
            return 0.0

    # ── Modus-Umschaltung ─────────────────────────────────────────────────────

    def _on_mode_changed(self, _idx):
        is_extended = self.mode_combo.currentData() == "extended"
        self.grp_extended.setVisible(is_extended)
        # Spalte 'th [m]' nur im erweiterten Modus zeigen
        self.table.setColumnHidden(6, not is_extended)

    # ── Berechnung ────────────────────────────────────────────────────────────

    def _calculate(self):
        """Führt den Rückwärtsschnitt durch und zeigt die Ergebnisse an."""
        # Eingaben sammeln
        try:
            self._observations = self._collect_observations()
        except ValueError as e:
            QMessageBox.warning(self, "Eingabefehler", str(e))
            return

        n = len(self._observations)
        if n < 2:
            QMessageBox.warning(
                self,
                "Zu wenige Beobachtungen",
                "Mindestens 2 Zeilen mit vollständigen Messungen (SD + ZA) "
                "sind erforderlich.\n"
                "Für eine statistisch abgesicherte Lösung werden ≥ 3 empfohlen."
            )
            return

        obs = self._observations

        # Arrays aufbauen
        observed_points = np.array([[o["X"], o["Y"], o["Z"]] for o in obs])
        slant_distances = np.array([o["sd_m"] for o in obs])

        # Zenitwinkel (gon) → Höhenwinkel (rad)
        # ZA = 100 gon entspricht horizontal (v = 0)
        # ZA < 100: aufwärts (v > 0); ZA > 100: abwärts (v < 0)
        v_angles = np.array([
            _gon_to_rad(100.0 - o["za_gon"]) for o in obs
        ])

        # Horizontalrichtungen (gon) → Radiant für den Ausgleich
        # Hz-Werte auf [0, 400) normieren (Tachymeter kann > 400 gon liefern)
        hz_angles = np.array([_gon_to_rad(o["hz_gon"] % 400.0) for o in obs])

        # Resektionsberechnung – Modus auswerten
        # Standard: bisheriger Algorithmus, Höhenwinkel-Modell, gleiche Gewichte.
        # Erweitert: konformes Modell mit ih/th, Refraktion, optional
        # Maßstab/Add für SD und a-priori-Sigmen pro Beobachtungstyp.
        mode = self.mode_combo.currentData() if hasattr(self, "mode_combo") else "standard"
        try:
            if mode == "extended":
                # Reflektorhöhen aus Tabelle, Instrumentenhöhe aus Eingabefeld
                target_heights = np.array([o["th_m"] for o in obs])
                try:
                    ih_val = _parse_float(self.input_ih.text())
                except ValueError:
                    ih_val = 0.0

                try:
                    k_val = _parse_float(self.input_k.text())
                    R_val = _parse_float(self.input_R.text())
                    sigma_sd = _parse_float(self.input_sigma_sd.text())
                    sigma_hz_mgon = _parse_float(self.input_sigma_hz.text())
                    sigma_za_mgon = _parse_float(self.input_sigma_za.text())
                except ValueError as ex:
                    QMessageBox.warning(
                        self, "Eingabefehler",
                        f"Erweiterte Parameter ungültig:\n{ex}")
                    return
                # mgon → rad
                sigma_hz_rad = _gon_to_rad(sigma_hz_mgon / 1000.0)
                sigma_za_rad = _gon_to_rad(sigma_za_mgon / 1000.0)

                # Zenitwinkel als ZA (rad), nicht als Höhenwinkel
                zenith_rad = np.array([_gon_to_rad(o["za_gon"]) for o in obs])

                result = resection_extended(
                    observed_points,
                    measured_slant_distances=slant_distances,
                    measured_hz_angles=hz_angles,
                    measured_zenith_angles=zenith_rad,
                    instrument_height=ih_val,
                    target_heights=target_heights,
                    refraction_coefficient=k_val,
                    apply_earth_curvature=self.chk_earth_curv.isChecked(),
                    earth_radius=R_val,
                    sigma_sd=max(sigma_sd, 1e-9),
                    sigma_hz=max(sigma_hz_rad, 1e-12),
                    sigma_za=max(sigma_za_rad, 1e-12),
                    estimate_scale_sd=self.chk_estimate_scale.isChecked(),
                    estimate_add_sd=self.chk_estimate_add.isChecked(),
                    obs_labels=[o["name"] for o in obs],
                )
            else:
                result = resection(
                    observed_points,
                    measured_slant_distances=slant_distances,
                    measured_v_angles=v_angles,
                    measured_hz_angles=hz_angles,
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Berechnungsfehler",
                f"Die Berechnung konnte nicht durchgeführt werden:\n\n{e}"
            )
            return

        X_P, Y_P, Z_P = result.position

        # Orientierung z₀ aus dem Ausgleich (4. Unbekannte)
        if result.orientation is not None:
            z0_rad = result.orientation
            z0_gon = _normalize_gon(_rad_to_gon(z0_rad))
        else:
            # Fallback: post-hoc aus Hz-Messungen ableiten
            z0_list = []
            for o in obs:
                t_rad = math.atan2(o["X"] - X_P, o["Y"] - Y_P)
                t_gon = _normalize_gon(_rad_to_gon(t_rad))
                z0_i = _normalize_gon(t_gon - o["hz_gon"])
                z0_list.append(z0_i)
            z0_gon = _mean_angle_gon(z0_list)
            z0_rad = _gon_to_rad(z0_gon)

        self._result_z0_rad = z0_rad
        self._result = result

        self._display_results(result, obs, X_P, Y_P, Z_P, z0_gon)
        self.btn_use.setEnabled(True)

    def _display_results(self, result, obs, X_P, Y_P, Z_P, z0_gon):
        """Füllt alle Ergebniswidgets mit den berechneten Werten."""
        std = result.std_dev

        self.lbl_x.setText(f"X (Rechts): {X_P:.4f} m")
        self.lbl_y.setText(f"Y (Hoch):   {Y_P:.4f} m")
        self.lbl_z.setText(f"Z (Höhe):   {Z_P:.4f} m")
        self.lbl_z0.setText(f"z₀: {z0_gon:.4f} gon")

        self.lbl_sx.setText(f"σX: {std[0]:.4f} m")
        self.lbl_sy.setText(f"σY: {std[1]:.4f} m")
        self.lbl_sz.setText(f"σZ: {std[2]:.4f} m")
        self.lbl_sigma0.setText(f"σ₀: {result.sigma0:.4f}")
        self.lbl_dof.setText(f"Freiheitsgrade f: {result.dof}")

        # Residualtabelle füllen
        self.res_table.setRowCount(0)

        # Schwellwerte für Liegenschaftsvermessungen Baden-Württemberg (DIN 18709)
        # WARN: ±20 mgon/mm, ERROR: ±50 mgon/mm
        THRESHOLD_HZ_WARN = 20.0    # mgon
        THRESHOLD_HZ_ERROR = 50.0   # mgon
        THRESHOLD_SD_WARN = 20.0    # mm
        THRESHOLD_SD_ERROR = 50.0   # mm
        THRESHOLD_ZA_WARN = 20.0    # mgon
        THRESHOLD_ZA_ERROR = 50.0   # mgon

        # Orientierung z₀ bereits berechnet, wird benötigt für Hz-Residuen
        # (z0_gon wurde oben schon berechnet)

        for i, o in enumerate(obs):
            r = self.res_table.rowCount()
            self.res_table.insertRow(r)

            # Berechnete Sollwerte
            diff = np.array([o["X"], o["Y"], o["Z"]]) - result.position
            sd_calc = float(np.linalg.norm(diff))
            dh = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
            if dh < 1e-10:
                dh = 1e-10
            za_calc_gon = 100.0 - _rad_to_gon(math.atan2(diff[2], dh))

            # Residuen (Verbesserungen)
            sd_res = o["sd_m"] - sd_calc
            za_res = o["za_gon"] - za_calc_gon

            # Richtungswinkel t (berechneter Sollazimut)
            t_rad = math.atan2(diff[0], diff[1])
            t_gon = _normalize_gon(_rad_to_gon(t_rad))

            # Hz-Residuum
            hz_res_gon = _normalize_gon(t_gon - z0_gon - o["hz_gon"])
            if hz_res_gon > 200.0:
                hz_res_gon -= 400.0
            hz_res_mgon = hz_res_gon * 1000.0

            items = [
                o["name"],
                f"{sd_calc:.4f}",
                f"{sd_res:+.4f}",
                f"{za_calc_gon:.4f}",
                f"{za_res:+.4f}",
                f"{t_gon:.4f}",
            ]
            for col, text in enumerate(items):
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignCenter)
                self.res_table.setItem(r, col, it)

            # Auffällige Residuen rot/orange hinterlegen
            bg_color = None
            if abs(hz_res_mgon) > THRESHOLD_HZ_ERROR or abs(sd_res * 1000) > THRESHOLD_SD_ERROR or abs(za_res * 1000) > THRESHOLD_ZA_ERROR:
                # ERROR: Tiefrot
                bg_color = QColor(255, 100, 100)
            elif abs(hz_res_mgon) > THRESHOLD_HZ_WARN or abs(sd_res * 1000) > THRESHOLD_SD_WARN or abs(za_res * 1000) > THRESHOLD_ZA_WARN:
                # WARN: Hellorange
                bg_color = QColor(255, 200, 150)

            if bg_color is not None:
                for col in range(6):
                    self.res_table.item(r, col).setBackground(bg_color)

    # ── Übernahme des Ergebnisses ─────────────────────────────────────────────

    def _use_result(self):
        """Emittiert das Signal mit berechneten Koordinaten und Orientierung."""
        if self._result is None:
            return
        X_P, Y_P, Z_P = self._result.position
        z0_gon = _rad_to_gon(self._result_z0_rad)

        # Per-Punkt-Details für das Protokoll berechnen
        points = []
        for o in self._observations:
            diff = np.array([o["X"], o["Y"], o["Z"]]) - self._result.position
            sd_calc = float(np.linalg.norm(diff))
            dh = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
            if dh < 1e-10:
                dh = 1e-10
            za_calc_gon = 100.0 - _rad_to_gon(math.atan2(diff[2], dh))
            sd_res = o["sd_m"] - sd_calc
            za_res = o["za_gon"] - za_calc_gon
            t_rad = math.atan2(diff[0], diff[1])
            t_gon = _normalize_gon(_rad_to_gon(t_rad))
            hz_res_gon = _normalize_gon(t_gon - z0_gon - o["hz_gon"])
            if hz_res_gon > 200.0:
                hz_res_gon -= 400.0
            hz_res_mgon = hz_res_gon * 1000.0
            points.append({
                'name':     o['name'],
                'ap_x':     o['X'],
                'ap_y':     o['Y'],
                'ap_z':     o['Z'],
                'hz_gon':   o['hz_gon'],
                'za_gon':   o['za_gon'],
                'sd_m':     o['sd_m'],
                'sd_calc':  sd_calc,
                'sd_res_mm': sd_res * 1000.0,
                'za_calc':  za_calc_gon,
                'za_res_mgon': za_res * 1000.0,
                't_gon':    t_gon,
                'hz_res_mgon': hz_res_mgon,
            })

        mode = self.mode_combo.currentData() if hasattr(self, "mode_combo") else "standard"
        details = {
            'std_dev':    self._result.std_dev.tolist(),
            'sigma0':     self._result.sigma0,
            'dof':        self._result.dof,
            'redundancy': self._result.redundancy,
            'z0_gon':     z0_gon,
            'points':     points,
            'ih':         _parse_float(self.input_ih.text()),  # Instrumentenhöhe
            'sp_id':      self.input_sp_id.text().strip() or "SP",  # Standpunktnummer
            'mode':       mode,  # "standard" oder "extended"
        }
        # Erweiterte Parameter nur im Modus 'extended' hinzufügen
        if mode == "extended":
            details['scale_sd'] = getattr(self._result, 'scale_sd', None)
            details['add_sd'] = getattr(self._result, 'add_sd', None)
            details['scale_hd'] = getattr(self._result, 'scale_hd', None)
            details['add_hd'] = getattr(self._result, 'add_hd', None)
            details['num_obs'] = getattr(self._result, 'num_obs', None)
            details['num_unknowns'] = getattr(self._result, 'num_unknowns', None)
            details['rms_residual'] = getattr(self._result, 'rms_residual', None)
            details['refraction_coefficient'] = getattr(self._result, 'refraction_coefficient', None)
            details['earth_radius'] = getattr(self._result, 'earth_radius', None)
            details['instrument_height'] = getattr(self._result, 'instrument_height', None)
            # A-priori Sigmen aus Eingabefeldern
            try:
                details['sigma_sd_mm'] = _parse_float(self.input_sigma_sd.text()) * 1000.0
                details['sigma_hz_mgon'] = _parse_float(self.input_sigma_hz.text())
                details['sigma_za_mgon'] = _parse_float(self.input_sigma_za.text())
            except (ValueError, AttributeError):
                pass

        self.result_accepted.emit(
            float(X_P), float(Y_P), float(Z_P),
            float(self._result_z0_rad), details
        )
        # Standpunkt-Dialog schließen, wenn vorhanden
        if self._standort_dlg:
            self._standort_dlg.hide()
        self.accept()
