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

import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QComboBox, QHeaderView, QMessageBox,
    QSizePolicy, QFrame,
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


# ─────────────────────────────────────────────────────────────────────────────
# Winkel-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _gon_to_rad(gon: float) -> float:
    return gon * math.pi / 200.0


def _rad_to_gon(rad: float) -> float:
    return rad * 200.0 / math.pi


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

    result_accepted = pyqtSignal(float, float, float, float)  # x, y, z, z0_rad

    def __init__(self, iface, mlayer, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.mlayer = mlayer
        self._default_mlayer = mlayer
        self._result = None          # ResectionResult-Objekt nach Berechnung
        self._result_z0_rad = None   # berechnete Orientierung in Radiant
        self._observations = []      # aktuelle Beobachtungsliste

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

        ll.addWidget(QLabel("Y / Hoch:"))
        self.field_y = QgsFieldComboBox()
        self.field_y.setAllowEmptyFieldName(True)
        ll.addWidget(self.field_y, 1)

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

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Messung (Pkt.Nr.)",
            "Hz [gon]",
            "ZA [gon]",
            "SD [m]",
            "Anschlusspunkt",
            "Bekannte Koordinaten",
        ])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.Stretch)
        for col in (1, 2, 3):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumHeight(150)
        al.addWidget(self.table)

        main.addWidget(grp_assign)

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
        self._refresh_ap_combos()

    def _load_measurements(self, layer=None):
        """Lädt alle Messungen aus dem angegebenen bzw. Standard-Messlayer."""
        self._measurements = []
        active_layer = layer if layer is not None else self.mlayer
        if active_layer is None:
            return
        for feat in active_layer.getFeatures():
            pnr = str(feat["Punktnummer"] or "")
            hz = feat["mess_ha"]
            za = feat["mess_za"]
            sd = feat["mess_sd"]
            if hz is None or za is None or sd is None:
                continue
            try:
                self._measurements.append({
                    "label": f"{pnr}  |  Hz={float(hz):.4f}  ZA={float(za):.4f}  SD={float(sd):.4f}",
                    "pnr": pnr,
                    "hz": float(hz),
                    "za": float(za),
                    "sd": float(sd),
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
        X/Y/Z werden aus den gewählten Feldern gelesen oder aus der Geometrie.
        """
        layer = self.layer_combo.currentLayer()
        if layer is None:
            return None
        feat = layer.getFeature(feature_id)
        x_field = self.field_x.currentField()
        y_field = self.field_y.currentField()
        z_field = self.field_z.currentField()
        try:
            if x_field and y_field:
                X = float(feat[x_field])
                Y = float(feat[y_field])
                Z = float(feat[z_field]) if z_field else 0.0
            else:
                pt = feat.geometry().asPoint()
                X, Y = pt.x(), pt.y()
                Z = 0.0
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

    def _remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

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
            })
        return observations

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

        # Resektionsberechnung
        try:
            result = resection(
                observed_points,
                measured_slant_distances=slant_distances,
                measured_v_angles=v_angles,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Berechnungsfehler",
                f"Die Berechnung konnte nicht durchgeführt werden:\n\n{e}"
            )
            return

        X_P, Y_P, Z_P = result.position

        # Orientierung z₀ aus Hz-Messungen ableiten
        # Richtungswinkel t (gon, NN = Nord, CW) vom Standpunkt zum AP:
        #   t = atan2(ΔX, ΔY) [Rechts = X-Achse, Hoch = Y-Achse]
        # z₀ = t - Hz  (Orientierungsunbekannte)
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
        self.lbl_sigma0.setText(f"σ₀: {result.sigma0:.4f} m")
        self.lbl_dof.setText(f"Freiheitsgrade f: {result.dof}")

        # Residualtabelle füllen
        self.res_table.setRowCount(0)

        # 3-sigma-Schwellwert für Markierung
        threshold_sd = 3.0 * result.sigma0

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

            # Auffällige Residuen rot hinterlegen (> 3σ bei SD)
            if abs(sd_res) > threshold_sd:
                for col in range(6):
                    self.res_table.item(r, col).setBackground(
                        QColor(255, 200, 200))

    # ── Übernahme des Ergebnisses ─────────────────────────────────────────────

    def _use_result(self):
        """Emittiert das Signal mit berechneten Koordinaten und Orientierung."""
        if self._result is None:
            return
        X_P, Y_P, Z_P = self._result.position
        self.result_accepted.emit(
            float(X_P), float(Y_P), float(Z_P), float(self._result_z0_rad)
        )
        self.accept()
