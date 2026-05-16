# -*- coding: utf-8 -*-
"""
Absteckungsdialog – fährt das Tachymeter auf einen Punkt eines QGIS-Layers an.

Ablauf:
  1. Punktlayer und Anzeige-Feld auswählen.
  2. Gewünschten Punkt im Dropdown wählen.
  3. 2D (nur Hz) oder 3D (Hz + ZA) wählen; bei 3D Höhenfeld oder Geometrie-Z.
  4. "Anfahren" sendet den *DHA…VA…-Befehl an den Tachymeter.
"""

import math

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QRadioButton,
    QButtonGroup, QFrame, QSizePolicy,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont

from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox
from qgis.core import QgsMapLayerProxyModel, QgsWkbTypes


def _normalize_gon(gon: float) -> float:
    """Normiert einen Winkel auf [0, 400) gon. 400 gon = Vollkreis."""
    gon = gon % 400.0
    if gon < 0:
        gon += 400.0
    return gon


def _gon_fmt(value: float) -> str:
    """Formatiert einen Gon-Wert als 7-stelligen String für den *DHA-Befehl."""
    s = f"{value:.4f}".replace('.', '')
    while len(s) < 7:
        s = '0' + s
    return s


class AbsteckungDialog(QDialog):
    """Nicht-modaler Absteckungsdialog."""

    # Stil-Konstante für Ergebnisfelder
    _RESULT_STYLE = (
        "font-size:13px;font-weight:bold;padding:3px 8px;"
        "background:#263238;color:#80cbc4;border-radius:3px;font-family:monospace;"
    )
    _WARN_STYLE = (
        "font-size:13px;font-weight:bold;padding:3px 8px;"
        "background:#263238;color:#ff7043;border-radius:3px;font-family:monospace;"
    )

    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self._plugin = plugin          # Referenz auf QSokkiaPlugin-Instanz
        self.setWindowTitle("Absteckung")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setMinimumWidth(480)
        self._build_ui()
        self._on_layer_changed()

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        _ss = (
            "QGroupBox{font-weight:bold;border:1px solid #c0c0c0;border-radius:5px;"
            "margin-top:8px;padding-top:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;"
            "padding:0 4px;color:#333;}"
            "QPushButton{border:1px solid #aaa;border-radius:3px;padding:3px 8px;"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f8f8f8,stop:1 #e0e0e0);"
            "min-height:22px;}"
            "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #e8f0fe,stop:1 #c8d8f8);border-color:#6699cc;}"
            "QPushButton:pressed{background:#c0c8d8;}"
            "QPushButton:disabled{color:#aaa;background:#f0f0f0;border-color:#d0d0d0;}"
        )
        self.setStyleSheet(_ss)

        # ── 1. Layer & Feld ───────────────────────────────────────────────────
        grp_layer = QGroupBox("Punktlayer & Anzeige-Feld")
        ll = QHBoxLayout(grp_layer)

        ll.addWidget(QLabel("Layer:"))
        self._layer_combo = QgsMapLayerComboBox()
        self._layer_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        self._layer_combo.layerChanged.connect(self._on_layer_changed)
        ll.addWidget(self._layer_combo, 3)

        ll.addWidget(QLabel("Feld:"))
        self._field_combo = QgsFieldComboBox()
        self._field_combo.setAllowEmptyFieldName(False)
        self._field_combo.fieldChanged.connect(self._on_field_changed)
        ll.addWidget(self._field_combo, 2)

        self._layer_combo.layerChanged.connect(self._field_combo.setLayer)

        root.addWidget(grp_layer)

        # ── 2. Punktauswahl ───────────────────────────────────────────────────
        grp_pt = QGroupBox("Abzusteckender Punkt")
        pl = QHBoxLayout(grp_pt)

        pl.addWidget(QLabel("Punkt:"))
        self._point_combo = QComboBox()
        self._point_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._point_combo.currentIndexChanged.connect(self._update_result)
        pl.addWidget(self._point_combo)

        root.addWidget(grp_pt)

        # ── 3. Modus 2D / 3D ──────────────────────────────────────────────────
        grp_mode = QGroupBox("Modus")
        ml = QVBoxLayout(grp_mode)
        ml.setSpacing(4)

        mode_row = QHBoxLayout()
        self._rb_2d = QRadioButton("2D  (nur Hz – keine Höhe)")
        self._rb_3d = QRadioButton("3D  (Hz + ZA – mit Höhe)")
        self._rb_2d.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._rb_2d)
        bg.addButton(self._rb_3d)
        mode_row.addWidget(self._rb_2d)
        mode_row.addWidget(self._rb_3d)
        mode_row.addStretch()
        ml.addLayout(mode_row)

        # Höhen-Unterzeile (nur bei 3D aktiv)
        z_row = QHBoxLayout()
        self._rb_z_field = QRadioButton("Höhe aus Feld:")
        self._rb_z_geom = QRadioButton("Höhe aus Geometrie-Z")
        self._rb_z_geom.setChecked(True)
        zg = QButtonGroup(self)
        zg.addButton(self._rb_z_field)
        zg.addButton(self._rb_z_geom)
        self._z_field_combo = QgsFieldComboBox()
        self._z_field_combo.setAllowEmptyFieldName(True)
        self._z_field_combo.setEnabled(False)
        z_row.addWidget(self._rb_z_field)
        z_row.addWidget(self._z_field_combo, 2)
        z_row.addSpacing(12)
        z_row.addWidget(self._rb_z_geom)
        z_row.addStretch()
        ml.addLayout(z_row)

        self._rb_2d.toggled.connect(self._on_mode_changed)
        self._rb_3d.toggled.connect(self._on_mode_changed)
        self._rb_z_field.toggled.connect(
            lambda on: self._z_field_combo.setEnabled(on))
        self._rb_z_field.toggled.connect(lambda _: self._update_result())
        self._rb_z_geom.toggled.connect(lambda _: self._update_result())
        self._z_field_combo.fieldChanged.connect(self._update_result)

        # Höhen-Felder standardmäßig ausblenden
        self._rb_z_field.setVisible(False)
        self._rb_z_geom.setVisible(False)
        self._z_field_combo.setVisible(False)

        root.addWidget(grp_mode)

        # ── 4. Ergebnisanzeige ────────────────────────────────────────────────
        grp_res = QGroupBox("Berechnete Abstechwerte")
        rl = QVBoxLayout(grp_res)
        rl.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Richtungswinkel t:"))
        self._lbl_t = QLabel("—")
        self._lbl_t.setStyleSheet(self._RESULT_STYLE)
        row1.addWidget(self._lbl_t)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Hz (Rohwert):"))
        self._lbl_hz = QLabel("—")
        self._lbl_hz.setStyleSheet(self._RESULT_STYLE)
        row1.addWidget(self._lbl_hz)
        row1.addStretch()
        rl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("ZA (Zenitwinkel):"))
        self._lbl_za = QLabel("—")
        self._lbl_za.setStyleSheet(self._RESULT_STYLE)
        row2.addWidget(self._lbl_za)
        row2.addSpacing(12)
        row2.addWidget(QLabel("Horizontaldistanz:"))
        self._lbl_hd = QLabel("—")
        self._lbl_hd.setStyleSheet(self._RESULT_STYLE)
        row2.addWidget(self._lbl_hd)
        row2.addSpacing(12)
        row2.addWidget(QLabel("\u0394H:"))
        self._lbl_dh = QLabel("—")
        self._lbl_dh.setStyleSheet(self._RESULT_STYLE)
        row2.addWidget(self._lbl_dh)
        row2.addStretch()
        rl.addLayout(row2)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color:#b71c1c;font-style:italic;")
        self._lbl_status.setWordWrap(True)
        rl.addWidget(self._lbl_status)

        root.addWidget(grp_res)

        # ── 5. Schaltflächen ──────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_goto = QPushButton("▶  Anfahren")
        self._btn_goto.setToolTip(
            "Tachymeter auf den berechneten Hz- (und ZA-)Wert drehen")
        self._btn_goto.setStyleSheet(
            "QPushButton:enabled{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #e3f2fd,stop:1 #bbdefb);border-color:#64b5f6;font-weight:bold;}")
        self._btn_goto.setEnabled(False)
        self._btn_goto.clicked.connect(self._goto)
        btn_row.addWidget(self._btn_goto)

        btn_close = QPushButton("Schließen")
        btn_close.clicked.connect(self.hide)
        btn_row.addWidget(btn_close)

        root.addLayout(btn_row)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_layer_changed(self, layer=None):
        layer = layer or self._layer_combo.currentLayer()
        self._field_combo.setLayer(layer)
        self._z_field_combo.setLayer(layer)
        self._rebuild_point_combo()

    def _on_field_changed(self, _field=None):
        self._rebuild_point_combo()

    def _on_mode_changed(self):
        is_3d = self._rb_3d.isChecked()
        self._rb_z_field.setVisible(is_3d)
        self._rb_z_geom.setVisible(is_3d)
        self._z_field_combo.setVisible(is_3d)
        self._update_result()

    def _rebuild_point_combo(self):
        """Füllt das Punkt-Dropdown neu aus dem gewählten Layer/Feld."""
        self._point_combo.blockSignals(True)
        self._point_combo.clear()
        layer = self._layer_combo.currentLayer()
        field = self._field_combo.currentField()
        if layer is not None:
            for feat in layer.getFeatures():
                label = str(feat[field]) if field else f"FID {feat.id()}"
                self._point_combo.addItem(label, feat.id())
        self._point_combo.blockSignals(False)
        self._update_result()

    def _get_target_coords(self):
        """
        Gibt (x, y, z_or_None) des gewählten Punktes zurück.
        z ist None im 2D-Modus.
        """
        layer = self._layer_combo.currentLayer()
        if layer is None:
            return None
        fid = self._point_combo.currentData()
        if fid is None:
            return None
        feat = layer.getFeature(fid)
        geom = feat.geometry()
        if geom is None or geom.isNull():
            return None
        pt = geom.asPoint()
        x, y = pt.x(), pt.y()
        z = None
        if self._rb_3d.isChecked():
            if self._rb_z_field.isChecked():
                zf = self._z_field_combo.currentField()
                try:
                    z = float(feat[zf]) if zf else None
                except (TypeError, ValueError):
                    z = None
            else:
                # Geometrie-Z
                if geom.constGet() and hasattr(geom.constGet(), 'z'):
                    try:
                        z = geom.constGet().z()
                    except Exception:
                        z = None
                if z is None:
                    # WKB-Fallback
                    try:
                        pt3d = geom.vertexAt(0)
                        z = pt3d.z() if pt3d.z() == pt3d.z() else None  # NaN-Check
                    except Exception:
                        z = None
        return x, y, z

    def _update_result(self):
        """Berechnet und zeigt die Abstechwerte."""
        self._btn_goto.setEnabled(False)
        self._lbl_t.setText("—")
        self._lbl_hz.setText("—")
        self._lbl_za.setText("—")
        self._lbl_hd.setText("—")
        self._lbl_dh.setText("—")
        self._lbl_status.setText("")

        coords = self._get_target_coords()
        if coords is None:
            self._lbl_status.setText("Kein Punkt gewählt.")
            return

        tx, ty, tz = coords
        p = self._plugin
        sp = p.sp
        sp_x = sp.get('RECHTS', 0.0)
        sp_y = sp.get('HOCH', 0.0)
        sp_z = sp.get('H', 0.0)
        ih = sp.get('ih', 0.0)

        dx = tx - sp_x
        dy = ty - sp_y
        hd = math.hypot(dx, dy)

        if hd < 1e-6:
            self._lbl_status.setText(
                "Ziel liegt auf dem Standpunkt – Richtungswinkel unbestimmt.")
            return

        # Geodätischer Richtungswinkel zum Ziel [gon]
        t_gon = _normalize_gon(math.atan2(dx, dy) * 200.0 / math.pi)

        # Hz-Rohwert (Instrumentenablesung) = t - z0
        orientation_gon = p.orientation * 200.0 / math.pi
        hz_raw = _normalize_gon(t_gon - orientation_gon)

        self._lbl_t.setText(f"{t_gon:.4f} gon")
        self._lbl_hz.setText(f"{hz_raw:.4f} gon")
        self._lbl_hd.setText(f"{hd:.4f} m")

        # ZA-Berechnung (3D)
        if self._rb_3d.isChecked() and tz is not None:
            dh = tz - (sp_z + ih)
            za_gon = (100.0 - math.atan2(dh, hd) * 200.0 / math.pi) % 400.0
            self._lbl_za.setText(f"{za_gon:.4f} gon")
            self._lbl_dh.setText(f"{dh:+.4f} m")
        elif self._rb_3d.isChecked() and tz is None:
            self._lbl_status.setText(
                "Keine Höhe für den Zielpunkt verfügbar (Feld leer oder keine Geometrie-Z).")
            self._lbl_za.setText("—")
        else:
            self._lbl_za.setText("2D – nicht gesetzt")

        self._btn_goto.setEnabled(True)

    def _goto(self):
        """Sendet den *DHA…VA…-Befehl an das Tachymeter."""
        p = self._plugin
        if not hasattr(p, 'serial') or p.serial is None or not p.serial.is_open:
            self._lbl_status.setText(
                "Kein Tachymeter verbunden – bitte zuerst verbinden.")
            return

        coords = self._get_target_coords()
        if coords is None:
            return

        tx, ty, tz = coords
        sp = p.sp
        sp_x = sp.get('RECHTS', 0.0)
        sp_y = sp.get('HOCH', 0.0)
        sp_z = sp.get('H', 0.0)
        ih = sp.get('ih', 0.0)

        dx = tx - sp_x
        dy = ty - sp_y
        hd = math.hypot(dx, dy)
        if hd < 1e-6:
            return

        orientation_gon = p.orientation * 200.0 / math.pi
        hz_raw = _normalize_gon(math.atan2(dx, dy) * 200.0 / math.pi - orientation_gon)

        # ZA: bei 2D aktuellen ZA-Wert des Instruments beibehalten
        if self._rb_3d.isChecked() and tz is not None:
            dh = tz - (sp_z + ih)
            za_gon = _normalize_gon(100.0 - math.atan2(dh, hd) * 200.0 / math.pi)
        else:
            za_gon = p.measureValues.get('za', 100.0)

        ha_str = _gon_fmt(hz_raw)
        za_str = _gon_fmt(za_gon)

        command = f"*DHA{ha_str}VA{za_str}\r\n".encode('utf-8')
        try:
            p.serial.write(command)
            self._lbl_status.setStyleSheet("color:#1b5e20;font-style:italic;")
            self._lbl_status.setText(
                f"Befehl gesendet: Hz={hz_raw:.4f} gon, ZA={za_gon:.4f} gon")
        except Exception as e:
            self._lbl_status.setStyleSheet("color:#b71c1c;font-style:italic;")
            self._lbl_status.setText(f"Fehler beim Senden: {e}")

    def closeEvent(self, event):
        self.hide()
        event.ignore()
