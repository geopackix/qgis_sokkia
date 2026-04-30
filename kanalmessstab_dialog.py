# -*- coding: utf-8 -*-
"""
Dialog für die Messung mit einem Kanalmessstab (zwei Prismen am Stab).

Workflow:
  1. Benutzer öffnet den Dialog und gibt Punktnummer + Stab-Geometrie an.
  2. Es werden nacheinander zwei Prismen am Stab gemessen:
       - Prisma 1 (oben, weiter weg vom Ziel)
       - Prisma 2 (unten, näher am Ziel)
  3. Aus den beiden gemessenen 3D-Punkten wird der Vektor `V = P2 − P1`
     gebildet (zeigt vom oberen Prisma zur Spitze des Stabs).
  4. Der Zielpunkt T (z.B. Sohle eines Schachts) wird in Verlängerung des
     Stabs berechnet:  ``T = P2 + V_unit · d_tip``
     mit ``d_tip`` = Abstand zwischen unterem Prisma und Stabspitze.
  5. Die beiden Hilfsmesspunkte werden in einem eigenen Layer
     (``HilfsMesspunkte``) gespeichert, der berechnete Zielpunkt im
     normalen Mess-Layer (``mlayer``).
  6. Der Stab wird als Linienfeature P1 → P2 → T in einem Linienlayer
     (``KanalmessstabLinien``) visualisiert; während des Dialogs zeigt
     ein RubberBand den aktuellen Stand an.

Der Dialog ist nicht-modal und blockiert die normale Messung nicht;
solange er aktiv ist und auf einen Prisma-Treffer wartet, fängt er
die nächste Distanzmessung ab.
"""

import math

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QCheckBox, QFrame, QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal


def _parse_float(value, default=0.0):
    if value is None:
        return default
    s = str(value).strip().replace(",", ".")
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


class KanalmessstabDialog(QDialog):
    """Nicht-modaler Dialog für die Kanalmessstab-Messung.

    Signale:
        request_measurement():  Wird emittiert, wenn der Benutzer eine
            Streckenmessung auslösen möchte (das Plugin sendet den
            Triggerbefehl an das Tachymeter).
        save_target(targetid, x, y, z, p1_xyz, p2_xyz, length_prisms,
                    d_tip):  Wird emittiert, wenn der berechnete Zielpunkt
            gespeichert werden soll.
        save_helper(label, x, y, z, sd, za, ha):  Wird emittiert, sobald
            ein Hilfsmesspunkt (Prisma) gespeichert werden soll.
        rod_visualization(p1_xy, p2_xy, target_xy):  Übergibt die
            aktuelle Stab-Geometrie zur Visualisierung (RubberBand).
    """

    request_measurement = pyqtSignal()
    request_target_settings = pyqtSignal()
    save_target = pyqtSignal(str, float, float, float, object, object, float, float)
    save_helper = pyqtSignal(str, float, float, float, float, float, float)
    rod_visualization = pyqtSignal(object, object, object)

    # ── Konstruktion ──────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kanalmessstab – Verlängerter Punkt")
        self.setWindowFlags(Qt.Tool)
        self.setMinimumWidth(420)

        # Zustand
        self._capture_target = None     # 'P1' | 'P2' | None
        self._p1 = None                 # tuple(x, y, z) | None
        self._p2 = None
        self._target = None
        self._p1_meas = None            # (sd, za, ha)
        self._p2_meas = None

        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(6)

        # ── Zieleinstellungs-Hinweis ──────────────────────────────────────
        grp_target = QGroupBox("Zieleinstellungen am Tachymeter")
        tl = QVBoxLayout(grp_target)
        tl.setSpacing(4)

        self.lbl_target_status = QLabel("Zieltyp: — (noch nicht gesetzt)")
        self.lbl_target_status.setAlignment(Qt.AlignCenter)
        self.lbl_target_status.setStyleSheet(
            "font-weight: bold; font-size: 12px; padding: 6px; "
            "background: #eceff1; border: 1px solid #b0bec5; "
            "border-radius: 4px;")
        tl.addWidget(self.lbl_target_status)

        hint = QLabel(
            "Hinweis: Zieltyp und Prismenkonstante gelten geräteweit für "
            "<b>jede</b> Streckenmessung. Beide Stab-Prismen müssen daher "
            "denselben Zieltyp/Konstante verwenden. Bei Wechsel zwischen "
            "Stabmessung und Sondermessung die Zieleinstellung neu setzen.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color: #5d4037; font-size: 10px; padding: 4px; "
            "background: #fff8e1; border-left: 3px solid #ffb300;")
        tl.addWidget(hint)

        self.btn_open_target = QPushButton("🎯  Zieleinstellungen öffnen …")
        self.btn_open_target.setMinimumHeight(28)
        self.btn_open_target.setToolTip(
            "Öffnet den Zielpunkt-Dialog (Zieltyp, Zielhöhe, Prismenkonstante)")
        self.btn_open_target.clicked.connect(self.request_target_settings.emit)
        tl.addWidget(self.btn_open_target)

        main.addWidget(grp_target)

        # ── Geometrie / Eingaben ───────────────────────────────────────────
        grp_in = QGroupBox("Stab-Geometrie und Zielpunkt")
        form = QFormLayout(grp_in)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.input_pid = QLineEdit("KP.1")
        self.input_pid.setToolTip("Punktnummer des berechneten Zielpunktes")
        form.addRow("Punkt-Nr.:", self.input_pid)

        self.input_d_tip = QLineEdit("0.500")
        self.input_d_tip.setToolTip(
            "Abstand vom unteren Prisma (P2) zur Stabspitze in Metern.\n"
            "Der Zielpunkt liegt in Verlängerung des Stabs jenseits von P2.")
        form.addRow("Abstand P2 → Spitze [m]:", self.input_d_tip)

        self.input_d_known = QLineEdit("")
        self.input_d_known.setPlaceholderText("optional, z.B. 1.000")
        self.input_d_known.setToolTip(
            "Bekannter Soll-Abstand zwischen den beiden Prismen am Stab "
            "(nur zur Qualitätskontrolle – wird mit der gemessenen "
            "Distanz verglichen).")
        form.addRow("Sollabstand P1↔P2 [m]:", self.input_d_known)

        self.cb_visualize = QCheckBox("Stab live visualisieren")
        self.cb_visualize.setChecked(True)
        form.addRow("", self.cb_visualize)

        main.addWidget(grp_in)

        # ── Messsteuerung ─────────────────────────────────────────────────
        grp_meas = QGroupBox("Messung")
        ml = QVBoxLayout(grp_meas)

        # Buttons für die zwei Prismen
        btn_row = QHBoxLayout()
        self.btn_meas_p1 = QPushButton("⊕  Prisma 1 (oben) messen")
        self.btn_meas_p1.setMinimumHeight(34)
        self.btn_meas_p1.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; "
            "font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:checked { background-color: #ff8f00; }")
        self.btn_meas_p1.setCheckable(True)
        self.btn_meas_p1.clicked.connect(self._on_meas_p1)

        self.btn_meas_p2 = QPushButton("⊕  Prisma 2 (unten) messen")
        self.btn_meas_p2.setMinimumHeight(34)
        self.btn_meas_p2.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:checked { background-color: #ff8f00; }")
        self.btn_meas_p2.setCheckable(True)
        self.btn_meas_p2.clicked.connect(self._on_meas_p2)

        btn_row.addWidget(self.btn_meas_p1)
        btn_row.addWidget(self.btn_meas_p2)
        ml.addLayout(btn_row)

        # Status pro Prisma
        self.lbl_p1 = QLabel("P1 (oben):  —")
        self.lbl_p2 = QLabel("P2 (unten): —")
        for lbl in (self.lbl_p1, self.lbl_p2):
            lbl.setStyleSheet(
                "padding: 4px; background: #f5f5f5; border: 1px solid #ccc; "
                "border-radius: 3px; font-family: monospace;")
            ml.addWidget(lbl)

        main.addWidget(grp_meas)

        # ── Berechnetes Ergebnis ──────────────────────────────────────────
        grp_res = QGroupBox("Berechneter Zielpunkt")
        rl = QVBoxLayout(grp_res)

        self.lbl_dist = QLabel("|P1−P2|: —")
        self.lbl_dist.setStyleSheet("padding: 2px;")
        rl.addWidget(self.lbl_dist)

        self.lbl_target = QLabel("Zielpunkt T:  —")
        self.lbl_target.setAlignment(Qt.AlignCenter)
        self.lbl_target.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px; "
            "background: #fff8e1; border: 1px solid #ffd54f; "
            "border-radius: 4px;")
        self.lbl_target.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        rl.addWidget(self.lbl_target)

        main.addWidget(grp_res)

        # Trennlinie
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        main.addWidget(sep)

        # ── Aktionen ──────────────────────────────────────────────────────
        act_row = QHBoxLayout()
        self.btn_save = QPushButton("✓  Zielpunkt speichern")
        self.btn_save.setEnabled(False)
        self.btn_save.setMinimumHeight(34)
        self.btn_save.setStyleSheet(
            "QPushButton:enabled { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #bdbdbd; color: #757575; }")
        self.btn_save.clicked.connect(self._on_save)

        self.btn_reset = QPushButton("Zurücksetzen")
        self.btn_reset.setMinimumHeight(34)
        self.btn_reset.clicked.connect(self._reset_measurements)

        btn_close = QPushButton("Schließen")
        btn_close.setMinimumHeight(34)
        btn_close.clicked.connect(self.hide)

        act_row.addWidget(self.btn_save)
        act_row.addWidget(self.btn_reset)
        act_row.addStretch()
        act_row.addWidget(btn_close)
        main.addLayout(act_row)

    # ── Mess-Triggerbuttons ───────────────────────────────────────────────────

    def _on_meas_p1(self):
        if not self.btn_meas_p1.isChecked():
            self._capture_target = None
            return
        self.btn_meas_p2.setChecked(False)
        self._capture_target = "P1"
        self.lbl_p1.setText("P1 (oben):  warte auf Messung …")
        self.request_measurement.emit()

    def _on_meas_p2(self):
        if not self.btn_meas_p2.isChecked():
            self._capture_target = None
            return
        self.btn_meas_p1.setChecked(False)
        self._capture_target = "P2"
        self.lbl_p2.setText("P2 (unten): warte auf Messung …")
        self.request_measurement.emit()

    def is_capturing(self) -> bool:
        """True solange auf einen Prisma-Treffer gewartet wird."""
        return self._capture_target is not None

    # ── Zielstatus aktualisieren ─────────────────────────────────────────────

    def update_target_status(self, target_type: str, prism_constant=None,
                             is_set: bool = True):
        """Aktualisiert die Zieltyp-Anzeige im Dialog.

        target_type: 'Prisma' | 'Reflexfolie' | 'Reflektorlos' | ''
        prism_constant: int|float|None
        is_set: True wenn die Einstellung am Tachymeter aktiv übertragen wurde
        """
        if not is_set or not target_type:
            self.lbl_target_status.setText("Zieltyp: — (noch nicht gesetzt)")
            self.lbl_target_status.setStyleSheet(
                "font-weight: bold; font-size: 12px; padding: 6px; "
                "background: #ffebee; color: #b71c1c; "
                "border: 1px solid #e57373; border-radius: 4px;")
            return
        # Farbe je nach Typ
        if target_type.lower().startswith("prism"):
            bg, fg, bd = "#e3f2fd", "#0d47a1", "#64b5f6"
        elif target_type.lower().startswith("refl") and "los" in target_type.lower():
            bg, fg, bd = "#fff3e0", "#e65100", "#ffb74d"
        else:
            bg, fg, bd = "#e8f5e9", "#1b5e20", "#81c784"
        pk_txt = ""
        if prism_constant is not None:
            pk_txt = f"   |   PK: {prism_constant} mm"
        self.lbl_target_status.setText(
            f"Zieltyp: {target_type}{pk_txt}")
        self.lbl_target_status.setStyleSheet(
            f"font-weight: bold; font-size: 12px; padding: 6px; "
            f"background: {bg}; color: {fg}; "
            f"border: 1px solid {bd}; border-radius: 4px;")

    # ── Messdaten konsumieren ────────────────────────────────────────────────

    def consume_measurement(self, x: float, y: float, z: float,
                            sd: float, za: float, ha: float) -> bool:
        """Verarbeitet eine eingehende Tachymeter-Messung.

        Liefert ``True``, wenn die Messung verbraucht wurde.
        Berechnet 3D-Koordinaten muss der Aufrufer bereitstellen
        (Standpunkt + Orientierung sind dort bekannt).
        """
        if self._capture_target is None:
            return False

        if self._capture_target == "P1":
            self._p1 = (x, y, z)
            self._p1_meas = (sd, za, ha)
            self.lbl_p1.setText(
                f"P1 (oben):  X={x:.3f}  Y={y:.3f}  Z={z:.3f}  "
                f"(SD={sd:.3f})")
            self.btn_meas_p1.setChecked(False)
            self.save_helper.emit(
                f"{self.input_pid.text().strip()}_P1",
                x, y, z, sd, za, ha)
        else:  # P2
            self._p2 = (x, y, z)
            self._p2_meas = (sd, za, ha)
            self.lbl_p2.setText(
                f"P2 (unten): X={x:.3f}  Y={y:.3f}  Z={z:.3f}  "
                f"(SD={sd:.3f})")
            self.btn_meas_p2.setChecked(False)
            self.save_helper.emit(
                f"{self.input_pid.text().strip()}_P2",
                x, y, z, sd, za, ha)

        self._capture_target = None
        self._update_target()
        return True

    # ── Berechnung ────────────────────────────────────────────────────────────

    def _update_target(self):
        if self._p1 is None or self._p2 is None:
            self.lbl_dist.setText("|P1−P2|: —")
            self.lbl_target.setText("Zielpunkt T:  —")
            self.btn_save.setEnabled(False)
            self._target = None
            self._emit_visualization()
            return

        p1 = self._p1
        p2 = self._p2
        vx, vy, vz = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        v_len = math.sqrt(vx * vx + vy * vy + vz * vz)

        # QC-Vergleich mit Sollabstand
        d_known_str = self.input_d_known.text().strip()
        if d_known_str:
            d_known = _parse_float(d_known_str, 0.0)
            if d_known > 0:
                diff_mm = (v_len - d_known) * 1000.0
                self.lbl_dist.setText(
                    f"|P1−P2|: {v_len:.4f} m   (Soll {d_known:.4f} m, "
                    f"Δ {diff_mm:+.1f} mm)")
            else:
                self.lbl_dist.setText(f"|P1−P2|: {v_len:.4f} m")
        else:
            self.lbl_dist.setText(f"|P1−P2|: {v_len:.4f} m")

        if v_len < 1e-6:
            self.lbl_target.setText("⚠ P1 und P2 sind identisch.")
            self.btn_save.setEnabled(False)
            self._target = None
            self._emit_visualization()
            return

        d_tip = _parse_float(self.input_d_tip.text(), 0.0)
        ux, uy, uz = vx / v_len, vy / v_len, vz / v_len
        tx = p2[0] + ux * d_tip
        ty = p2[1] + uy * d_tip
        tz = p2[2] + uz * d_tip
        self._target = (tx, ty, tz)

        self.lbl_target.setText(
            f"X={tx:.4f}   Y={ty:.4f}   Z={tz:.4f}")
        self.btn_save.setEnabled(True)
        self._emit_visualization()

    def _emit_visualization(self):
        if not self.cb_visualize.isChecked():
            self.rod_visualization.emit(None, None, None)
            return
        p1_xy = (self._p1[0], self._p1[1]) if self._p1 else None
        p2_xy = (self._p2[0], self._p2[1]) if self._p2 else None
        t_xy = (self._target[0], self._target[1]) if self._target else None
        self.rod_visualization.emit(p1_xy, p2_xy, t_xy)

    # ── Speichern / Reset ────────────────────────────────────────────────────

    def _on_save(self):
        if self._target is None or self._p1 is None or self._p2 is None:
            return
        targetid = self.input_pid.text().strip() or "KP.0"
        v_len = math.sqrt(sum(
            (self._p2[i] - self._p1[i]) ** 2 for i in range(3)))
        d_tip = _parse_float(self.input_d_tip.text(), 0.0)
        self.save_target.emit(
            targetid,
            self._target[0], self._target[1], self._target[2],
            self._p1, self._p2, v_len, d_tip)
        # Punkt-Nr. inkrementieren falls möglich
        self.input_pid.setText(_increment_id(targetid))
        # Reset für nächste Messung
        self._reset_measurements()

    def _reset_measurements(self):
        self._capture_target = None
        self._p1 = None
        self._p2 = None
        self._p1_meas = None
        self._p2_meas = None
        self._target = None
        self.btn_meas_p1.setChecked(False)
        self.btn_meas_p2.setChecked(False)
        self.lbl_p1.setText("P1 (oben):  —")
        self.lbl_p2.setText("P2 (unten): —")
        self.lbl_dist.setText("|P1−P2|: —")
        self.lbl_target.setText("Zielpunkt T:  —")
        self.btn_save.setEnabled(False)
        self._emit_visualization()

    # ── Schließverhalten ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Capture beenden + Visualisierung löschen, aber Werte beibehalten.
        self._capture_target = None
        self.btn_meas_p1.setChecked(False)
        self.btn_meas_p2.setChecked(False)
        self.rod_visualization.emit(None, None, None)
        self.hide()
        event.ignore()


def _increment_id(s: str) -> str:
    """Inkrementiert das letzte numerische Segment einer Punkt-Nr."""
    import re
    match = re.search(r"[\.\-_]([^.\-_]+)$", s)
    if not match:
        # Nur Zahl?
        if s.isdigit():
            return str(int(s) + 1)
        return s
    teil = match.group(1)
    if teil.isdigit():
        return s[: -len(teil)] + str(int(teil) + 1)
    return s
