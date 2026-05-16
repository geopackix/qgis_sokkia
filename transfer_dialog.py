# -*- coding: utf-8 -*-
"""Transfer-Dialog: Koordinaten zwischen QGIS-Layer und Sokkia-Tachymeter austauschen."""

import math
import os
import threading
import time

from qgis.PyQt.QtCore import QTimer, QVariant, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMapLayerProxyModel,
    QgsPointXY,
    QgsProject,
    QgsVectorLayer,
)
from qgis.gui import QgsFieldComboBox, QgsMapLayerComboBox

from .sdr33.coordinate import Coordinate
from .sdr33.sdr33message import Sdr33Export


class TransferDialog(QDialog):
    """Dialog zum Upload (Layer → SDR33) und Download (SDR33-Datei / seriell → Layer)."""

    _serial_data_received = pyqtSignal(str)
    _serial_send_finished = pyqtSignal(bool, str)   # success, message

    def __init__(self, iface, serial_connection=None, parent=None,
                 transfer_mode_setter=None):
        """
        Args:
            iface:  QGIS interface
            serial_connection:  offenes ``serial.Serial``-Objekt (oder None)
            parent:  Parent-Widget
            transfer_mode_setter:  Callable(bool) zum (De-)Aktivieren des
                Transfer-Modus im Haupt-Plugin (pausiert normalen readSerial).
        """
        super().__init__(parent)
        self.iface = iface
        self.serial = serial_connection
        self._set_transfer_mode = transfer_mode_setter
        self.setWindowTitle("Koordinaten-Transfer (SDR33)")
        self.setMinimumWidth(560)

        # ── Empfangs-State ────────────────────────────────────────────
        self._rx_buffer = ""
        self._rx_stop = threading.Event()
        self._rx_thread = None
        self._tx_thread = None

        self._build_ui()

        # Thread-sicheres Signal → UI
        self._serial_data_received.connect(self._on_serial_rx)
        self._serial_send_finished.connect(self._on_send_finished)

    def closeEvent(self, event):
        """Empfangsthread sauber beenden wenn Dialog geschlossen wird."""
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_stop.set()
            if self._set_transfer_mode:
                self._set_transfer_mode(False)
        event.accept()

    # ==================================================================
    #  UI aufbauen
    # ==================================================================
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Upload ────────────────────────────────────────────────────
        upload_group = QGroupBox("Upload (Layer → Tachymeter / Datei)")
        ul = QVBoxLayout(upload_group)

        form = QFormLayout()
        form.setHorizontalSpacing(8)

        # Layerauswahl
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        form.addRow("Punktlayer:", self.layer_combo)

        # Koordinatenzuordnung
        self.x_mapping = QComboBox()
        self.x_mapping.addItems(["Rechtswert (Easting)", "Hochwert (Northing)"])
        form.addRow("X-Koordinate ist:", self.x_mapping)

        self.y_label = QLabel("→ Y = Hochwert (Northing)")
        self.y_label.setStyleSheet("color:#666; font-style:italic;")
        form.addRow("", self.y_label)

        # Punktname
        self.name_field = QgsFieldComboBox()
        self.name_field.setAllowEmptyFieldName(True)
        form.addRow("Punktname aus:", self.name_field)

        # Beschreibung
        self.desc_field = QgsFieldComboBox()
        self.desc_field.setAllowEmptyFieldName(True)
        form.addRow("Beschreibung aus:", self.desc_field)

        # Jobname
        self.job_name = QLineEdit("Job1")
        form.addRow("Jobname:", self.job_name)

        ul.addLayout(form)

        # Höhenwert
        height_group = QGroupBox("Höhenwert")
        hl = QVBoxLayout(height_group)

        self.height_z_radio = QRadioButton("Z-Wert der Geometrie")
        self.height_z_radio.setChecked(True)
        self.height_attr_radio = QRadioButton("Aus Attribut:")
        self._height_btn_group = QButtonGroup(self)
        self._height_btn_group.addButton(self.height_z_radio)
        self._height_btn_group.addButton(self.height_attr_radio)

        self.height_field = QgsFieldComboBox()
        self.height_field.setAllowEmptyFieldName(True)
        self.height_field.setEnabled(False)

        hl.addWidget(self.height_z_radio)
        row = QHBoxLayout()
        row.addWidget(self.height_attr_radio)
        row.addWidget(self.height_field)
        hl.addLayout(row)
        ul.addWidget(height_group)

        # Buttons – Upload
        btn_row = QHBoxLayout()
        self.btn_save_file = QPushButton("Als Datei speichern …")
        self.btn_send_serial = QPushButton("📡 An Gerät senden (seriell)")
        self._update_serial_btn_state()
        btn_row.addWidget(self.btn_save_file)
        btn_row.addWidget(self.btn_send_serial)
        ul.addLayout(btn_row)

        layout.addWidget(upload_group)

        # ── Download ──────────────────────────────────────────────────
        download_group = QGroupBox("Download (Tachymeter / Datei → Layer)")
        dl = QVBoxLayout(download_group)

        # -- Datei-Import --
        file_row = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("SDR33-Datei auswählen …")
        self.btn_browse = QPushButton("…")
        self.btn_browse.setMaximumWidth(40)
        file_row.addWidget(self.file_path)
        file_row.addWidget(self.btn_browse)
        dl.addLayout(file_row)

        form_dl = QFormLayout()
        self.dl_x_mapping = QComboBox()
        self.dl_x_mapping.addItems(["Rechtswert (Easting)", "Hochwert (Northing)"])
        form_dl.addRow("X-Koordinate ist:", self.dl_x_mapping)
        dl.addLayout(form_dl)

        self.btn_import = QPushButton("Datei als Layer importieren")
        dl.addWidget(self.btn_import)

        # -- Serieller Empfang --
        serial_rx_group = QGroupBox("Serieller Empfang (Tachymeter → QGIS)")
        srl = QVBoxLayout(serial_rx_group)

        self.lbl_rx_status = QLabel("Bereit")
        self.lbl_rx_status.setStyleSheet(
            "padding:3px 6px;background:#f5f5f5;border:1px solid #ccc;"
            "border-radius:3px;font-weight:bold;")
        srl.addWidget(self.lbl_rx_status)

        self.txt_rx_log = QTextEdit()
        self.txt_rx_log.setReadOnly(True)
        self.txt_rx_log.setMaximumHeight(100)
        self.txt_rx_log.setStyleSheet(
            "font-family:monospace;font-size:11px;background:#263238;color:#80cbc4;")
        srl.addWidget(self.txt_rx_log)

        rx_btn_row = QHBoxLayout()
        self.btn_rx_start = QPushButton("▶ Empfang starten")
        self.btn_rx_start.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #e8f5e9,stop:1 #c8e6c9);border-color:#81c784;font-weight:bold;}")
        self.btn_rx_stop = QPushButton("■ Empfang stoppen")
        self.btn_rx_stop.setEnabled(False)
        self.btn_rx_stop.setStyleSheet(
            "QPushButton:enabled{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #ffebee,stop:1 #ffcdd2);border-color:#e57373;font-weight:bold;}")
        self.btn_rx_import = QPushButton("Empfangene Daten als Layer importieren")
        self.btn_rx_import.setEnabled(False)
        rx_btn_row.addWidget(self.btn_rx_start)
        rx_btn_row.addWidget(self.btn_rx_stop)
        srl.addLayout(rx_btn_row)
        srl.addWidget(self.btn_rx_import)

        dl.addWidget(serial_rx_group)
        layout.addWidget(download_group)

        # Schließen
        self.btn_close = QPushButton("Schließen")
        layout.addWidget(self.btn_close, alignment=Qt.AlignmentFlag.AlignRight)

        # ── Signale ───────────────────────────────────────────────────
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.x_mapping.currentIndexChanged.connect(self._update_y_label)
        self.height_attr_radio.toggled.connect(self.height_field.setEnabled)

        self.btn_save_file.clicked.connect(self._upload_to_file)
        self.btn_send_serial.clicked.connect(self._upload_to_serial)

        self.btn_browse.clicked.connect(self._browse_file)
        self.btn_import.clicked.connect(self._import_file)

        self.btn_rx_start.clicked.connect(self._rx_start)
        self.btn_rx_stop.clicked.connect(self._rx_stop_receive)
        self.btn_rx_import.clicked.connect(self._rx_import)

        self.btn_close.clicked.connect(self.close)

        # Initiale Befüllung
        self._on_layer_changed(self.layer_combo.currentLayer())

    # ==================================================================
    #  Hilfsmethoden
    # ==================================================================
    def _has_serial(self):
        return self.serial is not None and self.serial.is_open

    def _update_serial_btn_state(self):
        ok = self._has_serial()
        self.btn_send_serial.setEnabled(ok)

    def _on_layer_changed(self, layer):
        self.name_field.setLayer(layer)
        self.height_field.setLayer(layer)
        self.desc_field.setLayer(layer)

    def _update_y_label(self, index):
        if index == 0:
            self.y_label.setText("→ Y = Hochwert (Northing)")
        else:
            self.y_label.setText("→ Y = Rechtswert (Easting)")

    # ==================================================================
    #  Upload  (Layer → Tachymeter / Datei)
    # ==================================================================
    def _build_export(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            QMessageBox.warning(self, "Fehler", "Kein Punktlayer ausgewählt.")
            return None

        x_is_easting = self.x_mapping.currentIndex() == 0
        use_z_geometry = self.height_z_radio.isChecked()
        name_attr = self.name_field.currentField()
        height_attr = self.height_field.currentField() if not use_z_geometry else None
        desc_attr = self.desc_field.currentField()

        export = Sdr33Export(self.job_name.text())
        count = 0

        for i, feature in enumerate(layer.getFeatures()):
            geom = feature.geometry()
            if geom.isNull() or geom.isEmpty():
                continue

            pt = geom.constGet()
            x_val = pt.x()
            y_val = pt.y()

            if x_is_easting:
                easting, northing = x_val, y_val
            else:
                easting, northing = y_val, x_val

            # Höhenwert
            if use_z_geometry:
                z = pt.z() if pt.is3D() else 0.0
                elevation = 0.0 if math.isnan(z) else z
            else:
                try:
                    elevation = float(feature[height_attr]) if height_attr else 0.0
                except (ValueError, TypeError):
                    elevation = 0.0

            # Punktname
            point_name = str(feature[name_attr]) if name_attr else str(i + 1)

            # Beschreibung
            description = ""
            if desc_attr:
                try:
                    description = str(feature[desc_attr])
                except Exception:
                    pass

            coord = Coordinate(
                point_name=point_name,
                northing=northing,
                easting=easting,
                elevation=elevation,
                description=description,
            )
            export.add_coordinate(coord)
            count += 1

        if count == 0:
            QMessageBox.warning(self, "Fehler", "Keine Punkte im Layer gefunden.")
            return None

        return export

    def _upload_to_file(self):
        export = self._build_export()
        if not export:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "SDR33-Datei speichern",
            "",
            "SDR33 Dateien (*.sdr);;Alle Dateien (*)",
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(export.get_message())

        self.iface.messageBar().pushSuccess(
            "Transfer", f"SDR33-Datei gespeichert: {path}"
        )

    def _upload_to_serial(self):
        if not self._has_serial():
            QMessageBox.warning(self, "Fehler", "Keine serielle Verbindung aktiv.")
            return

        export = self._build_export()
        if not export:
            return

        message = export.get_message()

        # Transfer-Modus aktivieren → readSerial pausieren
        if self._set_transfer_mode:
            self._set_transfer_mode(True)

        # Buttons sperren während des Sendens
        self.btn_send_serial.setEnabled(False)
        self.btn_send_serial.setText("⏳ Sende …")

        # Senden im Hintergrund-Thread (Zeile für Zeile mit Delay)
        self._tx_thread = threading.Thread(
            target=self._tx_send_loop, args=(message,), daemon=True
        )
        self._tx_thread.start()

    def _tx_send_loop(self, message):
        """Sendet SDR33-Daten zeilenweise mit CR+LF und 200ms Pause (Hintergrund-Thread)."""
        try:
            lines = message.split("\n")
            for line in lines:
                if not line and line != lines[-1]:
                    continue
                self.serial.write(line.encode("utf-8"))
                self.serial.write(b"\r\n")
                time.sleep(0.2)
            self._serial_send_finished.emit(True, "Koordinaten an Gerät gesendet.")
        except Exception as e:
            self._serial_send_finished.emit(False, str(e))

    def _on_send_finished(self, success, msg):
        """Slot: Sende-Thread ist fertig (im Haupt-Thread)."""
        if self._set_transfer_mode:
            self._set_transfer_mode(False)
        self.btn_send_serial.setText("📡 An Gerät senden (seriell)")
        self._update_serial_btn_state()
        if success:
            self.iface.messageBar().pushSuccess("Transfer", msg)
        else:
            QMessageBox.critical(self, "Sendefehler", msg)

    # ==================================================================
    #  Download – Datei
    # ==================================================================
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "SDR33-Datei öffnen",
            "",
            "SDR33 Dateien (*.sdr);;Alle Dateien (*)",
        )
        if path:
            self.file_path.setText(path)

    def _import_file(self):
        path = self.file_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(
                self, "Fehler", "Bitte eine gültige SDR33-Datei auswählen."
            )
            return

        with open(path, "r", encoding="utf-8") as f:
            data = f.read()

        coordinates = self._parse_sdr33(data)
        if not coordinates:
            QMessageBox.warning(
                self, "Fehler", "Keine Koordinaten in der Datei gefunden."
            )
            return

        x_is_easting = self.dl_x_mapping.currentIndex() == 0

        layer_name = os.path.splitext(os.path.basename(path))[0]
        self._create_layer_from_coords(coordinates, layer_name, x_is_easting)

    # ==================================================================
    #  Download – Serieller Empfang
    # ==================================================================
    def _rx_start(self):
        """Startet den seriellen Empfangsmodus."""
        if not self._has_serial():
            QMessageBox.warning(self, "Fehler", "Keine serielle Verbindung aktiv.")
            return

        self._rx_buffer = ""
        self._rx_stop.clear()
        self.txt_rx_log.clear()

        # normalen readSerial pausieren
        if self._set_transfer_mode:
            self._set_transfer_mode(True)

        self._rx_thread = threading.Thread(target=self._rx_read_loop, daemon=True)
        self._rx_thread.start()

        self.btn_rx_start.setEnabled(False)
        self.btn_rx_stop.setEnabled(True)
        self.btn_rx_import.setEnabled(False)
        self.btn_send_serial.setEnabled(False)
        self.lbl_rx_status.setText("⏳ Empfange Daten …")
        self.lbl_rx_status.setStyleSheet(
            "padding:3px 6px;background:#fff8e1;border:1px solid #ffd54f;"
            "border-radius:3px;font-weight:bold;color:#f57f17;")

    def _rx_read_loop(self):
        """Hintergrund-Thread: liest serielle Daten bis Stop oder ETX empfangen."""
        try:
            while not self._rx_stop.is_set() and self.serial.is_open:
                raw = self.serial.readline()
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    # Signal in den GUI-Thread
                    self._serial_data_received.emit(text)
                    # Prüfe ob Nachrichtenende (ETX 0x03) enthalten
                    if chr(0x03) in text:
                        break
        except Exception as e:
            self._serial_data_received.emit(f"[FEHLER] {e}\n")

    def _on_serial_rx(self, text):
        """Slot: wird im Haupt-Thread aufgerufen wenn Daten empfangen."""
        self._rx_buffer += text
        # Log aktualisieren – nur druckbare Zeichen
        display = text.replace(chr(0x02), "<STX>").replace(chr(0x03), "<ETX>")
        self.txt_rx_log.append(display.rstrip("\r\n"))

        coords = self._parse_sdr33(self._rx_buffer)
        self.lbl_rx_status.setText(
            f"⏳ Empfange … ({len(coords)} Punkte bisher)")

        # Automatisch stoppen wenn ETX empfangen
        if chr(0x03) in text:
            self._rx_finish()

    def _rx_stop_receive(self):
        """Manueller Stop des Empfangs."""
        self._rx_stop.set()
        self._rx_finish()

    def _rx_finish(self):
        """Empfangsmodus beenden und Ergebnis bereitstellen."""
        self._rx_stop.set()

        # readSerial wieder aktivieren
        if self._set_transfer_mode:
            self._set_transfer_mode(False)

        coords = self._parse_sdr33(self._rx_buffer)
        n = len(coords)

        self.btn_rx_start.setEnabled(True)
        self.btn_rx_stop.setEnabled(False)
        self.btn_rx_import.setEnabled(n > 0)
        self._update_serial_btn_state()

        if n > 0:
            self.lbl_rx_status.setText(f"✓ {n} Punkte empfangen")
            self.lbl_rx_status.setStyleSheet(
                "padding:3px 6px;background:#e8f5e9;border:1px solid #a5d6a7;"
                "border-radius:3px;font-weight:bold;color:#1b5e20;")
        else:
            self.lbl_rx_status.setText("Keine Koordinaten empfangen")
            self.lbl_rx_status.setStyleSheet(
                "padding:3px 6px;background:#ffebee;border:1px solid #ef9a9a;"
                "border-radius:3px;font-weight:bold;color:#b71c1c;")

    def _rx_import(self):
        """Empfangene Daten als Layer importieren."""
        coordinates = self._parse_sdr33(self._rx_buffer)
        if not coordinates:
            QMessageBox.warning(self, "Fehler", "Keine Koordinaten im Puffer.")
            return

        x_is_easting = self.dl_x_mapping.currentIndex() == 0
        from datetime import datetime
        layer_name = f"SDR33-Empfang-{datetime.now().strftime('%H%M%S')}"
        self._create_layer_from_coords(coordinates, layer_name, x_is_easting)

    # ==================================================================
    #  Layer aus Koordinatenliste erzeugen
    # ==================================================================
    def _create_layer_from_coords(self, coordinates, layer_name, x_is_easting):
        crs = QgsProject.instance().crs().authid() or "EPSG:25832"
        layer = QgsVectorLayer(f"PointZ?crs={crs}", layer_name, "memory")

        layer.dataProvider().addAttributes(
            [
                QgsField("Punktname", QVariant.String),
                QgsField("Rechtswert", QVariant.Double),
                QgsField("Hochwert", QVariant.Double),
                QgsField("Hoehe", QVariant.Double),
                QgsField("Beschreibung", QVariant.String),
                QgsField("ih", QVariant.Double),
            ]
        )
        layer.updateFields()

        features = []
        for coord in coordinates:
            if x_is_easting:
                x, y = coord["easting"], coord["northing"]
            else:
                x, y = coord["northing"], coord["easting"]

            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            feat.setAttributes(
                [
                    coord["point_id"],
                    coord["easting"],
                    coord["northing"],
                    coord["elevation"],
                    coord["description"],
                    coord.get("ih"),
                ]
            )
            features.append(feat)

        layer.dataProvider().addFeatures(features)
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)

        self.iface.messageBar().pushSuccess(
            "Transfer",
            f"{len(features)} Punkte importiert als Layer '{layer_name}'.",
        )

    # ==================================================================
    #  SDR33-Parser
    # ==================================================================
    @staticmethod
    def _parse_sdr33(data):
        """Coordinate records (type code 08 und 02) aus SDR33-Daten extrahieren.

        Typ 08 = bekannter Koordinatenpunkt (Known Point)
        Typ 02 = besetzter Standpunkt (Occupied Point / Station)
        Beide Typen haben dieselbe Feldstruktur ab Position 4:
          4-19  Punktname (16)
          20-35 Hochwert / Northing (16)
          36-51 Rechtswert / Easting (16)
          52-67 Höhe / Elevation (16)
          68-83 Beschreibung (Typ 08) / Instrumentenhöhe (Typ 02)
        """
        coordinates = []
        clean = data.replace(chr(0x02), "").replace(chr(0x03), "")
        for line in clean.split("\n"):
            line = line.strip("\r")
            type_code = line[:2] if len(line) >= 2 else ""
            if type_code not in ("08", "02"):
                continue
            if len(line) < 68:
                continue
            try:
                point_id = line[4:20].strip()
                northing = float(line[20:36].strip())
                easting = float(line[36:52].strip())
                elevation = float(line[52:68].strip())
                if type_code == "08":
                    description = line[68:84].strip() if len(line) >= 84 else ""
                    ih = None
                else:
                    # Typ 02: Feld 68-83 enthält die Instrumentenhöhe
                    description = ""
                    try:
                        ih_str = line[68:84].strip() if len(line) >= 84 else ""
                        ih = float(ih_str) if ih_str else None
                    except ValueError:
                        ih = None
                coordinates.append(
                    {
                        "point_id": point_id,
                        "northing": northing,
                        "easting": easting,
                        "elevation": elevation,
                        "description": description,
                        "ih": ih,
                    }
                )
            except (ValueError, IndexError):
                continue
        return coordinates
