# -*- coding: utf-8 -*-
import os
import json
from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtCore import Qt

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'zielpunkt_dialog_base.ui'))


class ZielpunktDialog(QtWidgets.QDialog, FORM_CLASS):
    """Nicht-modaler Dialog für die Zielpunktdefinition."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(Qt.Tool)
        self._point_types = {}  # {label: {prefix, qml_file, description}}
        self._load_point_types()
        self.combo_point_type.currentIndexChanged.connect(self._on_point_type_changed)

    def _load_point_types(self):
        """Lädt die Punkttypen aus pointTypes.json und befüllt die ComboBox."""
        json_path = os.path.join(os.path.dirname(__file__), 'pointTypes.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            point_prefix_data = data.get('PointPrefixNumbers', {})
            # Normalisiere die Struktur: {label: {prefix, qml_file, description}}
            self._point_types = {}
            for label, value in point_prefix_data.items():
                if isinstance(value, dict):
                    # Neue Struktur: nested object
                    self._point_types[label] = value
                else:
                    # Alte Struktur: simple string (Abwärtskompatibilität)
                    self._point_types[label] = {"prefix": value}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[ZielpunktDialog] pointTypes.json nicht geladen: {e}")
            self._point_types = {}

        self.combo_point_type.clear()
        for label, data in self._point_types.items():
            prefix = data.get('prefix') if isinstance(data, dict) else data
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
        current_id = self.input_targetid.text()
        # Bestehenden Prefix ersetzen: alles vor der letzten Zahl beibehalten
        # Strategie: ersetze den Prefix-Teil (bis inkl. letztem '.')
        # Finde, ob der aktuelle Text einen bekannten Prefix hat
        for label, data in self._point_types.items():
            old_prefix = data.get('prefix') if isinstance(data, dict) else data
            if current_id.startswith(old_prefix):
                # Entferne den alten Prefix und setze den neuen
                suffix = current_id[len(old_prefix):]
                self.input_targetid.setText(prefix + suffix)
                return
        # Kein bekannter Prefix gefunden – setze neuen mit "0" als Startnummer
        self.input_targetid.setText(prefix + "0")

    def get_current_prefix(self):
        """Gibt den aktuellen Prefix aus der ComboBox zurück."""
        return self.combo_point_type.currentData() or ""

    def get_current_point_type_label(self):
        """Gibt das Label des aktuell gewählten Punkttyps zurück."""
        idx = self.combo_point_type.currentIndex()
        if idx >= 0:
            text = self.combo_point_type.itemText(idx)
            # Entferne den " (prefix)"-Teil
            if " (" in text:
                return text.split(" (")[0]
        return ""

    def closeEvent(self, event):
        self.hide()
        event.ignore()
