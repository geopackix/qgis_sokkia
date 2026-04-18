# -*- coding: utf-8 -*-
import os
from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtCore import Qt

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'standort_dialog_base.ui'))


class StandortDialog(QtWidgets.QDialog, FORM_CLASS):
    """Nicht-modaler Dialog für die Standortdefinition."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(Qt.Tool)

    def closeEvent(self, event):
        self.hide()
        event.ignore()
