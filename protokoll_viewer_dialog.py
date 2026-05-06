# -*- coding: utf-8 -*-
"""
Dialog zur Anzeige des Messprotokolls.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QMessageBox
)
from qgis.PyQt.QtCore import Qt
import os


class ProtokollViewerDialog(QDialog):
    """Dialog zur Anzeige und zum Export des Messprotokolls."""

    def __init__(self, protokoll_text: str, protokoll_file: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Messprotokoll anzeigen")
        self.setGeometry(100, 100, 800, 600)
        self.protokoll_text = protokoll_text
        self.protokoll_file = protokoll_file
        
        self._build_ui()

    def _build_ui(self):
        """Baut die Benutzeroberfläche auf."""
        layout = QVBoxLayout(self)
        
        # Textbereich
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.protokoll_text)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFontFamily("Courier New")
        self.text_edit.setFontPointSize(9)
        layout.addWidget(self.text_edit)
        
        # Button-Zeile
        btn_layout = QHBoxLayout()
        
        # "Speichern unter" Button
        btn_save = QPushButton("💾  Speichern unter...")
        btn_save.clicked.connect(self._save_as)
        btn_layout.addWidget(btn_save)
        
        # "In Editor öffnen" Button (falls Datei vorhanden)
        if self.protokoll_file and os.path.exists(self.protokoll_file):
            btn_edit = QPushButton("📝  In Editor öffnen")
            btn_edit.clicked.connect(self._open_in_editor)
            btn_layout.addWidget(btn_edit)
        
        btn_layout.addStretch()
        
        # "Schließen" Button
        btn_close = QPushButton("Schließen")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)

    def _save_as(self):
        """Speichert das Protokoll unter einem neuen Namen."""
        default_name = "Messprotokoll.txt"
        if self.protokoll_file:
            default_name = os.path.basename(self.protokoll_file)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Protokoll speichern",
            default_name,
            "Textdateien (*.txt);;Alle Dateien (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.protokoll_text)
                QMessageBox.information(
                    self,
                    "Erfolg",
                    f"Protokoll gespeichert:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Fehler",
                    f"Fehler beim Speichern:\n{e}"
                )

    def _open_in_editor(self):
        """Öffnet die Protokolldatei im Standard-Editor."""
        if self.protokoll_file and os.path.exists(self.protokoll_file):
            try:
                import subprocess
                import sys
                if sys.platform == 'win32':
                    os.startfile(self.protokoll_file)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', self.protokoll_file])
                else:
                    subprocess.Popen(['xdg-open', self.protokoll_file])
                QMessageBox.information(
                    self,
                    "Erfolg",
                    "Protokoll wird im Standard-Editor geöffnet..."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Fehler",
                    f"Fehler beim Öffnen der Datei:\n{e}"
                )
