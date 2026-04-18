# -*- coding: utf-8 -*-
"""
/***************************************************************************
 QGISSokkia
 QGIS plugin to connect a sokkia tachymeter (sdr)
-------------------
        begin                : 2025-04-19
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Manuel Hart (geokoord.com)
        email                : mh@geokoord.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt, QVariant, QDateTime, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QIcon, QColor
import os

from datetime import datetime
from qgis.gui import QgsMapCanvas, QgsRubberBand, QgsMapToolEmitPoint, QgsMapLayerComboBox, QgsMapTool, QgsSnapIndicator
from qgis.core import QgsPointXY, QgsPoint, QgsWkbTypes, QgsVectorLayer, QgsFeature, QgsGeometry, QgsProject, QgsField, QgsMapLayerProxyModel, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsPointLocator
from qgis.PyQt.QtWidgets import QAction, QInputDialog, QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QDialogButtonBox, QFileDialog
import serial
import serial.tools.list_ports
import threading
import time
import math
import queue

from .q_sokkia_orientation_arrow import OrientationArrow
from .resection.resection_dialog import ResectionDialog
from .transfer_dialog import TransferDialog
from .standort_dialog import StandortDialog
from .zielpunkt_dialog import ZielpunktDialog
from .fernsteuerung_dialog import FernsteuerungDialog
from .absteckung_dialog import AbsteckungDialog

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the DockWidget
from .q_sokkia_plugin_dockwidget import QGISSokkiaDockWidget
import os.path


def remove_all_rubber_bands(canvas):
    # Alle RubberBand-Objekte entfernen
    items = canvas.scene().items()
    for item in items:
        if isinstance(item, QgsRubberBand):
            canvas.scene().removeItem(item)

    # Aktualisiere die Karte
    canvas.refresh()



class SnapPointTool(QgsMapTool):
    """Map-Tool mit Objektfang-Indikator. Snap wird während Mausbewegung berechnet."""
    pointPicked = pyqtSignal(QgsPointXY)

    def __init__(self, canvas):
        super().__init__(canvas)
        self._snap_indicator = QgsSnapIndicator(canvas)
        self._last_match = None

    def canvasMoveEvent(self, event):
        match = self.canvas().snappingUtils().snapToMap(event.pos())
        self._snap_indicator.setMatch(match)
        self._last_match = match

    def canvasPressEvent(self, event):
        if self._last_match and self._last_match.isValid():
            point = self._last_match.point()
        else:
            point = self.toMapCoordinates(event.pos())
        self._snap_indicator.setMatch(QgsPointLocator.Match())
        self.pointPicked.emit(point)
        self.canvas().unsetMapTool(self)

    def deactivate(self):
        self._snap_indicator.setMatch(QgsPointLocator.Match())
        super().deactivate()


class SavePointDialog(QDialog):
    """Kleiner Dialog vor dem Speichern: Punktnummer + Zielhöhe editierbar, Koordinaten sichtbar."""

    def __init__(self, point_id, x, y, z, th, sd, za, sp_h, sp_ih, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Punkt speichern")
        self._sd = sd
        self._za = za
        self._sp_h = sp_h
        self._sp_ih = sp_ih

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._input_id = QLineEdit(str(point_id))
        form.addRow("Punktnummer:", self._input_id)

        self._input_th = QLineEdit(f"{th:.4f}")
        self._input_th.textChanged.connect(self._recalc_z)
        form.addRow("Zielh\xf6he [m]:", self._input_th)

        self._lbl_x = QLabel(f"{x:.4f}")
        self._lbl_y = QLabel(f"{y:.4f}")
        self._lbl_z = QLabel(f"{z:.4f}")
        for lbl in (self._lbl_x, self._lbl_y, self._lbl_z):
            lbl.setStyleSheet("font-weight:bold;")
        form.addRow("X (Rechts):", self._lbl_x)
        form.addRow("Y (Hoch):", self._lbl_y)
        form.addRow("Z (H\xf6he):", self._lbl_z)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Discard)
        buttons.accepted.connect(self.accept)
        discard_btn = buttons.button(QDialogButtonBox.Discard)
        discard_btn.setText("Verwerfen")
        discard_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _recalc_z(self, text):
        try:
            th = float(text)
            z = self._sp_h + self._sp_ih + self._sd * math.cos(self._za * math.pi / 200) - th
            self._lbl_z.setText(f"{z:.4f}")
        except ValueError:
            pass

    def get_values(self):
        """Gibt (point_id, th) zurück."""
        return self._input_id.text(), float(self._input_th.text())


class QGISSokkia:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'QGISSokkia_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&QGIS Sokkia Plugin')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'QGISSokkia')
        self.toolbar.setObjectName(u'QGISSokkia')
        
        
        #map canvas
        self.canvas = iface.mapCanvas()

        #print "** INITIALIZING QGISSokkia"

        self.pluginIsActive = False
        self.dockwidget = None
        self._standort_dlg = None
        self._zielpunkt_dlg = None
        self._fernsteuerung_dlg = None
        self._absteckung_dlg = None
        
        self.serial = None
        
        
        #Threads
        self.serialStopEvent = threading.Event()
        self.serialthread = None
        self.serialPeriodicEvent = threading.Event()
        self.periodicThread = None
        self.laserState = False
        self.target = 2
        self.targetPrismConstant = 0
        self.targetHeight = 0
        self.sp = {"ID": "SP1", "RECHTS": 0, "HOCH":0, "H": 0, "ih": 0}             #standpunkt
        self.ap = {"ID": "", "RECHTS": 0, "HOCH":0}             #standpunkt
        self.measureValues = {"ha": 0, "za": 0, "sd": 0}        #Messungen
        self.crsName = "EPSG:25832"
        self.orientation = 0
        self.orientationArrow = OrientationArrow(self.crsName)
        self._direction_rubber_band = None  # Live-Richtungslinie auf der Karte
        
        
        #remove_all_rubber_bands(self.canvas)
        #self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        
        #Layer
        self.mlayer = None
        self.splayer = None
        self.aplayer = None
        self._measure_queue = queue.Queue()
        self._transfer_mode = False
        self._protokoll = []          # Protokolleinträge dieser Sitzung
        self._protokoll_meta = {}     # Verbindungsinfos (Port, Baudrate, Gerät, CRS)
        self._protokoll_tempfile = None  # Pfad der automatisch gespeicherten Protokolldatei



    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('QGISSokkia', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/q_sokkia_plugin/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'QSDR'),
            callback=self.run,
            parent=self.iface.mainWindow())

    #--------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # Serielle Schnittstelle zwingend freigeben
        self._close_serial()
        self._clear_direction_rubber_band()

        # Detaildialoge verstecken (nicht zerstören)
        for dlg in (self._standort_dlg, self._zielpunkt_dlg, self._fernsteuerung_dlg):
            if dlg is not None:
                dlg.hide()

        self.pluginIsActive = False


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        # Serielle Schnittstelle freigeben falls noch offen
        self._close_serial()

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&QGIS Sokkia Plugin'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    #--------------------------------------------------------------------------

    def _refresh_serial_ports(self):
        """Füllt die Port-ComboBox mit den aktuell verfügbaren seriellen Ports."""
        combo = self.dockwidget.combo_port
        previous = combo.currentText()
        combo.clear()
        ports = sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)
        for p in ports:
            combo.addItem(p.device, p.description)
            combo.setItemData(combo.count() - 1, f"{p.device} – {p.description}", Qt.ToolTipRole)
        # Gespeicherten Port wiederherstellen
        saved = QSettings().value('qgis_sokkia/last_port', '')
        restore = saved if saved else previous
        idx = combo.findText(restore)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # Zustands-Helper: setzt Enable + Tooltip für alle gesteuerten Btns #
    # ------------------------------------------------------------------ #
    _TT_NEED_CONNECT   = "Bitte zuerst verbinden (\u25b6 Verbinden) oder \u26a1 Initialisieren"
    _TT_NEED_CONNECT_M = "Bitte zuerst mit dem Tachymeter verbinden (\u25b6 Verbinden)"
    _TT_NEED_SP        = "Bitte zuerst einen Standpunkt setzen (\U0001f4cd Standort \u2026)"

    def _apply_connection_state(self, connected: bool, initialized: bool):
        """Setzt Enable-Status und erkl\u00e4rende Tooltips f\u00fcr alle abh\u00e4ngigen Buttons."""
        dw  = self.dockwidget
        sdlg = self._standort_dlg
        zdlg = self._zielpunkt_dlg

        # --- Verbinden / Trennen ---
        dw.btn_connect.setEnabled(not connected)
        dw.btn_connect.setToolTip(
            "Bereits verbunden" if connected
            else "Serielle Verbindung zum Tachymeter herstellen")
        dw.btn_disconnect.setEnabled(connected)
        dw.btn_disconnect.setToolTip(
            "Verbindung zum Tachymeter trennen" if connected
            else "Kein aktiver Tachymeter verbunden")

        # --- Messtasten (nur mit Ger\u00e4t) ---
        for btn, label in [
            (dw.btn_laser,        "Laserpointer ein-/ausschalten"),
            (dw.btn_measure,      "Streckenmessung ausl\u00f6sen und Punkt speichern"),
            (dw.btn_measure_a,    "Winkelmessung ausl\u00f6sen"),
            (dw.btn_measure_stop, "Messung stoppen"),
        ]:
            btn.setEnabled(connected)
            btn.setToolTip(label if connected else self._TT_NEED_CONNECT_M)

        # --- Ziel setzen (nur mit Ger\u00e4t) ---
        zdlg.btn_setTarget.setEnabled(connected)
        zdlg.btn_setTarget.setToolTip(
            "Zieltyp und Prismenkonstante an den Tachymeter senden" if connected
            else self._TT_NEED_CONNECT_M)

        # --- Standpunkt setzen (mit Ger\u00e4t ODER nach Initialisieren) ---
        sp_ok = connected or initialized
        sdlg.btn_setSp_confirm.setEnabled(sp_ok)
        sdlg.btn_setSp_confirm.setToolTip(
            "Standpunkt mit eingegebenen Werten setzen und Orientierung berechnen" if sp_ok
            else self._TT_NEED_CONNECT)

    def _query_device_info(self) -> str:
        """Liest Geräteinformationen vom Sokkia SET RS232.
        1. Liest ein evtl. vorhandenes Begrüßungs-Telegramm (Greeting)
        2. Fragt Firmware-Version mit *R8 ab
        Gibt einen formatierten Info-String zurück."""
        try:
            result_parts = []
            self.serial.timeout = 1

            # 1. Greeting lesen (manche Geräte senden beim Verbinden spontan Daten)
            self.serial.reset_input_buffer()
            time.sleep(0.3)
            waiting = self.serial.in_waiting
            if waiting > 0:
                greeting = self.serial.read(waiting).decode('utf-8', errors='replace').strip()
                if greeting:
                    result_parts.append(greeting)

            # 2. Firmware-Version abfragen (*R8)
            self.serial.reset_input_buffer()
            self.serial.write(b'*R8\r\n')
            time.sleep(0.5)
            lines = []
            for _ in range(4):
                line = self.serial.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace').strip()
                if decoded and decoded != '\x15':
                    lines.append(decoded)
            if lines:
                result_parts.append('  '.join(lines))

            self.serial.timeout = 1
            return '  |  '.join(result_parts) if result_parts else ''
        except Exception as e:
            print(f"[Geräteinfo] {e}")
            return ''

    def connectToSerial(self):
    
        try:    
            print("connect to serial");
            
            
            port = self.dockwidget.combo_port.currentText().strip()
            if not port:
                self.iface.messageBar().pushWarning("Verbindung", "Kein Port angegeben.")
                return
            QSettings().setValue('qgis_sokkia/last_port', port)
            try:
                baudrate = int(self.dockwidget.input_baud.text().strip())
            except ValueError:
                self.iface.messageBar().pushWarning("Verbindung", "Ungültige Baudrate (ganzzahlig erforderlich).")
                return
            # Sicherstellen, dass vorherige Thread-Stopp-Flag zurückgesetzt ist
            self.serialStopEvent.clear()
            self.serial = serial.Serial(port, baudrate, timeout=1)
            
            
            crs = self.dockwidget.mQgsProjectionSelectionWidget.crs()
            self.crsName = crs.authid()
            print(f"Using CRS: {self.crsName}")
            
            
            time.sleep(1)
            
            if self.serial.is_open:
                print("Connection established successfully!")

                # Geräteinfo anfragen (SDR33: ?I = Instrument Info)
                device_info = self._query_device_info()

                self.serialthread = threading.Thread(target=self.readSerial)
                self.serialthread.daemon = True  # makes the thread a daemon thread
                self.serialthread.start() 
                
                self.periodicThread = threading.Thread(target=self.sendPeriodicAngleMeasureCommand)
                self.periodicThread.daemon = True
                #self.periodicThread.start()
                

                # add temp layer with layername
                self.addTempLayer(f"Messungen-{datetime.now().strftime('%d%m%y-%H%M')}")
                self.addSpTempLayer(f"Station-{datetime.now().strftime('%d%m%y-%H%M')}")
                self.addApTempLayer(f"APs-{datetime.now().strftime('%d%m%y-%H%M')}")
                
                self._apply_connection_state(connected=True, initialized=True)
            
                # Layer zur Karte hinzufügen
                QgsProject.instance().addMapLayer(self.mlayer)
                QgsProject.instance().addMapLayer(self.splayer)
                QgsProject.instance().addMapLayer(self.aplayer)
                info_text = f"Verbunden mit {port} ({baudrate} Baud)  |  CRS: {self.crsName}"
                if device_info:
                    info_text += f"  |  {device_info}"
                self.iface.messageBar().pushSuccess("Verbindung", info_text)
                self.dockwidget.lbl_device_info.setText(
                    device_info if device_info else f"{port} · {baudrate} Baud")
                # Protokoll für diese Sitzung starten
                self._protokoll = []
                self._protokoll_meta = {
                    'port': port, 'baudrate': baudrate,
                    'device': device_info or '(unbekannt)',
                    'crs': self.crsName,
                    'start': datetime.now(),
                }
                # Temp-Protokolldatei anlegen
                temp_dir = os.path.join(self.plugin_dir, 'temp_protocols')
                os.makedirs(temp_dir, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                self._protokoll_tempfile = os.path.join(temp_dir, f'Protokoll_{ts}.txt')
                self._protokoll_add('VERBINDUNG', f"Port: {port}  Baudrate: {baudrate}  Gerät: {device_info or '?'}  CRS: {self.crsName}")
            else:
                raise serial.SerialException('Port konnte nicht geöffnet werden.')

        except serial.SerialException as e:
            self.iface.messageBar().pushCritical("Verbindungsfehler", str(e))
        except Exception as e:
            self.iface.messageBar().pushCritical("Fehler", str(e))
    
    
    def addTempLayer(self, name):
        
        self.mlayer = QgsVectorLayer("Point?crs="+self.crsName, name, "memory") #EPSG:25832
        
        attr_pkno = QgsField('Punktnummer', QVariant.String)
        attr_sp = QgsField('Standpunkt', QVariant.String)
        attr_datetime = QgsField('Recordtime', QVariant.DateTime)
        attr_ih = QgsField('ih', QVariant.Double)
        attr_th = QgsField('th', QVariant.Double)
        
        attr_messung_sd = QgsField('mess_sd', QVariant.Double)
        attr_messung_za = QgsField('mess_za', QVariant.Double)
        attr_messung_ha = QgsField('mess_ha', QVariant.Double)
        calc_hd = QgsField('calc_hd', QVariant.Double)
        calc_x = QgsField('calc_x', QVariant.Double)
        calc_y = QgsField('calc_y', QVariant.Double)
        calc_z = QgsField('calc_z', QVariant.Double)
        prismConst = QgsField('prism_const', QVariant.Double)
        
        qml_file = f'{self.plugin_dir}/messung.qml'
        self.mlayer.loadNamedStyle(qml_file)
        
        self.mlayer.dataProvider().addAttributes([attr_pkno,attr_sp,attr_datetime,attr_ih,attr_th,attr_messung_sd,attr_messung_za,attr_messung_ha, calc_hd, calc_x, calc_y, calc_z,prismConst])
        
        self.mlayer.updateFields() 
        
    
    def addSpTempLayer(self, name):
        
        self.splayer = QgsVectorLayer("Point?crs="+self.crsName, name, "memory")
        
        attr_pkno = QgsField('Punktnummer', QVariant.String)
        attr_apno = QgsField('Anschluss', QVariant.String)
        attr_datetime = QgsField('Recordtime', QVariant.DateTime)
        attr_ih = QgsField('ih', QVariant.Double)
        x = QgsField('x', QVariant.Double)
        y = QgsField('y', QVariant.Double)
        z = QgsField('z', QVariant.Double)
        
        qml_file = f'{self.plugin_dir}/sp.qml'
        self.splayer.loadNamedStyle(qml_file)
        self.splayer.dataProvider().addAttributes([attr_pkno,attr_datetime,attr_ih,x,y,z,attr_apno])
        self.splayer.updateFields()  
        
    def addApTempLayer(self, name):
        
        self.aplayer = QgsVectorLayer("Point?crs="+self.crsName, name, "memory")
        
        attr_pkno = QgsField('Punktnummer', QVariant.String)
        attr_datetime = QgsField('Recordtime', QVariant.DateTime)
        x = QgsField('x', QVariant.Double)
        y = QgsField('y', QVariant.Double)
        
        qml_file = f'{self.plugin_dir}/ap.qml'
        self.aplayer.loadNamedStyle(qml_file)
        self.aplayer.dataProvider().addAttributes([attr_pkno,attr_datetime,x,y])
        self.aplayer.updateFields()  
        
    def sendPeriodicAngleMeasureCommand(self):
        while not self.serialPeriodicEvent.is_set() and self.serial.is_open:
            command = bytes([0x13])
            self.serial.write(command)
            time.sleep(1)

    def initLayers(self):
        """Legt alle nötigen Layer an, ohne eine serielle Verbindung zu benötigen."""
        crs = self.dockwidget.mQgsProjectionSelectionWidget.crs()
        self.crsName = crs.authid() if crs.isValid() else "EPSG:25832"

        self.addTempLayer(f"Messungen-{datetime.now().strftime('%d%m%y-%H%M')}")
        self.addSpTempLayer(f"Station-{datetime.now().strftime('%d%m%y-%H%M')}")
        self.addApTempLayer(f"APs-{datetime.now().strftime('%d%m%y-%H%M')}")

        QgsProject.instance().addMapLayer(self.mlayer)
        QgsProject.instance().addMapLayer(self.splayer)
        QgsProject.instance().addMapLayer(self.aplayer)

        self._apply_connection_state(connected=False, initialized=True)

        self.iface.messageBar().pushSuccess(
            "Initialisiert", f"Layer angelegt (offline)  |  CRS: {self.crsName}")

    def _protokoll_add(self, typ: str, text: str):
        """Fügt einen Eintrag zum laufenden Protokoll hinzu."""
        self._protokoll.append({
            'time': datetime.now(),
            'typ': typ,
            'text': text,
        })

    def _autosave_protokoll(self):
        """Speichert das Protokoll automatisch in temp_protocols/ im Plugin-Verzeichnis."""
        if not self._protokoll_tempfile:
            return
        try:
            self._write_protokoll_to_file(self._protokoll_tempfile)
        except Exception as e:
            print(f'[Protokoll Autosave] {e}')

    def _write_protokoll_to_file(self, filepath: str):
        """Schreibt das Protokoll in die angegebene Datei (intern genutzt)."""
        SEP  = '=' * 80
        SEP2 = '-' * 80

        def fmt(v, decimals=4):
            try:
                return f'{float(v):.{decimals}f}'
            except (TypeError, ValueError):
                return str(v) if v is not None else '—'

        meta = self._protokoll_meta
        lines = []
        lines.append(SEP)
        lines.append('  QGIS Sokkia Plugin  —  Messprotokoll')
        lines.append(SEP)
        lines.append(f"  Erstellt am  : {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        lines.append(f"  Sitzungsstart: {meta.get('start', datetime.now()).strftime('%d.%m.%Y %H:%M:%S')}")
        lines.append(f"  Gerät        : {meta.get('device', '?')}")
        lines.append(f"  Port / Baud  : {meta.get('port', '?')}  /  {meta.get('baudrate', '?')}")
        lines.append(f"  Koordinaten  : {meta.get('crs', '?')}")
        lines.append(SEP)
        lines.append('')

        n_station = sum(1 for e in self._protokoll if e['typ'] == 'STATIONIERUNG')
        n_messung = sum(1 for e in self._protokoll if e['typ'] == 'MESSUNG')
        lines.append(f'  Stationierungen: {n_station}    Messungen: {n_messung}')
        lines.append('')

        current_station = None
        station_nr = 0
        messung_nr = 0

        for entry in self._protokoll:
            t = entry['time'].strftime('%H:%M:%S.%f')[:-3]
            typ = entry['typ']
            if typ == 'VERBINDUNG':
                lines.append(f'[{t}] VERBINDUNG')
                lines.append(f"  {entry['text']}")
                lines.append('')
            elif typ == 'STATIONIERUNG':
                station_nr += 1
                d = entry.get('data', {})
                lines.append(SEP2)
                lines.append(f'[{t}] STATIONIERUNG #{station_nr}')
                lines.append(f"  Standpunkt-Nr.    : {d.get('sp_id','?')}")
                lines.append(f"  Rechts (X)        : {fmt(d.get('x'))} m")
                lines.append(f"  Hoch   (Y)        : {fmt(d.get('y'))} m")
                lines.append(f"  Höhe   (H)        : {fmt(d.get('h'))} m")
                lines.append(f"  Instrumentenhöhe  : {fmt(d.get('ih'))} m")
                lines.append(f"  Orientierung z0   : {fmt(d.get('orientation_gon'))} gon")
                res = d.get('resection')
                if res:
                    pts = res.get('points', [])
                    std = res.get('std_dev', [0, 0, 0])
                    lines.append(f"  [Freie Stationierung  —  {len(pts)} Anschlusspunkte]")
                    lines.append(f"  Genauigkeit:  "
                                 f"σX: ±{std[0]:.4f} m   "
                                 f"σY: ±{std[1]:.4f} m   "
                                 f"σZ: ±{std[2]:.4f} m   "
                                 f"σ₀: {res.get('sigma0', 0):.4f} m   "
                                 f"f: {res.get('dof', 0)}")
                    lines.append(f"  {'Punkt':<14}  {'X-AP [m]':>14}  {'Y-AP [m]':>14}  {'Z-AP [m]':>9}  "
                                 f"{'Hz [gon]':>10}  {'t [gon]':>10}  {'vHz [mgon]':>10}  "
                                 f"{'SD [m]':>9}  {'SDber [m]':>9}  {'vSD [mm]':>8}  "
                                 f"{'ZA [gon]':>10}  {'ZAber [gon]':>11}  {'vZA [mgon]':>10}")
                    lines.append(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*9}  "
                                 f"{'-'*10}  {'-'*10}  {'-'*10}  "
                                 f"{'-'*9}  {'-'*9}  {'-'*8}  "
                                 f"{'-'*10}  {'-'*11}  {'-'*10}")
                    for pt in pts:
                        lines.append(
                            f"  {pt['name']:<14}  {pt['ap_x']:>14.4f}  {pt['ap_y']:>14.4f}  {pt['ap_z']:>9.4f}  "
                            f"{pt['hz_gon']:>10.4f}  {pt['t_gon']:>10.4f}  {pt['hz_res_mgon']:>+10.1f}  "
                            f"{pt['sd_m']:>9.4f}  {pt['sd_calc']:>9.4f}  {pt['sd_res_mm']:>+8.2f}  "
                            f"{pt['za_gon']:>10.4f}  {pt['za_calc']:>11.4f}  {pt['za_res_mgon']:>+10.1f}"
                        )
                else:
                    lines.append(f"  Anschlussrichtung : {d.get('ap_id','—')}  "
                                 f"X={fmt(d.get('ap_x'))}  Y={fmt(d.get('ap_y'))}")
                lines.append('')
                current_station = d
                messung_nr = 0
            elif typ == 'MESSUNG':
                messung_nr += 1
                d = entry.get('data', {})
                lines.append(f"[{t}] Messung #{messung_nr}  —  Pkt: {d.get('id','?')}")
                if current_station:
                    lines.append(f"  Standpunkt        : {current_station.get('sp_id','?')}")
                lines.append(f"  Hz (Messwert)     : {fmt(d.get('ha_raw'))} gon")
                lines.append(f"  Hz (orientiert)   : {fmt(d.get('ha_oriented'))} gon")
                lines.append(f"  ZA (Zenitwinkel)  : {fmt(d.get('za'))} gon")
                lines.append(f"  SD (Schrägdistanz): {fmt(d.get('sd'))} m")
                lines.append(f"  HD (Horizontaldist): {fmt(d.get('hd'))} m")
                lines.append(f"  Zielh. (th)       : {fmt(d.get('th'))} m")
                lines.append(f"  Prismenkonstante  : {fmt(d.get('prism_const'),1)} mm")
                lines.append(f"  Ber. X (Rechts)   : {fmt(d.get('x'))} m")
                lines.append(f"  Ber. Y (Hoch)     : {fmt(d.get('y'))} m")
                lines.append(f"  Ber. Z (Höhe)     : {fmt(d.get('z'))} m")
                lines.append('')
            elif typ == 'TRENNUNG':
                lines.append(SEP2)
                lines.append(f'[{t}] TRENNUNG')
                lines.append('')
            else:
                lines.append(f"[{t}] {typ}: {entry['text']}")
                lines.append('')

        lines.append(SEP)
        lines.append(f'  Ende des Protokolls  —  {n_station} Stationierung(en)  /  {n_messung} Messung(en)')
        lines.append(SEP)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def export_protokoll(self):
        """Schreibt das Messprotokoll der aktuellen Sitzung in eine TXT-Datei."""
        if not self._protokoll and not self._protokoll_meta:
            self.iface.messageBar().pushWarning("Protokoll", "Keine Protokolldaten vorhanden.")
            return

        # Speicherort abfragen – temp-Datei als Vorschlag
        default_name = os.path.basename(self._protokoll_tempfile) if self._protokoll_tempfile \
            else "Protokoll_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
        default_dir = QSettings().value('qgis_sokkia/last_protokoll_dir', os.path.expanduser('~'))
        filepath, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(),
            "Protokoll speichern",
            os.path.join(default_dir, default_name),
            "Textdateien (*.txt)"
        )
        if not filepath:
            return
        QSettings().setValue('qgis_sokkia/last_protokoll_dir', os.path.dirname(filepath))

        n_station = sum(1 for e in self._protokoll if e['typ'] == 'STATIONIERUNG')
        n_messung = sum(1 for e in self._protokoll if e['typ'] == 'MESSUNG')
        try:
            self._write_protokoll_to_file(filepath)
            self.iface.messageBar().pushSuccess(
                "Protokoll", f"Gespeichert: {filepath}  ({n_station} Stat. / {n_messung} Mess.)")
        except Exception as e:
            self.iface.messageBar().pushCritical("Protokoll-Fehler", str(e))

    def _close_serial(self):
        """Gibt die serielle Schnittstelle sicher frei (Thread-sicher, idempotent)."""
        self.serialStopEvent.set()
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
            except Exception:
                pass

    def disconnectFromSerial(self):
        self._protokoll_add('TRENNUNG', 'Verbindung getrennt')
        self._close_serial()
        self._clear_direction_rubber_band()
        #Enable/disable buttons
        self._apply_connection_state(connected=False, initialized=False)
        self.iface.messageBar().pushInfo("Verbindung", "Getrennt.")

    def readSerial(self):
        """Liest serielle Daten im Hintergrund-Thread und legt sie thread-sicher in die Queue."""
        def parse_and_format_string(raw):
            decoded = raw.decode('utf-8', errors='replace').strip().replace("\x15", "")
            numbers = decoded.split()
            return [n[:3] + '.' + n[3:] for n in numbers]

        import errno as _errno
        while not self.serialStopEvent.is_set() and self.serial.is_open:
            try:
                # Im Transfermodus überlässt der readSerial die Daten dem Transfer-Dialog
                if self._transfer_mode:
                    time.sleep(0.1)
                    continue
                data = self.serial.readline()
                if not data:
                    continue
                text = data.decode('utf-8', errors='replace')
                if not text.startswith('\x06'):
                    parsed = parse_and_format_string(data)
                    if len(parsed) >= 3:
                        sd = float(parsed[0])
                        za = float(parsed[1])
                        ha = float(parsed[2])
                        self._measure_queue.put({
                            'sd': sd, 'za': za, 'ha': ha,
                            'is_distance': sd > 0,
                        })
            except serial.SerialTimeoutException:
                # Normaler Timeout bei leerem Port – einfach weiterlesen
                continue
            except serial.SerialException as e:
                # Windows-Semaphor-Timeout (errno 121) – nicht-fataler Port-Hitch
                if hasattr(e, 'args') and len(e.args) >= 4 and e.args[3] == 121:
                    time.sleep(0.05)
                    continue
                # Echter Verbindungsfehler
                self._measure_queue.put({'error': str(e)})
                if not self.serial.is_open:
                    break
            except Exception as e:
                self._measure_queue.put({'error': str(e)})
                    
    def _process_measure_queue(self):
        """Verarbeitet Messdaten aus der seriellen Queue (läuft im Haupt-Thread via QTimer)."""
        try:
            while not self._measure_queue.empty():
                item = self._measure_queue.get_nowait()
                if 'error' in item:
                    err = item['error']
                    # Semaphor-Timeout (Windows Error 121) nicht im Log anzeigen
                    if '121' not in err:
                        print(f"[Seriell] {err}")
                    continue
                sd = item['sd']
                za = item['za']
                ha = item['ha']
                self.measureValues['ha'] = ha
                self.measureValues['za'] = za
                if item['is_distance']:
                    self.measureValues['sd'] = sd
                    self.addMPoint(sd, za, ha)
                if self.dockwidget:
                    self.dockwidget.lbl_ha.setText(f"HZ: {self.measureValues['ha']:.4f} gon")
                    self.dockwidget.lbl_za.setText(f"VZ: {self.measureValues['za']:.4f} gon")
                    self.dockwidget.lbl_sd.setText(f"SD: {self.measureValues['sd']:.4f} m")
                self._update_direction_rubber_band(ha)
        except Exception as e:
            print(f"[Queue] {e}")

    def _update_direction_rubber_band(self, ha_raw: float):
        """Zeichnet eine 100m-Linie in Richtung des aktuellen Hz-Werts vom Standpunkt aus."""
        try:
            if not self.canvas:
                print("[RubberBand] kein canvas")
                return
            sp_x = self.sp.get('RECHTS', 0)
            sp_y = self.sp.get('HOCH', 0)
            # Hz-Rohwert + Orientierung -> Nordrichtung in Gon
            ha_oriented = (ha_raw + self.orientation * 200.0 / math.pi) % 400
            ha_rad = ha_oriented * math.pi / 200.0
            length = 100.0
            end_x = sp_x + length * math.sin(ha_rad)
            end_y = sp_y + length * math.cos(ha_rad)

            # Koordinaten in Projekt-CRS transformieren
            try:
                src_crs = QgsCoordinateReferenceSystem(self.crsName)
                dst_crs = QgsProject.instance().crs()
                if src_crs.isValid() and dst_crs.isValid() and src_crs != dst_crs:
                    transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
                    p1 = transform.transform(QgsPointXY(sp_x, sp_y))
                    p2 = transform.transform(QgsPointXY(end_x, end_y))
                else:
                    p1 = QgsPointXY(sp_x, sp_y)
                    p2 = QgsPointXY(end_x, end_y)
            except Exception as te:
                print(f"[RubberBand] Transform-Fehler: {te}")
                p1 = QgsPointXY(sp_x, sp_y)
                p2 = QgsPointXY(end_x, end_y)

            if self._direction_rubber_band is None:
                self._direction_rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
                self._direction_rubber_band.setColor(QColor(255, 80, 0, 220))
                self._direction_rubber_band.setWidth(3)
                self._direction_rubber_band.setZValue(100)

            self._direction_rubber_band.reset(QgsWkbTypes.LineGeometry)
            self._direction_rubber_band.addPoint(p1, False)
            self._direction_rubber_band.addPoint(p2, True)
            print(f"[RubberBand] {p1.x():.2f},{p1.y():.2f} -> {p2.x():.2f},{p2.y():.2f}")
        except Exception as e:
            print(f"[RubberBand] Fehler: {e}")

    def _clear_direction_rubber_band(self):
        """Entfernt die Richtungslinie von der Karte."""
        if self._direction_rubber_band is not None:
            self._direction_rubber_band.reset(QgsWkbTypes.LineGeometry)
            self._direction_rubber_band = None

    def _center_map(self, x: float, y: float):
        """Zentriert die Karte auf den Punkt (x, y) in self.crsName."""
        try:
            pt = QgsPointXY(x, y)
            src_crs = QgsCoordinateReferenceSystem(self.crsName)
            dst_crs = QgsProject.instance().crs()
            if src_crs.isValid() and dst_crs.isValid() and src_crs != dst_crs:
                transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
                pt = transform.transform(pt)
            self.canvas.setCenter(pt)
            self.canvas.refresh()
        except Exception as e:
            print(f"[CenterMap] {e}")

    def addMPoint(self, sd, za, ha):

        def increment_last_segment(s):
            import re
            # Suche nach dem letzten Vorkommen von '.', '-' oder '_'
            match = re.search(r'[\.\-_]([^.\-_]+)$', s)
            if not match:
                return s  # Kein passendes Zeichen gefunden, gib den Originalstring zurück
            teil = match.group(1)
            if teil.isdigit():
                # Inkrementiere die Zahl um 1
                inkrementiert = str(int(teil) + 1)
                # Ersetze den alten Teil durch den neuen im Originalstring
                return s[:-len(teil)] + inkrementiert
            else:
                # Wenn kein Zahl, gib den String unverändert zurück
                return s

        if self.mlayer is None:
            return

        try:
            hd = sd * math.sin(za*math.pi/200)
            
            #orientierung
            ha_raw_proto = ha % 400  # Rohwert in Gon für Protokoll
            ha = (ha + self.orientation * 200.0 / math.pi) % 400
            
            th = float(self._zielpunkt_dlg.input_th.text())
            
            z = self.sp['H'] + self.sp['ih'] + sd * math.cos(za*math.pi/200) - th 
            
            x = self.sp['RECHTS'] + hd  * math.sin(ha*math.pi/200)
            y = self.sp['HOCH'] + hd * math.cos(ha*math.pi/200)
            
            print(f"Neuer Punkt X:{x} Y:{y} Z:{z}")
            
            point = QgsPointXY(x, y) 
            
            
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(point))
            #feature.setAttributes([1]) # ID auf 1 setzen

            #get properties from UI
            prism_constant = float(self._zielpunkt_dlg.input_prismConstant.text())
            targetid = self._zielpunkt_dlg.input_targetid.text()

            #add values to ui
            self.dockwidget.lbl_calc_x.setText('X:' + str(f"{x:.4f}"))
            self.dockwidget.lbl_calc_y.setText('Y:' + str(f"{y:.4f}"))
            self.dockwidget.lbl_calc_z.setText('Z:' + str(f"{z:.4f}"))

            self._calc_line_distance(x, y)

            # Speichern: automatisch oder mit Abfrage
            auto_save = self._zielpunkt_dlg.cb_auto_save.isChecked()
            if auto_save:
                save_id = targetid
                do_save = True
            else:
                dlg = SavePointDialog(
                    targetid, x, y, z, th,
                    sd, za, self.sp['H'], self.sp['ih'],
                    parent=self.iface.mainWindow()
                )
                if dlg.exec_() == QDialog.Accepted:
                    save_id, th = dlg.get_values()
                    # Z neu berechnen falls Zielhöhe geändert
                    z = self.sp['H'] + self.sp['ih'] + sd * math.cos(za * math.pi / 200) - th
                    do_save = save_id.strip() != ''
                else:
                    do_save = False

            if do_save:
                feature.setAttributes([save_id, self.sp['ID'], QDateTime.currentDateTime(),
                                        self.sp['ih'], th, sd, za, ha, hd, x, y, z, prism_constant])
                self.mlayer.dataProvider().addFeature(feature)
                print('Punkt gespeichert:', save_id)
                self.mlayer.updateExtents()
                self.mlayer.triggerRepaint()
                self._center_map(x, y)
                # Protokoll-Eintrag
                self._protokoll_add('MESSUNG', '', )
                self._protokoll[-1]['data'] = {
                    'id': save_id, 'ha_raw': ha_raw_proto, 'ha_oriented': ha,
                    'za': za, 'sd': sd, 'hd': hd, 'th': th,
                    'prism_const': prism_constant, 'x': x, 'y': y, 'z': z,
                }
                self._autosave_protokoll()
                # Autoinkrement nur wenn Punkt wirklich gespeichert
                if self._zielpunkt_dlg.cb_autoincerement.isChecked():
                    newid = increment_last_segment(save_id)
                    self._zielpunkt_dlg.input_targetid.setText(newid)
        except Exception as e:
            print(e)
    
    def addStation(self):
        if self.splayer is None:
            self.iface.messageBar().pushWarning("Standpunkt", "Kein Stations-Layer vorhanden – bitte zuerst verbinden.")
            return
        point = QgsPointXY(self.sp["RECHTS"], self.sp["HOCH"])

        feature = QgsFeature(self.splayer.fields())
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        feature.setAttributes([self.sp['ID'], QDateTime.currentDateTime(), self.sp['ih'], self.sp["RECHTS"], self.sp["HOCH"], self.sp["H"], self.ap['ID']])

        self.splayer.dataProvider().addFeature(feature)
        print('Station gespeichert')

        self.splayer.updateExtents()
        self.splayer.triggerRepaint()
        
    def addAp(self):
        if self.aplayer is None:
            return
        point = QgsPointXY(self.ap["RECHTS"], self.ap["HOCH"])
            
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        

        feature.setAttributes([self.ap['ID'],QDateTime.currentDateTime(),self.ap["RECHTS"], self.ap["HOCH"]])


        self.aplayer.dataProvider().addFeature(feature)
        print('Station gespeichert')
        
        self.aplayer.updateExtents()
        self.aplayer.triggerRepaint() #re-draw layer
    
    def _connect_line_layer_signals(self, layer):
        """Verbindet selectionChanged des aktuellen Linienlayers zur Live-Anzeige."""
        # Vorherigen Layer trennen
        if hasattr(self, '_line_layer_connected') and self._line_layer_connected is not None:
            try:
                self._line_layer_connected.selectionChanged.disconnect(self._update_line_selection_label)
            except Exception:
                pass
        self._line_layer_connected = layer
        if layer is not None:
            layer.selectionChanged.connect(self._update_line_selection_label)
        self._update_line_selection_label()

    def _update_line_selection_label(self, *args):
        """Aktualisiert das Status-Label für die Linienauswahl."""
        if not self.dockwidget:
            return
        layer = self.dockwidget.combo_line_layer.currentLayer()
        if layer is None:
            self.dockwidget.lbl_line_selection.setText("Kein Layer gewählt")
            return
        count = layer.selectedFeatureCount()
        if count > 0:
            names = []
            for f in layer.selectedFeatures():
                val = f.attribute(layer.displayField()) if layer.displayField() else None
                names.append(str(val) if val else f"ID {f.id()}")
            label = f"Selektiert ({count}): " + ", ".join(names[:3])
            if count > 3:
                label += " …"
            self.dockwidget.lbl_line_selection.setText(label)
            self.dockwidget.lbl_line_selection.setStyleSheet(
                "color:#1b5e20;font-style:normal;font-size:9px;font-weight:bold;")
        else:
            self.dockwidget.lbl_line_selection.setText(
                "Kein Objekt ausgewählt \u2013 nächste Linie wird verwendet")
            self.dockwidget.lbl_line_selection.setStyleSheet(
                "color:#999;font-style:italic;font-size:9px;")

    def _calc_line_distance(self, x, y):
        """Berechnet den orthogonalen Abstand vom Punkt (x,y) zum nächsten Liniensegment.
        Ist im Layer ein Objekt selektiert, wird nur dieses verwendet; sonst alle Objekte."""
        if not self.dockwidget or not self.dockwidget.groupBox_linedist.isChecked():
            return
        layer = self.dockwidget.combo_line_layer.currentLayer()
        if layer is None:
            self.dockwidget.lbl_line_distance.setText("Abstand: kein Layer")
            return

        selected = layer.selectedFeatures()
        features = selected if selected else list(layer.getFeatures())

        def segments_from_geom(geom):
            """Liefert alle (ax,ay,bx,by)-Segmente aus einer Liniengeometrie."""
            segs = []
            wkb_type = geom.wkbType()
            if QgsWkbTypes.isMultiType(wkb_type):
                lines = geom.asMultiPolyline()
            else:
                lines = [geom.asPolyline()]
            for line in lines:
                for i in range(len(line) - 1):
                    segs.append((line[i].x(), line[i].y(), line[i+1].x(), line[i+1].y()))
            return segs

        px, py = x, y
        min_dist = None
        for feat in features:
            geom = feat.geometry()
            if geom.isNull() or geom.isEmpty():
                continue
            for ax, ay, bx, by in segments_from_geom(geom):
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx*dx + dy*dy
                if seg_len_sq == 0:
                    d = math.hypot(px - ax, py - ay)
                else:
                    t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / seg_len_sq))
                    d = math.hypot(px - ax - t*dx, py - ay - t*dy)
                if min_dist is None or d < min_dist:
                    min_dist = d

        if min_dist is not None:
            self.dockwidget.lbl_line_distance.setText(f"Abstand: {min_dist:.3f} m")
        else:
            self.dockwidget.lbl_line_distance.setText("Abstand: keine Geometrie")

    def calc_orientation(self):
        try:
            ap_x = float(self._standort_dlg.input_ap_x.text())
            ap_y = float(self._standort_dlg.input_ap_y.text())
            ap_name = self._standort_dlg.input_ap.text().strip()
            sp_x = float(self._standort_dlg.input_sp_x.text())
            sp_y = float(self._standort_dlg.input_sp_y.text())
        except ValueError as e:
            self.iface.messageBar().pushWarning("Orientierung", f"Ungültige Koordinate: {e}")
            return

        if abs(ap_x - sp_x) < 1e-9 and abs(ap_y - sp_y) < 1e-9:
            self.iface.messageBar().pushWarning(
                "Orientierung",
                "Standpunkt und Anschlusspunkt sind identisch – Orientierung nicht berechenbar.")
            return

        o = math.atan2(ap_x - sp_x, ap_y - sp_y)
        if o < 0:
            o += 2 * math.pi
        self.orientation = o
        z0_gon = o * 200.0 / math.pi
        self._standort_dlg.input_orientation.setText(f"{z0_gon:.4f} gon")
        self.ap = {"ID": ap_name, "RECHTS": ap_x, "HOCH": ap_y}
        self.addAp()
        self.orientationArrow.addFeature(sp_x, sp_y, ap_x, ap_y)
        self.orientationArrow.addLayerToMapInstance()

    def set_orientation_zero(self):
        """Setzt die Orientierung auf 0 gon (z₀ = 0)."""
        self.orientation = 0.0
        self._standort_dlg.input_orientation.setText("0.0000 gon")
        self.iface.messageBar().pushInfo(
            "Orientierung", "Orientierung z\u2080 auf 0.0000 gon gesetzt.")

    def draw_line(self, theta):
        # Erstelle eine RubberBand-Instanz
        
        self.rubber_band.reset(QgsWkbTypes.LineGeometry)
        #self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        
        
        
        # Definiere die Punkte der Linie
        start_point = QgsPointXY(0, 0)
        
        dist = 100
        
        theta = theta * math.pi / 200       #gon --> rad
        
        
        x = dist * math.cos(theta)
        y = dist * math.sin(theta)

    
        end_point = QgsPointXY(x, y)

        # Füge die Punkte zur RubberBand hinzu
        self.rubber_band.addPoint(start_point, True)
        self.rubber_band.addPoint(end_point, True)

        # Setze die Farbe und Breite der Linie
        self.rubber_band.setColor(Qt.red)
        self.rubber_band.setWidth(1)
        
        
        # Aktualisiere die Karte
        self.canvas.refresh()
        #self.canvas.update()

    
    def selectCoordinatesFromMap(self):
        """Aktiviert ein Kartenwerkzeug zum Aufnehmen des Standpunkts per Mausklick."""
        def capture_coordinate(point):
            self._standort_dlg.input_sp_x.setText(f"{point.x():.4f}")
            self._standort_dlg.input_sp_y.setText(f"{point.y():.4f}")
            self.dockwidget.lbl_x.setText(f"RECHTS: {point.x():.4f}")
            self.dockwidget.lbl_y.setText(f"HOCH: {point.y():.4f}")
            self.canvas.unsetMapTool(self._sp_map_tool)
            self._standort_dlg.show()
            self._standort_dlg.raise_()
            self.iface.messageBar().pushInfo(
                "Standpunkt", f"Koordinaten übernommen: X={point.x():.4f}  Y={point.y():.4f}")

        self._sp_map_tool = SnapPointTool(self.canvas)
        self._sp_map_tool.pointPicked.connect(capture_coordinate)
        self.canvas.setMapTool(self._sp_map_tool)
        self.iface.messageBar().pushInfo("Standpunkt", "Klicken Sie in die Karte, um Koordinaten zu übernehmen.")

    def selectApFromMap(self):
        """Aktiviert ein Kartenwerkzeug zum Aufnehmen der Anschlussrichtung per Mausklick."""
        def capture_ap(point):
            self._standort_dlg.input_ap_x.setText(f"{point.x():.4f}")
            self._standort_dlg.input_ap_y.setText(f"{point.y():.4f}")
            self.canvas.unsetMapTool(self._ap_map_tool)
            self._standort_dlg.show()
            self._standort_dlg.raise_()
            self.iface.messageBar().pushInfo(
                "Anschlussrichtung", f"Koordinaten übernommen: X={point.x():.4f}  Y={point.y():.4f}")

        self._ap_map_tool = SnapPointTool(self.canvas)
        self._ap_map_tool.pointPicked.connect(capture_ap)
        self.canvas.setMapTool(self._ap_map_tool)
        self.iface.messageBar().pushInfo("Anschlussrichtung", "Klicken Sie in die Karte, um den Anschlusspunkt zu übernehmen.")
   
    def switchLaser(self):
        
        laser_command = b'*/PF 2,1\r\n'
        self.serial.write(laser_command)
        
        # get laser state
        state = self.laserState
        
        #toggle lader
        if state:
            laser_command = b'*GLOFF\r\n'
            self.laserState= False
            #button text
            self.dockwidget.btn_laser.setText('Laser einschalten')
            print('Laser off')
        else:
            laser_command = b'*GLON\r\n'
            self.laserState = True
            self.dockwidget.btn_laser.setText('Laser ausschalten')
            print('Laser on')
        
        self.serial.write(laser_command)
        
    def selectTarget(self):
        
        vprism = self._zielpunkt_dlg.radio_prism
        vreflex = self._zielpunkt_dlg.radio_reflex
        vreflectorless = self._zielpunkt_dlg.radio_reflectorless
        
        if vprism.isChecked():
            self.target = 0
            self.targetPrismConstant = -35   #sokkia default
            self._zielpunkt_dlg.input_prismConstant.setText(str(self.targetPrismConstant))
        elif vreflex.isChecked():
            self.target =1 
            self.targetPrismConstant = 0
            self._zielpunkt_dlg.input_prismConstant.setText(str(self.targetPrismConstant))
        elif vreflectorless.isChecked():
            self.target = 2
            self.targetPrismConstant = 0
            self._zielpunkt_dlg.input_prismConstant.setText(str(self.targetPrismConstant))
        else:
            self.target = 2    
            
        
         
    
    def setTarget(self):
        
        targetType = 'None'
        
        command = None
        if self.target == 0:
            command = b'/C 0\r\n'   #prism
            targetType = 'Prisma'
            
        elif self.target == 1:
            command = b'/C 1\r\n'   #sheet
            targetType = 'Reflexfolie'
            
        else:
            command = b'/C 2\r\n'   #reflectorless
            targetType = 'Reflektorlos'
            
            
        self.targetPrismConstant = int(self._zielpunkt_dlg.input_prismConstant.text())
            
        pc1 = b'/B 0,0,0,'
        pc2 = b',1,0,0,0,0,0,0,0\r\n'
        command2 = pc1 + str(self.targetPrismConstant).encode('utf-8') + pc2
        
        print(command2)
        
        self.serial.write(command)  
        self.serial.write(command2) 
        
        status_text = f"Zieltyp: {targetType}  |  th: {float(self._zielpunkt_dlg.input_th.text()):.3f} m  |  PK: {self.targetPrismConstant}"
        self.dockwidget.lbl_target.setText(status_text)
        self._zielpunkt_dlg.lbl_target_dialog.setText(status_text)
        self._zielpunkt_dlg.hide()
        
        
        
        
    def mesaure(self):
        print('Streckenmessung')
        command = bytes([0x11])
        self.serial.write(command)
    
    def mesaure_angle(self):
        print('Winkelmessung')
        command = bytes([0x13])
        self.serial.write(command)   
    
    def mesaure_stop(self):
        print('Messung stoppen')
        command = bytes([0x12])
        self.serial.write(command) 
        
    def setSp(self):
        try:
            sp_id = self._standort_dlg.input_standpoint.text().strip() or "SP"
            x = float(self._standort_dlg.input_sp_x.text())
            y = float(self._standort_dlg.input_sp_y.text())
            z = float(self._standort_dlg.input_sp_z.text())
            ih = float(self._standort_dlg.input_ih.text())
        except ValueError as e:
            self.iface.messageBar().pushWarning("Standpunkt", f"Ungültige Eingabe: {e}")
            return

        self.sp = {"ID": sp_id, "RECHTS": x, "HOCH": y, "H": z, "ih": ih}
        if self._standort_dlg.groupBox_8.isChecked():
            try:
                self.calc_orientation()
            except Exception as e:
                print(f"[Orientierung] {e}")
        self.addStation()
        status_text = f"ID: {sp_id}  |  X: {x:.4f}  Y: {y:.4f}  H: {z:.4f}  ih: {ih:.4f}"
        self.dockwidget.lbl_sp.setText(status_text)
        self.dockwidget.lbl_x.setText(f"X: {x:.4f}")
        self.dockwidget.lbl_y.setText(f"Y: {y:.4f}")
        self.dockwidget.lbl_z.setText(f"H: {z:.4f}")
        self._standort_dlg.lbl_sp_dialog.setText(status_text)
        self.iface.messageBar().pushSuccess(
            "Standpunkt", f"Standpunkt '{sp_id}' gesetzt und in Layer gespeichert.")
        self._center_map(x, y)
        # Protokoll-Eintrag
        ap_id = self._standort_dlg.input_ap.text() if self._standort_dlg.groupBox_8.isChecked() else ''
        ap_x = self._standort_dlg.input_ap_x.text() if self._standort_dlg.groupBox_8.isChecked() else ''
        ap_y = self._standort_dlg.input_ap_y.text() if self._standort_dlg.groupBox_8.isChecked() else ''
        self._protokoll_add('STATIONIERUNG', '')
        self._protokoll[-1]['data'] = {
            'sp_id': sp_id, 'x': x, 'y': y, 'h': z, 'ih': ih,
            'orientation_gon': self.orientation * 200.0 / math.pi,
            'ap_id': ap_id, 'ap_x': ap_x, 'ap_y': ap_y,
        }
        self._autosave_protokoll()
    
    def open_standort_dialog(self):
        """Standort-Dialog anzeigen."""
        self._standort_dlg.show()
        self._standort_dlg.raise_()
        self._standort_dlg.activateWindow()

    def open_zielpunkt_dialog(self):
        """Zielpunkt-Dialog anzeigen."""
        self._zielpunkt_dlg.show()
        self._zielpunkt_dlg.raise_()
        self._zielpunkt_dlg.activateWindow()

    def open_fernsteuerung_dialog(self):
        """Fernsteuerungs-Dialog anzeigen."""
        self._fernsteuerung_dlg.show()
        self._fernsteuerung_dlg.raise_()
        self._fernsteuerung_dlg.activateWindow()

    def open_absteckung_dialog(self):
        """Absteckungs-Dialog anzeigen."""
        self._absteckung_dlg.show()
        self._absteckung_dlg.raise_()
        self._absteckung_dlg.activateWindow()

    def open_resection_dialog(self):
        """Öffnet den Dialog für die Freie Stationierung."""
        dlg = ResectionDialog(
            self.iface,
            self.mlayer,
            parent=self.iface.mainWindow()
        )
        dlg.result_accepted.connect(self._apply_resection_result)
        dlg.exec_()

    def open_transfer_dialog(self):
        """Öffnet den Koordinaten-Transfer-Dialog (Upload/Download)."""
        dlg = TransferDialog(
            self.iface,
            serial_connection=self.serial,
            parent=self.iface.mainWindow(),
            transfer_mode_setter=self._set_transfer_mode,
        )
        dlg.exec_()

    def _set_transfer_mode(self, active):
        """Aktiviert/deaktiviert den Transfermodus (pausiert readSerial)."""
        self._transfer_mode = active

    def _apply_resection_result(self, x: float, y: float, z: float, z0_rad: float, resection_details: dict = None):
        """
        Übernimmt das Ergebnis des Rückwärtsschnitts in den Standpunkt
        und die Orientierung des Plugins.
        """
        sp_id = self._standort_dlg.input_standpoint.text() or "SP"
        ih = float(self._standort_dlg.input_ih.text() or 0)
        self.sp = {"ID": sp_id, "RECHTS": x, "HOCH": y, "H": z, "ih": ih}

        self.orientation = z0_rad
        z0_gon = z0_rad * 200.0 / math.pi

        # UI-Felder aktualisieren
        self._standort_dlg.input_sp_x.setText(f"{x:.4f}")
        self._standort_dlg.input_sp_y.setText(f"{y:.4f}")
        self._standort_dlg.input_sp_z.setText(f"{z:.4f}")
        self._standort_dlg.input_orientation.setText(f"{z0_gon:.4f}")
        self.dockwidget.lbl_x.setText(f"RECHTS: {x:.4f}")
        self.dockwidget.lbl_y.setText(f"HOCH: {y:.4f}")
        self.dockwidget.lbl_z.setText(f"HÖHE: {z:.4f}")
        status_text = (
            f"ID: {sp_id}  X: {x:.4f}  Y: {y:.4f}  Z: {z:.4f}  "
            f"z\u2080: {z0_gon:.4f} gon  [Freie Stationierung]"
        )
        self.dockwidget.lbl_sp.setText(status_text)
        self._standort_dlg.lbl_sp_dialog.setText(status_text)

        # Orientierungspfeil setzen (100 m in z₀-Richtung)
        end_x = x + 100.0 * math.sin(z0_rad)
        end_y = y + 100.0 * math.cos(z0_rad)
        self.orientationArrow.addFeature(x, y, end_x, end_y)
        self.orientationArrow.addLayerToMapInstance()

        # Standpunkt in Layer speichern
        self.addStation()

        # Protokoll-Eintrag
        self._protokoll_add('STATIONIERUNG', '')
        self._protokoll[-1]['data'] = {
            'sp_id': sp_id, 'x': x, 'y': y, 'h': z, 'ih': ih,
            'orientation_gon': z0_gon,
            'ap_id': '', 'ap_x': '', 'ap_y': '',
            'resection': resection_details,
        }
        self._autosave_protokoll()

        self.iface.messageBar().pushSuccess(
            "Freie Stationierung",
            f"Standpunkt gesetzt \u2192 X={x:.4f} m, Y={y:.4f} m, "
            f"Z={z:.4f} m, z\u2080={z0_gon:.4f} gon"
        )

    def control(self, direction, step):
        
        print("Control totalstation in " + direction + ' direction')
        
        stepsize = float(self._fernsteuerung_dlg.input_control_step.text())
        
        ha = self.measureValues["ha"]
        za = self.measureValues["za"]
        
        if direction == 'h':
            ha = (ha + step * stepsize) % 400
            print(ha)
            
        if direction == 'v':
            za = (za + step * stepsize) % 400
            print(za)
            
        
        ha_string = f"{ha:.4f}".replace('.', '')
        
        if len(ha_string) < 7:
            ha_string = '0' + ha_string
        za_string = f"{za:.4f}".replace('.', '')
        
        if len(za_string) < 7:                  #führende Null hinzufügen bei Zahlen < 100
            za_string = '0' + za_string
        
        
        #generate new value and send command, then request angle update
        command = f"*DHA{ha_string}VA{za_string}\r\n".encode('utf-8')
        self.serial.write(command)
        # Sofort optimistisch aktualisieren, damit schnelle Folge-Klicks
        # auf den bereits gesendeten Winkel aufaddieren
        self.measureValues["ha"] = ha
        self.measureValues["za"] = za
        # Winkelmessung verzögert anfordern – Gerät muss erst ankommen
        QTimer.singleShot(800, self.mesaure_angle)
        
        
           
    
    #--------------------------------------------------------------------------

    



    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive:
            self.pluginIsActive = True

            #print "** STARTING QGISSokkia"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = QGISSokkiaDockWidget()

            # Detaildialoge einmalig erstellen
            if self._standort_dlg is None:
                self._standort_dlg = StandortDialog(parent=self.iface.mainWindow())
            if self._zielpunkt_dlg is None:
                self._zielpunkt_dlg = ZielpunktDialog(parent=self.iface.mainWindow())
            if self._fernsteuerung_dlg is None:
                self._fernsteuerung_dlg = FernsteuerungDialog(parent=self.iface.mainWindow())

            if self._absteckung_dlg is None:
                self._absteckung_dlg = AbsteckungDialog(self, parent=self.iface.mainWindow())

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)

            # Serielle Ports befüllen und gespeicherten Port wiederherstellen
            self._refresh_serial_ports()
            self.dockwidget.btn_refresh_ports.clicked.connect(self._refresh_serial_ports)

            # CRS-Widget mit Projekt-CRS vorbelegen
            project_crs = QgsProject.instance().crs()
            if project_crs.isValid():
                self.dockwidget.mQgsProjectionSelectionWidget.setCrs(project_crs)
            else:
                self.dockwidget.mQgsProjectionSelectionWidget.setCrs(
                    QgsCoordinateReferenceSystem("EPSG:25832"))

            #connect 'connect' btn
            self.dockwidget.btn_connect.clicked.connect(self.connectToSerial)
            
            #connect 'disconnect' btn
            self.dockwidget.btn_disconnect.clicked.connect(self.disconnectFromSerial)

            #connect 'initialize without device' btn
            self.dockwidget.btn_init.clicked.connect(self.initLayers)

            # Hauptdock: "Standort …" Button öffnet den Standort-Dialog
            self.dockwidget.btn_setSp.clicked.connect(self.open_standort_dialog)

            # Hauptdock: Ziel- und Fernsteuerungs-Dialoge öffnen
            self.dockwidget.btn_open_zielpunkt.clicked.connect(self.open_zielpunkt_dialog)
            self.dockwidget.btn_open_fernsteuerung.clicked.connect(self.open_fernsteuerung_dialog)
            self.dockwidget.btn_absteckung.clicked.connect(self.open_absteckung_dialog)

            # Standort-Dialog Verbindungen
            self._standort_dlg.btn_select_sp.clicked.connect(self.selectCoordinatesFromMap)
            self._standort_dlg.btn_select_ap.clicked.connect(self.selectApFromMap)
            self._standort_dlg.btn_setSp_confirm.clicked.connect(self.setSp)
            self._standort_dlg.btn_resection.clicked.connect(self.open_resection_dialog)
            self._standort_dlg.btn_zero_orientation.clicked.connect(self.set_orientation_zero)

            #default values
            self._standort_dlg.input_standpoint.setText(self.sp['ID'])

            # Zielpunkt-Dialog Verbindungen
            self._zielpunkt_dlg.radio_prism.clicked.connect(self.selectTarget)
            self._zielpunkt_dlg.radio_reflex.clicked.connect(self.selectTarget)
            self._zielpunkt_dlg.radio_reflectorless.clicked.connect(self.selectTarget)
            self._zielpunkt_dlg.btn_setTarget.clicked.connect(self.setTarget)
            self._zielpunkt_dlg.input_prismConstant.setText(str(self.targetPrismConstant))

            # Fernsteuerungs-Dialog Verbindungen
            self._fernsteuerung_dlg.btn_control_left.clicked.connect(lambda: self.control('h', -1))
            self._fernsteuerung_dlg.btn_control_right.clicked.connect(lambda: self.control('h', 1))
            self._fernsteuerung_dlg.btn_control_up.clicked.connect(lambda: self.control('v', -1))
            self._fernsteuerung_dlg.btn_control_down.clicked.connect(lambda: self.control('v', 1))

            #connect 'toggle Laser from Map' btn
            self.dockwidget.btn_laser.clicked.connect(self.switchLaser)

            #Mess buttons
            self.dockwidget.btn_measure.clicked.connect(self.mesaure)
            self.dockwidget.btn_measure_a.clicked.connect(self.mesaure_angle)
            self.dockwidget.btn_measure_stop.clicked.connect(self.mesaure_stop)

            #Koordinaten-Transfer
            self.dockwidget.btn_transfer.clicked.connect(self.open_transfer_dialog)
            self.dockwidget.btn_export_protokoll.clicked.connect(self.export_protokoll)
            # Linienabstand: nur Linienlayer anzeigen
            self.dockwidget.combo_line_layer.setFilters(QgsMapLayerProxyModel.LineLayer)
            self.dockwidget.combo_line_layer.layerChanged.connect(self._connect_line_layer_signals)
            self._connect_line_layer_signals(self.dockwidget.combo_line_layer.currentLayer())

            # Initiale Tooltips auf deaktivierten Buttons setzen
            self._apply_connection_state(connected=False, initialized=False)

            # QTimer für thread-sichere Queue-Verarbeitung (Messdaten aus seriellem Thread)
            if not hasattr(self, '_queue_timer'):
                self._queue_timer = QTimer()
                self._queue_timer.timeout.connect(self._process_measure_queue)
                self._queue_timer.start(100)  # alle 100 ms prüfen

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(Qt.TopDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
