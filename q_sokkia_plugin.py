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
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt, QVariant, QDateTime
from qgis.PyQt.QtGui import QIcon
import os

from datetime import datetime
from qgis.gui import QgsMapCanvas, QgsRubberBand, QgsMapToolEmitPoint
from qgis.core import QgsPointXY, QgsPoint, QgsWkbTypes, QgsVectorLayer, QgsFeature, QgsGeometry, QgsProject, QgsField
from qgis.PyQt.QtWidgets import QAction
import serial
import threading
import time
import math

from .q_sokkia_orientation_arrow import OrientationArrow

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
        
        
        #remove_all_rubber_bands(self.canvas)
        #self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        
        #Layer
        self.mlayer = None
        self.splayer = None
        self.aplayer = None



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

        icon_path = ':/plugins/q_sokkia_plugin/icon2.png'
        self.add_action(
            icon_path,
            text=self.tr(u'QSDR'),
            callback=self.run,
            parent=self.iface.mainWindow())

    #--------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        #print "** CLOSING QGISSokkia"

        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # remove this statement if dockwidget is to remain
        # for reuse if plugin is reopened
        # Commented next statement since it causes QGIS crashe
        # when closing the docked window:
        # self.dockwidget = None

        self.pluginIsActive = False


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        #print "** UNLOAD QGISSokkia"

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&QGIS Sokkia Plugin'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    #--------------------------------------------------------------------------

    def connectToSerial(self):
    
        try:    
            print("connect to serial");
            
            
            port = self.dockwidget.input_port.text()
            baudrate = self.dockwidget.input_baud.text()
            self.serial = serial.Serial(port, baudrate, timeout=1)
            
            
            crs = self.dockwidget.mQgsProjectionSelectionWidget.crs()
            self.crsName = crs.authid()
            print(f"Using CRS: {self.crsName}")
            
            
            time.sleep(1)
            
            if self.serial.is_open:
                print("Connection established successfully!")
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
                
                self.dockwidget.btn_connect.setEnabled(False)
                self.dockwidget.btn_disconnect.setEnabled(True)
                self.dockwidget.btn_laser.setEnabled(True)      #enable laser button
                self.dockwidget.btn_measure.setEnabled(True)
                self.dockwidget.btn_measure_a.setEnabled(True)
                self.dockwidget.btn_measure_stop.setEnabled(True)
                self.dockwidget.btn_setTarget.setEnabled(True)
                self.dockwidget.btn_setSp.setEnabled(True)
            
            
                # Layer zur Karte hinzufügen
                QgsProject.instance().addMapLayer(self.mlayer)
                QgsProject.instance().addMapLayer(self.splayer)
                QgsProject.instance().addMapLayer(self.aplayer)
            else:
                raise Exception('Serielle verbindung fehlgeschlagen')
        
        except serial.SerialException as e:
            print(e)
        except Exception as e:
            print(e)
    
    
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

    def disconnectFromSerial(self):
        print("disconnect from serial");
        self.serialStopEvent.set()
        self.serial.close()
        
        #Enable/disable buttons
        self.dockwidget.btn_connect.setEnabled(True)
        self.dockwidget.btn_disconnect.setEnabled(False)
        self.dockwidget.btn_laser.setEnabled(False)      #enable laser button
        self.dockwidget.btn_measure.setEnabled(False)
        self.dockwidget.btn_measure_a.setEnabled(False)
        self.dockwidget.btn_measure_stop.setEnabled(False)
        self.dockwidget.btn_setTarget.setEnabled(False)
        self.dockwidget.btn_setSp.setEnabled(False)
    
        
            
    def readSerial(self):
        print("[thread] reading from serial")
        
        def parse_and_format_string(input_string):
            # Decode the input string
            decoded_string = input_string.decode('utf-8').strip()
            decoded_string = decoded_string.replace("\x15", "")

            # Split the string into individual numbers
            numbers = decoded_string.split()

            # Insert a decimal point at the third position of each number
            formatted_numbers = [number[:3] + '.' + number[3:] for number in numbers]

            return formatted_numbers
        
        while not self.serialStopEvent.is_set() and self.serial.is_open:
            try:
                data = self.serial.readline()       #self.serial.read(128)
                
                if data:
                    print(data)
                    
                    if not data.decode('utf-8').startswith('\x06'):
                        parsedData = parse_and_format_string(data)
                        print(parsedData)
                        
                        sd = float(parsedData[0])
                        za = float(parsedData[1])
                        ha = float(parsedData[2])
                        
                        
                        if sd > 0:   #streckenmessung
                            print(f"SD: {sd} ZA: {za} HA: {ha}")
                            self.measureValues['ha'] = ha
                            self.measureValues['za'] = za
                            self.measureValues['sd'] = sd
                            self.addMPoint(sd,za,ha)
                        else:
                            print(f"SD: {sd} ZA: {za} HA: {ha}")   
                            self.measureValues['ha'] = ha
                            self.measureValues['za'] = za
                                
                        self.dockwidget.lbl_ha.setText('HZ:' + str(f"{self.measureValues['ha']:.4f}") + ' gon')
                        self.dockwidget.lbl_za.setText('VZ:' + str(f"{self.measureValues['za']:.4f}") + ' gon')
                        self.dockwidget.lbl_sd.setText('SD:' + str(f"{self.measureValues['sd']:.4f}") + ' m')
                    else:
                        print('OK')
    
            except Exception as e:
                print(e)
                    
    def addMPoint(self, sd,za,ha):
        
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

                
        try:
            hd = sd * math.sin(za*math.pi/200)
            
            #orientierung
            print(f"{ha *200 / math.pi}")
            ha = ha + self.orientation*200 / math.pi
            print(f"{ha *200 / math.pi}")
            
            th = float(self.dockwidget.input_th.text())
            
            z = self.sp['H'] + self.sp['ih'] + sd * math.cos(za*math.pi/200) - th 
            
            x = self.sp['RECHTS'] + hd  * math.sin(ha*math.pi/200)
            y = self.sp['HOCH'] + hd * math.cos(ha*math.pi/200)
            
            print(f"Neuer Punkt X:{x} Y:{y} Z:{z}")
            
            point = QgsPointXY(x, y) 
            
            
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(point))
            #feature.setAttributes([1]) # ID auf 1 setzen

            #get properties from UI
            
            prism_constant = float(self.dockwidget.input_prismConstant.text())
            targetid = self.dockwidget.input_targetid.text()
            
            feature.setAttributes([targetid,self.sp['ID'],QDateTime.currentDateTime(),self.sp['ih'],th,sd,za,ha,hd, x, y, z, prism_constant])

            #add values to ui
            self.dockwidget.lbl_calc_x.setText('X:' + str(f"{x:.4f}"))
            self.dockwidget.lbl_calc_y.setText('Y:' + str(f"{y:.4f}"))
            self.dockwidget.lbl_calc_z.setText('Z:' + str(f"{z:.4f}"))

            #increment target id
            newid = increment_last_segment(targetid)
            self.dockwidget.input_targetid.setText(newid)



            self.mlayer.dataProvider().addFeature(feature)
            print('Punkt gespeichert')
            
            self.mlayer.updateExtents()
            self.mlayer.triggerRepaint() #re-draw layer
        except Exception as e:
            print(e)
    
    def addStation(self):
        
        #self.sp = {"ID": "SP1", "RECHTS": 0, "HOCH":0, "H": 0, "ih": 0}
        
        point = QgsPointXY(self.sp["RECHTS"], self.sp["HOCH"]) 
            
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        

        feature.setAttributes([self.sp['ID'],QDateTime.currentDateTime(),self.sp['ih'],self.sp["RECHTS"], self.sp["HOCH"],self.sp["H"],self.ap['ID']])


        self.splayer.dataProvider().addFeature(feature)
        print('Station gespeichert')
        
        self.splayer.updateExtents()
        self.splayer.triggerRepaint() #re-draw layer
        
    def addAp(self):
        
        #self.ap = {"ID": "SP1", "RECHTS": 0, "HOCH":0, "H": 0, "ih": 0}
        
        point = QgsPointXY(self.ap["RECHTS"], self.ap["HOCH"]) 
            
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        

        feature.setAttributes([self.ap['ID'],QDateTime.currentDateTime(),self.ap["RECHTS"], self.ap["HOCH"]])


        self.aplayer.dataProvider().addFeature(feature)
        print('Station gespeichert')
        
        self.aplayer.updateExtents()
        self.aplayer.triggerRepaint() #re-draw layer
    
    def calc_orientation(self):
        
        #Get values from inpout fields
        
        #Anschluss (ap)
        ap_x = float(self.dockwidget.input_ap_x.text())
        ap_y = float(self.dockwidget.input_ap_y.text())
        ap_name = self.dockwidget.input_ap.text()
        
        #Standpunkt
        sp_x = float(self.dockwidget.input_sp_x.text())
        sp_y = float(self.dockwidget.input_sp_y.text())
        sp_name = self.dockwidget.input_standpoint.text()
        
        print(f"Berechne Orientierung von Standpunkt {sp_name} nach {ap_name}")
        
        o = math.atan2((ap_x -sp_x), (ap_y - sp_y))
        self.orientation = o;   #setze Orientierung
        
        print(f"Orientierung: {o*200/math.pi} gon")
        self.dockwidget.input_orientation.setText(str(self.orientation*200/math.pi) +'gon')
        
        self.ap = {"ID": ap_name, "RECHTS": ap_x, "HOCH":ap_y}             #standpunkt
        

        self.addAp()
        self.orientationArrow.addFeature(sp_x,sp_y,ap_x,ap_y)
        self.orientationArrow.addLayerToMapInstance()
        
    
            
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
        print('Select coordinates from map')
        
        
        def capture_coordinate(point, button):
            x = point.x()
            y = point.y()
            print(f"Captured coordinate: ({x}, {y})")

        
        self.action = QAction("Capture Coordinate", self.iface.mainWindow())
        tool = QgsMapToolEmitPoint(self.canvas)
        tool.canvasClicked.connect(capture_coordinate)
              
        self.canvas.setMapTool(tool)
        print('map tool set')
   
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
        
        vprism = self.dockwidget.radio_prism
        vreflex = self.dockwidget.radio_reflex
        vreflectorless = self.dockwidget.radio_reflectorless
        
        if vprism.isChecked():
            self.target = 0
            self.targetPrismConstant = -35   #sokkia default
            self.dockwidget.input_prismConstant.setText(str(self.targetPrismConstant))
        elif vreflex.isChecked():
            self.target =1 
            self.targetPrismConstant = 0
            self.dockwidget.input_prismConstant.setText(str(self.targetPrismConstant))
        elif vreflectorless.isChecked():
            self.target = 2
            self.targetPrismConstant = 0
            self.dockwidget.input_prismConstant.setText(str(self.targetPrismConstant))
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
            
            
        self.targetPrismConstant = int(self.dockwidget.input_prismConstant.text())
            
        pc1 = b'/B 0,0,0,'
        pc2 = b',1,0,0,0,0,0,0,0\r\n'
        command2 = pc1 + str(self.targetPrismConstant).encode('utf-8') + pc2
        
        print(command2)
        
        self.serial.write(command)  
        self.serial.write(command2) 
        
        self.dockwidget.lbl_target.setText(f"Zieltyp: {targetType}, th: {float(self.dockwidget.input_th.text())}, Pismentkonstante: {self.targetPrismConstant}")
        
        
        
        
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
        print('Setze Standpunkt')
        
        #get values from ui
        id = self.dockwidget.input_standpoint.text()
        x = float(self.dockwidget.input_sp_x.text())
        y = float(self.dockwidget.input_sp_y.text())
        z = float(self.dockwidget.input_sp_z.text())
        ih = float(self.dockwidget.input_ih.text())
        
        self.sp = {"ID": id, "RECHTS": x, "HOCH":y, "H": z, "ih": ih} 
        
        #Orientierung Berechnen
        self.calc_orientation()
        self.dockwidget.input_orientation.setText(str(self.orientation))
        
        self.addStation()
        
        self.dockwidget.lbl_sp.setText(f"ID: {self.sp['ID']}, RECHTS: {self.sp['RECHTS']}, HOCH: {self.sp['HOCH']}, H: {self.sp['H']}, ih: {self.sp['ih']}")
    
    def control(self, direction, step):
        
        print("Control totalstation in " + direction + ' direction')
        
        #get current angle value
        self.mesaure_angle()
        
        #time.sleep(0.1)
        
        stepsize = float(self.dockwidget.input_control_step.text())
        
        ha = self.measureValues["ha"]
        za = self.measureValues["za"]
        
        if direction == 'h':
            ha = ha + step* stepsize
            
            if ha > 400:
                ha = ha - 400
            
            if ha < 0:
                ha = ha + 400   
            print(ha)
            
        if direction == 'v':
            za = za + step* stepsize
            
            
            if za > 400:
                za = za - 400
            
            if za < 0:
                za = za + 400
                
            print(za)
            
        
        ha_string = f"{ha:.4f}".replace('.', '')
        
        if len(ha_string) < 7:
            ha_string = '0' + ha_string
        za_string = f"{za:.4f}".replace('.', '')
        
        if len(za_string) < 7:                  #führende Null hinzufügen bei Zahlen < 100
            za_string = '0' + za_string
        
        
        #generate new value and send command
        command = f"*DHA{ha_string}VA{za_string}".encode('utf-8')
        self.serial.write(command)
        
        
           
    
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

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)


            #connect 'connect' btn
            self.dockwidget.btn_connect.clicked.connect(self.connectToSerial)
            
            #connect 'disconnect' btn
            self.dockwidget.btn_disconnect.clicked.connect(self.disconnectFromSerial)

            #connect 'select coorinates from Map' btn
            self.dockwidget.btn_select_sp.clicked.connect(self.selectCoordinatesFromMap)
            self.dockwidget.btn_setSp.clicked.connect(self.setSp)
            
            #default values
            #self.sp = {"ID": "SP1", "RECHTS": 0, "HOCH":0, "H": 0, "ih": 0} 
            self.dockwidget.input_standpoint.setText(self.sp['ID'])
            
            
            #connect control buttons
            
            self.dockwidget.btn_control_left.clicked.connect(lambda: self.control('h', -1))
            self.dockwidget.btn_control_right.clicked.connect(lambda: self.control('h',1))
            self.dockwidget.btn_control_up.clicked.connect(lambda: self.control('v', -1))
            self.dockwidget.btn_control_down.clicked.connect(lambda: self.control('v', 1))
            

            #connect 'toggle Laser from Map' btn
            self.dockwidget.btn_laser.clicked.connect(self.switchLaser)

            #radio buttons for target selection
            self.dockwidget.radio_prism.clicked.connect(self.selectTarget)
            self.dockwidget.radio_reflex.clicked.connect(self.selectTarget)
            self.dockwidget.radio_reflectorless.clicked.connect(self.selectTarget)
            self.dockwidget.btn_setTarget.clicked.connect(self.setTarget)
            
            #set prismconstant to default value
            self.dockwidget.input_prismConstant.setText(str(self.targetPrismConstant))
            
            #Mess buttons
            self.dockwidget.btn_measure.clicked.connect(self.mesaure)
            self.dockwidget.btn_measure_a.clicked.connect(self.mesaure_angle)
            self.dockwidget.btn_measure_stop.clicked.connect(self.mesaure_stop)
            
            
            

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(Qt.TopDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
