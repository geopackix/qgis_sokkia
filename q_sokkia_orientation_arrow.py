from qgis.core import QgsPointXY, QgsPoint, QgsWkbTypes, QgsVectorLayer, QgsFeature, QgsGeometry, QgsProject, QgsField, QgsFields
from PyQt5.QtCore import QVariant
import os
class OrientationArrow:
    
    def __init__(self, crs):
      
      self.layer =  QgsVectorLayer("LineString?crs="+crs, 'Orientation', "memory")
      fields = QgsFields()
      fields.append(QgsField("id", QVariant.Int))
      self.layer.dataProvider().addAttributes(fields)
      self.layer.updateFields()
      qml_file = f'{os.path.dirname(__file__)}/orientation_arrow.qml'
      self.layer.loadNamedStyle(qml_file)

    def addLayerToMapInstance(self):
      # Add layer to project
      QgsProject.instance().addMapLayer(self.layer)
      
    def addFeature(self,X1,Y1,X2,Y2):
      # Create a line feature
      feature = QgsFeature()
      feature.setFields(self.layer.fields())  # Important to set fields before attributes
      feature.setAttribute("id", 1)

      # Create line geometry (list of points)
      points = [
          QgsPointXY(X1, Y1),  
          QgsPointXY(X2, Y2)
      ]
      line_geometry = QgsGeometry.fromPolylineXY(points)
      feature.setGeometry(line_geometry)

      # Add feature to layer
      self.layer.dataProvider().addFeature(feature)
      self.layer.updateExtents()