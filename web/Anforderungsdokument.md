# Anforderungsdokument: Webbasierte Version des Sokkia QGIS-Plugins

## Zielsetzung
Entwicklung einer webbasierten Anwendung zur Steuerung und Datenauswertung von Sokkia-Tachymetern, die unabhängig von QGIS und Desktop-Umgebungen funktioniert. Die Web-App soll die wichtigsten Funktionen des bestehenden QGIS-Plugins abbilden und für verschiedene Endgeräte (Desktop, Tablet, Smartphone) nutzbar sein.

## Funktionale Anforderungen

### 1. Gerätekommunikation
- Verbindung zu Sokkia-Tachymetern über serielle Schnittstelle (backend (node.js) mit Websocket Schnittstelle zum Frontend)
- Initialisierung, Steuerung und Statusabfrage des Geräts
- Senden und Empfangen von Messdaten

### 2. Mess- und Steuerfunktionen
- Fernsteuerung (z.B. Drehen, Zielen, Lasersteuerung)
- Punktmessung (Koordinaten, Winkel, Entfernung)
- Absteckung (Zielpunkt anfahren, Reststrecke anzeigen)
- Freie Stationierung (mehrere Anschlusspunkte, Berechnung der Standpunktkoordinaten)
- EInstellen der Zielparameter (Prisma, Reflektorlos, Prismenkonstante)

### 3. Karten- und Punktverwaltung
- Anzeige und Verwaltung von Punktlisten (Import/Export von GeoJSON, CSV, etc.)
- Visualisierung der Messpunkte und Zielpunkte auf einer Karte (z.B. Leaflet)
- Auswahl und Markierung von Punkten

### 4. Protokollierung und Berichte
- Automatische Protokollierung aller Messungen und Aktionen
- Export von Protokollen (z.B. als PDF, TXT)
- Anzeige von Restklaffen, Residuen und Genauigkeiten bei Stationierung

### 5. Benutzeroberfläche
- Intuitive, responsive Web-Oberfläche (Karte zentral)
- Mehrsprachigkeit (mind. Deutsch/Englisch)
- Benutzer- und Rechteverwaltung (optional)
- Benutzereinstellungen
- optimiert für kleinere displays (7 Zoll - z.b. 1280 x 800)

## Nicht-funktionale Anforderungen
- Plattformunabhängigkeit (Browser-basiert)
- Keine QGIS-Abhängigkeit
- Modularer Aufbau (Frontend (vue3) /Backend-Trennung)
- Erweiterbarkeit für weitere Tachymeter-Modelle
- Datenschutz und Datensicherheit (lokale Speicherung, ggf. Serverbetrieb)

## Technische Anforderungen
- Frontend: Moderne Webtechnologien (z.B. Vue3, Svelte oder Vanilla JS)
- Backend (optional): Node.js
- Schnittstellen: Websocket, WebSerial/WebUSB, REST-API
- Kartenkomponente: Leaflet
- Datenformate: GeoJSON, CSV, TXT

## Abgrenzung
- Keine direkte Integration in QGIS
- Keine Desktop-Only-Lösung
- Keine Abhängigkeit von proprietären GIS-Systemen

---
Letzte Aktualisierung: 19.04.2026
