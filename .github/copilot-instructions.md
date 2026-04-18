# Copilot Instructions – qgis_sokkia

## Qt UI Files (.ui) – Zeichenkodierung

In Qt `.ui`-Dateien (XML) **niemals** rohe Unicode- oder Emoji-Zeichen direkt einfügen.
Stattdessen ausschließlich **XML-Zeichenentitäten** (`&#xXXXX;`) verwenden.
So wird jede Encoding-Korruption durch Tools (PowerShell, Git, Editoren) zuverlässig verhindert,
da die Datei dann reines ASCII-XML ist.

### Häufig verwendete Zeichen – Referenztabelle

| Zeichen | Unicode | XML-Entität      | Verwendung              |
|---------|---------|------------------|-------------------------|
| ⟳       | U+27F3  | `&#x27F3;`       | Refresh / Aktualisieren |
| ▶       | U+25B6  | `&#x25B6;`       | Verbinden / Play        |
| ■       | U+25A0  | `&#x25A0;`       | Stop / Trennen          |
| ⚡      | U+26A1  | `&#x26A1;`       | Initialisieren          |
| 📡      | U+1F4E1 | `&#x1F4E1;`      | Koordinaten-Transfer    |
| 📍      | U+1F4CD | `&#x1F4CD;`      | Standort / Karte        |
| ⊕       | U+2295  | `&#x2295;`       | Freie Stationierung     |
| 🎯      | U+1F3AF | `&#x1F3AF;`      | Ziel                    |
| 🕹       | U+1F579 | `&#x1F579;`      | Fernsteuerung           |
| ↔       | U+2194  | `&#x2194;`       | Strecke / Bidirektional |
| ∠       | U+2220  | `&#x2220;`       | Winkel                  |
| —       | U+2014  | `&#x2014;`       | Em-Dash (Platzhalter)   |
| …       | U+2026  | `&#x2026;`       | Ellipsis                |
| ▲       | U+25B2  | `&#x25B2;`       | Hoch / Up               |
| ▼       | U+25BC  | `&#x25BC;`       | Runter / Down           |
| ◀       | U+25C0  | `&#x25C0;`       | Links / Left            |
| ▶       | U+25B6  | `&#x25B6;`       | Rechts / Right          |
| ✓       | U+2713  | `&#x2713;`       | Bestätigen / Setzen     |
| ä       | U+00E4  | `&#xE4;`         | Umlaut ä                |
| ö       | U+00F6  | `&#xF6;`         | Umlaut ö                |
| ü       | U+00FC  | `&#xFC;`         | Umlaut ü                |
| Ä       | U+00C4  | `&#xC4;`         | Umlaut Ä                |
| Ö       | U+00D6  | `&#xD6;`         | Umlaut Ö                |
| Ü       | U+00DC  | `&#xDC;`         | Umlaut Ü                |
| ß       | U+00DF  | `&#xDF;`         | Eszett ß                |

### Beispiel (korrekt)

```xml
<property name="text"><string>&#x25B6;  Verbinden</string></property>
<property name="text"><string>&#x25A0;  Trennen</string></property>
<property name="text"><string>&#x1F4CD; Standort &#x2026;</string></property>
<property name="text"><string>&#x2220; Winkel</string></property>
<property name="text"><string>Hz: &#x2014;</string></property>
```

### Beispiel (falsch – nicht verwenden)

```xml
<property name="text"><string>▶  Verbinden</string></property>
<property name="text"><string>📍 Standort …</string></property>
```

## Python-Dateien (.py)

In Python-Quelldateien dürfen Unicode-Strings normal verwendet werden (UTF-8 ist Standard).
Die Einschränkung gilt **nur für `.ui`-Dateien**.

## Allgemein

- Dieses Projekt ist ein QGIS-Plugin (Python/PyQt5).
- Alle Dialoge verwenden Qt `.ui`-Dateien (uic.loadUiType).
- Hauptdock: `q_sokkia_plugin_dockwidget_base.ui`
- Detaildialoge: `standort_dialog_base.ui`, `zielpunkt_dialog_base.ui`, `fernsteuerung_dialog_base.ui`
- Plugin-Logik: `q_sokkia_plugin.py`
