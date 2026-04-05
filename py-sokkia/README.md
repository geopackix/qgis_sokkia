# Toolbox for Sokkia SDR33 format (Python)

## Motivation

This toolbox allows you to create survey job data in SDR33 format from Sokkia.
Currently the point/coordinate exchange is in the focus of the development.

## Scope

- Import GeoJSON file
- Add coordinates manually
- Create SDR33 export message

## Installation

```bash
pip install -e .
```

## Usage

**Import GeoJSON file**

```python
from sdr33 import Sdr33Export

sdrmessage = Sdr33Export.from_geojson("./ref.geojson")
```

**Add coordinates manually**

```python
from sdr33 import Sdr33Export, Coordinate

# Job name
sdrmessage = Sdr33Export("New-Job")

# Point name, North value, East value, Elevation, Point description
sdrmessage.add_coordinate(Coordinate("103.0001", 0, 0, 0, ""))
```

**Create SDR33 export message**

```python
sdr33message = sdrmessage.get_message()
```
