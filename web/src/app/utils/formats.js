/**
 * Import/Export utilities for GeoJSON, CSV, and TXT formats.
 */

// ── GeoJSON ──

export function pointsToGeoJSON(points) {
  return {
    type: 'FeatureCollection',
    features: points.map(p => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [p.easting, p.northing, p.height || 0],
      },
      properties: {
        id: p.id,
        description: p.description || '',
        code: p.code || '',
        ...(p.ha !== undefined && { ha: p.ha }),
        ...(p.za !== undefined && { za: p.za }),
        ...(p.sd !== undefined && { sd: p.sd }),
        ...(p.hd !== undefined && { hd: p.hd }),
        ...(p.ih !== undefined && { ih: p.ih }),
        ...(p.th !== undefined && { th: p.th }),
      },
    })),
  }
}

export function parseGeoJSON(text) {
  const geojson = JSON.parse(text)
  if (geojson.type !== 'FeatureCollection' || !Array.isArray(geojson.features)) {
    throw new Error('Ungültiges GeoJSON FeatureCollection')
  }

  return geojson.features
    .filter(f => f.geometry && f.geometry.type === 'Point')
    .map(f => ({
      id: f.properties?.id || f.properties?.name || f.properties?.Punktname || '',
      easting: f.geometry.coordinates[0],
      northing: f.geometry.coordinates[1],
      height: f.geometry.coordinates[2] || f.properties?.height || f.properties?.Hoehe || 0,
      description: f.properties?.description || f.properties?.Beschreibung || '',
      code: f.properties?.code || '',
    }))
}

// ── CSV ──

export function pointsToCSV(points, separator = ';') {
  const header = ['ID', 'Easting', 'Northing', 'Height', 'Description', 'Code'].join(separator)
  const rows = points.map(p =>
    [p.id, p.easting?.toFixed(4), p.northing?.toFixed(4), p.height?.toFixed(4), p.description || '', p.code || ''].join(separator)
  )
  return [header, ...rows].join('\n')
}

export function parseCSV(text, separator) {
  const lines = text.split(/\r?\n/).filter(l => l.trim())
  if (lines.length < 2) return []

  // Auto-detect separator if not given
  if (!separator) {
    const first = lines[0]
    if (first.includes('\t')) separator = '\t'
    else if (first.includes(';')) separator = ';'
    else separator = ','
  }

  const header = lines[0].split(separator).map(h => h.trim().toLowerCase())
  const points = []

  // Try to find columns by common names
  const colMap = {
    id: findCol(header, ['id', 'punkt', 'punktname', 'point', 'name', 'nr']),
    easting: findCol(header, ['easting', 'e', 'x', 'rechtswert', 'rechts', 'east']),
    northing: findCol(header, ['northing', 'n', 'y', 'hochwert', 'hoch', 'north']),
    height: findCol(header, ['height', 'h', 'z', 'hoehe', 'höhe', 'elevation', 'ele']),
    description: findCol(header, ['description', 'desc', 'beschreibung', 'bemerkung']),
    code: findCol(header, ['code', 'typ']),
  }

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(separator)
    points.push({
      id: colMap.id >= 0 ? cols[colMap.id]?.trim() : `P${i}`,
      easting: colMap.easting >= 0 ? parseFloat(cols[colMap.easting]) || 0 : 0,
      northing: colMap.northing >= 0 ? parseFloat(cols[colMap.northing]) || 0 : 0,
      height: colMap.height >= 0 ? parseFloat(cols[colMap.height]) || 0 : 0,
      description: colMap.description >= 0 ? cols[colMap.description]?.trim() || '' : '',
      code: colMap.code >= 0 ? cols[colMap.code]?.trim() || '' : '',
    })
  }

  return points
}

function findCol(headers, candidates) {
  for (const c of candidates) {
    const idx = headers.indexOf(c)
    if (idx >= 0) return idx
  }
  return -1
}

// ── File helpers ──

export function downloadFile(content, filename, mimeType = 'text/plain') {
  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })
}
