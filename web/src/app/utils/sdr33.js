/**
 * SDR33 message builder and parser.
 * Ported from the Python sdr33 module.
 */

const STX = '\x02'
const ETX = '\x03'
const LF = '\x0A'

/** Pad/truncate string to fixed width (right-padded with spaces). */
function fillUp(str, len) {
  return (str || '').substring(0, len).padEnd(len, ' ')
}

/**
 * Compute SDR33 checksum: sum of ASCII codes excluding STX/ETX/CR/LF, mod 65536, zero-padded 5 digits.
 */
function computeChecksum(content) {
  let sum = 0
  for (const ch of content) {
    const code = ch.charCodeAt(0)
    if (code !== 0x02 && code !== 0x03 && code !== 0x0D && code !== 0x0A) {
      sum += code
    }
  }
  return (sum % 65536).toString().padStart(5, '0')
}

// ── Builder ──

/**
 * Build SDR33 Header record (type 00, 46 bytes).
 */
export function buildHeader(serial = '', dateStr = '') {
  if (!dateStr) {
    const now = new Date()
    dateStr = now.toISOString().slice(0, 10).replace(/-/g, '-') + ' ' + now.toTimeString().slice(0, 8)
  }
  // "00" + version(20) + serial(10) + datetime(16)
  const version = fillUp('SDR33 V04-04.02', 20)
  const ser = fillUp(serial, 10)
  const dt = fillUp(dateStr, 16)
  return `00${version}${ser}${dt}`
}

/**
 * Build SDR33 Job record (type 10, 26 bytes).
 */
export function buildJob(jobName = 'WebExport') {
  // "10" + jobName(16) + settings(8)
  const name = fillUp(jobName, 16)
  const settings = '10210100' // point_id=1, include_ele=0, angle_unit=2(gon), dist_unit=1(m), ...
  return `10${name}${settings}`
}

/**
 * Build SDR33 Coordinate record (type 08, 84 bytes).
 *
 * @param {string} pointId
 * @param {number} northing
 * @param {number} easting
 * @param {number} elevation
 * @param {string} description
 * @param {string} derivationCode - 'KI' = keyboard, 'NM' = not measured, 'CO' = coordinates
 */
export function buildCoordinate(pointId, northing, easting, elevation, description = '', derivationCode = 'KI') {
  const dc = fillUp(derivationCode, 2)
  const pid = fillUp(pointId, 16)
  const n = fillUp(northing.toFixed(4), 16)
  const e = fillUp(easting.toFixed(4), 16)
  const h = fillUp(elevation.toFixed(4), 16)
  const desc = fillUp(description, 16)
  return `08${dc}${pid}${n}${e}${h}${desc}`
}

/**
 * Build a complete SDR33 message from records.
 */
export function buildMessage(records) {
  const body = records.join('\r\n')
  const content = `${STX}${LF}${body}\r\n${ETX}`
  const checksum = computeChecksum(content)
  return `${content}${checksum}`
}

/**
 * Build SDR33 export for a list of points.
 */
export function buildExport(points, jobName = 'WebExport') {
  const records = [
    buildHeader(),
    buildJob(jobName),
    ...points.map(p =>
      buildCoordinate(
        p.id || p.name || '',
        p.northing || p.y || 0,
        p.easting || p.x || 0,
        p.height || p.z || 0,
        p.description || '',
        'KI'
      )
    ),
  ]
  return buildMessage(records)
}

// ── Parser ──

/**
 * Parse SDR33 data string into coordinate records.
 *
 * @param {string} data - Raw SDR33 text
 * @returns {Array<{id: string, northing: number, easting: number, height: number, description: string}>}
 */
export function parseSdr33(data) {
  const lines = data.split(/\r?\n/)
  const points = []

  for (const line of lines) {
    if (line.startsWith('08') && line.length >= 84) {
      // Type 08 = Coordinate record
      const pointId = line.substring(4, 20).trim()
      const northing = parseFloat(line.substring(20, 36).trim()) || 0
      const easting = parseFloat(line.substring(36, 52).trim()) || 0
      const elevation = parseFloat(line.substring(52, 68).trim()) || 0
      const description = line.substring(68, 84).trim()

      points.push({
        id: pointId,
        northing,
        easting,
        height: elevation,
        description,
      })
    }
  }

  return points
}
