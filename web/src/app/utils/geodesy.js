/**
 * Geodetic calculation utilities.
 * All angles in GON (400 gon = 360°) unless otherwise noted.
 * Internal computations use radians.
 */

const GON_TO_RAD = Math.PI / 200
const RAD_TO_GON = 200 / Math.PI

/**
 * Compute geodetic orientation (azimuth) from station to target.
 * Returns orientation in radians [0, 2π).
 */
export function calcOrientation(spE, spN, apE, apN) {
  const dx = apE - spE
  const dy = apN - spN
  let o = Math.atan2(dx, dy) // atan2(ΔE, ΔN) = geodetic convention (from North, clockwise)
  if (o < 0) o += 2 * Math.PI
  return o
}

/**
 * Compute 3D coordinates from station + polar measurement.
 *
 * @param {number} spE  - Station Easting
 * @param {number} spN  - Station Northing
 * @param {number} spH  - Station Height
 * @param {number} ih   - Instrument height [m]
 * @param {number} z0   - Orientation z₀ [radians]
 * @param {number} haGon - Horizontal angle reading [gon]
 * @param {number} zaGon - Zenith angle reading [gon]
 * @param {number} sd   - Slope distance [m]
 * @param {number} th   - Target height [m]
 * @returns {{ easting: number, northing: number, height: number, hd: number, vd: number }}
 */
export function computeCoordinates(spE, spN, spH, ih, z0, haGon, zaGon, sd, th) {
  const zaRad = zaGon * GON_TO_RAD
  const hd = sd * Math.sin(zaRad)
  const vd = sd * Math.cos(zaRad)

  // Apply orientation to get geodetic azimuth
  const haOriented = ((haGon + z0 * RAD_TO_GON) % 400 + 400) % 400
  const haRad = haOriented * GON_TO_RAD

  const easting = spE + hd * Math.sin(haRad)
  const northing = spN + hd * Math.cos(haRad)
  const height = spH + ih + vd - th

  return { easting, northing, height, hd, vd }
}

/**
 * Compute stakeout angles from station to target point.
 *
 * @param {number} spE, spN, spH - Station coordinates
 * @param {number} ih - Instrument height
 * @param {number} z0 - Orientation [radians]
 * @param {number} tE, tN, tH - Target coordinates
 * @param {boolean} mode3d - If true, compute vertical angle too
 * @returns {{ hzGon: number, zaGon: number, hd: number, dh: number }}
 */
export function computeStakeoutAngles(spE, spN, spH, ih, z0, tE, tN, tH, mode3d = true) {
  const dx = tE - spE
  const dy = tN - spN
  const hd = Math.sqrt(dx * dx + dy * dy)

  // Geodetic azimuth in gon
  let tGon = Math.atan2(dx, dy) * RAD_TO_GON
  tGon = ((tGon % 400) + 400) % 400

  // Instrument Hz reading (remove orientation)
  let hzGon = ((tGon - z0 * RAD_TO_GON) % 400 + 400) % 400

  // Zenith angle for 3D mode
  let zaGon = 100 // default: horizontal
  let dh = 0
  if (mode3d) {
    dh = tH - (spH + ih)
    const vAngle = Math.atan2(dh, hd) // elevation angle
    zaGon = ((100 - vAngle * RAD_TO_GON) % 400 + 400) % 400
  }

  return { hzGon, zaGon, hd, dh }
}

/**
 * Compute distance and direction between two points.
 */
export function distance2d(e1, n1, e2, n2) {
  const dx = e2 - e1
  const dy = n2 - n1
  return Math.sqrt(dx * dx + dy * dy)
}

export function distance3d(e1, n1, h1, e2, n2, h2) {
  const dx = e2 - e1
  const dy = n2 - n1
  const dz = h2 - h1
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/**
 * Compute azimuth in gon from point 1 to point 2.
 */
export function azimuth(e1, n1, e2, n2) {
  const dx = e2 - e1
  const dy = n2 - n1
  let az = Math.atan2(dx, dy) * RAD_TO_GON
  return ((az % 400) + 400) % 400
}

/**
 * Circular mean of angles in gon.
 */
export function circularMeanGon(angles) {
  if (angles.length === 0) return 0
  let sinSum = 0, cosSum = 0
  for (const a of angles) {
    const rad = a * GON_TO_RAD
    sinSum += Math.sin(rad)
    cosSum += Math.cos(rad)
  }
  let mean = Math.atan2(sinSum, cosSum) * RAD_TO_GON
  return ((mean % 400) + 400) % 400
}

export { GON_TO_RAD, RAD_TO_GON }
