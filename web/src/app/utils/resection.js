/**
 * Free Stationing / Resection via Gauss-Newton least squares adjustment.
 * Ported from the Python resection module.
 *
 * Supports:
 * - Slant distances (SD)
 * - Vertical angles (ZA → elevation angle)
 * - Horizontal angles (Hz with orientation unknown)
 */

const GON_TO_RAD = Math.PI / 200
const RAD_TO_GON = 200 / Math.PI

/**
 * Perform free stationing computation.
 *
 * @param {Array<{id: string, easting: number, northing: number, height: number}>} controlPoints
 * @param {Array<{id: string, ha: number, za: number, sd: number}>} measurements - angles in gon, distance in m
 * @param {number[]} initialGuess - [E, N, H] initial position estimate (optional)
 * @returns {ResectionResult}
 */
export function resection(controlPoints, measurements, initialGuess) {
  const n = controlPoints.length
  if (n < 2) throw new Error('Mindestens 2 Anschlusspunkte erforderlich')

  // Match measurements to control points by ID
  const pairs = []
  for (const m of measurements) {
    const cp = controlPoints.find(p => p.id === m.id)
    if (cp) {
      pairs.push({ cp, m })
    }
  }

  if (pairs.length < 2) throw new Error('Zu wenige Zuordnungen zwischen Messungen und Festpunkten')

  // Initial estimate: centroid of control points or provided
  let x0 = initialGuess
    ? [...initialGuess]
    : [
        pairs.reduce((s, p) => s + p.cp.easting, 0) / pairs.length,
        pairs.reduce((s, p) => s + p.cp.northing, 0) / pairs.length,
        pairs.reduce((s, p) => s + p.cp.height, 0) / pairs.length,
      ]

  const maxIter = 20
  const tolerance = 1e-8

  // Unknowns: [E, N, H, z0]
  // z0 = orientation unknown in radians
  let params = [...x0, 0] // E, N, H, z0

  for (let iter = 0; iter < maxIter; iter++) {
    const { A, l } = buildSystem(pairs, params)
    const AtA = matMul(transpose(A), A)
    const Atl = matVecMul(transpose(A), l)

    // Solve normal equations: AtA * dx = Atl
    const dx = solveLinear(AtA, Atl)
    if (!dx) throw new Error('Singuläre Normalgleichungsmatrix')

    // Update parameters
    for (let i = 0; i < params.length; i++) {
      params[i] += dx[i]
    }

    // Check convergence
    const maxDx = Math.max(...dx.map(Math.abs))
    if (maxDx < tolerance) break
  }

  // Final residuals and statistics
  const { A, l } = buildSystem(pairs, params)
  const v = matVecMul(A, [0, 0, 0, 0]) // Residuals = A*0 - l (since we already converged)
  // Actually residuals = observed - computed (l vector already contains these)
  const residuals = l.map(x => -x) // sign convention

  const numObs = l.length
  const numUnknowns = 4 // E, N, H, z0
  const dof = numObs - numUnknowns

  // Variance factor
  let vPv = 0
  for (const r of residuals) vPv += r * r
  const sigma0 = dof > 0 ? Math.sqrt(vPv / dof) : 0

  // Covariance matrix of unknowns
  const AtA = matMul(transpose(A), A)
  const Qxx = invertMatrix(AtA)

  const std_dev = Qxx
    ? [
        Math.sqrt(Math.abs(Qxx[0][0])) * sigma0,
        Math.sqrt(Math.abs(Qxx[1][1])) * sigma0,
        Math.sqrt(Math.abs(Qxx[2][2])) * sigma0,
      ]
    : [0, 0, 0]

  // Per-point residuals
  const pointResiduals = computePointResiduals(pairs, params)

  return {
    position: [params[0], params[1], params[2]],
    orientation: params[3],
    orientationGon: ((params[3] * RAD_TO_GON) % 400 + 400) % 400,
    std_dev,
    sigma0,
    rms_residual: Math.sqrt(vPv / Math.max(numObs, 1)),
    dof,
    residuals: pointResiduals,
    success: true,
  }
}

function buildSystem(pairs, params) {
  const [E, N, H, z0] = params
  const rows_A = []
  const rows_l = []

  for (const { cp, m } of pairs) {
    const dE = cp.easting - E
    const dN = cp.northing - N
    const dH = cp.height - H
    const hd_calc = Math.sqrt(dE * dE + dN * dN)
    const sd_calc = Math.sqrt(dE * dE + dN * dN + dH * dH)

    // Observation 1: Slant distance
    if (m.sd > 0) {
      // ∂SD/∂E = -dE/sd, ∂SD/∂N = -dN/sd, ∂SD/∂H = -dH/sd, ∂SD/∂z0 = 0
      rows_A.push([
        -dE / sd_calc,
        -dN / sd_calc,
        -dH / sd_calc,
        0,
      ])
      rows_l.push(m.sd - sd_calc)
    }

    // Observation 2: Vertical angle (ZA → elevation angle)
    const zaRad = m.za * GON_TO_RAD
    const vAngle_obs = Math.PI / 2 - zaRad  // elevation angle from ZA
    const vAngle_calc = hd_calc > 0.001 ? Math.atan2(dH, hd_calc) : 0

    if (hd_calc > 0.001) {
      const denom = sd_calc * sd_calc
      // ∂v/∂E, ∂v/∂N, ∂v/∂H
      rows_A.push([
        (dH * dE) / (denom * hd_calc),
        (dH * dN) / (denom * hd_calc),
        -hd_calc / denom,
        0,
      ])
      rows_l.push(vAngle_obs - vAngle_calc)
    }

    // Observation 3: Horizontal angle (Hz)
    const haRad = m.ha * GON_TO_RAD
    const azCalc = Math.atan2(dE, dN) // geodetic azimuth
    const haCalcRad = azCalc - z0

    let dAz = haRad - haCalcRad
    // Normalize to [-π, π]
    while (dAz > Math.PI) dAz -= 2 * Math.PI
    while (dAz < -Math.PI) dAz += 2 * Math.PI

    if (hd_calc > 0.001) {
      const hd2 = hd_calc * hd_calc
      rows_A.push([
        dN / hd2,
        -dE / hd2,
        0,
        -1,
      ])
      rows_l.push(dAz)
    }
  }

  return {
    A: rows_A,
    l: rows_l,
  }
}

function computePointResiduals(pairs, params) {
  const [E, N, H, z0] = params
  return pairs.map(({ cp, m }) => {
    const dE = cp.easting - E
    const dN = cp.northing - N
    const dH = cp.height - H
    const hd_calc = Math.sqrt(dE * dE + dN * dN)
    const sd_calc = Math.sqrt(dE * dE + dN * dN + dH * dH)

    const dSd = m.sd > 0 ? (m.sd - sd_calc) * 1000 : 0 // mm

    const zaRad = m.za * GON_TO_RAD
    const vAngle_obs = Math.PI / 2 - zaRad
    const vAngle_calc = hd_calc > 0.001 ? Math.atan2(dH, hd_calc) : 0
    const dV = (vAngle_obs - vAngle_calc) * RAD_TO_GON * 1000 // mgon

    const haRad = m.ha * GON_TO_RAD
    const azCalc = Math.atan2(dE, dN)
    let dHz = (haRad - (azCalc - z0))
    while (dHz > Math.PI) dHz -= 2 * Math.PI
    while (dHz < -Math.PI) dHz += 2 * Math.PI
    dHz = dHz * RAD_TO_GON * 1000 // mgon

    return {
      id: cp.id,
      dSd: Math.round(dSd * 10) / 10,
      dHz: Math.round(dHz * 10) / 10,
      dV: Math.round(dV * 10) / 10,
    }
  })
}

// --- Linear algebra helpers ---

function transpose(M) {
  const rows = M.length, cols = M[0].length
  const T = Array.from({ length: cols }, () => new Array(rows))
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      T[j][i] = M[i][j]
  return T
}

function matMul(A, B) {
  const m = A.length, n = B[0].length, p = B.length
  const C = Array.from({ length: m }, () => new Array(n).fill(0))
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < p; k++)
        C[i][j] += A[i][k] * B[k][j]
  return C
}

function matVecMul(M, v) {
  return M.map(row => row.reduce((s, val, i) => s + val * v[i], 0))
}

function solveLinear(A, b) {
  // Gauss elimination with partial pivoting
  const n = A.length
  const M = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    // Pivoting
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]]

    if (Math.abs(M[col][col]) < 1e-14) return null

    // Eliminate
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col]
      for (let j = col; j <= n; j++) {
        M[row][j] -= factor * M[col][j]
      }
    }
  }

  // Back-substitute
  const x = new Array(n)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n]
    for (let j = i + 1; j < n; j++) {
      x[i] -= M[i][j] * x[j]
    }
    x[i] /= M[i][i]
  }

  return x
}

function invertMatrix(M) {
  const n = M.length
  // Augment with identity
  const aug = M.map((row, i) => {
    const id = new Array(n).fill(0)
    id[i] = 1
    return [...row, ...id]
  })

  for (let col = 0; col < n; col++) {
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]]

    if (Math.abs(aug[col][col]) < 1e-14) return null

    const pivot = aug[col][col]
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot

    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const factor = aug[row][col]
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j]
      }
    }
  }

  return aug.map(row => row.slice(n))
}
