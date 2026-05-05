import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useStationStore } from './station.js'
import { useProtocolStore } from './protocol.js'

export const useMeasurementStore = defineStore('measurement', () => {
  // All measured points
  const points = ref([])

  // Imported/loaded control points
  const controlPoints = ref([])

  // Current measurement counter
  const nextPointId = ref(100)
  const autoIncrement = ref(true)
  const autoSave = ref(false)

  // Target settings
  const targetHeight = ref(2.000)
  const targetType = ref('prism')       // 'prism' | 'reflective' | 'reflectorless'
  const prismConstant = ref(-35)         // mm
  const code = ref('')

  // Points filtered for display
  const sortedPoints = computed(() =>
    [...points.value].sort((a, b) => b.timestamp - a.timestamp)
  )

  /**
   * Add a measured point from raw measurement values.
   */
  function addMeasuredPoint(haGon, zaGon, sdMeter, pointName) {
    const station = useStationStore()
    const protocol = useProtocolStore()

    if (!station.isSet) {
      protocol.log('WARNUNG: Kein Standpunkt gesetzt!', 'warning')
    }

    const coords = station.computePoint(haGon, zaGon, sdMeter, targetHeight.value)
    const hdist = sdMeter * Math.sin(zaGon * Math.PI / 200)
    const vdist = sdMeter * Math.cos(zaGon * Math.PI / 200)

    const point = {
      id: pointName || String(nextPointId.value),
      stationId: station.id,
      timestamp: Date.now(),
      // Raw measurements
      ha: haGon,
      za: zaGon,
      sd: sdMeter,
      hd: hdist,
      vd: vdist,
      // Computed coordinates
      easting: coords.easting,
      northing: coords.northing,
      height: coords.height,
      // Settings at time of measurement
      ih: station.instrumentHeight,
      th: targetHeight.value,
      prismConst: prismConstant.value,
      code: code.value,
    }

    points.value.push(point)

    if (autoIncrement.value) {
      nextPointId.value++
    }

    protocol.log(
      `Punkt ${point.id}: E=${point.easting.toFixed(3)} N=${point.northing.toFixed(3)} H=${point.height.toFixed(3)} ` +
      `(Hz=${haGon.toFixed(4)} V=${zaGon.toFixed(4)} SD=${sdMeter.toFixed(3)})`
    )

    return point
  }

  function removePoint(index) {
    points.value.splice(index, 1)
  }

  function clearPoints() {
    points.value = []
  }

  /**
   * Import points from parsed data array.
   */
  function importPoints(data, target = 'control') {
    const list = target === 'control' ? controlPoints : points
    for (const pt of data) {
      list.value.push({
        id: pt.id || pt.name || `P${list.value.length}`,
        easting: pt.easting || pt.x || 0,
        northing: pt.northing || pt.y || 0,
        height: pt.height || pt.z || 0,
        description: pt.description || pt.desc || '',
        timestamp: Date.now(),
        code: pt.code || '',
        imported: true,
      })
    }
    useProtocolStore().log(`${data.length} Punkte importiert (${target})`)
  }

  function clearControlPoints() {
    controlPoints.value = []
  }

  /**
   * Find control point by ID.
   */
  function findControlPoint(id) {
    return controlPoints.value.find(p => p.id === id)
  }

  return {
    points, controlPoints, nextPointId,
    autoIncrement, autoSave,
    targetHeight, targetType, prismConstant, code,
    sortedPoints,
    addMeasuredPoint, removePoint, clearPoints,
    importPoints, clearControlPoints, findControlPoint,
  }
})
