import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useProtocolStore } from './protocol.js'
import { calcOrientation, computeCoordinates } from '../utils/geodesy.js'

export const useStationStore = defineStore('station', () => {
  // Station position
  const id = ref('SP1')
  const easting = ref(0)   // Rechtswert
  const northing = ref(0)  // Hochwert
  const height = ref(0)    // Höhe
  const instrumentHeight = ref(1.600)

  // Orientation
  const orientation = ref(0)  // z₀ in radians

  // Control point for manual orientation
  const apId = ref('')
  const apEasting = ref(0)
  const apNorthing = ref(0)

  // Station is set flag
  const isSet = ref(false)

  // Last resection result
  const resectionResult = ref(null)

  const orientationGon = computed(() => orientation.value * 200 / Math.PI)

  function setStation(data) {
    const protocol = useProtocolStore()
    id.value = data.id || id.value
    easting.value = data.easting
    northing.value = data.northing
    height.value = data.height
    instrumentHeight.value = data.instrumentHeight
    isSet.value = true

    protocol.log(`Standpunkt gesetzt: ${id.value} E=${easting.value.toFixed(3)} N=${northing.value.toFixed(3)} H=${height.value.toFixed(3)} ih=${instrumentHeight.value.toFixed(3)}`)
  }

  function setOrientationFromAP(apE, apN) {
    const o = calcOrientation(easting.value, northing.value, apE, apN)
    orientation.value = o
    useProtocolStore().log(`Orientierung berechnet: z₀=${(o * 200 / Math.PI).toFixed(4)} gon`)
    return o
  }

  function setOrientationDirect(radians) {
    orientation.value = radians
  }

  function zeroOrientation() {
    orientation.value = 0
    useProtocolStore().log('Orientierung auf 0 gesetzt')
  }

  function applyResection(result) {
    resectionResult.value = result
    easting.value = result.position[0]
    northing.value = result.position[1]
    height.value = result.position[2]
    orientation.value = result.orientation || 0
    isSet.value = true

    useProtocolStore().log(
      `Freie Stationierung: E=${easting.value.toFixed(3)} N=${northing.value.toFixed(3)} H=${height.value.toFixed(3)} ` +
      `σ=[${result.std_dev.map(s => (s * 1000).toFixed(1)).join(',')}] mm, RMS=${(result.rms_residual * 1000).toFixed(1)} mm`
    )
  }

  /**
   * Compute 3D coordinates from current station + measurement.
   */
  function computePoint(haGon, zaGon, sdMeter, targetHeight) {
    return computeCoordinates(
      easting.value, northing.value, height.value,
      instrumentHeight.value, orientation.value,
      haGon, zaGon, sdMeter, targetHeight
    )
  }

  return {
    id, easting, northing, height, instrumentHeight,
    orientation, orientationGon,
    apId, apEasting, apNorthing,
    isSet, resectionResult,
    setStation, setOrientationFromAP, setOrientationDirect, zeroOrientation,
    applyResection, computePoint,
  }
})
