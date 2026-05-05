<template>
  <div class="map-wrapper" ref="mapContainer">
    <div ref="mapEl" style="width: 100%; height: 100%"></div>

    <!-- Station info overlay -->
    <div v-if="station" class="map-overlay map-overlay-station">
      <div><span class="ov-label">SP:</span> <span class="ov-value">{{ station.id }}</span></div>
      <div><span class="ov-label">E:</span> <span class="ov-value">{{ station.easting.toFixed(3) }}</span></div>
      <div><span class="ov-label">N:</span> <span class="ov-value">{{ station.northing.toFixed(3) }}</span></div>
    </div>

    <!-- Coordinate readout overlay -->
    <div class="map-overlay map-overlay-coords">
      <div><span class="ov-label">E:</span> <span class="ov-value">{{ cursorE.toFixed(3) }}</span></div>
      <div><span class="ov-label">N:</span> <span class="ov-value">{{ cursorN.toFixed(3) }}</span></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import L from 'leaflet'
import { useSettingsStore } from '../../stores/settings.js'
import { MAP_TILE_PROVIDERS } from '../../utils/constants.js'

const props = defineProps({
  points: Array,
  controlPoints: Array,
  station: Object,
  aimLine: Object,
})

const settings = useSettingsStore()

const mapEl = ref(null)
const mapContainer = ref(null)
const cursorE = ref(0)
const cursorN = ref(0)

let map = null
let tileLayer = null
let pointsLayer = null
let controlPointsLayer = null
let stationMarker = null
let aimLineLayer = null

// Marker styles
const pointStyle = {
  radius: 5,
  fillColor: '#e94560',
  color: '#fff',
  weight: 2,
  fillOpacity: 0.9,
}

const controlPointStyle = {
  radius: 5,
  fillColor: '#00b0ff',
  color: '#fff',
  weight: 2,
  fillOpacity: 0.9,
}

const stationIcon = L.divIcon({
  html: '<div style="width:14px;height:14px;background:#00c853;border:2px solid #fff;border-radius:50%;box-shadow:0 0 6px #00c853"></div>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
  className: '',
})

onMounted(() => {
  nextTick(() => {
    initMap()
    // Force Leaflet to recalculate size after DOM is fully laid out
    setTimeout(() => map?.invalidateSize(), 100)
  })
})

onUnmounted(() => {
  if (map) {
    map.remove()
    map = null
  }
})

function initMap() {
  if (!mapEl.value) return

  map = L.map(mapEl.value, {
    center: [51.0, 10.0], // Default center Germany
    zoom: 6,
    zoomControl: true,
    attributionControl: false,
  })

  // Tile layer
  const provider = MAP_TILE_PROVIDERS[settings.mapTiles] || MAP_TILE_PROVIDERS.osm
  tileLayer = L.tileLayer(provider.url, {
    attribution: provider.attribution,
    maxZoom: provider.maxZoom,
  }).addTo(map)

  // Feature layers
  pointsLayer = L.layerGroup().addTo(map)
  controlPointsLayer = L.layerGroup().addTo(map)
  aimLineLayer = L.layerGroup().addTo(map)

  // Cursor coordinate tracking
  map.on('mousemove', (e) => {
    cursorE.value = e.latlng.lng
    cursorN.value = e.latlng.lat
  })

  // Initial render
  updatePoints()
  updateStation()
}

// Watch for point changes
watch(() => props.points, updatePoints, { deep: true })
watch(() => props.controlPoints, updateControlPoints, { deep: true })
watch(() => props.station, updateStation, { deep: true })
watch(() => props.aimLine, updateAimLine, { deep: true })

watch(() => settings.mapTiles, () => {
  if (!map || !tileLayer) return
  map.removeLayer(tileLayer)
  const provider = MAP_TILE_PROVIDERS[settings.mapTiles] || MAP_TILE_PROVIDERS.osm
  tileLayer = L.tileLayer(provider.url, {
    attribution: provider.attribution,
    maxZoom: provider.maxZoom,
  }).addTo(map)
})

function updatePoints() {
  if (!pointsLayer) return
  pointsLayer.clearLayers()

  for (const p of (props.points || [])) {
    if (p.easting && p.northing) {
      const marker = L.circleMarker([p.northing, p.easting], pointStyle)
      marker.bindTooltip(`${p.id}<br>E: ${p.easting.toFixed(3)}<br>N: ${p.northing.toFixed(3)}<br>H: ${(p.height || 0).toFixed(3)}`, {
        className: 'map-tooltip',
      })
      pointsLayer.addLayer(marker)
    }
  }

  fitBounds()
}

function updateControlPoints() {
  if (!controlPointsLayer) return
  controlPointsLayer.clearLayers()

  for (const p of (props.controlPoints || [])) {
    if (p.easting && p.northing) {
      const marker = L.circleMarker([p.northing, p.easting], controlPointStyle)
      marker.bindTooltip(`${p.id} (AP)`)
      controlPointsLayer.addLayer(marker)
    }
  }
}

function updateStation() {
  if (!map) return

  if (stationMarker) {
    map.removeLayer(stationMarker)
    stationMarker = null
  }

  if (props.station) {
    stationMarker = L.marker(
      [props.station.northing, props.station.easting],
      { icon: stationIcon }
    ).addTo(map)
    stationMarker.bindTooltip(`SP: ${props.station.id}`)
  }
}

function updateAimLine() {
  if (!aimLineLayer) return
  aimLineLayer.clearLayers()

  if (props.aimLine) {
    const line = L.polyline(
      [
        [props.aimLine.from[1], props.aimLine.from[0]],
        [props.aimLine.to[1], props.aimLine.to[0]],
      ],
      { color: '#ff9800', weight: 2, dashArray: '8,4', opacity: 0.7 }
    )
    aimLineLayer.addLayer(line)
  }
}

function fitBounds() {
  if (!map) return
  const allLayers = [...(pointsLayer?.getLayers() || []), stationMarker].filter(Boolean)
  if (allLayers.length === 0) return

  const group = L.featureGroup(allLayers)
  if (group.getBounds().isValid()) {
    map.fitBounds(group.getBounds(), { padding: [40, 40], maxZoom: 20 })
  }
}

// Expose for parent to trigger resize
defineExpose({
  invalidateSize: () => nextTick(() => map?.invalidateSize()),
})
</script>

<style scoped>
.map-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}
</style>
