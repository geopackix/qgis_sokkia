<template>
  <div id="sokkia-app">
    <AppHeader @open-connect="showConnect = true" />
    <AppNav v-model="activeTab" />

    <div class="main-content">
      <!-- Map is always rendered (background layer) -->
      <div class="map-layer">
        <MapView
          :points="measureStore.points"
          :controlPoints="measureStore.controlPoints"
          :station="stationData"
          :aimLine="aimLine"
        />
      </div>

      <!-- Panel overlay (hides when Map tab is active) -->
      <div class="panel-overlay" :class="{ hidden: activeTab === 'map' }">
        <StationSetup v-if="activeTab === 'station'" />
        <MeasurePanel v-if="activeTab === 'measure'" />
        <StakeoutPanel v-if="activeTab === 'stakeout'" />
        <RemoteControl v-if="activeTab === 'remote'" />
        <TransferDialog v-if="activeTab === 'transfer'" />
        <ProtocolView v-if="activeTab === 'protocol'" />
        <SettingsDialog v-if="activeTab === 'settings'" />
      </div>
    </div>

    <StatusBar />

    <!-- Connection modal -->
    <ConnectionDialog v-if="showConnect" @close="showConnect = false" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useDeviceStore } from './stores/device.js'
import { useStationStore } from './stores/station.js'
import { useMeasurementStore } from './stores/measurement.js'
import { useProtocolStore } from './stores/protocol.js'
import { GON_TO_RAD, RAD_TO_GON } from './utils/geodesy.js'

import AppHeader from './components/layout/AppHeader.vue'
import AppNav from './components/layout/AppNav.vue'
import StatusBar from './components/layout/StatusBar.vue'
import MapView from './components/map/MapView.vue'
import StationSetup from './components/station/StationSetup.vue'
import MeasurePanel from './components/measure/MeasurePanel.vue'
import StakeoutPanel from './components/stakeout/StakeoutPanel.vue'
import RemoteControl from './components/remote/RemoteControl.vue'
import TransferDialog from './components/transfer/TransferDialog.vue'
import ProtocolView from './components/protocol/ProtocolView.vue'
import SettingsDialog from './components/settings/SettingsDialog.vue'
import ConnectionDialog from './components/layout/ConnectionDialog.vue'

const deviceStore = useDeviceStore()
const stationStore = useStationStore()
const measureStore = useMeasurementStore()
const protocolStore = useProtocolStore()

const activeTab = ref('map')
const showConnect = ref(false)

// Station data for map
const stationData = computed(() => {
  if (!stationStore.isSet) return null
  return {
    easting: stationStore.easting,
    northing: stationStore.northing,
    id: stationStore.id,
  }
})

// Aim line (current instrument direction)
const aimLine = computed(() => {
  if (!stationStore.isSet) return null
  const haOriented = ((deviceStore.ha + stationStore.orientation * RAD_TO_GON) % 400 + 400) % 400
  const haRad = haOriented * GON_TO_RAD
  const len = 100 // 100m line
  return {
    from: [stationStore.easting, stationStore.northing],
    to: [
      stationStore.easting + len * Math.sin(haRad),
      stationStore.northing + len * Math.cos(haRad),
    ],
  }
})

// Listen for measurements and auto-add points if auto-save is on
function onMeasurement(data) {
  if (measureStore.autoSave && stationStore.isSet && data.sd > 0) {
    measureStore.addMeasuredPoint(data.ha, data.za, data.sd)
  }
}

onMounted(() => {
  deviceStore.on('measurement', onMeasurement)
  protocolStore.log('Sokkia Web Control gestartet')
})

onUnmounted(() => {
  deviceStore.off('measurement', onMeasurement)
})
</script>
