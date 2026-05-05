<template>
  <footer class="status-bar">
    <span>{{ station.isSet ? `SP: ${station.id}` : '—' }}</span>
    <span>Hz: {{ device.ha.toFixed(4) }} gon | V: {{ device.za.toFixed(4) }} gon | SD: {{ device.sd.toFixed(3) }} m</span>
    <span>{{ settings.crs }}</span>
  </footer>
</template>

<script setup>
import { useDeviceStore } from '../../stores/device.js'
import { useStationStore } from '../../stores/station.js'
import { useSettingsStore } from '../../stores/settings.js'

const device = useDeviceStore()
const station = useStationStore()
const settings = useSettingsStore()
</script>
