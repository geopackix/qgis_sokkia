<template>
  <div>
    <div class="card">
      <div class="card-title">🎯 {{ t('stakeout.title') }}</div>

      <!-- Point source -->
      <div class="form-row">
        <div class="form-group" style="flex: 2">
          <label>{{ t('stakeout.targetPoint') }}</label>
          <select v-model="selectedPointId">
            <option value="">{{ t('stakeout.selectPoint') }}</option>
            <option v-for="p in allPoints" :key="p.id" :value="p.id">
              {{ p.id }} (E: {{ p.easting.toFixed(1) }}, N: {{ p.northing.toFixed(1) }})
            </option>
          </select>
        </div>
      </div>

      <!-- Manual coordinates -->
      <div class="form-row">
        <div class="form-group">
          <label>E</label>
          <input type="number" v-model.number="targetE" step="0.001" />
        </div>
        <div class="form-group">
          <label>N</label>
          <input type="number" v-model.number="targetN" step="0.001" />
        </div>
        <div class="form-group">
          <label>H</label>
          <input type="number" v-model.number="targetH" step="0.001" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Modus</label>
          <select v-model="mode3d">
            <option :value="false">{{ t('stakeout.mode2d') }}</option>
            <option :value="true">{{ t('stakeout.mode3d') }}</option>
          </select>
        </div>
      </div>

      <button
        class="btn btn-primary btn-lg"
        @click="driveToTarget"
        :disabled="!station.isSet || targetE === 0"
      >
        🎯 {{ t('stakeout.drive') }}
      </button>
    </div>

    <!-- Live residuals -->
    <div v-if="isActive" class="card">
      <div class="card-title">{{ t('stakeout.title') }} — {{ selectedPointId || 'Manuell' }}</div>

      <div class="residual-display">
        <div class="residual-cell" :class="residualClass(residualHd)">
          <div class="reading-label">{{ t('stakeout.residualDist') }}</div>
          <div class="residual-value">{{ residualHd.toFixed(3) }}</div>
          <div class="reading-unit">m</div>
        </div>
        <div class="residual-cell" :class="residualClass(residualDHz * 10)">
          <div class="reading-label">{{ t('stakeout.residualHz') }}</div>
          <div class="residual-value">{{ residualDHz.toFixed(4) }}</div>
          <div class="reading-unit">gon</div>
        </div>
        <div class="residual-cell" :class="residualClass(residualDV * 10)">
          <div class="reading-label">ΔH</div>
          <div class="residual-value">{{ residualDH.toFixed(3) }}</div>
          <div class="reading-unit">m</div>
        </div>
      </div>

      <!-- Arrow indicator -->
      <div style="text-align: center; margin-top: 12px; font-size: 2rem">
        <span v-if="residualHd > 0.05">
          {{ directionArrow }}
        </span>
        <span v-else style="color: var(--success)">✓ Ziel erreicht</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'
import { useStationStore } from '../../stores/station.js'
import { useMeasurementStore } from '../../stores/measurement.js'
import { computeStakeoutAngles, distance2d, RAD_TO_GON } from '../../utils/geodesy.js'

const { t } = useI18n()
const device = useDeviceStore()
const station = useStationStore()
const measureStore = useMeasurementStore()

const selectedPointId = ref('')
const targetE = ref(0)
const targetN = ref(0)
const targetH = ref(0)
const mode3d = ref(true)
const isActive = ref(false)

// All available points (measured + control/imported)
const allPoints = computed(() => [
  ...measureStore.points,
  ...measureStore.controlPoints,
])

// Fill target coords when point selected
watch(selectedPointId, (id) => {
  const pt = allPoints.value.find(p => p.id === id)
  if (pt) {
    targetE.value = pt.easting
    targetN.value = pt.northing
    targetH.value = pt.height || 0
  }
})

// Live residuals
const stakeoutAngles = computed(() => {
  if (!station.isSet) return null
  return computeStakeoutAngles(
    station.easting, station.northing, station.height,
    station.instrumentHeight, station.orientation,
    targetE.value, targetN.value, targetH.value, mode3d.value
  )
})

const residualHd = computed(() => {
  if (!isActive.value || device.sd <= 0) return 0
  // Current measured position
  const coords = station.computePoint(device.ha, device.za, device.sd, measureStore.targetHeight)
  return distance2d(coords.easting, coords.northing, targetE.value, targetN.value)
})

const residualDHz = computed(() => {
  if (!stakeoutAngles.value || !isActive.value) return 0
  let diff = device.ha - stakeoutAngles.value.hzGon
  while (diff > 200) diff -= 400
  while (diff < -200) diff += 400
  return diff
})

const residualDH = computed(() => {
  if (!isActive.value || device.sd <= 0) return 0
  const coords = station.computePoint(device.ha, device.za, device.sd, measureStore.targetHeight)
  return coords.height - targetH.value
})

const residualDV = computed(() => residualDH.value) // alias

const directionArrow = computed(() => {
  const dHz = residualDHz.value
  if (Math.abs(dHz) > 0.5) return dHz > 0 ? '⟵ Links' : '⟶ Rechts'
  return '↕ Strecke korrigieren'
})

function residualClass(val) {
  const abs = Math.abs(val)
  if (abs < 0.01) return 'ok'
  if (abs < 0.05) return 'warn'
  return 'bad'
}

function driveToTarget() {
  if (!stakeoutAngles.value) return

  device.driveTo(stakeoutAngles.value.hzGon, stakeoutAngles.value.zaGon)
  isActive.value = true

  // Start continuous angle measurement for tracking
  device.measureAngle()
}
</script>
