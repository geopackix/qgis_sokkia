<template>
  <div>
    <div class="card">
      <div class="card-title">⊕ {{ t('station.freeStation') }}</div>

      <div class="form-row">
        <div class="form-group">
          <label>{{ t('station.instrumentHeight') }}</label>
          <input type="number" v-model.number="ih" step="0.001" />
        </div>
      </div>

      <!-- Measurements table -->
      <div class="scroll-y" style="max-height: 220px; margin: 8px 0">
        <table>
          <thead>
            <tr>
              <th>{{ t('station.pointId') }}</th>
              <th>Hz [gon]</th>
              <th>V [gon]</th>
              <th>SD [m]</th>
              <th>AP-E</th>
              <th>AP-N</th>
              <th>AP-H</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(m, i) in measurements" :key="i">
              <td><input v-model="m.id" style="width: 60px" /></td>
              <td>{{ m.ha.toFixed(4) }}</td>
              <td>{{ m.za.toFixed(4) }}</td>
              <td>{{ m.sd.toFixed(3) }}</td>
              <td><input type="number" v-model.number="m.cpE" step="0.001" style="width: 85px" /></td>
              <td><input type="number" v-model.number="m.cpN" step="0.001" style="width: 85px" /></td>
              <td><input type="number" v-model.number="m.cpH" step="0.001" style="width: 75px" /></td>
              <td><button class="btn btn-sm btn-danger" @click="removeMeasurement(i)">✕</button></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="btn-row">
        <button class="btn" @click="addMeasurement">⌖ {{ t('station.addMeasurement') }}</button>
        <button class="btn btn-primary" @click="calculate" :disabled="measurements.length < 2">
          ✓ {{ t('station.calculate') }}
        </button>
      </div>
    </div>

    <!-- Results -->
    <div v-if="result" class="card">
      <div class="card-title">{{ t('station.residuals') }}</div>

      <div class="readings-grid" style="margin-bottom: 10px">
        <div class="reading-cell">
          <div class="reading-label">E</div>
          <div class="reading-value">{{ result.position[0].toFixed(3) }}</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">N</div>
          <div class="reading-value">{{ result.position[1].toFixed(3) }}</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">H</div>
          <div class="reading-value">{{ result.position[2].toFixed(3) }}</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">z₀ [gon]</div>
          <div class="reading-value">{{ result.orientationGon.toFixed(4) }}</div>
        </div>
      </div>

      <div class="form-row" style="font-size: 0.8rem">
        <div><strong>{{ t('station.sigmaPos') }}:</strong> {{ result.std_dev.map(s => (s * 1000).toFixed(1)).join(' / ') }} mm</div>
        <div><strong>{{ t('station.rms') }}:</strong> {{ (result.rms_residual * 1000).toFixed(1) }} mm</div>
        <div><strong>{{ t('station.dof') }}:</strong> {{ result.dof }}</div>
      </div>

      <!-- Per-point residuals -->
      <table style="margin-top: 8px">
        <thead>
          <tr>
            <th>{{ t('station.pointId') }}</th>
            <th>{{ t('station.dSd') }}</th>
            <th>{{ t('station.dHz') }}</th>
            <th>{{ t('station.dV') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in result.residuals" :key="r.id">
            <td>{{ r.id }}</td>
            <td>{{ r.dSd.toFixed(1) }}</td>
            <td>{{ r.dHz.toFixed(1) }}</td>
            <td>{{ r.dV.toFixed(1) }}</td>
          </tr>
        </tbody>
      </table>

      <button class="btn btn-success btn-lg" style="margin-top: 10px" @click="applyResult">
        ✓ {{ t('station.setStation') }}
      </button>
    </div>

    <div v-if="error" class="card" style="border-color: var(--danger)">
      <div style="color: var(--danger)">{{ error }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'
import { useStationStore } from '../../stores/station.js'
import { useSettingsStore } from '../../stores/settings.js'
import { resection } from '../../utils/resection.js'

const { t } = useI18n()
const device = useDeviceStore()
const station = useStationStore()
const settings = useSettingsStore()

const ih = ref(settings.defaultIh)
const measurements = ref([])
const result = ref(null)
const error = ref('')

function addMeasurement() {
  // Capture current instrument readings
  measurements.value.push({
    id: `AP${measurements.value.length + 1}`,
    ha: device.ha,
    za: device.za,
    sd: device.sd,
    cpE: 0,
    cpN: 0,
    cpH: 0,
  })
}

// Also listen for live measurements from device
function onDeviceMeasurement(data) {
  // Auto-fill last empty row or just update display
}

onMounted(() => {
  device.on('measurement', onDeviceMeasurement)
})

onUnmounted(() => {
  device.off('measurement', onDeviceMeasurement)
})

function removeMeasurement(index) {
  measurements.value.splice(index, 1)
}

function calculate() {
  error.value = ''
  result.value = null

  try {
    const controlPoints = measurements.value.map(m => ({
      id: m.id,
      easting: m.cpE,
      northing: m.cpN,
      height: m.cpH,
    }))

    const obs = measurements.value.map(m => ({
      id: m.id,
      ha: m.ha,
      za: m.za,
      sd: m.sd,
    }))

    result.value = resection(controlPoints, obs)
  } catch (e) {
    error.value = e.message
  }
}

function applyResult() {
  if (!result.value) return
  station.applyResection(result.value)
  station.instrumentHeight = ih.value
  station.id = 'FS'
}
</script>
