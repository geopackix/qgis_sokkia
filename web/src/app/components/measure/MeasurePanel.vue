<template>
  <div>
    <!-- Target Settings -->
    <div class="card">
      <div class="card-title">⌖ {{ t('measure.title') }}</div>

      <div class="form-row">
        <div class="form-group">
          <label>{{ t('measure.pointNumber') }}</label>
          <input v-model="currentPointId" />
        </div>
        <div class="form-group">
          <label>{{ t('measure.code') }}</label>
          <input v-model="measureStore.code" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>{{ t('measure.targetHeight') }} [m]</label>
          <input type="number" v-model.number="measureStore.targetHeight" step="0.001" />
        </div>
        <div class="form-group">
          <label>{{ t('measure.targetType') }}</label>
          <select v-model="measureStore.targetType" @change="applyTargetType">
            <option value="prism">{{ t('measure.prism') }}</option>
            <option value="reflectorless">{{ t('measure.reflectorless') }}</option>
            <option value="reflective">{{ t('measure.reflectiveSheet') }}</option>
          </select>
        </div>
        <div class="form-group">
          <label>{{ t('measure.prismConstant') }}</label>
          <input type="number" v-model.number="measureStore.prismConstant" />
        </div>
      </div>

      <div class="toggle-row">
        <input type="checkbox" v-model="measureStore.autoIncrement" id="auto-inc" />
        <label for="auto-inc">{{ t('measure.autoIncrement') }}</label>
        <input type="checkbox" v-model="measureStore.autoSave" id="auto-save" style="margin-left: 16px" />
        <label for="auto-save">{{ t('measure.autoSave') }}</label>
      </div>
    </div>

    <!-- Measurement buttons -->
    <div class="card">
      <div class="btn-row">
        <button class="btn" @click="measureAngle">{{ t('measure.measureAngle') }}</button>
        <button class="btn btn-primary btn-lg" @click="measureAndSave" style="flex: 2">
          ⌖ {{ t('measure.measureAndSave') }}
        </button>
        <button class="btn btn-danger" @click="device.stopMeasurement()">{{ t('measure.stopMeasure') }}</button>
      </div>
    </div>

    <!-- Live readings -->
    <div class="card">
      <div class="readings-grid">
        <div class="reading-cell reading-big">
          <div class="reading-label">{{ t('measure.hz') }}</div>
          <div class="reading-value">{{ device.ha.toFixed(4) }}</div>
          <div class="reading-unit">gon</div>
        </div>
        <div class="reading-cell reading-big">
          <div class="reading-label">{{ t('measure.v') }}</div>
          <div class="reading-value">{{ device.za.toFixed(4) }}</div>
          <div class="reading-unit">gon</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">{{ t('measure.sd') }}</div>
          <div class="reading-value">{{ device.sd.toFixed(3) }}</div>
          <div class="reading-unit">m</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">{{ t('measure.hd') }}</div>
          <div class="reading-value">{{ hdCalc.toFixed(3) }}</div>
          <div class="reading-unit">m</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">{{ t('measure.vd') }}</div>
          <div class="reading-value">{{ vdCalc.toFixed(3) }}</div>
          <div class="reading-unit">m</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">E</div>
          <div class="reading-value" style="font-size: 1rem">{{ liveCoords.easting.toFixed(3) }}</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">N</div>
          <div class="reading-value" style="font-size: 1rem">{{ liveCoords.northing.toFixed(3) }}</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">H</div>
          <div class="reading-value" style="font-size: 1rem">{{ liveCoords.height.toFixed(3) }}</div>
        </div>
      </div>
    </div>

    <!-- Point List -->
    <PointList />
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'
import { useMeasurementStore } from '../../stores/measurement.js'
import { useStationStore } from '../../stores/station.js'
import PointList from './PointList.vue'

const { t } = useI18n()
const device = useDeviceStore()
const measureStore = useMeasurementStore()
const station = useStationStore()

const currentPointId = computed({
  get: () => String(measureStore.nextPointId),
  set: (v) => { measureStore.nextPointId = parseInt(v, 10) || measureStore.nextPointId },
})

const hdCalc = computed(() => {
  if (device.sd <= 0) return 0
  return device.sd * Math.sin(device.za * Math.PI / 200)
})

const vdCalc = computed(() => {
  if (device.sd <= 0) return 0
  return device.sd * Math.cos(device.za * Math.PI / 200)
})

const liveCoords = computed(() => {
  if (!station.isSet || device.sd <= 0) return { easting: 0, northing: 0, height: 0 }
  return station.computePoint(device.ha, device.za, device.sd, measureStore.targetHeight)
})

function measureAngle() {
  device.measureAngle()
}

function measureAndSave() {
  device.measureDistance()
  // The measurement listener in App.vue (or manual) will handle saving
  // For manual trigger, we also listen once:
  const handler = (data) => {
    if (data.sd > 0) {
      measureStore.addMeasuredPoint(data.ha, data.za, data.sd)
      device.off('measurement', handler)
    }
  }
  if (!measureStore.autoSave) {
    device.on('measurement', handler)
  }
}

function applyTargetType() {
  device.setTarget(measureStore.targetType, measureStore.prismConstant)
}
</script>
