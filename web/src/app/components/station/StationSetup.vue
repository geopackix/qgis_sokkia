<template>
  <div>
    <!-- Mode selection -->
    <div class="card">
      <div class="card-title">⊕ {{ t('station.title') }}</div>
      <div class="btn-row" style="margin-bottom: 12px">
        <button class="btn" :class="{ 'btn-primary': mode === 'manual' }" @click="mode = 'manual'">
          {{ t('station.manualStation') }}
        </button>
        <button class="btn" :class="{ 'btn-primary': mode === 'free' }" @click="mode = 'free'">
          {{ t('station.freeStation') }}
        </button>
      </div>
    </div>

    <!-- Manual Station -->
    <div v-if="mode === 'manual'" class="card">
      <div class="form-row">
        <div class="form-group">
          <label>{{ t('station.stationId') }}</label>
          <input v-model="station.id" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>{{ t('station.easting') }}</label>
          <input type="number" v-model.number="stE" step="0.001" />
        </div>
        <div class="form-group">
          <label>{{ t('station.northing') }}</label>
          <input type="number" v-model.number="stN" step="0.001" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>{{ t('station.height') }}</label>
          <input type="number" v-model.number="stH" step="0.001" />
        </div>
        <div class="form-group">
          <label>{{ t('station.instrumentHeight') }}</label>
          <input type="number" v-model.number="stIh" step="0.001" />
        </div>
      </div>

      <!-- Orientation via control point -->
      <div class="card" style="margin-top: 8px">
        <div class="toggle-row">
          <input type="checkbox" v-model="useAP" id="use-ap" />
          <label for="use-ap">{{ t('station.orientationAP') }}</label>
        </div>
        <div v-if="useAP" class="form-row">
          <div class="form-group">
            <label>AP E</label>
            <input type="number" v-model.number="apE" step="0.001" />
          </div>
          <div class="form-group">
            <label>AP N</label>
            <input type="number" v-model.number="apN" step="0.001" />
          </div>
        </div>
        <div v-if="!useAP" class="form-row">
          <div class="form-group">
            <label>{{ t('station.orientation') }} [gon]</label>
            <input :value="station.orientationGon.toFixed(4)" readonly style="opacity: 0.6" />
          </div>
          <button class="btn btn-sm" @click="station.zeroOrientation()">{{ t('station.zeroOrientation') }}</button>
        </div>
      </div>

      <button class="btn btn-primary btn-lg" @click="setManualStation" style="margin-top: 10px">
        ✓ {{ t('station.setStation') }}
      </button>
    </div>

    <!-- Free Stationing -->
    <div v-if="mode === 'free'">
      <ResectionPanel />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useStationStore } from '../../stores/station.js'
import { useSettingsStore } from '../../stores/settings.js'
import ResectionPanel from './ResectionPanel.vue'

const { t } = useI18n()
const station = useStationStore()
const settings = useSettingsStore()

const mode = ref('manual')
const stE = ref(0)
const stN = ref(0)
const stH = ref(0)
const stIh = ref(settings.defaultIh)
const useAP = ref(false)
const apE = ref(0)
const apN = ref(0)

function setManualStation() {
  station.setStation({
    id: station.id,
    easting: stE.value,
    northing: stN.value,
    height: stH.value,
    instrumentHeight: stIh.value,
  })

  if (useAP.value) {
    station.setOrientationFromAP(apE.value, apN.value)
  }
}
</script>
