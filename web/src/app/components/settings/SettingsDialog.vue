<template>
  <div>
    <div class="card">
      <div class="card-title">⚙ {{ t('settings.title') }}</div>

      <!-- Language -->
      <div class="form-group" style="margin-bottom: 12px">
        <label>{{ t('settings.language') }}</label>
        <select v-model="settings.locale" @change="changeLocale">
          <option value="de">{{ t('settings.german') }}</option>
          <option value="en">{{ t('settings.english') }}</option>
        </select>
      </div>

      <!-- CRS -->
      <div class="form-group" style="margin-bottom: 12px">
        <label>{{ t('settings.crs') }}</label>
        <select v-model="settings.crs">
          <option v-for="c in crsOptions" :key="c.value" :value="c.value">{{ c.label }}</option>
        </select>
      </div>

      <!-- Theme -->
      <div class="form-group" style="margin-bottom: 12px">
        <label>{{ t('settings.theme') }}</label>
        <select v-model="settings.theme">
          <option value="dark">{{ t('settings.dark') }}</option>
          <option value="light">{{ t('settings.light') }}</option>
        </select>
      </div>

      <!-- Map tiles -->
      <div class="form-group" style="margin-bottom: 12px">
        <label>{{ t('settings.mapTiles') }}</label>
        <select v-model="settings.mapTiles">
          <option value="osm">{{ t('settings.osm') }}</option>
          <option value="satellite">{{ t('settings.satellite') }}</option>
        </select>
      </div>
    </div>

    <!-- Default values -->
    <div class="card">
      <div class="card-title">Standardwerte</div>

      <div class="form-row">
        <div class="form-group">
          <label>{{ t('station.instrumentHeight') }} [m]</label>
          <input type="number" v-model.number="settings.defaultIh" step="0.001" />
        </div>
        <div class="form-group">
          <label>{{ t('measure.targetHeight') }} [m]</label>
          <input type="number" v-model.number="settings.defaultTh" step="0.001" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>{{ t('measure.prismConstant') }}</label>
          <input type="number" v-model.number="settings.defaultPrismConst" />
        </div>
        <div class="form-group">
          <label>{{ t('connection.baudRate') }}</label>
          <select v-model.number="settings.defaultBaudRate">
            <option v-for="b in baudRates" :key="b" :value="b">{{ b }}</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Info -->
    <div class="card">
      <div class="card-title">Info</div>
      <div style="font-size: 0.8rem; color: var(--text-secondary)">
        <p><strong>Sokkia Web Control</strong> v1.0.0</p>
        <p>Webbasierte Steuerung für Sokkia-Tachymeter</p>
        <p style="margin-top: 8px">Unterstützte Formate: SDR33, GeoJSON, CSV</p>
        <p>Kommunikation: WebSocket → SerialPort Bridge</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
import { useSettingsStore } from '../../stores/settings.js'
import { CRS_OPTIONS, BAUD_RATES } from '../../utils/constants.js'

const { t, locale } = useI18n()
const settings = useSettingsStore()

const crsOptions = CRS_OPTIONS
const baudRates = BAUD_RATES

function changeLocale() {
  locale.value = settings.locale
  localStorage.setItem('sokkia-locale', settings.locale)
}
</script>
