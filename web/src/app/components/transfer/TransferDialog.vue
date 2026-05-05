<template>
  <div>
    <!-- Mode Selection -->
    <div class="card">
      <div class="card-title">📡 {{ t('transfer.title') }}</div>

      <div class="btn-row" style="margin-bottom: 12px">
        <button class="btn" :class="{ 'btn-primary': mode === 'import' }" @click="mode = 'import'">
          📥 {{ t('transfer.importFile') }}
        </button>
        <button class="btn" :class="{ 'btn-primary': mode === 'export' }" @click="mode = 'export'">
          📤 {{ t('transfer.exportFile') }}
        </button>
        <button class="btn" :class="{ 'btn-primary': mode === 'upload' }" @click="mode = 'upload'">
          ⬆ {{ t('transfer.upload') }}
        </button>
        <button class="btn" :class="{ 'btn-primary': mode === 'download' }" @click="mode = 'download'">
          ⬇ {{ t('transfer.download') }}
        </button>
      </div>
    </div>

    <!-- File Import -->
    <div v-if="mode === 'import'" class="card">
      <div class="form-group" style="margin-bottom: 10px">
        <label>{{ t('transfer.format') }}</label>
        <select v-model="importFormat">
          <option value="geojson">{{ t('formats.geojson') }}</option>
          <option value="csv">{{ t('formats.csv') }}</option>
          <option value="sdr33">{{ t('formats.sdr33') }}</option>
        </select>
      </div>

      <div class="form-group" style="margin-bottom: 10px">
        <label>Ziel</label>
        <select v-model="importTarget">
          <option value="control">Festpunkte (AP)</option>
          <option value="measured">Messpunkte</option>
        </select>
      </div>

      <input ref="importFileInput" type="file" :accept="acceptStr" style="display: none" @change="handleFileImport" />
      <button class="btn btn-primary btn-lg" @click="importFileInput?.click()">📂 Datei wählen</button>

      <div v-if="importResult" style="margin-top: 10px; color: var(--success)">
        {{ importResult }}
      </div>
    </div>

    <!-- File Export -->
    <div v-if="mode === 'export'" class="card">
      <div class="form-group" style="margin-bottom: 10px">
        <label>{{ t('transfer.format') }}</label>
        <select v-model="exportFormat">
          <option value="geojson">{{ t('formats.geojson') }}</option>
          <option value="csv">{{ t('formats.csv') }}</option>
          <option value="sdr33">{{ t('formats.sdr33') }}</option>
        </select>
      </div>

      <div class="form-group" style="margin-bottom: 10px">
        <label>Quelle</label>
        <select v-model="exportSource">
          <option value="measured">Messpunkte ({{ measureStore.points.length }})</option>
          <option value="control">Festpunkte ({{ measureStore.controlPoints.length }})</option>
        </select>
      </div>

      <button class="btn btn-primary btn-lg" @click="handleExport" :disabled="exportPointCount === 0">
        💾 {{ t('transfer.exportFile') }} ({{ exportPointCount }} Punkte)
      </button>
    </div>

    <!-- Serial Upload -->
    <div v-if="mode === 'upload'" class="card">
      <p style="margin-bottom: 10px; font-size: 0.85rem; color: var(--text-secondary)">
        Punkte als SDR33 zum Tachymeter senden.
      </p>
      <div class="form-group" style="margin-bottom: 10px">
        <label>Quelle</label>
        <select v-model="uploadSource">
          <option value="control">Festpunkte ({{ measureStore.controlPoints.length }})</option>
          <option value="measured">Messpunkte ({{ measureStore.points.length }})</option>
        </select>
      </div>
      <button class="btn btn-primary btn-lg" @click="handleUpload" :disabled="!device.connected || transferBusy">
        {{ transferBusy ? t('transfer.sending') : t('transfer.startTransfer') }}
      </button>
    </div>

    <!-- Serial Download -->
    <div v-if="mode === 'download'" class="card">
      <p style="margin-bottom: 10px; font-size: 0.85rem; color: var(--text-secondary)">
        SDR33-Daten vom Tachymeter empfangen.
      </p>
      <button class="btn btn-primary btn-lg" @click="handleDownload" :disabled="!device.connected || transferBusy">
        {{ transferBusy ? t('transfer.receiving') : t('transfer.startTransfer') }}
      </button>
      <div v-if="downloadBuffer" style="margin-top: 10px">
        <pre style="background: #000; padding: 8px; max-height: 200px; overflow-y: auto; font-size: 0.75rem; color: #0f0">{{ downloadBuffer }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'
import { useMeasurementStore } from '../../stores/measurement.js'
import { useProtocolStore } from '../../stores/protocol.js'
import { parseGeoJSON, parseCSV, pointsToGeoJSON, pointsToCSV, downloadFile, readFileAsText } from '../../utils/formats.js'
import { parseSdr33, buildExport } from '../../utils/sdr33.js'

const { t } = useI18n()
const device = useDeviceStore()
const measureStore = useMeasurementStore()
const protocol = useProtocolStore()

const mode = ref('import')
const importFormat = ref('geojson')
const importTarget = ref('control')
const exportFormat = ref('geojson')
const exportSource = ref('measured')
const uploadSource = ref('control')
const importResult = ref('')
const transferBusy = ref(false)
const downloadBuffer = ref('')
const importFileInput = ref(null)

const acceptStr = computed(() => {
  switch (importFormat.value) {
    case 'geojson': return '.geojson,.json'
    case 'csv': return '.csv,.txt'
    case 'sdr33': return '.sdr,.txt'
    default: return '*'
  }
})

const exportPointCount = computed(() => {
  return exportSource.value === 'measured'
    ? measureStore.points.length
    : measureStore.controlPoints.length
})

async function handleFileImport(event) {
  const file = event.target.files?.[0]
  if (!file) return

  try {
    const text = await readFileAsText(file)
    let points = []

    switch (importFormat.value) {
      case 'geojson':
        points = parseGeoJSON(text)
        break
      case 'csv':
        points = parseCSV(text)
        break
      case 'sdr33':
        points = parseSdr33(text)
        break
    }

    measureStore.importPoints(points, importTarget.value)
    importResult.value = `${points.length} Punkte importiert`
    protocol.log(`Datei-Import: ${file.name} → ${points.length} Punkte (${importTarget.value})`)
  } catch (e) {
    importResult.value = `Fehler: ${e.message}`
  }

  event.target.value = ''
}

function handleExport() {
  const pts = exportSource.value === 'measured'
    ? measureStore.points
    : measureStore.controlPoints

  if (pts.length === 0) return

  const timestamp = new Date().toISOString().slice(0, 10)
  let content, filename, mime

  switch (exportFormat.value) {
    case 'geojson':
      content = JSON.stringify(pointsToGeoJSON(pts), null, 2)
      filename = `punkte_${timestamp}.geojson`
      mime = 'application/json'
      break
    case 'csv':
      content = pointsToCSV(pts)
      filename = `punkte_${timestamp}.csv`
      mime = 'text/csv'
      break
    case 'sdr33':
      content = buildExport(pts)
      filename = `punkte_${timestamp}.sdr`
      mime = 'text/plain'
      break
  }

  downloadFile(content, filename, mime)
  protocol.log(`Export: ${filename} (${pts.length} Punkte)`)
}

async function handleUpload() {
  const pts = uploadSource.value === 'control'
    ? measureStore.controlPoints
    : measureStore.points

  if (pts.length === 0) return

  transferBusy.value = true
  try {
    const sdrData = buildExport(pts, 'WebUpload')
    device.startTransfer()
    device.transferSend(sdrData)

    // Wait for transfer-complete event
    await new Promise((resolve) => {
      const handler = () => {
        device.off('transfer-complete', handler)
        resolve()
      }
      device.on('transfer-complete', handler)
      // Timeout after 30s
      setTimeout(() => { device.off('transfer-complete', handler); resolve() }, 30000)
    })

    device.endTransfer()
    protocol.log(`Upload: ${pts.length} Punkte an Tachymeter gesendet`)
  } finally {
    transferBusy.value = false
  }
}

function handleDownload() {
  transferBusy.value = true
  downloadBuffer.value = ''

  device.startTransfer()

  const handler = (data) => {
    downloadBuffer.value += data

    // Check for ETX (end of transmission)
    if (data.includes('\x03')) {
      device.off('transfer-data', handler)
      device.endTransfer()

      // Parse received SDR33 data
      const points = parseSdr33(downloadBuffer.value)
      measureStore.importPoints(points, 'control')
      protocol.log(`Download: ${points.length} Punkte empfangen`)
      transferBusy.value = false
    }
  }

  device.on('transfer-data', handler)

  // Timeout after 60s
  setTimeout(() => {
    if (transferBusy.value) {
      device.off('transfer-data', handler)
      device.endTransfer()
      transferBusy.value = false

      if (downloadBuffer.value) {
        const points = parseSdr33(downloadBuffer.value)
        measureStore.importPoints(points, 'control')
      }
    }
  }, 60000)
}
</script>
