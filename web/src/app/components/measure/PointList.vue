<template>
  <div class="card">
    <div class="card-title">{{ t('points.title') }}</div>

    <div class="toolbar">
      <input
        type="text"
        v-model="searchQuery"
        :placeholder="t('points.search')"
        style="flex: 1; min-width: 100px"
      />
      <button class="btn btn-sm" @click="triggerImport">{{ t('points.import') }}</button>
      <button class="btn btn-sm" @click="exportPoints">{{ t('points.export') }}</button>
      <button class="btn btn-sm btn-danger" @click="clearPoints">{{ t('points.clear') }}</button>
      <input ref="fileInput" type="file" accept=".geojson,.json,.csv,.txt,.sdr" style="display: none" @change="handleImport" />
    </div>

    <div class="scroll-y" style="max-height: 250px">
      <table v-if="filteredPoints.length > 0">
        <thead>
          <tr>
            <th>{{ t('points.id') }}</th>
            <th>{{ t('points.e') }}</th>
            <th>{{ t('points.n') }}</th>
            <th>{{ t('points.z') }}</th>
            <th>{{ t('points.desc') }}</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(p, i) in filteredPoints" :key="i">
            <td>{{ p.id }}</td>
            <td>{{ p.easting.toFixed(3) }}</td>
            <td>{{ p.northing.toFixed(3) }}</td>
            <td>{{ (p.height || 0).toFixed(3) }}</td>
            <td>{{ p.description || p.code || '' }}</td>
            <td>
              <button class="btn btn-sm btn-danger" @click="measureStore.removePoint(i)">✕</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else style="color: var(--text-muted); text-align: center; padding: 20px">
        {{ t('points.noPoints') }}
      </div>
    </div>

    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px">
      {{ measureStore.points.length }} {{ t('points.title') }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMeasurementStore } from '../../stores/measurement.js'
import { parseGeoJSON, parseCSV, pointsToGeoJSON, pointsToCSV, downloadFile, readFileAsText } from '../../utils/formats.js'
import { parseSdr33 } from '../../utils/sdr33.js'

const { t } = useI18n()
const measureStore = useMeasurementStore()

const searchQuery = ref('')
const fileInput = ref(null)

const filteredPoints = computed(() => {
  const q = searchQuery.value.toLowerCase()
  if (!q) return measureStore.sortedPoints
  return measureStore.sortedPoints.filter(p =>
    p.id.toLowerCase().includes(q) || (p.description || '').toLowerCase().includes(q)
  )
})

function triggerImport() {
  fileInput.value?.click()
}

async function handleImport(event) {
  const file = event.target.files?.[0]
  if (!file) return

  try {
    const text = await readFileAsText(file)
    let points = []

    if (file.name.endsWith('.geojson') || file.name.endsWith('.json')) {
      points = parseGeoJSON(text)
    } else if (file.name.endsWith('.sdr')) {
      points = parseSdr33(text)
    } else {
      // CSV / TXT
      points = parseCSV(text)
    }

    measureStore.importPoints(points, 'measured')
  } catch (e) {
    alert(`Import-Fehler: ${e.message}`)
  }

  // Reset file input
  event.target.value = ''
}

function exportPoints() {
  if (measureStore.points.length === 0) return

  const geojson = pointsToGeoJSON(measureStore.points)
  downloadFile(JSON.stringify(geojson, null, 2), 'punkte.geojson', 'application/json')
}

function clearPoints() {
  if (confirm(t('protocol.clearConfirm'))) {
    measureStore.clearPoints()
  }
}
</script>
