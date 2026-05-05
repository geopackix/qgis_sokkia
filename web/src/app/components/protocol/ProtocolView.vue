<template>
  <div>
    <div class="card">
      <div class="card-title">📋 {{ t('protocol.title') }}</div>
      <div class="toolbar">
        <button class="btn btn-sm" @click="protocol.exportTxt()">💾 {{ t('protocol.export') }}</button>
        <button class="btn btn-sm btn-danger" @click="clearProtocol">🗑 {{ t('protocol.clear') }}</button>
      </div>
    </div>

    <div class="protocol-log" ref="logContainer">
      <template v-if="protocol.entries.length > 0">
        <div
          v-for="(entry, i) in protocol.entries"
          :key="i"
          :class="`entry-${entry.level}`"
        >
          <span style="color: #666">[{{ formatTime(entry.timestamp) }}]</span> {{ entry.message }}
        </div>
      </template>
      <div v-else style="color: var(--text-muted)">{{ t('protocol.noEntries') }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { useProtocolStore } from '../../stores/protocol.js'

const { t } = useI18n()
const protocol = useProtocolStore()

const logContainer = ref(null)

// Auto-scroll to bottom
watch(
  () => protocol.entries.length,
  () => {
    nextTick(() => {
      if (logContainer.value) {
        logContainer.value.scrollTop = logContainer.value.scrollHeight
      }
    })
  }
)

function formatTime(date) {
  return date.toLocaleTimeString('de-DE')
}

function clearProtocol() {
  if (confirm(t('protocol.clearConfirm'))) {
    protocol.clear()
  }
}
</script>

<style scoped>
.protocol-log {
  height: calc(100vh - 260px);
  min-height: 200px;
}
</style>
