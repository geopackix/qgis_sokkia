import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useProtocolStore = defineStore('protocol', () => {
  const entries = ref([])
  const maxEntries = 5000

  function log(message, level = 'info') {
    const entry = {
      timestamp: new Date(),
      message,
      level, // 'info' | 'warning' | 'error' | 'measurement'
    }
    entries.value.push(entry)

    // Trim old entries
    if (entries.value.length > maxEntries) {
      entries.value = entries.value.slice(-maxEntries)
    }
  }

  function clear() {
    entries.value = []
    log('Protokoll geleert')
  }

  function exportTxt() {
    const lines = entries.value.map(e => {
      const ts = e.timestamp.toLocaleString('de-DE')
      const lvl = e.level === 'info' ? '' : ` [${e.level.toUpperCase()}]`
      return `[${ts}]${lvl} ${e.message}`
    })

    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const now = new Date()
    const dateStr = now.toISOString().slice(0, 10).replace(/-/g, '')
    const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '')
    a.download = `Protokoll_${dateStr}_${timeStr}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return { entries, log, clear, exportTxt }
})
