import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useProtocolStore } from './protocol.js'

export const useDeviceStore = defineStore('device', () => {
  // Connection state
  const connected = ref(false)
  const connecting = ref(false)
  const portPath = ref('')
  const baudRate = ref(9600)
  const availablePorts = ref([])
  const firmwareVersion = ref('')

  // WebSocket
  let ws = null
  const wsConnected = ref(false)

  // Measurement values (live from instrument)
  const ha = ref(0)     // Hz angle [gon]
  const za = ref(0)     // Zenith angle [gon]
  const sd = ref(0)     // Slope distance [m]

  // Laser state
  const laserOn = ref(false)

  // Event listeners registered by other stores/components
  const listeners = new Map()

  function on(event, callback) {
    if (!listeners.has(event)) listeners.set(event, [])
    listeners.get(event).push(callback)
  }

  function off(event, callback) {
    if (!listeners.has(event)) return
    const cbs = listeners.get(event)
    const idx = cbs.indexOf(callback)
    if (idx !== -1) cbs.splice(idx, 1)
  }

  function emit(event, data) {
    const cbs = listeners.get(event) || []
    cbs.forEach(cb => cb(data))
  }

  let wsReconnectDelay = 2000

  // WebSocket connection to backend
  function connectWs() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    ws = new WebSocket(`${protocol}//${host}/ws`)

    ws.onopen = () => {
      wsConnected.value = true
      wsReconnectDelay = 2000 // reset on success
      console.log('[WS] Connected to backend')
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        handleMessage(msg)
      } catch (e) {
        console.error('[WS] Parse error:', e)
      }
    }

    ws.onclose = () => {
      wsConnected.value = false
      connected.value = false
      // Exponential backoff: 2s → 4s → 8s → max 30s
      wsReconnectDelay = Math.min(wsReconnectDelay * 2, 30000)
      console.log(`[WS] Disconnected. Reconnecting in ${wsReconnectDelay / 1000}s...`)
      setTimeout(connectWs, wsReconnectDelay)
    }

    ws.onerror = () => {
      // Will trigger onclose
    }
  }

  function handleMessage(msg) {
    const protocol = useProtocolStore()

    switch (msg.type) {
      case 'status':
        connected.value = !!msg.payload.connected
        connecting.value = false
        if (msg.payload.error) {
          protocol.log(`Fehler: ${msg.payload.error}`, 'error')
        }
        break

      case 'serial-data':
        if (msg.payload.type === 'measurement') {
          ha.value = msg.payload.ha
          za.value = msg.payload.za
          sd.value = msg.payload.sd
          emit('measurement', { ha: msg.payload.ha, za: msg.payload.za, sd: msg.payload.sd })
          protocol.log(`Messwert: Hz=${msg.payload.ha.toFixed(4)} V=${msg.payload.za.toFixed(4)} SD=${msg.payload.sd.toFixed(3)}`)
        } else if (msg.payload.type === 'transfer') {
          emit('transfer-data', msg.payload.raw)
        } else {
          emit('raw-data', msg.payload.raw)
          protocol.log(`RAW: ${msg.payload.raw}`)
        }
        break

      case 'transfer-complete':
        emit('transfer-complete', null)
        break

      case 'error':
        protocol.log(`Server-Fehler: ${msg.payload}`, 'error')
        break
    }
  }

  function wsSend(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg))
    }
  }

  // Serial commands
  async function connectDevice(port, baud) {
    connecting.value = true
    wsSend({ type: 'connect', payload: { port, baudRate: baud } })
  }

  async function disconnectDevice() {
    wsSend({ type: 'disconnect' })
  }

  async function fetchPorts() {
    try {
      const res = await fetch('/api/ports')
      availablePorts.value = await res.json()
    } catch {
      availablePorts.value = []
    }
  }

  // Instrument commands
  function sendCommand(cmd) {
    wsSend({ type: 'send', payload: cmd })
  }

  function sendBytes(bytes) {
    wsSend({ type: 'send-bytes', payload: Array.from(bytes) })
  }

  function measureDistance() {
    sendBytes(new Uint8Array([0x11]))
    useProtocolStore().log('Streckenmessung ausgelöst')
  }

  function measureAngle() {
    sendBytes(new Uint8Array([0x13]))
  }

  function stopMeasurement() {
    sendBytes(new Uint8Array([0x12]))
  }

  function toggleLaser() {
    if (laserOn.value) {
      sendCommand('*/PF 2,1\r\n')
      sendCommand('*GLOFF\r\n')
      laserOn.value = false
    } else {
      sendCommand('*/PF 2,1\r\n')
      sendCommand('*GLON\r\n')
      laserOn.value = true
    }
  }

  function setTarget(type, prismConstant = 0) {
    const cmds = { prism: '/C 0\r\n', reflective: '/C 1\r\n', reflectorless: '/C 2\r\n' }
    sendCommand(cmds[type] || cmds.prism)
    sendCommand(`/B 0,0,0,${prismConstant},1,0,0,0,0,0,0,0\r\n`)
    useProtocolStore().log(`Zieltyp: ${type}, PK=${prismConstant} mm`)
  }

  /**
   * Motor drive to Hz/ZA angles.
   * Encodes gon values as 7-digit integers (implicit 4 decimal places).
   */
  function driveTo(hzGon, zaGon) {
    const hzStr = Math.round(hzGon * 10000).toString().padStart(7, '0')
    const zaStr = Math.round(zaGon * 10000).toString().padStart(7, '0')
    sendCommand(`*DHA${hzStr}VA${zaStr}\r\n`)
  }

  function startTransfer() {
    wsSend({ type: 'transfer-start' })
  }

  function endTransfer() {
    wsSend({ type: 'transfer-end' })
  }

  function transferSend(data) {
    wsSend({ type: 'transfer-send', payload: data })
  }

  // Computed
  const statusText = computed(() => {
    if (connecting.value) return 'connecting'
    if (connected.value) return 'connected'
    return 'disconnected'
  })

  // Initialize WebSocket on store creation
  connectWs()

  return {
    connected, connecting, portPath, baudRate, availablePorts, firmwareVersion,
    wsConnected, ha, za, sd, laserOn, statusText,
    on, off, emit,
    connectDevice, disconnectDevice, fetchPorts,
    sendCommand, sendBytes,
    measureDistance, measureAngle, stopMeasurement,
    toggleLaser, setTarget, driveTo,
    startTransfer, endTransfer, transferSend,
  }
})
