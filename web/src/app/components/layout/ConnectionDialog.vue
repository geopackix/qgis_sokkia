<template>
  <div class="modal-backdrop" @click.self="$emit('close')">
    <div class="modal">
      <div class="modal-title">{{ t('connection.connect') }}</div>

      <div class="form-group" style="margin-bottom: 12px">
        <label>{{ t('connection.port') }}</label>
        <div style="display: flex; gap: 6px">
          <select v-model="selectedPort" style="flex: 1">
            <option value="">{{ t('connection.selectPort') }}</option>
            <option v-for="p in device.availablePorts" :key="p.path" :value="p.path">
              {{ p.path }} {{ p.manufacturer ? `(${p.manufacturer})` : '' }}
            </option>
          </select>
          <button class="btn btn-sm" @click="device.fetchPorts()">↻</button>
        </div>
      </div>

      <div class="form-group" style="margin-bottom: 16px">
        <label>{{ t('connection.baudRate') }}</label>
        <select v-model.number="selectedBaud">
          <option v-for="b in baudRates" :key="b" :value="b">{{ b }}</option>
        </select>
      </div>

      <div class="btn-row">
        <button
          v-if="!device.connected"
          class="btn btn-primary"
          style="flex: 1"
          :disabled="!selectedPort || device.connecting"
          @click="connect"
        >
          {{ device.connecting ? t('connection.connecting') : t('connection.connect') }}
        </button>
        <button
          v-else
          class="btn btn-danger"
          style="flex: 1"
          @click="disconnect"
        >
          {{ t('connection.disconnect') }}
        </button>
        <button class="btn" @click="$emit('close')">✕</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'
import { BAUD_RATES } from '../../utils/constants.js'

const { t } = useI18n()
const device = useDeviceStore()

const selectedPort = ref('')
const selectedBaud = ref(9600)
const baudRates = BAUD_RATES

defineEmits(['close'])

onMounted(() => {
  device.fetchPorts()
})

function connect() {
  device.connectDevice(selectedPort.value, selectedBaud.value)
}

function disconnect() {
  device.disconnectDevice()
}
</script>
