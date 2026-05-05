<template>
  <header class="app-header">
    <div class="logo">
      <span>📡</span>
      <span>{{ t('app.title') }}</span>
    </div>
    <div class="status-group">
      <span class="conn-badge" :class="device.statusText">
        <span v-if="device.connected">●</span>
        <span v-else>○</span>
        {{ t(`connection.${device.statusText}`) }}
      </span>
      <button class="btn btn-sm" @click="$emit('openConnect')">
        {{ device.connected ? t('connection.disconnect') : t('connection.connect') }}
      </button>
    </div>
  </header>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'

const { t } = useI18n()
const device = useDeviceStore()

defineEmits(['openConnect'])
</script>
