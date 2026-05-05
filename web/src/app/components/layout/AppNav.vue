<template>
  <nav class="app-nav">
    <button
      v-for="tab in tabs"
      :key="tab.id"
      :class="{ active: modelValue === tab.id }"
      @click="$emit('update:modelValue', tab.id)"
    >
      <span class="nav-icon">{{ tab.icon }}</span>
      <span class="nav-label">{{ t(`nav.${tab.id}`) }}</span>
    </button>
  </nav>
</template>

<script setup>
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

defineProps({ modelValue: String })
defineEmits(['update:modelValue'])

const tabs = [
  { id: 'map', icon: '🗺' },
  { id: 'station', icon: '⊕' },
  { id: 'measure', icon: '⌖' },
  { id: 'stakeout', icon: '🎯' },
  { id: 'remote', icon: '🕹' },
  { id: 'transfer', icon: '📡' },
  { id: 'protocol', icon: '📋' },
  { id: 'settings', icon: '⚙' },
]
</script>
