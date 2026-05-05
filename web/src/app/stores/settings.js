import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useSettingsStore = defineStore('settings', () => {
  const crs = ref(localStorage.getItem('sokkia-crs') || 'EPSG:25832')
  const theme = ref(localStorage.getItem('sokkia-theme') || 'dark')
  const locale = ref(localStorage.getItem('sokkia-locale') || 'de')
  const mapTiles = ref(localStorage.getItem('sokkia-tiles') || 'osm')
  const defaultBaudRate = ref(parseInt(localStorage.getItem('sokkia-baud') || '9600', 10))
  const defaultIh = ref(parseFloat(localStorage.getItem('sokkia-ih') || '1.600'))
  const defaultTh = ref(parseFloat(localStorage.getItem('sokkia-th') || '2.000'))
  const defaultPrismConst = ref(parseInt(localStorage.getItem('sokkia-pc') || '-35', 10))

  // Persist settings
  watch(crs, v => localStorage.setItem('sokkia-crs', v))
  watch(theme, v => {
    localStorage.setItem('sokkia-theme', v)
    document.documentElement.setAttribute('data-theme', v)
  })
  watch(locale, v => localStorage.setItem('sokkia-locale', v))
  watch(mapTiles, v => localStorage.setItem('sokkia-tiles', v))
  watch(defaultBaudRate, v => localStorage.setItem('sokkia-baud', String(v)))
  watch(defaultIh, v => localStorage.setItem('sokkia-ih', String(v)))
  watch(defaultTh, v => localStorage.setItem('sokkia-th', String(v)))
  watch(defaultPrismConst, v => localStorage.setItem('sokkia-pc', String(v)))

  // Apply theme on init
  document.documentElement.setAttribute('data-theme', theme.value)

  return {
    crs, theme, locale, mapTiles,
    defaultBaudRate, defaultIh, defaultTh, defaultPrismConst,
  }
})
