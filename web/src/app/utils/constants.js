/**
 * Application constants.
 */

export const TARGET_TYPES = {
  prism: { label: 'Prisma', command: '/C 0\r\n', defaultPC: -35 },
  reflective: { label: 'Reflektierendes Ziel', command: '/C 1\r\n', defaultPC: 0 },
  reflectorless: { label: 'Reflektorlos', command: '/C 2\r\n', defaultPC: 0 },
}

export const STEP_SIZES = [
  { label: '10 gon', value: 10 },
  { label: '1 gon', value: 1 },
  { label: '0.1 gon', value: 0.1 },
  { label: '0.01 gon', value: 0.01 },
  { label: '0.001 gon', value: 0.001 },
]

export const BAUD_RATES = [1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]

export const CRS_OPTIONS = [
  { label: 'EPSG:25832 (UTM 32N)', value: 'EPSG:25832' },
  { label: 'EPSG:25833 (UTM 33N)', value: 'EPSG:25833' },
  { label: 'EPSG:31466 (GK Zone 2)', value: 'EPSG:31466' },
  { label: 'EPSG:31467 (GK Zone 3)', value: 'EPSG:31467' },
  { label: 'EPSG:31468 (GK Zone 4)', value: 'EPSG:31468' },
  { label: 'EPSG:2056 (CH1903+)', value: 'EPSG:2056' },
  { label: 'EPSG:32632 (UTM 32N WGS84)', value: 'EPSG:32632' },
  { label: 'EPSG:32633 (UTM 33N WGS84)', value: 'EPSG:32633' },
  { label: 'EPSG:4326 (WGS84)', value: 'EPSG:4326' },
]

export const MAP_TILE_PROVIDERS = {
  osm: {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  },
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri',
    maxZoom: 18,
  },
}
