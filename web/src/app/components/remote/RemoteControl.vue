<template>
  <div>
    <div class="card">
      <div class="card-title">🕹 {{ t('remote.title') }}</div>

      <!-- Step size selector -->
      <div class="form-row" style="margin-bottom: 16px">
        <div class="form-group">
          <label>{{ t('remote.stepSize') }}</label>
          <select v-model.number="stepSize">
            <option v-for="s in stepSizes" :key="s.value" :value="s.value">{{ s.label }}</option>
          </select>
        </div>
      </div>

      <!-- Joystick -->
      <div class="joystick">
        <!-- Row 1 -->
        <div></div>
        <button class="joy-btn" @pointerdown="startRepeat('v', -1)" @pointerup="stopRepeat" @pointerleave="stopRepeat">▲</button>
        <div></div>
        <!-- Row 2 -->
        <button class="joy-btn" @pointerdown="startRepeat('h', -1)" @pointerup="stopRepeat" @pointerleave="stopRepeat">◀</button>
        <button class="joy-btn joy-center" @click="device.measureDistance()">⌖</button>
        <button class="joy-btn" @pointerdown="startRepeat('h', 1)" @pointerup="stopRepeat" @pointerleave="stopRepeat">▶</button>
        <!-- Row 3 -->
        <div></div>
        <button class="joy-btn" @pointerdown="startRepeat('v', 1)" @pointerup="stopRepeat" @pointerleave="stopRepeat">▼</button>
        <div></div>
      </div>

      <!-- Quick actions -->
      <div class="btn-row" style="margin-top: 16px; justify-content: center">
        <button class="btn" :class="{ 'btn-warning': device.laserOn }" @click="device.toggleLaser()">
          {{ device.laserOn ? t('remote.laserOff') : t('remote.laserOn') }}
        </button>
      </div>
    </div>

    <!-- Live angle display -->
    <div class="card">
      <div class="readings-grid">
        <div class="reading-cell reading-big">
          <div class="reading-label">Hz</div>
          <div class="reading-value">{{ device.ha.toFixed(4) }}</div>
          <div class="reading-unit">gon</div>
        </div>
        <div class="reading-cell reading-big">
          <div class="reading-label">V</div>
          <div class="reading-value">{{ device.za.toFixed(4) }}</div>
          <div class="reading-unit">gon</div>
        </div>
        <div class="reading-cell">
          <div class="reading-label">SD</div>
          <div class="reading-value">{{ device.sd.toFixed(3) }}</div>
          <div class="reading-unit">m</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDeviceStore } from '../../stores/device.js'
import { STEP_SIZES } from '../../utils/constants.js'

const { t } = useI18n()
const device = useDeviceStore()

const stepSize = ref(0.1)
const stepSizes = STEP_SIZES

let repeatTimer = null

function move(axis, direction) {
  let ha = device.ha
  let za = device.za

  if (axis === 'h') {
    ha = ((ha + direction * stepSize.value) % 400 + 400) % 400
  } else {
    za = ((za + direction * stepSize.value) % 400 + 400) % 400
  }

  device.driveTo(ha, za)

  // Request angle measurement after motor command
  setTimeout(() => device.measureAngle(), 300)
}

function startRepeat(axis, direction) {
  move(axis, direction)
  repeatTimer = setInterval(() => move(axis, direction), 400)
}

function stopRepeat() {
  if (repeatTimer) {
    clearInterval(repeatTimer)
    repeatTimer = null
  }
}

onUnmounted(() => stopRepeat())
</script>
