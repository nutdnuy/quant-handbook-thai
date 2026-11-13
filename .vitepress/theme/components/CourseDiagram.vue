<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import Mermaid from 'vitepress-plugin-mermaid/Mermaid.vue'

// The markdown plugin passes class="mermaid". Do not forward it to the
// viewer: Mermaid's automatic scan must never parse the toolbar as a graph.
defineOptions({ inheritAttrs: false })
const props = defineProps<{ graph: string; id: string }>()
const viewport = ref<HTMLElement>()
const canvas = ref<HTMLElement>()
const scale = ref(1)
const fit = ref(false)
const ready = ref(false)
const scrollLeft = ref(0)
const maxScroll = ref(0)
const percentage = computed(() => Math.round(scale.value * 100))
let svg: SVGSVGElement | null = null
let naturalWidth = 0
let naturalHeight = 0
let mutation: MutationObserver | undefined
let resize: ResizeObserver | undefined
let disposed = false

function measureScroll() {
  const el = viewport.value
  if (!el) return
  maxScroll.value = Math.max(0, el.scrollWidth - el.clientWidth)
  scrollLeft.value = el.scrollLeft
}
function applySize() {
  if (!svg || !naturalWidth || !viewport.value) return
  if (fit.value) scale.value = Math.min(1, (viewport.value.clientWidth - 32) / naturalWidth)
  svg.style.width = `${naturalWidth * scale.value}px`
  svg.style.height = `${naturalHeight * scale.value}px`
  measureScroll()
}
function readDiagram() {
  const next = canvas.value?.querySelector<SVGSVGElement>(':scope > .mermaid > svg')
  if (!next || next === svg) return
  svg = next
  const box = svg.viewBox.baseVal
  naturalWidth = box.width
  naturalHeight = box.height
  ready.value = naturalWidth > 0 && naturalHeight > 0
  applySize()
}
function zoom(value: number) {
  fit.value = false
  scale.value = Math.max(.15, Math.min(2, value))
  applySize()
}
function fitDiagram() {
  fit.value = true
  applySize()
}
function pan(value: number) {
  if (!viewport.value) return
  viewport.value.scrollLeft = value
  measureScroll()
}
function step(direction: number) {
  pan(scrollLeft.value + direction * (viewport.value?.clientWidth ?? 300) * .75)
}
function onKey(event: KeyboardEvent) {
  if (event.target !== viewport.value) return
  if (event.key === 'ArrowLeft') step(-1)
  else if (event.key === 'ArrowRight') step(1)
  else if (event.key === 'Home') pan(0)
  else if (event.key === 'End') pan(maxScroll.value)
  else return
  event.preventDefault()
}
onMounted(async () => {
  await nextTick()
  if (disposed || !canvas.value || !viewport.value) return
  mutation = new MutationObserver(readDiagram)
  mutation.observe(canvas.value, { childList: true, subtree: true })
  resize = new ResizeObserver(() => { applySize(); measureScroll() })
  resize.observe(viewport.value)
  readDiagram()
})
onBeforeUnmount(() => { disposed = true; mutation?.disconnect(); resize?.disconnect() })
</script>

<template>
  <figure class="course-diagram" aria-label="แผนภาพบทเรียน">
    <div class="diagram-toolbar">
      <span class="diagram-label">แผนภาพบทเรียน</span>
      <div class="diagram-tools" role="group" aria-label="การซูมแผนภาพ">
        <button type="button" aria-label="ย่อแผนภาพ" :disabled="!ready || scale <= .15" @click="zoom(scale - .15)">−</button>
        <button type="button" class="diagram-scale" aria-label="คืนค่าขนาดเดิมของแผนภาพ" :disabled="!ready" @click="zoom(1)">{{ percentage }}%</button>
        <button type="button" aria-label="ขยายแผนภาพ" :disabled="!ready || scale >= 2" @click="zoom(scale + .15)">＋</button>
        <button type="button" class="diagram-fit" :aria-pressed="fit" :disabled="!ready" @click="fitDiagram">เต็มภาพ</button>
      </div>
    </div>
    <div ref="viewport" class="diagram-viewport" tabindex="0" role="region" aria-label="พื้นที่เลื่อนแผนภาพ สามารถใช้ปุ่มลูกศรซ้ายขวา ปุ่ม Home และปุ่ม End ได้" @scroll="measureScroll" @keydown="onKey">
      <div ref="canvas" class="diagram-canvas"><Mermaid :id="props.id" :graph="props.graph" /></div>
    </div>
    <div v-if="maxScroll > 1" class="diagram-navigation">
      <button type="button" aria-label="เลื่อนดูแผนภาพไปทางซ้าย" :disabled="scrollLeft <= 1" @click="step(-1)">←</button>
      <input type="range" min="0" :max="maxScroll" :value="scrollLeft" aria-label="ตำแหน่งแนวนอนของแผนภาพ" @input="pan(Number(($event.target as HTMLInputElement).value))">
      <button type="button" aria-label="เลื่อนดูแผนภาพไปทางขวา" :disabled="scrollLeft >= maxScroll - 1" @click="step(1)">→</button>
    </div>
    <figcaption>{{ maxScroll > 1 ? 'เลื่อนซ้ายขวาหรือลากแถบเลื่อนเพื่อดูแผนภาพฉบับเต็ม คลิก「เต็มภาพ」เพื่อดูภาพรวมทั้งหมด' : 'สามารถขยายเพื่อดูรายละเอียด คลิกที่เปอร์เซ็นต์เพื่อคืนค่าขนาดเดิม' }}</figcaption>
  </figure>
</template>
