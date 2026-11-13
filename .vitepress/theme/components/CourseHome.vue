<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { modules, lessonCount } from '../../curriculum'

const filters = [
  { id: 'all', label: 'ทุกโมดูล' },
  { id: 'foundation', label: 'ข้อมูลและวิศวกรรม' },
  { id: 'research', label: 'แฟกเตอร์ (Factor) และโมเดล' },
  { id: 'portfolio', label: 'พอร์ตและการเทรดจริง (Live Trading)' }
]
const activeFilter = ref('all')
const visibleModules = computed(() => modules.filter(module => activeFilter.value === 'all' || module.stage === activeFilter.value))
const stages = [
  { number: '01—02', label: 'ฐานข้อมูลรากฐาน', detail: 'Engineering Stack / ความสอดคล้องเชิงจุดเวลา', module: modules[0] },
  { number: '03—05', label: 'วิจัยสัญญาณ', detail: 'การทดสอบแฟกเตอร์ / Machine Learning', module: modules[2] },
  { number: '06—07', label: 'การส่งคำสั่งพอร์ต', detail: 'Risk Budgeting / ข้อจำกัดการเทรด', module: modules[5] },
  { number: '08', label: 'การแจกแจงที่มา (Attribution) และมอนิเตอร์', detail: 'การแยกองค์ประกอบผลตอบแทน / ผลสะท้อนความเสี่ยง', module: modules[7] }
]
// An in-page module link must reveal its target even after filtering.
function resetFilter() { activeFilter.value = 'all' }
</script>

<template>
  <main class="course-home">
    <section class="course-hero" aria-labelledby="course-title">
      <div class="hero-copy">
        <div class="hero-kicker"><span class="kicker-line" /> QUANTML <span class="kicker-divider">/</span> RESEARCH SERIES</div>
        <h1 id="course-title"><span>คู่มือการวิจัยเชิงปริมาณ (Quant)</span></h1>
        <p class="hero-title-en" lang="en">Quantitative Research Handbook</p>
        <p class="hero-subtitle">ตั้งแต่ข้อมูล ฟีเจอร์ AI สู่ลูปปิดของกองทุนดัชนีเสริมผลตอบแทน (Index Enhancement) ในการเทรดจริง</p>
        <p class="hero-description">ร้อยเรียงองค์ความรู้ด้าน Quant ตามกระบวนการวิจัยที่ครบวงจร เชื่อมโยงตรรกะทางการเงิน ระเบียบวิธีทางคณิตศาสตร์ และการลงมือปฏิบัติจริงด้วย Python เข้าด้วยกัน</p>
        <div class="hero-actions">
          <a class="q-button primary" :href="withBase(`${modules[0].lessons[0].link}.html`)">เริ่มบทที่ 1 <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" /></svg></a>
          <a class="q-button secondary" href="#curriculum" @click="resetFilter">ดูโครงสร้างหลักสูตร <span aria-hidden="true">↓</span></a>
        </div>
        <div class="hero-audience"><span class="audience-dot" />เหมาะสำหรับผู้เรียนที่มีพื้นฐาน Python<span class="audience-separator">·</span>ต่อยอดการเงินและคณิตศาสตร์อย่างเป็นระบบ</div>
      </div>
      <div class="research-map" aria-label="กระบวนการวิจัย Quant">
        <div class="map-topline"><span>THE RESEARCH LOOP</span><span class="map-edition">A-SHARE / 01</span></div>
        <div class="map-heading"><h2>จากการวิจัยสู่การตัดสินใจ</h2><span>เส้นทางการวิจัยและการลงทุนที่ครบวงจร</span></div>
        <ol class="research-stages">
          <li v-for="(stage, index) in stages" :key="stage.label">
            <a :href="withBase(`${stage.module.lessons[0].link}.html`)">
              <span class="stage-symbol" aria-hidden="true"><svg viewBox="0 0 24 24"><template v-if="index === 0"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v14c0 4 14 4 14 0V5M5 12c0 4 14 4 14 0"/></template><template v-else-if="index === 1"><path d="M3 18 8 11l5 3 8-10M3 21h18"/><circle cx="8" cy="11" r="1"/><circle cx="13" cy="14" r="1"/></template><template v-else-if="index === 2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></template><template v-else><path d="M3 12h4l3-7 4 14 3-7h4"/></template></svg></span>
              <span class="stage-copy"><strong>{{ stage.label }}</strong><span>{{ stage.detail }}</span></span>
              <span class="stage-number">{{ stage.number }}</span>
            </a>
          </li>
        </ol>
        <div class="map-bottomline"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9a8 8 0 0 1 14-3l2 3M20 3v6h-6M20 15A8 8 0 0 1 6 18l-2-3M4 21v-6h6"/></svg><span>ตรวจสอบ · ส่งคำสั่ง · Attribution · ปรับปรุงซ้ำ</span><span class="map-dot" /></div>
      </div>
    </section>

    <section class="course-facts" aria-label="ภาพรวมหลักสูตร">
      <div class="course-fact"><strong>{{ String(modules.length).padStart(2, '0') }}</strong><span>โมดูลหลัก<small>กระบวนการวิจัยและการลงทุนครบวงจร</small></span></div>
      <div class="course-fact"><strong>{{ lessonCount }}</strong><span>บทเรียนอย่างเป็นระบบ<small>ตั้งแต่พื้นฐานสู่การปฏิบัติจริง</small></span></div>
      <div class="course-principle"><span class="fact-marker">01 / REPRODUCIBLE</span><strong>การวิจัยเชิงปฏิบัติที่รันโค้ดได้จริง</strong><span>ตัวอย่างโค้ด Python และการทดสอบวิธีวิทยา</span></div>
      <div class="course-principle"><span class="fact-marker">02 / OPEN KNOWLEDGE</span><strong>ระบบความรู้แบบเปิด</strong><span>ตำราเผยแพร่ทั่วไป งานวิจัย และเครื่องมือมาตรฐาน</span></div>
    </section>

    <section id="curriculum" class="curriculum-section" aria-labelledby="curriculum-heading">
      <div class="section-heading"><div><span class="eyebrow">THE CURRICULUM</span><h2 id="curriculum-heading">สร้างระบบความรู้ Quant อย่างครบวงจร</h2></div><p>ตั้งแต่ข้อมูลที่เชื่อถือได้ สัญญาณที่มีประสิทธิภาพ ไปจนถึงพอร์ตที่นำไปปฏิบัติได้จริง<br>เจาะลึกไปตามขั้นตอนการวิจัย หรือเลือกศึกษาตามหัวข้อที่สนใจ</p></div>
      <div class="curriculum-toolbar">
        <div class="curriculum-filters" role="group" aria-label="กรองโมดูลตามขั้นตอนการเรียนรู้"><button v-for="filter in filters" :key="filter.id" type="button" :aria-pressed="activeFilter === filter.id" @click="activeFilter = filter.id">{{ filter.label }}</button></div>
        <span class="filter-count" aria-live="polite">{{ visibleModules.length }} โมดูล · {{ visibleModules.reduce((n, m) => n + m.lessons.length, 0) }} บท</span>
      </div>
      <div class="module-grid">
        <article v-for="module in visibleModules" :id="`module-${module.id}`" :key="module.id" class="module-card">
          <div class="module-card-top"><span class="module-index">{{ module.id }}</span><span class="module-range">บทที่ {{ module.lessons[0].number }}–{{ module.lessons[module.lessons.length - 1].number }}</span></div>
          <span class="module-en">{{ module.en }}</span>
          <h3><a :href="withBase(`${module.lessons[0].link}.html`)">{{ module.title }}<span aria-hidden="true">↗</span></a></h3>
          <p>{{ module.description }}</p>
          <div class="module-tags"><span v-for="tag in module.tags" :key="tag">{{ tag }}</span></div>
          <details class="module-lessons"><summary><span>ดูบทเรียนในโมดูลนี้</span><span class="lesson-toggle"><span>{{ module.lessons.length }} บท</span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></span></summary>
            <ol><li v-for="lesson in module.lessons" :key="lesson.number"><a :href="withBase(`${lesson.link}.html`)"><span>{{ String(lesson.number).padStart(2, '0') }}</span><span>{{ lesson.title.replace(/^บทที่ \d+\s*[:：]\s*/, '') }}</span><span aria-hidden="true">↗</span></a></li></ol>
          </details>
        </article>
      </div>
    </section>

    <section class="learning-resources" aria-labelledby="resources-heading">
      <div class="resource-intro"><span class="eyebrow">BEYOND THE LESSONS</span><h2 id="resources-heading">ผสานความรู้ให้เป็นระบบ<br>สร้างงานวิจัยที่ทำซ้ำได้จริง</h2><p>ยึดหลักสูตรเป็นแกนหลัก เสริมพื้นฐานด้วยแหล่งข้อมูลเปิด<br>เปลี่ยนทุกบทเรียนให้เป็นทักษะการวิจัยที่ตรวจสอบได้จริง</p></div>
      <div class="resource-links">
        <a :href="withBase('/learning_path.html')"><span class="resource-index">01</span><span><strong>เส้นทางการเรียนรู้</strong><small>แบบทดสอบก่อนเรียน การปูพื้นฐาน และเกณฑ์การประเมินแต่ละบท</small></span><span aria-hidden="true">↗</span></a>
        <a :href="withBase('/resources.html')"><span class="resource-index">02</span><span><strong>แหล่งข้อมูลเปิดแนะนำ</strong><small>คอร์สเรียนมหาวิทยาลัย ตำราคลาสสิก เปเปอร์ต้นฉบับ และเอกสารทางการ</small></span><span aria-hidden="true">↗</span></a>
        <a :href="withBase('/README.html')"><span class="resource-index">03</span><span><strong>โค้ดและสภาพแวดล้อม</strong><small>การตั้งค่าสภาพแวดล้อมในเครื่อง เพื่อรันโค้ดตัวอย่างและการทดสอบ</small></span><span aria-hidden="true">↗</span></a>
      </div>
    </section>
    <div class="toolchain"><span>RESEARCH TOOLKIT</span><span>Python</span><span>Polars</span><span>DuckDB</span><span>LightGBM</span><span>CVXPY</span></div>
  </main>
</template>
