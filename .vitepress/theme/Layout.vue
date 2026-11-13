<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { modules, lessonCount } from '../curriculum'

const { page, frontmatter } = useData()
const currentModule = computed(() => modules.find(module => module.lessons.some(
  lesson => `${lesson.link.slice(1)}.md` === page.value.relativePath
)))
const currentLesson = computed(() => currentModule.value?.lessons.find(
  lesson => `${lesson.link.slice(1)}.md` === page.value.relativePath
))
</script>

<template>
  <DefaultTheme.Layout>
    <template #nav-bar-title-after><span class="brand-caption">คู่มือการวิจัยเชิงปริมาณ (Quant)</span></template>
    <template #sidebar-nav-before>
      <a class="sidebar-heading" :href="withBase('/#curriculum')">
        <span class="eyebrow">COURSE HANDBOOK</span>
        <span>สารบัญหลักสูตร <span class="sidebar-count">{{ modules.length }} โมดูล / {{ lessonCount }} บท</span></span>
      </a>
    </template>
    <template #doc-before>
      <div v-if="currentModule && currentLesson" class="lesson-context">
        <nav class="lesson-breadcrumb" aria-label="ตำแหน่งปัจจุบัน">
          <a :href="withBase('/#curriculum')">ภาพรวมหลักสูตร</a>
          <span aria-hidden="true">/</span>
          <a :href="withBase(`/#module-${currentModule.id}`)">{{ currentModule.title }}</a>
        </nav>
        <div class="lesson-meta">
          <span class="lesson-number">LECTURE {{ String(currentLesson.number).padStart(2, '0') }}</span>
          <span>ทั้งหมด {{ lessonCount }} บท</span>
          <span v-if="frontmatter.reading_time" class="reading-time">เวลาอ่านโดยประมาณ {{ frontmatter.reading_time }}</span>
        </div>
      </div>
      <div v-else class="guide-context"><span class="eyebrow">QUANTML / LEARNING GUIDE</span></div>
    </template>
    <template #aside-outline-before>
      <div v-if="currentModule" class="aside-module">
        <span class="eyebrow">MODULE {{ currentModule.id }}</span>
        <strong>{{ currentModule.title }}</strong>
        <span>{{ currentModule.lessons.length }} บท · {{ currentModule.en }}</span>
      </div>
    </template>
    <template #aside-outline-after>
      <div class="aside-resources">
        <span>เอกสารประกอบการเรียน</span>
        <a :href="withBase('/resources.html')">ตำราเรียนสาธารณะและเอกสารเครื่องมือ <span aria-hidden="true">↗</span></a>
        <a :href="withBase('/learning_path.html')">เส้นทางการเรียนรู้และเกณฑ์วัดความรู้ <span aria-hidden="true">↗</span></a>
      </div>
    </template>
  </DefaultTheme.Layout>
</template>
