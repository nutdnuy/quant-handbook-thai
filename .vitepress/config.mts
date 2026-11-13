import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { modules } from './curriculum'

export default withMermaid(
  defineConfig({
    lang: 'th-TH',
    base: '/quant-handbook-thai/',
    title: 'คู่มือวิจัย Quant (ฉบับภาษาไทย)',
    description: 'ตั้งแต่ข้อมูล ฟีเจอร์ และ AI สู่ลูปการเทรดจริง (Live Trading) ของกองทุนดัชนีเสริมผลตอบแทน (Index Enhancement) แบบครบวงจร',
    
    appearance: 'dark',
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: '/quant-handbook-thai/favicon.svg' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
      ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap' }]
    ],
    ignoreDeadLinks: true,

    // เปิดใช้งานสูตรคณิตศาสตร์ในตัว (MathJax) พร้อมเลขบรรทัดโค้ด
    markdown: {
      math: true,
      lineNumbers: true,
      config(md) {
        const renderFence = md.renderer.rules.fence!
        md.renderer.rules.fence = (...args) => {
          const rendered = renderFence(...args)
          return args[0][args[1]].info.trim() === 'mermaid'
            ? rendered.replace(/<Mermaid\b/g, '<CourseDiagram').replace(/<\/Mermaid>/g, '</CourseDiagram>')
            : rendered
        }
        // Timeline uses ':' as a separator. Format clock punctuation for display
        // before Mermaid receives it, without changing the lesson source.
        md.core.ruler.after('block', 'timeline-clock-labels', state => {
          for (const token of state.tokens) {
            if (token.type === 'fence' && token.info.trim() === 'mermaid' && /^timeline\s/m.test(token.content)) {
              token.content = token.content.replace(/\b(\d{2}):(\d{2})\b/g, '$1：$2')
            }
          }
        })
      }
    },

    mermaid: {
      theme: 'base',
      themeVariables: {
        primaryColor: '#eaf2ff', primaryTextColor: '#182337', primaryBorderColor: '#2f80ed',
        lineColor: '#61748f', secondaryColor: '#e7f7f4', tertiaryColor: '#f3f6fb',
        fontFamily: 'IBM Plex Sans, IBM Plex Sans Thai, Noto Sans Thai, Noto Sans SC, sans-serif'
      },
      flowchart: { useMaxWidth: false },
      sequence: { useMaxWidth: false },
      timeline: { useMaxWidth: false }
    },

    vite: {
      optimizeDeps: {
        include: ['mermaid', 'fastdom']
      }
    },

    themeConfig: {
      siteTitle: 'QuantML',
      logo: { src: '/favicon.svg', alt: '' },
      sidebarMenuLabel: 'สารบัญบทเรียน',
      returnToTopLabel: 'กลับสู่ด้านบน',
      skipToContentLabel: 'ข้ามไปยังเนื้อหา',
      darkModeSwitchLabel: 'รูปลักษณ์',
      darkModeSwitchTitle: 'สลับเป็นโหมดมืด',
      lightModeSwitchTitle: 'สลับเป็นโหมดสว่าง',
      nav: [
        { text: 'ภาพรวมคอร์ส', link: '/#curriculum' },
        { text: 'เส้นทางการเรียนรู้', link: '/learning_path' },
        { text: 'แหล่งข้อมูลสาธารณะ', link: '/resources' },
        { text: 'โครงสร้างหลักสูตร', link: '/syllabus' },
        { text: 'ต้นฉบับ (จีน)', link: 'https://github.com/QuantMLResearch/quant-handbook' },
        { text: 'เว็บไซต์ทางการ', link: 'https://www.quantml.cn/' }
      ],
      sidebar: [
        {
          text: 'คู่มือการเรียน',
          collapsed: true,
          items: [
            { text: 'เส้นทางการเรียนรู้และเกณฑ์การประเมิน', link: '/learning_path' },
            { text: 'คำแนะนำแหล่งข้อมูลสาธารณะ', link: '/resources' },
            { text: 'โครงสร้างหลักสูตร', link: '/syllabus' }
          ]
        },
        ...modules.map(module => ({
          text: `${module.id} · ${module.title}`,
          collapsed: true,
          items: module.lessons.map(lesson => ({ text: lesson.title, link: lesson.link }))
        }))
      ],
      search: {
        provider: 'local',
        options: {
          locales: {
            root: {
              translations: {
                button: {
                  buttonText: 'ค้นหาบทเรียน',
                  buttonAriaLabel: 'ค้นหาบทเรียน'
                },
                modal: {
                  noResultsText: 'ไม่พบผลลัพธ์ที่เกี่ยวข้อง',
                  resetButtonTitle: 'ล้างคำค้นหา',
                  footer: {
                    selectText: 'เลือก',
                    navigateText: 'นำทาง',
                    closeText: 'ปิด'
                  }
                }
              }
            }
          }
        }
      },
      outline: {
        level: [2, 3],
        label: 'หัวข้อในบทนี้'
      },
      docFooter: {
        prev: 'บทก่อนหน้า',
        next: 'บทถัดไป'
      },
      footer: {
        message: 'QUANTML · Quantitative Research Handbook',
        copyright: 'เนื้อหาต้นฉบับ © 2026 QuantMLResearch · ฉบับแปลภาษาไทยอย่างไม่เป็นทางการ'
      }
    }
  })
)
