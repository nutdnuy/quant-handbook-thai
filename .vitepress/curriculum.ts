export const modules = [
  {
    "id": "01",
    "title": "วิศวกรรมและความเข้าใจเชิงปริมาณ (Quant)",
    "en": "FOUNDATIONS",
    "description": "กำหนดขอบเขตงานวิจัยและสร้างสภาพแวดล้อมด้านข้อมูลและการประมวลผลสมัยใหม่",
    "tags": [
      "แหล่งที่มาของผลตอบแทน",
      "Backtest Bias",
      "Polars · DuckDB"
    ],
    "stage": "foundation",
    "lessons": [
      {
        "number": 1,
        "title": "บทที่ 1: ภาพรวมระบบนิเวศ Quant และโมเดลทำกำไรในหุ้น A-Share",
        "link": "/module_01/01_ashare_quant_landscape"
      },
      {
        "number": 2,
        "title": "บทที่ 2: “4 หลุมพรางมรณะ” ของ Quant Backtest และแนวทางป้องกัน",
        "link": "/module_01/02_backtest_pitfalls"
      },
      {
        "number": 3,
        "title": "บทที่ 3: บอกลาเทคสแต็กแบบเดิมที่เชื่องช้า: Polars + DuckDB",
        "link": "/module_01/03_modern_quant_stack"
      },
      {
        "number": 4,
        "title": "บทที่ 4: เวิร์กช็อปเชื่อมต่อข้อมูลหุ้น A-Share จากหลายแหล่งและจัดเก็บลงฐานข้อมูลเครื่อง",
        "link": "/module_01/04_data_acquisition_pipeline"
      }
    ]
  },
  {
    "id": "02",
    "title": "วิศวกรรมข้อมูลและ Point-in-Time (PIT)",
    "en": "POINT-IN-TIME",
    "description": "ตั้งแต่ข้อมูลราคาตลาดจนถึงงบการเงิน สร้างฐานข้อมูลที่สอดคล้องตามช่วงเวลาและตรวจสอบย้อนหลังได้",
    "tags": [
      "Universe หุ้นแบบไดนามิก",
      "การจัดตำแหน่งงบการเงิน",
      "PIT Engine"
    ],
    "stage": "foundation",
    "lessons": [
      {
        "number": 5,
        "title": "บทที่ 5: การปรับสิทธิ์ข้อมูลราคาตลาดและการสร้าง Universe หุ้นแบบไดนามิก",
        "link": "/module_02/05_universe_filtering"
      },
      {
        "number": 6,
        "title": "บทที่ 6: การจัดข้อมูลงบการเงินแบบ Point-in-Time ไร้ข้อมูลรั่วไหล",
        "link": "/module_02/06_pit_financials"
      },
      {
        "number": 7,
        "title": "บทที่ 7: วิวัฒนาการของการจัดกลุ่มอุตสาหกรรมและการจัดการแท็กธีมหุ้น",
        "link": "/module_02/07_industry_classification"
      },
      {
        "number": 8,
        "title": "บทที่ 8: [ปฏิบัติการ] เขียน PIT Search Engine สำหรับหุ้น A-Share ด้วยตนเอง",
        "link": "/module_02/08_pit_engine_practice"
      }
    ]
  },
  {
    "id": "03",
    "title": "วิศวกรรมฟีเจอร์และการค้นหาแฟกเตอร์ (Factor)",
    "en": "FACTOR RESEARCH",
    "description": "เชื่อมโยงตรรกะทางเศรษฐศาสตร์กับคุณลักษณะทางสถิติ พร้อมสร้างและทำ Standardization ให้ Factor",
    "tags": [
      "ราคา-ปริมาณและปัจจัยพื้นฐาน",
      "Microstructure",
      "Neutralization"
    ],
    "stage": "research",
    "lessons": [
      {
        "number": 9,
        "title": "บทที่ 9: ระบบ Factor ราคาและปริมาณแบบคลาสสิกและการเร่งความเร็วด้วย Polars Vectorization",
        "link": "/module_03/09_price_volume_factors"
      },
      {
        "number": 10,
        "title": "บทที่ 10: การค้นหา Factor ปัจจัยพื้นฐานและการคาดการณ์ของนักวิเคราะห์",
        "link": "/module_03/10_fundamental_factors"
      },
      {
        "number": 11,
        "title": "บทที่ 11: วิศวกรรมฟีเจอร์แปลงข้อมูลความถี่สูงเป็นความถี่ต่ำ (ผลตอบแทนส่วนเกิน (Excess Return) หลักของฝั่ง Buy-side)",
        "link": "/module_03/11_high_freq_to_low_freq"
      },
      {
        "number": 12,
        "title": "บทที่ 12: กระบวนการ 4 ขั้นตอนระดับอุตสาหกรรมในการ Preprocess Factor",
        "link": "/module_03/12_factor_preprocessing_pipeline"
      },
      {
        "number": 13,
        "title": "บทที่ 13: การทำ Orthogonalization ให้ Factor และการกำจัดปัญหา Collinearity",
        "link": "/module_03/13_factor_orthogonalization"
      }
    ]
  },
  {
    "id": "04",
    "title": "การประเมินและการตรวจสอบ Factor",
    "en": "VALIDATION",
    "description": "ทดสอบพลังการพยากรณ์ ความเสถียร และความเป็นไปได้ในการเทรด พร้อมคัดกรอง Excess Return ลวงตา",
    "tags": [
      "IC · ICIR",
      "Quantile Backtest",
      "Alpha Decay"
    ],
    "stage": "research",
    "lessons": [
      {
        "number": 14,
        "title": "บทที่ 14: ดัชนีชี้วัดหลักวัดพลังการพยากรณ์ของ Factor (IC / Rank IC / ICIR)",
        "link": "/module_04/14_factor_ic_metrics"
      },
      {
        "number": 15,
        "title": "บทที่ 15: การ Backtest แบบแบ่งกลุ่มและการทดสอบ Monotonicity",
        "link": "/module_04/15_quantile_and_monotonicity"
      },
      {
        "number": 16,
        "title": "บทที่ 16: อัตรา Turnover ของ Factor และกราฟการเสื่อมถอย (Alpha Decay)",
        "link": "/module_04/16_turnover_and_alpha_decay"
      },
      {
        "number": 17,
        "title": "บทที่ 17: [กรณีศึกษาหลบหลุมพราง] เจาะลึกการตรวจสอบ Factor ปลอมที่ใช้งานจริงไม่ได้",
        "link": "/module_04/17_fake_alpha_forensics"
      }
    ]
  },
  {
    "id": "05",
    "title": "Machine Learning ทางการเงิน",
    "en": "MACHINE LEARNING",
    "description": "ตั้งแต่การตรวจสอบแบบแยกช่วงเวลาไปจนถึงการจัดอันดับภาคตัดขวาง (Cross-sectional) สู่กระบวนการคัดเลือกหุ้นที่อธิบายได้",
    "tags": [
      "Purging · Embargo",
      "LightGBM",
      "SHAP"
    ],
    "stage": "research",
    "lessons": [
      {
        "number": 18,
        "title": "บทที่ 18: วิธีแบ่งข้อมูล Cross-Validation (CV) สำหรับอนุกรมเวลา (Time-series) ทางการเงินอย่างถูกต้อง",
        "link": "/module_05/18_financial_cv_purging_embargo"
      },
      {
        "number": 19,
        "title": "บทที่ 19: ปฏิบัติการ Gradient Boosted Trees ข้อมูลตาราง (LightGBM / CatBoost)",
        "link": "/module_05/19_lightgbm_catboost_ranking"
      },
      {
        "number": 20,
        "title": "บทที่ 20: ความสำคัญของฟีเจอร์และความสามารถในการอธิบายโมเดล (SHAP Value)",
        "link": "/module_05/20_shap_interpretability"
      },
      {
        "number": 21,
        "title": "บทที่ 21: วิเคราะห์โมเดล Deep Learning ข้อมูล Time-series โดยสังเขป (ขอบเขตการใช้งาน LSTM / GRU / Transformer)",
        "link": "/module_05/21_deep_learning_boundaries"
      },
      {
        "number": 22,
        "title": "บทที่ 22: [ปฏิบัติการ] จากการให้คะแนน Multi-factor แบบ Cross-sectional สู่ไปป์ไลน์คัดเลือกหุ้นอัตโนมัติ",
        "link": "/module_05/22_ml_stock_selection_pipeline"
      }
    ]
  },
  {
    "id": "06",
    "title": "แบบจำลองความเสี่ยง (Risk Model) และการหาพอร์ตที่เหมาะสม (Portfolio Optimization)",
    "en": "PORTFOLIO CONSTRUCTION",
    "description": "แปลงสัญญาณคาดการณ์เป็นพอร์ตลงทุนที่สอดคล้องกับข้อจำกัดด้านความเสี่ยง ต้นทุน และ Turnover",
    "tags": [
      "Barra",
      "Convex Optimization",
      "Risk Parity"
    ],
    "stage": "portfolio",
    "lessons": [
      {
        "number": 23,
        "title": "บทที่ 23: ถอดรหัสคณิตศาสตร์ของ Barra Structural Risk Model",
        "link": "/module_06/23_barra_risk_model_architecture"
      },
      {
        "number": 24,
        "title": "บทที่ 24: ปฏิบัติการตัวแก้โจทย์ Convex Optimization (cvxpy)",
        "link": "/module_06/24_cvxpy_portfolio_optimization"
      },
      {
        "number": 25,
        "title": "บทที่ 25: ข้อจำกัดอัตรา Turnover และการเพิ่มพจน์บทลงโทษต้นทุนธุรกรรม",
        "link": "/module_06/25_turnover_constraints_and_costs"
      },
      {
        "number": 26,
        "title": "บทที่ 26: Risk Parity และ Hierarchical Risk Parity (HRP)",
        "link": "/module_06/26_risk_parity_and_hrp"
      }
    ]
  },
  {
    "id": "07",
    "title": "Backtest เสมือนจริงและการส่งคำสั่งเทรด",
    "en": "BACKTEST & EXECUTION",
    "description": "นำกฎเกณฑ์การซื้อขาย ข้อจำกัดการจับคู่คำสั่ง และต้นทุน Market Impact เข้ามาทดสอบกลยุทธ์",
    "tags": [
      "Event-driven",
      "Transaction Friction",
      "กฎการจับคู่คำสั่ง (Matching)"
    ],
    "stage": "portfolio",
    "lessons": [
      {
        "number": 27,
        "title": "บทที่ 27: สถาปัตยกรรมเอนจิน Backtest หุ้น A-Share: Vectorized vs Event-driven",
        "link": "/module_07/27_vectorized_vs_event_driven_backtest"
      },
      {
        "number": 28,
        "title": "บทที่ 28: จำลอง Transaction Friction ของการซื้อขายจริงอย่างสมบูรณ์แบบ",
        "link": "/module_07/28_transaction_friction_and_market_impact"
      },
      {
        "number": 29,
        "title": "บทที่ 29: พัฒนาโค้ดจำลองกฎ Matching พิเศษของหุ้น A-Share",
        "link": "/module_07/29_matching_rules_limits_suspensions"
      }
    ]
  },
  {
    "id": "08",
    "title": "การแจกแจงที่มา (Attribution) ของผลตอบแทนและการติดตามพอร์ตเทรดจริง (Live Trading)",
    "en": "ATTRIBUTION & MONITORING",
    "description": "แยกแยะแหล่งที่มาของผลตอบแทน พร้อมสร้างระบบติดตามการทำงาน การแจ้งเตือนความเสี่ยง และวงรอบการทบทวนกลยุทธ์",
    "tags": [
      "Brinson · Barra",
      "การแจ้งเตือนความเสี่ยง",
      "การติดตามระบบงาน"
    ],
    "stage": "portfolio",
    "lessons": [
      {
        "number": 30,
        "title": "บทที่ 30: ปฏิบัติการ Attribution ผลตอบแทน: กำไรที่ได้มาจาก Beta หรือ Pure Alpha กันแน่?",
        "link": "/module_08/30_performance_attribution_brinson_barra"
      },
      {
        "number": 31,
        "title": "บทที่ 31: ระบบมอนิเตอร์การ Live Trading และแดชบอร์ดเตือนภัยความเสี่ยง",
        "link": "/module_08/31_live_monitoring_risk_dashboard"
      }
    ]
  }
] as const

export const lessonCount = modules.reduce((total, module) => total + module.lessons.length, 0)
