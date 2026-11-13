# คู่มือการวิจัยเชิงปริมาณ (Quantitative Research Handbook)

> [!NOTE]
> **ฉบับแปลภาษาไทย (Unofficial Thai Translation)**
> repo นี้เป็นฉบับแปลภาษาไทยอย่างไม่เป็นทางการของ [QuantMLResearch/quant-handbook](https://github.com/QuantMLResearch/quant-handbook) (《量化研究手册》) ลิขสิทธิ์เนื้อหาต้นฉบับทั้งหมดเป็นของผู้เขียนเดิม — ทีม QuantMLResearch / [quantml.cn](https://www.quantml.cn/)
> แปลจากต้นฉบับ commit `29665c7` (29 ต.ค. 2026) โดยใช้ AI ช่วยแปลและตรวจโครงสร้าง หากพบคำแปลคลาดเคลื่อน โปรดเปิด Issue หรือ Pull Request — หากต้องการอ้างอิงที่แม่นยำ ให้ยึดต้นฉบับภาษาจีนเป็นหลัก
>
> 🌐 อ่านเป็นเว็บไซต์: **https://nutdnuy.github.io/quant-handbook-thai/**
>
> โค้ด Python แปลเฉพาะคอมเมนต์และ docstring ส่วนข้อความใน string (เช่น error message ชื่ออุตสาหกรรม) คงภาษาจีนไว้ เพื่อให้ชุดทดสอบเดิมยังผ่านทั้งหมด


<p align="center">
  <strong>ระบบการวิจัยการลงทุนเชิงปริมาณ (Quant) และวิศวกรรมปฏิบัติการระดับมืออาชีพ</strong><br>
  ตั้งแต่วิศวกรรมข้อมูล, การค้นหาแฟกเตอร์ (Factor), Machine Learning ไปจนถึงวงจรสมบูรณ์ของการเทรดจริง (Live Trading) ในกองทุนดัชนีเสริมผลตอบแทน (Index Enhancement)
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20~%203.13-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Package%20Manager-uv-blueviolet.svg" alt="uv">
  <img src="https://img.shields.io/badge/Docs-VitePress-42b883.svg" alt="VitePress">
  <img src="https://img.shields.io/badge/Data%20Engine-Polars%20%7C%20DuckDB-ff69b4.svg" alt="Polars and DuckDB">
  <img src="https://img.shields.io/badge/Optimization-cvxpy-orange.svg" alt="cvxpy">
  <img src="https://img.shields.io/badge/Test%20Suite-100%25%20Offline%20Passed-success.svg" alt="Tests">
</p>

---

## 📌 ภาพรวมโครงการ

《คู่มือการวิจัยเชิงปริมาณ》เป็นชุดบทเรียนสำหรับการวิจัยการลงทุนเชิงปริมาณและการพัฒนาเชิงวิศวกรรมระดับอุตสาหกรรมสำหรับมืออาชีพ ทั้งชุดครอบคลุม **8 โมดูลหลัก และ 31 บทเจาะลึก** พร้อมเอกสารเว็บแบบ Interactive ที่สมบูรณ์ ตลอดจน **31 สคริปต์ปฏิบัติการ Python แบบสแตนด์อโลนที่พร้อมใช้งานทันทีและชุด Unit Test ออฟไลน์ครบวงจร**

โครงการนี้มุ่งเป้าไปที่นักวิจัยและนักพัฒนาที่มีพื้นฐานการเขียนโปรแกรม Python และต้องการสร้าง**วงจรวิศวกรรมการลงทุน Quant ครบวงจรระดับสถาบันฝั่ง Buy-side/กองทุนเอกชน (Private Fund)** อย่างเป็นระบบ โดยเชื่อมโยงสายงานทั้งหมดอย่างสมบูรณ์ตั้งแต่ "Data Cleaning $\rightarrow$ การกำกับดูแลข้อมูล PIT $\rightarrow$ Feature Engineering $\rightarrow$ การประเมิน Factor $\rightarrow$ Machine Learning $\rightarrow$ แบบจำลองความเสี่ยง (Risk Model) $\rightarrow$ การหาพอร์ตที่เหมาะสม (Portfolio Optimization) $\rightarrow$ การจับคู่คำสั่ง (Matching) เสมือนจริง $\rightarrow$ การติดตามและแจกแจงที่มา (Attribution)"

> [!IMPORTANT]
> **ข้อชี้แจงขอบเขตการสอนกับการเทรดจริง**：ตัวอย่างทั้งหมดในโครงการนี้มาพร้อมกับข้อมูลสังเคราะห์เพื่อการสอนและ Contract Test ในตัว โดยมีวัตถุประสงค์เพื่อตรวจสอบความถูกต้องของตรรกะสถาปัตยกรรมทางวิศวกรรมและอัลกอริทึมทางคณิตศาสตร์ การทดสอบผ่านไม่ได้เป็นข้อพิสูจน์ถึงผลตอบแทนในการเทรดจริง และข้อมูลสังเคราะห์ไม่ได้แสดงถึงสภาวะตลาดจริง ก่อนเชื่อมต่อระบบเทรดจริง จำเป็นต้องเชื่อมต่อระบบส่งคำสั่งและข้อมูลราคา, อินเทอร์เฟซการเทรด และแหล่งข้อมูลย้อนหลังที่ได้รับอนุญาตตามกฎระเบียบแยกต่างหาก

---

## 🌟 จุดเด่นหลักและหลักการออกแบบ

```mermaid
flowchart LR
    A["01 ความเข้าใจและโครงสร้างพื้นฐาน\n(กฎเกณฑ์หุ้น A-Share และ Modern Stack)"] --> B["02 วิศวกรรมข้อมูลและการป้องกันข้อมูลรั่วไหล\n(งบการเงิน PIT และ Dynamic Universe)"]
    B --> C["03 ฟีเจอร์และการแปลงความถี่สูงเป็นความถี่ต่ำ\n(ราคาและปริมาณ/ปัจจัยพื้นฐาน/OFI ระดับจุลภาค)"]
    C --> D["04 การประเมิน Factor ระดับอุตสาหกรรม\n(IC/IR/Monotonicity/Orthogonalization)"]
    D --> E["05 ML ภาคตัดขวางและอนุกรมเวลา\n(Purged CV/LightGBM)"]
    E --> F["06 Barra และ Portfolio Optimization\n(Risk Model/ข้อจำกัด cvxpy)"]
    F --> G["07 การ Backtest เสมือนจริงและแรงเสียดทานการเทรด\n(ข้อจำกัดราคาชนเพดาน/พื้น/ต้นทุน Market Impact)"]
    G --> H["08 การติดตาม Attribution และการนำไปใช้เทรดจริง\n(Attribution/การควบคุมความเสี่ยง/การซ้อมกู้คืนระบบ)"]
```

1. **ยึดตามกลไกระดับจุลภาคและข้อจำกัดการเทรดจริงของหุ้น A-Share**
   - จำลองเชิงลึกสำหรับกลไกเฉพาะของหุ้น A-Share ได้แก่ **ระบบ T+1**, **ข้อจำกัดราคาชนเพดาน/พื้น (Price Limit ±10% / ±20% / ±30%) พร้อมตรรกะคำสั่งไม่สามารถจับคู่ได้**, **การจัดการการพักการซื้อขาย (Suspension) และการเพิกถอนหลักทรัพย์ (Delisting)**, **บัญชีบันทึกเงินปันผล/หุ้นปันผลและสิทธิ์ XD/XR แบบ 4 จุดเวลา**, ตลอดจนอากรแสตมป์ (Stamp Duty) และต้นทุน Market Impact ปฏิเสธ "Backtest ของเล่น" ที่ตัดขาดจากความเป็นจริง
2. **สถาปัตยกรรมการกำกับดูแลข้อมูล Point-in-Time (PIT) ที่รัดกุม**
   - แยกความแตกต่างอย่างเข้มงวดระหว่าง**เวลาที่มีผลบังคับใช้ (`effective_date`)** กับ**เวลาที่กลยุทธ์มองเห็นข้อมูล (`available_at`)** รองรับการย้อนรอยประวัติการแก้ไขงบการเงินหลายครั้ง ขจัด Look-ahead Bias และ Survivorship Bias อย่างสิ้นเชิงตั้งแต่ระดับรากฐานของข้อมูล
3. **เปิดรับ Modern High-Performance Quant Engineering Stack**
   - นำ **Polars** (Vectorized Expressions และการประมวลผลคู่ขนานความเร็วสูง) และ **DuckDB** (การวิเคราะห์แบบ Embedded ความเร็วสูง) มาใช้อย่างเต็มรูปแบบ เพื่อแทนที่การคำนวณแบบ Single-thread บน Pandas แบบดั้งเดิมที่ไม่มีประสิทธิภาพ รองรับการรวบรวมข้อมูลระดับสิบล้านแถว ทั้งข้อมูลความถี่สูงและข้อมูลภาคตัดขวาง (Cross-sectional) ภายในหลักวินาที
4. **Quant Machine Learning ระดับ Buy-side และการสร้างพอร์ตด้วย Convex Optimization**
   - นำการทำ Cross-Validation แบบ **Purged / Embargo K-Fold** ตามอนุกรมเวลา (Time-series) มาปรับใช้อย่างเคร่งครัด; ปรับเทียบการคาดการณ์ผลตอบแทนด้วย `RankReturnCalibrator`; สร้าง Barra Multi-factor Structural Risk Model; ใช้ `cvxpy` ในการแก้ปัญหา Convex Optimization เพื่อจำกัดเพดานแข็งสำหรับ Tracking Error (TE) รายปี และเพิ่มบทลงโทษต้นทุน Impact จาก Turnover
5. **การตรวจสอบที่ทำซ้ำได้แบบออฟไลน์ 100% โดยสมบูรณ์ในตัวเอง**
   - ทั้ง 31 บทมีซอร์สโค้ดที่รันได้แยกเป็นอิสระ พร้อม Test Fixtures ข้อมูลสังเคราะห์ (`fixtures.py`, `common.py`) สามารถรันชุด Automated Regression Tests ทั้งหมดได้ในคลิกเดียว (`npm run test:examples`) โดยไม่ต้องซื้อข้อมูลเชิงพาณิชย์

---

## 🗺️ แผนผังภาพรวมหลักสูตรและการจับคู่ซอร์สโค้ดปฏิบัติการ

หลักสูตรนี้ประกอบด้วย 8 โมดูลหลัก และ 31 บท โดยแต่ละบทมีซอร์สโค้ด Python และ Contract Verification แยกเป็นอิสระ:

| โมดูล | บทที่ | หัวข้อหลัก (ลิงก์บทเรียน) | Entry Point ของ Python | Extra Dependencies | ประเด็นสำคัญในการตรวจสอบ |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **M1 วิศวกรรมและความรู้พื้นฐาน** | 01 | [ภาพรวมระบบนิเวศ Quant หุ้น A-Share และโมเดลการทำกำไร](./module_01/01_ashare_quant_landscape.md) | `examples.lecture_01_comprehensive_cost_engine` | พื้นฐาน | แหล่งที่มาของผลตอบแทน, Basis ในการ Hedging และการวัดความอ่อนไหวต่อต้นทุน |
| | 02 | ["4 สาเหตุหลักที่ทำให้พัง" ในการ Backtest เชิงปริมาณและหลักการป้องกัน](./module_01/02_backtest_pitfalls.md) | `examples.lecture_02_backtest_sanitizer` | พื้นฐาน | Look-ahead Bias, Survivorship Bias และการกระจายตัวของการทดสอบ DSR |
| | 03 | [บอกลา Stack แบบดั้งเดิมที่เชื่องช้า: Polars + DuckDB](./module_01/03_modern_quant_stack.md) | `examples.lecture_03_benchmark_massive` | พื้นฐาน | การ Benchmark ความเร็วระดับล้านแถวและการตรวจสอบความสอดคล้องของผลลัพธ์ |
| | 04 | [การเชื่อมต่อข้อมูลหลายแหล่งของหุ้น A-Share และการบันทึกลงฐานข้อมูลท้องถิ่นภาคปฏิบัติ](./module_01/04_data_acquisition_pipeline.md) | `examples.lecture_04_industrial_data_pipeline` | `data` (ทางเลือก) | Data Lake แบบส่วนเพิ่ม, การเขียนดิสก์แบบ Atomic Replacement และการป้องกันข้อผิดพลาดของ API |
| **M2 วิศวกรรมข้อมูลและ PIT** | 05 | [การปรับสิทธิ์ข้อมูลราคาและการสร้าง Dynamic Universe](./module_02/05_universe_filtering.md) | `examples.lecture_05_dynamic_universe_builder` | พื้นฐาน | การคำนวณราคาปรับสิทธิ์ย้อนหลัง, การคัดกรองอายุการจดทะเบียนในตลาด และ Dynamic Universe |
| | 06 | [การจัดเรียงงบการเงินแบบ Point-in-Time โดยไม่มีข้อมูลรั่วไหล](./module_02/06_pit_financials.md) | `examples.lecture_06_pit_financial_engine` | พื้นฐาน | เวลาเปิดเผยงบการเงิน, การเลือกรุ่นที่แก้ไข และการจัดเรียงเพื่อป้องกัน Look-ahead |
| | 07 | [วิวัฒนาการการจัดกลุ่มอุตสาหกรรมและการจัดการแท็กหมวดหมู่แนวคิด](./module_02/07_industry_classification.md) | `examples.lecture_07_industry_engine` | พื้นฐาน | การเปลี่ยนแปลงเวอร์ชันการจัดกลุ่มอุตสาหกรรม, หมวดหมู่ที่ไม่ทราบค่า และเหตุการณ์ของธีมแนวคิด |
| | 08 | [【ปฏิบัติการ】พัฒนาเอนจินค้นหาข้อมูล PIT ของหุ้น A-Share ด้วยตนเอง](./module_02/08_pit_engine_practice.md) | `examples.lecture_08_pit_engine_practice` | พื้นฐาน | การคิวรีข้อมูลภาคตัดขวาง ณ เวลาปิดตลาด 15:00 น. และการประกอบข้อมูล PIT หลายตารางแบบครอบคลุม |
| **M3 ฟีเจอร์และการค้นหา Factor** | 09 | [ระบบ Factor ราคาและปริมาณแบบคลาสสิกและการเร่งความเร็วด้วย Polars](./module_03/09_price_volume_factors.md) | `examples.lecture_09_price_volume_factors` | พื้นฐาน | ฟีเจอร์ราคาและปริมาณปรับสิทธิ์, Rolling Window และ Vectorized Expressions |
| | 10 | [การค้นหา Factor ปัจจัยพื้นฐานและการคาดการณ์ของนักวิเคราะห์](./module_03/10_fundamental_factors.md) | `examples.lecture_10_fundamental_factors` | พื้นฐาน | การวิเคราะห์แบบ DuPont, การเทียบรายไตรมาส YoY และการสร้างตัวชี้วัดทางการเงิน TTM |
| | 11 | [Feature Engineering แปลงข้อมูลความถี่สูงเป็นความถี่ต่ำ (OFI ระดับจุลภาค)](./module_03/11_high_freq_to_low_freq.md) | `examples.lecture_11_high_freq_to_low_freq` | พื้นฐาน | การคำนวณ Order Flow Imbalance (OFI) 3 เงื่อนไข และการรวมข้อมูลภาคตัดขวางระดับรายวัน |
| | 12 | [กระบวนการ 4 ขั้นตอนระดับอุตสาหกรรมในการเตรียมข้อมูล Factor](./module_03/12_factor_preprocessing_pipeline.md) | `examples.lecture_12_factor_preprocessing_pipeline` | พื้นฐาน | การทำ Winsorization, Standardization, การเติมค่าที่ขาดหาย และการทำ Neutralization ถ่วงน้ำหนักด้วย WLS |
| | 13 | [การทำ Orthogonalization สำหรับ Factor และการกำจัด Multicollinearity](./module_03/13_factor_orthogonalization.md) | `examples.lecture_13_factor_orthogonalization` | พื้นฐาน | การทำ Gram-Schmidt/Symmetric Orthogonalization, ภาคตัดขวางแบบ Full Rank และการแจ้งเตือน Collinearity |
| **M4 การประเมินและการตรวจสอบ Factor** | 14 | [ตัวชี้วัดหลักของพลังการคาดการณ์ของ Factor (IC/Rank IC/ICIR)](./module_04/14_factor_ic_metrics.md) | `examples.lecture_14_factor_ic_metrics` | พื้นฐาน | สหสัมพันธ์อันดับภาคตัดขวาง, t-test ตามอนุกรมเวลา และการปรับแก้ HAC |
| | 15 | [การทำ Quantile Backtest และการทดสอบความเป็นโมโนโทนิก](./module_04/15_quantile_and_monotonicity.md) | `examples.lecture_15_quantile_and_monotonicity` | พื้นฐาน | การทำ Long-Short Hedge ตาม Quantile, คะแนนความเป็นโมโนโทนิก และ Maximum Drawdown |
| | 16 | [อัตรา Turnover ของ Factor และเส้นโค้งการลดทอน (Alpha Decay)](./module_04/16_turnover_and_alpha_decay.md) | `examples.lecture_16_turnover_and_alpha_decay` | พื้นฐาน | ความสัมพันธ์ในตัวเองของ Factor, ค่าครึ่งชีวิตของ Marginal IC และการประมาณการสถานการณ์การปรับพอร์ต (Rebalance) |
| | 17 | [【กรณีศึกษาภาคปฏิบัติ】การตรวจสอบและแยกแยะ Factor ปลอม](./module_04/17_fake_alpha_forensics.md) | `examples.lecture_17_fake_alpha_forensics` | พื้นฐาน | การแฝงตัวจาก Exposure ของ Market Cap, Win Rate ปลอมจากราคาชนเพดาน/พื้น และเงินสดที่ยังไม่จับคู่คำสั่ง |
| **M5 Machine Learning ทางการเงิน** | 18 | [วิธีการแบ่งชุดข้อมูล Cross-Validation (CV) สำหรับอนุกรมเวลาทางการเงินอย่างถูกต้อง](./module_05/18_financial_cv_purging_embargo.md) | `examples.lecture_18_financial_cv_purging_embargo` | พื้นฐาน | การแบ่งช่วงเวลาแบบ Purged & Embargo และการป้องกันข้อมูลรั่วไหลทั้งสองด้าน |
| | 19 | [Gradient Boosted Decision Trees สำหรับข้อมูลตาราง (LightGBM) ภาคปฏิบัติ](./module_05/19_lightgbm_catboost_ranking.md) | `examples.lecture_19_lightgbm_catboost_ranking` | `ml` | การควบคุมขอบเขตความพร้อมของ Label และการจัดอันดับภาคตัดขวางด้วย LambdaRank |
| | 20 | [ความสำคัญและความสามารถในการอธิบายฟีเจอร์ (SHAP)](./module_05/20_shap_interpretability.md) | `examples.lecture_20_shap_interpretability` | `ml` | การแยกสัดส่วนอิทธิพลของฟีเจอร์ด้วย TreeSHAP และการตรวจสอบความสอดคล้องของการรวมผลลัพธ์ |
| | 21 | [การวิเคราะห์เบื้องต้นและขอบเขตข้อจำกัดของโมเดล Deep Learning สำหรับอนุกรมเวลา](./module_05/21_deep_learning_boundaries.md) | `examples.lecture_21_deep_learning_boundaries` | `deep` | การเทรน 3D Time-series Tensor ด้วย CNN/GRU และขอบเขตการเกิด Overfitting |
| | 22 | [【ปฏิบัติการ】จากคะแนนภาคตัดขวางสู่ Pipeline คัดเลือกหุ้นอัตโนมัติ](./module_05/22_ml_stock_selection_pipeline.md) | `examples.lecture_22_ml_stock_selection_pipeline` | `ml` | การคาดการณ์นอกกลุ่มตัวอย่าง, ตัวปรับเทียบผลตอบแทน และการสร้างน้ำหนักพอร์ตเป้าหมาย |
| **M6 แบบจำลองความเสี่ยงและการหาพอร์ตที่เหมาะสม** | 23 | [การถอดโครงสร้างทางคณิตศาสตร์ของ Barra Structural Risk Model](./module_06/23_barra_risk_model_architecture.md) | `examples.lecture_23_barra_risk_model_architecture` | พื้นฐาน | การประมาณผลตอบแทน Factor ด้วย WLS, Covariance แบบ EWMA และการทำ Shrinkage สำหรับ Specific Risk |
| | 24 | [Convex Optimization Solver (cvxpy) ภาคปฏิบัติ](./module_06/24_cvxpy_portfolio_optimization.md) | `examples.lecture_24_cvxpy_portfolio_optimization` | `optimization` | ข้อจำกัดความเป็นกลางของอุตสาหกรรม/สไตล์ และข้อจำกัดกำลังสองของ Tracking Error (TE) รายปี |
| | 25 | [ข้อจำกัดอัตรา Turnover และการกำหนดพจน์บทลงโทษต้นทุนการเทรด](./module_06/25_turnover_constraints_and_costs.md) | `examples.lecture_25_turnover_constraints_and_costs` | `optimization` | ต้นทุน Impact แบบไม่เป็นเชิงเส้นตาม AUM/ADV, การแก้ปัญหาข้อจำกัด Turnover แบบยืดหยุ่นและแบบเข้มงวด |
| | 26 | [Risk Parity และ HRP](./module_06/26_risk_parity_and_hrp.md) | `examples.lecture_26_risk_parity_and_hrp` | พื้นฐาน | การกระจายความเสี่ยงเท่ากัน (ERC), Hierarchical Risk Parity (HRP) |
| **M7 การ Backtest เสมือนจริงและการส่งคำสั่ง** | 27 | [เอนจิน Backtest หุ้น A-Share: Vectorized vs Event-driven](./module_07/27_vectorized_vs_event_driven_backtest.md) | `examples.lecture_27_vectorized_vs_event_driven_backtest` | พื้นฐาน | การจับคู่คำสั่ง ณ ราคาเปิดของวันถัดไป, การกระทบยอดบัญชีแยกประเภทแบบคู่สำหรับเงินสด/ลูกหนี้/หุ้น |
| | 28 | [การจำลองแรงเสียดทานการเทรดจริงเสมือนจริง](./module_07/28_transaction_friction_and_market_impact.md) | `examples.lecture_28_transaction_friction_and_market_impact` | พื้นฐาน | ค่าคอมมิชชัน, ค่าธรรมเนียมการโอน, การคำนวณอากรแสตมป์แบบขั้นบันได และแบบจำลองขีดความสามารถการเทรดด้านเดียว |
| | 29 | [การพัฒนากฎการจับคู่คำสั่งพิเศษของหุ้น A-Share ในระดับโค้ด](./module_07/29_matching_rules_limits_suspensions.md) | `examples.lecture_29_matching_rules_limits_suspensions` | พื้นฐาน | ข้อจำกัดราคาเปิดชนเพดาน/พื้นแบบคงที่ตลอดวัน, สัดส่วนสภาพคล่อง, ลำดับการเกิดสิทธิ์ 4 จุดเวลาสำหรับเงินปันผลและหุ้นปันผล |
| **M8 การแจกแจงที่มาของผลการดำเนินงานและการติดตาม** | 30 | [Performance Attribution ภาคปฏิบัติ: Beta หรือ Pure Alpha?](./module_08/30_performance_attribution_brinson_barra.md) | `examples.lecture_30_performance_attribution_brinson_barra` | พื้นฐาน | การจัดสรรสินทรัพย์ Brinson แบบช่วงเวลาเดียว และการวิเคราะห์ผลตอบแทน Factor ตามโมเดล Barra แบบปิด |
| | 31 | [แดชบอร์ดติดตามการเทรดจริงและการแจ้งเตือนความเสี่ยง](./module_08/31_live_monitoring_risk_dashboard.md) | `examples.lecture_31_live_monitoring_risk_dashboard` | พื้นฐาน | การติดตามมูลค่าสินทรัพย์สุทธิสัมพัทธ์, การเคลื่อนตัวของฟีเจอร์ (PSI Feature Drift), กลไก Circuit Breaker 3 ระดับพร้อม Hysteresis |
---

## 🚀 คู่มือเริ่มต้นใช้งานอย่างรวดเร็ว

### 1. ข้อกำหนดสภาพแวดล้อมเบื้องต้น
- **Python**: `>= 3.11, < 3.14`
- **Node.js**: `>= 18`
- **Package Manager ที่แนะนำ**: [uv](https://docs.astral.sh/uv/) (เครื่องมือจัดการ Package ของ Python ยุคใหม่ที่มีความเร็วสูง) และ `npm`

### 2. รัน VitePress Interactive Documentation Site บนเครื่อง Local
```bash
# 1. ติดตั้ง Dependencies ของเอกสารฝั่ง Frontend
npm ci

# 2. เริ่มต้น Development Server บนเครื่อง Local (รองรับ Hot Reload, เข้าใช้งานเริ่มต้นที่ http://localhost:5173)
npm run dev

# 3. Build ไฟล์ Static Site
npm run build

# 4. Preview ไฟล์ที่ Build บนเครื่อง Local
npm run preview
```

### 3. ตั้งค่าสภาพแวดล้อมรัน Python และรันโค้ดตัวอย่าง
โปรเจกต์นี้แนะนำให้ใช้ `uv` สำหรับจัดการ Lockfile สภาพแวดล้อมเพื่อให้ทำซ้ำได้ (Dependencies ทั้งหมดถูกล็อกไว้อย่างสมบูรณ์ใน `uv.lock`)

```bash
# ซิงค์ Dependencies พื้นฐาน (ครอบคลุม 25 บทหลัก)
uv sync --locked

# ซิงค์ Dependencies ทั้งหมด (รวม Machine Learning, Deep Learning และ Convex Optimization Solver)
uv sync --locked --extra ml --extra deep --extra optimization
```

รันโมดูลตัวอย่างการใช้งานจริงแบบเดี่ยว (**ต้องรันจาก Root Directory ของโปรเจกต์ในโหมดโมดูลด้วย `-m` เสมอ**):

```bash
# รันตัวอย่างพื้นฐาน (เช่น บทที่ 8: เขียน PIT Search Engine สำหรับหุ้น A-Share ด้วยตนเอง)
uv run --locked python -m examples.lecture_08_pit_engine_practice

# รัน Pipeline คัดเลือกหุ้นด้วย Machine Learning (บทที่ 22, ต้องใช้ --extra ml)
uv run --locked --extra ml python -m examples.lecture_22_ml_stock_selection_pipeline

# รัน Convex Optimization ภายใต้ข้อจำกัด Turnover ของพอร์ต (บทที่ 25, ต้องใช้ --extra optimization)
uv run --locked --extra optimization python -m examples.lecture_25_turnover_constraints_and_costs

# รัน Backtest Engine แบบ Event-driven เสมือนจริงเต็มรูปแบบ (บทที่ 27)
uv run --locked python -m examples.lecture_27_vectorized_vs_event_driven_backtest

# รัน Dashboard มอนิเตอร์การเทรดจริง (Live Trading) และระบบเตือน Circuit Breaker (บทที่ 31)
uv run --locked python -m examples.lecture_31_live_monitoring_risk_dashboard
```

> [!NOTE]
> **การดาวน์โหลดข้อมูลจริง (ทางเลือก)**: หากต้องการเชื่อมต่ออินเทอร์เน็ตเพื่อดาวน์โหลดข้อมูลตัวอย่างประวัติศาสตร์จริง และบันทึกลงใน Data Lake ท้องถิ่น `data_lake/` สามารถใช้ `--extra data`:
> ```bash
> uv sync --locked --extra data
> uv run --locked --extra data python -m examples.lecture_04_industrial_data_pipeline --fetch
> ```

### 4. รันชุด Automated Test ทั้งหมด
โปรเจกต์นี้มาพร้อมกับ Unit Test สำหรับ Contract และตรรกะทางคณิตศาสตร์อย่างครบถ้วน โดยการทดสอบทั้งหมดใช้ Synthetic Data ภายในตัว จึงไม่ต้องเชื่อมต่ออินเทอร์เน็ต:

```bash
# รัน Regression Test เต็มชุด
npm run test:examples

# หรือเรียกใช้ uv เพื่อรัน unittest โดยตรง:
uv run --locked --extra ml --extra deep --extra optimization python -m unittest discover -s tests -v
```

คำอธิบายการจำแนกชุดการทดสอบ:
- `tests/test_tutorial.py`: ครอบคลุมโมดูล 01–02 (การเขียน Data Lake แบบ Atomic, การวินิจฉัย Look-ahead Bias, การค้นหาข้อมูลย้อนหลังหลายเวอร์ชันด้วย Point-in-Time (PIT) และ Dynamic Universe)
- `tests/test_factor_modules.py`: ครอบคลุมโมดูล 03–04 (OFI ระดับจุลภาคของราคาและปริมาณการซื้อขาย, WLS Weighted Orthogonalization, การตรวจจับ Collinearity, สถิติ IC/IR, Monotonicity และการตรวจสอบเงินสดที่ยังไม่จับคู่คำสั่ง)
- `tests/test_ml_risk_modules.py`: ครอบคลุมโมดูล 05–06 (การแยกช่วงเวลาด้วย Purged CV, การรวมผลรวมของ SHAP, แบบจำลองความเสี่ยง Barra (Barra Risk Model), การแก้ปัญหา Convex Optimization ภายใต้ข้อจำกัด Tracking Error)
- `tests/test_execution_monitoring.py`: ครอบคลุมโมดูล 07–08 (กฎการอนุรักษ์ตามหลักบัญชีคู่, ข้อจำกัดห้ามจับคู่คำสั่งซื้อขายตอนเปิดตลาด, วิวัฒนาการ 4 จุดเวลาของการจ่ายปันผลและหุ้นปันผล, การปิดงบ Brinson Attribution และ Circuit Breaker แบบ Hysteresis สำหรับการควบคุมความเสี่ยง)

---

## 📐 ข้อกำหนดสัญญาข้อมูลหลัก (Point-in-Time Schema)

ในโมดูล 02 (บทที่ 4–8) โปรเจกต์ได้กำหนด Data Contract มาตรฐานระดับอุตสาหกรรมที่ใช้ร่วมกัน ซึ่งกระบวนการสกัด Feature, การเทรนโมเดล และการจับคู่คำสั่ง (Matching) ใน Backtest ทั้งหมดหลังจากนี้จะต้องปฏิบัติตาม Contract นี้อย่างเคร่งครัด:

### 1. มาตรฐานเวลาและ Timezone
- **ฟิลด์วันที่**: กำหนดใช้ `pl.Date` เหมือนกันทั้งหมด
- **Timestamp สำหรับข้อมูลความถี่สูงและเหตุการณ์**: ต้องใช้ **`pl.Datetime` แบบไม่มี Timezone ของเวลาท้องถิ่น Asia/Shanghai** เหมือนกันทั้งหมด เวลาภายนอกที่มี Timezone ต้องแปลงเป็นเวลาเซี่ยงไฮ้ก่อนแล้วจึงตัด Timezone ออก ห้ามตัดสัญลักษณ์ Timezone UTC ทิ้งโดยตรงเด็ดขาด
- **ค่าตัวเลขและสถานะ**: ราคาและปริมาณการซื้อขายต้องเป็นทศนิยมหรือจำนวนเต็มที่ไม่ติดลบ ส่วนสถานะให้ใช้ค่าบูลีน `Boolean` ทั้งหมด

### 2. คำจำกัดความโครงสร้างตารางหลักทั้ง 7 ตาราง

| ชื่อตาราง | ฟิลด์ที่จำเป็น (Primary Fields) | ความหมายของฟิลด์และข้อกำหนดด้าน Data Governance |
| :--- | :--- | :--- |
| **ปฏิทินการซื้อขาย** (`trading_calendar`) | `trade_date` | วันเปิดทำการของตลาดหลักทรัพย์ที่ผ่านการตรวจสอบแล้ว ครอบคลุมวงจรชีวิตการจดทะเบียนและการซื้อขายของตัวอย่างทดสอบทั้งหมด |
| **ทะเบียนหุ้น** (`stock_universe`) | `symbol, list_date, delist_date` | ต้องรวมหุ้นที่ถูกเพิกถอนหลักทรัพย์ (Delisting) ในอดีต (เพื่อป้องกัน Survivorship Bias); วันที่เพิกถอนคือขอบเขตการคัดออก ส่วนหุ้นที่ยังไม่ถูกเพิกถอนให้ระบุค่าว่างของ Date |
| **ตารางราคาตลาด** (`daily_quote`) | `symbol, trade_date, open, high, low, close, close_hfq, volume, turnover` | OHLC คือราคาดิบที่ยังไม่ปรับสิทธิ์; `close_hfq` จัดเก็บแยกต่างหาก; ปริมาณการซื้อขายเป็นหุ้น มูลค่าการซื้อขายเป็นหยวน; คีย์เอกลักษณ์คือ `(symbol, trade_date)` |
| **เหตุการณ์สถานะ** (`status_event`) | `symbol, effective_date, available_at, is_st, is_delisting` | รวมเครื่องหมาย ST และช่วงเตรียมการเพิกถอนในอดีต บันทึกแบบ Append-only รองรับการย้อนรอยแก้ไขข้อมูลในอดีต |
| **เวอร์ชันงบการเงิน** (`financial_pit`) | `symbol, report_period, available_at, roe, net_profit_yoy` | เก็บรักษาเวอร์ชันที่เปิดเผยครั้งแรกดั้งเดิมและเวอร์ชันที่มีการแก้ไขในเวลาต่อมาทั้งหมด; ห้ามไม่ให้เกิดการปนเปื้อนข้ามเวอร์ชันในการสืบค้นครั้งเดียวกัน |
| **เหตุการณ์อุตสาหกรรม** (`industry_event`) | `symbol, taxonomy, industry_name, effective_date, available_at` | บันทึกประวัติการเปลี่ยนแปลงการจัดกลุ่มอุตสาหกรรม; หุ้นที่ไม่มีการจับคู่กลุ่มให้ระบุแยกต่างหากเป็น `ind_未知` ห้ามจัดเข้าหมวด "เบ็ดเตล็ด" หรือเติมค่าตามอำเภอใจโดยเด็ดขาด |
| **สแนปชอตดัชนีอ้างอิง** (`index_weight`) | `symbol, asof_date, weight, available_at` | บันทึกแบบครบทั้ง Batch; ช่วงของ `weight` อยู่ระหว่าง $[0, 1]$; เวลาที่สังเกตได้จริงต้องไม่ย้อนหลังไปเป็นวันที่จัดทำดัชนี |
### 3. ความหมายของฟิลด์สำคัญและข้อจำกัดทางธุรกิจ
- **`effective_date` vs `available_at`**：
  - `effective_date` คือวันที่ทางธุรกิจที่สถานะมีผลบังคับใช้จริงในโลกความเป็นจริง
  - `available_at` คือเวลาทางกายภาพตามวัตถุวิสัยที่ระบบเชิงปริมาณ (Quant) **รับรู้ได้จริงและสามารถส่งคำสั่งได้** หลังจากข้อมูลได้รับการประมวลผลจากแหล่งข้อมูลต้นน้ำ ทั้งสองสิ่งนี้ห้ามสับสนปนเปกันโดยเด็ดขาด
- **กฎจำนวนวันทำการซื้อขายหลังเข้าตลาด**：
  - จำนวนวันทำการซื้อขายหลังเข้าตลาดจะนับทีละวันตามวันที่ตลาดหลักทรัพย์เปิดทำการ โดยนับรวมช่วงพักการซื้อขาย (Suspension) ด้วย และไม่เท่ากับจำนวนแถวของข้อมูลราคาตลาดที่ดาวน์โหลดมาอย่างแน่นอน หากปฏิทินขาดหายต้องแจ้ง Error ทันที
- **คุณสมบัติผู้มีสิทธิ์ใน Dynamic Universe (`in_universe`)**：
  - แสดงเพียงว่าหลักทรัพย์ดังกล่าวมีคุณสมบัติเป็นตัวเลือกในการคัดเลือกหุ้นในภาคตัดขวาง (Cross-sectional) นี้เท่านั้น ไม่ได้หมายความว่าเมื่อสถานะการถือครองถูกคัดออกแล้วจะสามารถแปลงเป็นเงินสดตามราคาล่าสุดได้ทันที
- **สมมติฐานภาคตัดขวางเวลาปิดตลาด 15:00**：
  - เอนจิน Point-in-Time (PIT) จำลองการคัดเลือกหุ้นหลังตลาดปิด โดยสมมติว่าข้อมูลราคารายวัน ณ เวลา 15:00 ก่อตัวสมบูรณ์แล้ว ในสภาพแวดล้อม Production จริงต้องคำนึงถึงความล่าช้าในการบันทึกข้อมูลเข้าฐานข้อมูล โดยสัญญาณที่สร้างขึ้นจะดำเนินการได้เร็วที่สุดในช่วงเวลาเปิดตลาดของวันถัดไป

---

## 🛡️ ขอบเขตการอิมพลีเมนต์และข้อกำหนดทางวิศวกรรมในแต่ละขั้นตอน

### 1. โมดูล 01–02：วิศวกรรมข้อมูลและเอนจินการสืบค้น PIT
- **กฎการเลือกเวอร์ชัน**：เมื่อสืบค้นงบการเงิน ให้กรองตามเวลาที่ระบุ `available_at <= query_time` ก่อน จากนั้นจึงเลือกเวอร์ชันแก้ไขล่าสุดที่มองเห็นได้ในขณะนั้นของแต่ละรอบระยะเวลารายงาน การอัปเดตรายงานประจำปีในอดีตจะไม่ทำให้ตัวชี้วัดทางการเงินของไตรมาสปัจจุบันย้อนกลับไปเป็นงวดเก่า
- **การจัดเก็บไฟล์ข้อมูลราคาตลาด**：รับประกันความเป็น Unique ของ Primary Key ตาม `(symbol, trade_date)` ตรวจสอบค่าจำกัด (Finite Value) และความสัมพันธ์ $Low \le Open, Close \le High$ อย่างเข้มงวดก่อนเขียนลงดิสก์ สำหรับไฟล์ Parquet รายปีให้ใช้การแทนที่แบบ Atomic (Atomic Rename) ด้วยไฟล์ชั่วคราว เพื่อป้องกันข้อมูลเสียหายจากการหยุดชะงักระหว่างการเขียน

### 2. โมดูล 03–04：Feature Engineering และการตรวจสอบประเมินแฟกเตอร์ (Factor)
- **เกณฑ์การคำนวณผลตอบแทน**：การคำนวณผลตอบแทนข้ามวันต้องใช้ราคาปรับสิทธิย้อนหลังแบบสะสม `close_hfq` เท่านั้น อัตราการเติบโตของงบการเงินเมื่อเทียบกับปีก่อนหน้า (YoY) ไม่ควรนำมาสวมรอยเป็นตัวชี้วัด SUE ที่ดีกว่าคาด
- **โครงสร้างจุลภาคความถี่สูง**：บทที่ 11 OFI ใช้ Local Timestamp ที่ไม่มี Timezone อย่างเคร่งครัด ไม่รองรับการเชื่อมต่ออย่างราบรื่นข้ามวันทำการซื้อขาย Snapshot ตามช่วงเวลาไม่เท่ากับเหตุการณ์คำสั่งซื้อขายแบบ Tick by Tick
- **Orthogonalization และ Neutralization**：บทที่ 12 ใช้ NumPy หาคำตอบแบบ Weighted Least Squares (WLS) เพื่อตรวจสอบว่า Residuals และ Constraint Vectors ตั้งฉากกันเชิงถ่วงน้ำหนัก สำหรับบทที่ 13 ก่อนทำ Orthogonalization ต้องแจ้ง Error อย่างชัดเจนเมื่อพบคอลัมน์ค่าคงที่หรืออินพุตที่เกิด Collinearity เพื่อขจัด Factor อิสระปลอม
- **การแยกส่วนการประเมิน**：บทที่ 14–15 เลเบลต้องถูกจัดเรียง (Align) โดยผู้เรียกใช้ตามจังหวะเวลาที่เกิดสัญญาณจริงและเวลาที่เกิดการจับคู่คำสั่ง (Matching) จริง ส่วนบทที่ 17 ในการตรวจสอบ Pseudo Alpha คำสั่งที่ยังไม่จับคู่ต้องคงไว้เป็นเงินสด เพื่อขจัดการ "สร้างกำไรมหาศาลสมมติจากการเปิดชนเพดาน (Limit-up) แบบแท่งเดียว"

### 3. โมดูล 05–06：แมชชีนเลิร์นนิงทางการเงินและการหาพอร์ตที่เหมาะสม (Portfolio Optimization)
- **การแยกส่วนทางเวลา**：ชุดข้อมูล Train และ Test ต้องใช้กลไก Purged และ Embargo ห้ามใช้ข้อมูลในอนาคตในการฟิตตัวแปลง Preprocessing ข้อมูล (เช่น Scaler/Imputer) โดยเด็ดขาด
- **การสอบเทียบผลตอบแทน**：เอาต์พุตการจัดอันดับของโมเดลต้นไม้ (คะแนน LambdaRank) ไม่สามารถส่งเข้าไปเป็นผลตอบแทนที่คาดหวังให้กับ Optimizer ได้โดยตรง ต้องผ่าน `RankReturnCalibrator` เพื่อฟิตผลตอบแทนส่วนเกิน (Excess Return) ของช่วงเวลาถือครองนอกกลุ่มตัวอย่าง (Out-of-sample) ในอดีตที่สมบูรณ์
- **Barra และการหาผลลัพธ์ Optimization**：
  - แบบจำลองความเสี่ยง (Risk Model) รับพาเนล Exposure ต้นงวดในอดีตขนาด $(T, N, K)$ และ Exposure ต้นงวดปัจจุบันที่เป็นอิสระ
  - Optimizer ใช้ตัวแก้ปัญหา Convex Optimization `Clarabel` เพื่อจัดการกับ Quadratic Hard Constraint ของ Tracking Error (TE) ต่อปี และตรวจสอบ Dual Residuals กับความเป็นไปได้ของคำตอบ (Feasibility) โดยอัตโนมัติหลังการหาคำตอบ
- **ข้อควรระวังเรื่องการแยกโพรเซส**：
  > [!WARNING]
  > การรันโมเดล Deep Learning ของ PyTorch ใน Python Process เดียวกันกับที่ฝึกสอน LightGBM ไปแล้ว อาจทำให้เกิดการบล็อกของ OpenMP / Convolution Thread Pool ในระดับล่าง แนะนำให้เรียกใช้บทที่ 21 และการทดสอบโมเดล Deep Learning ในโพรเซสแยกต่างหาก

### 4. โมดูล 07–08：การจับคู่คำสั่ง Backtest, Friction ในการซื้อขาย และการมอนิเตอร์การแจกแจงที่มา (Attribution)
- **วินัยในการจับคู่คำสั่ง**：การจับคู่คำสั่งตอนเปิดตลาดของวันถัดไปห้ามอ่านราคาสูงสุด ราคาต่ำสุด หรือราคาปิดของวันนั้นโดยเด็ดขาด ปริมาณที่จับคู่สำเร็จต้องอยู่ภายใต้ Hard Constraint ของขีดจำกัดสัดส่วนปริมาณการซื้อขายเฉลี่ยต่อวัน (ADV Limit) คำสั่งที่ยังไม่จับคู่จะคงค้างอยู่ในพอร์ตหรือเงินสดต่อไป และหุ้นที่จับคู่สำเร็จในวันนั้นจะถูกล็อกตามเกณฑ์ T+1 อย่างเคร่งครัด
- **การเปลี่ยนแปลงตาม 4 ลำดับเวลาของเงินปันผลและการเพิ่มทุน/แจกหุ้นปันผล**：ล็อกสิทธิประโยชน์เมื่อปิดตลาดในวันกำหนดรายชื่อผู้ถือหุ้น (Record Date), ยืนยันเงินปันผลค้างรับและล็อกหุ้นปันผลในวันขึ้นเครื่องหมาย XD/XR, ปลดล็อกเงินสดและหุ้นที่ใช้ได้จริงในวันจ่ายเงินปันผลและวันเข้าซื้อขายของหุ้นใหม่ตามลำดับ เพื่อให้เกิด Idempotency ของสมุดบัญชี
- **ความสมบูรณ์ของการแจกแจงที่มาและ Hysteresis ในการควบคุมความเสี่ยง**：บทที่ 30 การแจกแจงที่มาจะรวมเงินสดคงเหลือและค่าธรรมเนียมการซื้อขายอย่างชัดเจน เพื่อให้แน่ใจว่าผลตอบแทนของพอร์ตและรายการย่อยของ Attribution สอดคล้องตามสมการอย่างสมบูรณ์ ส่วนบทที่ 31 การมอนิเตอร์ควบคุมความเสี่ยงจะใช้กลไก Circuit Breaker แบบ Hysteresis เมื่อทริกเกอร์การแจ้งเตือนระดับ 3 ซึ่งเป็นระดับสูงสุด จะห้ามเปิดคำสั่งซื้อใหม่ และต้องรีเซ็ตตามเงื่อนไขที่กำหนดไว้อย่างชัดเจนพร้อมทั้งบันทึกการอ้างอิงการอนุมัติ

---

## 📂 โครงสร้างไดเรกทอรีของโปรเจกต์

```text
quant-handbook/
├── .vitepress/              # การกำหนดค่าไซต์ VitePress, ปลั๊กอิน Mermaid และนิยามโครงสร้างหลักสูตร
│   ├── config.mts           # การกำหนดค่าระดับโกลบอลของไซต์และการนำทาง
│   └── curriculum.ts        # แหล่งข้อมูลเค้าโครงหลักสูตร 8 โมดูล 31 บท
├── examples/                # ซอร์สโค้ด Python อิสระและไลบรารีเครื่องมือส่วนกลางสำหรับ 31 บท
│   ├── common.py            # การตรวจสอบ Data Contract และ Assertion ทั่วไป
│   ├── fixtures.py          # ตัวสร้างชุดข้อมูลสังเคราะห์แบบออฟไลน์
│   ├── factor_utils.py      # เครื่องมือคำนวณแฟกเตอร์และ Weighted Neutralization
│   ├── ml_utils.py          # เครื่องมือตรวจสอบแมชชีนเลิร์นนิงและการสอบเทียบผลตอบแทน
│   ├── portfolio_utils.py   # เครื่องมือ Portfolio Optimization และการแจกแจงที่มาความเสี่ยง
│   ├── lecture_01_*.py      # ตัวอย่างที่รันได้ประกอบบทที่ 1
│   └── ...                  # ซอร์สโค้ดตัวอย่างที่สอดคล้องกับบทที่ 2 ~ 31
├── module_01/ ~ module_08/  # เนื้อหา Markdown ของบทเรียนแบบละเอียดทั้ง 31 บท
├── tests/                   # ชุดแบบทดสอบอัตโนมัติ Regression Testing (ผ่านแบบออฟไลน์ 100%)
│   ├── test_tutorial.py             # Unit Test สำหรับโมดูล 01–02
│   ├── test_factor_modules.py       # Unit Test สำหรับโมดูล 03–04
│   ├── test_ml_risk_modules.py      # Unit Test สำหรับโมดูล 05–06
│   └── test_execution_monitoring.py # Unit Test สำหรับโมดูล 07–08
├── index.md                 # หน้าหลักของเว็บไซต์
├── learning_path.md         # เส้นทางการเรียนรู้ แบบทดสอบประเมินตนเองเบื้องต้น และเกณฑ์การประเมินความเข้าใจทีละบท
├── resources.md             # รายการหนังสืออ้างอิงสาธารณะ หลักสูตรมหาวิทยาลัย และคู่มือกฎเกณฑ์ทางการ
├── syllabus.md              # เค้าโครงภาพรวมหลักสูตรฉบับสมบูรณ์
├── package.json             # นิยามสคริปต์ส่วนหน้าและการทดสอบระดับโกลบอล
├── pyproject.toml           # การกำหนดค่า Dependencies และ Extras ของโปรเจกต์ Python
└── uv.lock                  # ไฟล์ล็อก Dependency อย่างแม่นยำ
```

---

## 📚 คู่มือเพิ่มเติมและเอกสารอ้างอิง

### คู่มือเอกสารหลักภายในไซต์
- [เส้นทางการเรียนรู้และเกณฑ์การประเมินความเข้าใจทีละบท](./learning_path.md)：รวบรวมแบบทดสอบประเมินตนเองเบื้องต้นเกี่ยวกับพื้นฐานทั้ง 6 ด้านของ Quant, เส้นทางการเรียนรู้คู่ขนาน 3 เส้นทาง และรายการตรวจสอบหลักฐานผลลัพธ์การส่งมอบทีละบท
- [คู่มือแนะนำแหล่งข้อมูลสาธารณะและหนังสืออ้างอิง](./resources.md)：รวบรวมรายการตรวจสอบ เช่น MIT 18.06 (พีชคณิตเชิงเส้น), Boyd Convex Optimization, กฎการซื้อขายล่าสุดของตลาดหลักทรัพย์เซี่ยงไฮ้, ข้อกำหนดอินเทอร์เฟซ AKShare
- [เค้าโครงภาพรวมหลักสูตรฉบับสมบูรณ์](./syllabus.md)：แผนผังความรู้ฉบับสมบูรณ์ 31 บท, แนวคิดการออกแบบในแต่ละบท และเป้าหมายขั้นสูง
### เอกสารทางการและระบบนิเวศโอเพนซอร์ส (Open Source Ecosystem)
- **สแตกวิศวกรรมสมัยใหม่ (Modern Engineering Stack)**: [เอกสารทางการ Polars](https://docs.pola.rs/) · [เอกสารทางการ DuckDB](https://duckdb.org/) · [ตัวจัดการแพ็กเกจ uv (Package Manager)](https://docs.astral.sh/uv/)
- **การหาค่าที่เหมาะสมและการสร้างโมเดล (Optimization & Modeling)**: [ไลบรารี Convex Optimization CVXPY](https://www.cvxpy.org/) · [เอกสาร LightGBM](https://lightgbm.readthedocs.io/) · [ความสามารถในการอธิบายผล SHAP (SHAP Interpretability)](https://shap.readthedocs.io/)
- **กฎระเบียบตลาดและเกณฑ์มาตรฐาน (Market Structure & Benchmark)**: [กฎการซื้อขายตลาดหลักทรัพย์เซี่ยงไฮ้ (SSE)](https://www.sse.com.cn/) · [เกณฑ์การคำนวณและดูแลรักษาดัชนี CSI](https://www.csindex.com.cn/) · [ข้อมูลทางการเงินโอเพนซอร์ส AKShare](https://akshare.akfamily.xyz/)

---

<p align="center">
  การวิจัยเชิงปริมาณ (Quant Research) ที่ยึดมั่นความจริงและเน้นปฏิบัติจริง · รอบคอบรัดกุมและยำเกรงต่อตลาด · สร้างผลตอบแทนทบต้นอย่างมั่นคง
</p>
