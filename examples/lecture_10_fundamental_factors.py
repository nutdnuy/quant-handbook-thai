"""
lecture_10_fundamental_factor_engine.py
เอนจินคำนวณแบบเวกเตอร์สำหรับแฟกเตอร์ (Factor) ปัจจัยพื้นฐานและคุณภาพการเติบโตแบบ DuPont ระดับฝั่ง Buy-side
"""
import time
from datetime import date
import numpy as np
import polars as pl
import pandas as pd

class AshareFundamentalFactorEngine:
    def __init__(self):
        pass

    def compute_fundamental_factors(
        self,
        snapshot_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Input ต้องประกอบด้วยตารางแบบกว้างระดับภาคตัดขวาง (Cross-sectional) ที่ผ่านการจัดเรียงแบบ PIT:
        [symbol, trade_date, market_cap, net_profit_ttm, revenue_ttm, 
         total_equity, total_assets, ocf_ttm, net_profit_q, net_profit_ly_same_q]
        """
        print("[*] 正在计算买方基本面与杜邦质量特征矩阵...")
        t0 = time.perf_counter()

        factors_df = snapshot_df.lazy().with_columns([
            # ================= 1. ระบบการประเมินมูลค่า (EP / BP / OCFP) =================
            # ส่วนกลับของ P/E คือ EP (Earnings Yield)
            (pl.col("net_profit_ttm") / pl.when(pl.col("market_cap") > 0).then(pl.col("market_cap")).otherwise(None)).alias("factor_ep_ttm"),
            # ส่วนกลับของ P/B คือ BP (Book-to-Market Ratio)
            (pl.col("total_equity") / pl.when(pl.col("market_cap") > 0).then(pl.col("market_cap")).otherwise(None)).alias("factor_bp"),
            # อัตราผลตอบแทนกระแสเงินสดจากการดำเนินงาน OCFP
            (pl.col("ocf_ttm") / pl.when(pl.col("market_cap") > 0).then(pl.col("market_cap")).otherwise(None)).alias("factor_ocfp"),

            # ================= 2. การแยกองค์ประกอบ 3 แฟกเตอร์ตามแบบ DuPont =================
            # อัตรากำไรสุทธิ (Net Profit Margin)
            (pl.col("net_profit_ttm") / pl.when(pl.col("revenue_ttm") > 0).then(pl.col("revenue_ttm")).otherwise(None)).alias("dupont_margin"),
            # อัตราการหมุนเวียนของสินทรัพย์รวม (Asset Turnover)
            (pl.col("revenue_ttm") / pl.when(pl.col("total_assets") > 0).then(pl.col("total_assets")).otherwise(None)).alias("dupont_turnover"),
            # ตัวคูณส่วนของผู้ถือหุ้น (Financial Leverage)
            (pl.col("total_assets") / pl.when(pl.col("total_equity") > 0).then(pl.col("total_equity")).otherwise(None)).alias("dupont_leverage"),
            # ตัวชี้วัด ROE รวมขั้นสุดท้าย
            (pl.col("net_profit_ttm") / pl.when(pl.col("total_equity") > 0).then(pl.col("total_equity")).otherwise(None)).alias("roe_ttm"),

            # ================= 3. ความแท้จริงของกำไรและคุณภาพคงค้าง (แฟกเตอร์ตรวจจับความเสี่ยง) =================
            # อัตราส่วนกระแสเงินสดต่อกำไรสำหรับบริษัทที่มีกำไรเป็นบวก; ไม่สามารถใช้ตัดสินความโปร่งใสทางบัญชีได้โดยตรง
            (pl.col("ocf_ttm") / pl.when(pl.col("net_profit_ttm") > 0).then(pl.col("net_profit_ttm")).otherwise(None)).alias("cash_to_profit_ratio"),
            # สัดส่วนกำไรคงค้างต่อสินทรัพย์รวม (ใส่เครื่องหมายลบ ยิ่งต่ำยิ่งดี)
            (
                -(pl.col("net_profit_ttm") - pl.col("ocf_ttm")) / pl.when(pl.col("total_assets") > 0).then(pl.col("total_assets")).otherwise(None)
            ).alias("neg_accrual_ratio"),

            # ================= 4. การเติบโตเทียบช่วงเดียวกันของปีก่อนหน้า (ไม่ใช่ SUE ที่ผ่าน Standardization) =================
            # อัตราการเติบโต YoY ของกำไรสุทธิรายไตรมาส
            (
                (pl.col("net_profit_q") - pl.col("net_profit_ly_same_q")) /
                pl.when(pl.col("net_profit_ly_same_q").abs() > 0).then(pl.col("net_profit_ly_same_q").abs()).otherwise(None)
            ).alias("profit_yoy_growth")
        ]).collect()

        elapsed = time.perf_counter() - t0
        print(f"✅ 基本面特征矩阵计算完成！耗时: {elapsed:.3f} 秒")
        return factors_df

# =========================================================================
# การทดสอบความถูกต้องระดับยูนิต
# =========================================================================
if __name__ == "__main__":
    # จำลองข้อมูล 3 บริษัทตัวแทนที่มีโมเดลธุรกิจแตกต่างกันอย่างสิ้นเชิง
    # บริษัท A: บริษัทกำไรสูงเชิงสังเคราะห์ (Net Profit Margin สูงเป็นพิเศษ, Turnover สูง, Leverage ต่ำ)
    # บริษัท B: ซูเปอร์มาร์เก็ตแบบดั้งเดิมที่เน้นหมุนเวียนสินค้าเร็ว (เน้นขายเยอะกำไรต่อชิ้นต่ำ, Turnover สูง, Leverage ปานกลาง)
    # บริษัท C: บริษัทอสังหาริมทรัพย์ที่มีปัญหาหนี้สินและ Leverage สูง (จ่อขาดทุน, หนี้สินมหาศาลใช้ Leverage พยุง ROE, กระแสเงินสดติดลบ)
    
    mock_pit_data = pl.DataFrame([
        {
            "symbol": "DEMO_A", # สินทรัพย์สมมติ
            "trade_date": date(2023, 10, 16),
            "market_cap": 22000.0, # 2.2 ล้านล้าน
            "net_profit_ttm": 700.0,
            "revenue_ttm": 1400.0,
            "total_equity": 2100.0,
            "total_assets": 2600.0,
            "ocf_ttm": 780.0, # กระแสเงินสดสภาพคล่องสูงมาก
            "net_profit_q": 175.0,
            "net_profit_ly_same_q": 145.0
        },
        {
            "symbol": "DEMO_B", # บริษัทค้าปลีกสมมติ
            "trade_date": date(2023, 10, 16),
            "market_cap": 300.0,
            "net_profit_ttm": 6.0,
            "revenue_ttm": 900.0,
            "total_equity": 80.0,
            "total_assets": 450.0,
            "ocf_ttm": 35.0,
            "net_profit_q": 1.5,
            "net_profit_ly_same_q": 1.125
        },
        {
            "symbol": "DEMO_C", # บริษัท Leverage สูงสมมติ
            "trade_date": date(2023, 10, 16),
            "market_cap": 200.0,
            "net_profit_ttm": 15.0,
            "revenue_ttm": 500.0,
            "total_equity": 30.0,  # ส่วนของผู้ถือหุ้นบางมาก
            "total_assets": 1200.0, # หนี้สินระดับล้านล้าน
            "ocf_ttm": -80.0, # กระแสเงินสดเหือดแห้ง
            "net_profit_q": 3.75,
            "net_profit_ly_same_q": 10.0
        }
    ])

    engine = AshareFundamentalFactorEngine()
    result = engine.compute_fundamental_factors(mock_pit_data)

    print("=" * 95)
    print("三类不同商业质量标的的杜邦拆解与基本面特征对比：")
    print("=" * 95)
    cols = ["symbol", "factor_ep_ttm", "roe_ttm", "dupont_margin", "dupont_turnover", "dupont_leverage", "cash_to_profit_ratio"]
    print(result.select(cols))
