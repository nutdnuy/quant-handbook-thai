"""
ตัวอย่างการสอนตรวจสอบความเสี่ยงแฟกเตอร์ (Factor) ในหุ้น A-Share (AshareFakeAlphaSanitizer)
ประกอบด้วย: การทดสอบ Market Cap Parasitism, การตรวจสอบ Survivorship Bias จากการซื้อที่ราคาชนเพดาน (Limit-up), การตรวจสอบความทนทานแยกตามกลุ่ม Market Cap และการตรวจสอบโดยรวม
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
from scipy import stats
from examples.factor_utils import validate_panel


@dataclass
class ForensicReport:
    """รายงานการวินิจฉัยและตรวจสอบแฟกเตอร์ปลอมระดับ Forensic"""
    factor_name: str
    raw_rank_ic: float
    neutralized_rank_ic: float
    style_ic_drop_ratio: float
    raw_top_annual_ret: float
    realistic_tradeable_top_ret: float
    tradability_haircut_ratio: float
    large_cap_ic: float
    mid_cap_ic: float
    micro_cap_ic: float
    verdict: str
    reasons: List[str]

    def print_forensic_dossier(self):
        """จัดรูปแบบและพิมพ์รายงานผลการตรวจสอบระดับ Forensic"""
        print("=" * 75)
        print(f"🔬 因子法医级审计排查卷宗: [{self.factor_name}]")
        print("=" * 75)
        status_symbol = "✅ [PASS - 本组有限检查未触发预警]" if self.verdict == "PASS" else "⚠ [REVIEW - 需要进一步研究]"
        print(f"【最终审计裁决】: {status_symbol}")
        if self.reasons:
            print("【拦截与风险预警原因】:")
            for r in self.reasons:
                print(f"  🚨 {r}")
        print("-" * 75)
        print("【Test 1: 市值风格隐形寄生渗透测试】")
        print(f"  • 原始未中性化 Rank IC   : {self.raw_rank_ic:8.4f}")
        print(f"  • 仅市值中性化后 Rank IC: {self.neutralized_rank_ic:8.4f}")
        print(f"  • IC 缩水流失率 (Drop)   : {self.style_ic_drop_ratio * 100:7.2f}% (警报阈值: > 60%)")
        print("-" * 75)
        print("【Test 2: 涨停打板可交易性穿透测试】")
        print(f"  • 理想假设下多头年化收益 : {self.raw_top_annual_ret * 100:8.2f}%")
        print(f"  • 剔除涨停无法买入后收益 : {self.realistic_tradeable_top_ret * 100:8.2f}%")
        print(f"  • 可交易性利润折让率     : {self.tradability_haircut_ratio * 100:7.2f}% (警报阈值: > 40%)")
        print("-" * 75)
        print("【Test 3: 分市值池稳健性检查（不是容量测算）】")
        print(f"  • 头部大盘股池 Rank IC (Top 30% 市值)   : {self.large_cap_ic:8.4f}")
        print(f"  • 中盘核心股池 Rank IC (Mid 40% 市值)   : {self.mid_cap_ic:8.4f}")
        print(f"  • 尾部微盘股池 Rank IC (Bottom 30% 市值): {self.micro_cap_ic:8.4f}")
        print("=" * 75)


class AshareFakeAlphaSanitizer:
    """
    เครื่องมือตรวจสอบแฟกเตอร์ปลอมในหุ้น A-Share ระดับ Forensic
    """

    def __init__(
        self,
        date_col: str = "trade_date",
        symbol_col: str = "symbol",
        cap_col: str = "market_cap",
        ret_col: str = "ret_fwd_1d",
        is_limit_up_col: str = "is_limit_up_next_day"
    ):
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.cap_col = cap_col
        self.ret_col = ret_col
        self.is_limit_up_col = is_limit_up_col

    def _calc_rank_ic(self, df: pl.DataFrame, col1: str, col2: str) -> float:
        """คำนวณค่าเฉลี่ย Rank IC ภาคตัดขวาง (Cross-sectional) ของสองคอลัมน์อย่างรวดเร็ว"""
        df = df.filter(pl.col(col1).is_finite() & pl.col(col2).is_finite())
        ranked = df.with_columns([
            pl.col(col1).rank().over(self.date_col).alias("r1"),
            pl.col(col2).rank().over(self.date_col).alias("r2")
        ])
        daily_corr = ranked.group_by(self.date_col).agg(pl.corr("r1", "r2").alias("ic"), pl.len().alias("n"))
        values = daily_corr.filter((pl.col("n") >= 3) & pl.col("ic").is_finite())["ic"]
        return float(values.mean()) if len(values) else float("nan")

    def audit_factor(
        self,
        df: pl.DataFrame,
        factor_col: str
    ) -> ForensicReport:
        """
        ดำเนินการวินิจฉัยเพื่อการสอน 3 ชุดกับแฟกเตอร์ทิศทางบวกที่กำหนด โดยค่า Threshold เป็นค่าคอนฟิกตัวอย่าง
        """
        validate_panel(df, self.date_col, self.symbol_col)
        # ตรวจสอบความสมบูรณ์ของ Label หลังตรึง Signal Pool โดยไม่ใช้การซื้อขายได้จริงในอนาคตมาเปลี่ยนการเลือกหุ้น
        df = df.filter(pl.col(factor_col).is_finite())
        if df.is_empty():
            raise ValueError("没有有效因子")
        if df.select(pl.any_horizontal([
            (~pl.col(c).is_finite()).fill_null(True) for c in [self.cap_col, self.ret_col]
        ]).any()).item():
            raise ValueError("市值或前瞻收益缺失，不能重新分配缺失样本权重")
        if df.filter((pl.col(self.cap_col) <= 0) | (pl.col(self.ret_col) < -1)).height:
            raise ValueError("市值必须为正，简单收益率不能小于 -100%")
        if df.select((~pl.col(self.is_limit_up_col).is_in([0, 1])).fill_null(True).any()).item():
            raise ValueError("成交阻断标记必须已知且为 0/1")
        reasons = []

        # ----------------------------------------------------
        # Test 1: การทดสอบ Residual ของ Market Cap Style (แยกส่วนด้วย Neutralization)
        # ----------------------------------------------------
        raw_ic = self._calc_rank_ic(df, factor_col, self.ret_col)

        # ดึง Residual จาก OLS เทียบกับ Log Market Cap
        neutral_records = []
        dates = df[self.date_col].unique().sort().to_list()
        
        for d in dates:
            sub = df.filter(pl.col(self.date_col) == d)
            y = sub[factor_col].to_numpy()
            x = np.log(np.maximum(sub[self.cap_col].to_numpy(), 1e-4))
            # Neutralization ภาคตัดขวางแบบง่าย: y = alpha + beta * ln_cap + residual
            X_mat = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
            residuals = y - X_mat @ beta
            if np.std(residuals) <= 1e-10:
                residuals[:] = np.nan  # ไม่หา Rank ของ Residual ทศนิยม หาก Cross-section นั้นถูกอธิบายด้วย Style ทั้งหมด

            sub_with_res = sub.with_columns(
                pl.Series("factor_neutral", residuals)
            )
            neutral_records.append(sub_with_res)

        neutral_df = pl.concat(neutral_records)
        neutral_ic = self._calc_rank_ic(neutral_df, "factor_neutral", self.ret_col)

        ic_drop = (raw_ic - neutral_ic) / raw_ic if np.isfinite(raw_ic) and abs(raw_ic) > 1e-12 else float("nan")
        if not np.isfinite(raw_ic) or not np.isfinite(neutral_ic):
            reasons.append("原始或市值残差 IC 不可估计（样本不足、常数截面或完全被风格解释）")
        if ic_drop > 0.60:
            reasons.append(
                f"市值寄生极度严重: 中性化后 Rank IC 从 {raw_ic:.4f} 断崖下跌至 {neutral_ic:.4f} (缩水 {ic_drop*100:.1f}%)"
            )

        # ----------------------------------------------------
        # Test 2: การทดสอบ Tradability ของการไล่ซื้อหุ้นชนเพดาน (Limit-up)
        # ----------------------------------------------------
        # พอร์ตเป้าหมาย: เลือก ceil(N*10%) ตามแฟกเตอร์ที่เห็น ณ ขณะนั้น หากคะแนนเท่ากันให้เรียงตาม symbol อย่างเสถียร
        top_df = df.sort([self.date_col, self.symbol_col]).with_columns(
            pl.col(factor_col).rank(method="ordinal", descending=True).over(self.date_col).alias("_rank"),
            (pl.len().over(self.date_col) * .10).ceil().cast(pl.Int64).alias("_top_n")
        )
        ideal_top = top_df.filter(pl.col("_rank") <= pl.col("_top_n"))
        ideal_daily_ret = ideal_top.group_by(self.date_col).agg(pl.col(self.ret_col).mean().alias("r"))
        ideal_ann_ret = float(ideal_daily_ret["r"].mean() * 250.)

        # ในแต่ละวันเป็นการทดลองส่งคำสั่งอิสระจากเงินสด น้ำหนักที่ไม่ Match จะคงเป็นเงินสดไว้ (ผลตอบแทนเป็น 0)
        # แฟล็กสถานะมาจากการจำลอง/การเทรดจริง (Live Trading) ณ เวลาส่งคำสั่ง ห้ามใช้ราคาสูงสุดของวันหรือผลตอบแทนหลังเกิดเหตุการณ์มาแทนที่
        tradeable_daily_ret = ideal_top.group_by(self.date_col).agg(
            pl.when(pl.col(self.is_limit_up_col) == 0).then(pl.col(self.ret_col))
            .otherwise(0.).mean().alias("r")
        )
        tradeable_ann_ret = float(tradeable_daily_ret["r"].mean() * 250.)

        haircut = (ideal_ann_ret - tradeable_ann_ret) / abs(ideal_ann_ret) if abs(ideal_ann_ret) > 1e-12 else float("nan")
        if haircut > 0.40:
            reasons.append(
                f"涨停打板幸存者陷阱: 剔除无法买入的涨停股票后，多头年化收益从 {ideal_ann_ret*100:.1f}% 骤降至 {tradeable_ann_ret*100:.1f}% (折让 {haircut*100:.1f}%)"
            )

        # ----------------------------------------------------
        # Test 3: การตรวจสอบความทนทานแยกตามกลุ่ม Market Cap (ไม่ใช่การวัด Capacity)
        # ----------------------------------------------------
        # แบ่งหุ้นทั้งตลาดออกเป็น 3 ส่วนเท่าๆ กันตาม Market Cap
        bucket_df = df.with_columns(
            ((pl.col(self.cap_col).rank().over(self.date_col) - 1.0) /
             pl.len().over(self.date_col)).alias("cap_bucket")
        )

        micro_sub = bucket_df.filter(pl.col("cap_bucket") < 0.30)
        mid_sub = bucket_df.filter((pl.col("cap_bucket") >= 0.30) & (pl.col("cap_bucket") < 0.70))
        large_sub = bucket_df.filter(pl.col("cap_bucket") >= 0.70)

        micro_ic = self._calc_rank_ic(micro_sub, factor_col, self.ret_col)
        mid_ic = self._calc_rank_ic(mid_sub, factor_col, self.ret_col)
        large_ic = self._calc_rank_ic(large_sub, factor_col, self.ret_col)

        if not np.isfinite(large_ic) or large_ic < 0.015:
            reasons.append(
                f"大市值子池 Rank IC 为 {large_ic:.4f}，需要复核风格依赖；不能据此推断资金容量。"
            )

        verdict = "REVIEW" if reasons else "PASS"

        return ForensicReport(
            factor_name=factor_col,
            raw_rank_ic=raw_ic,
            neutralized_rank_ic=neutral_ic,
            style_ic_drop_ratio=ic_drop,
            raw_top_annual_ret=ideal_ann_ret,
            realistic_tradeable_top_ret=tradeable_ann_ret,
            tradability_haircut_ratio=haircut,
            large_cap_ic=large_ic,
            mid_cap_ic=mid_ic,
            micro_cap_ic=micro_ic,
            verdict=verdict,
            reasons=reasons
        )


# ================= Unit Test และการสาธิตการตรวจสอบระดับ Forensic =================
if __name__ == "__main__":
    np.random.seed(42)
    n_days = 150
    n_stocks = 500

    import pandas as pd
    dates = pd.bdate_range("2025-01-01", periods=n_days).date.tolist()
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]

    records = []
    # จำลอง "แฟกเตอร์เทพจอมปลอม" ทั่วไป:
    # สังเคราะห์ Market Cap Exposure และการติด Price Limit จนซื้อไม่ได้ โดย IC และผลตอบแทนที่แน่นอนขึ้นกับผลลัพธ์จริง
    for d in dates:
        # การกระจายตัวของ Log Market Cap (3 พันล้าน ถึง 5 แสนล้าน)
        market_caps = np.exp(np.random.uniform(np.log(30), np.log(5000), n_stocks))
        
        # ผลของส่วนต่างผลตอบแทนหุ้นขนาดจิ๋ว (ยิ่ง Market Cap เล็ก ผลตอบแทนคาดหวังยิ่งสูง)
        size_premium = -np.log(market_caps) * 0.002
        stock_returns = size_premium + np.random.normal(0, 0.02, n_stocks)

        # สร้างแฟกเตอร์ปลอม: มีสหสัมพันธ์เชิงลบระดับสูงกับ Market Cap (แฝงตัวเป็นแฟกเตอร์ Growth/Momentum บางประเภท)
        fake_factor = -np.log(market_caps) * 1.5 + np.random.normal(0, 0.5, n_stocks)

        # จำลองการติด Limit ตอนเปิดตลาดก่อน แล้วจึงสร้างผลตอบแทนตามช่วงถือครอง ไม่ใช้ผลตอบแทนในอนาคตมาตัดสินว่าจะ Match คำสั่งได้หรือไม่
        is_limit_up = (np.random.uniform(size=n_stocks) < .15).astype(int)
        stock_returns = stock_returns + .06 * is_limit_up

        for s_idx, s in enumerate(symbols):
            records.append({
                "trade_date": d,
                "symbol": s,
                "market_cap": market_caps[s_idx],
                "fake_volatility_factor": fake_factor[s_idx],
                "ret_fwd_1d": stock_returns[s_idx],
                "is_limit_up_next_day": is_limit_up[s_idx]
            })

    test_df = pl.DataFrame(records)

    # รันเครื่องมือตรวจสอบระดับ Forensic
    sanitizer = AshareFakeAlphaSanitizer()
    dossier = sanitizer.audit_factor(test_df, factor_col="fake_volatility_factor")

    dossier.print_forensic_dossier()
