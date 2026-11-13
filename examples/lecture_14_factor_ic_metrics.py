"""
ตัวอย่างการสอนสำหรับหุ้น A-Share: เอนจินประเมินค่า Information Coefficient (IC) ของแฟกเตอร์ (Factor)
รองรับ: การคำนวณแบบขนานระดับภาคตัดขวาง (Cross-sectional) ของ Pearson IC / Spearman Rank IC, ค่าสถิติอนุกรมเวลา (Time-series) และการทดสอบสมมติฐาน
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import polars as pl
from scipy import stats
from examples.factor_utils import validate_panel, mean_test


@dataclass
class ICReport:
    """รายงานสรุปการประเมิน IC ของ Factor"""
    factor_name: str
    sample_days: int
    mean_ic: float
    ic_std: float
    icir: float
    annualized_icir: float
    t_stat: float
    p_value: float
    positive_ratio: float
    mean_rank_ic: float
    rank_ic_std: float
    rank_icir: float
    annualized_rank_icir: float
    rank_t_stat: float
    rank_p_value: float
    rank_positive_ratio: float
    skewness: float
    kurtosis: float
    excluded_days: int
    hac_lags: int

    def print_summary(self):
        """พิมพ์รายงานสรุปผลการประเมินในรูปแบบที่จัดไว้"""
        print("=" * 65)
        print(f"📊 因子全面 IC 评估报告: [{self.factor_name}] (评估交易日: {self.sample_days} 天)")
        print("=" * 65)
        print(f"无效截面: {self.excluded_days}；均值检验 HAC lag={self.hac_lags}（0 表示 iid t）")
        print(f"【Spearman Rank IC (买方核心)】")
        print(f"  • Rank IC 均值         : {self.mean_rank_ic:8.4f}")
        print(f"  • Rank IC 标准差       : {self.rank_ic_std:8.4f}")
        print(f"  • 日频 Rank ICIR       : {self.rank_icir:8.4f}")
        print(f"  • 年化 Rank ICIR       : {self.annualized_rank_icir:8.4f}")
        print(f"  • 均值检验统计量   : {self.rank_t_stat:8.4f} (p值: {self.rank_p_value:.4e})")
        print(f"  • 正 IC 比例 (胜率)    : {self.rank_positive_ratio * 100:7.2f}%")
        print("-" * 65)
        print(f"【Pearson Normal IC (线性参考)】")
        print(f"  • Pearson IC 均值      : {self.mean_ic:8.4f}")
        print(f"  • 年化 Pearson ICIR    : {self.annualized_icir:8.4f}")
        print(f"  • t 统计量             : {self.t_stat:8.4f} (p值: {self.p_value:.4e})")
        print(f"  • 胜率                 : {self.positive_ratio * 100:7.2f}%")
        print("-" * 65)
        print(f"【分布形态与稳健性】")
        print(f"  • IC 偏度 (Skewness)   : {self.skewness:8.4f}")
        print(f"  • IC 峰度 (Kurtosis)   : {self.kurtosis:8.4f}")
        print("=" * 65)


class AshareFactorICEngine:
    """
    เอนจินประเมินค่า Information Coefficient ของ Factor หุ้น A-Share
    """

    def __init__(
        self,
        date_col: str = "trade_date",
        symbol_col: str = "symbol",
        annual_factor: float = 250.0,
        hac_lags: int = 0
    ):
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.annual_factor = annual_factor
        self.hac_lags = hac_lags

    def compute_daily_ic(
        self,
        df: pl.DataFrame,
        factor_col: str,
        forward_ret_col: str
    ) -> pl.DataFrame:
        """
        คำนวณ Pearson IC และ Spearman Rank IC รายวันแบบขนานอย่างมีประสิทธิภาพบนภาคตัดขวางด้วย Polars expression
        """
        validate_panel(df, self.date_col, self.symbol_col)
        # 1. กรองค่าว่าง, NaN และค่าอนันต์
        valid_df = df.filter(
            pl.col(factor_col).is_not_null() & 
            pl.col(forward_ret_col).is_not_null() &
            pl.col(factor_col).is_finite() &
            pl.col(forward_ret_col).is_finite()
        )

        # 2. จัดอันดับ Rank บนภาคตัดขวางของแต่ละวัน
        ranked_df = valid_df.with_columns([
            pl.col(factor_col).rank().over(self.date_col).alias("factor_rank"),
            pl.col(forward_ret_col).rank().over(self.date_col).alias("ret_rank")
        ])

        # 3. จัดกลุ่มตามวันที่และรวมข้อมูลเพื่อคำนวณค่าสหสัมพันธ์
        daily_ic_df = (
            ranked_df.group_by(self.date_col)
            .agg([
                # คำนวณจำนวนหุ้นตัวอย่าง
                pl.len().alias("stock_count"),
                # ค่าสหสัมพันธ์เชิงเส้นของ Pearson
                pl.corr(factor_col, forward_ret_col).alias("ic"),
                # ค่าสหสัมพันธ์อันดับของ Spearman (ค่าสหสัมพันธ์ Pearson หลังจัดอันดับ)
                pl.corr("factor_rank", "ret_rank").alias("rank_ic")
            ])
            .sort(self.date_col)
        )

        # คงทุกวันที่ไว้และระบุภาคตัดขวางที่ไม่สมบูรณ์อย่างชัดเจน เส้นสะสมจะไม่ถูกรบกวนด้วย NaN
        daily_ic_df = df.select(self.date_col).unique().join(daily_ic_df, on=self.date_col, how="left").sort(self.date_col)
        daily_ic_df = daily_ic_df.with_columns([
            pl.col("stock_count").fill_null(0),
            *[pl.when((pl.col("stock_count") >= 3) & pl.col(c).is_finite())
              .then(pl.col(c)).otherwise(None).alias(c) for c in ["ic", "rank_ic"]]
        ]).with_columns([
            (pl.col("ic").is_not_null() & pl.col("rank_ic").is_not_null()).alias("valid_ic"),
            pl.col("ic").fill_null(0).cum_sum().alias("cum_ic"),
            pl.col("rank_ic").fill_null(0).cum_sum().alias("cum_rank_ic")
        ])

        return daily_ic_df

    def evaluate(
        self,
        daily_ic_df: pl.DataFrame,
        factor_name: str
    ) -> ICReport:
        """
        คำนวณชุดค่าสถิติ Time-series แบบครบวงจรและการทดสอบความเชื่อมั่นจากอนุกรม IC รายวัน
        """
        valid = daily_ic_df.filter(pl.col("ic").is_finite() & pl.col("rank_ic").is_finite())
        if self.hac_lags and valid.height != daily_ic_df.height:
            raise ValueError("HAC 需要连续有效截面；请选择完整区间，不能删行压缩时间")
        ic_series = valid["ic"].to_numpy()
        rank_ic_series = valid["rank_ic"].to_numpy()

        T = len(rank_ic_series)
        if T < 5:
            raise ValueError("至少需要 5 个有效截面交易日")

        # --- ค่าสถิติ Pearson IC ---
        mean_ic = float(np.mean(ic_series))
        ic_std = float(np.std(ic_series, ddof=1))
        icir = mean_ic / ic_std if ic_std > 1e-12 else float("nan")
        annualized_icir = icir * np.sqrt(self.annual_factor)
        t_stat, p_val = mean_test(ic_series, self.hac_lags)
        pos_ratio = float(np.mean(ic_series > 0))

        # --- ค่าสถิติ Spearman Rank IC ---
        mean_rank_ic = float(np.mean(rank_ic_series))
        rank_ic_std = float(np.std(rank_ic_series, ddof=1))
        rank_icir = mean_rank_ic / rank_ic_std if rank_ic_std > 1e-12 else float("nan")
        annualized_rank_icir = rank_icir * np.sqrt(self.annual_factor)
        rank_t_stat, rank_p_val = mean_test(rank_ic_series, self.hac_lags)
        rank_pos_ratio = float(np.mean(rank_ic_series > 0))

        # ค่าความเบ้และค่าความโด่ง
        skewness = float(stats.skew(rank_ic_series)) if rank_ic_std > 1e-12 else float("nan")
        kurtosis = float(stats.kurtosis(rank_ic_series)) if rank_ic_std > 1e-12 else float("nan")

        report = ICReport(
            factor_name=factor_name,
            sample_days=T,
            mean_ic=mean_ic,
            ic_std=ic_std,
            icir=icir,
            annualized_icir=annualized_icir,
            t_stat=t_stat,
            p_value=p_val,
            positive_ratio=pos_ratio,
            mean_rank_ic=mean_rank_ic,
            rank_ic_std=rank_ic_std,
            rank_icir=rank_icir,
            annualized_rank_icir=annualized_rank_icir,
            rank_t_stat=rank_t_stat,
            rank_p_value=rank_p_val,
            rank_positive_ratio=rank_pos_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            excluded_days=daily_ic_df.height - T,
            hac_lags=self.hac_lags
        )

        return report


# ================= สาธิตด้วยข้อมูลสังเคราะห์ (ผลลัพธ์ทางสถิติไม่ใช่ข้อพิสูจน์ผลตอบแทนจากการเทรดจริง (Live Trading)) =================
if __name__ == "__main__":
    np.random.seed(42)
    n_days = 250   # จำลองวันทำการซื้อขาย 1 ปี
    n_stocks = 500 # คล้ายกับ Stock Universe ของ CSI 500

    import pandas as pd
    dates = pd.bdate_range("2025-01-01", periods=n_days).date.tolist()
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]

    records = []
    # จำลองตลาดจริง: สร้าง Alpha Factor ที่แข็งแกร่งและสมจริง
    # คุณลักษณะ: ค่าเฉลี่ย Rank IC ที่แท้จริงอยู่ที่ประมาณ 0.045 และมีสัญญาณรบกวนในระดับรายวัน
    for d in dates:
        # ผลตอบแทนภาคตัดขวางที่แท้จริงของวันนั้น
        true_cross_return = np.random.normal(0.0005, 0.025, n_stocks)
        
        # Factor มีสัญญาณคาดการณ์ที่แท้จริง (สัมประสิทธิ์ 0.05) + สัญญาณรบกวนของตลาด (0.95)
        factor_signal = true_cross_return * 2.0 + np.random.normal(0, 1.0, n_stocks)
        
        # ค่าผิดปกติสุดขั้วที่เกิดขึ้นเป็นครั้งคราว (จำลองเหตุการณ์หุ้น A-Share มีการปรับโครงสร้างกะทันหันแล้วชนเพดานต่อเนื่องจนเบี่ยงเบนค่า Pearson)
        outlier_idx = np.random.choice(n_stocks, 2, replace=False)
        factor_signal[outlier_idx[0]] += 50.0  # ค่า Factor หลุดกรอบอย่างสุดขั้ว

        for s_idx, s in enumerate(symbols):
            records.append({
                "trade_date": d,
                "symbol": s,
                "alpha_factor": factor_signal[s_idx],
                "ret_fwd_1d": true_cross_return[s_idx]
            })

    test_df = pl.DataFrame(records)

    # กำหนดค่าเริ่มต้นให้กับเอนจินประเมินผล
    engine = AshareFactorICEngine()
    
    # คำนวณ IC รายวัน
    daily_ic_df = engine.compute_daily_ic(
        test_df,
        factor_col="alpha_factor",
        forward_ret_col="ret_fwd_1d"
    )

    print(">>> 每日 IC 时序与累积 IC (前 5 行):")
    print(daily_ic_df.head(5))

    # สร้างรายงานการประเมินตามมาตรฐานฝั่ง Buy-side
    report = engine.evaluate(daily_ic_df, factor_name="alpha_robust_momentum")
    report.print_summary()
