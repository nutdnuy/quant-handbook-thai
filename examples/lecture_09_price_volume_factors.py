"""
lecture_09_price_volume_engine.py
Engine คำนวณแฟกเตอร์ (Factor) ด้านราคาและปริมาณการซื้อขายระดับ Production สำหรับตลาดหุ้น A-Share ทั้งหมด (บน Polars Expression)
"""
import time
import numpy as np
import polars as pl
import pandas as pd

class AsharePriceVolumeFactorEngine:
    def __init__(self):
        pass

    def compute_all_factors(self, daily_bars_df: pl.DataFrame) -> pl.DataFrame:
        """
        คำนวณ Feature Matrix ด้านราคาและปริมาณการซื้อขายแบบ Batch เต็มรูปแบบ (Vectorization แบบมัลติคอร์ล้วน)
        Input ต้องประกอบด้วย: [symbol, trade_date, open, high, low, close, close_hfq, volume, turnover]
        """
        print("[*] 正在启动 Polars 多核并行量价特征流水线...")
        t0 = time.perf_counter()

        prices = ["open", "high", "low", "close", "close_hfq"]
        if daily_bars_df.select(pl.any_horizontal([
            (~pl.col(c).is_finite() | (pl.col(c) <= 0)).fill_null(True) for c in prices
        ]).any()).item():
            raise ValueError("原始和复权价格必须为有限正数")
        if daily_bars_df.select(pl.struct("symbol", "trade_date").is_duplicated().any()).item():
            raise ValueError("行情的股票日期键必须唯一")
        # แปลงเป็น Lazy Execution Graph
        lazy_df = daily_bars_df.lazy().sort(["symbol", "trade_date"])

        factors_lazy = lazy_df.with_columns([
            # ================= 1. ระบบ Momentum และ Reversal =================
            # อัตราผลตอบแทน 1 วัน (ผลตอบแทนรายวัน)
            (pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0).alias("ret_1d"),
            # Momentum 5 วัน
            (pl.col("close_hfq") / pl.col("close_hfq").shift(5).over("symbol") - 1.0).alias("mom_5d"),
            # Momentum 20 วัน
            (pl.col("close_hfq") / pl.col("close_hfq").shift(20).over("symbol") - 1.0).alias("mom_20d"),
            # Momentum ระยะกลาง 60 วัน
            (pl.col("close_hfq") / pl.col("close_hfq").shift(60).over("symbol") - 1.0).alias("mom_60d"),

            # Factor การกลับตัวระยะสั้น 5 วันและ 20 วัน (ใส่เครื่องหมายลบ)
            (-(pl.col("close_hfq") / pl.col("close_hfq").shift(5).over("symbol") - 1.0)).alias("rev_5d"),
            (-(pl.col("close_hfq") / pl.col("close_hfq").shift(20).over("symbol") - 1.0)).alias("rev_20d"),

            # ================= 2. การแยกส่วนย่อยข้ามคืน (Overnight) และระหว่างวัน (Intraday) =================
            # ผลตอบแทนข้ามคืน (เปิด / ปิดก่อนหน้า - 1)
            ((pl.col("open") * pl.col("close_hfq") / pl.col("close")) / pl.col("close_hfq").shift(1).over("symbol") - 1.0).alias("ret_overnight"),
            # ผลตอบแทนระหว่างวัน (ปิด / เปิด - 1)
            (pl.col("close") / pl.col("open") - 1.0).alias("ret_intraday"),

            # ================= 3. ระบบความผันผวน (Volatility) และผลตอบแทนสุดขั้ว (MAX) =================
            # Volatility ต่อปี 20 วัน
            (
                (pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0)
                .rolling_std(window_size=20)
                .over("symbol") * (252.0 ** 0.5)
            ).alias("volatility_20d"),

            # แอมพลิจูดที่แท้จริงระหว่างวัน (High / Low - 1)
            (pl.col("high") / pl.col("low") - 1.0).alias("amplitude_daily"),
            # แอมพลิจูดเฉลี่ย 20 วัน
            ((pl.col("high") / pl.col("low") - 1.0).rolling_mean(20).over("symbol")).alias("amplitude_20d"),

            # Factor ปรากฏการณ์ MAX: ผลตอบแทนรายวันสูงสุดในรอบ 20 วันที่ผ่านมา (ใส่เครื่องหมายลบ คาดหวัง Alpha ติดลบ)
            (
                -(pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0)
                .rolling_max(window_size=20)
                .over("symbol")
            ).alias("neg_max_20d"),

            # ================= 4. ระบบสภาพคล่อง (Liquidity) และความคึกคักของ Turnover =================
            # ค่าเฉลี่ยเคลื่อนที่ของปริมาณการซื้อขาย 20 วัน
            (pl.col("volume").rolling_mean(20).over("symbol")).alias("vol_ma20"),
            # อัตราส่วน Volume พุ่งสูง (ปริมาณวันนี้ / ปริมาณเฉลี่ย 20 วัน)
            (pl.col("volume") / (pl.col("volume").rolling_mean(20).over("symbol") + 1e-8)).alias("vol_surge_ratio"),

            # Factor ต้นทุน Market Impact ของ Amihud (ค่าเฉลี่ยของ |ผลตอบแทน| / มูลค่าการซื้อขาย ย้อนหลัง 20 วัน)
            (
                (
                    (pl.col("close_hfq") / pl.col("close_hfq").shift(1).over("symbol") - 1.0).abs() /
                    pl.when(pl.col("turnover") > 0).then(pl.col("turnover") / 1e8).otherwise(None) # แปลงเป็นมูลค่าการซื้อขายหน่วยร้อยล้านหยวน
                )
                .rolling_mean(window_size=20)
                .over("symbol")
            ).alias("amihud_illiq_20d")
        ])

        # สั่งให้ประมวลผลการคำนวณจริง (Physical Execution)
        res_df = factors_lazy.collect()
        elapsed = time.perf_counter() - t0
        print(f"✅ 量价特征计算完毕！总耗时: {elapsed:.3f} 秒 (包含 {len(res_df.columns)} 个特征字段)")
        return res_df

# =========================================================================
# การทดสอบระดับยูนิตและการทดสอบ Throughput
# =========================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 500
    n_days = 250
    total_len = n_stocks * n_days
    print(f"[*] 正在构造规模为 {total_len:,} 行的仿真 A 股行情面板...")

    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    symbols = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    records = []
    for s in symbols:
        ret = np.random.normal(0.0003, 0.02, size=n_days)
        prices = 10.0 * np.exp(np.cumsum(ret))
        opens = prices * (1 + np.random.normal(0, 0.004, size=n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, size=n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, size=n_days)))
        vols = np.random.uniform(500000, 5000000, size=n_days)
        turnovers = vols * prices

        for i in range(n_days):
            records.append({
                "symbol": s,
                "trade_date": dates[i].date(),
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": prices[i],
                "close_hfq": prices[i], # ข้อมูลสังเคราะห์ไม่มีการจ่ายปันผลและการแตกหุ้น
                "volume": vols[i],
                "turnover": turnovers[i]
            })

    mock_bars = pl.DataFrame(records)

    # รัน Feature Engine
    engine = AsharePriceVolumeFactorEngine()
    features_df = engine.compute_all_factors(mock_bars)

    # สุ่มพิมพ์ตัวอย่างผลลัพธ์
    sample_sym = "000001.SZ"
    sample_res = features_df.filter(pl.col("symbol") == sample_sym).tail(3)
    
    print("\n" + "=" * 90)
    print(f"标的 {sample_sym} 最新特征计算切片抽样：")
    print("=" * 90)
    cols_to_show = ["symbol", "trade_date", "close", "rev_20d", "ret_overnight", "ret_intraday", "volatility_20d", "neg_max_20d", "amihud_illiq_20d"]
    print(sample_res.select(cols_to_show))
