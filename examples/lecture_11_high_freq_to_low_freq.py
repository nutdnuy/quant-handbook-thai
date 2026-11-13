"""
lecture_11_high_freq_ofi_engine.py
การจำลอง Order Flow Imbalance (OFI) บนสมุดคำสั่งซื้อ (Order Book) ระดับ Level-2 และเอนจินคำนวณการรวมข้อมูลความถี่สูงสู่ความถี่ต่ำ
"""
from datetime import datetime, timedelta, date
import time
import numpy as np
import polars as pl
import pandas as pd

class AshareHighFreqOFIEngine:
    def __init__(self):
        pass

    def compute_intraday_ofi(self, l2_snapshots_df: pl.DataFrame) -> pl.DataFrame:
        """
        คำนวณลำดับ OFI จากสแนปช็อตระหว่างวันตลอดทั้งวันของหุ้นรายตัว (State Machine ระดับจุลภาค)
        อินพุตประกอบด้วย: [symbol, timestamp, bid_price, bid_vol, ask_price, ask_vol]
        """
        if (l2_snapshots_df.schema["timestamp"] != pl.Datetime
                or l2_snapshots_df.schema["timestamp"].time_zone is not None):
            raise ValueError("timestamp 必须为上海本地无时区 Datetime")
        if l2_snapshots_df.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item():
            raise ValueError("快照的股票时间键必须唯一")
        if l2_snapshots_df.select(pl.any_horizontal([
            (~pl.col(c).is_finite() | (pl.col(c) <= 0)).fill_null(True)
            for c in ["bid_price", "ask_price", "bid_vol", "ask_vol"]
        ]).any()).item():
            raise ValueError("报价和深度必须为有限正数；空盘口需上游处理")
        l2_snapshots_df = l2_snapshots_df.with_columns(pl.col("timestamp").dt.date().alias("trade_date"))
        # ใช้ฟังก์ชัน lag ของ Polars ดึงสถานะของสแนปช็อตก่อนหน้า (shift 1)
        ofi_calc = (
            l2_snapshots_df.lazy()
            .sort(["symbol", "timestamp"])
            .with_columns([
                pl.col("bid_price").shift(1).over(["symbol", "trade_date"]).alias("prev_bid_p"),
                pl.col("bid_vol").shift(1).over(["symbol", "trade_date"]).alias("prev_bid_v"),
                pl.col("ask_price").shift(1).over(["symbol", "trade_date"]).alias("prev_ask_p"),
                pl.col("ask_vol").shift(1).over(["symbol", "trade_date"]).alias("prev_ask_v")
            ])
            # คำนวณส่วนต่างฝั่ง Buy-side (I_bid)
            .with_columns([
                pl.when(pl.col("bid_price") > pl.col("prev_bid_p"))
                .then(pl.col("bid_vol"))
                .when(pl.col("bid_price") == pl.col("prev_bid_p"))
                .then(pl.col("bid_vol") - pl.col("prev_bid_v"))
                .when(pl.col("bid_price") < pl.col("prev_bid_p"))
                .then(-pl.col("prev_bid_v"))
                .otherwise(0.0) # ใช้เฟรมแรกของแต่ละวันเป็นเบสไลน์ (Baseline)
                .alias("i_bid"),

                # คำนวณส่วนต่างฝั่ง Sell-side (I_ask)
                pl.when(pl.col("ask_price") > pl.col("prev_ask_p"))
                .then(-pl.col("prev_ask_v"))
                .when(pl.col("ask_price") == pl.col("prev_ask_p"))
                .then(pl.col("ask_vol") - pl.col("prev_ask_v"))
                .when(pl.col("ask_price") < pl.col("prev_ask_p"))
                .then(pl.col("ask_vol"))
                .otherwise(0.0)
                .alias("i_ask")
            ])
            # คำนวณ OFI ระดับจุลภาคในแต่ละจังหวะสแนปช็อต
            .with_columns([
                (pl.col("i_bid") - pl.col("i_ask")).alias("snapshot_ofi"),
                (pl.col("bid_vol") + pl.col("ask_vol")).alias("depth_sum")
            ])
            .collect()
        )
        return ofi_calc

    def aggregate_to_daily_factors(self, ofi_snapshots_df: pl.DataFrame, trade_date: str | None = None) -> pl.DataFrame:
        """
        ลดมิติและรวบรวมสแนปช็อตระหว่างวันหลายพันรายการให้เป็น【แฟกเตอร์ (Factor) Alpha ความถี่รายวันที่มีมูลค่าสูง】
        """
        if trade_date is not None:
            requested = date.fromisoformat(trade_date)
            if ofi_snapshots_df["trade_date"].unique().to_list() != [requested]:
                raise ValueError("指定日期必须与全部快照日期一致；多日输入请省略 trade_date")
        daily_features = (
            ofi_snapshots_df.group_by(["symbol", "trade_date"])
            .agg([
                # 1. OFI สะสมสุทธิตลอดทั้งวัน
                pl.col("snapshot_ofi").sum().alias("daily_cum_ofi"),
                # 2. ความลึกเฉลี่ยของคำสั่งซื้อขายที่ตั้งรอตลอดทั้งวัน
                pl.col("depth_sum").mean().alias("avg_lob_depth"),
                # 3. ความเข้มข้นสัมพัทธ์ของ OFI แบบปรับมาตรฐาน (Standardization) (หารด้วยความลึกของคำสั่งซื้อขาย)
                (pl.col("snapshot_ofi").sum() / (pl.col("depth_sum").mean() + 1e-5)).alias("factor_norm_ofi"),
                # 4. ความผันผวนของ OFI (ความดุเดือดในการแย่งชิงของกระแสคำสั่งซื้อระหว่างวัน)
                pl.col("snapshot_ofi").std().alias("factor_ofi_volatility")
            ])
            .sort("factor_norm_ofi", descending=True)
        )
        return daily_features

# =========================================================================
# การทดสอบ Unit Test: จำลองการคำนวณจากสแนปช็อต L2 ตลอดวันจำนวน 4,800 จุด
# =========================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_snapshots = 4800 # จำลองสตรีมสแนปช็อต L2 ทุกๆ 3 วินาทีแบบเต็มวัน
    print(f"[*] 正在生成高密度 Level-2 订单簿盘口快照仿真流 (共 {n_snapshots} 条快照)...")

    # จำลองสินทรัพย์ 2 ตัว:
    # สินทรัพย์ A: คำสั่งซื้อขนาดใหญ่ของสถาบันตั้งซื้อดันราคากวาดของต่อเนื่อง (ซื้อแกร่ง ขายอ่อน)
    # สินทรัพย์ B: รายย่อยตั้งรับแบบ Passive, มีคำสั่งใหญ่เทขายกดราคาต่อเนื่อง (ขายแกร่ง ซื้ออ่อน)
    symbols = ["000001.SZ", "600519.SH"]
    all_records = []

    for sym in symbols:
        p_base = 100.0 if sym == "600519.SH" else 15.0
        # สินทรัพย์ A มี Drift ขาขึ้นเล็กน้อย ขณะที่สินทรัพย์ B มีแรงขายกดลงมา
        drift = 0.00002 if sym == "600519.SH" else -0.00002

        curr_p = p_base
        for i in range(n_snapshots):
            curr_p += np.random.normal(drift, 0.01)
            bid_p = np.round(curr_p - 0.01, 2)
            ask_p = np.round(curr_p + 0.01, 2)
            
            # Moutai จำลองฝั่ง Long ที่ตั้ง Bid หนุนอย่างต่อเนื่องเชิงรุก
            bid_v = np.random.randint(100, 500) if sym == "600519.SH" else np.random.randint(50, 200)
            ask_v = np.random.randint(50, 200) if sym == "600519.SH" else np.random.randint(100, 500)

            all_records.append({
                "symbol": sym,
                "timestamp": datetime(2023, 10, 16, 9, 30) + timedelta(seconds=3*i + (5400 if i >= 2400 else 0)),
                "bid_price": bid_p,
                "bid_vol": float(bid_v),
                "ask_price": ask_p,
                "ask_vol": float(ask_v)
            })

    df_l2 = pl.DataFrame(all_records)

    # รันเอนจินสกัดฟีเจอร์ความถี่สูง
    engine = AshareHighFreqOFIEngine()
    t0 = time.perf_counter()
    df_ofi_snaps = engine.compute_intraday_ofi(df_l2)
    daily_res = engine.aggregate_to_daily_factors(df_ofi_snaps, trade_date="2023-10-16")
    elapsed = time.perf_counter() - t0

    print(f"✅ 高频低频化聚合计算完毕！耗时: {elapsed:.3f} 秒")
    print("\n" + "=" * 80)
    print("日频标准化订单流不平衡 (Norm OFI) 降维提取结果：")
    print("=" * 80)
    print(daily_res)
