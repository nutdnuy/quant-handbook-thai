"""
lecture_02_backtest_sanitizer.py
กรณีศึกษาข้อมูลสังเคราะห์ความถี่รายวัน: เปรียบเทียบจังหวะเวลาสัญญาณกับข้อจำกัดการจับคู่คำสั่ง (Matching) พร้อมสาธิตการคำนวณเงินสดและสถานะถือครอง รวมถึง DSR
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass(frozen=True)
class TradeExecutionRules:
    can_buy_on_limit_up: bool = False   # ห้ามซื้อเมื่อชนเพดาน (Limit-up) โดยเด็ดขาด
    can_sell_on_limit_down: bool = False # ห้ามขายเมื่อชนพื้น (Limit-down) โดยเด็ดขาด
    one_way_commission: float = 0.00015  # ค่าคอมมิชชันขาเดียว 1.5 ใน 10,000
    stamp_duty: float = 0.0005           # อากรแสตมป์ (Stamp Duty) ฝั่ง Sell-side 0.05%
    slippage_bps: float = 0.0005         # Slippage พื้นฐาน 5 ใน 10,000

class AshareBacktestSanitizer:
    def __init__(self, rules: TradeExecutionRules | None = None):
        self.rules = rules or TradeExecutionRules()

    def simulate_mock_ashare_universe(self, n_days: int = 500, n_stocks: int = 5) -> pd.DataFrame:
        """
        สร้างข้อมูลราคาตลาดสังเคราะห์เพื่อการเรียนการสอน ไม่ได้สะท้อนถึงปฏิทินจริงของตลาดหลักทรัพย์ กฎเกณฑ์ราคาชนเพดาน/พื้น (Price Limit) หรือกระบวนการเพิกถอนหลักทรัพย์ (Delisting)
        """
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        all_dfs = []

        for stock_id in [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]:
            # ผลตอบแทน Random Walk พื้นฐาน
            base_ret = np.random.normal(0.0003, 0.022, size=n_days)
            
            # จำลองโอกาส 3% ที่จะชนเพดานสนิท และ 2% ที่จะชนพื้นสนิท
            limit_up_mask = np.random.binomial(1, 0.03, size=n_days).astype(bool)
            limit_down_mask = np.random.binomial(1, 0.02, size=n_days).astype(bool) & ~limit_up_mask
            
            base_ret[limit_up_mask] = 0.0998  # ชนเพดานสนิท
            base_ret[limit_down_mask] = -0.0998 # ชนพื้นสนิท

            close = 10.0 * np.cumprod(1 + base_ret)
            open_p = close * (1 + np.random.normal(0, 0.003, size=n_days))
            # กำหนดให้ความสัมพันธ์ของราคาสมเหตุสมผลในเชิงกายภาพ
            high_p = np.maximum(open_p, close) * (1 + np.abs(np.random.normal(0, 0.005, size=n_days)))
            low_p = np.minimum(open_p, close) * (1 - np.abs(np.random.normal(0, 0.005, size=n_days)))

            locked = limit_up_mask | limit_down_mask
            open_p[locked] = high_p[locked] = low_p[locked] = close[locked]

            df = pd.DataFrame({
                "trade_date": dates,
                "symbol": stock_id,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close,
                "returns": base_ret,
                "is_limit_up": limit_up_mask,
                "is_limit_down": limit_down_mask
            })
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    def run_backtest_engine(
        self,
        df: pd.DataFrame,
        use_future_leakage: bool = False,
        enforce_liquidity_constraints: bool = True
    ) -> Dict:
        """
        เครื่องยนต์หลักสำหรับรัน Backtest: รองรับการสลับเปรียบเทียบระหว่างโหมดตามกฎเกณฑ์จริงกับโหมดที่มีข้อบกพร่อง
        """
        if df.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("股票日期主键重复")
        df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        symbols = df["symbol"].unique()
        dates = df["trade_date"].unique()
        if len(dates) < 2 or len(df) != len(dates) * len(symbols):
            raise ValueError("示例要求至少两天且每个交易日覆盖相同股票全集")
        if not np.isfinite(df[["open", "close", "returns"]].to_numpy()).all() or (df[["open", "close"]] <= 0).any().any():
            raise ValueError("行情需为有效有限数值")
        df["signal"] = (df["returns"] > 0).astype(int)
        if not use_future_leakage:
            df["signal"] = df.groupby("symbol")["signal"].shift(1).fillna(0).astype(int)
        # จัดสรรเงินทุนเริ่มต้นแยกแต่ละหุ้น ไม่มีการปรับพอร์ต (Rebalance) ระหว่างหุ้น ส่วนที่ยังไม่ลงทุนจะถือเป็นเงินสด
        # หุ้นแต่ละตัวจะเทรดแค่วันละครั้งตอนเปิดตลาด สถานะที่เพิ่งซื้อใหม่จะไม่ขายในวันเดียวกัน
        cash = {symbol: 1.0 / len(symbols) for symbol in symbols}
        shares = {symbol: 0.0 for symbol in symbols}
        previous_nav = 1.0
        portfolio_daily_returns = []
        buy_cost = self.rules.one_way_commission + self.rules.slippage_bps
        sell_cost = buy_cost + self.rules.stamp_duty
        if min(buy_cost, sell_cost) < 0 or sell_cost >= 1:
            raise ValueError("交易成本率不合法")
        for day in dates[1:]:
            nav = 0.0
            for _, row in df[df["trade_date"] == day].iterrows():
                symbol = row["symbol"]
                buy = row["signal"] == 1 and shares[symbol] == 0
                sell = row["signal"] == 0 and shares[symbol] > 0
                suspended = ("volume" in row and (pd.isna(row["volume"]) or row["volume"] <= 0))
                if enforce_liquidity_constraints:
                    buy = buy and not suspended and (not row["is_limit_up"] or self.rules.can_buy_on_limit_up)
                    sell = sell and not suspended and (not row["is_limit_down"] or self.rules.can_sell_on_limit_down)
                if buy:
                    shares[symbol] = cash[symbol] / (row["open"] * (1 + buy_cost))
                    cash[symbol] = 0.0
                elif sell:
                    cash[symbol] = shares[symbol] * row["open"] * (1 - sell_cost)
                    shares[symbol] = 0.0
                nav += cash[symbol] + shares[symbol] * row["close"]
            portfolio_daily_returns.append(nav / previous_nav - 1)
            previous_nav = nav
        returns_arr = np.asarray(portfolio_daily_returns)
        std_ret = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 0.0
        annual_sharpe = np.sqrt(252) * np.mean(returns_arr) / std_ret if std_ret > 0 else 0.0
        return dict(sharpe_ratio=annual_sharpe, cumulative_return=previous_nav - 1,
                    mean_daily_return=np.mean(returns_arr), returns_series=returns_arr)

    @staticmethod
    def calculate_deflated_sharpe_ratio(
        returns: np.ndarray, num_trials: int = 1, benchmark_sharpe: float = 0.0,
        trial_sharpe_std: float | None = None,
    ) -> float:
        """สูตรอ้างอิง DSR ค่า Sharpe Ratio, Benchmark และส่วนเบี่ยงเบนมาตรฐานระหว่างการทดลองล้วนเป็นเกณฑ์ต่องวด (Single-period)

        num_trials คือข้อสมมติจำนวนครั้งของการทดลองอิสระ หากทดลองหลายครั้งต้องระบุระดับการกระจายตัวของ Sharpe Ratio จากการค้นหาทั้งหมด
        ผลลัพธ์ไม่ใช่ความน่าจะเป็นแบบเบย์เซียนภายหลัง (Bayesian Posterior Probability) ว่า "กลยุทธ์เป็นของจริง" และไม่ได้จัดการกับ Autocorrelation ของผลตอบแทนเป็นการเฉพาะ
        ที่มา: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 1 or len(returns) < 3 or not np.isfinite(returns).all():
            raise ValueError("至少需要三个有限的单期收益观测")
        if not isinstance(num_trials, int) or num_trials < 1 or not np.isfinite(benchmark_sharpe):
            raise ValueError("试验次数和基准不合法")
        standard_deviation = np.std(returns, ddof=1)
        if standard_deviation <= 0:
            raise ValueError("常量收益没有可估计的夏普标准误")
        sr = np.mean(returns) / standard_deviation
        threshold = benchmark_sharpe
        if num_trials > 1:
            if trial_sharpe_std is None or not np.isfinite(trial_sharpe_std) or trial_sharpe_std < 0:
                raise ValueError("多次试验需要提供单期夏普的试验间标准差")
            gamma = 0.5772156649
            max_z = ((1 - gamma) * norm.ppf(1 - 1 / num_trials) +
                     gamma * norm.ppf(1 - 1 / (num_trials * np.e)))
            threshold += trial_sharpe_std * max_z
        variance_term = 1 - skew(returns, bias=False) * sr + (kurtosis(returns, fisher=False, bias=False) - 1) * sr**2 / 4
        if not np.isfinite(variance_term) or variance_term <= 0:
            raise ValueError("无法估计有效夏普标准误")
        return float(norm.cdf((sr - threshold) * np.sqrt(len(returns) - 1) / np.sqrt(variance_term)))

if __name__ == "__main__":
    sanitizer = AshareBacktestSanitizer()
    data = sanitizer.simulate_mock_ashare_universe(n_days=600, n_stocks=10)

    print("=" * 70)
    print("实验 1：包含未来函数 + 忽略涨跌停限制的【业余漏洞版本】")
    print("=" * 70)
    res_flawed = sanitizer.run_backtest_engine(
        data, use_future_leakage=True, enforce_liquidity_constraints=False
    )
    print(f"  虚假回测夏普比率 (Sharpe Ratio) : {res_flawed['sharpe_ratio']:.2f}")
    print(f"  虚假累计总收益 (Cumulative Ret): {res_flawed['cumulative_return']*100:.2f}%")

    print("\n" + "=" * 70)
    print("实验 2：买方防御型【严格时序 + 涨跌停封板撮合拦截版本】")
    print("=" * 70)
    res_clean = sanitizer.run_backtest_engine(
        data, use_future_leakage=False, enforce_liquidity_constraints=True
    )
    print(f"  约束版本夏普比率 (Clean Sharpe) : {res_clean['sharpe_ratio']:.2f}")
    print(f"  约束版本累计收益 (Clean Return)    : {res_clean['cumulative_return']*100:.2f}%")

    print("\n" + "=" * 70)
    print("实验 3：过拟合与数据窥探检验 (Deflated Sharpe Ratio)")
    print("=" * 70)
    # สมมติว่ากลยุทธ์นี้คือผลลัพธ์ที่ดีที่สุดที่คัดเลือกมาหลังจากการทดสอบพารามิเตอร์ต่างๆ ไป 500 ครั้ง
    dsr_val = AshareBacktestSanitizer.calculate_deflated_sharpe_ratio(
        res_clean["returns_series"], num_trials=500, trial_sharpe_std=0.02  # เป็นเพียงสมมติฐานเพื่อการสาธิต ไม่ใช่ผลลัพธ์จากการค้นหาจริง
    )
    print(f"  单次回测名义夏普值               : {res_clean['sharpe_ratio']:.2f}")
    print(f"  假设 500 次独立试验、单期夏普标准差 0.02 的 DSR: {dsr_val:.4f} (仅示范公式，不能据此判定策略真实概率)")
