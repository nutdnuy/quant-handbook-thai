"""
lecture_01_comprehensive_cost_engine.py
ต้นทุนแรงเสียดทานของการลงทุน Quant ในหุ้น A-Share, กฎพื้นฐานของการบริหารเชิงรุก (FLAM) และ Engine การแจกแจงที่มา (Attribution) ของผลตอบแทนสุทธิจริง
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MarketFrictionConfig:
    stamp_duty_sell: float = 0.0005     # อากรแสตมป์ (Stamp Duty) (เกณฑ์หลัง ส.ค. 2023: ฝั่ง Sell-side ขาเดียว 0.05%)
    broker_commission: float = 0.00015  # ค่าคอมมิชชันรวมโบรกเกอร์ (สองฝั่งข้างละ 1.5 ในหมื่น)
    transfer_fee: float = 0.00001       # ค่าธรรมเนียมโอนหลักทรัพย์ (สองฝั่งข้างละ 1 ในแสน)
    slippage_one_way: float = 0.0004    # Bid-Ask Spread และ Slippage จากต้นทุน Market Impact ขาเดียว (ขาเดียว 4 ในหมื่น)
    im_annual_basis_cost: float = 0.075 # สัญญาฟิวเจอร์สดัชนี CSI 1000 (IM) ต้นทุน Discount เฉลี่ยต่อปี (7.5%)
    ic_annual_basis_cost: float = 0.055 # สัญญาฟิวเจอร์สดัชนี CSI 500 (IC) ต้นทุน Discount เฉลี่ยต่อปี (5.5%)
    if_annual_basis_cost: float = 0.025 # สัญญาฟิวเจอร์สดัชนี CSI 300 (IF) ต้นทุน Discount เฉลี่ยต่อปี (2.5%)


@dataclass
class FLAMParameters:
    ic: float = 0.045                   # ค่าเฉลี่ยภาคตัดขวาง (Cross-sectional) ของ Rank IC ของสัญญาณ (เช่น 0.045)
    breadth_universe_size: int = 1000   # ความกว้างของ Universe หุ้น (เช่น CSI 1000 มี 1,000 ตัว)
    rebalance_frequency_days: int = 5   # รอบการปรับพอร์ต (Rebalance) (วัน)
    transfer_coefficient: float = 0.45  # สัมประสิทธิ์การถ่ายทอด TC (ถูกจำกัดจากการห้าม Short, ราคาชนเพดาน/พื้น (Price Limit), ข้อจำกัดการเบี่ยงเบนอุตสาหกรรม)
    active_risk_vol: float = 0.08       # เป้าหมายความผันผวนของ Tracking Error เชิงรุกต่อปี TE (8%)


class QuantCostAttributionEngine:
    def __init__(self, config: MarketFrictionConfig = MarketFrictionConfig()):
        self.cfg = config

    def calculate_flam_theoretical_metrics(self, params: FLAMParameters) -> dict:
        """
        คำนวณ Information Ratio (IR) ตามทฤษฎี และผลตอบแทนส่วนเกิน (Excess Return) ขั้นต้นตามทฤษฎี (Gross Alpha)
        อิงตามกฎพื้นฐานของการบริหารเชิงรุก (Fundamental Law of Active Management) ของ Grinold & Kahn (1999)
        สูตร:
          N_annual = (Universe_Size) * (250 / Rebalance_Days)
          IR = IC * sqrt(N_annual) * TC
          Gross_Alpha = IR * Active_Risk
        """
        decisions_per_year = (250.0 / params.rebalance_frequency_days)
        annual_breadth = params.breadth_universe_size * decisions_per_year
        theoretical_ir = params.ic * np.sqrt(annual_breadth) * params.transfer_coefficient
        gross_alpha = theoretical_ir * params.active_risk_vol

        return {
            "rank_ic": params.ic,
            "universe_size": params.breadth_universe_size,
            "annual_decisions": int(annual_breadth),
            "transfer_coefficient_tc": params.transfer_coefficient,
            "active_risk_te": params.active_risk_vol,
            "theoretical_ir": theoretical_ir,
            "theoretical_gross_alpha": gross_alpha
        }

    def calculate_one_round_friction(self, custom_slippage: float = None) -> float:
        """
        คำนวณอัตราต้นทุนแรงเสียดทานรวมของการ Rebalance เต็มรอบหนึ่งครั้ง (ขายพอร์ตเดิม 100% และซื้อพอร์ตใหม่ 100%)
        ประกอบด้วย: อากรแสตมป์ฝั่งขาย + ค่าคอมมิชชันสองฝั่ง + ค่าธรรมเนียมโอนสองฝั่ง + Slippage สองฝั่ง
        """
        slip = self.cfg.slippage_one_way if custom_slippage is None else custom_slippage
        total_friction = (
            self.cfg.stamp_duty_sell +
            (self.cfg.broker_commission * 2) +
            (self.cfg.transfer_fee * 2) +
            (slip * 2)
        )
        return total_friction

    def simulate_strategy_net_returns(
        self,
        gross_alpha_annual: float,
        turnover_annual: float,
        strategy_type: str = "index_enhancement",
        hedge_index: str = "IM",
        custom_slippage: float = None
    ) -> dict:
        """
        จำลองผลตอบแทนส่วนเกินสุทธิและการลดทอนผลตอบแทนของกลยุทธ์ Quant แบบเต็มรูปแบบ
        :param gross_alpha_annual: อัตรา Gross Alpha ตามทฤษฎีต่อปี (เช่น 0.20 หมายถึง 20%)
        :param turnover_annual: อัตรา Turnover ขาเดียวต่อปี (เช่น 30 หมายถึงหมุนเวียนพอร์ต 30 รอบต่อปี ถือครองเฉลี่ยประมาณ 8 วัน)
        :param strategy_type: "index_enhancement" (กองทุนดัชนีเสริมผลตอบแทน (Index Enhancement)) หรือ "market_neutral" (Market Neutral)
        :param hedge_index: สินทรัพย์อ้างอิงสำหรับ Hedge "IM" (CSI 1000), "IC" (CSI 500), "IF" (CSI 300)
        :param custom_slippage: อัตรา Slippage ขาเดียวกำหนดเอง
        """
        round_trip_cost = self.calculate_one_round_friction(custom_slippage)
        total_trading_cost = turnover_annual * round_trip_cost

        basis_cost = 0.0
        if strategy_type == "market_neutral":
            if hedge_index == "IM":
                basis_cost = self.cfg.im_annual_basis_cost
            elif hedge_index == "IC":
                basis_cost = self.cfg.ic_annual_basis_cost
            else:
                basis_cost = self.cfg.if_annual_basis_cost

        net_alpha = gross_alpha_annual - total_trading_cost - basis_cost
        cost_erosion_rate = (1.0 - (net_alpha / gross_alpha_annual)) if gross_alpha_annual > 0 else 1.0

        return {
            "strategy_type": strategy_type,
            "hedge_index": hedge_index if strategy_type == "market_neutral" else "None",
            "gross_alpha": gross_alpha_annual,
            "annual_turnover": turnover_annual,
            "round_trip_friction_bps": round_trip_cost * 10000,
            "annual_trading_friction": total_trading_cost,
            "annual_basis_cost": basis_cost,
            "net_alpha": net_alpha,
            "erosion_percentage": cost_erosion_rate * 100.0,
            "is_viable": net_alpha > 0.05
        }

    def generate_sensitivity_table(self, gross_alphas: list, turnovers: list) -> pd.DataFrame:
        """
        สร้างเมทริกซ์การวิเคราะห์ความอ่อนไหวระหว่าง Turnover กับ Gross Return (ยกตัวอย่าง Index Enhancement CSI 1000)
        """
        rows = []
        for g_alpha in gross_alphas:
            row = {"Gross_Alpha": f"{g_alpha*100:.0f}%"}
            for t in turnovers:
                res = self.simulate_strategy_net_returns(g_alpha, t, "index_enhancement")
                row[f"Turnover_{t}x"] = f"{res['net_alpha']*100:.2f}% (损耗{res['erosion_percentage']:.0f}%)"
            rows.append(row)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    engine = QuantCostAttributionEngine()
    
    print("=" * 75)
    print("0. 主动管理基本法则 (FLAM) 理论超额测算 (以中证1000周频调仓为例)")
    print("=" * 75)
    flam_res = engine.calculate_flam_theoretical_metrics(FLAMParameters())
    for k, v in flam_res.items():
        if isinstance(v, float):
            print(f"  {k:28s} : {v:.4f}")
        else:
            print(f"  {k:28s} : {v}")

    print("\n" + "=" * 75)
    print("案例 1：中证 1000 指数增强策略 (中频价量模型，年化换手 35 倍)")
    print("=" * 75)
    res_enhanced = engine.simulate_strategy_net_returns(
        gross_alpha_annual=flam_res["theoretical_gross_alpha"],
        turnover_annual=35.0,
        strategy_type="index_enhancement"
    )
    for k, v in res_enhanced.items():
        if isinstance(v, float):
            print(f"  {k:28s} : {v:.4f}")
        else:
            print(f"  {k:28s} : {v}")

    print("\n" + "=" * 75)
    print("案例 2：相同 Alpha 引擎构造的市场中性策略 (IM 期指对冲，年化贴水 7.5%)")
    print("=" * 75)
    res_neutral = engine.simulate_strategy_net_returns(
        gross_alpha_annual=flam_res["theoretical_gross_alpha"],
        turnover_annual=35.0,
        strategy_type="market_neutral",
        hedge_index="IM"
    )
    for k, v in res_neutral.items():
        if isinstance(v, float):
            print(f"  {k:28s} : {v:.4f}")
        else:
            print(f"  {k:28s} : {v}")

    print("\n" + "=" * 75)
    print("敏感性矩阵：不同换手率对 A 股指增策略净超额的影响")
    print("=" * 75)
    df_sens = engine.generate_sensitivity_table(
        gross_alphas=[0.12, 0.18, 0.25],
        turnovers=[10.0, 25.0, 40.0, 60.0]
    )
    print(df_sens.to_string(index=False))
