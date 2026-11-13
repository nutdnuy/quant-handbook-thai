"""
Convex Optimizer สำหรับกองทุนดัชนีเสริมผลตอบแทน (Index Enhancement) หุ้น A-Share เพื่อการเรียนการสอน (AshareIndexEnhancementOptimizer)
สร้างขึ้นบน cvxpy และแบบจำลองความเสี่ยง (Risk Model) เชิงโครงสร้าง Barra
รองรับ: Hard Neutrality ของกลุ่มอุตสาหกรรม, ข้อจำกัด Market Cap, ขีดจำกัดบน-ล่างของหุ้นรายตัว และโทษปรับ Tracking Error แบบกำลังสอง
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
import numpy as np
from examples.portfolio_utils import validate_risk_inputs, solve_checked


@dataclass
class OptimizationResult:
    """ผลลัพธ์การหาพอร์ตที่เหมาะสม (Portfolio Optimization) และรายงานตรวจสุขภาพพอร์ตการลงทุนแบบองค์รวม"""
    horizon_days: int
    optimal_weights: np.ndarray
    expected_active_return: float
    annualized_tracking_error: float
    active_share: float
    effective_stocks_count: int
    industry_max_deviation: float
    size_style_deviation: float
    status: str

    def print_dossier(self):
        """พิมพ์รายงานผลลัพธ์การหาพอร์ตที่เหมาะสมแบบจัดรูปแบบ"""
        print("=" * 70)
        print(f"🎯 指数增强凸优化求解报告 [求解状态: {self.status}]")
        print("=" * 70)
        print("【核心收益与主动风险】")
        print(f"  • 预期持有期主动超额 (Alpha)  : {self.expected_active_return * 100:8.2f}%")
        print(f"  • 预期年化跟踪误差 (TE)     : {self.annualized_tracking_error * 100:8.2f}%")
        print(f"  • 收益预测持有期             : {self.horizon_days} 个交易日")
        print(f"  • 主动份额 (Active Share)    : {self.active_share * 100:8.2f}%")
        print(f"  • 有效持仓股票数 (ENP)      : {self.effective_stocks_count:8d} 支")
        print("-" * 70)
        print("【风控合规穿透检查】")
        print(f"  • 给定行业最大净暴露偏差     : {self.industry_max_deviation * 100:8.4f}% ")
        print(f"  • 对数市值风格净偏离 (Size) : {self.size_style_deviation:8.4f} ")
        print("=" * 70)


class AshareIndexEnhancementOptimizer:
    """
    Quadratic Convex Optimizer สำหรับ Index Enhancement หุ้น A-Share
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,      # สัมประสิทธิ์การหลีกเลี่ยงความเสี่ยง (Risk Aversion) lambda
        max_active_weight: float = 0.015, # สัดส่วน Active Overweight สูงสุดของหุ้นรายตัว +1.5%
        size_tolerance: float = 0.05,     # เพดาน Hard Tolerance สำหรับการเบี่ยงเบนสไตล์ Market Cap
        max_tracking_error: float = 0.045,
        horizon_days: int = 5,
        annual_factor: float = 250.
    ):
        if risk_aversion < 0 or max_active_weight < 0 or size_tolerance < 0 or max_tracking_error <= 0 or horizon_days < 1 or annual_factor <= 0:
            raise ValueError("风险/约束参数范围不合法")
        self.max_tracking_error = max_tracking_error
        self.horizon_days, self.annual_factor = horizon_days, annual_factor
        self.risk_aversion = risk_aversion
        self.max_active_weight = max_active_weight
        self.size_tolerance = size_tolerance

    def optimize(
        self,
        expected_returns: np.ndarray,          # (N,) อัตราผลตอบแทนส่วนเกิน (Excess Return) ที่คาดหวังในกรอบ H วัน; ห้ามส่ง Ranking Score ที่ยังไม่ผ่านการ Calibrate
        benchmark_weights: np.ndarray,     # (N,) น้ำหนักของหุ้นที่เป็นองค์ประกอบในดัชนี Benchmark (sum=1)
        exposure_matrix: np.ndarray,       # (N, K) เมทริกซ์การเปิดรับแฟกเตอร์ (Factor Exposure Matrix) (K_ind คอลัมน์แรกคือกลุ่มอุตสาหกรรม หลังจากนั้นคือสไตล์)
        factor_covariance: np.ndarray,     # (K, K) เมทริกซ์ Covariance ของ Factor F
        specific_variance: np.ndarray,     # (N,) สมาชิกบนเส้นทแยงมุมของ Specific Variance Δ
        n_industries: int = 6,             # จำนวนคอลัมน์ของ Industry Factor
        size_col_idx: int = 6              # ดัชนีคอลัมน์ของ Size Factor
    ) -> OptimizationResult:
        """
        สร้างโจทย์ Convex Optimization ที่มี Quadratic Risk Constraint และคำนวณหาน้ำหนักการถือครองที่เหมาะสมที่สุด
        """
        expected_returns, benchmark_weights, exposure_matrix, factor_covariance, specific_variance = validate_risk_inputs(
            expected_returns, benchmark_weights, exposure_matrix, factor_covariance, specific_variance, n_industries, size_col_idx)
        N = len(expected_returns)

        # 1. กำหนดตัวแปรตัดสินใจ (Decision Variable): น้ำหนักพอร์ตสุดท้าย w (N,)
        w = cp.Variable(N, nonneg=True) # ข้อจำกัด Long-only: w_i >= 0
        h = w - benchmark_weights       # เวกเตอร์ Active Deviation h

        # 2. ฟังก์ชันเป้าหมาย (Objective Function): max h^T alpha - (lambda / 2) * (h^T Σ h)
        # ใช้โครงสร้างของ Barra เพื่อลดมิติ: h^T Σ h = y^T F y + h^T Δ h
        y = exposure_matrix.T @ h       # Active Factor Exposure (K,)
        
        # องค์ประกอบความเสี่ยง: Factor Risk + Specific Risk
        factor_risk = cp.quad_form(y, factor_covariance)
        specific_risk = cp.sum(cp.multiply(specific_variance, cp.square(h)))
        portfolio_risk = factor_risk + specific_risk

        expected_return = expected_returns @ h
        period_risk = portfolio_risk * self.horizon_days / self.annual_factor
        objective = cp.Maximize(expected_return - (self.risk_aversion / 2.0) * period_risk)

        # 3. การสร้างเงื่อนไขข้อจำกัด (Constraints)
        constraints = [
            cp.sum(w) == 1.0,           # ข้อจำกัดการลงทุนเต็มจำนวน (Fully Invested Budget Constraint)
            w >= 0,
            portfolio_risk <= self.max_tracking_error ** 2,
        ]

        # 3.1 ข้อจำกัด Hard Neutrality ของกลุ่มอุตสาหกรรม: X_ind^T (w - w_b) = 0
        X_ind = exposure_matrix[:, :n_industries]
        constraints.append(X_ind.T @ h == 0.0)

        # 3.2 ข้อจำกัด Soft Tolerance สำหรับการเบี่ยงเบนสไตล์ Market Cap: |X_size^T h| <= delta
        size_exp = exposure_matrix[:, size_col_idx]
        constraints.append(cp.abs(size_exp @ h) <= self.size_tolerance)

        # 3.3 ข้อจำกัดเพดานน้ำหนัก Active Weight ของหุ้นรายตัว: w_i - w_bi <= max_active_weight
        constraints.append(h <= self.max_active_weight)

        # ข้อจำกัด Hard Constraint ของ Quadratic TE จำเป็นต้องใช้ Conic Solver; ตรวจสอบทุกข้อจำกัดหลังการคำนวณ โดยไม่มีการตัดทอนน้ำหนัก
        prob = cp.Problem(objective, constraints)
        opt_w = solve_checked(prob, w, constraints)

        # 5. การคำนวณตัวชี้วัดตรวจสุขภาพพอร์ต
        active_w = opt_w - benchmark_weights
        exp_ret = float(expected_returns @ active_w)
        
        # Tracking Error
        y_opt = exposure_matrix.T @ active_w
        f_risk = float(y_opt.T @ factor_covariance @ y_opt)
        s_risk = float(np.sum(specific_variance * (active_w ** 2)))
        te_ann = np.sqrt(f_risk + s_risk)

        # Active Share (Active Share = 1/2 * sum(|w - w_b|))
        active_share = float(0.5 * np.sum(np.abs(active_w)))

        # จำนวนหุ้นที่ถือครองอย่างมีประสิทธิผล (Herfindahl-Hirschman: 1 / sum(w_i^2))
        enp = int(np.round(1.0 / np.sum(opt_w ** 2)))

        # ส่วนเบี่ยงเบน Exposure สูงสุดของกลุ่มอุตสาหกรรม
        ind_bias = np.abs(X_ind.T @ active_w)
        max_ind_dev = float(np.max(ind_bias))

        # ส่วนเบี่ยงเบน Exposure ของ Market Cap
        size_dev = float(size_exp @ active_w)

        return OptimizationResult(
            horizon_days=self.horizon_days,
            optimal_weights=opt_w,
            expected_active_return=exp_ret,
            annualized_tracking_error=te_ann,
            active_share=active_share,
            effective_stocks_count=enp,
            industry_max_deviation=max_ind_dev,
            size_style_deviation=size_dev,
            status=prob.status
        )


# ================= Unit Test และการประยุกต์ใช้งานจริง Portfolio Optimization สำหรับ Index Enhancement =================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 500  # จำลอง Universe ของหุ้นองค์ประกอบ CSI 500
    n_factors = 7   # 6 Industry Factors + 1 Size Factor

    # 1. จำลองน้ำหนักดัชนี Benchmark (เช่น การเบี่ยงเบนจาก Equal Weight ของ CSI 500 โดยหุ้นแต่ละตัวมีน้ำหนักระหว่าง 0.1% ~ 0.5%)
    raw_b = np.random.uniform(0.1, 0.5, n_stocks)
    w_benchmark = raw_b / np.sum(raw_b)

    # 2. จำลอง Factor Exposure ของ Barra
    X_mat = np.zeros((n_stocks, n_factors))
    ind_assign = np.random.choice(6, n_stocks)
    for i in range(n_stocks):
        X_mat[i, ind_assign[i]] = 1.0  # One-Hot สำหรับกลุ่มอุตสาหกรรม
    # คอลัมน์สุดท้ายคือ Size Factor ที่ผ่าน Standardization
    X_mat[:, 6] = np.random.normal(0, 1.0, n_stocks)

    # 3. จำลองพารามิเตอร์ความเสี่ยงของ Barra
    F_cov = np.eye(n_factors) * 0.04
    spec_var = np.random.uniform(0.04, 0.12, n_stocks) # Annualized Specific Variance ของหุ้นรายตัว

    # 4. จำลองการคาดการณ์ผลตอบแทนช่วงเวลาถือครอง โดยแยกอิสระจากคะแนนโมเดลจัดอันดับ
    alpha = np.random.normal(0.02, 0.05, n_stocks) * 5 / 250  # สังเคราะห์ผลตอบแทนที่คาดหวัง 5 วัน ไม่ใช่คะแนนจัดอันดับ

    # 5. สร้าง Instance และแก้ปัญหา Optimization สำหรับ Index Enhancement
    optimizer = AshareIndexEnhancementOptimizer(
        risk_aversion=3.0,
        max_active_weight=0.012, # หุ้นรายตัว Overweight ได้สูงสุด 1.2%
        size_tolerance=0.03      # จำกัดการเบี่ยงเบนของ Market Cap อย่างเข้มงวดไม่เกิน 0.03
    )

    report = optimizer.optimize(
        expected_returns=alpha,
        benchmark_weights=w_benchmark,
        exposure_matrix=X_mat,
        factor_covariance=F_cov,
        specific_variance=spec_var,
        n_industries=6,
        size_col_idx=6
    )

    # แสดงรายงานตรวจสุขภาพพอร์ตสำหรับฝั่ง Buy-side
    report.print_dossier()
