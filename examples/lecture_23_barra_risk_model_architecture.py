"""
เอนจินแบบจำลองความเสี่ยง (Risk Model) เชิงโครงสร้างรูปแบบ Barra สำหรับการเรียนการสอน (AshareBarraRiskEngine)
การทำงาน: การสร้าง Σ = X F X^T + Δ ขึ้นมาใหม่อย่างแม่นยำ, EWMA สำหรับความแปรปรวนร่วมของแฟกเตอร์ (Factor) และการหดตัว (Shrinkage) ของความแปรปรวนเฉพาะตัวด้วยความเข้มคงที่
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl


@dataclass
class BarraRiskMatrices:
    """คอนเทนเนอร์เก็บเมทริกซ์ผลลัพธ์ของ Risk Model แบบ Barra"""
    exposure_matrix: np.ndarray        # X: (N, K)
    factor_covariance: np.ndarray      # F: (K, K)
    specific_variance: np.ndarray      # Δ: (N,) สมาชิกบนเส้นทแยงมุม
    factor_names: List[str]
    symbols: List[str]

    def compute_portfolio_variance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        คำนวณความแปรปรวนต่อปีรวม ความแปรปรวนของความเสี่ยงแฟกเตอร์ และความแปรปรวนของความเสี่ยงเฉพาะตัวตามเวกเตอร์น้ำหนักที่กำหนด
        คืนค่า: (ความแปรปรวนต่อปีรวม, ความแปรปรวนแฟกเตอร์ต่อปี, ความแปรปรวนเฉพาะตัวต่อปี)
        """
        # Factor Exposure: beta = X^T @ w (K,)
        factor_exp = self.exposure_matrix.T @ weights
        
        # Factor Risk: w^T (X F X^T) w = beta^T F beta
        factor_var = float(factor_exp.T @ self.factor_covariance @ factor_exp)
        
        # Specific Risk: w^T Δ w = sum(w_i^2 * delta_i)
        specific_var = float(np.sum((weights ** 2) * self.specific_variance))
        
        total_var = factor_var + specific_var
        return total_var, factor_var, specific_var


class AshareBarraRiskEngine:
    """
    ตัวประมาณค่าแบบจำลองความเสี่ยงเชิงโครงสร้างแบบหลายแฟกเตอร์ (Multi-factor) ของ Barra
    """

    def __init__(
        self,
        half_life_days: int = 60,
        annual_factor: float = 250.0,
        shrinkage_intensity: float = 0.30
    ):
        if half_life_days <= 0 or annual_factor <= 0 or not 0 <= shrinkage_intensity <= 1:
            raise ValueError("半衰期/年化系数须为正，收缩强度须在 [0,1]")
        self.half_life = half_life_days
        self.annual_factor = annual_factor
        self.shrinkage = shrinkage_intensity

    def _ewma_weights(self, T: int) -> np.ndarray:
        """สร้างเวกเตอร์น้ำหนักแบบลดทอนถ่วงน้ำหนักเอ็กซ์โพเนนเชียล (ปรับสเกลให้ผลรวมเป็น 1)"""
        decay_factor = 2.0 ** (-1.0 / self.half_life)
        powers = np.arange(T - 1, -1, -1)
        w = decay_factor ** powers
        return w / np.sum(w)

    def estimate_risk_model(
        self,
        historical_returns: np.ndarray, # (T, N) อนุกรมเวลา (Time-series) ของผลตอบแทนย้อนหลัง
        exposure_matrix: np.ndarray,    # (T, N, K) Exposure ย้อนหลังที่ทราบก่อนเริ่มผลตอบแทนแต่ละงวด
        market_caps: np.ndarray,        # (T, N) มูลค่าตามราคาตลาด ณ จุดเวลาเดียวกับ Exposure ย้อนหลัง
        factor_names: List[str],
        symbols: List[str],
        current_exposure_matrix: Optional[np.ndarray] = None,
        assume_static: bool = False
    ) -> BarraRiskMatrices:
        """
        กระบวนการประมาณค่าโมเดลหลัก
        """
        historical_returns = np.asarray(historical_returns, dtype=float)
        exposures = np.asarray(exposure_matrix, dtype=float)
        caps = np.asarray(market_caps, dtype=float)
        if historical_returns.ndim != 2 or historical_returns.shape[0] < 2:
            raise ValueError("收益必须为至少两期的 (T,N) 矩阵")
        T, N = historical_returns.shape
        if exposures.ndim == 2:
            if not assume_static:
                raise ValueError("固定暴露仅用于显式 assume_static=True 的合成实验")
            exposures = np.broadcast_to(exposures, (T, *exposures.shape))
            caps = np.broadcast_to(caps, (T, N))
            if current_exposure_matrix is None:
                current_exposure_matrix = exposures[-1]
        if exposures.ndim != 3 or exposures.shape[:2] != (T, N) or caps.shape != (T, N):
            raise ValueError("历史暴露/市值须按收益行对齐为 (T,N,K)/(T,N)")
        K = exposures.shape[2]
        current = np.asarray(current_exposure_matrix, dtype=float)
        if current.shape != (N, K) or len(factor_names) != K or len(symbols) != N:
            raise ValueError("必须提供预测时点暴露 (N,K)，并对齐因子和证券名称")
        if not all(np.isfinite(v).all() for v in (historical_returns, exposures, caps, current)) or np.any(caps <= 0):
            raise ValueError("收益/暴露须有限，市值须为正")
        factor_returns = np.zeros((T, K))
        specific_residuals = np.zeros((T, N))
        for t in range(T):
            X = exposures[t]
            # น้ำหนักเป้าหมายของ WLS คือ sqrt(cap); คูณเมทริกซ์การออกแบบ/ผลตอบแทนด้วย sqrt(W)
            sqrt_w = caps[t] ** .25
            f_t = np.linalg.lstsq(X * sqrt_w[:, None], historical_returns[t] * sqrt_w, rcond=None)[0]
            factor_returns[t] = f_t
            specific_residuals[t] = historical_returns[t] - X @ f_t

        # 2. คำนวณเมทริกซ์ความแปรปรวนร่วมของแฟกเตอร์ F (K, K) ด้วย Exponential Weighted Moving Average (EWMA)
        ewma_w = self._ewma_weights(T).reshape(-1, 1) # (T, 1)
        centered_f = factor_returns - np.sum(factor_returns * ewma_w, axis=0)
        
        # F = (centered_f * w)^T @ centered_f * annual_factor
        weighted_f = centered_f * np.sqrt(ewma_w)
        F_matrix = (weighted_f.T @ weighted_f) * self.annual_factor

        # 3. ปรับ Residual ให้ศูนย์กลาง คำนวณความแปรปรวนเฉพาะตัวและทำ Shrinkage ด้วยความเข้มคงที่
        centered_u = specific_residuals - np.sum(specific_residuals * ewma_w, axis=0)
        weighted_u = centered_u * np.sqrt(ewma_w)
        raw_specific_var = np.sum(weighted_u ** 2, axis=0) * self.annual_factor

        # เพื่อความง่ายในการสอน: ทำ Shrinkage เข้าหาค่ามัธยฐานความแปรปรวนของทั้งตลาดด้วยความเข้มคงที่; ยังไม่ได้ทำ Grouped Adaptive Bayesian Estimation
        prior_mean_var = np.median(raw_specific_var)
        shrunk_specific_var = (
            (1.0 - self.shrinkage) * raw_specific_var + 
            self.shrinkage * prior_mean_var
        )

        return BarraRiskMatrices(
            exposure_matrix=current,
            factor_covariance=F_matrix,
            specific_variance=shrunk_specific_var,
            factor_names=factor_names,
            symbols=symbols
        )


# ================= Unit Test และการสาธิตการแจกแจงความเสี่ยง =================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 500  # ตัวอย่างคล้ายหุ้น CSI 500
    t_history = 250 # วันทำการย้อนหลัง 1 ปี
    n_factors = 10  # 6 แฟกเตอร์อุตสาหกรรม + 4 แฟกเตอร์สไตล์ (Size, Value, Momentum, Vol)

    factor_names = [f"industry_{i}" for i in range(6)] + ["size", "value", "momentum", "volatility"]
    symbols = [f"{s:06d}.SH" for s in range(n_stocks)]

    # 1. จำลองเมทริกซ์ Factor Exposure X
    # 6 คอลัมน์แรกเป็น One-Hot สำหรับอุตสาหกรรม, 4 คอลัมน์หลังเป็นคุณลักษณะสไตล์ผ่านการ Standardization
    X_mat = np.zeros((n_stocks, n_factors))
    ind_assign = np.random.choice(6, n_stocks)
    for s_idx in range(n_stocks):
        X_mat[s_idx, ind_assign[s_idx]] = 1.0
    X_mat[:, 6:] = np.random.normal(0, 1.0, (n_stocks, 4))

    # จำลองมูลค่าตามราคาตลาด
    market_caps = np.exp(np.random.uniform(np.log(50), np.log(3000), n_stocks))

    # 2. จำลองผลตอบแทน Time-series จริง
    true_f_cov = np.eye(n_factors) * 0.04 / 250
    f_returns_sim = np.random.multivariate_normal(np.zeros(n_factors), true_f_cov, t_history)
    u_sim = np.random.normal(0, 0.02, (t_history, n_stocks))
    # Exposure ย้อนหลังทราบก่อนเริ่มผลตอบแทนที่สอดคล้องกัน; จุดเวลาคาดการณ์จะส่ง X_mat แยกต่างหาก
    X_history = np.broadcast_to(X_mat, (t_history, n_stocks, n_factors)).copy()
    X_history[:, :, 6:] += np.random.normal(0, .1, (t_history, n_stocks, 4))
    R_sim = np.einsum('tnk,tk->tn', X_history, f_returns_sim) + u_sim

    # 3. รันเอนจินแบบจำลองความเสี่ยง Barra
    engine = AshareBarraRiskEngine(half_life_days=60, shrinkage_intensity=0.25)
    risk_model = engine.estimate_risk_model(
        historical_returns=R_sim,
        exposure_matrix=X_history,
        current_exposure_matrix=X_mat,
        market_caps=np.broadcast_to(market_caps, (t_history, n_stocks)),
        factor_names=factor_names,
        symbols=symbols
    )

    print(">>> Barra 风险模型估计成功！")
    print(f"  • 因子协方差矩阵 F 形状 : {risk_model.factor_covariance.shape}")
    print(f"  • 特异性风险对角矩阵形状: {risk_model.specific_variance.shape}")

    # 4. จำลองพอร์ตกองทุนดัชนีแบบถ่วงน้ำหนักเท่ากัน vs พอร์ตกองทุนดัชนีเสริมผลตอบแทน (Index Enhancement) ที่มีความเบี่ยงเบนเชิงรุก
    benchmark_weights = np.ones(n_stocks) / n_stocks
    active_weights = benchmark_weights.copy()
    # เพิ่มน้ำหนักเชิงรุก (Active Overweight) 50 ตัวแรก และลดน้ำหนัก (Underweight) 50 ตัวหลัง
    active_weights[:50] += 0.001
    active_weights[-50:] -= 0.001
    tracking_diff_weights = active_weights - benchmark_weights

    # คำนวณความคลาดเคลื่อนในการติดตาม (Tracking Error) เชิงรุก
    te_var, te_factor_var, te_spec_var = risk_model.compute_portfolio_variance(tracking_diff_weights)
    annual_te = np.sqrt(te_var) * 100.0

    print("\n>>> 指数增强组合主动风险 (Tracking Error) 分解报告:")
    print(f"  • 年化总跟踪误差 (Annualized TE)  : {annual_te:6.2f}%")
    print(f"  • 公共因子风险占比 (Factor Risk) : {te_factor_var / te_var * 100:6.2f}%")
    print(f"  • 个股特异风险占比 (Specific Risk): {te_spec_var / te_var * 100:6.2f}%")
