"""
ตัวอย่างการสอนสำหรับหุ้น A-Share: ตัวประมวลผล Orthogonalization ของแฟกเตอร์ (Factor) (AshareFactorOrthogonalizer)
ประกอบด้วย: Symmetric Orthogonalization (Löwdin) และ Gram-Schmidt Sequential Orthogonalization
"""

from typing import List, Literal, Tuple, Dict
from datetime import date
import numpy as np
import polars as pl


class AshareFactorOrthogonalizer:
    """
    เอนจิน Orthogonalization แฟกเตอร์แบบภาคตัดขวาง (Cross-sectional) สำหรับหุ้น A-Share
    """

    def __init__(
        self,
        method: Literal["symmetric", "gram_schmidt"] = "symmetric",
        regularization: float = 1e-6
    ):
        """
        พารามิเตอร์:
            method: อัลกอริทึม Orthogonalization เลือกได้ระหว่าง 'symmetric' (Symmetric Orthogonalization/Löwdin) หรือ 'gram_schmidt' (Gram-Schmidt Sequential Orthogonalization)
            regularization: ค่าความเผื่อไอเกนแวลูสัมพัทธ์ขั้นต่ำ; ปฏิเสธข้อมูลเข้าที่มีภาวะไม่เสถียร/collinear ชัดเจน
        """
        self.method = method
        if not 0 < regularization < 1:
            raise ValueError("相对特征值容忍度必须在 0 和 1 之间")
        self.regularization = regularization

    def _symmetric_orthogonalize(self, F: np.ndarray) -> np.ndarray:
        """
        การคำนวณ Löwdin Symmetric Orthogonalization
        อินพุต F: (N, K) เมทริกซ์ภาคตัดขวาง (mean 0, variance 1 แล้ว)
        เอาต์พุต F_star: (N, K) เมทริกซ์เชิงตั้งฉาก
        """
        N, K = F.shape
        # คำนวณเมทริกซ์สหสัมพันธ์ของกลุ่มตัวอย่าง S = (F^T @ F) / (N - 1)
        # เพื่อให้ตรงกับสูตรโดยสมบูรณ์ จึงใช้ S = F^T @ F โดยตรง
        S = F.T @ F

        # การแยกตัวประกอบไอเกนแวลูและไอเกนเวกเตอร์ S = V @ diag(eigenvals) @ V^T
        # eigh ออกแบบมาสำหรับเมทริกซ์สมมาตร/เฮอร์มิเชียนโดยเฉพาะ มีความเสถียรเชิงตัวเลขสูงกว่า eig ทั่วไปมาก
        eigenvals, V = np.linalg.eigh(S)

        # ปฏิเสธเมทริกซ์ที่มีภาวะไม่เสถียร ไม่ตัดทอนไอเกนแวลูเพื่อสร้างฟูลแรงก์เทียม
        if eigenvals[0] <= self.regularization * eigenvals[-1]:
            raise ValueError("因子矩阵秩不足或近似共线，请删除冗余因子或先降维")

        # 计算 S^{-1/2} = V @ diag(1 / sqrt(lambda)) @ V^T
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvals))
        S_inv_sqrt = V @ diag_inv_sqrt @ V.T

        # F* = F @ S^{-1/2}
        F_star = F @ S_inv_sqrt

        # ปรับสเกลข้อมูลแต่ละคอลัมน์ใหม่ (mean 0, variance 1) เพื่อความสะดวกในการจับคู่ให้คะแนนกับผลตอบแทนในภายหลัง
        # หมายเหตุ: การคูณด้วยค่าคงที่ไม่เปลี่ยนความสัมพันธ์ระดับ 0 ระหว่างคอลัมน์
        col_stds = np.std(F_star, axis=0, ddof=1)
        col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)
        F_star = (F_star - np.mean(F_star, axis=0)) / col_stds

        return F_star

    def _gram_schmidt_orthogonalize(self, F: np.ndarray) -> np.ndarray:
        """
        การคำนวณ Gram-Schmidt Orthogonalization แบบดั้งเดิม (กำจัดเศษเหลือจากการโปรเจกชันทีละขั้นตามลำดับคอลัมน์)
        """
        N, K = F.shape
        Q = np.zeros_like(F)

        for j in range(K):
            v = F[:, j].copy()
            # หักลบโปรเจกชันบนเบสิสเชิงตั้งฉากก่อนหน้าทั้งหมดตามลำดับ
            for i in range(j):
                q_i = Q[:, i]
                proj = np.dot(v, q_i) / (np.dot(q_i, q_i) + 1e-12)
                v = v - proj * q_i

            norm = np.linalg.norm(v)
            if norm < 1e-8:
                Q[:, j] = v  # เวกเตอร์ศูนย์ที่แทบจะขึ้นต่อกันเชิงเส้นสมบูรณ์
            else:
                Q[:, j] = v / norm

        # Standardization ให้มีความแปรปรวนเท่ากับหนึ่ง
        col_stds = np.std(Q, axis=0, ddof=1)
        col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)
        Q = (Q - np.mean(Q, axis=0)) / col_stds
        return Q

    def fit_transform_matrix(
        self,
        F: np.ndarray,
        factor_names: List[str]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        การคำนวณ Orthogonalization เมทริกซ์หลักและเอาต์พุตตัวชี้วัดการวินิจฉัย
        """
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or not np.isfinite(F).all():
            raise ValueError("因子矩阵必须为有限的二维数组")
        N, K = F.shape
        if K < 1 or N <= K or len(factor_names) != K or len(set(factor_names)) != K:
            raise ValueError("需要 N > K >= 1，且因子名称唯一、数量等于列数")
        F = F - F.mean(axis=0)
        std = F.std(axis=0, ddof=1)
        if np.any(std <= 1e-12):
            raise ValueError("常数列导致矩阵秩不足")
        F = F / std
        eigenvalues = np.linalg.eigvalsh(F.T @ F)
        if eigenvalues[0] <= self.regularization * eigenvalues[-1]:
            raise ValueError("因子矩阵秩不足或近似共线，请删除冗余因子或先降维")

        # 1. วินิจฉัยความสมบูรณ์ของเมทริกซ์ดั้งเดิม
        orig_corr = np.atleast_2d(np.corrcoef(F, rowvar=False))
        orig_cond = np.linalg.cond(orig_corr)

        # 2. คำนวณ Orthogonalization
        if self.method == "symmetric":
            F_ortho = self._symmetric_orthogonalize(F)
        elif self.method == "gram_schmidt":
            F_ortho = self._gram_schmidt_orthogonalize(F)
        else:
            raise ValueError(f"未知的正交化方法: {self.method}")

        # 3. ตรวจสอบเมทริกซ์หลังทำ Orthogonalization
        ortho_corr = np.atleast_2d(np.corrcoef(F_ortho, rowvar=False))
        ortho_cond = np.linalg.cond(ortho_corr)

        # 4. คำนวณความสัมพันธ์เพียร์สันของการคงอยู่ (Retention Correlation) ระหว่างแต่ละแฟกเตอร์ใหม่กับแฟกเตอร์ตั้งต้นที่สอดคล้องกัน
        retentions = {}
        for idx, name in enumerate(factor_names):
            ret_corr = np.corrcoef(F[:, idx], F_ortho[:, idx])[0, 1]
            retentions[name] = float(ret_corr)

        diagnostics = {
            "original_condition_number": float(orig_cond),
            "orthogonal_condition_number": float(ortho_cond),
            "retention_correlations": retentions,
            "max_abs_off_diagonal_corr": float(np.max(np.abs(ortho_corr - np.eye(K))))
        }

        return F_ortho, diagnostics

    def transform_polars(
        self,
        df: pl.DataFrame,
        factor_cols: List[str],
        date_col: str = "trade_date"
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        วนซ้ำตามภาคตัดขวางของวันที่และดำเนินการ Orthogonalization
        ส่งกลับ: (DataFrame ใหม่พร้อมคอลัมน์แฟกเตอร์เชิงตั้งฉาก, DataFrame ตัวชี้วัดการวินิจฉัยภาคตัดขวาง)
        """
        ortho_cols = [f"{col}_ortho" for col in factor_cols]
        diag_records = []
        result_dfs = []

        # จัดกลุ่มตามวันที่ภาคตัดขวางเพื่อทำ Orthogonalization
        dates = df[date_col].unique().sort().to_list()

        for cur_date in dates:
            sub_df = df.filter(pl.col(date_col) == cur_date)
            # ดึงเมทริกซ์แฟกเตอร์
            f_mat = sub_df.select(factor_cols).to_numpy()

            # ดำเนินการ Orthogonalization
            f_ortho, diag = self.fit_transform_matrix(f_mat, factor_cols)

            # บันทึกการวินิจฉัย
            diag_record = {
                date_col: cur_date,
                "orig_cond": diag["original_condition_number"],
                "ortho_cond": diag["orthogonal_condition_number"],
                "max_residual_corr": diag["max_abs_off_diagonal_corr"]
            }
            # บันทึกค่าการคงอยู่ของแต่ละแฟกเตอร์
            for col in factor_cols:
                diag_record[f"retention_{col}"] = diag["retention_correlations"][col]
            diag_records.append(diag_record)

            # รวมกลับเข้า Polars DataFrame
            ortho_pl_dict = {
                ortho_cols[i]: f_ortho[:, i] for i in range(len(factor_cols))
            }
            sub_ortho_df = sub_df.with_columns([
                pl.Series(k, v) for k, v in ortho_pl_dict.items()
            ])
            result_dfs.append(sub_ortho_df)

        final_df = pl.concat(result_dfs)
        diag_df = pl.DataFrame(diag_records)
        return final_df, diag_df


# ================= สาธิตด้วยข้อมูลสังเคราะห์ (ผลลัพธ์ทางสถิติไม่ใช่หลักฐานรับรองผลตอบแทนการเทรดจริง (Live Trading)) =================
if __name__ == "__main__":
    np.random.seed(42)
    n_stocks = 800  # คล้ายกับภาคตัดขวางของ Stock Universe ดัชนี CSI 800

    # 1. จำลองชุดแฟกเตอร์สังเคราะห์ที่มีความสัมพันธ์กัน
    # สมมติว่ามีตัวแปรแฝงร่วมอยู่เบื้องหลัง (เช่น สไตล์โมเมนตัมการเติบโตของหุ้นขนาดกลางและเล็กที่ยังไม่ได้ถูก Neutralization อย่างหมดจด)
    latent_growth = np.random.normal(0, 1, n_stocks)
    latent_value = np.random.normal(0, 1, n_stocks)

    # สร้างแฟกเตอร์กลุ่มโมเมนตัม 3 ตัวที่มีภาวะร่วมเส้นตรงระดับสูง
    mom_20d = latent_growth * 0.8 + np.random.normal(0, 0.4, n_stocks)
    mom_60d = latent_growth * 0.75 + np.random.normal(0, 0.45, n_stocks)
    macd_power = latent_growth * 0.7 + np.random.normal(0, 0.5, n_stocks)

    # สร้างแฟกเตอร์กลุ่มคุณค่า 2 ตัวที่มีภาวะร่วมเส้นตรงระดับสูง
    ep_ratio = latent_value * 0.85 + np.random.normal(0, 0.35, n_stocks)
    bp_ratio = latent_value * 0.80 + np.random.normal(0, 0.40, n_stocks)

    raw_data = {
        "trade_date": [date(2026, 3, 31)] * n_stocks,
        "symbol": [f"{i:06d}.SH" for i in range(n_stocks)],
        "mom_20d": mom_20d,
        "mom_60d": mom_60d,
        "macd_power": macd_power,
        "ep_ratio": ep_ratio,
        "bp_ratio": bp_ratio,
    }
    sample_df = pl.DataFrame(raw_data)
    factor_list = ["mom_20d", "mom_60d", "macd_power", "ep_ratio", "bp_ratio"]

    print(">>> 原始因子截面相关系数矩阵:")
    orig_corr = np.corrcoef(sample_df.select(factor_list).to_numpy(), rowvar=False)
    for i, f1 in enumerate(factor_list):
        row_str = " ".join([f"{orig_corr[i, j]:6.3f}" for j in range(len(factor_list))])
        print(f"  {f1:12s} | {row_str}")

    # 2. รัน Löwdin Symmetric Orthogonalization
    engine = AshareFactorOrthogonalizer(method="symmetric")
    ortho_df, diag_df = engine.transform_polars(sample_df, factor_list)

    print("\n>>> 对称正交化后的因子相关系数矩阵:")
    ortho_cols = [f"{f}_ortho" for f in factor_list]
    ortho_corr = np.corrcoef(ortho_df.select(ortho_cols).to_numpy(), rowvar=False)
    for i, f1 in enumerate(ortho_cols):
        row_str = " ".join([f"{ortho_corr[i, j]:6.3f}" for j in range(len(ortho_cols))])
        print(f"  {f1:18s} | {row_str}")

    print("\n>>> 正交化诊断指标报告:")
    print(diag_df)

    # 3. เปรียบเทียบกับ Gram-Schmidt Orthogonalization
    gs_engine = AshareFactorOrthogonalizer(method="gram_schmidt")
    gs_ortho_df, gs_diag_df = gs_engine.transform_polars(sample_df, factor_list)
    print("\n>>> 施密特正交化信息保留度 (越排在后面丢失越严重):")
    for f in factor_list:
        sym_ret = diag_df[f"retention_{f}"][0]
        gs_ret = gs_diag_df[f"retention_{f}"][0]
        print(f"  因子 {f:12s} -> 对称正交化保留度: {sym_ret:6.3f} | 施密特保留度: {gs_ret:6.3f}")
