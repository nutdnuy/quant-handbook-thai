"""
เอนจินจัดสรรพอร์ตแบบกระจายความเสี่ยงตามลำดับขั้น (Hierarchical Risk Parity: HRP) เพื่อการศึกษา (AshareHierarchicalRiskParity)
ใช้ HRP รูปแบบที่คำนวณ Single-linkage โดยตรงจากระยะห่างสหสัมพันธ์ (Correlation Distance); ไม่รับประกัน ERC
ประกอบด้วย: การจัดกลุ่มเมทริกซ์ระยะห่าง, การจัดเรียงลำดับกึ่งทแยงมุม (Quasi-diagonalization), การจัดสรรน้ำหนักผกผันกับความเสี่ยงแบบแบ่งครึ่งซ้ำ (Recursive Bisection) และการวัดสัดส่วนความเสี่ยง TRC (Total Risk Contribution)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


class AshareHierarchicalRiskParity:
    """
    ออปติไมเซอร์การจัดสรรสินทรัพย์แบบ HRP
    """

    def __init__(self, annual_factor: float = 250.0):
        if annual_factor <= 0:
            raise ValueError("annual_factor 必须为正")
        self.annual_factor = annual_factor

    def _correl_distance(self, corr: np.ndarray) -> np.ndarray:
        """
        คำนวณเมทริกซ์ระยะห่างของสัมประสิทธิ์สหสัมพันธ์ d_ij = sqrt(0.5 * (1 - rho_ij))
        """
        dist = np.sqrt(0.5 * np.maximum(0.0, 1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        return dist

    def _get_quasi_diag_order(self, link_matrix: np.ndarray) -> List[int]:
        """
        ดึงลำดับกึ่งทแยงมุมของ leaf node จาก Linkage Matrix ของการจัดกลุ่มตามลำดับขั้น
        """
        root = to_tree(link_matrix, rd=False)
        order = []

        def _traverse(node):
            if node.is_leaf():
                order.append(node.id)
            else:
                _traverse(node.left)
                _traverse(node.right)

        _traverse(root)
        return order

    def _get_cluster_var(self, cov: np.ndarray, cluster_items: List[int]) -> float:
        """
        คำนวณความแปรปรวนความเสี่ยงรวมแบบถ่วงน้ำหนักด้วยส่วนกลับของความแปรปรวนของคลัสเตอร์ย่อย
        """
        sub_cov = cov[np.ix_(cluster_items, cluster_items)]
        # ถ่วงน้ำหนักด้วยส่วนกลับของความแปรปรวน (Inverse-variance weighting)
        diag_inv = 1.0 / np.maximum(np.diag(sub_cov), 1e-8)
        w = diag_inv / np.sum(diag_inv)
        cluster_var = float(w.T @ sub_cov @ w)
        return cluster_var

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        ordered_indices: List[int]
    ) -> np.ndarray:
        """
        จัดสรรน้ำหนักแบบแบ่งครึ่งซ้ำจากบนลงล่าง
        """
        weights = np.ones(len(ordered_indices))
        clusters = [ordered_indices]

        while len(clusters) > 0:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # แบ่งคลัสเตอร์ปัจจุบันออกเป็นครึ่งแรกและครึ่งหลัง
                    mid = len(cluster) // 2
                    c1 = cluster[:mid]
                    c2 = cluster[mid:]

                    # คำนวณความแปรปรวนรวมของคลัสเตอร์ย่อยทั้งสอง
                    v1 = self._get_cluster_var(cov, c1)
                    v2 = self._get_cluster_var(cov, c2)

                    # คำนวณสัมประสิทธิ์การจัดสรรน้ำหนักแบบผกผันกับความแปรปรวน alpha
                    alpha1 = v2 / (v1 + v2 + 1e-12)
                    alpha2 = 1.0 - alpha1

                    # ปรับสเกลน้ำหนักตำแหน่งเดิมของคลัสเตอร์ย่อย
                    weights[c1] *= alpha1
                    weights[c2] *= alpha2

                    next_clusters.append(c1)
                    next_clusters.append(c2)
            clusters = next_clusters

        return weights

    def compute_hrp_weights(
        self,
        cov_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        คำนวณน้ำหนักการจัดสรรแบบ HRP ทั้งหมด
        """
        cov_matrix = np.asarray(cov_matrix, dtype=float)
        if cov_matrix.ndim != 2 or not len(cov_matrix) or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("协方差须为非空方阵")
        if not np.isfinite(cov_matrix).all() or not np.allclose(cov_matrix, cov_matrix.T) or np.linalg.eigvalsh(cov_matrix).min() < -1e-10 or np.any(np.diag(cov_matrix) <= 0):
            raise ValueError("协方差须对称半正定，各资产方差为正")
        N = cov_matrix.shape[0]
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(N)]

        if len(asset_names) != N or len(set(asset_names)) != N:
            raise ValueError("资产名须唯一且与矩阵维度一致")
        if N == 1:
            return {asset_names[0]: 1.}

        # 1. คำนวณเมทริกซ์สัมประสิทธิ์สหสัมพันธ์และระยะห่างสหสัมพันธ์
        vols = np.sqrt(np.diag(cov_matrix))
        outer_vols = np.outer(vols, vols)
        corr = cov_matrix / np.maximum(outer_vols, 1e-8)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        dist_mat = self._correl_distance(corr)

        # 2. การจัดกลุ่มตามลำดับขั้น (Single Linkage)
        condensed_dist = squareform(dist_mat, checks=False)
        link = linkage(condensed_dist, method="single")

        # 3. จัดเรียงลำดับ leaf node แบบกึ่งทแยงมุม
        ordered_ids = self._get_quasi_diag_order(link)

        # 4. กำหนดน้ำหนักแบบแบ่งครึ่งซ้ำผกผันกับความเสี่ยง
        raw_weights = self._recursive_bisection(cov_matrix, ordered_ids)

        # แมปกลับไปยังลำดับสินทรัพย์เดิมและทำ Normalization
        final_weights = raw_weights / np.sum(raw_weights)
        
        weight_dict = {
            asset_names[i]: float(final_weights[i])
            for i in range(N)
        }
        return weight_dict

    def evaluate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        คำนวณสัดส่วนความเสี่ยงส่วนเพิ่ม (Marginal Risk Contribution: MRC) และสัดส่วนความเสี่ยงรวม (Total Risk Contribution: TRC)
        รับค่าความแปรปรวนร่วมระดับรายวัน; ส่งคืนค่าปรับเป็นรายปีทั้งหมด (เวกเตอร์ MRC, เวกเตอร์ TRC, ความผันผวนของพอร์ต)
        """
        port_var = float(weights.T @ cov_matrix @ weights)
        if port_var <= 0:
            raise ValueError("零或负组合方差时波动率风险贡献未定义")
        port_vol = np.sqrt(port_var)

        # MRC = (Σ w) / sigma_p
        mrc = (cov_matrix @ weights) / port_vol
        # TRC = w * MRC
        trc = weights * mrc

        ann_port_vol = port_vol * np.sqrt(self.annual_factor)
        return mrc * np.sqrt(self.annual_factor), trc * np.sqrt(self.annual_factor), ann_port_vol


# ================= ยูนิตเทสต์และการสาธิตการจัดสรรสินทรัพย์แบบ All-Weather =================
if __name__ == "__main__":
    np.random.seed(42)

    # จำลองพอร์ตหลายสินทรัพย์แบบ All-Weather ครอบคลุม 6 สินทรัพย์หลัก:
    # 1. สินทรัพย์ตราสารทุน (หุ้น A-Share ขนาดใหญ่ CSI 300, CSI 1000) - ความผันผวนต่อปี ~22%
    # 2. สินทรัพย์ตราสารหนี้ (พันธบัตรรัฐบาลอายุ 10 ปี ETF, หุ้นกู้เอกชน) - ความผันผวนต่อปี ~4%
    # 3. สินค้าโภคภัณฑ์ (ทองคำ ETF, ดัชนีสินค้าโภคภัณฑ์ Nanhua) - ความผันผวนต่อปี ~15%
    asset_classes = [
        "沪深300_ETF", "中证1000_ETF", 
        "十年国债_ETF", "企业信用债_ETF", 
        "黄金_ETF", "南华商品_ETF"
    ]
    N = len(asset_classes)

    # กำหนดความผันผวนต่อปีและความสัมพันธ์ที่ใกล้เคียงกับตลาดจริง
    vols_annual = np.array([0.20, 0.26, 0.035, 0.045, 0.16, 0.18])
    corr_matrix = np.array([
        [ 1.00,  0.82, -0.15, -0.05,  0.08,  0.35],
        [ 0.82,  1.00, -0.18, -0.08,  0.05,  0.42],
        [-0.15, -0.18,  1.00,  0.65,  0.25, -0.12],
        [-0.05, -0.08,  0.65,  1.00,  0.18, -0.08],
        [ 0.08,  0.05,  0.25,  0.18,  1.00,  0.30],
        [ 0.35,  0.42, -0.12, -0.08,  0.30,  1.00]
    ])

    # สร้างเมทริกซ์ความแปรปรวนร่วมระดับรายวันขึ้นมาใหม่
    outer_v = np.outer(vols_annual / np.sqrt(250.0), vols_annual / np.sqrt(250.0))
    daily_cov = corr_matrix * outer_v

    # 1. รันเอนจิน HRP
    hrp_engine = AshareHierarchicalRiskParity(annual_factor=250.0)
    hrp_weights_dict = hrp_engine.compute_hrp_weights(daily_cov, asset_classes)
    hrp_w = np.array(list(hrp_weights_dict.values()))

    # 2. เปรียบเทียบกับพอร์ตแบบถ่วงน้ำหนักเท่ากัน (Equal Weight)
    ew_w = np.ones(N) / N

    ivp_w = 1 / np.diag(daily_cov)
    ivp_w /= ivp_w.sum()
    _, ivp_trc, ivp_vol = hrp_engine.evaluate_risk_contributions(ivp_w, daily_cov)

    # 3. ประเมินสัดส่วนความเสี่ยง
    _, ew_trc, ew_vol = hrp_engine.evaluate_risk_contributions(ew_w, daily_cov)
    _, hrp_trc, hrp_vol = hrp_engine.evaluate_risk_contributions(hrp_w, daily_cov)

    print("=" * 80)
    print(f"📊 全天候多资产 HRP 风险平价对比报告")
    print("=" * 80)
    print(f"{'资产名称':<16} | {'等权资金比例':<12} | {'等权风险占比':<12} | {'HRP资金权重':<12} | {'HRP风险占比':<12}")
    print("-" * 80)
    for i, name in enumerate(asset_classes):
        ew_r_share = ew_trc[i] / np.sum(ew_trc) * 100
        hrp_r_share = hrp_trc[i] / np.sum(hrp_trc) * 100
        print(f"{name:<16} | {ew_w[i]*100:10.2f}%   | {ew_r_share:10.2f}%   | {hrp_w[i]*100:10.2f}%   | {hrp_r_share:10.2f}%")
    print("-" * 80)
    print(f"IVP 权重: {np.round(ivp_w, 4)}；风险占比: {np.round(ivp_trc / ivp_vol, 4)}；年化波动: {ivp_vol:.2%}")
    np.testing.assert_allclose(hrp_trc.sum(), hrp_vol)
    print("HRP 不保证资产风险贡献相等；未提供收益序列，故不计算夏普比率。")
    print(f"【组合总年化波动率对比】")
    print(f"  • 传统等权组合 (EW) 年化波动  : {ew_vol * 100:6.2f}%")
    print(f"  • 层级风险平价 (HRP) 年化波动 : {hrp_vol * 100:6.2f}%")
    print("=" * 80)
