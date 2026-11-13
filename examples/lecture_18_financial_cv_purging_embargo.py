"""Forward Validation จัดกลุ่มตามวันที่และการแบ่งชุดข้อมูล CPCV; ช่วงปิดที่ทับซ้อนกันใช้การล้างข้อมูล (Purging) แบบรัดกุม"""
import itertools
import numpy as np
from examples.ml_utils import demo_dates, interval_coordinates


class PurgedGroupTimeSeriesSplit:
    """Forward Validation แบบหน้าต่างขยาย (Expanding Window): ใช้เฉพาะวันฝึกสอนที่อยู่ก่อนจุดเริ่มต้นทดสอบและลาเบลสิ้นสุดลงแล้วเท่านั้น

    คง embargo_pct ไว้เพื่อความเข้ากันได้ โฟลด์ปัจจุบันไม่ใช้ข้อมูลฝึกสอนในอนาคต จึงไม่จำเป็นต้องมี post-embargo
    หากไม่ได้ส่ง label_end_dates เข้ามา H จะนับตามลำดับวันที่ซื้อขายทั้งหมดที่ป้อนเข้ามา
    """
    def __init__(self, n_splits=5, forward_horizon=5, embargo_pct=0.02):
        if n_splits < 1 or not 0 <= embargo_pct < 1:
            raise ValueError('折数必须为正，embargo_pct 必须在 [0,1)')
        self.n_splits, self.forward_horizon = n_splits, forward_horizon
        self.embargo_pct = embargo_pct

    def split(self, dates, label_end_dates=None):
        unique, starts, ends = interval_coordinates(dates, label_end_dates, self.forward_horizon)
        if len(unique) < self.n_splits + 1:
            raise ValueError('日期不足以构建训练窗口与测试折')
        blocks = np.array_split(np.arange(len(unique)), self.n_splits + 1)
        for test_days in blocks[1:]:
            train_days = np.flatnonzero((np.arange(len(unique)) < test_days[0])
                                        & (ends < test_days[0]))
            if not len(train_days):
                raise ValueError('清洗后训练折为空；增加历史窗口或减少折数')
            yield np.flatnonzero(np.isin(starts, train_days)), np.flatnonzero(np.isin(starts, test_days))


class CombinatorialPurgedCV:
    """ตัวแบ่ง CPCV สองด้านสำหรับงานวิจัย; ไม่ต่อเส้นทาง Backtest และไม่คำนวณ PBO"""
    def __init__(self, n_blocks=6, n_test_blocks=2, forward_horizon=5, embargo_pct=0.01):
        if not 1 <= n_test_blocks < n_blocks or not 0 <= embargo_pct < 1:
            raise ValueError('需要 1 <= n_test_blocks < n_blocks 且 0 <= embargo_pct < 1')
        self.n_blocks, self.n_test_blocks = n_blocks, n_test_blocks
        self.forward_horizon, self.embargo_pct = forward_horizon, embargo_pct

    def split(self, dates, label_end_dates=None):
        unique, starts, ends = interval_coordinates(dates, label_end_dates, self.forward_horizon)
        if len(unique) < self.n_blocks:
            raise ValueError('日期数不能少于时间块数')
        blocks = np.array_split(np.arange(len(unique)), self.n_blocks)
        embargo = int(np.ceil(len(unique) * self.embargo_pct))
        day_ids = np.arange(len(unique))
        for ids in itertools.combinations(range(self.n_blocks), self.n_test_blocks):
            test_days = np.concatenate([blocks[b] for b in ids])
            allowed = ~np.isin(day_ids, test_days)
            for b in ids:
                block = blocks[b]
                # ยูเนียนของช่วงลาเบล: วันที่สังเกตการณ์ภายในบล็อกต่อเนื่องกัน ลาเบลรายวันครอบคลุมอย่างน้อยวันปัจจุบัน
                left, right = block[0], int(ends[block].max())
                allowed &= ~((day_ids <= right) & (ends >= left))
                # embargo ส่วนเพิ่มจะนับถัดจากจุดสิ้นสุดของลาเบลทดสอบที่ช้าที่สุด ไม่ใช่ส่วนท้ายของวันที่ฟีเจอร์
                allowed &= ~((day_ids > right) & (day_ids <= right + embargo))
            if not allowed.any():
                raise ValueError('清洗后训练折为空；调整块数、标签跨度或 embargo')
            yield np.flatnonzero(allowed[starts]), np.flatnonzero(np.isin(starts, test_days)), ids


if __name__ == '__main__':
    calendar = demo_dates(750, '2023-01-02')
    dates = np.repeat(calendar, 40)
    for fold, (train, test) in enumerate(PurgedGroupTimeSeriesSplit(n_splits=3).split(dates), 1):
        unique = np.unique(dates)
        assert np.searchsorted(unique, dates[train].max()) + 5 < np.searchsorted(unique, dates[test].min())
        print(f'前向折 {fold}: 训练 {dates[train].min()}～{dates[train].max()}，测试 {dates[test].min()}～{dates[test].max()}')
    count = sum(1 for _ in CombinatorialPurgedCV().split(dates))
    print(f'CPCV 生成 {count} 组有标签清洗的研究切分；不等同于独立样本或实盘前向路径。')
