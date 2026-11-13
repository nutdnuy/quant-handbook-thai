"""ข้อมูลสังเคราะห์แบบกำหนดผลได้ (Deterministic) ขนาดเล็ก ทั้งหลักทรัพย์ ตัวเลขงบการเงิน และสถานะล้วนเป็นข้อมูลจำลองเพื่อการสอน ไม่ได้สะท้อนตลาดจริง"""
from datetime import date, datetime, timedelta

import polars as pl


def sample_data() -> dict[str, pl.DataFrame]:
    # กำหนดชัดเจนว่าเป็นปฏิทินจำลอง สำหรับ Production ต้องเปลี่ยนไปใช้ปฏิทินตลาดหลักทรัพย์ที่ผ่านการตรวจสอบแล้ว
    days = [date(2023, 1, 2) + timedelta(days=i) for i in range(124)]
    days = [day for day in days if day.weekday() < 5]
    calendar = pl.DataFrame({"trade_date": days})
    basics = pl.DataFrame({
        "symbol": ["OLD.SH", "HALT.SZ", "NEW.SZ"],
        "list_date": [date(2023, 1, 2), date(2023, 1, 2), date(2023, 4, 10)],
        "delist_date": [None, None, None],
    }, schema_overrides={"delist_date": pl.Date})
    rows = []
    for symbol, listed, _ in basics.iter_rows():
        for i, day in enumerate(days):
            if day < max(listed, date(2023, 3, 1)):
                continue  # โหลดข้อมูลราคาตลาดเฉพาะตั้งแต่เดือนมีนาคม แต่วันที่เข้าจดทะเบียนยังคงต้องคำนวณตั้งแต่เดือนมกราคม
            volume = 0.0 if symbol == "HALT.SZ" and date(2023, 4, 10) <= day <= date(2023, 4, 20) else 1000.0
            rows.append({"symbol": symbol, "trade_date": day, "open": 100.0 + i,
                         "high": 102.0 + i, "low": 99.0 + i, "close": 101.0 + i,
                         "close_hfq": 101.0 + i, "volume": volume, "turnover": volume * (101.0 + i)})
    status = pl.DataFrame([
        {"symbol": symbol, "effective_date": listed,
         "available_at": datetime.combine(listed, datetime.min.time()),
         "is_st": False, "is_delisting": False}
        for symbol, listed, _ in basics.iter_rows()
    ])
    financials = pl.DataFrame([
        {"symbol": "OLD.SH", "report_period": date(2022, 12, 31),
         "available_at": datetime(2023, 3, 31, 20), "roe": .30, "net_profit_yoy": .19},
        {"symbol": "OLD.SH", "report_period": date(2023, 3, 31),
         "available_at": datetime(2023, 4, 26, 20), "roe": .09, "net_profit_yoy": .20},
        # การแก้ไขที่มาล่าช้าของงวดรายงานเก่า ไม่ควรเขียนทับรายงานไตรมาส 1 ปัจจุบันให้กลายเป็นรายงานประจำปี
        {"symbol": "OLD.SH", "report_period": date(2022, 12, 31),
         "available_at": datetime(2023, 4, 27, 10), "roe": .25, "net_profit_yoy": .15},
    ])
    industries = pl.DataFrame([
        {"symbol": symbol, "effective_date": listed,
         "available_at": datetime.combine(listed, datetime.min.time()),
         "taxonomy": "demo-v1", "industry_name": industry}
        for (symbol, listed, _), industry in zip(basics.iter_rows(), ["食品饮料", "房地产", "电子"])
    ])
    return dict(calendar=calendar, basics=basics, bars=pl.DataFrame(rows),
                status=status, financials=financials, industries=industries)
