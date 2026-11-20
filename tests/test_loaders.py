"""
tests/test_loaders.py
"""

from datetime import date
import polars as pl
import pytest

from data_loaders.trading_calendar import TradingCalendar
from data_loaders.mixins import (
    GridAlignmentMixin,
    AsOfAlignmentMixin,
    DateFilterMixin
)
from data_loaders.loaders import BaseDataLoader


# =====================================================================
# 1. TradingCalendar 单例与日期算子测试
# =====================================================================

def test_calendar_singleton_identity(initialized_calendar):
    """测试单例模式的全局唯一性"""
    tc1 = TradingCalendar()
    tc2 = TradingCalendar()
    assert tc1 is tc2
    assert tc1 is initialized_calendar


def test_calendar_is_trading_day(initialized_calendar):
    """测试 O(1) 交易日判定"""
    tc = TradingCalendar()
    # 正常交易日
    assert tc.is_trading_day(date(2024, 4, 1)) is True
    assert tc.is_trading_day(date(2024, 4, 3)) is True
    # 清明节休市日
    assert tc.is_trading_day(date(2024, 4, 4)) is False
    assert tc.is_trading_day(date(2024, 4, 5)) is False
    # 周末
    assert tc.is_trading_day(date(2024, 4, 6)) is False


def test_calendar_get_range(initialized_calendar):
    """测试二分切片区间选择器 (闭区间)"""
    tc = TradingCalendar()
    # 字符串入参
    df_range = tc.get_range("2024-04-02", "2024-04-09", as_df=True)
    expected_dates = [
        date(2024, 4, 2), date(2024, 4, 3),
        date(2024, 4, 8), date(2024, 4, 9)
    ]
    assert df_range["date"].to_list() == expected_dates

    # 列表返回模式
    list_range = tc.get_range(date(2024, 4, 10), date(2024, 4, 12), as_df=False)
    assert list_range == [date(2024, 4, 10), date(2024, 4, 11), date(2024, 4, 12)]


def test_calendar_get_last_n_trading_days(initialized_calendar):
    """测试最近 N 个交易日滑动窗口截取"""
    tc = TradingCalendar()
    # 包含锚点日自身
    res = tc.get_last_n_trading_days(anchor_date="2024-04-09", n=3, include_anchor=True, as_df=False)
    assert res == [date(2024, 4, 3), date(2024, 4, 8), date(2024, 4, 9)]

    # 不包含锚点日
    res_ex = tc.get_last_n_trading_days(anchor_date="2024-04-09", n=2, include_anchor=False, as_df=False)
    assert res_ex == [date(2024, 4, 3), date(2024, 4, 8)]


def test_calendar_weekly_trading_days(initialized_calendar):
    """
    测试周频调仓日选择：
    2024年4月第一周（4月1日-4月3日开市，4日-5日休市），
    mode='last' 应正确识别 4月3日（周三）为当周调仓点，而非跳过该周。
    """
    tc = TradingCalendar()
    weekly_last = tc.get_weekly_trading_days(mode="last")
    dates = weekly_last["date"].to_list()

    # 第一周最后交易日应为周三 4月3日，第二周最后交易日应为周五 4月12日
    assert date(2024, 4, 3) in dates
    assert date(2024, 4, 12) in dates


# =====================================================================
# 2. GridAlignmentMixin 测试 (股票行情截面网格化、停牌与前向填充)
# =====================================================================

class DummyPriceLoader(BaseDataLoader, GridAlignmentMixin):
    def load(self, file_name: str, **kwargs) -> pl.DataFrame:
        pass


def test_grid_alignment_pause_and_ffill(initialized_calendar):
    """
    测试股票网格对齐：
    1. 某标的在中间交易日缺失（如停牌），应自动打上 is_paused=True 标签；
    2. 收盘价 close 应正确执行 forward fill。
    """
    loader = DummyPriceLoader()

    # 模拟原始行情：000001 在 2024-04-02 停牌（数据行缺失）
    # 000002 正常交易
    raw_prices = pl.DataFrame({
        "date": [
            date(2024, 4, 1), date(2024, 4, 3), # 000001 缺失 4-02
            date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3)
        ],
        "sec_id": ["000001", "000001", "000002", "000002", "000002"],
        "close": [10.0, 10.5, 20.0, 20.2, 20.1],
        "volume": [1000.0, 1200.0, 500.0, 600.0, 800.0]
    })

    # 执行网格对齐
    aligned = loader.align_by_grid(raw_prices, ffill_cols=["close"])

    # 截取 000001 在 4月2日 的表现
    p1_day2 = aligned.filter(
        (pl.col("sec_id") == "000001") & (pl.col("date") == date(2024, 4, 2))
    )

    assert p1_day2.height == 1
    # 校验停牌标记与前向填充值
    assert p1_day2["is_paused"][0] is True
    assert p1_day2["close"][0] == 10.0  # 沿用 4月1日的 10.0
    assert p1_day2["volume"][0] == 0.0


# =====================================================================
# 3. AsOfAlignmentMixin 测试 (宏观与财报点时对齐，杜绝未来函数)
# =====================================================================

class DummyMacroLoader(BaseDataLoader, AsOfAlignmentMixin):
    def load(self, file_name: str, **kwargs) -> pl.DataFrame:
        pass


def test_asof_pit_no_lookahead(initialized_calendar):
    """
    测试宏观指标严格的点时 (PIT) 对齐：
    宏观指标在 2024-04-09 发布新数值。
    在 4月1日~4月8日期间，系统读取的宏观值必须是旧值，绝对不能看到 4月9日的新值。
    """
    loader = DummyMacroLoader()

    # 模拟宏观公布事件：
    # 3月31日发布旧值 50.8，4月9日早间发布新值 51.5
    raw_macro = pl.DataFrame({
        "notice_date": [date(2024, 3, 31), date(2024, 4, 9)],
        "pmi": [50.8, 51.5]
    })

    aligned = loader.align_asof(raw_macro, notice_date_col="notice_date")

    # 验证 4月8日（公告日前一天）：必须只能看到 50.8
    day_before = aligned.filter(pl.col("date") == date(2024, 4, 8))
    assert day_before["pmi"][0] == 50.8

    # 验证 4月9日（公告日当天）：开始生效 51.5
    day_announcement = aligned.filter(pl.col("date") == date(2024, 4, 9))
    assert day_announcement["pmi"][0] == 51.5


# =====================================================================
# 4. DateFilterMixin 测试 (非开市日净值剔除)
# =====================================================================

class DummyNAVLoader(BaseDataLoader, DateFilterMixin):
    def load(self, file_name: str, **kwargs) -> pl.DataFrame:
        pass


def test_date_filter_trading_days(initialized_calendar):
    """测试周末与节假日净值剔除"""
    loader = DummyNAVLoader()

    raw_nav = pl.DataFrame({
        "date": [
            date(2024, 4, 3), # 交易日
            date(2024, 4, 4), # 清明休市
            date(2024, 4, 6), # 周六
            date(2024, 4, 8)  # 交易日
        ],
        "nav": [1.01, 1.02, 1.025, 1.03]
    })

    filtered = loader.filter_trading_days(raw_nav, date_col="date")
    assert filtered["date"].to_list() == [date(2024, 4, 3), date(2024, 4, 8)]


# =====================================================================
# 5. 异常边界与防御性测试
# =====================================================================

def test_uninitialized_calendar_raises_error():
    """测试未初始化日历时调用算子抛出清晰异常"""
    tc = TradingCalendar()
    with pytest.raises(RuntimeError, match="TradingCalendar 尚未初始化"):
        tc.is_trading_day(date(2024, 1, 1))