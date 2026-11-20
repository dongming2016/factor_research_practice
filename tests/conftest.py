"""
tests/conftest.py
"""

from datetime import date
import polars as pl
import pytest
from data_loaders.trading_calendar import TradingCalendar


@pytest.fixture(autouse=True)
def reset_calendar_singleton():
    """每个测试函数执行前，强制重置 TradingCalendar 单例状态"""
    TradingCalendar._instance = None
    yield
    TradingCalendar._instance = None


@pytest.fixture
def mock_calendar_df():
    """
    提供模拟的 2024 年开市交易日（含常规周五、节前调休等情况）
    2024-04-01 至 2024-04-12 期间交易日
    注: 4月4日~4月5日清明休市，周五非交易日
    """
    return pl.DataFrame({
        "trade_date": [
            "2024-04-01", "2024-04-02", "2024-04-03",
            # 4月4日、5日休市
            "2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12"
        ],
        "is_open": [1, 1, 1, 1, 1, 1, 1, 1]
    })


@pytest.fixture
def initialized_calendar(tmp_path, mock_calendar_df):
    """在临时目录创建日历文件并初始化单例"""
    cal_file = tmp_path / "test_calendar.csv"
    mock_calendar_df.write_csv(cal_file)

    tc = TradingCalendar().initialize(
        file_path=cal_file,
        date_col="trade_date",
        is_open_col="is_open"
    )
    return tc