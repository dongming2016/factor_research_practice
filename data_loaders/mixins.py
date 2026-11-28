"""
mixins.py
提供可插拔的对齐与数据处理能力
"""
import os
from pathlib import Path
from typing import List, Optional
import polars as pl
from .trading_calendar import TradingCalendar


class CalendarAwareMixin:
    """提供统一日历访问能力的基石 Mixin"""
    @property
    def calendar(self) -> TradingCalendar:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(Path(base_path).parent, 'datas/trading_calendar.parquet')
        return TradingCalendar().initialize(file_path = file_path, date_col = 'date')


class GridAlignmentMixin(CalendarAwareMixin):
    """
    能力：笛卡尔积截面网格对齐
    适用标的：个股、ETF、转债等具有 (date, sec_id) 属性且存在停牌断流的面板数据
    """
    def align_by_grid(
        self,
        df: pl.DataFrame,
        id_col: str = "sec_id",
        date_col: str = "date",
        ffill_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        if ffill_cols is None:
            ffill_cols = ["close"]

        # 1. 提取所有标的代码
        unique_ids = df.select(id_col).unique()

        # 2. 生成 (全量交易日 x 标的代码) 网格
        grid = self.calendar.df.select(pl.col("date").alias(date_col)).join(
            unique_ids, how="cross"
        )

        # 3. 合并并打上停牌标记
        aligned = (
            grid.join(df, on=[date_col, id_col], how="left")
            .with_columns([
                # 判定停牌：价格缺失或成交量为 0
                (
                    pl.col("close").is_null()
                    | (pl.col("volume").fill_null(0.0) == 0.0)
                ).alias("is_paused"),
                pl.col("volume").fill_null(0.0),
            ])
            .with_columns([
                pl.col(c).fill_null(strategy="forward").over(id_col)
                for c in ffill_cols
            ])
            .sort([id_col, date_col])
        )
        return aligned


class AsOfAlignmentMixin(CalendarAwareMixin):
    """
    能力：基于实际公告日/发布日的向前点时对齐 (Point-in-Time)
    适用标的：宏观时序指标 (PMI/CPI)、分析师预期、财报基本面指标
    """
    def align_asof(
        self,
        df: pl.DataFrame,
        notice_date_col: str = "notice_date",
        by_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        if by_col is not None and 'sec_id' in df.columns:
            raise ValueError(
                '检测到输入面板包含 "sec_id" 列， 多标的pit对齐必须显示指定 by_col="sec_id",否则会导致跨股票数据串扰！'
            )
        cal = self.calendar.df.select("date").sort("date")
        if start_date:
            cal = cal.filter(pl.col("date") >= pl.lit(start_date))
        if end_date:
            cal = cal.filter(pl.col("date") <= pl.lit(end_date))
        cal = cal.sort('date')

        join_kwargs = {
            "left_on": "date",
            "right_on": notice_date_col,
            "strategy": "backward",  # 严格向前查找最新已公开数据
        }
        if by_col:
            # 需要按照个股来构建一个个股 * 日期的面板，以为每个个股构建独立的交易日历
            unique_ids = df.select(by_col).unique()
            left_grid = cal.join(unique_ids, how="cross").sort([by_col, 'date'])
            data = (
                df.filter(pl.col(notice_date_col).is_not_null())
                .unique(subset=[by_col, notice_date_col], keep='last')
                .sort([by_col, notice_date_col])
            )
            join_kwargs["by"] = by_col
            return  left_grid.join(data, **join_kwargs)
        data = (df.filter(pl.col(notice_date_col).is_null())
                .sort(notice_date_col))
        aligned = cal.join_asof(data, **join_kwargs)
        return aligned


class DateFilterMixin(CalendarAwareMixin):
    """
    能力：简单交易日过滤与非交易日剔除
    适用标的：基金净值 (NAV) 或指数日频收益率
    """
    def filter_trading_days(
        self, df: pl.DataFrame, date_col: str = "date"
    ) -> pl.DataFrame:
        cal_dates = self.calendar.df.select(pl.col("date").alias(date_col))
        return df.join(cal_dates, on=date_col, how="inner").sort(date_col)