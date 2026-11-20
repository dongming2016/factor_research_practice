"""
trading_calendar.py
全局唯一的交易日历单例，提供全局统一的时间基准与快速检索能力
"""
import bisect
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Optional, Union, List, Set,  Literal
import polars as pl

def _normalize_date(date_: Union[date, str]) -> date:
    if isinstance(date_, str):
        return date.strptime(date_.replace('/', '-'), "%Y-%m-%d")
    return date_

class TradingCalendar:
    _instance: Optional["TradingCalendar"] = None
    _lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        # 双重检查锁定 (Double-Checked Locking)，保障线程安全单例
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        file_path: Union[str, Path] = "./data/sse_calendar.csv",
        date_col: str = "trade_date",
        is_open_col: Optional[str] = "is_open",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> "TradingCalendar":
        """
        全生命周期仅需在系统启动阶段显式初始化一次。
        若后续重复调用，直接返回当前已初始化的单例。
        """
        if getattr(self, "_initialized", False):
            return self

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"日历文件未找到: {path}")

        # 1. 读盘解析
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pl.read_parquet(path)
        elif suffix == ".csv":
            df = pl.read_csv(path)
        else:
            df = pl.read_excel(path)

        # 2. 开市过滤
        if is_open_col and is_open_col in df.columns:
            df = df.filter(
                pl.col(is_open_col).cast(pl.Utf8).str.to_uppercase().is_in(["1", "TRUE", "Y", "交易"])
            )

        # 3. 标准化为 pl.Date
        clean_cal = (
            df.select(pl.col(date_col).alias("date"))
            .with_columns(
                pl.when(pl.col("date").cast(pl.Utf8).str.contains(r"^\d{8}$"))
                .then(pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d"))
                .otherwise(
                    pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").str.slice(0, 10).str.to_date("%Y-%m-%d")
                )
            )
            .filter(pl.col("date").is_not_null())
            .unique()
            .sort("date")
        )

        if start_date:
            clean_cal = clean_cal.filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        if end_date:
            clean_cal = clean_cal.filter(pl.col("date") <= pl.lit(end_date).str.to_date())

        # 缓存数据
        self._df: pl.DataFrame = clean_cal
        self._date_list: List[date] = clean_cal["date"].to_list()
        self._date_set: Set[date] = set(self._date_list)
        self._initialized = True
        return self

    @property
    def df(self) -> pl.DataFrame:
        self._check_ready()
        return self._df

    @property
    def date_list(self) -> List[date]:
        self._check_ready()
        return self._date_list

    def is_trading_day(self, target_date: date) -> bool:
        """O(1) 判定是否为开市交易日"""
        self._check_ready()
        return target_date in self._date_set

    def shift_trading_day(self, current_date: date, n: int = 1) -> Optional[date]:
        """T+N 交易日位移算子"""
        self._check_ready()
        try:
            idx = self._date_list.index(current_date)
            target_idx = idx + n
            if 0 <= target_idx < len(self._date_list):
                return self._date_list[target_idx]
            return None
        except ValueError:
            future = [d for d in self._date_list if d > current_date]
            if future and n > 0:
                return future[n - 1] if (n - 1) < len(future) else None
            return None

    def get_month_ends(self) -> pl.DataFrame:
        """获取月频调仓日"""
        self._check_ready()
        return (
            self._df.with_columns([
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month")
            ])
            .group_by(["year", "month"])
            .agg(pl.col("date").max().alias("date"))
            .sort("date")
            .select("date")
        )

    def _check_ready(self):
        if not getattr(self, "_initialized", False):
            raise RuntimeError("TradingCalendar 尚未初始化，请先调用 TradingCalendar().initialize(...)")

    """
    trading_calendar.py 中新增的区间选择与周频选择算子
    """
    def get_range(
            self,
            start_date: Optional[Union[str, date]] = None,
            end_date: Optional[Union[str, date]] = None,
            as_df: bool = True,
    ) -> Union[pl.DataFrame, List[date]]:
        """
        获取指定日期区间内的所有开市交易日 (闭区间 [start_date, end_date])
        利用二分查找 (bisect)，时间复杂度 O(log N)，极速响应切片。

        :param start_date: 起始日期 (None 表示从日历最早开始)
        :param end_date: 截止日期 (None 表示至日历最后结束)
        :param as_df: True 返回 Polars DataFrame，False 返回 List[date]
        """
        self._check_ready()

        # 统一转为 date 对象
        s_date = (
            _normalize_date(start_date)
            if isinstance(start_date, str)
            else start_date
        )
        e_date = (
            _normalize_date(end_date)
            if isinstance(end_date, str)
            else end_date
        )

        dates = self._date_list
        # 二分查找左右截断点
        left_idx = bisect.bisect_left(dates, s_date) if s_date else 0
        right_idx = bisect.bisect_right(dates, e_date) if e_date else len(dates)

        sliced_dates = dates[left_idx:right_idx]

        if not as_df:
            return sliced_dates

        return pl.DataFrame({"date": sliced_dates})

    def get_last_n_trading_days(
            self,
            anchor_date: Union[str, date],
            n: int,
            include_anchor: bool = True,
            as_df: bool = True,
    ) -> Union[pl.DataFrame, List[date]]:
        """
        截取某个锚点日期前（如最近 60 个交易日）的时间窗口（用于滚动估计协方差/计算动量）
        """
        self._check_ready()
        a_date = (
            _normalize_date(anchor_date)
            if isinstance(anchor_date, str)
            else anchor_date
        )
        dates = self._date_list
        # 寻找 <= anchor_date 的位置
        idx = bisect.bisect_right(dates, a_date)
        if not include_anchor and idx > 0 and dates[idx - 1] == a_date:
            idx -= 1

        start_idx = max(0, idx - n)
        window = dates[start_idx:idx]

        if not as_df:
            return window
        return pl.DataFrame({"date": window})

    # ================= 2. 周频日期选择器 (Weekly Selectors) =================

    def get_weekly_trading_days(
            self,
            mode: Literal["last", "first"] = "last",
            start_date: Optional[Union[str, date]] = None,
            end_date: Optional[Union[str, date]] = None,
    ) -> pl.DataFrame:
        """
        获取周度调仓基准日列表（严格对齐真实节假日）

        :param mode:
            - 'last': 每周最后一个有效交易日（最常用，通常为周五；若周五休市则前推至周四/周三）
            - 'first': 每周第一个有效交易日（通常为周一；若遇假期顺延至周二）
        :param start_date: 起始范围过滤
        :param end_date: 结束范围过滤
        :return: Schema [date: pl.Date] 的周度调仓日期表
        """
        self._check_ready()

        # 1. 取得区间范围内的完整日历
        cal_slice = self.get_range(start_date, end_date, as_df=True)

        # 2. 向量化计算 ISO 周年份和周号
        # 注意: dt.iso_year() 与 dt.week() 能够完美处理跨年周问题
        with_week_info = cal_slice.with_columns([
            pl.col("date").dt.iso_year().alias("iso_year"),
            pl.col("date").dt.week().alias("iso_week"),
        ])

        # 3. 按 (年, 周) 分组聚合选取极值
        agg_expr = (
            pl.col("date").max().alias("date")
            if mode == "last"
            else pl.col("date").min().alias("date")
        )

        weekly_df = (
            with_week_info.group_by(["iso_year", "iso_week"])
            .agg(agg_expr)
            .sort("date")
            .select("date")
        )

        return weekly_df