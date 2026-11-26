"""
loaders.py
通过继承 BaseDataLoader 与相关 Mixins 组合出具体 Loader
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List
import polars as pl
from .mixins import GridAlignmentMixin, AsOfAlignmentMixin, DateFilterMixin


class BaseDataLoader(ABC):
    """纯粹的抽象基类，只规范数据加载契约"""
    def __init__(self, data_dir: Union[str, Path] = "./datas"):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self, file_name: str, **kwargs) -> pl.DataFrame:
        """主入口：加载原始数据并调用对应 Mixin 完成对齐标准化"""
        pass


class PriceDataLoader(BaseDataLoader, GridAlignmentMixin):
    """
    股票/ETF行情加载器：
    拥有 BaseDataLoader 的加载能力 + GridAlignmentMixin 的截面对齐能力
    """
    def load(
        self,
        file_name: str = "daily_prices.parquet",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        file_path = self.data_dir / file_name
        q = pl.scan_parquet(file_path).with_columns(
            pl.col("date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d")
        )
        if start_date:
            q = q.filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        if end_date:
            q = q.filter(pl.col("date") <= pl.lit(end_date).str.to_date())

        raw_df = q.collect()

        # 直接调用从 GridAlignmentMixin 混入的方法
        return self.align_by_grid(raw_df, ffill_cols=["close"])


class MacroDataLoader(BaseDataLoader, AsOfAlignmentMixin):
    """
    宏观数据加载器：
    拥有 BaseDataLoader 的加载能力 + AsOfAlignmentMixin 的点时对齐能力
    """
    def load(
        self,
        file_name: str = "macro.parquet",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        file_path = self.data_dir / file_name
        q = pl.scan_parquet(file_path).with_columns(
            pl.col("notice_date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d")
        )
        if start_date:
            q = q.filter(pl.col("notice_date") >= pl.lit(start_date).str.to_date())
        if end_date:
            q = q.filter(pl.col("notice_date") <= pl.lit(end_date).str.to_date())

        raw_df = q.collect()

        # 直接调用从 AsOfAlignmentMixin 混入的方法
        return self.align_asof(raw_df, notice_date_col="notice_date")


class NAVDataLoader(BaseDataLoader, DateFilterMixin):
    """
    基金净值加载器：
    混入 DateFilterMixin 进行非开市日剔除
    """
    def load(self, file_name: str = "fund_nav.parquet") -> pl.DataFrame:
        raw_df = pl.read_parquet(self.data_dir / file_name)
        # 调用 DateFilterMixin 混入的方法
        filtered_df = self.filter_trading_days(raw_df)
        return filtered_df.with_columns(
            (pl.col("nav") / pl.col("nav").shift(1) - 1.0)
            .over("fund_id")
            .alias("ret")
        )
