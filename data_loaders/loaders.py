"""
loaders.py
通过继承 BaseDataLoader 与相关 Mixins 组合出具体 Loader
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List
import polars as pl
from mixins import GridAlignmentMixin, AsOfAlignmentMixin, DateFilterMixin


class BaseDataLoader(ABC):
    """纯粹的抽象基类，只规范数据加载契约"""
    def __init__(self, data_dir: Union[str, Path] = "../datas"):
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

class ValuationDataLoader(BaseDataLoader, DateFilterMixin):
    '''
    估值特征总线组装加载器
    '''
    METRIC_COLS: tuple[str, ...] = (
        "pe_ttm",
        "pb",
        "roe",
        "ps_ttm",
        "pcf_ttm",
        "div_yield_ttm",
    )

    def _scan_filtered(self, file_name: str, s_dt, e_dt) -> pl.LazyFrame:
        lf = pl.scan_parquet(self.data_dir / file_name)
        if s_dt:
            lf = lf.filter(pl.col("date") >= s_dt)
        if e_dt:
            lf = lf.filter(pl.col("date") <= e_dt)
        return lf

    def load(
            self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            base_file: str = "valuation_base.parquet",
            industry_file: str = "industry_daily.parquet",
            suspend_file: Optional[str] = None,
            restruct_file: Optional[str] = None,
            mask_suspend: bool = True,
    ):
        s_dt = pl.lit(start_date).str.to_date() if start_date else None
        e_dt = pl.lit(end_date).str.to_date() if end_date else None

        q = self._scan_filtered(base_file, s_dt, e_dt)

        if industry_file and (self.data_dir / industry_file).exists():
            ind_lf = self._scan_filtered(industry_file, s_dt, e_dt)
            q = q.join(ind_lf, on = ['sec_id', 'date'], how = 'left')

        if suspend_file and (self.data_dir / suspend_file).exists():
            susp_lf = self._scan_filtered(suspend_file, s_dt, e_dt)
            q = q.join(susp_lf, on = ['sec_id', 'date'], how = 'left')

        if restruct_file and (self.data_dir / restruct_file).exists():
            rest_lf = self._scan_filtered(restruct_file, s_dt, e_dt)
            q = q.join(rest_lf, on = ['sec_id', 'date'], how = 'left')

        df = q.collect()

        if mask_suspend and 'is_suspend' in df.columns:
            val_cols = [c for c in self.METRIC_COLS if c in df.columns]
            if val_cols:
                df = df.with_columns(
                    pl.when(pl.col('is_suspend').fill_null(0) == 1)
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in val_cols
                )

        post_exprs: list[pl.Expr] = [
            (
                pl.col('industry').cast(pl.String).fill_null('UNKNOWN')
                if 'industry' in df.columns
                else pl.lit('UNKNOWN', dtype=pl.String).alias('industry')
            ),
            (
                pl.col('is_restructured').fill_null(0).cast(pl.Int8)
                if 'is_restructured' in df.columns
                else pl.lit(0, dtype=pl.Int8).alias('is_restructured')
            )
        ]
        df = df.with_columns(post_exprs)

        return self.filter_trading_days(df).sort(['sec_id', 'date'])


if __name__ == '__main__':
    val_data_loader = ValuationDataLoader()
    val_datas = val_data_loader.load()

