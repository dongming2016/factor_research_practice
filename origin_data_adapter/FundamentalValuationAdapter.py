from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping, Optional, Union, List, Iterable
import os
import pandas as pd
import polars as pl


from data_loaders.trading_calendar import TradingCalendar
from strategy.factors.valuation.config import PanelSchema

class FundamentalValuationAdapter:
    '''
    1. 首次运行历史 pandas 宽表转换为标准 PanelSchema 并罗盘为 Parquet
    2. 后续直接 则柔-copy 读取标准 parquet
    3. 支持幂等追加最新交易日截面
    '''

    DEFAULT_COL_MAPPING: dict[str, str] = {
        'pe_ratio_ttm': 'pe_ttm',
        'pb_ratio_ttm': 'pb',
        'ps_ratio_ttm': 'ps_ttm',
        'pcf_ratio_ttm': 'pcf_ttm',
        'dividend_yield_ttm': 'dividend_ttm',
    }

    KNOWN_METRIC_COLS: tuple[str, ...] = (
        "pe_ttm",
        "pb",
        "roe",
        "ps_ttm",
        "pcf_ttm",
        "div_yield_ttm",
    )

    @property
    def calendar(self) -> TradingCalendar:
        return TradingCalendar().initialize(file_path = '../datas/trading_calendar.parquet', date_col = 'date')

    def __init__(self,
                 schema: Optional[PanelSchema] = None,
                 col_mapping: Optional[Mapping[str, str]] = None,):
        self.schema = schema or PanelSchema()
        self.col_mapping = dict(self.DEFAULT_COL_MAPPING)
        if col_mapping:
            self.col_mapping.update(col_mapping)

    def _normalize_pandas(self,
                          path_: Union[str, Path])->pl.DataFrame:
        s = self.schema
        path_ = Path(path_)
        if path_.suffix in {'.pkl', '.plk', '.pickle'}:
            data_ = pd.read_pickle(path_)
        else:
            raise TypeError(f'path {path_} is not a valid path, file must be .pkl or .pickle')

        # if isinstance(data_.index, pd.MultiIndex):
        data_.reset_index(inplace=True)
        columns = data_.columns
        data_.rename(columns={columns[0]: s.date_col, columns[1]: s.sec_id_col}, inplace=True)
        df_pl = pl.from_pandas(data_)

        return self._apply_column_mapping(df_pl)

    def _apply_column_mapping(self, df: pl.DataFrame) -> pl.DataFrame:
        s = self.schema
        renames = {k : v for k, v in self.col_mapping.items() if k in df.columns and k != v}
        if renames:
            df = df.rename(renames)

        cast_exprs = []
        if s.sec_id_col in df.columns:
            cast_exprs.append(pl.col(s.sec_id_col).cast(pl.String))
        if s.date_col in df.columns:
            cast_exprs.append(pl.col(s.date_col).cast(pl.Date))

        return df.with_columns(cast_exprs) if cast_exprs else df

    def enforce_contract(self, df: pl.DataFrame) -> pl.DataFrame:
        '''强制收口为符合 PanelSchema 契约的干净面板（类型对齐、NaN清晰、去重、严格排序）'''
        s = self.schema
        metric_col = [c for c in self.KNOWN_METRIC_COLS if c in df.columns]

        return (
            df
            .unique(subset=[s.sec_id_col, s.date_col], keep='last')
            .sort([s.sec_id_col, s.date_col])
        )

    # =========================================================================
    # 阶段 1: 合成基础估值 + 公告日 PIT 对齐财报指标
    # =========================================================================
    def build_base_panel(
            self,
            valuation_source: Union[str, Path]
    ) -> pl.DataFrame:
        '''
        将日频估值宽表与带真实披露日期的财报数据合成 PIT 基准面板：
        - 严格按 notice_date <= date 执行 backward join_asof, 杜绝未来函数
        :param valuation_source:
        :return:
        '''
        s = self.schema
        base_df = self._normalize_pandas(valuation_source)
        trading_dates = self.calendar.df
        base_df = base_df.join(trading_dates, on=[s.date_col], how='inner')

        base_df = base_df.unique(subset=[s.sec_id_col, s.date_col], keep='last').sort([s.sec_id_col, s.date_col])

        return self.enforce_contract(base_df)

    # 阶段 2: 添加财报指标
    @staticmethod
    def build_fin_df(
            # self,
            fin_source: Union[str, Path],
            disclosure_source: Union[str, Path]
    ) -> pl.DataFrame:
        fin_panel = pd.read_pickle(fin_source)
        disclosure_panel = pd.read_pickle(disclosure_source)['panel']
        fin_panel_2 = fin_panel.stack().dropna().reset_index().rename(columns={'level_1': 'sec_id', 0: 'roe', 'date': 'notice_date'})
        disclosure_panel_2 = disclosure_panel.stack().dropna().reset_index().rename(columns={'wind': 'sec_id', 'level_1': 'report_period', 0: 'notice_date'})
        fin_panel_2.notice_date = pd.to_datetime(fin_panel_2.notice_date)
        disclosure_panel_2.notice_date = pd.to_datetime(disclosure_panel_2.notice_date)
        merged_fin_panel = fin_panel_2.merge(disclosure_panel_2, on=['sec_id', 'notice_date'], how='inner')

        return pl.from_pandas(merged_fin_panel)

    @staticmethod
    def save(
            panel_df: pl.DataFrame,
            output_path: Union[str, Path],
            compression: str = 'zstd') -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out.with_suffix(f'{out.suffix}.tmp')
        panel_df.write_parquet(tmp_path, compression=compression, statistics=True)
        os.replace(tmp_path, out)
        return out

    def save_value_df(
            self,
            panel_df: pl.DataFrame,
            output_path: Union[str, Path],
            compression: str = 'zstd'
    ) -> Path:
        '''原子写入标准 Parquet 基准仓（先写 .tmp 再 os.replace 防止损坏'''
        clean_df = self.enforce_contract(panel_df)
        return self.__class__.save(panel_df=clean_df, output_path=output_path, compression=compression)

if __name__ == '__main__':
    adapter = FundamentalValuationAdapter()
    STORE_PATH = '../datas/valuation_base.parquet'
    base_df = adapter.build_base_panel(
        valuation_source= '../origin_datas/value_valuation_panel.plk',
    )
    adapter.save_value_df(base_df, STORE_PATH)
    fin_pit_df = adapter.build_fin_df('../origin_datas/roe_ttm.plk', '../origin_datas/_ckpt_ashare_regular_report_actual_dd.pkl')
    FIN_STORE_PATH = '../datas/financial_base.parquet'

    adapter.save(fin_pit_df, FIN_STORE_PATH)