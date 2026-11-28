from datetime import date

import polars as pl

from strategy.factors.registry import FactorRegistry
from base import BaseValuationGenerator
from strategy.factors.valuation.config import MetricCatalog, ValuationBusConfig
from strategy.factors.valuation.utils import build_valuation_feature_bus
from typing import Optional


@FactorRegistry.register('valuation.time_series')
class TimeSeriesValuation(BaseValuationGenerator):
    factor_family = 'valuation'

    def __init__(
            self,
            catalog: MetricCatalog = MetricCatalog(),
            bus_config: ValuationBusConfig = ValuationBusConfig(),
            start_date: Optional[date] = None,
    ):
        self.catalog = catalog
        self.bus_config = bus_config
        self.start_date = start_date

    def generate(self, df: pl.DataFrame, keep_tmp = False) -> pl.DataFrame:
        '''
        执行时序特征计算
        :param df:
        :return:
        '''
        # 1. 接入统一特征总线
        if self.start_date:
            df = df.filter(pl.col('date') >= self.start_date)
        bus_df = build_valuation_feature_bus(df = df, catalog = self.catalog, config = self.bus_config)

        # 2. 提取目标时序
        score_cols = [
            f'{k}_ts_alpha_zscore'
            for k in self.catalog.keys
            if k in bus_df.columns
        ]

        # 3. 严格收敛瘦表
        schema = self.bus_config.schema

        if not keep_tmp:
            return bus_df.select([schema.sec_id_col, schema.date_col] + score_cols)
        return bus_df

if __name__ == '__main__':
    # from specs import METRIC_REGISTRY
    from data_loaders.loaders import ValuationDataLoader
    tsv = TimeSeriesValuation(catalog = MetricCatalog().with_custom(['pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb']), start_date = date(2010, 1, 1))
    # df = pl.read_parquet('../../../datas/valuation_base.parquet')
    # df = df.rename({'pe_ttm': 'pe', 'ps_ttm': 'ps', 'pcf_ttm': 'pcf', 'pb_ttm': 'pb'})
    df = ValuationDataLoader('../../../datas').load()
    result = tsv.generate(df, keep_tmp=True)