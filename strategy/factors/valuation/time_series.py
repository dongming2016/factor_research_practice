import polars as pl

from strategy.factors.registry import FactorRegistry
from base import BaseValuationGenerator
from config import MetricCatalog, ValuationBusConfig
from utils import build_valuation_feature_bus


@FactorRegistry.register('valuation.time_series')
class TimeSeriesValuation(BaseValuationGenerator):
    factor_family = 'valuation'

    def __init__(
            self,
            catalog: MetricCatalog = MetricCatalog(),
            bus_config: ValuationBusConfig = ValuationBusConfig(),
    ):
        self.catalog = catalog
        self.bus_config = bus_config

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        '''
        执行时序特征计算
        :param df:
        :return:
        '''
        # 1. 接入统一特征总线
        bus_df = build_valuation_feature_bus(df = df, catalog = self.catalog, config = self.bus_config)

        # 2. 提取目标时序
        score_cols = [
            f'{k}_ts_alpha_zscore'
            for k in self.catalog.keys
            if k in bus_df.columns
        ]

        # 3. 严格收敛瘦表
        schema = self.bus_config.schema

        return bus_df.select([schema.sec_id_col, schema.date_col] + score_cols)

if __name__ == '__main__':
    from specs import METRIC_REGISTRY
    tsv = TimeSeriesValuation(catalog = MetricCatalog().with_custom(['pe', 'ps', 'pcf', 'pb']))
    df = pl.read_parquet('../../../datas/valuation_base.parquet')
    df = df.rename({'pe_ttm': 'pe', 'ps_ttm': 'ps', 'pcf_ttm': 'pcf', 'pb_ttm': 'pb'})
    result = tsv.generate(df)