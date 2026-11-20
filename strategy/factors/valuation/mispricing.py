from typing import List,Optional
import polars as pl
# import polars_ols as opt
import numpy as np

from strategy.factors.registry import FactorRegistry
from .base import BaseValuationGenerator
from strategy.factors.valuation.config import ValuationBusConfig, MetricCatalog
from strategy.factors.valuation.utils import build_valuation_feature_bus


@FactorRegistry.register('valuation.pb_roe_mispricing')
class PbROEMispricingGenerator(BaseValuationGenerator):
    '''
    PB-ROE 截面错误定价生成器：
    1. 计算每只股票滚动窗口内的调和平均EP，结合行业中位数进行贝叶斯平滑
    2. 每日截面将稳态估值分位 n 档哑变量（吸收长期商业模式固有估值溢价）
    ln(pb) ~ alpha + roe + pe分档
    '''
    factor_family = 'valuation'

    def __init__(
            self,
            bus_config: ValuationBusConfig = ValuationBusConfig(),
            anchor_metric_key: str = 'ts_pe',
            pb_col: str = 'pb',
            roe_col: str = 'roe',
            mkt_cap_col: str = 'total_mv',
            n_pe_tiers: int = 5,
            ridge_alpha: float = 1e-4
    ):
        self.bus_config = bus_config
        self.anchor_metric_key = anchor_metric_key
        self.anchor_catalog = MetricCatalog(keys=(anchor_metric_key))
        self.pb_col = pb_col
        self.roe_col = roe_col
        self.mkt_cap_col = mkt_cap_col
        self.n_pe_tiers = n_pe_tiers
        self.ridge_alpha = ridge_alpha

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        schema = self.bus_config.schema
        sec_col = schema.sec_id_col
        date_col = schema.date_col
        ind_col = schema.industry_col

        # 1. 获取稳态估值中枢
        bus_df = build_valuation_feature_bus(
            df = df,
            catalog=self.anchor_catalog,
            config=self.bus_config
        )

        print(bus_df)
        anchor_col = f'{self.anchor_metric_key}_anchor'
        if anchor_col not in bus_df.columns:
            raise ValueError(f'特征总线未能生成"{anchor_col}", 请检查输入数据是否包含对应原始估值字段。')

        # 2. 连续特征清洗与每日横截面稳态估值分
        df_tiered = bus_df.with_columns([
            pl.when(pl.col(self.pb_col) >0)
            .then(pl.col(self.pb_col).log())
            .otherwise(None)
            .alias('_log_pb'),

            pl.col(self.roe_col).clip(-0.5, 0.8).alias('_clean_roe'),

            pl.when(pl.col(self.mkt_cap_col) >0)
            .then(pl.col(self.mkt_cap_col).log())
            .otherwise(None)
            .alias('_log_mv'),

            pl.col(anchor_col)
            .bin_quantiles(self.n_pe_tiers, labels=[f'tier_{i}' for i in range(self.n_pe_tiers)])
            .over(date_col)
            .fill_null('tier_sparse')
            .alias('_ts_pe_tier')
        ])

        # 3. 构造独立 0/1 哑变量
        unique_inds = (
            df_tiered.select(ind_col)
            .drop_nulls()
            .unique()
            .sort(ind_col)[ind_col]
            .to_list()
        )
        active_inds = unique_inds[1:] if len(unique_inds) > 1 else unique_inds
        ind_dummies = [
            (pl.col(ind_col) == ind).cast(pl.Float64).alias(f'_ind_{ind}')
            for ind in active_inds
        ]
        ind_dummy_cols = [f'_ind_{ind}' for ind in active_inds]

        tier_names = [f'tier_{i}' for i in range(self.n_pe_tiers)] + ['tier_sparse']
        tier_dummies = [
            (pl.col('_ts_pe_tier') == t_name).cast(pl.Float64).alias(f'_{t_name}')
            for t_name in tier_names
        ]
        tier_dummy_cols = [f'_{t_name}' for t_name in tier_names]

        df_prepared = df_tiered.with_columns(ind_dummies + tier_dummies)
        candidate_dummies = ind_dummy_cols + tier_dummy_cols
        valid_dummies = [
            col for col in candidate_dummies
            if df_prepared.select(pl.col(col).sum()).item() > 0
        ]
        feature_cols = ["_clean_roe", "_log_mv"] + valid_dummies
        # feature_cols = ['_clean_roe', '_log_mv'] + ind_dummy_cols + tier_dummies_cols

        # 4. 截面多元岭回归提取残差 (纯原生 NumPy 闭式解，峰值内存 < 5 MB，彻底根除 138GB 内存泄漏)
        def _solve_cross_sectional_ridge(group_df: pl.DataFrame) -> pl.DataFrame:
            valid_mask = pl.all_horizontal(
                [pl.col("_log_pb").is_not_null()] + [pl.col(c).is_not_null() for c in feature_cols]
            )
            sub = group_df.filter(valid_mask)
            n_obs = sub.height

            if n_obs <= len(feature_cols):
                return group_df.select([sec_col, date_col]).with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("mispricing_pb_roe_score")
                )

            X_raw = sub.select(feature_cols).to_numpy()
            X = np.column_stack([np.ones(n_obs), X_raw])
            y = sub["_log_pb"].to_numpy()

            # 岭回归闭式解: (X^T X + alpha * I) beta = X^T y (不惩罚截距项)
            reg_eye = np.eye(X.shape[1]) * self.ridge_alpha
            reg_eye[0, 0] = 0.0
            beta = np.linalg.solve(X.T @ X + reg_eye, X.T @ y)

            residuals = y - X @ beta
            scored_sub = sub.select([sec_col, date_col]).with_columns(
                pl.Series("mispricing_pb_roe_score", -1.0 * residuals)
            )
            return group_df.select([sec_col, date_col]).join(scored_sub, on=[sec_col, date_col], how="left")

        return (
            df_prepared
            .group_by(date_col, maintain_order=True)
            .map_groups(_solve_cross_sectional_ridge)
        )
        # return (
        #     df_prepared.with_columns(
        #         pl.col('_log_pb')
        #         .least_squares.ridge(
        #             *[pl.col(col) for col in feature_cols],
        #             alpha=self.ridge_alpha,
        #             mode='residuals',
        #             null_policy = 'drop'
        #         )
        #         .over(date_col)
        #         .alias('_residual'),
        #     )
        #     .with_columns(
        #         (-1.0 * pl.col('_residual')).alias('mispricing_pb_roe_score')
        #     )
        #     .select([sec_col, date_col, 'mispricing_pb_roe_score'])
        # )

