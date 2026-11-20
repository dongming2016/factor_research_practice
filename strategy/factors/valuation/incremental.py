'''
每日增量估值特征总线状态及
'''

import math
from typing import Dict, List, Optional, Tuple

import polars as pl
from pathlib import Path
from polars.series.utils import call_expr
from rqdatac.services.basic import industry

from .config import MetricCatalog, ValuationBusConfig
from .utils import MetricSpec, _compute_batch_internal


class IncrementalValuationBus:
    '''
    每日增量更新状态机
    1. 直接消费 utils.py 中的 METRIC_SPECS 声明式契约， 确保全量与增量单一真源；
    2. 维护个股与行业两层 EMA 状态快找 （一阶矩 mu 与 二阶矩 s2），支持断点重置与贝叶斯收缩
    3. 严格保留 burn_in 隔离期与非法值的原生 Null 状态
    '''
    def __init__(self
                 , catalog: MetricCatalog = MetricCatalog()
                 , config: ValuationBusConfig = ValuationBusConfig()):
        self.catalog = catalog
        self.config = config
        self.schema = config.schema
        self.alpha = 1.0 - math.pow(2.0, -1.0 / config.half_life_days)

    def create_empty_states(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        '''
        构造严格对齐的空状态快照 用于历史第 1 个交易日的冷启动起点
        :return:
        '''
        stk_schema = {
            self.schema.sec_id_col: pl.Utf8,
            '_prev_industry': pl.Utf8,
            '_prev_age': pl.Int64
        }

        ind_schema = {
            self.schema.industry_col: pl.Utf8,
        }

        for prefix in self.catalog.keys:
            stk_schema[f'_prev_stk_n_{prefix}'] = pl.Int64
            stk_schema[f'_prev_stk_mu_{prefix}'] = pl.Float64
            stk_schema[f'_prev_stk_s2_{prefix}'] = pl.Float64

            ind_schema[f'_prev_ind_mu{prefix}'] = pl.Float64
            ind_schema[f'_prev_ind_s2_{prefix}'] = pl.Float64

        return pl.DataFrame(stk_schema), pl.DataFrame(ind_schema)

    def step_daily(
            self,
            today_df: pl.DataFrame,
            stock_state_df: pl.DataFrame,
            ind_state_df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        '''
        单日增量推进
        1. 先计算断点
        2. 生成个股维度的增量更新表达式
        3. 生成行业维度的增量更新表达式

        :param today_df:
        :param stock_state_df:
        :param ind_state_df:
        :return:
        '''
        alpha_ = self.alpha
        sec_id_col = self.schema.sec_id_col
        industry_col = self.schema.industry_col
        re_col = self.schema.restructure_col
        burn_in = self.config.burn_in_days
        half_life_days = self.config.half_life_days
        min_periods = self.config.min_periods

        # 解析当前 DataFrame 中实际存在的生效指标
        active_specs: Dict[str, MetricSpec] = self.catalog.resolve_active_specs(today_df.columns)
        if not active_specs:
            return today_df, stock_state_df, ind_state_df

        today_clean = today_df.unique(subset=[sec_id_col], keep='last')
        stk_state_clean = stock_state_df.unique(subset=[sec_id_col], keep='last')
        ind_state_clean = ind_state_df.filter(pl.col(industry_col).is_not_null()).unique(subset=[industry_col], keep='last')
        # ===============================================
        # 1. 关联昨日个股状态，判定是否出发行业漂移或重大重组断点
        # ===============================================
        merged_stk = today_clean.join(stock_state_df, on=sec_id_col, how='inner')
        assert merged_stk.height == today_clean.height, (
            f'个股 join 发生膨胀： {today_clean.height} -> {merged_stk.height} '
        )

        is_break_expr = (
        (pl.col(industry_col) != pl.col('_prev_industry').fill_null(True)) # fill_null非常有必要，表明是新股票
            | (pl.col(re_col).fill_null(0).cast(pl.Boolean)
               if re_col in today_df.columns
               else pl.lit(False))
        )

        step_df = (merged_stk.with_columns([
            is_break_expr.alias('_is_break'),
            pl.when(is_break_expr)
            .then(0) # 如果变了重新记时长
            .otherwise(pl.col('_prev_age').fill_null(-1) + 1)
            .alias('_segment_age')
        ])
        .with_columns([
            ((pl.col('_segment_age') - burn_in) / (half_life_days * 1.5))
            .clip(0.0, 1.0)
            .alias('_bayes_weight')
        ]))

        # =============================================================
        # 2. 依据 MetricSpec 映射当日值，并更新个股 EMA 状态（mu 与 s2）
        # =============================================================
        val_exprs, ind_med_exprs, stk_update_exprs = [], [], []

        for prefix, spec in active_specs.items():
            c_val = f'_val_{prefix}'
            c_ind_med = f'_ind_med_{prefix}'
            prev_mu = f'_prev_stk_mu_{prefix}'
            prev_s2 = f'_prev_stk_s2_{prefix}'
            prev_n = f'_prev_stk_n_{prefix}'

            # 复用 MetricSpec 唯一真源
            val_exprs.append(spec.build_transorm_expr('_segment_age', burn_in).alias(c_val))
            ind_med_exprs.append(pl.col(c_val).median().over(industry_col).alias(c_ind_med))

            # 发生断点时清空历史
            base_mu = pl.when(pl.col('_is_break')).then(None).otherwise(pl.col(prev_mu))
            base_s2 = pl.when(pl.col('_is_break')).then(None).otherwise(pl.col(prev_s2))
            base_n = pl.when(pl.col('_is_break')).then(0).otherwise(pl.col(prev_n).fill_null(0))

            x = pl.col(c_val)
            # 当日有效则按指数衰减递推；停牌或隔离期保持昨日状态不变
            new_n = pl.when(x.is_not_null()).then(base_n + 1).otherwise(base_n)
            new_mu = (
                pl.when(x.is_null()).then(base_mu)
                .when(base_mu.is_null()).then(x)
                .otherwise((1 - alpha_) * base_mu + alpha_ * x)
            )
            new_s2 = (
                pl.when(x.is_null()).then(base_s2)
                .when(base_s2.is_null()).then(x * x)
                .otherwise((1.0 - alpha_) * base_s2 + alpha_ * (x * x))
            )

            stk_update_exprs.extend([
                new_n.alias(f'_new_stk_n_{prefix}'),
                new_mu.alias(f'_new_stk_mu_{prefix}'),
                new_s2.alias(f'_new_stk_s2_{prefix}'),
            ])

        step_df = (step_df.with_columns(val_exprs)
                   .with_columns(ind_med_exprs)
                   .with_columns(stk_update_exprs))

        # =============================================================
        # 3. 更新行业先验 EMA 状态
        # =============================================================
        today_ind_slice = (
            step_df.select([industry_col] + [f'_ind_med_{p}' for p in active_specs])
            .filter(pl.col(industry_col).is_not_null())
            .unique(subset=[industry_col])
        )

        merged_ind = today_ind_slice.join(ind_state_clean, on=industry_col, how='full', coalesce=True)

        ind_update_exprs = []
        for prefix in active_specs:
            med_col = f'_ind_med_{prefix}'
            p_mu = f'_prev_ind_mu_{prefix}'
            p_s2 = f'_prev_ind_s2_{prefix}'

            x_ind = pl.col(med_col)
            new_ind_mu = (
                pl.when(x_ind.is_null()).then(pl.col(p_mu))
                .when(pl.col(p_mu).is_null()).then(x_ind)
                .otherwise((1.0 - alpha_) * pl.col(p_mu) + alpha_ * x_ind)
            )
            new_ind_s2 = (
                pl.when(x_ind.is_null()).then(pl.col(p_s2))
                .when(pl.col(p_s2).is_null()).then(x_ind * x_ind)
                .otherwise((1.0 - alpha_) * pl.col(p_s2) + alpha_ * (x_ind * x_ind))
            )
            ind_update_exprs.extend([new_ind_mu.alias(f'_new_ind_mu_{prefix}'),
                                     new_ind_s2.alias(f'_new_ind_s2_{prefix}'), ])

        updated_ind_df = merged_ind.with_columns(ind_update_exprs)

        # =============================================================
        # 4. 经验贝叶斯合成并输出今日总线
        # =============================================================
        final_df = step_df.join(updated_ind_df, on=industry_col, how='left')
        assert final_df.height == step_df.height, (
            f"行业状态 join 发生行膨胀: {step_df.height} -> {final_df.height}"
        )

        out_exprs = []

        for prefix, spec in active_specs.items():
            valid_stk_mu = (
                pl.when(pl.col(f'_new_stk_n_{prefix}') >= min_periods)
                .then(pl.col(f'_new_stk_mu_{prefix}'))
                .otherwise(None)
            )

            valid_stk_s2 = (
                pl.when(pl.col(f'_new_stk_n_{prefix}') >= min_periods)
                .then(pl.col(f'_new_stk_s2_{prefix}'))
                .otherwise(None)
            )

            stk_std = (valid_stk_s2 - valid_stk_mu.pow(2)).clip(0.0, None).sqrt()

            ind_mu = pl.col(f'_new_ind_mu_{prefix}')
            ind_s2 = pl.col(f'_new_ind_s2_{prefix}')
            ind_std = (ind_s2 - ind_mu.pow(2)).clip(0.0, None).sqrt() # clip的目的是防抖动

            w = pl.col('_bayes_weight')
            final_mu = w * valid_stk_mu.fill_null(ind_mu) + (1.0 - w) * ind_mu
            final_std = w * stk_std.fill_null(ind_std) + (1.0 - w) * ind_std
            direction_sign = 1 if spec.higher_is_alpha else -1

            out_exprs.extend([
                spec.restore_anchor_expr(final_mu).alias(f'{prefix}_anchor'),
                (direction_sign * (pl.col(f'_val_{prefix}') - final_mu) / (final_std + 1e-4))
                .alias(f'{prefix}_ts_alpha_zscore')
            ])

        today_bus_df = final_df.with_columns(out_exprs)

        # =============================================================
        # 5. 打包导出新的状态快照， 并清理今日总线表的内部临时状态列
        # =============================================================
        next_stk_cols = [
            pl.col(sec_id_col),
            pl.col(industry_col).alias('_prev_industry'),
            pl.col('_segment_age').alias('_prev_age'),
        ]

        next_ind_cols = [pl.col(industry_col)]

        for prefix in active_specs:
            next_stk_cols.extend([
                pl.col(f'_new_stk_n_{prefix}').alias(f'_prev_stk_n_{prefix}'),
                pl.col(f'_new_stk_mu_{prefix}').alias(f'_prev_stk_mu_{prefix}'),
                pl.col(f'_new_stk_s2_{prefix}').alias(f'_prev_stk_s2_{prefix}'),
            ])
            next_ind_cols.extend([
                pl.col(f'_new_ind_mu_{prefix}').alias(f'_prev_ind_mu_{prefix}'),
                pl.col(f'_new_ind_s2_{prefix}').alias(f'_prev_ind_s2_{prefix}'),
            ])

        today_updated_stk = today_bus_df.select(next_stk_cols)

        # 保留当日停牌快照，防止复牌后记忆丢失
        frozen_suspend_stk = stk_state_clean.join(today_updated_stk, on = sec_id_col, how='anti').select(
            today_updated_stk.columns
        )

        next_stock_state = pl.concat([today_updated_stk, frozen_suspend_stk], how='vertical')
        next_ind_state = updated_ind_df.select(next_ind_cols)

        # 清理内部临时列，保持输出总线表的Schema 与全量版完全一致
        temp_prefixes = ('_prev_', '_new_', '_val_', '_ind_med_', '_is_break', '_segment_age', '_bayes_weight')
        clean_bus_cols = [c for c in today_bus_df.columns if not c.startswith(temp_prefixes)]
        # print(clean_bus_cols, today_bus_df.columns)
        return today_bus_df.select(clean_bus_cols), next_stock_state, next_ind_state

    def initialize_state_snapshots(
            self,
            history_df: pl.DataFrame,
            cutoff_date: Optional[object] = None,
            stock_state_path: str = 'state/val_stock_state.parquet',
            ind_state_path: str = 'state/val_ind_state.parquet',
            need_history_bus: bool = True,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame]]:
        '''
        利用历史全量面板一次性跑完状态递推，并导出至 cutoff_date 的状态快照
        :param history_df:
        :param cutoff_date:
        :param stock_state_path:
        :param ind_state_path:
        :param need_history_bus:
        :return:
        '''
        date_col = self.schema.date_col
        sec_col = self.schema.sec_id_col
        ind_col = self.schema.industry_col

        df_filtered = (
            history_df.filter(pl.col(date_col)  <= cutoff_date)
            if cutoff_date is not None
            else history_df
        )

        sorted_df = df_filtered.sort([date_col, sec_col])
        daily_partitions = sorted_df.partition_by(date_col, maintain_order=True)

        if not daily_partitions:
            raise ValueError('历史面板数据为空，无法初始化状态快照。')


        # 1. 调用向量化批处理引擎
        batch_df, active_specs = _compute_batch_internal(
            df = df_filtered,
            catalog = self.catalog,
            config = self.config,
            keep_state_cols = True
        )

        # 2. 抽取个股截止最后一天的最新状态快照
        stk_agg_exprs = [
            pl.col(ind_col).last().alias(f'_prev_industry'),
            pl.col('_tmp_seg_age').last().alias(f'_prev_age'),
        ]
        for prefix in active_specs:
            c_n = f'_tmp_stk_n_{prefix}'
            c_mu = f'_tmp_stk_mu_{prefix}'
            # c_std = f'_tmp_stk_std_{prefix}'

            # 由 std = squrt(s2 - mu^2)
            stk_agg_exprs.extend([
                pl.col(c_n).last().cast(pl.Int64).alias(f'_prev_stk_n_{prefix}'),
                pl.col(c_mu).last().cast(pl.Float64).alias(f'_prev_stk_mu_{prefix}'),
                pl.col(f'_tmp_stk_s2_{prefix}').last().cast(pl.Float64).alias(f'_prev_stk_s2_{prefix}'),
            ])

        stk_state_df =  (batch_df.sort([sec_col, date_col])
                         .group_by(sec_col, maintain_order=True)
                         .agg(stk_agg_exprs))

        # 3. 抽取行业截至最后一天的最新先验状态快照
        ind_daily_cols = [ind_col, date_col]
        for prefix in active_specs:
            ind_daily_cols.extend([f'_tmp_ind_mu_{prefix}', f'_tmp_ind_s2_{prefix}'])

        ind_agg_exprs = []
        for prefix in active_specs:
            c_imu = f'_tmp_ind_mu_{prefix}'
            # c_istd = f'_tmp_ind_std_{prefix}'
            ind_agg_exprs.extend([
                pl.col(c_imu).last().alias(f'_prev_ind_mu_{prefix}'),
                pl.col(f'_tmp_ind_s2_{prefix}').last().alias(f'_prev_ind_s2_{prefix}'),
                # (pl.col(c_istd).last().pow(2) + pl.col(c_imu).last().pow(2)).alias(f'_prev_ind_s2_{prefix}')
            ])

        # print(batch_df.columns, ind_col)
        ind_state_df = (
            batch_df.select(ind_daily_cols)
            .filter(pl.col(ind_col).is_not_null())
            .unique(subset=[ind_col, date_col], keep='last')
            .sort([ind_col, date_col])
            .group_by(ind_col, maintain_order=True)
            .agg(ind_agg_exprs)
        )

        # 4. 原子落盘快照文件

        Path(stock_state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ind_state_path).parent.mkdir(parents=True, exist_ok=True)

        stk_state_df.write_parquet(stock_state_path)
        ind_state_df.write_parquet(ind_state_path)
        full_bus_df = None
        if need_history_bus:
            tmp_cols = [c for c in batch_df.columns if c.startswith('_tmp_')]
            full_bus_df = batch_df.drop(tmp_cols)

        return stk_state_df, ind_state_df, full_bus_df