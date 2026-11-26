'''
估值因子家族共享特征总线： 一次计算全量稳态中枢与时序偏离，供全家族零冗余消费
'''
import polars as pl
from typing import Dict, Optional, List

from config import MetricCatalog, ValuationBusConfig
from specs import MetricSpec

# 定义跨
# DEFAULT_VALUATION_METRICS: Dict[str, str] = {
#     'ts_pe': 'pe_ttm',
#     'ts_pb': 'pb_ttm',
#     'ts_ps': 'ps_ttm',
#     'ts_pcf': 'pc_ttm'
# }

BUS_READY_FLAG = '_val_bus_computed'

def prepare_segment_context(
        df: pl.DataFrame,
        industry_col: str = 'industry',
        sec_id_col: str = 'sec_id',
        date_col: str = 'date',
        restructure_col: str = 'is_restructured',
        half_life_days: int = 250,
        burn_in_days: int = 60
):
    '''
    1. 行业变化或发生重大事件，如重组等事项时，视为不同的序列；
    2. 上市前burn_in_days日股票的因子不具参考性，使用行业中位数的相应因子填充，上市后burn_in_days+half_life_days*1.5日后全部用股票的因子，中间部分股票因子的权重为(age-burn_in_days)/(1.5 * half_life_days)
    :param df:
    :param industry_col:
    :param sec_id_col:
    :param date_col:
    :param restructure_col:
    :param half_life_days:
    :param burn_in_days:
    :return:
    '''
    return (
        df.sort([sec_id_col, date_col]).with_columns([
            (pl.col(industry_col) != pl.col(industry_col).shift(1))
            .fill_null(False)
            .over(sec_id_col)
            .alias('_tmp_ind_changed'),
            pl.col(restructure_col).fill_null(0).cast(pl.Boolean).alias('_tmp_re')
        ])
        .with_columns([
            ((pl.col('_tmp_ind_changed') | pl.col('_tmp_re')).cast(pl.Int32))
            .cum_sum()
            .over(sec_id_col)
            .alias('_tmp_seg_id')
        ])
        .with_columns([
            pl.int_range(0, pl.len()).over([sec_id_col, '_tmp_seg_id']).alias('_tmp_seg_age'),
        ])
        .with_columns([
            ((pl.col('_tmp_seg_age') - burn_in_days) / (half_life_days * 1.5))
            .clip(0.0, 1.0)
            .alias('_tmp_bayes_w')
        ])
    )


def build_valuation_feature_bus(
        df: pl.DataFrame,
        catalog: MetricCatalog = MetricCatalog(),
        config: ValuationBusConfig = ValuationBusConfig(),
):
    '''
    【单次执行】准备结构突变分段上下文与经验贝叶斯权重：
    这一段实际上只要是时序因子都可复用
    '''
    active_specs: Dict[str, MetricSpec] = catalog.resolve_active_specs(df.columns)
    if not active_specs:
        return df

    # 1. 幂等缓存： 目标列已全部存在则直接返回
    expected_cols = [
        col for prefix in active_specs
        for col in (f'{prefix}_anchor', f'{prefix}_ts_alpha_zscore')
    ] # 笛卡尔乘积
    if all(c in df.columns for c in expected_cols):
        return df

    res_df, _ = _compute_batch_internal(catalog, config, df)
    print(res_df)
    return res_df


def _compute_batch_internal(
                            catalog: MetricCatalog,
                            config: ValuationBusConfig,
                            df: pl.DataFrame,
                            keep_state_cols = False) -> tuple[pl.DataFrame, Dict[str, MetricSpec]]:
    active_specs: dict[str, MetricSpec] = catalog.resolve_active_specs(df.columns)
    if not active_specs:
        return df, active_specs

    schema = config.schema
    industry_col: str = schema.industry_col
    sec_id_col: str = schema.sec_id_col
    date_col: str = schema.date_col
    restructure_col: str = schema.restructure_col
    half_life_days: int = config.half_life_days
    burn_in_days: int = config.burn_in_days
    min_periods: int = config.min_periods

    ctx_df = prepare_segment_context(df, industry_col, sec_id_col, date_col, restructure_col, half_life_days,
                                     burn_in_days)

    segment_keys = [sec_id_col, '_tmp_seg_id']
    stage1_val, stage2_ind_med, stage3_ema, stage4_out = [], [], [], []

    # 3.批量组装所有指标的表达式
    # for prefix, spec in active_specs.items():
    #     c_val = f'_tmp_stk_n_{prefix}'
    #     c_ind_med = f'_tmp_ind_med_{prefix}'
    #     c_stk_mu = f'_tmp_stk_mu_{prefix}'
    #     c_stk_std = f'_tmp_stk_std_{prefix}'
    #     c_ind_mu = f'_tmp_ind_mu_{prefix}'
    #     c_ind_std = f'_tmp_ind_std_{prefix}'
    #
    #     # stage 1: 因子转换
    #     stage1_val.append(spec.build_transorm_expr('_tmp_seg_age', burn_in_days).alias(c_val))
    #
    #     # stage 2: 行业当日截面中位数
    #     stage2_ind_med.append(pl.col(c_val).median().over([industry_col, date_col]).alias(c_ind_med))
    #
    #     # stage 3: 个股EMA 与行业基准 EMA
    #     stage3_ema.extend([
    #         pl.col(c_val).ewm_mean(half_life=half_life_days, min_samples=min_periods).over(segment_keys).alias(
    #             c_stk_mu),
    #         pl.col(c_val).ewm_std(half_life=half_life_days, min_samples=min_periods).over(segment_keys).alias(
    #             c_stk_std),
    #         pl.col(c_ind_med).ewm_mean(half_life=half_life_days, min_samples=min_periods).over(segment_keys).alias(
    #             c_ind_mu),
    #         pl.col(c_ind_med).ewm_std(half_life=half_life_days, min_samples=min_periods).over(segment_keys).alias(
    #             c_ind_std),
    #     ])
    #
    #     # stage 4: 贝叶斯收缩与方向对齐输出
    #     w = pl.col('_tmp_bayes_w')
    #     final_mu = w * pl.col(c_stk_mu) + (1.0 - w) * pl.col(c_ind_mu)
    #     final_std = w * pl.col(c_stk_std) + (1.0 - w) * pl.col(c_ind_std)
    #     direction_sign = 1.0 if spec.higher_is_alpha else -1.0
    #
    #     stage4_out.extend([
    #         spec.restore_anchor_expr(final_mu).alias(f'{prefix}_anchor'),
    #         (direction_sign * (pl.col(c_val) - final_mu) / (final_std + 1e-4))
    #         .alias(f'{prefix}_ts_alpha_zscore')
    #     ])

    # 4. 顺序推进阶段并按前缀规则统一清理所有临时列

    stage1_exprs = []
    for prefix, spec in active_specs.items():
        val_expr = spec.build_transorm_expr('_tmp_seg_age', burn_in_days)
        stage1_exprs.extend([val_expr.alias(f'_tmp_val_{prefix}'),
                             val_expr.pow(2).alias(f'_tmp_val2_{prefix}')])
    ctx_df = ctx_df.with_columns(stage1_exprs)

    # ================================================
    # 3. 行业先验独立降维计算
    # ================================================
    ind_ts_exprs = [
        pl.col(f'_tmp_val_{prefix}').median().alias(f'_tmp_ind_med_{prefix}')
        for prefix in active_specs
    ]
    ind_panel = (
        ctx_df.filter(pl.col(industry_col).is_not_null())
        .group_by([industry_col, date_col])
        .agg(ind_ts_exprs)
        .sort([industry_col, date_col])
        .with_columns([
            pl.col(f'_tmp_ind_med_{prefix}').pow(2).alias(f'_tmp_ind_med2_{prefix}')
            for prefix in active_specs
        ])
    )

    ind_ema_exprs = []
    for prefix in active_specs:
        c_med = f'_tmp_ind_med_{prefix}'
        c_med2 = f'_tmp_ind_med2_{prefix}'
        ind_ema_exprs.extend([
            pl.col(c_med)
            .ewm_mean(half_life=half_life_days, min_samples=1, adjust=False, ignore_nulls=True)
            .forward_fill()
            .over(industry_col).alias(f'_tmp_ind_mu_{prefix}'),
            pl.col(c_med2)
            .ewm_mean(half_life=half_life_days, min_samples=1, adjust=False, ignore_nulls=True)
            .forward_fill().over(industry_col).alias(
                f'_tmp_ind_s2_{prefix}'),

        ])

    ind_panel = ind_panel.with_columns(ind_ema_exprs).select(
        [industry_col, date_col]
        + [col for p in active_specs for col in (f"_tmp_ind_mu_{p}", f"_tmp_ind_s2_{p}")]
    )

    # 将唯一行业序列 join 回个股明细表（每个 [industry, date] 唯一，无笛卡尔积膨胀风险）
    ctx_df = ctx_df.join(ind_panel, on=[industry_col, date_col], how="left").sort([sec_id_col, date_col])

    # -----------------------------------------------------------------
    # 4. 个股分段 EMA 一阶矩与二阶矩计算（纯 ewm_mean，零内存溢出）
    # -----------------------------------------------------------------
    segment_keys = [sec_id_col, "_tmp_seg_id"]
    stk_ema_exprs = []
    for prefix in active_specs:
        c_val = f"_tmp_val_{prefix}"
        c_val2 = f"_tmp_val2_{prefix}"
        stk_ema_exprs.extend([
            pl.col(c_val).is_not_null().cast(pl.Int64).cum_sum().over(segment_keys).alias(f"_tmp_stk_n_{prefix}"),
            pl.col(c_val).ewm_mean(half_life=half_life_days, min_samples=1, adjust=False, ignore_nulls=True)
            .forward_fill().over(segment_keys).alias(
                f"_tmp_stk_mu_{prefix}"),
            pl.col(c_val2).ewm_mean(half_life=half_life_days, min_samples=1, adjust=False, ignore_nulls=True)
            .forward_fill().over(segment_keys).alias(
                f"_tmp_stk_s2_{prefix}"),
        ])
    ctx_df = ctx_df.with_columns(stk_ema_exprs)

    # -----------------------------------------------------------------
    # 5. 贝叶斯收缩与输出对齐（严格保留原生 Null 状态）
    # -----------------------------------------------------------------
    out_exprs = []
    for prefix, spec in active_specs.items():
        c_val = f"_tmp_val_{prefix}"
        c_stk_n = f"_tmp_stk_n_{prefix}"
        c_stk_mu = f"_tmp_stk_mu_{prefix}"
        c_stk_s2 = f"_tmp_stk_s2_{prefix}"
        c_ind_mu = f"_tmp_ind_mu_{prefix}"
        c_ind_s2 = f"_tmp_ind_s2_{prefix}"

        valid_stk_mu = pl.when(pl.col(c_stk_n) >= min_periods).then(pl.col(c_stk_mu)).otherwise(None)
        valid_stk_s2 = pl.when(pl.col(c_stk_n) >= min_periods).then(pl.col(c_stk_s2)).otherwise(None)
        stk_std = (valid_stk_s2 - valid_stk_mu.pow(2)).clip(0.0, None).sqrt()

        ind_mu = pl.col(c_ind_mu)
        ind_std = (pl.col(c_ind_s2) - ind_mu.pow(2)).clip(0.0, None).sqrt()

        w = pl.col("_tmp_bayes_w")
        final_mu = w * valid_stk_mu.fill_null(ind_mu) + (1.0 - w) * ind_mu
        final_std = w * stk_std.fill_null(ind_std) + (1.0 - w) * ind_std
        direction_sign = 1.0 if spec.higher_is_alpha else -1.0

        out_exprs.extend([
            spec.restore_anchor_expr(final_mu).alias(f"{prefix}_anchor"),
            (direction_sign * (pl.col(c_val) - final_mu) / (final_std + 1e-4))
            .alias(f"{prefix}_ts_alpha_zscore"),
        ])

    res_df = ctx_df.with_columns(out_exprs)

    if not keep_state_cols:
        tmp_cols = [c for c in res_df.columns if c.startswith("_tmp_")]
        res_df = res_df.drop(tmp_cols)

    return res_df, active_specs
    # res_df = (
    #     ctx_df
    #     .with_columns(stage1_val)
    #     .with_columns(stage2_ind_med)
    #     .with_columns(stage3_ema)
    #     .with_columns(stage4_out)
    # )
    #
    # # return res_df
    # if not keep_state_cols:
    #     tmp_cols = [c for c in res_df.columns if c.startswith('_tmp_')]
    #     res_df = res_df.drop(tmp_cols)
    # # print(res_df.columns)
    #
    # return res_df, active_specs



