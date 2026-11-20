"""
preprocessor.py
- 截面特征工程预处理核心流水线
- 包含：MAD 去极值、robost Z-Score 标准化、OLS 截面行业与市值中性化
- 特点：零 Python 行循环、零方差防御、奇异矩阵容错
"""
from typing import List, Optional, Union
# import numpy as np
# import pandas as pd
import polars as pl
import polars_ols as pls # 倒入并自动注册 Polars OLS 命名空间拓展
# 使用polars_ols代替 numpy的好处
# polars-ols 是 Polars 官方生态推荐的 Rust 原生扩展插件
# 1. 纯 Rust 内存运算，零拷贝：回归过程完全停留在 Arrow 缓冲区中，无需向 Python / NumPy 转换数据；
# 2. 表达式原生支持（.over("date")）：可以直接配合 .over("date") 在 Rust 底层多线程并行执行每个截面的回归；
# 3. 原生支持分类变量（行业哑变量）与残差提取：直接通过 .least_squares.ols(..., mode="residuals") 提取残差。
# 中性化的意义是将市值和行业从因子中剥离出去，减少影响。


class FactorPreprocessor:
    '''
    多因子界面特征预处理器
    '''
    # ============== 1. 基础算子：MAD去极值 ==============
    @staticmethod
    def winsorized_mad(
            df: pl.DataFrame,
            factor_col: str,
            n: float = 3.0,
            group_col: str = 'date',
    ):
        '''
        中位数绝对偏差去极值(Median Absolute Deviation, MAD: 好处避免了方差去极值时带来的不稳定因素干扰导致的最终极值未去除干净)
        算法：
            median = Median(x)
            MAD = Median(|X - median|)
            upper = median + n * 1.4826 * MAD
            lower = median - n * 1.4826 * MAD

        :param df:
        :param factor_col:
        :param n:
        :param group_col:
        :return:
        '''
        # 1.计算截面中位数
        median_expr = pl.col(factor_col).median().over(group_col)

        # 2. 计算截面离差中位数 MAD
        mad_expr = (
            (pl.col(factor_col) - median_expr).abs().median().over(group_col)
        )

        # 3. 构造 1.4826 标准化渐进上下界
        scale = 1.4826
        upper_bound = median_expr + n * scale * mad_expr
        lower_bound = median_expr - n * scale * mad_expr

        # 4. 向量化 clip 截断
        return df.with_columns(
            pl.col(factor_col).clip(lower_bound, upper_bound).alias(factor_col)
        )

    @staticmethod
    def standardize_zscore(
            df: pl.DataFrame,
            factor_col: str,
            group_col: str = 'date',
            epsilon: float = 1e-8,
    ) -> pl.DataFrame:
        '''
        截面 Z-Score 标准化 (带零方差与极小方差防御）：
            z = (x - mean(x)) / std(x)
            当 std < eps 时， 强制置0，防止毒性数据（NaN/inf)污染优化器
        :param df:
        :param factor_col:
        :param group_col:
        :param epsilon:
        :return:
        '''
        mean_expr = pl.col(factor_col).mean().over(group_col)
        std_expr = pl.col(factor_col).std().over(group_col)

        return df.with_columns(
            pl.when(std_expr.fill_null(0.0) < epsilon)
            .then(0.0)
            .otherwise((pl.col(factor_col) - mean_expr) / std_expr)
            .alias(factor_col)
        )

    @staticmethod
    def neutralize(
            df: pl.DataFrame,
            factor_col: str,
            group_col: str = 'date',
            id_col: str = 'sec_id',
            industry_col: Optional[str] = 'industry',
            market_cap_col: Optional[str] = 'total_mv',
    ) -> pl.DataFrame:
        '''
        截面 OLS 残差中性化
        回归模型：
            Factor =  beta_0 + sum(beta_ind * Dummy_ind) + beta_mv * ln(MarketCap) + Residual(Alpha)
        :param df:
        :param factor_col:
        :param group_col:
        :param id_col:
        :param industry_col:
        :param market_cap_col:
        :return:
        '''
        if not industry_col and not market_cap_col:
            return df

        features_to_drop = []
        target_features = []

        # 1. 构造对数市值列
        working_df = df
        if market_cap_col and market_cap_col in working_df.columns:
            ln_mv_col = '__ln_market_cap'
            working_df = working_df.with_columns(
                pl.col(market_cap_col).clip(1.0).log().alias(ln_mv_col)
            )
            target_features.append(ln_mv_col)
            features_to_drop.append(ln_mv_col)

        # 2. 构造行业哑变量
        if industry_col and industry_col in working_df.columns:
            # 使用select的原因是ind_dummies variable后续需要被用到
            ind_dummies = working_df.select(industry_col).to_dummies(drop_first=True)
            dummy_cols = ind_dummies.columns
            target_features.extend(dummy_cols)
            features_to_drop.extend(dummy_cols)

            # 水平拼接至主表,这个比pandas的concat好记多了
            working_df = pl.concat([working_df, ind_dummies], how='horizontal')

        # 3. 核心：调用 polars-ols 的least_squares 表达式提取残差
        # add_intercept=True 自动加入常数截距项
        # mode='residuals' 直接返回回归残差项
        ols_expr = (
            pl.col(factor_col)
            .least_squares.ols(
                *target_features,
                mode='residuals',
                add_intercept=True,
                null_policy='drop'
            )
            .over(group_col)
            .alias(factor_col)
        )

        # 4. 执行回归并清理临时特征列
        neutralized_df = working_df.with_columns(ols_expr).drop(features_to_drop)
        return neutralized_df

    # ================== 4. 一键处理流水线 (Pipeline) ==================
    @staticmethod
    def process_pipeline(
            df: pl.DataFrame,
            factor_col: str,
            group_col: str = 'date',
            tradable_filter_col: Optional[str] = 'is_tradable',
            mad_n: float = 3.0,
            industry_col: Optional[str] = 'industry',
            market_cap_col: Optional[str] = 'total_mv',
    ) -> pl.DataFrame:
        '''
        工业级全流程预处理流水线 (纯 Rust 向量化算子流）
        1. 过滤不可交易样本
        2. MAD 去极值
        3. polars-ols 截面中性化
        4. 截面 Z-Score
        :param cls:
        :param df:
        :param factor_col:
        :param group_col:
        :param tradable_filter_col:
        :param mad_n:
        :param industry_col:
        :param market_cap_col:
        :return:
        '''
        working_df = df
        # 1. 过滤可交易股票
        if tradable_filter_col and tradable_filter_col in working_df.columns:
            working_df = working_df.filter(pl.col(tradable_filter_col))

        # 2. MAD
        working_df = FactorPreprocessor.winsorized_mad(
            working_df,factor_col = factor_col, n = mad_n, group_col = group_col
        )

        # 3. 截面 OLS 中性化, 如果没有行业或市场列就跳过
        if industry_col or market_cap_col:
            working_df = FactorPreprocessor.neutralize(
                working_df,
                factor_col = factor_col,
                group_col = group_col,
                industry_col = industry_col,
                market_cap_col = market_cap_col
            )

        # 4. z-score
        processed_df = FactorPreprocessor.standardize_zscore(
            working_df, factor_col = factor_col, group_col = group_col
        )

        return processed_df
