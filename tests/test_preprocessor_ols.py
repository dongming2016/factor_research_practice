from datetime import date
import numpy as np
import polars as pl
import pytest

from preprocessor.preprocessor import FactorPreprocessor

# 常规的测试用例

# ============================================
# 1. MAD 去极值算子测试
# ============================================

def test_winsorize_mad_resists_extreme_pollution():
    '''
    测试 MAD 去极值的高击穿特性：
    即使单点收到高达 10^7 的极端污染，阈值边界依然受中位数保护
    不能检查逻辑错误，
    异常值用 median +- 3 * 1.4826 * median(|x - median(x)|) 替换
    mad的好处就是不会被异常值拉偏
    :return:
    '''
    mock_df = pl.DataFrame(
        {
            'date': [date(2024, 4, 15)] * 5,
            'sec_id': ['000001', '000002', '000003', '000004', '000005'],
            'factors': [1.0, 1.0, 2.0, 3.0, 10_000_000.0]
        }
    )

    clipped = FactorPreprocessor.winsorized_mad(
        df=mock_df, factor_col='factors', n=3.0, group_col='date'
    )

    res = clipped['factors'].to_list()

    print(res)
    assert res[4] == pytest.approx(2.0 + 3.0 * 1.4826 * 1.0, rel=1e-4)
    assert res[0] == 1.0
    assert res[2] == 2.0

# ============================================
# 2. 截面 Z-Score 去极值算子测试
# ============================================
def test_zscore_zero_var_safety():
    '''
    测试0方差,不能是nan和inf
    :return:
    '''
    mock_df = pl.DataFrame({
        'date': [date(2024, 4, 15)] * 5,
        'sec_id': ['000001', '000002', '000003', '000004', '000005'],
        'factors': [5.0, 5.0, 5.0, 5.0, 5.0]
    })

    scaled = FactorPreprocessor.standardize_zscore(
        df=mock_df, factor_col='factors', group_col='date', epsilon=1e-8
    )
    vals = scaled['factors'].to_list()

    assert vals == [0.0] * len(vals)
    assert not any(np.isnan(vals))
    assert not any(np.isinf(vals))

def test_zscore_mean_var_property():
    '''
    检测平均值和标准差的特征是否满足平均值=0，标准差=1
    :return:
    '''
    mock_df = pl.DataFrame({
        'date': [date(2024, 4, 15)] * 5,
        'sec_id': ['000001', '000002', '000003', '000004', '000005'],
        'factors': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    scaled = FactorPreprocessor.standardize_zscore(
        df=mock_df, factor_col='factors', group_col='date', epsilon=1e-8
    )
    print(scaled)
    vals = scaled['factors'].to_numpy()
    print(vals)

    assert np.mean(vals) == pytest.approx(0.0, abs=1e-6)
    assert np.std(vals, ddof=1) == pytest.approx(1.0, abs=1e-6) # np.std(a)  # 默认 ddof=0 → 分母 n，总体标准差

# ============================================
# 3.截面中性正交性测试
# 中性的含义就是将因子与行业、市值剥离出来，barra因子因为本身就有行业和市值因子所以无需剥离
# ============================================
def test_neutralize_orthogonality():
    np.random.seed(42)
    n = 60
    # 模拟：原始因子被对数市值强烈驱动
    market_cap = np.random.uniform(1e9, high=1e11, size=n) # 均匀分布
    raw_factor = 2.5 * np.log(market_cap) + np.random.normal(0, 0.5 , n)
    mock_df = pl.DataFrame({
        'date': [date(2024, 4, 15)] * n,
        'sec_id': [f'{i: 07d}' for i in range(n)],
        'factors': raw_factor,
        'industry': ['银行', '医药', '电子', '消费'] * (n // 4),
        'total_mv': market_cap,
    })

    print(mock_df)
    neu_df = FactorPreprocessor.neutralize(
        df=mock_df, factor_col='factors', group_col='date', industry_col='industry', market_cap_col='total_mv'
    )

    residuals = neu_df['factors'].to_numpy()
    ln_mv = np.log(neu_df['total_mv'].to_numpy())

    # 1. 验证残差与自变量的相关性
    corr = np.corrcoef(residuals, ln_mv)[0, 1]
    assert abs(corr) < 1e-6

    # 2. 验证中性化没有一流临时生成的哑变量
    assert '__ln_market_cap__' not in neu_df.columns

def test_neutralize_cross_section_independence():
    '''
    测试独立性
    :return:
    '''
    mock_df = pl.DataFrame({
        'date': [date(2024, 4, 15)] * 20 + [date(2024, 4, 16)] * 20,
        'sec_id': [f'{i: 07d}' for i in range(20)] * 2,
        'factors': np.concatenate([np.linspace(1, 10, 20), np.linspace(10, 20, 20)]),
        'total_mv': np.concatenate([np.linspace(1e9, 1e11, 20), np.linspace(2e9, 2e11, 20)]),
        'industry': ['金融', '科技'] * 20
    })

    neu_df = FactorPreprocessor.neutralize(
        df=mock_df,
        factor_col='factors',
        group_col='date',
        industry_col='industry',
        market_cap_col='total_mv',)

    assert neu_df.height == 40

    r1 = neu_df.filter(pl.col('date') == date(2024, 4, 15))["factors"].to_numpy()
    r2 = neu_df.filter(pl.col('date') == date(2024, 4, 16))["factors"].to_numpy()

    print(r1, r2)
    assert not np.allclose(r1, r2)

# ============================================
# 4. 全流程流水线与样本过滤集成测试
# ============================================
def test_process_pipeline():
    '''
    测试一键预处理全流程(process_pipeline):
    1. 自动过滤 is_tradable == False 的停牌/ST 标的；
    2. 顺次执行 MAD -> OLS -> Z-Score;
    3. 最终阐述严格标准化分布 (均值 约等于 0， 方差 约等于 1）
    :return:
    '''
    np.random.seed(101)
    n = 40
    mv = np.random.uniform(1e9, 1e11, size=n)
    factors = 1.8 * np.log(mv) + np.random.normal(0, 1.0, n)
    factors[0] = 9999.0 # 注入极端值
    factors[-1] = 9999.0 # 注入极端值

    # 设置 2 只股票不可交易
    tradable = [True] * n
    tradable[3] = False
    tradable[9] = False

    mock_df = pl.DataFrame({
        'date': [date(2024, 4, 15)] * n,
        'sec_id': [f'{i: 07d}' for i in range(n)],
        'factors': factors,
        'industry': ['电子', '汽车', '化工', '家电'] * 10,
        'total_mv': mv,
        'is_tradable': tradable
    })

    processed_df = FactorPreprocessor.process_pipeline(
        df = mock_df,
        factor_col='factors',
        group_col='date',
        tradable_filter_col='is_tradable',
        mad_n=3.0,
        industry_col='industry',
        market_cap_col='total_mv',
    )

    # 1. 不可交易行已被剔除
    assert processed_df.height == 38
    assert '000003' not in processed_df['sec_id'].to_list()
    assert '000009' not in processed_df['sec_id'].to_list()

    # 2. 验证最终特征分布
    final_vals = processed_df['factors'].to_numpy()
    assert np.mean(final_vals) == pytest.approx(0.0, abs=1e-6)
    assert np.std(final_vals, ddof=1) == pytest.approx(1.0, abs=1e-6)
