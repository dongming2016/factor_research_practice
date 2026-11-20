"""
tests/factors/valuation/test_batch_vs_incremental.py
端到端集成测试：验证批处理冷启动 (initialize_state_snapshots) + 逐日增量 (step_daily)
与全量一次性批处理在极端市场边界下的数值比特级等价性 (atol < 1e-12)
"""

from datetime import date, timedelta
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from strategy.factors.valuation.config import MetricCatalog, PanelSchema, ValuationBusConfig
from strategy.factors.valuation.incremental import IncrementalValuationBus
from strategy.factors.valuation.utils import build_valuation_feature_bus
from strategy.factors.valuation.time_series import TimeSeriesValuation
from strategy.factors.valuation.mispricing import PbROEMispricingGenerator


@pytest.fixture(scope="module")
def complex_market_panel() -> pl.DataFrame:
    """
    构造 120 个交易日、包含 6 类极端边界场景的模拟全市场面板
    """
    rng = np.random.default_rng(42)
    base_date = date(2026, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(120)]

    records = []

    # 补充若干背景股票保证截面 qcut(5) 分箱和行业中位数统计稳健
    bg_stocks = [
        ("STK_BG_A1", "IND_A"), ("STK_BG_A2", "IND_A"), ("STK_BG_A3", "IND_A"),
        ("STK_BG_B1", "IND_B"), ("STK_BG_B2", "IND_B"), ("STK_BG_B3", "IND_B"),
    ]

    for day_idx, dt in enumerate(dates):
        # 1. 背景常规股票
        for sec_id, ind in bg_stocks:
            records.append({
                "sec_id": sec_id,
                "date": dt,
                "industry": ind,
                "is_restructured": 0,
                "pe_ttm": float(rng.uniform(12.0, 45.0)),
                "pb": float(rng.uniform(1.2, 6.0)),
                "roe": float(rng.uniform(0.04, 0.25)),
                "total_mv": float(rng.uniform(5e9, 1e11)),
            })

        # 2. STK_01_NORMAL: 全程连续交易
        records.append({
            "sec_id": "STK_01_NORMAL",
            "date": dt,
            "industry": "IND_A",
            "is_restructured": 0,
            "pe_ttm": float(rng.uniform(15.0, 35.0)),
            "pb": float(rng.uniform(1.5, 4.5)),
            "roe_ttm": float(rng.uniform(0.08, 0.22)),
            "total_mv": 3e10,
        })

        # 3. STK_02_IPO: 第 40 天上市的新股
        if day_idx >= 40:
            records.append({
                "sec_id": "STK_02_IPO",
                "date": dt,
                "industry": "IND_B",
                "is_restructured": 0,
                "pe_ttm": float(rng.uniform(25.0, 80.0)),
                "pb": float(rng.uniform(3.0, 9.0)),
                "roe_ttm": float(rng.uniform(0.10, 0.30)),
                "total_mv": 1.5e10,
            })

        # 4. STK_03_SUSPEND: 第 75~84 天停牌（直接无行情记录，模拟停牌剔除）
        if not (75 <= day_idx <= 84):
            records.append({
                "sec_id": "STK_03_SUSPEND",
                "date": dt,
                "industry": "IND_A",
                "is_restructured": 0,
                "pe_ttm": float(rng.uniform(10.0, 25.0)),
                "pb": float(rng.uniform(1.0, 3.0)),
                "roe_ttm": float(rng.uniform(0.05, 0.18)),
                "total_mv": 2e10,
            })

        # 5. STK_04_IND_SHIFT: 第 70 天跨行业重组 (IND_A -> IND_B, is_restructured=1)
        ind_04 = "IND_A" if day_idx < 70 else "IND_B"
        restruct_04 = 1 if day_idx == 70 else 0
        records.append({
            "sec_id": "STK_04_IND_SHIFT",
            "date": dt,
            "industry": ind_04,
            "is_restructured": restruct_04,
            "pe_ttm": float(rng.uniform(18.0, 50.0)),
            "pb": float(rng.uniform(2.0, 5.5)),
            "roe_ttm": float(rng.uniform(0.06, 0.20)),
            "total_mv": 2.5e10,
        })

        # 6. STK_05_LOSS_PE: 第 60~95 天亏损（PE < 0 且 ROE < 0），其余时间盈利
        is_loss = 60 <= day_idx <= 95
        records.append({
            "sec_id": "STK_05_LOSS_PE",
            "date": dt,
            "industry": "IND_B",
            "is_restructured": 0,
            "pe_ttm": -15.5 if is_loss else float(rng.uniform(14.0, 30.0)),
            "pb": float(rng.uniform(1.1, 2.8)),
            "roe_ttm": -0.08 if is_loss else float(rng.uniform(0.07, 0.19)),
            "total_mv": 1.2e10,
        })

        # 7. STK_06_IND_FREEZE: 独占冷门行业 IND_C，并在 Day 80~82 停牌（导致 IND_C 全行业无有效值）
        is_ind_c_suspended = 80 <= day_idx <= 82
        records.append({
            "sec_id": "STK_06_IND_FREEZE",
            "date": dt,
            "industry": "IND_C",
            "is_restructured": 0,
            "pe_ttm": None if is_ind_c_suspended else float(rng.uniform(20.0, 40.0)),
            "pb": None if is_ind_c_suspended else float(rng.uniform(2.0, 4.0)),
            "roe_ttm": None if is_ind_c_suspended else float(rng.uniform(0.09, 0.21)),
            "total_mv": 8e9,
        })

    return pl.DataFrame(records)


def test_batch_vs_incremental_exact_equivalence(complex_market_panel: pl.DataFrame, tmp_path):
    """
    核心集成断言：
    1. 前 60 天 (Day 0 ~ Day 59) 使用 initialize_state_snapshots 批处理冷启动并落盘快照；
    2. 后 60 天 (Day 60 ~ Day 119) 逐日调用 step_daily 增量推进（覆盖停牌、亏损、重组与全行业冻结）；
    3. 拼接全量 120 天增量总线表，断言与一次性全量 build_valuation_feature_bus 误差 < 1e-12；
    4. 断言下游 TimeSeriesValuationGenerator 与 PbRoeMispricingGenerator 双模输出完全一致；
    5. 显式断言 burn_in 隔离期原生 Null 行为。
    """
    # 配置较短的周期以便在 120 天窗口内充分触发 burn_in 结束与 min_periods 达标
    catalog = MetricCatalog.from_list(["ts_pe", "ts_pb", "ts_roe"], {})
    bus_cfg = ValuationBusConfig(
        half_life_days=30,
        burn_in_days=10,
        min_periods=8,
        schema=PanelSchema(),
    )

    all_dates = complex_market_panel.select("date").unique().sort("date")["date"].to_list()
    cutoff_date = all_dates[59]       # Day 59 作为历史批处理截止日
    incremental_dates = all_dates[60:]  # Day 60 ~ Day 119 逐日增量推演

    # =====================================================================
    # 路径 A：一次性全量批处理基准（Ground Truth）
    # =====================================================================
    full_batch_bus = build_valuation_feature_bus(
        df=complex_market_panel,
        catalog=catalog,
        config=bus_cfg,
    ).sort(["sec_id", "date"])

    # =====================================================================
    # 路径 B：批处理冷启动 (Day 0~59) + 逐日增量推进 (Day 60~119)
    # =====================================================================
    inc_engine = IncrementalValuationBus(catalog=catalog, config=bus_cfg)
    stk_snap_path = str(tmp_path / "val_stock_state.parquet")
    ind_snap_path = str(tmp_path / "val_ind_state.parquet")

    _, _, hist_bus_part = inc_engine.initialize_state_snapshots(
        history_df=complex_market_panel,
        cutoff_date=cutoff_date,
        stock_state_path=stk_snap_path,
        ind_state_path=ind_snap_path,
        need_history_bus=True,
    )

    # 模拟每日收盘从磁盘读取昨日快照并增量推进
    stk_state = pl.read_parquet(stk_snap_path)
    ind_state = pl.read_parquet(ind_snap_path)
    daily_bus_chunks = [hist_bus_part]

    for dt in incremental_dates:
        today_slice = complex_market_panel.filter(pl.col("date") == dt)
        today_bus, stk_state, ind_state = inc_engine.step_daily(
            today_df=today_slice,
            stock_state_df=stk_state,
            ind_state_df=ind_state,
        )
        daily_bus_chunks.append(today_bus)

    stitched_incremental_bus = (
        pl.concat(daily_bus_chunks, how="vertical")
        .select(full_batch_bus.columns)
        .sort(["sec_id", "date"])
    )

    # =====================================================================
    # 断言 1：特征总线全量列（含所有 anchor 与 ts_alpha_zscore）误差 < 1e-12
    # =====================================================================
    assert_frame_equal(
        full_batch_bus,
        stitched_incremental_bus,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # =====================================================================
    # 断言 2：下游两大生成器在批处理总线与增量拼接总线上输出完全一致
    # =====================================================================
    ts_gen = TimeSeriesValuation(catalog=catalog, bus_config=bus_cfg)
    mispricing_gen = PbROEMispricingGenerator(bus_config=bus_cfg, n_pe_tiers=3)

    assert_frame_equal(
        ts_gen.generate(full_batch_bus).sort(["sec_id", "date"]),
        ts_gen.generate(stitched_incremental_bus).sort(["sec_id", "date"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert_frame_equal(
        mispricing_gen.generate(full_batch_bus).sort(["sec_id", "date"]),
        mispricing_gen.generate(stitched_incremental_bus).sort(["sec_id", "date"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # =====================================================================
    # 断言 3：专项业务边界校验（burn_in 原生 Null 与行业突变重置）
    # =====================================================================
    # 3.1 新股 STK_02_IPO 在上市前 11 天 (age 0..10 <= burn_in=10) 必须严格为 Null，第 12 天恢复非空
    ipo_df = stitched_incremental_bus.filter(pl.col("sec_id") == "STK_02_IPO").sort("date")
    assert ipo_df.slice(0, 11)["ts_pe_ts_alpha_zscore"].null_count() == 11
    assert ipo_df.slice(11, 1)["ts_pe_ts_alpha_zscore"].null_count() == 0

    # 3.2 重组股 STK_04_IND_SHIFT 在 Day 70 发生重组断点后，必须重新进入 11 天原生 Null 隔离期
    shift_df = stitched_incremental_bus.filter(
        (pl.col("sec_id") == "STK_04_IND_SHIFT") & (pl.col("date") >= all_dates[70])
    ).sort("date")
    assert shift_df.slice(0, 11)["ts_pe_ts_alpha_zscore"].null_count() == 11
    assert shift_df.slice(11, 1)["ts_pe_ts_alpha_zscore"].null_count() == 0

    # 3.3 亏损股 STK_05_LOSS_PE 在亏损期 PE Z-Score 为 Null，但 ROE Z-Score 正常有值
    loss_df = stitched_incremental_bus.filter(
        (pl.col("sec_id") == "STK_05_LOSS_PE") & (pl.col("pe_ttm") < 0) & (pl.col("roe") < 0)
    )
    assert loss_df["ts_pe_ts_alpha_zscore"].is_null().all()
    assert loss_df["ts_roe_ts_alpha_zscore"].is_not_null().all()