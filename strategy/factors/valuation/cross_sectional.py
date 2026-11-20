from typing import Dict, List, Optional
import polars as pl
from strategy.factors.registry import FactorRegistry
from base import BaseValuationGenerator

@FactorRegistry.register('valuation.cs')
class CrossSectionalValuationGenerator(BaseValuationGenerator):
    '''
    截面相对低估因子：
    纯粹聚焦于横向指标的倒数清洗（1/PE， 1/PB， 1/PS， 1/PCF 等） 与股息率直通
    '''
    @staticmethod
    def expr_reciprocal(
            col_name: str,
            alias_name: str,
            default_val: float = 0.0,
    ) -> pl.Expr:
        '''
        通用估值倒数转换算子
        针对 PE， PB, PS, PCF 等指标，当值小于等于0或为null时统一退回为default_val
        :param col_name:
        :param alias_name:
        :param default_val:
        :return:
        '''
        return (
            pl.when(pl.col(col_name) > 0)
            .then(1.0 / pl.col(col_name))
            .otherwise(default_val)
            .alias(alias_name)
        )


    @staticmethod
    def expr_passthrough(
            col_name: str,
            alias_name: str,
            fill_val: float = 0.0
    ):
        '''非倒数类指标直通算子，负责空值安全填充'''
        return pl.col(col_name).fill(fill_val).alias(alias_name)

    # ================ 2. 规则驱动与按需路由 ================
    # 使用元数据配置驱动
    FACTOR_COFIGS: Dict[str, tuple[str, str]] = {
        'ep_ratio': ('pe_ttm', 'reciprocal'),
        'bp_ratio': ('pb', 'reciprocal'),
        'sp_ratio': ('ps_ttm', 'reciprocal'),
        'cfp_ratio': ('pcf_ttm', 'reciprocal'),
        'dividend_yield': ('dividend_yield_ratio', 'passthrough')
    }

    def __init__(self, selected_factors: Optional[List[str]] = None):
        self.selected_factors = selected_factors or list(self.FACTOR_COFIGS.keys())

    def _build_expr(self, factor_name: str) -> Optional[pl.Expr]:
        '''根据规则动态派发构建表达式'''
        if factor_name not in self.FACTOR_COFIGS:
            return None

        src_col, transform_type = self.FACTOR_COFIGS[factor_name]
        if transform_type == 'reciprocal':
            return self.expr_reciprocal(col_name=src_col, alias_name=factor_name)
        elif transform_type == 'passthrough':
            return self.expr_passthrough(col_name=src_col, alias_name=factor_name)
        return None

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs_to_run = [
            expr for name in self.selected_factors
            if (expr := self._build_expr(name)) is not None
        ]

        return df.with_columns(exprs_to_run)