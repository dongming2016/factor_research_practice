'''
声明式指标配置契约：统一定义全量总线与增量状态及的数学转换与方向逻辑
'''

from dataclasses import dataclass
from typing import Dict, Literal
import polars as pl

# 这里只注册了三种变换方式，后续可以结合需求进一步优化
TransformType = Literal['inverse', 'identity', 'log']

@dataclass(frozen=True)
class MetricSpec:
    raw_col: str # 原始数据表中的列名
    transform: TransformType = 'inverse' # 变换方式，inverse：1/x, identity: x, log: ln(x)
    higher_is_alpha: bool = True # 变换后，越大越正向
    clip_min: float | None = None
    clip_max: float | None = None

    def build_transorm_expr(self, segment_age_col: str, burn_in_days: int) -> pl.Expr:
        '''
        生成统一的 Polars 空间映射表达式 （全量与增量引擎公用此唯一真源）
        :param segment_age_col:
        :param burn_in_days:
        :return:
        '''
        col_expr = pl.col(self.raw_col)

        if self.clip_min is not None or self.clip_max is not None:
            if self.clip_min is not None or self.clip_max is not None:
                col_expr = col_expr.clip(lower_bound=self.clip_min, upper_bound=self.clip_max)


        if self.transform == 'inverse':
            transformed = pl.when(col_expr > 0).then(1 / col_expr).otherwise(None)
        elif self.transform == 'log':
            transformed = pl.log(col_expr > 0).then(col_expr.log()).otherwise(None)
        else:
            transformed = col_expr

        return (
            pl.when(pl.col(segment_age_col) > burn_in_days)
            .then(transformed)
            .otherwise(None)
        )

    def restore_anchor_expr(self, mean_col_expr: pl.Expr) -> pl.Expr:
        '''
        将平滑后的均值还原回原始物理量纲 （供截面分档或看板监控使用）
        :param mean_col_expr:
        :return:
        '''
        if self.transform == 'inverse':
            return 1 / mean_col_expr
        elif self.transform == 'log':
            return mean_col_expr.log()

        return mean_col_expr

# =========================================================
# 全局指标注册中心： 后续扩展指标在此追加
# =========================================================
METRIC_REGISTRY: Dict[str, MetricSpec] = {
    'ts_pe': MetricSpec(raw_col='pe_ttm', transform='inverse', higher_is_alpha=True),
    'ts_pb': MetricSpec(raw_col='pb', transform='inverse', higher_is_alpha=True),
    'ts_ps': MetricSpec(raw_col='ps', transform='inverse', higher_is_alpha=True),
    'ts_pcf': MetricSpec(raw_col='pcf', transform='inverse', higher_is_alpha=True),
    'ts_roe': MetricSpec(raw_col='roe', transform='identity', higher_is_alpha=True, clip_min=-0.5,clip_max=0.8)
}