'''
估值因子家族统一配置实体：隔离 Schema 列映射与时序统计超参数
'''

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
from .specs import MetricSpec, METRIC_REGISTRY
# from .. import registry


@dataclass(frozen=True)
class PanelSchema:
    '''底层面板数据列契约（全因子库通用）暂时放在valuation中，后续有需求再优化'''
    sec_id_col: str = 'sec_id'
    date_col: str = 'date'
    industry_col: str = 'industry'
    restructure_col: str = 'is_restructured'

@dataclass(frozen=True)
class MetricCatalog:
    '''
    1. 支持流式注册： 新增自定义指标时，支持catalog.with_custom(...)链式派生出新目录，不污染全局注册表
    2. 使用typle 存储键名，保障配置对象纯洁且支持快速派生
    '''
    keys: Tuple[str, ...] = ('sec_id_col', 'date_col', 'industry_col')
    custom_specs: Tuple[Tuple[str, MetricSpec], ...] = ()

    @classmethod
    def from_list(cls, keys: Iterable[str], custom_specs: Optional[Dict[str, MetricSpec]]) -> 'MetricCatalog':
        '''便捷方法：支持外部直接传入常规 Python List 和 Dict 构建定制的metrics'''
        specs_tuple = tuple(custom_specs.items()) if custom_specs else ()
        return cls(keys=tuple(keys), custom_specs=specs_tuple)

    def with_custom(self, extra_keys: Iterable[str], extra_specs: Optional[Dict[str, MetricSpec]] = None) -> 'MetricCatalog':
        '''派生新配置'''
        # 在保证元素不重复的前提下，完美保留合并后的先后顺序，并且具备极高的执行效率，最后将结果锁定为不可变的元组
        merged_keys = tuple(dict.fromkeys([*self.keys, *extra_keys]))
        merged_specs_dict = dict(self.custom_specs)
        if extra_specs:
            merged_specs_dict.update(extra_specs)

        return MetricCatalog(keys=merged_keys, custom_specs=tuple(merged_specs_dict.items()))

    def resolve_active_specs(self, available_columns: Iterable[str]) -> Dict[str, MetricSpec]:
        '''解析出active的metric, 若未配置或不在raw_col中则不返回'''
        col_set = set(available_columns)
        registry = {**METRIC_REGISTRY, **dict(self.custom_specs)}
        return {
            k: registry[k]
            for k in self.keys
            if k in registry and registry[k].raw_col in col_set
        }

@dataclass(frozen=True)
class ValuationBusConfig:
    '''时序特征总线与增量状态机共享超参数'''
    half_life_days: int = 250
    burn_in_days: int = 60
    min_periods: int = 50
    schema: PanelSchema = field(default_factory=PanelSchema)

    def __post_init__(self):
        if  self.half_life_days <= 0:
            raise ValueError(f'half_life_days 必须大于0，当前： {self.hal_life_days}')
        if self.burn_in_days < 0:
            raise ValueError(f'burn_in_days 不能为负，当前为：{self.burn_in_days}')
        if self.min_periods < 0:
            raise ValueError(f'min_periods 必须 >= 1,当前为 {self.min_periods}')
