'''
基本面因子
'''
from abc import ABC, abstractmethod
import polars as pl
from strategy.factors.base_factor import BaseFactorGenerator

class BaseValuationGenerator(BaseFactorGenerator, ABC):
    '''
    价值因子领域基类：
    仅声明估值大类标识与通用契约，不掺杂任何特定子类的数学计算逻辑
    '''
    factor_family = 'Valuation'

    @abstractmethod
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        '''所有估值生成器必须实现的主入口'''
        pass

