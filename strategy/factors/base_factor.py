'''
因子生成器的抽象基类
'''

from abc import ABC, abstractmethod
from typing import List, Optional
import polars as pl
class BaseFactorGenerator(ABC):
    @property
    @abstractmethod
    def factor_family(self):
        '''因子所属族类别'''
        pass

    @abstractmethod
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        '''
        核心生成入口，接收勾选的财务因子
        返回带一个或多个新因子列的 DataFrame
        :param df:
        :return:
        '''
        pass