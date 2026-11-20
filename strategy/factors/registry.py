'''
支持命名空间强制约束与领域属性自动校验的因子注册中心
'''

import inspect
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type
from strategy.factors.base_factor import BaseFactorGenerator

class FactorRegistry:
    # 统一命名格式： domain.factor_name
    NAMESPACE_PATTERN = re.compile('^[a-z0-9_]+\\.[a-z0-9_]+$')

    # 规范领域标识（前缀 -> 官方家族全称）
    # 允许领域使用标准简写， 也可以直接使用全称小谢
    VALID_DOMAIN_MAPPING: ClassVar[Dict[str, str]] = {
        'val': 'Valuation',
        'valuation': 'Valuation'
    }

    _registry: ClassVar[Dict[str, BaseFactorGenerator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseFactorGenerator]], Type[BaseFactorGenerator]]:
        '''
        强类型与命名空间装饰器
        用法：
            @FacorRegistry.register('val.cross_section')
            class ValuationCrossSection(BaseFactorGenerator):
        :param name:
        :return:
        '''
        def decorator(subcls: Type[BaseFactorGenerator]) -> Type[BaseFactorGenerator]:
            # 1. 继承检查
            if not inspect.isclass(subcls) or not issubclass(subcls, BaseFactorGenerator):
                raise TypeError(
                    f'注册失败：目标类"{subcls}"必须继承自 BaseFactorGenerator'
                )

            # 2. 格式校验：必须符合 'domain.factor_name'
            raw_key = name.strip()
            if not cls.NAMESPACE_PATTERN.match(raw_key):
                raise TypeError(
                    f'注册名称"{raw_key}" 格式不合法'
                    f'必须严格遵循 "domain.factor_name" 格式（仅小写字母、数字与下划线），例如 "val.cross_section"'
                )

            domain, factor_name = raw_key.split('.', 1)

            # 3. 校验 domain 是否支持
            if domain not in cls.VALID_DOMAIN_MAPPING:
                valid_domains = sorted(list(cls.VALID_DOMAIN_MAPPING.keys()))
                raise TypeError(
                    f'未知领域前缀 "{domain}", 支持的领域前缀如下：{valid_domains}'
                )

            # 4. 校验 factor_family 类属性是否存在
            family = getattr(subcls, 'factor_family', None)
            if not family or not isinstance(family, str) or not family.strip():
                raise AttributeError(
                    f'类 "{subcls.__name__}" 必须显示定义非空的类属性 "factor_family: str"'
                )

            # 5. 校验 domain 与 factor_family 的一致性
            expected_family = cls.VALID_DOMAIN_MAPPING[domain]
            if family.strip() != expected_family.lower():
                raise ValueError(
                    f'类 "{subcls.__name__}" 的 factor_family: "{family.lower()}" 与注册前缀 "{domain}" '
                    f'推断出的官方领域 "{expected_family.lower()}" 不匹配！'
                )

            # 6. 重复注册冲突检测
            key = raw_key.lower()
            if key in cls._registry:
                existing_cls = cls._registry[key]
                if existing_cls != subcls:
                    raise ValueError(
                        f'因子键名 "{key}" 已被 "{existing_cls.__module__}.{existing_cls.__name__}"注册.'
                        f'无法无予 "{subcls.__module__}.{subcls.__name__}"'
                    )

            cls._registry[key] = subcls
            return subcls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseFactorGenerator]:
        key = name.lower().strip()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f'未找到因子生成器 "{key}"。当前已注册：{available}')

        return cls._registry[key]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Type[BaseFactorGenerator]:
        gen_cls = cls.get(name)

        return gen_cls(**kwargs)

    @classmethod
    def list_factors(cls, domain: Optional[str] = None) -> List[str]:
        '''
        支持全量列举，或指定领域前缀过滤
        :param domain:
        :return:
        '''
        if domain is None:
            return sorted(list(cls._registry.keys()))

        prefix = f'{domain.lower().strip()}'
        return sorted([k for k in cls._registry if k.startswith(prefix)])

    @classmethod
    def clear(cls) -> None:
        '''单元测试专用的状态重置方法'''
        cls._registry.clear()