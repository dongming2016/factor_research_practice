import os

import rqdatac
from rqdatac import all_instruments

## 米筐接口服务相关
# 优先读环境变量；机器数超限 / 过期时依次尝试备用 license
_DEFAULT_LICENSES = [
    (
        'YlbeWQuF9V7ZzhGeIRsxv5SGphggCVhcXTKqhxAFJeDa0pq6fkP79TWdFwsbvaDUax3jIbIGAKPa3ABArXrvZsrVf8_vbIQ7F3mG3sIso0OcvIFFlItNgsKsnyFKfw_aI9urb8jcQM0eog6u-tqpwCJd9wdLgZoHwCz_JUYSFR4='
        'Do7D_z7iCrpl1bGT4mHYoU135YJ3JGz-wCvHThlreulRG-qVb27uyHJlB3xZrnvRQWg2svXxYHVPHwQrxOYAdmNInoPq0JxQvRqYR-oVtMCad7njkYft-R65W2hFiuMaPioYaaCXUr4YW2UtGZgvTw8Pp2ptZXW92nMoeKOC3SQ='
    ),
]

RQ_LICENSE = os.environ.get('RQDATAC_LICENSE', _DEFAULT_LICENSES[0])

def init_rq():
    """初始化米筐；额度超限或认证失败时自动尝试备用 license。"""
    licenses = [RQ_LICENSE] + [x for x in _DEFAULT_LICENSES if x != RQ_LICENSE]
    last_err = None
    for lic in licenses:
        try:
            rqdatac.init('license', lic)
            return
        except Exception as err:
            last_err = err
            msg = str(err).lower()
            retryable = (
                'QuotaExceeded' in type(err).__name__
                or 'quota' in msg
                or 'AuthenticationFailed' in type(err).__name__
                or 'auth failed' in msg
                or '过期' in str(err)
            )
            if not retryable:
                raise
    raise last_err

def get_cs_stocks():
    """沪深 A 股 CS（米筐 ``order_book_id``：``.XSHG`` / ``.XSHE``）。

    不含北交所 ``.BJSE``。与质量 / 价值现行 A 股输出宇宙对齐；港股仍走 ``get_hk_cs_stocks``。
    """
    stock_list = all_instruments(type='CS', market='cn', date=None) # 原始的参数type为None效率很低，改为CS后效率变高了很多，秒出
    cs = stock_list[stock_list['type'] == 'CS']
    oid = cs['order_book_id'].astype(str)
    mask = oid.str.endswith('.XSHG') | oid.str.endswith('.XSHE')
    return cs.loc[mask].copy()



init_rq()
cs_stocks = get_cs_stocks()