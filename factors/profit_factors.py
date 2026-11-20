from utils.rq_util import *
from rqdatac import get_factor

def get_cfoa():
    cs_stocks_list = cs_stocks.order_book_id.to_list()
    cfoa = get_factor(cs_stocks_list, 'cash_flow_from_operating_activities_ttm_0', start_date='2009-12-31', end_date='2026-12-31')
    total_asset = get_factor(cs_stocks_list, 'total_assets', start_date='2009-12-31', end_date='2026-12-31')

    cfoa.dropna(inplace=True)
    total_asset.dropna(inplace=True)
    return cfoa, total_asset

if __name__ == '__main__':
    cfoa, total_asset = get_cfoa()
    print(cfoa)
    print(total_asset)