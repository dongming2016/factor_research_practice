from typing import Optional

import polars as pl

from data_loaders.loaders import BaseDataLoader
from data_loaders.mixins import DateFilterMixin


class IndustryDataLoader(BaseDataLoader, DateFilterMixin):
    def load(self, file_name: str = 'industry_daily.parquet',
             start_date: Optional[str]=None,
             end_date: Optional[str] = None) -> pl.DataFrame:

        file_path = self.data_dir / file_name
        q = pl.scan_parquet(file_path)

        if start_date:
            q = q.filter(pl.col('date') >= pl.lit(start_date).str.to_date())
        if end_date:
            q = q.filter(pl.col('date') <= pl.lit(end_date).str.to_date())

        return self.filter_trading_days(q.collect()).sort(['sec_id', 'date'])


if __name__ == '__main__':
    from datetime import date
    data = IndustryDataLoader().load(end_date=date.today().strftime('%Y-%m-%d'))