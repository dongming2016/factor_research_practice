'''
data_loaders.IndustryDailyBuilder
公共行业维度构建器
- 将 区间流水表展开为日频 Parquet
- 仅保留真实开市交易日期，压缩存储体积，供基本面因子、截面中性化与风控模块共享
'''
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Union, Mapping, Optional
import os
import polars as pl

from data_loaders.mixins import CalendarAwareMixin

class IndustryDailyBuilder(CalendarAwareMixin):
    DEFAULT_COL_MAPPING = {
        'S_INFO_WINDCODE': 'sec_id',
        'SW_IND_CODE': 'industry',
        'INDUSTRIESNAME': 'industry_name',
        'ENTRY_DT': 'start_date',
        'REMOVE_DT': 'end_date',
    }

    def __init__(self, col_mapping: Optional[Mapping[str, str]] = None) -> None:
        self.col_mapping = dict(self.DEFAULT_COL_MAPPING)
        if col_mapping:
            self.col_mapping.update(col_mapping)

    def _to_polars(self, source: Union[str, Path]) -> pl.DataFrame:
        df = pl.read_csv(source)
        renames = {k: v for k, v in self.col_mapping.items() if k in df.columns}

        return df.rename(renames)

    @staticmethod
    def _parse_date(col_name: str) -> pl.Expr:
        s = pl.col(col_name).cast(pl.Utf8).str.strip_chars()
        return (
            pl.when(s.str.contains(r'^\d{8}$'))
            .then(s.str.to_date('%Y%m%d', strict=False))
            .otherwise(
                s.str.replace_all('/', '-')
                .str.slice(0, 10)
                .str.to_date('%Y-%m-%d', strict=False)
            )
            .alias(col_name)
        )

    def build_daily_panel(self,
                          interval_source: Union[str, Path],
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None,
                          end_inclusive: bool = True,) -> pl.DataFrame:
        cal_df = self.calendar.df.select('date')
        if start_date:
            cal_df = cal_df.filter(pl.col('date') >= start_date)
        if end_date:
            cal_df = cal_df.filter(pl.col('date') <= end_date)

        max_cal_date = cal_df['date'].max()
        min_cal_date = cal_df['date'].min()
        if max_cal_date is None or min_cal_date is None:
            raise ValueError('指定区间内无有效交易日，请检查交易日历初始化范围')

        raw_df = self._to_polars(interval_source)
        clean_intervals = (
            raw_df.with_columns(
                pl.col('sec_id').cast(pl.String),
                pl.col('industry').cast(pl.String),
                pl.col('industry_name').cast(pl.String),
                self._parse_date('start_date'),
                self._parse_date('end_date')
            )
            .filter(pl.col('start_date').is_not_null() & pl.col('industry').is_not_null())
            .with_columns([
                pl.max_horizontal(pl.col('start_date'), pl.lit(min_cal_date)).alias('eff_start'), # 开始日期早于最早的日历时，取日历作为有效开始日期
                pl.min_horizontal(pl.col('end_date').fill_null(max_cal_date), pl.lit(max_cal_date)).alias('eff_end')
            ])
        )
        if not end_inclusive:
            clean_intervals = clean_intervals.with_columns((pl.col('eff_end') - pl.duration(days=1)).alias('eff_end'))

        print(clean_intervals)
        daily_df = (
            clean_intervals.filter(pl.col('eff_start') <= pl.col('eff_end'))
            .with_columns(pl.date_ranges(pl.col('eff_start'), pl.col('eff_end'), interval='1d').alias('date'))
            .explode('date')
            .join(cal_df, on=['date'], how='inner')
            .sort(['sec_id', 'date', 'start_date'])
            .unique(['sec_id', 'date'], keep='last')
            .select('sec_id', 'date', 'industry', 'industry_name')
            .sort(['sec_id', 'date'])
        )

        return daily_df

    def build_and_save(self,
                       interval_source: Union[str, Path],
                       output_path: Union[str, Path] = '../datas/industry_daily.parquet',
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None,
                       end_inclusive: bool = True
                       ) -> Path:
        daily_df = self.build_daily_panel(interval_source, start_date, end_date, end_inclusive)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out.with_suffix(f'{out.suffix}.tmp')

        daily_df.write_parquet(tmp_path, compression='zstd', statistics=True)
        os.replace(tmp_path, out)

        return out

if __name__ == '__main__':
    industry_daily_builder = IndustryDailyBuilder()
    out_path = industry_daily_builder.build_and_save(
        interval_source='../origin_datas/A_industry.csv'
    )