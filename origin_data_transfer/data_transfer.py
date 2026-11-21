from pathlib import Path
import polars as pl

def conver_trade_date_excel_to_parquet(
        excel_path:str | Path = '../datas/trade_date.xlsx',
        sheet_name:str = 'day',
        output_dir:str | Path = '../datas/',
        output_name:str = 'trading_calendar.parquet', ):
    excel_path = Path(excel_path)
    output_path = Path(output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_data = pl.read_excel(excel_path, sheet_name=sheet_name)

    date_col = next(
        (c for c in ('trade_date', 'DateTime', 'date') if c in raw_data.columns),
        raw_data.columns[0]
    )

    clean_df = (
        raw_data.select(pl.col(date_col))
        .drop_nulls()
        .with_columns(pl.col(date_col).cast(pl.Date).alias('date'))
        .select('date')
        .unique()
        .sort('date')
    )

    clean_df.write_parquet(output_path)
    return clean_df

if __name__ == '__main__':
    # 导出主日频交易日历
    cal_df = conver_trade_date_excel_to_parquet()
    print(f'已生成日频交易日历: {cal_df.shape}, 范围: {cal_df['date'].min()} - {cal_df['date'].max()}')