import pandas as pd

value_factor = pd.read_pickle('tmp/价值因子A+H_20261106.pkl')
growth_b_factor = pd.read_pickle('tmp/成长黑马.pkl')
growth_factor = pd.read_pickle('tmp/成长因子_四因子数据_20261106.pkl')
quality_factor = pd.read_pickle('tmp/质量因子_日频_20261106.pkl')

value_factor.to_csv('tmp/价值因子A+H_20261106.csv')
growth_b_factor.to_csv('tmp/成长黑马.csv')
growth_factor.to_csv('tmp/成长因子_四因子数据_20261106.csv')
quality_factor.to_csv('tmp/质量因子_日频_20261106.csv')


