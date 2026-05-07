import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# 1. 讀取資料
df = pd.read_csv('Data/raw/YRBS_2007.csv')

# 2. 資料清理與重編碼
# 定義性別 (1=Male, 2=Female, 根據 YRBS 定義)
# 定義飲酒 (Success=1 for codes 2-7, Failure=0 for code 1)
df['alcohol_binary'] = np.where(df['CurrentAlcoholUse'].isin([2,3,4,5,6,7]), 1, 
                                np.where(df['CurrentAlcoholUse'] == 1, 0, np.nan))

# 移除缺失值
clean_df = df.dropna(subset=['WhatIsYourSex', 'alcohol_binary'])

# 3. 計算分組摘要
summary = clean_df.groupby('WhatIsYourSex')['alcohol_binary'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total_N', 'Alcohol_Users', 'Proportion']
print(summary)

# 4. 執行 Two-proportion z-test
count = summary['Alcohol_Users'].values
nobs = summary['Total_N'].values
stat, pval = proportions_ztest(count, nobs)

print(f'Z-statistic: {stat:.4f}')
print(f'P-value: {pval:.4f}')
