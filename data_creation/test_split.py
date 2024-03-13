import pandas as pd
lang = 'hi_qonly'
df = pd.read_parquet(f'data/{lang}.parquet')
df.reset_index(drop=True, inplace=True)


sec1 = df.iloc[:1000].copy()

# Get 1000 entries starting from index 120000
sec2 = df.iloc[120000:121000].copy()

# Get the last 1000 entries
sec3 = df.iloc[-1000:].copy()

# Remove the entries in sec1, sec2, and sec3 from the original DataFrame
df = pd.concat([df.iloc[1000:120000], df.iloc[121000:-1000]])
test = pd.concat([sec1, sec2, sec3])

df.to_parquet(f'{lang}.parquet')
test.to_parquet(f'{lang}_test.parquet')
