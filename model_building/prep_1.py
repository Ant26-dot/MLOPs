from huggingface_hub import hf_hub_download
import pandas as pd

#  Ensure that all columns will be shown
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv(hf_hub_download(repo_id="lomface/Tourism-Package-Prediction", filename="tourism.csv", repo_type="dataset"))
print(f'____Getting first five rows of the dataset____')
print(df.head())
print()
print(f'____Getting last last five rows of the dataset____')
print(df.tail())
print()
print(f'____Getting the shape of the dataset____')
print(df.shape)
print()
print(f'____Getting the information of the dataset____')
print(df.info())
print()
print(f'____Getting the summary statistics of the dataset____')
print(df.describe().T)
print()
print(f'____Getting the missing values of the dataset____')
print(df.isna().sum())
