
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.pandas.set_option('display.max_columns', None)

df = pd.read_csv("datasets/persona.csv")

# Dataset overview
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.values

# Unique number of sources and frequencies
df["SOURCE"].value_counts()

# Number of unique prices
df["PRICE"].nunique()

# How many sales were made from which PRICE?
df.groupby("PRICE").agg({"count"})
df.loc[:].groupby("PRICE").agg({"count"})

# How many sales from which country?
df.loc[:, ["PRICE", "COUNTRY"]].groupby("COUNTRY").agg({"count"})

# How much was earned in total from sales by country?
df.loc[:, ["PRICE", "COUNTRY"]].groupby("COUNTRY").agg({"sum"})

# What are the sales numbers according to SOURCE types?
df.loc[:, ["SOURCE", "PRICE"]].groupby("SOURCE").agg({"count"})

# What are the PRICE averages by country?
df.loc[:, ["PRICE", "COUNTRY"]].groupby("COUNTRY").agg({"mean"})

# What are the PRICE averages by SOURCE?
df.loc[:, ["PRICE", "SOURCE"]].groupby("SOURCE").agg({"mean"})

# What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.loc[:, ["COUNTRY", "SOURCE", "PRICE"]].groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# What are the total gains broken down by COUNTRY, SOURCE, SEX, AGE?
df.loc[:, ["COUNTRY", "SOURCE", "PRICE", "AGE", "SEX"]].groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "sum"})

# Sorting the output and saving as agg_df
df.loc[:, ["COUNTRY", "SOURCE", "PRICE", "AGE", "SEX"]].groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "sum"}).sort_values(["PRICE"], ascending=False)

agg_df = df.loc[:, ["COUNTRY", "SOURCE", "PRICE", "AGE", "SEX"]].groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "sum"}).sort_values(["PRICE"], ascending=False)

agg_df.head()

# Converting index names to variable names
agg_df = agg_df.reset_index()
agg_df.head()

# Converting the age variable to a categorical variable and adding it to agg_df
df["AGE"].max()
# '0_19', '20_24', '24_31', '31_41', '41_70
pd.cut(agg_df["AGE"], [0, 20, 24, 31, 41, 70], labels=["0_19", "20_24", "24_31", "31_41", "41_70"])

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 19, 24, 31, 41, 70], labels=["0_19", "20_24", "24_31", "31_41", "41_70"])

agg_df.head()
agg_df.tail()


# Creation of level-based customers (personas)
agg_df["customers_level_based"] = [x.upper()+"_"+y.upper()+"_"+z.upper()+"_"+str(t) for x, y, z, t in zip(agg_df['COUNTRY'], agg_df['SOURCE'], agg_df['SEX'], agg_df["AGE_CAT"])]
agg_df = agg_df.drop(columns=["COUNTRY", "SOURCE", "SEX", "AGE", "AGE_CAT"], axis=1)
columns_titles = ["customers_level_based", "PRICE"]
agg_df = agg_df.reindex(columns=columns_titles)

# Singularization
agg_df = agg_df.loc[:, ["customers_level_based", "PRICE"]].groupby(["customers_level_based"]).agg({"PRICE": "mean"}).sort_values(["PRICE"], ascending=False)
agg_df = agg_df.loc[:, ["customers_level_based", "PRICE"]].groupby(["customers_level_based"]).agg({"PRICE": "mean"})
agg_df.head()

# Segmentation of new customers (persona)
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.tail()

# Description of segments
agg_df.loc[:, ["PRICE", "SEGMENT"]].groupby(["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})
agg_df.pivot_table(values="PRICE", index="SEGMENT", aggfunc=["mean", "max", "sum"])


# Analysis of the C segment
agg_df["SEGMENT"] = "C"
agg_df["SEGMENT"].value_counts()
agg_df.pivot_table(values="PRICE", index="SEGMENT", aggfunc=["mean", "max", "sum"])

"""
-- What segment does a 33-year-old Turkish woman using ANDROID belong to and <br>how much income is expected to earn on average?
"""
agg_df.head()
agg_df = agg_df.reset_index()

new_user = "TUR_ANDROID_FEMALE_31_41"
agg_df[agg_df["customers_level_based"] == new_user]

# In which segment and on average how <br>much income would a 35-year-old French woman using iOS expect to earn?
new_user2 = "FRA_IOS_FEMALE_31_41"
agg_df[agg_df["customers_level_based"] == new_user2]


