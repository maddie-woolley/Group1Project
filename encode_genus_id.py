import pandas
from sklearn.preprocessing import LabelEncoder

df = pandas.read_csv("data/observations/observations_us_train.csv")

label_encoder = LabelEncoder()
df["genus_id"] = label_encoder.fit_transform(df["genus"])

df.to_csv("data/observations/observations_us_train.csv", index=False)