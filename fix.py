import pandas

adjusted_df = pandas.read_csv("data/observations.csv")
original_df = pandas.read_csv("../geolife/observations/observations_us_train.csv", sep=';')

adjusted_df = adjusted_df.join(original_df.set_index("observation_id"), on="observation_id", lsuffix="DROP").filter(regex="^(?!.*DROP)")
adjusted_df = adjusted_df.drop(columns=["Unnamed: 0"])

adjusted_df.to_csv("data/observations.csv", index=False)