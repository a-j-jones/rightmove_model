import pandas as pd
import re
import glob

# Get all image paths and Property IDs from downloaded images:
images = glob.glob("images/*.jpg")
df = pd.DataFrame(pd.Series(images), columns=["image_location"])
df["property_id"] = df.image_location.apply(lambda x: re.search(r'\\(\d{8})_', x).group(1)).astype(int)
print(f"{len(df):,.0f} potential images found")

# Get model data from Feather file:
df_model = pd.read_feather("ModelData.feather").set_index("property_id")

# Merge model data and exclude properties which are not in the model output:
df = pd.merge(df, df_model["price_log"], how="left", left_on="property_id", right_index=True)
df = df[~df.price_log.isna()]
df = df.rename(columns={"price_log": "price"})
print(f"{len(df):,.0f} images remaining after checking model data")

# Save feather:
df.reset_index().to_feather("ModelDataImage.feather")
df[:100].to_csv("ModelDataTest.csv")

