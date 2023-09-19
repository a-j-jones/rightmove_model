import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler


class ModelPreprocess:
	def __init__(self, df):
		self.scalers = {}
		self.df = df
		self.df_processed = None
		self.columns = None

	def pre_processing(self):
		# Remove outliers:
		df = (self.df
		      .copy()
		      .query("bedrooms < 15")
		      .query("bathrooms < 5")
		      .query("days_old < 700")
		      .query("price_amount < 10_000_000")
		      .query("price_amount > 50_000")
		      )

		# Create dummies for Property Type:
		subtype_dummies = pd.get_dummies(df.type)
		df = df.join(subtype_dummies)

		# Create additional features:
		df["semi_detached"] = df.subtype.str.contains("Semi-Detached").astype(bool).astype(float)
		df["detached"] = np.where(df.semi_detached, 0, df.subtype.str.contains("Detached")).astype(bool).astype(float)
		df["terraced"] = df.subtype.str.contains("Terraced").astype(bool).astype(float)
		df = df.dropna(subset=["price_amount"])
		df = df.fillna(0)

		# Create log values:
		df["days_log"] = np.log(np.where(df.days_old == 0, 1, df.days_old))
		df["price_log"] = np.log(df.price_amount)

		# Drop columns not required:
		df = df.drop(columns=["type", "subtype", "price_amount", "days_old"])

		# Normalise & Scale data:
		for col in df.columns:
			if col == "price_log":
				continue
			values = df[col].values.reshape(-1, 1)
			self.scalers[col] = StandardScaler().fit(values)
			values_scaled = self.scalers[col].transform(values)
			df[col] = values_scaled

		# Final columns to be expected in the model data:
		df = df[[
			'bedrooms',
			'bathrooms',
			'latitude',
			'longitude',
			'bungalow',
			'flat',
			'house',
			'other',
			'semi_detached',
			'detached',
			'terraced',
			'days_log',
			'crime',
			'price_log'
		]]

		self.df_processed = df.copy()
		self.columns = df.columns

		return self.df_processed

	def reverse_preprocess(self, features, labels):
		df = features.copy()
		df["price_log"] = labels

		for col in df.columns:
			if col == "price_log":
				continue
			values = df[col].values.reshape(-1, 1)
			values_descaled = self.scalers[col].inverse_transform(values)
			df[col] = values_descaled

		df["price_amount"] = np.exp(df.price_log)

		return df.copy()


