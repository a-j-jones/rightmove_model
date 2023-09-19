import re
import sqlite3
import tkinter as tk

import keras
import numpy as np
import pandas as pd
from selenium import webdriver

from preprocess import ModelPreprocess


def regexp(expr, item):
	"""
	Regex function for sqlite3
	"""
	reg = re.compile(expr)
	return reg.search(item) is not None


def create_dataframe():
	# Download data from the Database:
	print("Downloading data...")
	sql = """
	select a.* from 
	model_data as a
	left join manual_exclusions as b 
		on a.property_id=b.property_id
		where b.property_id is null"""
	df_import = pd.read_sql(sql, conn).set_index("property_id")

	# Preprocess data:
	print("Processing data...")
	preprocessor = ModelPreprocess(df_import)
	df = preprocessor.pre_processing()
	features = df.copy()
	labels = features.pop('price_log')

	# Load the model from checkpoint:
	print("Loading model...")
	filepath = "../models/20221112-011441/models/epoch30"
	model = keras.models.load_model(filepath)

	# Create price predictions, and create final DataFrame:
	print("Create prediction DataFrame...")
	labels_predicted = model.predict(features)
	dfp = preprocessor.reverse_preprocess(features, labels)
	dfp["price_predict"] = np.exp(labels_predicted)
	dfp["price_residual"] = np.abs(dfp.price_predict-dfp.price_amount)
	dfp["price_residual_pct"] = dfp.price_residual / dfp.price_amount
	dfp = dfp.query("price_predict > price_amount").sort_values("price_residual_pct", ascending=False)
	dfp = dfp[[col for col in dfp.columns if "price" in col]]

	return dfp


class ToolWindow:
	def __init__(self, df, conn):
		# Get ID list:
		self.df = df
		self.id_list = (x for x in self.df.index)

		# Set up selenium driver:
		self.driver = webdriver.Chrome()

		# Set up database connection:
		self.conn = conn
		self.cursor = self.conn.cursor()

		# Create main window:
		self.root = tk.Tk()
		self.root.title("Check Rightmove properties")
		self.root.geometry("400x300")

		# Create an inner frame:
		self.frame = tk.Frame(self.root, borderwidth=25)
		self.frame.pack(fill="both", expand=True)

		# Text box for folder path:
		self.reason_label_text = tk.StringVar()
		self.reason_label_text.set("")
		self.reason_label = tk.Label(self.frame, textvariable=self.reason_label_text)
		self.reason_label.grid(column=0, row=0, sticky=tk.W)
		self.reason_text = tk.Text(self.frame, height=1, width=30, wrap="none")
		self.reason_text.grid(column=0, row=1, columnspan=5, sticky=tk.W, padx=(0, 15), pady=(0, 10))

		# Submit:
		self.check_button = tk.Button(self.frame, height=1, width=10, text="Include", command=self.checked)
		self.check_button.grid(column=0, row=2, columnspan=4, pady=(0, 10))

		# Submit:
		self.exclude_button = tk.Button(self.frame, height=1, width=10, text="Exclude", command=self.excluded)
		self.exclude_button.grid(column=1, row=2, columnspan=4, pady=(0, 10))

		# Setup page and gui status:
		self.next_id()
		self.get_url()
		self.root.mainloop()


	def next_id(self):
		self.id = next(self.id_list)
		price = self.df.loc[self.id]["price_predict"]
		self.reason_label_text.set(f"Property ID {self.id} - Price Prediction: £{price:,.0f}")
		self.reason_text.delete("1.0", "end")

	def get_url(self):
		url = f"https://www.rightmove.co.uk/properties/{self.id}#/?channel=RES_BUY"
		self.driver.get(url)

	def checked(self):
		text = self.reason_text.get("1.0", tk.END)
		self.sql_submit(id=self.id, text=text, exclude=0)
		self.next_id()
		self.get_url()

	def excluded(self):
		text = self.reason_text.get("1.0", tk.END)
		self.sql_submit(id=self.id, text=text, exclude=1)
		self.next_id()
		self.get_url()

	def sql_submit(self, id, text, exclude):
		sql = f"""
		INSERT INTO manual_exclusions 
		(property_id, comment, exclude)
		VALUES
		({id}, '{text}', {exclude})
		"""

		self.cursor.execute(sql)
		self.conn.commit()


# Database setup:
print("Setting up database connection...")
database = r"E:\_github\rightmove_api\database.db"
conn = sqlite3.connect(database)
conn.create_function("REGEXP", 2, regexp)

dfp = create_dataframe()
window = ToolWindow(df=dfp, conn=conn)

while True:
	pass