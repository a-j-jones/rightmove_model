import pandas as pd
from preprocess import ModelPreprocess
import sqlite3
import re

# Database setup:
database = r"E:\_github\rightmove_api\database.db"


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None


conn = sqlite3.connect(database)
conn.create_function("REGEXP", 2, regexp)
sql = "select * from model_data"
df_import = pd.read_sql(sql, conn).set_index("property_id")

# Preprocess data:
preprocessor = ModelPreprocess(df_import)
df = preprocessor.pre_processing().reset_index()
df.to_feather("ModelData.feather")