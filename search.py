# Add references to ensure that JIT Compile functions correctly.
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"'

import pandas as pd
import tensorflow as tf
from keras import layers
# from preprocess import ModelPreprocess
from keras_tuner.tuners import BayesianOptimization
from keras_tuner.engine.hyperparameters import HyperParameters

"""
GET REFRESHED DATA FROM DATABASE
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
df = preprocessor.pre_processing()
"""

df = pd.read_feather("modelling_data.feather").set_index("property_id")

# Test / Train split:
test_data = df.sample(frac=0.9, random_state=0)
train_data = df.drop(test_data.index)

# Features / Labels split:
df = df.sample(frac=1, random_state=69420).reset_index(drop=True)
train_features = test_data.copy()
test_features = train_data.copy()
train_labels = train_features.pop('price_log')
test_labels = test_features.pop('price_log')

print(f"train_features - {train_features.shape}")
print(f"test_features - {test_features.shape}")
print(f"train_labels - {train_labels.shape}")
print(f"test_labels - {test_labels.shape}")


def build_model(hp: HyperParameters):
	# Base sequential model:
	model = tf.keras.models.Sequential([
		layers.Dense(1024, "relu"),
		layers.Dropout(hp.Float("dropout_rate", min_value=0.001, max_value=0.5, default=0.262834)),
		layers.Dense(32, "selu"),
		layers.Dense(1024, "relu"),
		layers.Dense(512, "selu"),
		layers.Dense(1)
	])

	# Compile the model:
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=hp.Float("learning_start_rate", min_value=0.00005, max_value=0.004, sampling="log",
		                               default=0.0005),
		decay_steps=10000,
		decay_rate=hp.Float("learning_decay_rate", min_value=0.1, max_value=0.99)
	)
	model.compile(
		loss='mean_absolute_error',
		optimizer=tf.keras.optimizers.Adam(lr_schedule),
		metrics=["mean_absolute_percentage_error"],
		jit_compile=True
	)

	return model


# Tensorboard setup:
log_dir = "logs/tuner_fit/"
project_name = "tuner_7_stop_early"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir + project_name, histogram_freq=1)
with open("activate_tensorboard.bat", "w") as f:
	f.write(f"tensorboard --logdir {log_dir}")

stop_early = tf.keras.callbacks.EarlyStopping(
	monitor='val_mean_absolute_percentage_error',
	patience=10
)

tuner = BayesianOptimization(
	build_model,
	objective="sk",
	max_trials=150,
	executions_per_trial=2,
	directory=log_dir,
	project_name=project_name
)

tuner.search(
	x=train_features,
	y=train_labels,
	epochs=150,
	batch_size=64,
	validation_data=(test_features, test_labels),
	callbacks=[tensorboard_callback, stop_early]
)

tuner.save_model(trial_id=133)
