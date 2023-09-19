# Add references to ensure that JIT Compile functions correctly.
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"'

import datetime as dt
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard

physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)

df = pd.read_feather("modelling_data.feather").set_index("property_id")

# Test / Train split:
train_data = df.sample(frac=0.8, random_state=0)
test_data = df.drop(train_data.index)

# Features / Labels split:
df = df.sample(frac=1, random_state=69420).reset_index(drop=True)
train_features = train_data.copy()
test_features = test_data.copy()
train_labels = train_features.pop('price_log')
test_labels = test_features.pop('price_log')


def build_model():
	# Base sequential model:
	model = tf.keras.models.Sequential([
		layers.Dense(1024, "relu"),
		layers.Dropout(0.07541912828345232),
		layers.Dense(32, "selu"),
		layers.Dense(1024, "relu"),
		layers.Dense(512, "selu"),
		layers.Dense(1)
	])

	# Compile the model:
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=0.0011908155324161704,
		decay_steps=10000,
		decay_rate=0.47646395150277243
	)
	model.compile(
		loss='mean_absolute_error',
		optimizer=tf.keras.optimizers.Adam(lr_schedule),
		metrics=["mean_absolute_percentage_error"],
		jit_compile=True
	)

	return model


# Tensorboard setup:
directory = f"models/{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(
	log_dir=f"{directory}/logs",
	histogram_freq=1
)

model_checkpoint = ModelCheckpoint(
	filepath=directory + "/models/epoch{epoch}",
	monitor="val_mean_absolute_percentage_error",
	verbose=1,
	save_best_only=True,
	mode="min"
)


model = build_model()
model.fit(
	train_features,
	train_labels,
	epochs=200,
	batch_size=64,
	validation_data=(test_features, test_labels),
	use_multiprocessing=True,
	callbacks=[tensorboard_callback, model_checkpoint]
)
