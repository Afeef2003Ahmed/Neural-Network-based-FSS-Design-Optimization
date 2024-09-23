import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/content/drive/MyDrive/FSS_6G_dataset.csv")
df = df.drop(df.columns[0], axis=1)
#df = df[(df['r_freq'] >= 1e9) & (df['r_freq'] <= 21e9)]
df = df[(df['g']<=1.5e-3) & (df['d']>=5e-3) & (df['d']<=35e-3)]
df['r_freq'] = df['r_freq']/(1e10)
df[['p','s','d','g']]= df[['p','s','d','g']]*(1e3)
df[['S11(dB)']] = df[['S11(dB)']]*(1e5)
df = df.drop(columns = df.columns[[7,8]])
print(df)

train_df = df.sample(frac=0.8, random_state=0)

test_df = df.drop(train_df.index)
#print(df)
# Convert data to NumPy arrays
train_labels = train_df[['p', 's', 'd','g']].to_numpy()
val_labels = test_df[['p', 's', 'd','g']].to_numpy()
train_features = train_df[['r_freq','S11(dB)','S21(dB)']].to_numpy()
val_features = test_df[['r_freq','S11(dB)','S21(dB)']].to_numpy()
print(f"Number of samples in train dataframe {len(train_df)}")
print(f"Number of samples in test dataframe {len(test_df)}")

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[3]),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)
])

# Compile model
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

from keras.callbacks import ModelCheckpoint


checkpoint = ModelCheckpoint("/content/drive/MyDrive/6G_model3_checkpoint.h5", save_best_only=True)


history = model.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    batch_size=64,
    epochs = 50 ,
    verbose=1,
    callbacks=[checkpoint]
)

# Evaluate model
loss = model.evaluate(val_features, val_labels, verbose=0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

print("Mean squared error on validation set: {:.3f}".format(loss))

