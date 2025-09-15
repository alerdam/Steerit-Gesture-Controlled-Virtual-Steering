import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import ds_prepFunctions

def MODEL_OneLayer():
    model = keras.models.Sequential([
    keras.layers.Input(shape=(8,)),  
    keras.layers.Dense(8, activation='tanh'), 
    keras.layers.Dense(units=1, activation='tanh') 
])
    optmzr = keras.optimizers.Adam(learning_rate=0.0001)
    lossfunc =  keras.losses.MeanAbsoluteError()

    # Model Options
    model.compile(optimizer=optmzr, loss = lossfunc, metrics=["mae"])
    return model

model = MODEL_OneLayer()
model.summary()

# Load Data
loaded_data = ds_prepFunctions.load_csv_to_numpy("DATA/DATABASE.csv")
if loaded_data is not None:
    xtrain, ytrain, xtest, ytest =ds_prepFunctions.Explore_DS(loaded_data)
    if xtrain is not None:
        print(" -------- DATA --------")
        print("xtrain shape:", xtrain.shape)
        print("ytrain shape:", ytrain.shape)
        print("xtest shape:", xtest.shape)
        print("ytest shape:", ytest.shape)
        print( "----------------------------------------------")


# Train the model
history = model.fit(xtrain, ytrain,batch_size=8, epochs=100)  

# Evaluate the model
print("")
print( "----------------------------------------------")
print("TEST RESULTS : OneLayer")
model.evaluate(xtest, ytest)

# Save the model 
model.save("M_OneLayer.keras")
print("Model created and saved")


