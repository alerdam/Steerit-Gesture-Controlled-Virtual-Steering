import tensorflow as tf
from tensorflow import keras
import numpy as np
from ModelMakers import ds_prepFunctions
import matplotlib.pyplot as plt

# Load the model
path_model = "M_trained.keras"
model = keras.models.load_model(path_model)
model.summary()

# Load Data
loaded_data = ds_prepFunctions.load_csv_to_numpy("DATA/DS_TEMP.csv")
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
model.fit(xtrain, ytrain,batch_size=8, epochs=30)  

# For use Connection Lines only
# model.fit(xtrain[:, :5], ytrain,batch_size=8, epochs=30)  

# Evaluate the model
print("")
print( "----------------------------------------------")
print("TEST RESULTS : Training Session")
model.evaluate(xtest, ytest)


# Save the model 
output_path = "M_trained.keras"
model.save("M_trained.keras") 
print(f"{output_path} updated")