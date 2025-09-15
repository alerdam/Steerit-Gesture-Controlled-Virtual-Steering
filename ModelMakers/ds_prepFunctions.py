import numpy as np
def Explore_DS(data):
    """Splits a NumPy array into train/test and returns the split data."""
    if data is not None:
        print("Data shape:", data.shape)

        xdata = data[:, :8]
        ydata = data[:, 8]
        splitVal = int(len(xdata) * 0.8)  # 80% train, 20% test
        xtrain = xdata[:splitVal, :]
        xtest = xdata[splitVal:, :]
        ytrain = ydata[:splitVal]
        ytest = ydata[splitVal:]
        return xtrain, ytrain, xtest, ytest
    else:
        return None, None, None, None  

def load_csv_to_numpy(filename=""):
    """Loads a CSV file into a NumPy array."""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)  # skip header row
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


"""
# Load Data
loaded_data = load_csv_to_numpy("DS_TEMP.csv")
if loaded_data is not None:
    xtrain, ytrain, xtest, ytest =Explore_DS(loaded_data)
    if xtrain is not None:
        print("xtrain shape:", xtrain.shape)
        print("ytrain shape:", ytrain.shape)
        print("xtest shape:", xtest.shape)
        print("ytest shape:", ytest.shape)
"""
