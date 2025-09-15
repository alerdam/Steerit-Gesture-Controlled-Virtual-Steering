import numpy as np
import os



"""
Load data
"""
slopes_connection_lines_path = "DATA/slopes_connection_lines.npy"
slopes_knuckle_lines_path = "DATA/slopes_knuckle_lines.npy"
y_diff_array_path = "DATA\y_diff_array.npy"
label_path = "DATA/label_data.npy"

try:
    slopes_connection_lines = np.load(slopes_connection_lines_path)
    slopes_knuckle_lines = np.load(slopes_knuckle_lines_path)
    y_diff_array = np.load(y_diff_array_path)
    label_array = np.load(label_path)

    # Print the loaded arrays (or do whatever you need with them)
    print("slopes_connection_lines:")
    print("Shape:", slopes_connection_lines.shape)

    print("\nslopes_knuckle_lines:")
    print("Shape:", slopes_knuckle_lines.shape)

    print("\ny_diff_array")
    print("Shape:", y_diff_array.shape)

    print("\nLABELS")
    print(f"Shape: {label_array.shape}")
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"An error occurred: {e}")



# Stack them along columns
final_array = np.column_stack((slopes_connection_lines, slopes_knuckle_lines, y_diff_array, label_array))
print(f"\nDatastore \nShape: {final_array.shape}")

column_names = "slopes_connection_lines_1, slopes_connection_lines_2, slopes_connection_lines_3, slopes_connection_lines_4, slopes_connection_lines_5, slopes_knuckle_lines_1, slopes_knuckle_lines_2, y_diff_array, label_array"

# Save to CSV
np.savetxt("DATA/DS_TEMP.csv", final_array, delimiter=",", fmt="%.2f", header=column_names, comments='')

print("CSV file saved successfully with column headers!")

# --- Code for saving to DATABASE.csv with appending ---

database_csv_path = "DATA/DATABASE.csv"

# Check if DATABASE.csv already exists
file_exists = os.path.exists(database_csv_path)

if not file_exists or os.stat(database_csv_path).st_size == 0:
    # If the file doesn't exist or is empty, write with header
    np.savetxt(database_csv_path, final_array, delimiter=",", fmt="%.2f", header=column_names, comments='')
    print(f"CSV file saved successfully as {database_csv_path} with header.")
else:
    with open(database_csv_path, 'ab') as f:
        np.savetxt(f, final_array, delimiter=",", fmt="%.2f", comments='')
    print(f"Data successfully appended to {database_csv_path}.")