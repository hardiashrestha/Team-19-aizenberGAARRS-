import os
import pandas as pd
import shutil

# Define paths
image_directory = ""
spreadsheet_file = ""

# Load the spreadsheet
df = pd.read_csv(spreadsheet_file)

# Ensure your columns include 'image' and the label columns
filenames = df['image'].tolist()
label_columns = [col for col in df.columns if col != 'image']

# Create directories for each label if they do not exist
for label in label_columns:
    label_dir = os.path.join(image_directory, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Move files to their respective label directories
for index, row in df.iterrows():
    filename = row['image']
    src_path = os.path.join(image_directory, filename + ".jpg")
    print(f"Source path: {src_path}") # Add this line to print the source path
    if os.path.exists(src_path):
        for label in label_columns:
            if row[label] == 1:
                dest_path = os.path.join(image_directory, label, )
                shutil.move(src_path, dest_path)  # Use copy instead of move to handle multiple labels
    else:
        print(f"File {filename} not found in the directory.")

print("Images have been sorted into respective folders.")
