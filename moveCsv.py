import os
import shutil

# Your designated folder
data_dir = "synth_data_collected"

# Your working directory
workdir = "exp"

# Traverse all subdirectories under the working directory
for dirpath, dirnames, filenames in os.walk(workdir):
    # Check if the current directory is the 'checked' directory
    print(f"Moving {dirpath}")
    if os.path.basename(dirpath) == 'check':
        # Go through each file in the 'checked' directory
        for filename in filenames:
            # Check if the file is a .csv file
            if filename.endswith('.csv'):
                # Construct the full original file path
                original_file_path = os.path.join(dirpath, filename)
                # Construct the new file path
                new_file_name = "TabDDPM_" + filename
                new_file_path = os.path.join(data_dir, new_file_name)
                if os.path.exists(new_file_path):
                    continue
                # Copy the file to the new location with the new name
                shutil.copy(original_file_path, new_file_path)
