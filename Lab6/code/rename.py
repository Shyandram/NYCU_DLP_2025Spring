import os

# Set the directory path where the images are located
directory = r'images\new test'  # Change this to your folder path

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.startswith("final_new_test_") and filename.endswith(".png"):
        # Extract the number from the filename
        number = filename.replace("final_new_test_", "").replace(".png", "")
        new_name = f"{number}.png"
        
        # Full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
