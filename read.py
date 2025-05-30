import cv2
import os
import glob

# Path to the folder containing images
folder_path = "../0148"

# Pattern to match filenames in the specified range
pattern = "IMG_240412_??????_????_GRE.TIF"

# Use glob to find matching filenames
file_paths = glob.glob(os.path.join(folder_path, pattern))

# Sort the filenames to ensure they are in the correct order
file_paths = sorted(file_paths)


print(file_paths)


# Function to check if the filename is within the required range
def is_within_range(filename, start, end):
    return start <= filename <= end


# Start and end filenames
start_filename = "IMG_240412_084417_0000_GRE.TIF"
end_filename = "IMG_240412_085218_0267_GRE.TIF"

# Filter filenames based on the start and end
file_paths = [
    f
    for f in file_paths
    if is_within_range(os.path.basename(f), start_filename, end_filename)
]

# Iterate over each file and read it
for file_path in file_paths:
    # Read the image using OpenCV
    image = cv2.imread(file_path)
    if image is not None:
        # Perform your processing here
        print(f"Processing {os.path.basename(file_path)}")
        cv2.imshow("Image", image)
        cv2.waitKey(1)  # Display each image for a short time
    else:
        print(f"Failed to load image {os.path.basename(file_path)}")

cv2.destroyAllWindows()
