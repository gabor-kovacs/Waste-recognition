import torch
from options.base_options import BaseOptions
import utils.utils as utils
import segmentation_models_pytorch as smp
import os
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.spatial import KDTree

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=5
    ).to(device)

    model.load_state_dict(torch.load(os.path.join("model", "model.pth")))

    transform = transforms.Compose(
        [
            # transforms.Resize((1280, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3582850575382683, 0.21850878253252187, 0.22273015377280164],
                std=[0.1959599683893488, 0.1373912334838446, 0.07926674805856508],
            ),
        ]
    )

    original_image = cv2.imread("filippo.png")
    input_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Define individual border widths
    border_left = 128
    border_top = 96
    border_right = 0
    border_bottom = 32

    # Retrieve the dimensions of the original image
    h, w, _ = input_image.shape

    # Calculate the cropping coordinates
    crop_top = border_top
    crop_bottom = h - border_bottom
    crop_left = border_left
    crop_right = w - border_right

    # Crop the image using the specified borders
    cropped_image = input_image[crop_top:crop_bottom, crop_left:crop_right]

    pil_image = Image.fromarray(cropped_image)

    input_tensor = transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    # Run the model
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to segmentation map
    output_predictions = output.argmax(1).squeeze(0).cpu().numpy()

    # resized_output_predictions = cv2.resize(
    #     output_predictions,
    #     (original_image.shape[1], original_image.shape[0]),
    #     interpolation=cv2.INTER_NEAREST,
    # )

    class_dict = {"grass": 0, "obstacle": 1, "road": 2, "trash": 3, "vegetation": 4}
    cmap = ListedColormap(["green", "red", "blue", "gray", "darkgreen"])

    # Plotting both original and segmented image
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(input_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    segm_image = ax[1].imshow(output_predictions, cmap=cmap, interpolation="nearest")
    ax[1].set_title("Segmentation Map with Class Colors")
    ax[1].axis("off")

    # Create a legend for the segmentation map
    colors = [segm_image.cmap(segm_image.norm(value)) for value in class_dict.values()]
    labels = list(class_dict.keys())
    patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=10,
        )
        for color, label in zip(colors, labels)
    ]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # plt.show()

    # trash_mask = (resized_output_predictions == class_dict["trash"]).astype(np.uint8)
    trash_obstacle_mask = (
        (output_predictions == class_dict["trash"])
        | (output_predictions == class_dict["obstacle"])
    ).astype(np.uint8)
    # Finding connected components
    num_labels, labels_im = cv2.connectedComponents(trash_obstacle_mask)

    def find_centroids_and_points(label_image, min_pixels=100, additional_points=4):
        centroids = []
        for label in range(1, num_labels):  # Starting from 1 to skip the background
            mask = (label_image == label).astype(np.uint8)
            area = np.sum(mask)
            if area > 0:
                M = cv2.moments(mask, binaryImage=True)
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                centroids.append((centroid_x, centroid_y))

                if area > min_pixels:
                    # Optionally find additional points by subdividing the area
                    # This example just uses evenly spaced points in the x-dimension
                    x_coords, y_coords = np.where(mask == 1)
                    x_sorted = np.sort(np.unique(x_coords))
                    split = np.array_split(
                        x_sorted, additional_points + 1
                    )  # +1 to account for the main centroid
                    for section in split:
                        if len(section) > 0:
                            sec_mask = np.isin(x_coords, section)
                            add_x = int(np.mean(x_coords[sec_mask]))
                            add_y = int(np.mean(y_coords[sec_mask]))
                            centroids.append((add_x, add_y))

        return centroids

    def find_centroids(label_image):
        centroids = []
        for label in range(1, num_labels):  # Start from 1 to skip the background
            mask = (label_image == label).astype(np.uint8)
            area = np.sum(mask)
            if area > 0:
                # Calculate moments for the current mask
                M = cv2.moments(mask, binaryImage=True)
                # Compute centroid coordinates
                centroid_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                centroid_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                # Append centroid if it is valid
                centroids.append((centroid_x, centroid_y))

        return centroids

    # Get centroids and additional points for large "trash" regions
    trash_centroids = find_centroids(labels_im)
    # trash_centroids = find_centroids_and_points(labels_im)

    filtered_centroids = [
        (x, y)
        for x, y in trash_centroids
        if 0 <= x < pil_image.size[0] and 0 <= y < pil_image.size[1]
    ]

    centroids = np.array(filtered_centroids)  # Convert list to numpy array for KDTree

    # Create a KDTree for fast spatial look-up
    tree = KDTree(centroids)

    # To keep track of which points have been merged
    visited = np.zeros(centroids.shape[0], dtype=bool)
    new_centroids = []

    # Merge points within 5 pixels
    for i in range(centroids.shape[0]):
        if not visited[i]:
            # Find all points within 5 pixels of the current point
            indices = tree.query_ball_point(centroids[i], 100)
            # Mark these points as visited
            visited[indices] = True
            # Calculate the centroid of these points
            cluster_centroid = centroids[indices].mean(axis=0)
            new_centroids.append(cluster_centroid)

    # # Convert new_centroids to a numpy array for plotting
    # new_centroids = np.array(new_centroids)

    adjusted_centroids = [(x + border_left, y + border_top) for x, y in trash_centroids]

    print("Trash centroids:", adjusted_centroids)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(input_image[:, :, 2])
    ax.set_title("Original Image with Centroids of 'Trash' Areas")
    ax.axis("off")

    # Plot each centroid
    for x, y in adjusted_centroids:
        ax.scatter(x, y, c="red", s=50)  # red circle with size 50

    plt.show()
