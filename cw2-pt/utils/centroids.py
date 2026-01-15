import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

from dataset import MalePelvicDataset

def compute_centroids(mask, class_labels):
    """ Compute centroids for each class in the mask. """
    # If 3D (ZYX), take middle slice
    if mask.ndim == 3:
        mask = mask[mask.shape[0] // 2]
    
    centroids = {}
    for label in class_labels:
        # Find all pixels with this class label
        coords = torch.where(mask == label)
        
        if len(coords[0]) > 0:  # If class exists in mask
            # Compute centroid (mean of coordinates)
            y_centroid = coords[0].float().mean().item()
            x_centroid = coords[1].float().mean().item()
            centroids[label] = (y_centroid, x_centroid)
        else:
            # If class doesn't exist, use None
            centroids[label] = None
    
    return centroids

def process_masks(file_paths, class_labels=None):
    """ Process all mask files and compute average centroid locations. """
    if class_labels is None:
        class_labels = list(range(1, 9))  # Classes 1-8
    
    # Storage for centroids across all files
    left_centroids_all = {label: [] for label in class_labels}
    right_centroids_all = {label: [] for label in class_labels}
    
    for i in range(dataset.len):
        # Load mask
        _, mask = dataset.__getitem__(i)
        
        # Handle 3D masks - take middle Z slice
        if mask.ndim == 3:
            mask = mask[mask.shape[0] // 2]
        
        height, width = mask.shape
        mid_x = width // 2
        
        # Split into left and right halves
        left_mask = mask[:, :mid_x]
        right_mask = mask[:, mid_x:]
        
        # Compute centroids for each half
        left_centroids = compute_centroids(left_mask, class_labels)
        right_centroids = compute_centroids(right_mask, class_labels)
        
        # Store centroids (adjust right coordinates by mid_x offset)
        for label in class_labels:
            if left_centroids[label] is not None:
                left_centroids_all[label].append(left_centroids[label])
            
            if right_centroids[label] is not None:
                y, x = right_centroids[label]
                right_centroids_all[label].append((y, x + mid_x))
    
    # Compute average centroids
    left_avg = {}
    right_avg = {}
    
    for label in class_labels:
        if left_centroids_all[label]:
            left_avg[label] = tuple(np.mean(left_centroids_all[label], axis=0))
        else:
            left_avg[label] = None
        
        if right_centroids_all[label]:
            right_avg[label] = tuple(np.mean(right_centroids_all[label], axis=0))
        else:
            right_avg[label] = None
    
    return left_avg, right_avg

def compute_distance_matrix(centroids, class_labels=None):
    """ Compute pairwise distances between class centroids. """
    if class_labels is None:
        class_labels = list(range(1, 9))
    
    # Extract centroid coordinates in order
    coords = []
    valid_labels = []
    
    for label in class_labels:
        if centroids[label] is not None:
            coords.append(centroids[label])
            valid_labels.append(label)
        else:
            coords.append((np.nan, np.nan))
    
    coords = np.array(coords)
    
    # Compute pairwise Euclidean distances
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    return dist_matrix

if __name__ == "__main__":

    dataset = MalePelvicDataset("dataset/train")

    class_labels = list(range(1, 9))  # Classes 1-8
    
    left_avg_centroids, right_avg_centroids = process_masks(dataset, class_labels)
    
    print("\nLeft half average centroids:")
    for label, centroid in left_avg_centroids.items():
        print(f"  Class {label}: {centroid}")
    
    print("\nRight half average centroids:")
    for label, centroid in right_avg_centroids.items():
        print(f"  Class {label}: {centroid}")
    
    left_dist_matrix = compute_distance_matrix(left_avg_centroids, class_labels)
    right_dist_matrix = compute_distance_matrix(right_avg_centroids, class_labels)
    
    print("\nLeft half distance matrix (8x8):")
    print(left_dist_matrix)
    
    print("\nRight half distance matrix (8x8):")
    print(right_dist_matrix)
    
    avg_dist_matrix = (left_dist_matrix + right_dist_matrix) / 2
    
    print("\nAverage distance matrix:")
    print(avg_dist_matrix)
    
    avg_dist_matrix_rounded = np.round(avg_dist_matrix / 10) * 10
    min_val = np.nanmin(avg_dist_matrix_rounded)
    max_val = np.nanmax(avg_dist_matrix_rounded)
    avg_dist_matrix_normalized = np.round((avg_dist_matrix_rounded - min_val) / (max_val - min_val), 2)

    print("\nAverage distance matrix quantized and normalized:")
    print(avg_dist_matrix_normalized)

    # Save to txt file
    with open("matrix.txt", "w") as f:
        for i, row in enumerate(avg_dist_matrix_normalized):
            row_str = ", ".join([f"{val:.2f}" for val in row])
            f.write(f"[{row_str}]\n")

    print("\nNormalized average distance matrix saved to: matrix.txt")