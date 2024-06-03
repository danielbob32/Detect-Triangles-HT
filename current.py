#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, img, cmap='gray'):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image, d, sigmaColor, sigmaSpace, blur_ksize):
    bilateral_filtered = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    blurred = cv2.GaussianBlur(bilateral_filtered, (blur_ksize, blur_ksize), 0)
    return blurred

def hough_transform(edge_map, rho_res, theta_res, threshold):
    height, width = edge_map.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rhos = np.linspace(-max_rho, max_rho, int(2 * max_rho / rho_res))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / theta_res))
    accumulator = np.zeros((len(rhos), len(thetas)))

    edge_points = np.argwhere(edge_map)
    for y, x in edge_points:
        for t_index, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = np.argmin(np.abs(rhos - rho))
            accumulator[rho_index, t_index] += 1

    lines = []
    for rho_index in range(len(rhos)):
        for theta_index in range(len(thetas)):
            if accumulator[rho_index, theta_index] >= threshold:
                rho = rhos[rho_index]
                theta = thetas[theta_index]
                lines.append((rho, theta))
    
    return lines, accumulator

def non_maximum_suppression(lines, distance_threshold, angle_threshold):
    suppressed_lines = []
    for i, (rho1, theta1) in enumerate(lines):
        is_suppressed = False
        for j in range(i + 1, len(lines)):
            rho2, theta2 = lines[j]
            if abs(rho1 - rho2) < distance_threshold and abs(theta1 - theta2) < angle_threshold:
                is_suppressed = True
                break
        if not is_suppressed:
            suppressed_lines.append((rho1, theta1))
    return suppressed_lines

def draw_lines_on_edge_map(edge_map, lines, color=(255, 255, 255)):
    img = np.copy(edge_map)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Calculate the maximum distance needed to draw the lines across the entire image
    diag_len = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend line endpoints from center to beyond the edges of the image
        x1 = int(x0 + diag_len * (-b))
        y1 = int(y0 + max(diag_len * (a), 1))
        x2 = int(x0 - diag_len * (-b))
        y2 = int(y0 - max(diag_len * (a), 1))
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    
    return img

def process_image(image_path, preprocess_params, hough_params, nms_params):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
    except Exception as e:
        print(f"Error: {e}")
        return

    preprocessed = preprocess_image(image, **preprocess_params)
    display_image("Preprocessed Image", preprocessed)
    
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    edges = cv2.Canny(blurred_image, 150, 200)  # Lower thresholds to capture more edges
    # Apply morphological operations to remove small noises
    #kernel = np.ones((3, 3), np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    display_image("Edge Map", edges)

    lines, accumulator = hough_transform(edges, **hough_params)
    lines = non_maximum_suppression(lines, **nms_params)

    # Display the Hough transform parameter space
    plt.figure(figsize=(10, 10))
    accumulator = np.log(accumulator + 1)
    plt.imshow(accumulator, cmap='gray', aspect='auto')
    plt.title("Hough Transform Parameter Space")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Rho (pixels)")
    plt.axis('on')
    plt.show()

    edges_with_lines = draw_lines_on_edge_map(edges, lines, color=(255, 255, 255))
    display_image("Detected Lines on Edge Map", edges_with_lines)

# Parameters for the image of a triangle
image_path2 = 'Assets/group_sketch/sample_triangles.png'
preprocess_params2 = {'d': 3, 'sigmaColor': 75, 'sigmaSpace': 75, 'blur_ksize': 5}
hough_params2 = {'rho_res': 0.5, 'theta_res': np.pi / 180, 'threshold': 70}
nms_params2 = {'distance_threshold': 20, 'angle_threshold': np.pi / 36}

#process_image(image_path1, preprocess_params1, hough_params1, nms_params1)
process_image(image_path2, preprocess_params2, hough_params2, nms_params2)