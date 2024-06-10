#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Display an image with a title and specified colormap.
def display_image(title, img, cmap='gray'):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Preprocess the image by applying bilateral filtering followed by Gaussian blurring.
def preprocess_image(image, d, sigmaColor, sigmaSpace, blur_ksize):
    bilateral_filtered = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    blurred = cv2.GaussianBlur(bilateral_filtered, (blur_ksize, blur_ksize), 0)
    return blurred

# Implement the Hough Transform to detect lines in the edge map.
def hough_transform(edge_map, rho_res, theta_res, threshold):
    height, width = edge_map.shape
    max_rho = int(np.sqrt(height**2 + width**2))  # The maximum rho value
    rhos = np.linspace(-max_rho, max_rho, int(2 * max_rho / rho_res))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / theta_res))
    
    # Initialize the Hough accumulator array.
    accumulator = np.zeros((len(rhos), len(thetas)))

    # Accumulate votes in the Hough accumulator.
    edge_points = np.argwhere(edge_map)
    for y, x in edge_points:
        for t_index, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = np.argmin(np.abs(rhos - rho))
            accumulator[rho_index, t_index] += 1

    # Extract lines based on the threshold.
    lines = []
    for rho_index in range(len(rhos)):
        for theta_index in range(len(thetas)):
            if accumulator[rho_index, theta_index] >= threshold:
                rho = rhos[rho_index]
                theta = thetas[theta_index]
                lines.append((rho, theta))
    
    return lines, accumulator

# Apply Non-Maximum Suppression to filter out less significant lines.
def non_maximum_suppression(lines, distance_threshold, angle_threshold, important_rho_range):
    suppressed_lines = []
    for i, (rho1, theta1) in enumerate(lines):
        is_suppressed = False
        for j in range(i + 1, len(lines)):
            rho2, theta2 = lines[j]
            if abs(rho1 - rho2) < distance_threshold and abs(theta1 - theta2) < angle_threshold:
                if not (important_rho_range[0] < rho1 < important_rho_range[1] and abs(np.cos(theta1)) < 0.01):
                    is_suppressed = True
                    break
        if not is_suppressed:
            suppressed_lines.append((rho1, theta1))
    return suppressed_lines

# Check if the edges form a triangle based on a threshold.
def triangle_edge_check(edge_map, pt1, pt2, pt3, threshold):
    height, width = edge_map.shape
    
    # Calculate the presence of edges between points.
    def edge_presence(ptA, ptB):
        x_values = np.linspace(ptA[0], ptB[0], num=100).astype(int)
        y_values = np.linspace(ptA[1], ptB[1], num=100).astype(int)
        valid_indices = (x_values >= 0) & (x_values < width) & (y_values >= 0) & (y_values < height)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]
        if len(x_values) == 0 or len(y_values) == 0:
            return 0
        return np.sum(edge_map[y_values, x_values] != 0) / len(x_values)

    edge1 = edge_presence(pt1, pt2)
    edge2 = edge_presence(pt2, pt3)
    edge3 = edge_presence(pt3, pt1)
    
    return (edge1 + edge2 + edge3) / 3 > threshold

# Classify the type of triangle based on side lengths and angles.
def classify_triangle(pt1, pt2, pt3):
    # Calculate the length between two points.
    def length(ptA, ptB):
        return np.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)

    len1 = length(pt1, pt2)
    len2 = length(pt2, pt3)
    len3 = length(pt3, pt1)
    
    def angle(ptA, ptB, ptC):
        ba = np.array(ptA) - np.array(ptB)
        bc = np.array(ptC) - np.array(ptB)
        epsilon = 1e-7  # Small constant to prevent division by zero
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + epsilon)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    angle1 = angle(pt1, pt2, pt3)
    angle2 = angle(pt2, pt3, pt1)
    angle3 = angle(pt3, pt1, pt2)

    # prioritize right triangles
    if np.isclose(angle1, 90, atol=2) or np.isclose(angle2, 90, atol=2) or np.isclose(angle3, 90, atol=2):
        return "right"
    elif np.isclose(len1, len2, atol=0.1) and np.isclose(len2, len3, atol=0.1):
        return "equilateral"
    elif np.isclose(len1, len2, atol=0.1) or np.isclose(len2, len3, atol=0.1) or np.isclose(len3, len1, atol=0.1):
        return "isosceles"
   
    return None

# Plot the Hough Transform results and detected lines.
def plot_hough_transform(window, accumulator, lines, window_number):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    ax[0].imshow(window, cmap='gray', aspect='auto')
    ax[0].set_title(f'Window {window_number}')
    ax[0].set_xlim([0, window.shape[1]])
    ax[0].set_ylim([window.shape[0], 0])

    ax[1].imshow(accumulator, cmap='gray', aspect='auto')
    ax[1].set_title(f'Hough Transform Accumulator for Window {window_number}')
    ax[1].set_xlim([0, len(accumulator[0])])
    ax[1].set_ylim([len(accumulator), 0])

    for line in lines:
        rho, theta = line
        x = np.linspace(0, window.shape[1], 1000)
        y = (rho - x * np.cos(theta)) / np.sin(theta)
        ax[0].plot(x, y, '-w',linewidth=4)
    
    plt.tight_layout()
    plt.show()

# Main function to detect triangles in an image using edge detection and Hough Transform.
def detect_triangles(image_path, preprocess_params, hough_params, nms_params, window_height, window_width, step_x, edge_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed = preprocess_image(image, **preprocess_params)
    display_image("Preprocessed Image", preprocessed)

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    edges = cv2.Canny(blurred_image, 150, 200)  # Lower thresholds to capture more edges
    display_image("Edge Map", edges)

    lines, accumulator = hough_transform(edges, **hough_params)
    accumulator = np.log10(accumulator + 1)
    lines = non_maximum_suppression(lines, **nms_params)

    # Display the Hough transform parameter space
    plt.figure(figsize=(6, 6))
    accumulator = np.log(accumulator + 1)
    plt.imshow(accumulator, cmap='gray', aspect='auto')
    plt.title("Hough Transform Parameter Space")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Rho (pixels)")
    plt.axis('on')
    plt.show()
    edges_with_lines = draw_lines_on_edge_map(edges, lines, color=(255, 255, 255))
    display_image("Detected Lines on Edge Map", edges_with_lines)
    
    height, width = edges.shape
    triangles = []  # List to store detected triangles.

    window_number = 0

    # Process each window to detect lines and triangles.
    for y in range(0, height - window_height + 1, window_height):
        for x in range(0, width - window_width + 1, step_x):
            window = edges[y:y + window_height, x:x + window_width]
            if np.sum(window) == 0:
                continue

            lines, accumulator = hough_transform(window, **hough_params)
            accumulator = np.log10(accumulator + 1)  # Logarithmic scaling for better visibility
            lines = non_maximum_suppression(lines, **nms_params)
            plot_hough_transform(window, accumulator, lines, window_number)
            window_number += 1

            # Check for triangles formed by the detected lines.
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    for k in range(j + 1, len(lines)):
                        pt1 = intersection(lines[i], lines[j], width, height)
                        pt2 = intersection(lines[j], lines[k], width, height)
                        pt3 = intersection(lines[k], lines[i], width, height)
                        if pt1 and pt2 and pt3:
                            pt1 = (pt1[0] + x, pt1[1] + y)
                            pt2 = (pt2[0] + x, pt2[1] + y)
                            pt3 = (pt3[0] + x, pt3[1] + y)
                            if triangle_edge_check(edges, pt1, pt2, pt3, edge_threshold):
                                triangle_type = classify_triangle(pt1, pt2, pt3)
                                if triangle_type:
                                    triangles.append((triangle_type, [pt1, pt2, pt3]))



    # Check if a point is within the bounds of the image.
    def is_within_bounds(point, height, width):
        x, y = point
        return 0 <= x < width and 0 <= y < height

    img_with_triangles = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for triangle_type, triangle in triangles:
        if all(is_within_bounds(pt, height, width) for pt in triangle):
            pts = np.array(triangle, np.int32)
            pts = pts.reshape((-1, 1, 2))
            if triangle_type == "equilateral":
                color = (0, 0, 255)  # Blue for equilateral
            elif triangle_type == "isosceles":
                color = (0, 255, 0)  # Green for isosceles
            else:
                color = (255, 0, 0)  # Red for right triangle
            cv2.polylines(img_with_triangles, [pts], isClosed=True, color=color, thickness=2)
    
    display_image("Detected Triangles", img_with_triangles)

# Check for intersection between two lines in Hough space.
def intersection(line1, line2, width, height):
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Check if lines are nearly parallel to avoid singular matrix error
    if np.isclose(theta1, theta2, atol=np.pi / 180 * 6):  #degrees tolerance
        return None

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])

    try:
        x0, y0 = np.linalg.solve(A, b)
        if 0 <= x0 < width and 0 <= y0 < height:
            return int(x0), int(y0)
    except np.linalg.LinAlgError:
        return None
    return None

# Draw lines on the edge map for visualization.
def draw_lines_on_edge_map(edge_map, lines, color=(255, 255, 255)):
    img_with_lines = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_with_lines, (x1, y1), (x2, y2), color, 2)
    return img_with_lines

# Define the image path and parameters for the detection process.
image_path = 'Assets/examples/sample_triangles.png'
preprocess_params = {'d': 3, 'sigmaColor': 75, 'sigmaSpace': 75, 'blur_ksize': 5}
hough_params = {'rho_res': 0.5, 'theta_res': np.pi / 180, 'threshold': 70}
nms_params = {'distance_threshold': 20, 'angle_threshold': np.pi / 30, 'important_rho_range': (-164, -163)}
window_height = 200
window_width = 300
step_x = 40

# Call the main function to start the triangle detection process.
detect_triangles(image_path, preprocess_params, hough_params, nms_params, window_height, window_width, step_x, edge_threshold=0.1)
