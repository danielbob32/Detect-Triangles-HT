#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images
def display_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the input image
input_image_path = 'Assets/four_triangles_example.png'
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Unable to load image at path: {input_image_path}")

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Print the dimensions of the image and edges to debug
print(f"Image shape: {image.shape}")
print(f"Edges shape: {edges.shape}")

# Sliding window parameters
window_size = 100  # Size of the sliding window
step_size = 50     # Step size of the sliding window

# Initialize Hough Transform accumulator for the whole image
height, width = edges.shape
diag_len = int(np.sqrt(width**2 + height**2))
thetas = np.deg2rad(np.arange(-90.0, 90.0))
rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
hough_space = np.zeros((2 * diag_len, len(thetas)))

# Create a blank image to draw Hough lines
hough_lines_image = np.zeros_like(image)

# Sliding window Hough Transform
for y in range(0, height, step_size):
    for x in range(0, width, step_size):
        # Define the window boundaries
        y_end = min(y + window_size, height)
        x_end = min(x + window_size, width)
        
        # Extract the window from the edge image
        window = edges[y:y_end, x:x_end]
        
        # Apply Hough Transform on the window
        lines = cv2.HoughLines(window, 1, np.pi / 180, 80)
        
        # Accumulate the results
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(hough_lines_image, (x1 + x, y1 + y), (x2 + x, y2 + y), (255, 255, 255), 1)
                
                # Correct theta_global calculation
                theta_global = int(np.rad2deg(theta)) + 90
                if 0 <= theta_global < len(thetas):
                    rho_global = int(rho + diag_len)
                    if 0 <= rho_global < 2 * diag_len:
                        hough_space[rho_global, theta_global] += 1

# Function to detect triangles from lines
def detect_triangles(lines, edges):
    if lines is None:
        return []

    triangles = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            for k in range(j + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                line3 = lines[k][0]

                points = []
                for line_a, line_b in [(line1, line2), (line2, line3), (line3, line1)]:
                    rho1, theta1 = line_a
                    rho2, theta2 = line_b
                    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                    B = np.array([rho1, rho2])
                    if np.abs(np.linalg.det(A)) > 1e-6:  # Check for nearly parallel lines
                        point = np.linalg.solve(A, B)
                        if 0 <= point[0] < edges.shape[1] and 0 <= point[1] < edges.shape[0]:  # Check if the point is within the image
                            if edges[int(point[1]), int(point[0])] > 0:  # Check if the point coincides with an edge
                                points.append(point)

                if len(points) == 3:
                    triangles.append(points)

    return triangles

# Draw the detected triangles on a black background
triangle_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
for idx, points in enumerate(triangles):
        # Convert points to a numpy array
    points = np.array(points)

    # Check if points contains any NaN values
    if np.isnan(points).any():
        # Handle NaN values (e.g., replace them with zero)
        points[np.isnan(points)] = 0

    # Check if points contains any infinite values
    if np.isinf(points).any():
        # Handle infinite values (e.g., replace them with a large finite number)
        points[np.isinf(points)] = np.finfo(points.dtype).max

    # Cast to int32 and reshape
    points = points.astype(np.int32).reshape((-1, 1, 2))
    color = colors[idx % len(colors)]
    cv2.polylines(triangle_image, [points], isClosed=True, color=color, thickness=2)

# Display the Hough transform parameter space with markers
plt.figure(figsize=(10, 10))
plt.imshow(hough_space, cmap='gray', extent=[-90, 90, -diag_len, diag_len], aspect='auto')
plt.title("Hough Transform Parameter Space")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (pixels)")
plt.axis('on')
plt.show()

# Display the detected lines
display_image("Detected Lines", hough_lines_image)

# Draw the detected triangles on a black background
triangle_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
for idx, points in enumerate(triangles):
    points = np.array(points, np.int32)
    if not np.any(np.isnan(points)):
        points = points.reshape((-1, 1, 2))
        color = colors[idx % len(colors)]
        cv2.polylines(triangle_image, [points], isClosed=True, color=color, thickness=2)

# Display the detected triangles
display_image("Detected Triangles", triangle_image, cmap=None)