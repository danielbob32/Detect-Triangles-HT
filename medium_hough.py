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

# Hough Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Create a blank image to draw Hough lines
hough_lines_image = np.zeros_like(image)

# Draw the detected lines on the edge map
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
        cv2.line(hough_lines_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # Small square markers
        cv2.rectangle(hough_lines_image, (int(x0) - 2, int(y0) - 2), (int(x0) + 2, int(y0) + 2), (255, 0, 0), -1)


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
        y1 = int(y0 + diag_len * (a))
        x2 = int(x0 - diag_len * (-b))
        y2 = int(y0 - diag_len * (a))
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    
    return img

# Draw the detected lines on the edge map
line_image = np.copy(edges)
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
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
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
# Plotting Hough Transform parameter space (rho, theta)
thetas = np.deg2rad(np.arange(-90.0, 90.0))
height, width = edges.shape
diag_len = int(np.sqrt(width**2 + height**2))
rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
hough_space = np.zeros((2 * diag_len, len(thetas)))

for y in range(height):
    for x in range(width):
        if edges[y, x] > 0:
            for theta_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
                if 0 <= rho < 2 * diag_len:
                    hough_space[rho, theta_idx] += 1

# Detect triangles
triangles = detect_triangles(lines, edges)

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
    

display_image("Input Image", image)
display_image("Edge Map", edges)

# Display the Hough transform parameter space
plt.figure(figsize=(10, 10))
plt.imshow(hough_space, cmap='gray', extent=[-90, 90, -diag_len, diag_len], aspect='auto')
plt.title("Hough Transform Parameter Space")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (pixels)")
plt.axis('on')
plt.show()

# Display the detected lines

display_image("Detected Lines", line_image)
edges_with_lines = draw_lines_on_edge_map(edges, lines, color=(255, 255, 255))
display_image("Detected Lines on Edge Map", edges_with_lines)
display_image("Detected Triangles", triangle_image, cmap=None)