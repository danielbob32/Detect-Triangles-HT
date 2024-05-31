#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def merge_close_points(points, threshold=10):
    # This function merges points that are closer than the specified threshold
    if not points:
        return []
    
    # Sort points based on the x-coordinate
    points = sorted(points, key=lambda x: (x[0], x[1]))
    merged_points = [points[0]]

    for current in points[1:]:
        last = merged_points[-1]
        if np.sqrt((current[0] - last[0])**2 + (current[1] - last[1])**2) <= threshold:
            # Average the two points
            merged_points[-1] = ((last[0] + current[0]) / 2, (last[1] + current[1]) / 2)
        else:
            merged_points.append(current)
    
    return merged_points

def detect_triangles(lines, max_angle_deviation=np.deg2rad(10), max_distance=50):
    triangles = []
    if lines is None:
        return triangles
    
    # First, filter and merge points
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            A = np.array([
                [np.cos(line1[1]), np.sin(line1[1])],
                [np.cos(line2[1]), np.sin(line2[1])]
            ])
            B = np.array([line1[0], line2[0]])
            if np.linalg.det(A) != 0:
                point = np.linalg.solve(A, B)
                if np.isfinite(point).all():
                    points.append((point[0], point[1]))
    
    # Merge close points
    merged_points = merge_close_points(points, threshold=max_distance)

    # Form triangles from the merged points using angular and distance criteria
    n_points = len(merged_points)
    for i in range(n_points):
        for j in range(i+1, n_points):
            for k in range(j+1, n_points):
                # Check angles formed by the triangle to avoid very acute or obtuse to consider
                if is_valid_triangle(merged_points[i], merged_points[j], merged_points[k], max_angle_deviation):
                    triangles.append([merged_points[i], merged_points[j], merged_points[k]])

    return triangles

def is_valid_triangle(p1, p2, p3, max_angle_deviation):
    # This function checks if the triangle formed by points p1, p2, p3 is valid based on angle criteria
    def angle(a, b, c):
        # Calculate angle between vectors ba and bc
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return np.pi  # Returning pi effectively ignores degenerate cases where points coincide
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        return np.arccos(np.clip(cosine_angle, -1, 1))


    angles = [
        angle(np.array(p1), np.array(p2), np.array(p3)),
        angle(np.array(p2), np.array(p3), np.array(p1)),
        angle(np.array(p3), np.array(p1), np.array(p2))
    ]

    return all(np.pi - max_angle_deviation < ang < np.pi + max_angle_deviation for ang in angles)


def draw_triangles(image_shape, triangles):
    triangle_image = np.zeros(image_shape, dtype=np.uint8)
    for triangle in triangles:
        pts = np.array(triangle, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(triangle_image, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return triangle_image

# Load and process the image
image_path = 'Assets/four_triangles_example.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

triangles = detect_triangles(lines)
triangle_image = draw_triangles(image.shape, triangles)
display_image("Detected Triangles", triangle_image)