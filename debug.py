#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your image
image_path = 'Assets/group_sketch/sample_triangles.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# List of rho and theta from your detections (only potential horizontal lines)
lines = [
 
    (-163.79750507687845, -1.5707963267948966),
    (-181.80272700899332, -1.5707963267948966),
    (-180.80243690165366, -1.5707963267948966),
    (-179.8021467943139, -1.5707963267948966),
    (-178.80185668697413, -1.5707963267948966),
    (-177.80156657963448, -1.5707963267948966),
    (-176.80127647229472, -1.5707963267948966),
    (-175.80098636495507, -1.5707963267948966),
    (-174.8006962576153, -1.5707963267948966),
    (-173.80040615027565, -1.5707963267948966),
    (193.80620829706982, -0.008775398473714668),
    (194.8064984044097, 0.008775398473714446),
    (817.9872352770524, -0.008775398473714668),
    (818.9875253843923, 0.008775398473714446)
]

# Function to draw lines
def draw_line(img, rho, theta, color=(0, 0, 255), thickness=2):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Points to draw the line
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Draw each line
for rho, theta in lines:
    draw_line(img_with_lines, rho, theta)

# Show the image with lines
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Lines on Image")
plt.axis('off')
plt.show()
