#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Apply Gaussian Blur
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(image_blur, 50, 150)
    return edges

def hough_transform_display(image, rho_res=1, theta_res=np.pi/180, threshold=100):
    edges = preprocess_image(image)
    hough_space, theta, rho = cv2.HoughLines(edges, rho_res, theta_res, threshold)
    return edges, hough_space, theta, rho

def plot_hough_peaks(image, accumulator, thetas, rhos, peaks):
    fig, ax = plt.subplots()
    ax.imshow(accumulator, cmap='hot')
    ax.set_title('Hough Transform')
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Rho (pixels)')
    ax.autoscale(False)
    # Plot the peaks
    for peak in peaks:
        ax.plot(peak[1]*np.pi/180, peak[0], 'ro')
    plt.show()

def draw_detected_lines(image, lines):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

def main(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges, hough_space, theta, rho = hough_transform_display(image)
    peaks = [(rho, theta) for rho, theta in zip(rho, theta)]  # Example peaks data
    plot_hough_peaks(image, hough_space, theta, rho, peaks)
    lines_image = draw_detected_lines(image.copy(), hough_space)
    plt.imshow(lines_image, cmap='gray')
    plt.title('Detected Lines')
    plt.show()

# Call the main function with your image path
main('Assets/four_triangles_example.png')
