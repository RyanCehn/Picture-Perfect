import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)

def reshape_to_matrix(flat_list, cols):
    return [flat_list[i:i + cols] for i in range(0, len(flat_list), cols)]

def convolve(matrix, kernel, factor):
    mat = np.array(matrix)
    ker = np.array(kernel)
    k_size = ker.shape[0]
    pad = k_size // 2
    padded = np.pad(mat, pad, mode='constant')
    result = np.zeros_like(mat, dtype=float)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            region = padded[r:r + k_size, c:c + k_size]
            result[r, c] = np.sum(region * ker) * factor
    return result.tolist()

def round_angle(angle):
    abs_angle = abs(angle) % 180
    if abs_angle <= 22.5 or abs_angle > 157.5:
        return 0
    elif 22.5 < abs_angle <= 67.5:
        return 45
    elif 67.5 < abs_angle <= 112.5:
        return 90
    else:
        return 135

def non_max_suppression(mag, ang):
    rows, cols = len(mag), len(mag[0])
    suppressed = np.zeros((rows, cols), dtype=int)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = ang[i][j]
            current_mag = mag[i][j]
            if direction == 0:
                neighbor1, neighbor2 = mag[i, j - 1], mag[i, j + 1]
            elif direction == 45:
                neighbor1, neighbor2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif direction == 90:
                neighbor1, neighbor2 = mag[i - 1, j], mag[i + 1, j]
            else:  # direction == 135
                neighbor1, neighbor2 = mag[i - 1, j - 1], mag[i + 1, j + 1]
            if current_mag >= neighbor1 and current_mag >= neighbor2 and current_mag > 36:
                suppressed[i, j] = 1
    return suppressed.flatten().tolist()

def process_img():
    # Step 1: Read and process the initial image
    image_path = 'uploads/panoPhoto.png'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    height, width = image.shape[:2]
    cv2.imwrite('uploads/1_panoPhoto_Original.png', image)

    # Step 2: Convert to grayscale and enhance contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_gray = clahe.apply(gray_image)
    cv2.imwrite('uploads/2_panoPhoto_Contrast_Gray.png', contrast_gray)

    # Step 3: Apply Gaussian blur
    smoothed_image = cv2.GaussianBlur(contrast_gray, (7, 7), 0)
    cv2.imwrite('uploads/3_panoPhoto_Smoothed.png', smoothed_image)

    # Step 4: Edge detection using custom implementation
    gaussian_kernel = [[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    blurred = convolve(smoothed_image.tolist(), gaussian_kernel, 1 / 159)
    grad_x = convolve(blurred, sobel_x, 1)
    grad_y = convolve(blurred, sobel_y, 1)

    magnitude = [[math.hypot(x, y) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(grad_x, grad_y)]
    direction = [[round_angle(math.degrees(math.atan2(y, x))) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(grad_x, grad_y)]

    suppressed = non_max_suppression(np.array(magnitude), np.array(direction))
    edge_image = np.array(suppressed).reshape((height, width)) * 255
    cv2.imwrite('uploads/4_panoPhoto_Edges.png', edge_image)

    # Step 5: Create dot representation
    dot_image = np.zeros_like(image)
    grid_size = 10
    for y in range(0, height - grid_size, grid_size):
        for x in range(0, width - grid_size, grid_size):
            if np.any(edge_image[y:y + grid_size, x:x + grid_size] > 0):
                center_x = x + grid_size // 2
                center_y = y + grid_size // 2
                cv2.circle(dot_image, (center_x, center_y), 1, (255, 255, 255), -1)

    cv2.imwrite('uploads/5_panoPhoto_Dots.png', dot_image)

    # Step 6: Connect dots and create line image
    gray_dot_image = cv2.cvtColor(dot_image, cv2.COLOR_BGR2GRAY)
    _, binary_dot_image = cv2.threshold(gray_dot_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_dot_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))

    line_image = np.zeros((height, width, 4), dtype=np.uint8)
    intersection_map = np.zeros((height, width), dtype=np.float32)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pt1 = points[i]
            pt2 = points[j]
            cv2.line(line_image, pt1, pt2, (0, 0, 255, 255), 1)
            cv2.line(intersection_map, pt1, pt2, 1, 1)

    intersection_map = cv2.normalize(intersection_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for y in range(height):
        for x in range(width):
            if intersection_map[y, x] > 0:
                alpha_value = 255 - intersection_map[y, x]
                line_image[y, x, 3] = alpha_value

    cv2.imwrite('uploads/6_connected_lines_transparent.png', cv2.cvtColor(line_image, cv2.COLOR_RGBA2BGR))

    # Step 7: Suggest camera position
    gray_line_image = cv2.cvtColor(cv2.cvtColor(line_image, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_line_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=20)

    slopes = []
    intercepts = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)

    if slopes:
        avg_slope = np.median(slopes)
        avg_intercept = np.median(intercepts)
        vp_x = int(-avg_intercept / avg_slope) if avg_slope != 0 else 0
        vp_y = int(avg_intercept)
        vanishing_point = (vp_x, vp_y)
        suggested_position = (vp_x, height // 3)
        suggested_angle = np.degrees(np.arctan(avg_slope))

        result_image = cv2.cvtColor(line_image, cv2.COLOR_RGBA2BGR).copy()
        cv2.circle(result_image, vanishing_point, 10, (0, 255, 0), -1)
        cv2.putText(result_image, 'Suggested Camera Position', suggested_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.arrowedLine(result_image, (width // 2, height), suggested_position, (255, 0, 0), 2)

        cv2.imwrite('uploads/7_camera_suggestion_visualization.png', result_image)

        with open('uploads/camera_suggestions.txt', 'w') as file:
            file.write(f"Estimated vanishing point: {vanishing_point}\n")
            file.write(f"Suggested camera position: {suggested_position}\n")
            file.write(f"Suggested camera angle: {suggested_angle:.2f}°\n")
    else:
        print("No lines detected. Unable to suggest camera position.")
        with open('uploads/camera_suggestions.txt', 'w') as file:
            file.write("No lines detected. Unable to suggest camera position.\n")

    # Display final result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Camera Position Suggestion')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    process_img()