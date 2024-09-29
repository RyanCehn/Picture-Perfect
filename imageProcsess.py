from PIL import Image
import numpy as np
import math

def reshape_to_matrix(flat_list, cols):
    # Convert a flat list into a 2D matrix
    return [flat_list[i:i + cols] for i in range(0, len(flat_list), cols)]

def convolve(matrix, kernel, factor):
    # Optimized convolution using NumPy for fast matrix operations
    mat = np.array(matrix)
    ker = np.array(kernel)
    k_size = ker.shape[0]
    pad = k_size // 2
    padded = np.pad(mat, pad, mode='constant')
    result = np.zeros_like(mat, dtype=float)

    # Perform convolution
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            region = padded[r:r + k_size, c:c + k_size]
            result[r, c] = np.sum(region * ker) * factor

    return result.tolist()

def round_angle(angle):
    # Round the angle to the nearest 0, 45, 90, or 135 degrees
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
    # Suppress non-maximum values based on gradient direction
    rows, cols = len(mag), len(mag[0])
    suppressed = np.zeros((rows, cols), dtype=int)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = ang[i][j]
            current_mag = mag[i][j]

            # Determine neighbors to compare based on the rounded direction
            if direction == 0:
                neighbor1, neighbor2 = mag[i, j - 1], mag[i, j + 1]
            elif direction == 45:
                neighbor1, neighbor2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif direction == 90:
                neighbor1, neighbor2 = mag[i - 1, j], mag[i + 1, j]
            else:  # direction == 135
                neighbor1, neighbor2 = mag[i - 1, j - 1], mag[i + 1, j + 1]

            # Apply non-maximum suppression and thresholding
            if current_mag >= neighbor1 and current_mag >= neighbor2 and current_mag > 36:
                suppressed[i, j] = 1

    return suppressed.flatten().tolist()

# Load and process the image
image_path = './uploads/panoPhoto.png'
gaussian_kernel = [[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# Convert image to grayscale matrix
image = Image.open(image_path).convert('L')
pixels = list(image.getdata())
width, height = image.width, image.height
gray_matrix = reshape_to_matrix(pixels, width)

# Apply Gaussian blur to the image
blurred = convolve(gray_matrix, gaussian_kernel, 1 / 159)

# Compute gradients using Sobel operators
grad_x = convolve(blurred, sobel_x, 1)
grad_y = convolve(blurred, sobel_y, 1)

# Calculate gradient magnitude and direction
magnitude = [[math.hypot(x, y) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(grad_x, grad_y)]
direction = [[round_angle(math.degrees(math.atan2(y, x))) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(grad_x, grad_y)]

# Perform non-maximum suppression
suppressed = non_max_suppression(np.array(magnitude), np.array(direction))

# Create and save the output image
output_image = Image.new('1', (width, height))  # Binary image mode
output_image.putdata(suppressed)
output_image.save('uploads/optimized-lines.png', 'PNG')