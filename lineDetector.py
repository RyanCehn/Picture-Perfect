import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def image_to_dots(image_path, output_path, dot_size=2, target_width=1600, target_height=900):
    # Step 1: Read the image from the /uploads folder
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return

    # Resize the image to fit the target grid size
    resized_image = cv2.resize(image, (target_width, target_height))

    # Step 2: Increase contrast using histogram equalization
    contrast_image = cv2.equalizeHist(resized_image)

    # Step 3: Threshold the image to create a binary (black and white) version
    _, binary_image = cv2.threshold(contrast_image, 128, 255, cv2.THRESH_BINARY)

    # Step 4: Calculate the grid size to fit 160 by 90 dots (total grid size)
    grid_size_x = target_width // 160
    grid_size_y = target_height // 90

    # Step 5: Create an empty canvas for drawing the dots
    dot_image = np.ones_like(binary_image) * 255  # Start with a white canvas

    # Step 6: Divide the image into a grid and place dots where there is white
    for y in range(0, target_height, grid_size_y):
        for x in range(0, target_width, grid_size_x):
            # Check if the grid cell contains any white pixel
            grid_region = binary_image[y:y + grid_size_y, x:x + grid_size_x]
            if np.any(grid_region == 255):  # If any pixel is white
                # Draw a dot on the new canvas
                cv2.circle(dot_image, (x + grid_size_x // 2, y + grid_size_y // 2), dot_size, 0, -1)  # Black dot

    # Step 7: Save and display the dot image
    cv2.imwrite(output_path, dot_image)

    plt.figure(figsize=(10, 8))
    plt.imshow(dot_image, cmap='gray')
    plt.title('Image to Dots')
    plt.axis('off')
    plt.show()

    # Step 8: Apply Hough Line Transform to find lines in the dot image
    edges = cv2.Canny(dot_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # Detect lines

    # Draw detected lines on the original image
    line_image = cv2.cvtColor(dot_image, cv2.COLOR_GRAY2BGR)
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
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw lines in red

    # Save and display the image with lines
    line_output_path = os.path.join(os.path.dirname(output_path), 'output_with_lines.png')
    cv2.imwrite(line_output_path, line_image)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title('Hough Transform Lines on Dots')
    plt.axis('off')
    plt.show()

# Example usage
input_image_path = 'uploads/optimized-lines.png'  # Path to the image in the /uploads folder
output_dot_path = 'uploads/output_dots.png'  # Output path for the dot image
image_to_dots(input_image_path, output_dot_path)