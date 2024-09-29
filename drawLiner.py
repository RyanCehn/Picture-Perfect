import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def connect_dots(image_path, output_path):
    """
    Connect dots in an image by detecting dots and drawing lines between them.
    
    Parameters:
    - image_path: str, path to the input dot image.
    - output_path: str, path to save the connected line image.
    """
    # Step 1: Read the dot image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return
    
    # Step 2: Detect the coordinates of all the dots in the image
    # Threshold the image to find the dots
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the dots
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract the center points of each detected dot
    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))
    
    # Step 3: Sort points to try and connect in a logical sequence (left-to-right, top-to-bottom)
    points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort primarily by y-coordinate, then by x-coordinate

    # Step 4: Create an output image to draw lines connecting the dots
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Step 5: Connect consecutive dots with lines
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        # Draw a line connecting the current point to the next
        cv2.line(line_image, pt1, pt2, (0, 0, 255), 2)  # Red line

    # Step 6: Save and display the result
    cv2.imwrite(output_path, line_image)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title('Connected Dots')
    plt.axis('off')
    plt.show()

# Example usage
input_dot_image_path = 'uploads/output_dots.png'  # Path to the dot image generated previously
output_line_image_path = 'uploads/connected_lines.png'  # Output path for the image with connected lines

connect_dots(input_dot_image_path, output_line_image_path)