import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def connect_all_dots_with_transparent_intersections(image_path, output_path):
    """
    Connect every dot to every other dot in an image, draw lines in opaque red, 
    and make intersection points more transparent.
    
    Parameters:
    - image_path: str, path to the input dot image.
    - output_path: str, path to save the connected line image with transparent intersections.
    """
    # Step 1: Read the dot image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return
    
    # Step 2: Detect the coordinates of all the dots in the image
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
    
    # Step 3: Create an output image to draw lines connecting the dots
    line_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)  # RGBA image for transparency

    # Create a heatmap to track intersection intensity
    intersection_map = np.zeros_like(image, dtype=np.float32)

    # Step 4: Connect each dot to every other dot and update the intersection map
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pt1 = points[i]
            pt2 = points[j]
            # Draw line in opaque red on line_image
            cv2.line(line_image, pt1, pt2, (0, 0, 255, 255), 1)
            # Increment the intersection map to track line density
            cv2.line(intersection_map, pt1, pt2, 1, 1)

    # Normalize the intersection map to the range [0, 255] to determine transparency levels
    intersection_map = cv2.normalize(intersection_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 5: Adjust transparency of intersection points based on intensity
    for y in range(intersection_map.shape[0]):
        for x in range(intersection_map.shape[1]):
            if intersection_map[y, x] > 0:  # If there's an intersection
                alpha_value = 255 - intersection_map[y, x]  # Higher intersection count = more transparent
                line_image[y, x, 3] = alpha_value  # Set the transparency in the alpha channel

    # Convert the image to BGR for display and saving, ignoring the alpha channel
    bgr_image = cv2.cvtColor(line_image, cv2.COLOR_BGRA2BGR)

    # Step 6: Save and display the result
    cv2.imwrite(output_path, bgr_image)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.title('Fully Connected Dots with Transparent Intersections')
    plt.axis('off')
    plt.show()

# Example usage
input_dot_image_path = 'uploads/output_dots.png'  # Path to the dot image generated previously
output_line_image_path = 'uploads/connected_lines_transparent.png'  # Output path for the image with connected lines

connect_all_dots_with_transparent_intersections(input_dot_image_path, output_line_image_path)