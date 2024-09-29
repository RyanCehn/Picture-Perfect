import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def draw_and_save_lines(input_image_path, output_image_path, vector_output_path):
    # Check if the input image exists
    if not os.path.exists(input_image_path):
        print(f"Error: The specified input image '{input_image_path}' does not exist.")
        return
    
    # Step 1: Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Image could not be read. Please check the file path and integrity of the file.")
        return
    
    # Step 2: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Threshold the image to isolate solid white areas
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Step 4: Use morphological operations to enhance continuous lines
    kernel = np.ones((5, 5), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # Closing to fill small gaps
    dilated_image = cv2.dilate(cleaned_image, kernel, iterations=1)  # Dilation to connect lines further

    # Step 5: Use Canny edge detection to detect edges in the enhanced white areas
    edges = cv2.Canny(dilated_image, 50, 150, apertureSize=3)

    # Step 6: Use Hough Line Transform to detect long and continuous white lines with adjusted parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=20)

    # Initialize a blank canvas to draw the red lines on the original image
    red_lines_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Step 7: Draw and save the detected long and continuous white lines in red
    with open(vector_output_path, 'w') as file:  # Open the output file to write vectors
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw lines in red only on the detected solid white areas
                cv2.line(red_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Write the line vector coordinates to the file
                file.write(f"Line: Start({x1}, {y1}) End({x2}, {y2})\n")

    # Save the resulting image with the completed lines in red
    cv2.imwrite(output_image_path, red_lines_image)

    # Display the image with completed red lines
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(red_lines_image, cv2.COLOR_BGR2RGB))
    plt.title('Completed Red Lines on Continuous Solid White Areas')
    plt.axis('off')
    plt.show()

# Example usage:
input_path = 'optimized-lines.png'  # Path to the input image
output_path = 'uploads/final_completed_lines_red.png'  # Path to save the output image
vector_output_path = 'uploads/lines_vectors.txt'  # Path to save the line vector data

draw_and_save_lines(input_path, output_path, vector_output_path)