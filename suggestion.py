import cv2
import numpy as np
import matplotlib.pyplot as plt

def suggest_camera_position(image_path):
    """
    Suggest camera placement based on detected lines in the image.

    Parameters:
    - image_path: str, path to the input image with detected lines.
    """
    # Step 1: Read the image with lines
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return

    # Step 2: Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Detect edges using Canny
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    
    # Step 4: Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=20)
    
    # Initialize variables for vanishing point calculation
    slopes = []
    intercepts = []
    
    if lines is not None:
        # Extract line parameters and calculate slopes and intercepts
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)
    
    # Step 5: Estimate vanishing points (simplified by averaging the slopes and intercepts)
    if slopes:
        avg_slope = np.median(slopes)
        avg_intercept = np.median(intercepts)
        # The vanishing point can be approximated by finding the intersection of average lines
        vp_x = int(-avg_intercept / avg_slope) if avg_slope != 0 else 0
        vp_y = int(avg_intercept)
        vanishing_point = (vp_x, vp_y)
        print(f"Estimated vanishing point at: {vanishing_point}")

        # Step 6: Suggest camera placement based on vanishing point and line analysis
        suggested_position = (vp_x, image.shape[0] // 3)  # Position camera a third from the top
        suggested_angle = avg_slope
        print(f"Suggested camera position: {suggested_position}, Suggested angle: {np.degrees(np.arctan(suggested_angle))}°")

        # Step 7: Visualize suggestions on the image
        result_image = image.copy()
        cv2.circle(result_image, vanishing_point, 10, (0, 255, 0), -1)  # Mark vanishing point
        cv2.putText(result_image, 'Suggested Camera Position', suggested_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.arrowedLine(result_image, (image.shape[1] // 2, image.shape[0]), suggested_position, (255, 0, 0), 2)  # Indicate camera direction

        # Display the result
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Camera Position Suggestion')
        plt.axis('off')
        plt.show()

    else:
        print("No lines detected. Unable to suggest camera position.")
    
# Example usage
input_line_image_path = 'uploads/connected_lines_transparent.png'  # Use the generated image with connected lines
suggest_camera_position(input_line_image_path)