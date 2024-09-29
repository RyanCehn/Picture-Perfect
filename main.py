import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)

def process_img():
    # Step 1: Read the image
    image = cv2.imread('uploads/panoPhoto.png')  # Replace with the path to your image
    if image is None:
        print("Error: Image not found.")
        return
    
    height, width = image.shape[:2]
    
    # Save the original image with step number in filename
    step1_path = 'uploads/1_panoPhoto_Original.png'
    cv2.imwrite(step1_path, image)

    # Step 2: Convert to grayscale with enhanced contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_gray = clahe.apply(gray_image)

    # Save the contrast-enhanced grayscale image
    step2_path = 'uploads/2_panoPhoto_Contrast_Gray.png'
    cv2.imwrite(step2_path, contrast_gray)

    # Step 3: Smooth out similar grayscale values to average out small variations
    smoothed_image = cv2.GaussianBlur(contrast_gray, (7, 7), 0)

    # Save the smoothed grayscale image
    step3_path = 'uploads/3_panoPhoto_Smoothed.png'
    cv2.imwrite(step3_path, smoothed_image)

    # Step 4: Create a dot representation based on significant contrast in neighboring cells
    dot_image = np.zeros_like(image)
    grid_size = 10  # Finer grid size to increase dot density

    # Loop through the image with a grid to find contrasting pixels
    for y in range(0, height - grid_size, grid_size):
        for x in range(0, width - grid_size, grid_size):
            # Define the current cell and its neighboring region
            cell = smoothed_image[y:y + grid_size, x:x + grid_size]
            center_value = np.mean(cell)

            # Check neighboring regions for significant contrast
            neighbors = [
                smoothed_image[max(y - grid_size, 0):y, x:x + grid_size],  # Above
                smoothed_image[y + grid_size:min(y + 2 * grid_size, height), x:x + grid_size],  # Below
                smoothed_image[y:y + grid_size, max(x - grid_size, 0):x],  # Left
                smoothed_image[y:y + grid_size, x + grid_size:min(x + 2 * grid_size, width)]  # Right
            ]

            # Calculate average contrast between the current cell and its neighbors
            contrasts = [np.abs(center_value - np.mean(neighbor)) for neighbor in neighbors if neighbor.size > 0]

            # Define a threshold to determine significant contrast
            if any(contrast > 15 for contrast in contrasts):  # Threshold value can be adjusted for sensitivity
                # Calculate the center of the cell
                center_x = x + grid_size // 2
                center_y = y + grid_size // 2
                cv2.circle(dot_image, (center_x, center_y), 1, (255, 255, 255), -1)  # Dot size set to 1 for higher density

    # Step 5: Save and display the resulting dot image
    step4_path = 'uploads/4_panoPhoto_Dots.png'
    cv2.imwrite(step4_path, dot_image)  # Save the dot representation

    # Display the dot image using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(dot_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title('Dot Representation of Contrasting Areas')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    process_img()