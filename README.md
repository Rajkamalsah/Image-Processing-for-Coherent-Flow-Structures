# Image Processing and Velocity Vector Field Calculation

This project involves preprocessing images, performing cross-correlation to determine shift vectors, calculating velocity vectors, and visualizing the velocity vector field. The project uses OpenCV for image processing, NumPy for numerical operations, and Matplotlib for visualization.

## Features

- **Image Preprocessing**: Load and resize images.
- **Cross-Correlation**: Calculate shift vectors between two images using cross-correlation.
- **Velocity Calculation**: Compute velocity vectors based on shift vectors and time interval.
- **Visualization**: Plot the velocity vector field.

## Tools and Technologies

- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-processing-velocity.git
    cd image-processing-velocity
    ```

2. Install the required libraries:
    ```bash
    pip install numpy opencv-python matplotlib
    ```

## Usage

1. **Preprocess Image**:
    ```python
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    def preprocess_image(image_path):
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return None
        
        # Resize the image
        resized_image = cv2.resize(image, (640, 480))
        
        return resized_image
    ```

2. **Cross-Correlation**:
    ```python
    def cross_correlation(image1, image2, window_size=32, overlap=16):
        # Get the dimensions of the images
        h, w = image1.shape
        
        # Initialize the shift vectors
        shift_vectors = np.zeros((h // overlap, w // overlap, 2))
        
        # Loop over the image in overlapping windows
        for i in range(0, h - window_size, overlap):
            for j in range(0, w - window_size, overlap):
                # Extract the windows
                window1 = image1[i:i + window_size, j:j + window_size]
                window2 = image2[i:i + window_size, j:j + window_size]
                
                # Perform cross-correlation
                result = cv2.matchTemplate(window2, window1, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                
                # Calculate the shift
                max_loc = np.array(max_loc)
                shift = max_loc - np.array([window_size // 2, window_size // 2])
                shift_vectors[i // overlap, j // overlap] = shift
        
        print("Shift Vectors Shape:", shift_vectors.shape)
        return shift_vectors
    ```

3. **Calculate Velocity**:
    ```python
    def calculate_velocity(shift_vectors, time_interval):
        # Calculate the velocity vectors
        velocity_vectors = shift_vectors / time_interval
        print("Velocity Vectors Shape:", velocity_vectors.shape)
        return velocity_vectors
    ```

4. **Plot Velocity Field**:
    ```python
    def plot_velocity_field(velocity_vectors):
        # Create a grid for plotting
        X, Y = np.meshgrid(np.arange(velocity_vectors.shape), np.arange(velocity_vectors.shape))
        
        # Plot the velocity field
        plt.quiver(X, Y, velocity_vectors[..., 0], velocity_vectors[..., 1])
        plt.title('Velocity Vector Field')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    ```

5. **Example Usage**:
    ```python
    # Example usage
    image_path1 = 'E:/github/Project_3/Frame1.tif'
    image_path2 = 'E:/github/Project_3/Frame2.tif'
    time_interval = 0.001  # Time interval in seconds (1 ms)

    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)

    if image1 is not None and image2 is not None:
        shift_vectors = cross_correlation(image1, image2)
        velocity_vectors = calculate_velocity(shift_vectors, time_interval)
        
        plot_velocity_field(velocity_vectors)
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
