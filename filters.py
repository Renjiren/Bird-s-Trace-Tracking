import cv2

def to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    """Apply Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, sigma)
    

#def canny_edge_detection(image, low_threshold, high_threshold):
#    """Apply Canny edge detection to an image."""
#    return cv2.Canny(image, low_threshold, high_threshold)


def gaussian_median_blur(image, kernel_size=(5, 5), sigma=1.0, m_ksize=5):
    """Apply Gaussian blur followed by median blur to an image."""
    g = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.medianBlur(g, m_ksize)