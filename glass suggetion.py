import cv2
import numpy as np

# Load the image of the eye (replace 'eye_image.jpg' with the path to your eye image)
image_path = 'eye_image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

# Use thresholding to detect the pupil (darker region of the eye)
_, threshold_image = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image to detect circular shapes (pupil or iris)
contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour which will likely be the pupil
largest_contour = max(contours, key=cv2.contourArea)

# Get the coordinates and radius of the enclosing circle of the pupil
(x, y), radius = cv2.minEnclosingCircle(largest_contour)
center = (int(x), int(y))
radius = int(radius)

# Draw a circle around the detected pupil
cv2.circle(image, center, radius, (0, 255, 0), 2)

# Simulate a suggestion for glasses based on pupil size
if radius < 10:
    suggestion = "You might need reading glasses."
elif 10 <= radius <= 20:
    suggestion = "Your eyesight seems average."
else:
    suggestion = "You might need distance glasses."

# Display the suggestion on the image
cv2.putText(image, suggestion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the result
cv2.imshow('Retina Scan Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
