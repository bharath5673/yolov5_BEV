import cv2

# Load the transparent image with alpha channel
image = cv2.imread('traffic light.png', cv2.IMREAD_UNCHANGED)

# Get the original image dimensions
original_height, original_width = image.shape[:2]

# Set the desired width and height for resizing
new_width = 25
new_height = 25

# Resize the image while preserving the aspect ratio
resized_image = cv2.resize(image, (new_width, new_height))

# Display or save the resized image
cv2.imshow('Resized_', resized_image)
cv2.imwrite('Resized_traffic light.png', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
