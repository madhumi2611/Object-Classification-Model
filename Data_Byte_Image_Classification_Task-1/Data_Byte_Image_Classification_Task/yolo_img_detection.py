from ultralytics import YOLO
import cv2
from datetime import datetime

# Load the model
model = YOLO('Models/best.pt')

# Perform inference on the image
results = model('Images Collected/Final Demon Slayer Data/Twinkle-Dolly-Demon-Devil39S-Blade-3-8-Pieces-Box-Shokugan-Japan-Figure-4549660627753-0_512x766.jpg')

# Draw the results on the frame
annotated_frame = results[0].plot()
    
# Display the frame with annotations
cv2.imshow('YOLO Detection', annotated_frame)

if cv2.waitKey(0) & 0xFF == ord('s'):
    # Get the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the output file name with the timestamp
    output_file_name = f"output/output_{timestamp}.jpg"
    # Save the image
    cv2.imwrite(output_file_name, annotated_frame)

cv2.destroyAllWindows()