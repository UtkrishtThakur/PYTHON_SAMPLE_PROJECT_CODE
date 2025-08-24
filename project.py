# Image processing
import cv2
import numpy as np

# For handling images easily
from PIL import Image

# Optional: ML / pose estimation
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def get_body_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    keypoints = {}
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[idx] = (lm.x, lm.y)  # Normalized coordinates
    return keypoints
def estimate_measurements(keypoints, image_height_cm):
    """
    image_height_cm: Approximate real-world height of person in cm
    keypoints: Dictionary from get_body_keypoints()
    """
    # Example: Shoulder width between left and right shoulders
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    
    # Pixel distance
    pixel_width = np.sqrt((left_shoulder[0]-right_shoulder[0])**2 +
                          (left_shoulder[1]-right_shoulder[1])**2)
    
    # Approximate scaling: image height pixels to real-world cm
    top = keypoints[0]  # nose/top of head
    bottom = keypoints[31]  # ankle/bottom
    pixel_height = np.sqrt((top[0]-bottom[0])**2 + (top[1]-bottom[1])**2)
    
    scale = image_height_cm / pixel_height
    shoulder_width_cm = pixel_width * scale
    
    return {
        "shoulder_width_cm": round(shoulder_width_cm, 2),
        "estimated_height_cm": image_height_cm
    }
def classify_body_type(measurements):
    shoulder = measurements['shoulder_width_cm']
    height = measurements['estimated_height_cm']
    
    # Very basic classification
    if shoulder/height > 0.25:
        return "Broad Shoulders (Athletic)"
    elif shoulder/height < 0.20:
        return "Slim / Lean"
    else:
        return "Average"
def suggest_clothes(body_type):
    suggestions = {
        "Broad Shoulders (Athletic)": ["Fitted T-shirts", "Structured jackets", "V-neck shirts"],
        "Slim / Lean": ["Layered clothing", "Horizontal stripes", "Slim-fit shirts"],
        "Average": ["Casual fit shirts", "Regular jeans", "Polos"]
    }
    return suggestions.get(body_type, ["Standard Clothing"])

image_path = "person.jpg"
x=int(input("Height of the person:   "))


keypoints = get_body_keypoints(image_path)
measurements = estimate_measurements(keypoints,x)
body_type = classify_body_type(measurements)
clothes = suggest_clothes(body_type)

print("Estimated Measurements:", measurements)
print("Body Type:", body_type)
print("Recommended Clothes:", clothes)

