import cv2
import numpy as np

# Read the video
cap = cv2.VideoCapture('/home/UNT/vk0318/Documents/Work/Code/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/Caucasian (American)/women/id00025.mp4')

# Parameters for Lucas-Kanade method
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables to store feature vectors
feature_vectors = []

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Calculate motion vectors
    motion_vectors = p1 - p0

    # Summarize motion vectors into a feature vector (e.g., mean, std deviation, etc.)
    feature_vector = np.mean(motion_vectors, axis=0)  # Example: Mean of motion vectors
    feature_vectors.append(feature_vector)

    # Update for next iteration
    old_gray = frame_gray.copy()
    p0 = p1

# Convert feature vectors to numpy array
feature_vectors = np.array(feature_vectors)

# Perform further processing or analysis with the feature vectors
print("Feature vectors shape:", feature_vectors.shape)

# Release resources
cap.release()
cv2.destroyAllWindows()
