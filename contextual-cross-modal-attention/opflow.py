import cv2
import numpy as np

# Function to compute optical flow between two frames
def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

# Read input video
input_video_path = '/home/UNT/vk0318/Documents/Work/Code/MultiModal-DeepFake/datasets/00162_id00097_wavtolip.mp4'
cap = cv2.VideoCapture(input_video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video properties:")
print("Frame width:", frame_width)
print("Frame height:", frame_height)
print("FPS:", fps)
print("Number of frames:", num_frames)

# Create VideoWriter object to save optical flow
output_video_path = './output_optical_flow.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Read first frame
ret, prev_frame = cap.read()

# Iterate through frames
while True:
    ret, next_frame = cap.read()
    if not ret:
        break
    
    # Compute optical flow
    flow = compute_optical_flow(prev_frame, next_frame)
    
    # Convert flow to color image for visualization
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Write optical flow frame to output video
    out.write(flow_rgb)
    
    # Update previous frame
    prev_frame = next_frame

# Release VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Optical flow video saved successfully!")
