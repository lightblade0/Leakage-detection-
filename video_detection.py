from ultralytics import YOLO
import cv2
import os
from datetime import datetime

def process_video(video_path, output_dir='output_frames', conf_threshold=0.5):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the YOLO model
    model = YOLO("best.pt")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Resolution: {frame_width}x{frame_height}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        results = model.predict(frame, conf=conf_threshold)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Save frame with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_filename = f"frame_{timestamp}_{frame_count}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # Save the frame
        cv2.imwrite(frame_path, annotated_frame)
        
        # Print detection information
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"Frame {frame_count}: Detected {class_name} with confidence {confidence:.2f}")
        
        frame_count += 1
        
        # Optional: Display the frame (uncomment to see live preview)
        # cv2.imshow('Frame', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release resources
    cap.release()
    # cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Example usage
    video_path = "video.mp4"  # Replace with your video path
    process_video(video_path) 