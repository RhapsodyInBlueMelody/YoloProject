import cv2
import math 
import time 
from ultralytics import YOLO

def detect_object_yolov12(video_path, model_path="yolov12n.pt", output_path="output.mp4", conf_threshold=0.15, iou_threshold=0.1):
    """
    Original function for batch processing and saving video
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    cocoClassName = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                     "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                     "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                     "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    ptime = 0 
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        print(f"Frame Number:{count}")
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)

                    class_id = int(box.cls[0])
                    className = cocoClassName[class_id]
                    conf = round(float(box.conf[0]), 2)
                    label = f"{className}: {conf}"

                    text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + text_size[0], y1 - text_size[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 255), 2)

        output_video.write(frame)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def generate_frames(video_path, model_path="yolov12n.pt", conf_threshold=0.15, iou_threshold=0.1):
    """
    Generator function for Flask streaming - yields frames as JPEG bytes
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    cocoClassName = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                     "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                     "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                     "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    ptime = 0
    count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            print(f"Processing Frame Number: {count}")
            
            # Run YOLO detection
            results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # Draw bounding boxes and labels
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)

                        class_id = int(box.cls[0])
                        className = cocoClassName[class_id]
                        conf = round(float(box.conf[0]), 2)
                        label = f"{className}: {conf}"

                        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                        c2 = x1 + text_size[0], y1 - text_size[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            
            # Add FPS counter
            ctime = time.time()
            fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
            ptime = ctime
            cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        cap.release()
        print("Video capture released")

def process_and_save_video(video_path, output_path, model_path="yolov12n.pt", conf_threshold=0.15, iou_threshold=0.1):
    """
    Process entire video and save to output file
    """
    return detect_object_yolov12(video_path, model_path, output_path, conf_threshold, iou_threshold)
