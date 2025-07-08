# run_detection.py

from ultralytics import YOLO
import os

def main():
    """
    Loads a YOLOv8 model, performs prediction on a sample image,
    and prints/saves the results.
    """
    print("Starting YOLO detection script...")

    # Define the path for the model and source image
    # The model will be downloaded to ~/.cache/ultralytics/ if not present
    model_name = "yolov8n.pt"
    image_url = "https://ultralytics.com/images/bus.jpg"
    output_dir = "." # Output directory inside the container

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    try:
        # Load a pre-trained YOLOv8n model
        print(f"Loading model: {model_name}...")
        model = YOLO(model_name)
        print("Model loaded successfully.")

        # Perform prediction on the image URL
        print(f"Running prediction on image: {image_url}...")
        # The 'save=True' argument will save the results (image with bounding boxes)
        # to the 'runs/detect/predict' directory within the container.
        # Since /usr/src/app is mounted, these results will appear on your host.
        results = model.predict(source=image_url, save=True, project=output_dir, name='predict')
        print("Prediction completed.")

        # Process and print results (optional, for demonstration)
        for i, r in enumerate(results):
            print(f"\n--- Results for image {i+1} ---")
            # Print bounding boxes, classes, and confidence scores
            if r.boxes:
                print("Detected Objects:")
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    xyxy = box.xyxy[0].tolist() # Bounding box coordinates [x1, y1, x2, y2]
                    print(f"  Class: {model.names[cls]}, Confidence: {conf:.2f}, BBox: {xyxy}")
            else:
                print("No objects detected.")

        print(f"\nResults (images with detections) saved to: {output_dir}/predict")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
