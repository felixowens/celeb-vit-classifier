# predict.py
import numpy as np
from transformers import pipeline  # type: ignore
import argparse
from PIL import Image
import os
import torch
from facenet_pytorch import MTCNN  # type: ignore
import cv2
import mediapipe as mp  # type: ignore


def load_model(model_name: str, device: str = "cuda"):
    """Load the model pipeline with specified device"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # return top 10 results
    return pipeline("image-classification", model=model_name, device=device, top_k=10)


def preprocess_image(
    image_path, target_size=(256, 256), padding_factor=0.3, y_offset=-0.1, debug=False
):
    """
    Preprocess image to 256x256 with padding around face.
    Args:
        image_path: Path to input image
        target_size: Final image dimensions
        padding_factor: Amount of padding around face (0.3 = 30%)
        y_offset: Vertical offset as fraction of face height (+ moves down, - moves up)
        debug: If True, saves image with drawn bounding box
    """
    # Open image
    img = Image.open(image_path)

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = face_detection.process(img_cv)

    ih, iw, _ = img_cv.shape

    # Get padded bounding box of face
    for landmark in results.detections:
        bboxC = landmark.location_data.relative_bounding_box

        # Calculate base coordinates
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        # Add padding
        pad_w = int(w * padding_factor)
        pad_h = int(h * padding_factor)

        # Apply y offset
        y_shift = int(h * y_offset)
        y += y_shift

        # Ensure coordinates stay within image bounds
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(iw, x + w + pad_w)
        y2 = min(ih, y + h + pad_h)

        if debug:
            debug_img = img_cv.copy()
            # Draw original face box in green
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw padded box in yellow
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            debug_path = "temp/temp_debug.jpg"
            cv2.imwrite(debug_path, debug_img)
            print(f"Debug image saved to {debug_path}")

        face = img_cv[y1:y2, x1:x2]
        break

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)
    face = face.resize(target_size)

    print("Image preprocessed")
    return face


def predict_image(pipe, image_path: str):
    """Run prediction on an image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = preprocess_image(image_path, debug=True)
        # save the image in a temporary file, always overwrite the previous one
        img.save("temp/temp.jpg")
        result = pipe(img)
        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on an image using a trained model"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flxowens/celebrity-classifier-alpha-2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    # Load model and run prediction
    try:
        pipe = load_model(args.model, args.device)
        print(f"Using device: {pipe.device}")
        results = predict_image(pipe, args.image)

        if results:
            for result in results:
                print(f"Label: {result['label']}, Score: {result['score']:.4f}")
    except Exception as e:
        print(f"Error: {e}")


# Usage: python predict.py --image path/to/image.jpg --model flxowens/celebrity-classifier-alpha-2
if __name__ == "__main__":
    main()
