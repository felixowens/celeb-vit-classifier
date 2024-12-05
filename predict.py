# predict.py
from transformers import pipeline  # type: ignore
import argparse
from PIL import Image
import os
import torch


def load_model(model_name: str, device: str = "cuda"):
    """Load the model pipeline with specified device"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # return top 10 results
    return pipeline("image-classification", model=model_name, device=device, top_k=10)


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image to 256x256 with resize and center crop
    """
    # Open image
    img = Image.open(image_path)

    # Calculate aspect ratio
    ratio = min(target_size[0] / img.size[0], target_size[1] / img.size[1])
    new_size = tuple([int(x * ratio) for x in img.size])

    # Resize maintaining aspect ratio
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Center crop
    left = (img.size[0] - target_size[0]) / 2
    top = (img.size[1] - target_size[1]) / 2
    right = (img.size[0] + target_size[0]) / 2
    bottom = (img.size[1] + target_size[1]) / 2

    img = img.crop((left, top, right, bottom))
    return img


def predict_image(pipe, image_path: str):
    """Run prediction on an image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = preprocess_image(image_path)
        # save the image in a temporary file, always overwrite the previous one
        img.save("test/temp.jpg")
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
