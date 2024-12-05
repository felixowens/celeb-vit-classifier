# Celeb ViT classifier

Training code for a simple ViT model to classify celeb faces. This model was trained on [tonyassi/celebrity-1000](https://huggingface.co/datasets/tonyassi/celebrity-1000) dataset using [flxowens/celebrity-classifier-alpha-1](https://huggingface.co/flxowens/celebrity-classifier-alpha-1) as a base.
It achieves the following results on the evaluation set:

- Loss: 1.1460
- Accuracy: 0.8155

The weights are available on HuggingFace: [flxowens/celebrity-classifier-alpha-2](https://huggingface.co/flxowens/celebrity-classifier-alpha-2).

## Requirements

Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

## Usage

### Training

To train the model, set your HuggingFace token as an environment variable and run main.py:

```sh
export HUGGINGFACE_TOKEN=your_token_here
python main.py
```

The training process includes:

- Dataset splitting (80% train, 20% test)
- Image augmentation (random crops, flips, rotations, color jittering)
- Model training with evaluation each epoch
- Automatic model checkpointing
- TensorBoard logging
- Pushing the trained model to HuggingFace Hub

### Inference

To run predictions on images:

```sh
python predict.py --image path/to/image.jpg --model flxowens/celebrity-classifier-alpha-2
```

Optional arguments:

- `--device`: Specify "cuda" or "cpu" (defaults to CUDA if available)
- `--model`: HuggingFace model name (defaults to flxowens/celebrity-classifier-alpha-2)
  The model will preprocess the image to 256x256 pixels and return the top 10 predicted celebrity matches with confidence scores.

## Project Structure

- main.py: Entry point for training
- trainer.py: Contains training logic and configuration
- predict.py: Inference script for running predictions
- logs: TensorBoard logs directory
- temp: Directory for temporary files during inference

## Training Details

The model uses the following training parameters:

- Batch size: 64
- Gradient accumulation steps: 4
- Learning rate: 1e-4
- Training epochs: 30
- Warmup ratio: 0.1
- Evaluation strategy: Per epoch
- Best model checkpointing based on accuracy
