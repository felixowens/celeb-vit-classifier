from datasets import load_dataset  # type: ignore
from transformers import AutoImageProcessor, DefaultDataCollator  # type: ignore
from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer as HuggingFaceTrainer,
)
import evaluate  # type: ignore
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomAffine, ColorJitter  # type: ignore
import numpy as np


class Trainer:
    def __init__(self, key, dataset_id, model_name, train_epochs):
        self.key = key
        self.model_name = model_name
        self.train_epochs = train_epochs
        self.dataset_id = dataset_id

    def train_model(self):
        # Load dataset
        ds = load_dataset(self.dataset_id, split="train")

        # Split data between test and train
        ds = ds.train_test_split(test_size=0.2)

        # Create a dictionary that maps the label name to an integer and vice versa
        labels = ds["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        # Preprocess: load a ViT image processor to process the image into a tensor
        checkpoint = "flxowens/celebrity-classifier-alpha-1"
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint,
            hidden_dropout_prob=0.2,
            attention_dropout_prob=0.2,
        )

        # Apply some image transformations to the images to make the model more robust against overfitting
        # Crop a random part of the image, resize it, and normalize it with the image mean and standard deviation
        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        _transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(p=0.5),
                RandomAffine(
                    degrees=(-10, 10),  # Slight rotation
                    translate=(0.05, 0.05),  # Minor position shifts
                    scale=(0.95, 1.05),  # Subtle scaling
                ),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ToTensor(),
                normalize,
            ]
        )

        # Preprocessing function to apply the transforms and return the pixel_values
        def transforms(examples):
            examples["pixel_values"] = [
                _transforms(img.convert("RGB")) for img in examples["image"]
            ]
            del examples["image"]
            return examples

        # Apply the preprocessing function over the entire dataset
        ds = ds.with_transform(transforms)

        # Create a batch of examples using DefaultDataCollator.
        data_collator = DefaultDataCollator()

        # Load an evaluation method
        accuracy = evaluate.load("accuracy")

        # Function that passes your predictions and labels to compute to calculate the accuracy:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        #  Load ViT with AutoModelForImageClassification, number of labels, and the label mappings
        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        # Define training hyperparameters
        training_args = TrainingArguments(
            output_dir=self.model_name,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=64,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=64,
            num_train_epochs=self.train_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=True,
            resume_from_checkpoint=True,
            hub_token=self.key,
            report_to="tensorboard",
            logging_dir="./logs",
            save_total_limit=5,  # Limit the total number of saved models
        )

        # Pass the training arguments to Trainer
        trainer = HuggingFaceTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
        )

        # Begin training
        trainer.train()

        # Push model to hub
        trainer.push_to_hub()
