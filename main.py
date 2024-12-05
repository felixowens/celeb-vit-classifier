import os
from trainer import Trainer

key = os.getenv("HUGGINGFACE_TOKEN")

trainer = Trainer(
    dataset_id="tonyassi/celebrity-1000",
    key=key,
    model_name="flxowens/celebrity-classifier-alpha-2",
    train_epochs=30,
)

print("Trainer initialized")

trainer.train_model()
