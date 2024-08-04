import torch
from timeit import default_timer as timer
import Path
from model import TinyVGG
from dataloader import (
    train_data_augmented,
    test_dataloader_simple,
    train_dataloader_augmented,
)
from utils import train
from torch import nn



NUM_EPOCHS = 50
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_1 = TinyVGG(
    input_features=1,
    hidden_features=64,
    output_features=len(train_data_augmented.classes),
).to(device)





loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

start_time = timer()

model_1_results = train(
    model=model_1,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    epochs=NUM_EPOCHS,
    test_dataloader=test_dataloader_simple,
    train_dataloader=train_dataloader_augmented,
)

end_time = timer()

print(f"Total Trainig Time {end_time-start_time} seconds ")


MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "CUSTOM_XRAY_DETECTION_BEST.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving Model to {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
