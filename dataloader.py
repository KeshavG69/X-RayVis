import pandas as pd
from PIL import Image

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


BATCH_SIZE = 32

train_transform_trivial = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
    ]
)

test_transform_simple = transforms.Compose(
    [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
)

DATA_PATH = Path(
    "/Users/keshav/Developer/pytorch-learning/Custom Datasets/archive/images/data"
)
DATA_PATH.mkdir(parents=True, exist_ok=True)

train_dir = DATA_PATH / "train"
test_dir = DATA_PATH / "test"


def find_classes(directory):
    # Get the Class Names by creating a Dataframe
    df = pd.read_csv(directory)
    classes = list(df.Target.unique())
    classes = [str(x) for x in classes]

    if not classes:
        raise FileNotFoundError(
            f"Could not find any classes in {directory} please check CSV file"
        )

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx


class ImageCSVCustom(Dataset):
    def __init__(self, targ_dir, transform=None):
        self.path = list(pd.read_csv(targ_dir).image_path)
        self.targets = list(pd.read_csv(targ_dir).Target)
        self.targets = [str(x) for x in self.targets]
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index):
        image_path = self.path[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = str(self.targets[index])
        class_idx = self.class_to_idx[class_name]
        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


train_data_augmented = ImageCSVCustom(
    targ_dir=train_dir, transform=train_transform_trivial
)
test_data_simple = ImageCSVCustom(targ_dir=test_dir, transform=test_transform_simple)


train_dataloader_augmented = DataLoader(
    dataset=train_data_augmented, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)
test_dataloader_simple = DataLoader(
    dataset=test_data_simple, batch_size=BATCH_SIZE, num_workers=0, shuffle=False
)
