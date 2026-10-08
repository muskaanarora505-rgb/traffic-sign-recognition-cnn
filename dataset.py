from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms():
    """Return the initial image preprocessing pipeline."""

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform


def create_dataset(data_dir):
    """Create an ImageFolder dataset."""

    transform = get_transforms()

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    return dataset


def create_dataloader(data_dir, batch_size=32):
    """Create a DataLoader for the dataset."""

    dataset = create_dataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader, dataset.classes
