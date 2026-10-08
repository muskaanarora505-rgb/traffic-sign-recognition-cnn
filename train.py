import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloader
from model import TrafficSignCNN


DATA_DIR = "data/train"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, classes = create_dataloader(
        DATA_DIR,
        BATCH_SIZE
    )

    model = TrafficSignCNN(
        num_classes=len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    torch.save(
        model.state_dict(),
        "traffic_sign_cnn.pth"
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    train()
