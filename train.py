import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader
from dataset import ESC50Dataset
from model import SoundCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ESC50Dataset(
    csv_path="data/ESC-50/meta/esc50.csv",
    audi_dir="data/ESC-50/audio",
    folds=[1,2,3,4]
)

test_dataset = ESC50Dataset(
    csv_path="data/ESC-50/meta/esc50.csv",
    audi_dir="data/ESC-50/audio",
    folds=[5]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 🔎 DEBUG SHAPE HERE
x, y = next(iter(train_loader))
print("Input shape:", x.shape)
print("Label shape:", y.shape)


model = SoundCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(15):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    #Validation testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
        
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
        
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = 100 * correct / total
    


    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%\n")

