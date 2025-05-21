import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
from importlib.metadata import distributions
import subprocess
import sys
import json


from sklearn.metrics import classification_report
import time
import pickle
from tqdm import tqdm
import os
import argparse
import numpy as np
from PIL import Image
from torchvision.models import ResNet152_Weights, resnet152

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="Starting learning rate of model")
    parser.add_argument("--data_path", type=str, default="", help="Path to the dataset")
    parser.add_argument("--build", action='store_true', help="Build data from scratch")
    parser.add_argument("--train", action='store_true', help="Flag to indicate training mode")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per iteration")
    parser.add_argument("--save_mdl", type=str, default="utils", help="Directory to save the trained model ONNX files")
    parser.add_argument("--model", type=str, default="utils/model.onnx", help="Path to ONNX model for inference")
    parser.add_argument("--mdl_name", type=str, default="model.onnx", help="Name of saved model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Specify device to run on: 'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--install_req", default=False, action='store_true', help="Flag to indicate installation of required packages")

    args = parser.parse_args()
    return args

args = get_args()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

def install_requirements():
    if args.install_req:
        installed_packages = {dist.metadata["Name"].lower(): dist.version for dist in distributions()}
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
        for package in requirements:
            if package not in installed_packages_list:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Installed {package}")
        print("All required packages are installed.")

def select_device(device_choice: str):
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        # device_choice == "cpu"
        return torch.device("cpu")

device = select_device(args.device)

data_path = args.data_path
file_path = os.path.join("utils", "training_data.pkl")
IMG_SIZE = 224

def createLabels(data_path):
    LABELS = {}
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    for idx, folder in enumerate(tqdm(folders, desc="Processing folders", delay=0.1)):
        LABELS[folder] = idx
    print(">>> data_path:", data_path)
    print(">>> folders found:", os.listdir(data_path))
    print(">>> LABELS dict:", LABELS)

    labels_save_dir = os.path.join("C:\\Users\\navsh\\OneDrive\\Documents\\GitHub\\animalrec\\models") # Or use os.path.dirname(ONNX_MODEL_PATH)
    labels_file_path = os.path.join(labels_save_dir, "labels.json")

    os.makedirs(labels_save_dir, exist_ok=True) # Ensure the directory exists
    with open(labels_file_path, "w") as f:
        json.dump(LABELS, f)
    print(f"LABELS dictionary saved to {labels_file_path}")
    return LABELS

LABELS = createLabels(data_path)

class buildData():
    def __init__(self, path):
        self.data_path = path
        self.traningData = []
        self.animalCount = {}
        
    def process_folders(self):
        for _, folder in enumerate(tqdm(os.listdir(self.data_path), desc="Processing folders", delay=0.1)):
            if os.path.isdir(os.path.join(self.data_path, folder)):
                self.animalCount[folder] = 0

    def trainBuild(self):
        for label in tqdm(LABELS, desc="Building Data"):
            class_path = os.path.join(self.data_path, label)
            if not os.path.isdir(class_path):
                print(f"Warning: {class_path} is not a directory, skipping.")
                continue

            for f in os.listdir(class_path):
                file_full_path = os.path.join(class_path, f)
                if os.path.isfile(file_full_path):
                    try:
                        self.traningData.append([file_full_path, LABELS[label]])
                        if label in self.animalCount:
                            self.animalCount[label] += 1
                    except Exception as e:
                        print(str(e))

        np.random.shuffle(self.traningData)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.traningData, f)
        #print(f"Data count: {self.animalCount}\nLabels: {LABELS}")

def loadData():
    try:
        with open(file_path, "rb") as f:
            training_data = pickle.load(f)
            return training_data
    except Exception as e:
        print("Please build the data with \"model.py --build\"")
        exit(1)

def load_resnet_model(lenClass):
    weights = ResNet152_Weights.DEFAULT
    resnet = resnet152(weights=weights)

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, lenClass)
    )
    return resnet

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def test_onnx_model(ort_session, image_tensor):
    inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    outputs = ort_session.run(None, inputs)
    return outputs

def ExportOnnx(model, onnx_file_path):
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    # Move model to CPU before export to ensure portability
    model_cpu = model.to("cpu")
    torch.onnx.export(
        model_cpu, 
        dummy_input.cpu(), 
        onnx_file_path, 
        input_names=['input'], 
        output_names=['output'], 
        opset_version=12
    )
    # Move back to original device if needed
    model.to(device)

def save_model(model, epoch):
    model_save_dir = args.save_mdl
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    onnx_file_path = os.path.join(model_save_dir, f"model_epoch_{epoch + 1}_{timestamp}.onnx")
    ExportOnnx(model, onnx_file_path) 
    print(f"Model saved at {onnx_file_path}")
    manage_saved_models()

def manage_saved_models():
    model_save_dir = args.save_mdl
    max_models = 3
    all_files = [f for f in os.listdir(model_save_dir) if f.endswith(".onnx")]
    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_save_dir, f)))    
    if len(all_files) > max_models:
        files_to_delete = all_files[:-max_models] 
        for file in files_to_delete:
            file_path = os.path.join(model_save_dir, file)
            os.remove(file_path)
            print(f"Deleted old model: {file_path}")

class ApplyTransform(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        print(f"Early Stopping Counter: {self.counter}")
        return self.early_stop

def train():
    training_data = loadData()
    model = load_resnet_model(len(LABELS))
    model.to(device)
    
    # Compute class counts and check if balanced
    sorted_labels = sorted(LABELS.items(), key=lambda x: x[1])
    class_names = [l[0] for l in sorted_labels]
    class_counts = []
    for label, idx in sorted_labels:
        class_dir = os.path.join(data_path, label)
        count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        class_counts.append(count)

    # Set all class weights to 1.0 (balanced classes)
    if len(set(class_counts)) == 1:  # All counts are equal
        class_weights = [1.0 for _ in class_counts]
    else:
        # Calculate weights as the inverse of class counts
        total_samples = sum(class_counts)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        
    class_weights = torch.tensor(class_weights).float().to(device)

    print("Class weights:", class_weights)
    print("Class names (in index order):", class_names)
    print("Class counts (in index order):", class_counts)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs

    # Shuffle and split the data
    np.random.shuffle(training_data)
    train_size = int(0.8 * len(training_data))
    train_data_list = training_data[:train_size]
    val_data_list = training_data[train_size:]

    # Simplified augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = ApplyTransform(train_data_list, transform=train_transform)
    val_dataset = ApplyTransform(val_data_list, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses = []
    val_losses = []
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []
        for batch_X, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(batch_y.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true_val = []
        y_pred_val = []
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{EPOCHS}"):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct / total

        scheduler.step(val_loss)

        # Save model every 3 epochs
        if (epoch + 1) % 3 == 0:
            save_model(model, epoch)

        target_names = [name for name, idx in sorted(LABELS.items(), key=lambda x: x[1])]

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        print("Classification report on training data: ")
        print(classification_report(y_true_train, y_pred_train, target_names=target_names, zero_division=0))

        if early_stopper(val_loss):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    model.eval()
    final_model_path = os.path.join(args.save_mdl, args.mdl_name)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    ExportOnnx(model, final_model_path)
    onnx.checker.check_model(final_model_path)

if args.build:
    Builder = buildData(data_path)
    Builder.process_folders()
    Builder.trainBuild()

if args.train:
    train()

if args.img is not None:
    onnx_model_path = args.model
    model = onnx.load(onnx_model_path)          # will throw if invalid
    onnx.checker.check_model(model)             # more checks
    ort_session = ort.InferenceSession(onnx_model_path)
    
    image_tensor = preprocess_image(args.img)
    output = test_onnx_model(ort_session, image_tensor)

    predictions = output[0][0]
    predicted_index = predictions.argmax()
    predicted_label = list(LABELS.keys())[list(LABELS.values()).index(predicted_index)]
    confidence = predictions[predicted_index]*2/10
    print(f"Predicted Animal: {predicted_label}, (Confidence: {confidence:.2f})")