# Satellite Image Understanding with PyTorch: From SAR Concepts to CNN Classification

This repository is a learning project for understanding satellite imagery, SAR imaging concepts, and deep learning implementation using PyTorch.

The goal is to connect ideas from GPR signal processing to satellite imagery processing.

In GPR, we may have:

[
A\text{-scan signal} \rightarrow Neural Network \rightarrow Continuous Labels
]

In satellite imagery, we may have:

[
Satellite Image \rightarrow CNN \rightarrow Class Prediction
]

or:

[
Satellite Image \rightarrow CNN/Transformer \rightarrow Continuous Prediction
]

For example, a satellite image can be used for:

* Land-cover classification
* Crop monitoring
* Soil moisture estimation
* Vegetation health analysis
* Flood detection
* Urban area detection
* Water body detection
* SAR-based surface roughness analysis

---

# 1. Conceptual Background

## 1.1 What Is Satellite Data?

Satellite data is information collected by sensors mounted on satellites orbiting Earth.

Depending on the sensor, satellite data can come in different forms.

Common types include:

| Data Type             | Description                                                 | Example Use                                                 |
| --------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| RGB imagery           | Red, green, and blue bands, similar to normal camera images | Visual land-cover classification                            |
| Multispectral imagery | Multiple spectral bands beyond RGB, such as NIR and SWIR    | Vegetation, crop health, soil moisture estimation           |
| Thermal imagery       | Measures heat emitted from Earth surface                    | Surface temperature, drought stress                         |
| SAR imagery           | Radar-based imaging using microwave signals                 | Cloud-independent imaging, surface roughness, soil moisture |
| Hyperspectral imagery | Hundreds of narrow spectral bands                           | Material identification, crop disease, mineral mapping      |

---

# 2. RGB Satellite Imagery

RGB satellite imagery is similar to normal photography.

Each pixel has three values:

[
Pixel = [R, G, B]
]

Where:

* (R) = red reflectance
* (G) = green reflectance
* (B) = blue reflectance

For a satellite image of size (64 \times 64), the data shape is:

[
3 \times 64 \times 64
]

This means:

```text
3 channels × 64 height × 64 width
```

RGB images are easy to visualize, but they do not contain all the physical information satellites can measure.

---

# 3. Multispectral Satellite Imagery

Multispectral imagery includes more than RGB.

For example, Sentinel-2 has bands such as:

* Blue
* Green
* Red
* Near-infrared
* Red edge
* Shortwave infrared

A multispectral pixel may look like:

[
Pixel = [B_1, B_2, B_3, ..., B_{13}]
]

So instead of only 3 values per pixel, we may have 13 values per pixel.

A multispectral image can be represented as:

[
13 \times H \times W
]

This is very powerful because different materials reflect different wavelengths differently.

Vegetation, soil, water, buildings, and roads may look similar in RGB but become more separable in multispectral space.

---

# 4. NDVI: A Simple Vegetation Index

One famous index is NDVI.

NDVI stands for:

```text
Normalized Difference Vegetation Index
```

The formula is:

[
NDVI = \frac{NIR - Red}{NIR + Red}
]

Where:

* (NIR) = near-infrared reflectance
* (Red) = red reflectance

Healthy vegetation usually reflects strongly in near-infrared and absorbs red light.

So high NDVI often means healthier vegetation.

Example:

[
NDVI = \frac{0.72 - 0.18}{0.72 + 0.18}
]

[
NDVI = \frac{0.54}{0.90} = 0.60
]

An NDVI value around 0.60 usually indicates healthy vegetation.

---

# 5. SAR Imaging Concept

SAR stands for:

```text
Synthetic Aperture Radar
```

SAR is different from RGB or infrared imaging.

RGB and infrared sensors passively receive light reflected from the Sun.

SAR actively sends microwave radar pulses toward Earth and listens for the returning echo.

The satellite does not need sunlight.

That means SAR can image Earth:

* during the day
* at night
* through clouds
* in many weather conditions

---

## 5.1 SAR Explained Simply

Imagine the satellite is like a bat.

A bat sends sound and listens to the echo.

SAR does something similar, but instead of sound, it sends microwave radar waves.

The radar wave travels from the satellite to Earth, hits the surface, and bounces back.

The satellite records:

1. How strong the echo is
2. How long the echo takes to return
3. The phase of the returning signal
4. How the echo changes as the satellite moves

From many echoes, SAR software creates an image.

---

## 5.2 What Type of Wave Does SAR Use?

SAR uses electromagnetic microwave waves.

These waves are usually in radar frequency bands such as:

| Band   | Approximate Frequency | Approximate Wavelength | Common Use                         |
| ------ | --------------------: | ---------------------: | ---------------------------------- |
| X-band |              8–12 GHz |           ~2.5–3.75 cm | High-resolution imaging            |
| C-band |               4–8 GHz |           ~3.75–7.5 cm | General Earth observation          |
| L-band |               1–2 GHz |              ~15–30 cm | Vegetation and soil penetration    |
| P-band |             0.3–1 GHz |             ~30–100 cm | Deeper vegetation/forest structure |

The wave frequency is not the same as the number of pulses per second.

Frequency means how fast the wave oscillates.

Pulse repetition frequency means how many radar pulses are sent per second.

---

## 5.3 Frequency vs Pulse Repetition Frequency

These are different concepts.

### Radar Frequency

Radar frequency describes the microwave itself.

Example:

[
f = 5.4 \ GHz
]

This means the wave oscillates 5.4 billion times per second.

### Pulse Repetition Frequency

Pulse repetition frequency, or PRF, describes how many radar pulses are emitted per second.

Example:

[
PRF = 5000 \ pulses/second
]

So SAR may send thousands of radar pulses per second, but each pulse itself is a GHz-frequency microwave signal.

---

## 5.4 Why Not Send Infinite Pulses?

More pulses can help improve image quality, but there are trade-offs.

Higher pulse rates require:

* More power
* More onboard storage
* More communication bandwidth
* More processing
* Careful timing to avoid signal ambiguity

So SAR systems are designed to balance resolution, coverage, power, and data volume.

---

# 6. SAR Grayscale Image Meaning

A raw SAR image is often shown in grayscale.

The brightness of each pixel represents the strength of the returned radar echo.

| SAR Pixel Brightness | Meaning                 |
| -------------------- | ----------------------- |
| Bright pixel         | Strong radar reflection |
| Dark pixel           | Weak radar reflection   |

Bright areas may be caused by:

* buildings
* metal structures
* rough surfaces
* rocky terrain
* strong geometric reflections

Dark areas may be caused by:

* calm water
* smooth surfaces
* shadow regions
* surfaces that reflect radar away from the satellite

Important:

A bright SAR pixel does not simply mean “dense material.”

It is more accurate to say SAR brightness depends on:

* surface roughness
* geometry
* moisture
* dielectric properties
* radar wavelength
* incidence angle
* object orientation

---

# 7. SAR vs GPR Conceptual Connection

In GPR, we used radar signals near the ground.

For example:

[
GPR \rightarrow A\text{-scan} \rightarrow 1D Signal
]

An A-scan may contain thousands of time-signal values.

In SAR, the radar is on a satellite.

Instead of one A-scan, SAR collects many radar echoes while the satellite moves.

Those echoes are processed into an image.

Conceptually:

[
Many\ Radar\ Echoes \rightarrow SAR\ Processing \rightarrow Grayscale\ Image
]

So the relationship is:

| GPR                     | SAR                             |
| ----------------------- | ------------------------------- |
| Ground-based radar      | Satellite-based radar           |
| Often near the surface  | Hundreds of kilometers away     |
| A-scan, B-scan, C-scan  | Radar image                     |
| Time-domain reflections | Processed microwave backscatter |
| Subsurface focus        | Surface and near-surface focus  |

---

# 8. Deep Learning Example: EuroSAT Classification

In this project, we use the EuroSAT dataset as a beginner-friendly satellite image example.

EuroSAT contains Sentinel-2 satellite image patches for land-use and land-cover classification.

The RGB version has images of shape:

[
3 \times 64 \times 64
]

The output is one of 10 classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

This is a classification problem.

Later, the same structure can be modified for regression problems such as soil moisture prediction.

---

# 9. Installation

```bash
pip install torch torchvision matplotlib scikit-learn seaborn numpy pandas
```

---

# 10. Full PyTorch Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
```

---

# 11. Select Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

If GPU is available, PyTorch uses GPU.

Otherwise, it uses CPU.

---

# 12. Image Transformations

```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

This does three things:

1. Resize every image to (64 \times 64)
2. Convert image to tensor
3. Normalize RGB values

The final tensor shape is:

[
3 \times 64 \times 64
]

---

# 13. Load EuroSAT Dataset

```python
dataset = EuroSAT(
    root="./data",
    download=True,
    transform=transform
)

class_names = dataset.classes

print("Number of images:", len(dataset))
print("Classes:", class_names)
```

---

# 14. Train / Validation / Test Split

```python
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

---

# 15. Plot Sample Satellite Images

```python
def unnormalize_image(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img

images, labels = next(iter(train_loader))

plt.figure(figsize=(12, 8))

for i in range(12):
    img = unnormalize_image(images[i])
    img = img.permute(1, 2, 0)

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
```

This plot shows random satellite image samples from the training data.

---

# 16. Plot Class Distribution

```python
all_labels = []

for _, label in dataset:
    all_labels.append(label)

label_counts = pd.Series(all_labels).value_counts().sort_index()

plt.figure(figsize=(12, 5))
plt.bar(class_names, label_counts.values)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("EuroSAT Class Distribution")
plt.tight_layout()
plt.show()
```

This plot shows how many images exist for each class.

---

# 17. Define CNN Model

```python
class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SatelliteCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

# 18. Create Model, Loss, and Optimizer

```python
model = SatelliteCNN(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

Since this is classification, we use:

[
CrossEntropyLoss
]

For regression, we would use:

[
MSELoss
]

---

# 19. Training Function

```python
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc
```

This function trains the model for one epoch.

Important steps:

```python
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Meaning:

1. Predict
2. Calculate error
3. Backpropagate
4. Update weights

---

# 20. Evaluation Function

```python
def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_predictions, all_labels
```

During evaluation, we use:

```python
with torch.no_grad():
```

because we do not need to update the model weights.

---

# 21. Train the Model

```python
num_epochs = 10

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )
```

---

# 22. Plot Training and Validation Loss

```python
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
```

If training loss decreases but validation loss increases, the model may be overfitting.

---

# 23. Plot Training and Validation Accuracy

```python
plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

This plot shows whether the model is improving over time.

---

# 24. Test the Model

```python
test_loss, test_acc, y_pred, y_true = evaluate(
    model,
    test_loader,
    criterion
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
```

---

# 25. Plot Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()
```

The confusion matrix shows which classes the model confuses with each other.

---

# 26. Plot Example Predictions

```python
model.eval()

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

images = images.cpu()
predictions = predictions.cpu()

plt.figure(figsize=(12, 8))

for i in range(12):
    img = unnormalize_image(images[i])
    img = img.permute(1, 2, 0)

    true_label = class_names[labels[i]]
    predicted_label = class_names[predictions[i]]

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

This plot shows example images with their true and predicted labels.

---

# 27. Simulated SAR Wave Concept Plot

This is not real SAR data.

This is a simple educational plot to understand the idea of transmitted and returned radar waves.

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)

frequency = 30
transmitted_wave = np.sin(2 * np.pi * frequency * t)

delay = 80
attenuation = 0.45

returned_wave = np.zeros_like(transmitted_wave)
returned_wave[delay:] = attenuation * transmitted_wave[:-delay]

plt.figure(figsize=(10, 5))
plt.plot(t, transmitted_wave, label="Transmitted Radar Wave")
plt.plot(t, returned_wave, label="Returned Echo")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Conceptual SAR Radar Pulse and Returned Echo")
plt.legend()
plt.grid(True)
plt.show()
```

This shows the idea that a radar signal is transmitted and a delayed weaker echo returns.

---

# 28. Simulated SAR Grayscale Image

This is also educational, not real SAR data.

It creates a synthetic grayscale image where brighter pixels represent stronger radar backscatter.

```python
np.random.seed(42)

height = 128
width = 128

sar_image = np.random.normal(loc=0.3, scale=0.1, size=(height, width))

# Add a bright urban-like region
sar_image[30:70, 40:90] += 0.5

# Add a dark water-like region
sar_image[85:115, 10:60] -= 0.25

sar_image = np.clip(sar_image, 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(sar_image, cmap="gray")
plt.title("Synthetic SAR-like Grayscale Image")
plt.colorbar(label="Backscatter Strength")
plt.axis("off")
plt.show()
```

Interpretation:

* Bright areas = stronger radar return
* Dark areas = weaker radar return

---

# 29. Synthetic SAR Histogram

```python
plt.figure(figsize=(8, 5))
plt.hist(sar_image.flatten(), bins=40)
plt.xlabel("Backscatter Strength")
plt.ylabel("Pixel Count")
plt.title("Histogram of Synthetic SAR Backscatter Values")
plt.grid(True)
plt.show()
```

This plot shows the distribution of SAR-like backscatter pixel values.

---

# 30. Classification vs Regression

The EuroSAT example is classification.

The output is a class:

```text
Forest
River
Residential
SeaLake
```

So the final layer is:

```python
nn.Linear(256, 10)
```

and the loss is:

```python
nn.CrossEntropyLoss()
```

For soil moisture estimation, we need regression.

The output is a continuous value:

```text
soil moisture = 0.34
```

So the final layer becomes:

```python
nn.Linear(256, 1)
```

and the loss becomes:

```python
nn.MSELoss()
```

For six continuous labels, like in a GPR-style problem:

```python
nn.Linear(256, 6)
```

and:

```python
nn.MSELoss()
```

---

# 31. Regression Model Version

```python
class SatelliteRegressionCNN(nn.Module):
    def __init__(self, output_dim=1):
        super(SatelliteRegressionCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
```

For one output:

```python
model = SatelliteRegressionCNN(output_dim=1).to(device)
criterion = nn.MSELoss()
```

For six outputs:

```python
model = SatelliteRegressionCNN(output_dim=6).to(device)
criterion = nn.MSELoss()
```

---

# 32. Connection to GPR Research

This satellite learning example is conceptually similar to GPR modeling.

In GPR:

[
Signal \in \mathbb{R}^{10000}
]

The model predicts:

[
Y \in \mathbb{R}^{6}
]

In satellite imagery:

[
Image \in \mathbb{R}^{C \times H \times W}
]

The model predicts either:

[
Class \in {0,1,2,...,9}
]

or:

[
Y \in \mathbb{R}^{k}
]

where (k) can be one continuous value or multiple continuous labels.

Examples:

```text
soil moisture
crop stress
vegetation index
surface temperature
roughness
biomass
```

---

# 33. CNN vs Transformer Intuition

A CNN is good for local spatial patterns.

It looks at small windows such as:

[
3 \times 3
]

This is useful for:

* edges
* textures
* small crop patterns
* building boundaries
* road shapes
* water boundaries

A Transformer is better for global relationships.

It can compare distant parts of the image.

This is useful when the meaning of one region depends on another region.

Conceptually:

| Model             | Strength                     |
| ----------------- | ---------------------------- |
| CNN               | Local details                |
| Transformer       | Global context               |
| CNN + Transformer | Local + global understanding |

This is similar to the idea that one method sees details and another method sees the broader structure.

---

# 34. Future Work

Possible next steps:

1. Use full 13-band Sentinel-2 data instead of RGB only.
2. Add NDVI as an additional input channel.
3. Train a regression model for soil moisture estimation.
4. Compare CNN, ResNet, Vision Transformer, and CNN-Transformer hybrid models.
5. Use SAR imagery for cloud-independent soil and surface analysis.
6. Use real soil moisture datasets such as SMAP or Sentinel-1-based products.
7. Combine optical, infrared, and SAR data for multimodal learning.

---

# 35. Summary

This repository demonstrates how to use PyTorch for satellite image understanding.

The key idea is:

[
Remote\ Sensing\ Data \rightarrow Deep\ Learning\ Model \rightarrow Prediction
]

For EuroSAT:

[
RGB\ Satellite\ Image \rightarrow CNN \rightarrow Land\ Cover\ Class
]

For future soil moisture estimation:

[
Satellite/SAR/Multispectral\ Image \rightarrow CNN/Transformer \rightarrow Moisture\ Value
]

This project is a beginner-friendly bridge between GPR-based machine learning and satellite remote sensing.
