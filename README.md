# MobilePlantViT-LDA

A Lightweight Hybrid CNN-Transformer Architecture for Plant Disease Classification with Linear Differential Attention

---

## 🌿 Overview

**MobilePlantViT-LDA** is a novel lightweight deep learning architecture designed for efficient plant disease classification on mobile and edge devices. It combines the local feature extraction capabilities of MobileNet-inspired CNNs with the global context understanding of Vision Transformers (ViT), featuring a custom **Linear Differential Attention (LDA)** mechanism for improved noise cancellation and feature extraction.

### Key Features

- **Hybrid Architecture**: Combines efficient CNN backbone with transformer attention
- **Linear Differential Attention**: Novel attention mechanism that computes the difference between two attention maps for noise cancellation
- **Mobile-First Design**: Optimized for deployment on resource-constrained devices (<5M parameters)
- **Multiple Variants**: Tiny (~220K), Small (~490K), Base (~867K), and Large (~1.9M) parameter configurations
- **Comprehensive Pipeline**: End-to-end training, evaluation, and deployment workflow

---

## 🏗️ Architecture

```markdown
Input (224×224×3)
       │
       ▼ CNN Stage
┌──────────────────┐
│    GhostConv     │  → Efficient feature generation with ghost features
├──────────────────┤
│   Fused-IR Block │  → Fused inverted residual for spatial processing
├──────────────────┤
│ Coordinate Attn  │  → Position-aware channel attention
└──────────────────┘
       │
       ▼ Transition Stage
┌──────────────────┐
│  Patch Embedding │  → Convert spatial features to sequence
├──────────────────┤
│ Positional Enc.  │  → Add position information
└──────────────────┘
       │
       ▼ Transformer Stage
┌──────────────────┐
│       LDA        │  → Linear Differential Attention
├──────────────────┤
│   Residual LN    │  → LayerNorm with skip connection
├──────────────────┤
│  Bottleneck FFN  │  → Parameter-efficient feed-forward
└──────────────────┘
       │
       ▼ Classifier Stage
┌──────────────────┐
│       GAP        │  → Global Average Pooling
├──────────────────┤
│  Classifier Head │  → Final classification
└──────────────────┘
       │
       ▼
Output (38 classes)
```



### Linear Differential Attention (LDA)

The core innovation of this architecture is the **Linear Differential Attention** mechanism:

```markdown
A_diff = α × (softmax(Q₁K₁ᵀ) - softmax(Q₂K₂ᵀ))
```



This differential approach:
- Cancels out noise common to both attention maps
- Enhances meaningful patterns that differ between maps
- Provides learnable noise cancellation via the α parameter

---

## 📊 Model Variants

| Variant | Parameters | embed_dim | num_heads | Use Case |
|---------|------------|-----------|-----------|----------|
| **Tiny** | ~220K | 128 | 4 | Edge devices, IoT, real-time inference |
| **Small** | ~490K | 192 | 6 | Mobile apps, balanced performance |
| **Base** | ~867K | 256 | 8 | Default choice, best accuracy/size trade-off |
| **Large** | ~1.9M | 384 | 12 | Maximum accuracy, server deployment |

---

## 📁 Project Structure

```markdown
MobilePlantViT-LDA/
├── src/
│   ├── __init__.py
│   ├── blocks/                    # Neural network building blocks
│   │   ├── __init__.py
│   │   ├── attention.py           # Linear Differential Attention
│   │   ├── classifier.py          # GAP and Classification Head
│   │   ├── coord_attention.py     # Coordinate Attention module
│   │   ├── ffn.py                 # Bottleneck Feed-Forward Network
│   │   ├── fused_ir.py            # Fused Inverted Residual Block
│   │   ├── ghost_conv.py          # Ghost Convolution
│   │   ├── patch_embed.py         # Patch Embedding
│   │   ├── positional_encoding.py # Sinusoidal Positional Encoding
│   │   └── utils.py               # Utility functions
│   └── models/
│       ├── __init__.py
│       └── mobile_plant_vit.py    # Main model implementation
├── preprocessing/
│   └── preprocessing_color.ipynb  # Data preprocessing pipeline
├── training/
│   └── training-color-mobileplantvit-large.ipynb  # Training notebook
└── README.md
```



---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/MobilePlantViT-LDA.git
cd MobilePlantViT-LDA
pip install torch torchvision numpy matplotlib pillow tqdm scikit-learn
```


### Basic Usage

```python
from src.models import MobilePlantViT, MobilePlantViTConfig

# Create model with default configuration
model = MobilePlantViT(num_classes=38)

# Or use a specific variant
from src.models import mobileplant_vit_base, mobileplant_vit_tiny

model = mobileplant_vit_base(num_classes=38)  # ~867K params
model = mobileplant_vit_tiny(num_classes=38)  # ~220K params

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)
output = model(x)  # Returns probabilities
logits = model.get_logits(x)  # Returns raw logits for CrossEntropyLoss
```


### Custom Configuration

```python
from src.models import MobilePlantViT, MobilePlantViTConfig

config = MobilePlantViTConfig(
    img_size=224,
    num_classes=38,
    ghost_out_channels=64,
    fused_ir_out_channels=64,
    embed_dim=256,
    num_heads=8,
    lda_dropout=0.1,
    ffn_bottleneck_ratio=0.25,
)

model = MobilePlantViT(config)
print(f"Parameters: {model.count_parameters():,}")
```


---

## 📦 Building Blocks

### GhostConv

Efficient convolution using ghost features to reduce computation:

```python
from src.blocks import GhostConv

ghost = GhostConv(inp=64, oup=128, kernel_size=1, ratio=2)
```


### Coordinate Attention

Position-aware channel attention mechanism:

```python
from src.blocks import CoordAtt

coord_att = CoordAtt(inp=64, oup=64, reduction=32)
```


### Linear Differential Attention

The core attention mechanism:

```python
from src.blocks import LinearDifferentialAttention

lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.1)
```


---

## 🔧 Training

### Data Preprocessing

The preprocessing pipeline handles:
- Dataset verification and corruption detection
- Duplicate image detection using perceptual hashing
- Class name harmonization
- Train/Val/Test splitting (70/15/15)
- Class imbalance analysis

Run the preprocessing notebook:

```bash
jupyter notebook preprocessing/preprocessing_color.ipynb
```


### Training Pipeline

The training notebook includes:
- Data augmentation (rotation, flip, color jitter, perspective)
- Mixed precision training (AMP)
- Cosine warmup learning rate schedule
- Gradient clipping for transformer stability
- Early stopping
- Comprehensive logging and visualization

Key training configurations:

```python
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 64,
    'optimizer': 'adamw',
    'learning_rate': 2e-4,
    'weight_decay': 0.01,
    'scheduler': 'cosine_warmup',
    'warmup_epochs': 3,
    'gradient_clip_max_norm': 1.0,
    'use_mixed_precision': True,
}
```


---

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 95%+ |
| Top-3 Accuracy | 99%+ |
| Top-5 Accuracy | 99.5%+ |
| Parameters | <5M (all variants) |
| Inference Time | <10ms/image (GPU) |

### Comparison with Baseline

| Model | Parameters | Accuracy | Size (MB) |
|-------|------------|----------|-----------|
| MobileNetV2 | 3.5M | ~96% | 13.5 |
| **MobilePlantViT-Base** | 867K | ~95% | 3.3 |
| **MobilePlantViT-Tiny** | 220K | ~93% | 0.9 |

---

## 📤 Model Export

The trained model can be exported in multiple formats:

### PyTorch Checkpoint
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model.get_config(),
}, 'model.pth')
```


### TorchScript
```python
traced_model = torch.jit.trace(model, sample_input)
traced_model.save('model_traced.pt')
```


### ONNX
```python
torch.onnx.export(model, sample_input, 'model.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})
```


---

## 🔬 Technical Details

### Attention Mechanism

The Linear Differential Attention computes:

1. Project input to Q₁, Q₂, K₁, K₂, V
2. Compute attention scores: `A₁ = softmax(Q₁K₁ᵀ/√d)`, `A₂ = softmax(Q₂K₂ᵀ/√d)`
3. Differential attention: `A_diff = α × (A₁ - A₂)`
4. Apply to values: `output = A_diff × V`

### Parameter Efficiency

The architecture achieves parameter efficiency through:
- **GhostConv**: Generates features cheaply via depthwise operations
- **Bottleneck FFN**: Contracts then expands (opposite of standard transformer)
- **Single Transformer Block**: Minimal transformer overhead
- **Efficient Projections**: Careful dimensionality choices

---

## 📚 References

- [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
- [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [DIFF Transformer](https://arxiv.org/abs/2410.05258)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## ⭐ Acknowledgments

- PlantVillage dataset for providing the plant disease images
- The PyTorch team for the excellent deep learning framework
- The research community for the foundational architectures and techniques