# NYCU Deep Learning Practice 2025 Spring

This repository contains lab assignments for the Deep Learning Practice course at National Yang Ming Chiao Tung University (NYCU) for Spring 2025.

## 📋 Course Overview

This course covers practical implementations of various deep learning models and techniques through hands-on laboratory exercises.

## 🗂️ Lab Assignments

### Lab 2: Binary Semantic Segmentation
**Topic**: Semantic segmentation for binary classification tasks
- **Framework**: PyTorch
- **Key Files**: 
  - `src/train.py` - Training script
  - `src/evaluate.py` - Evaluation utilities
  - `src/inference.py` - Model inference
  - `src/oxford_pet.py` - Dataset handling
- **Dependencies**: torch, torchvision, pillow, tqdm, albumentations, tensorboardX

### Lab 3: MaskGiT for Image Inpainting
**Topic**: Masked Generative Image Transformer for image inpainting tasks
- **Framework**: PyTorch with custom transformer implementation
- **Key Files**:
  - `training_transformer.py` - Transformer training
  - `inpainting.py` - Image inpainting inference
  - `models/VQGAN_Transformer.py` - VQGAN + Transformer architecture
- **Environment**: Use `environment.yml` for conda setup
- **Features**: VQGAN tokenization, masked transformer training

### Lab 4: Conditional Video Prediction
**Topic**: Conditional video prediction with VAE-based architecture for human pose sequences
- **Framework**: PyTorch
- **Dataset**: Dance dataset with human pose sequences
- **Key Files**:
  - `Trainer.py` - Training pipeline with PSNR evaluation
  - `Tester.py` - Testing and inference utilities
  - `dataloader.py` - Video sequence data loading
  - `modules/modules.py` - Generator, RGB_Encoder, Gaussian_Predictor, Decoder_Fusion, Label_Encoder
  - `modules/layers.py` - Custom convolutional layers and residual blocks
- **Features**: Conditional generation, pose-guided video prediction, multiple training strategies
- **Checkpoints**: Multiple configurations (cyclical_full, wo, 600 epochs)

### Lab 5: Deep Q-Network (DQN)
**Topic**: Reinforcement learning with Deep Q-Networks
- **Framework**: PyTorch + Gymnasium
- **Key Files**:
  - `dqn_task3.py` - DQN implementation
  - `test_model_task2.py` - Model testing
  - `find_seed.py` - Reproducibility utilities
- **Environment**: Atari Pong game
- **Features**: Video evaluation recording

### Lab 6: Generative Models (DDPM)
**Topic**: Conditional Denoising Diffusion Probabilistic Models
- **Framework**: PyTorch
- **Dataset**: i-CLEVR (conditional image generation)
- **Key Files**:
  - `train.py` - DDPM training
  - `test.py` - Model testing and evaluation
  - `ddpm.py` - DDPM implementation
  - `dataset.py` - Data loading utilities
- **Features**: Conditional generation based on object specifications

### Lab 7: Reinforcement Learning (A2C & PPO)
**Topic**: Actor-Critic methods and Proximal Policy Optimization
- **Framework**: PyTorch + Gymnasium
- **Environments**: Pendulum, BipedalWalker
- **Key Files**:
  - `a2c_pendulum.py` - Advantage Actor-Critic implementation
  - `ppo_pendulum.py` - PPO for Pendulum environment
  - `ppo_walker.py` - PPO for BipedalWalker environment
- **Features**: Multiple trained models with different configurations

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Shyandram/NYCU_DLP_2025Spring.git
cd NYCU_DLP_2025Spring
```

2. For each lab, navigate to the respective directory and install dependencies:
```bash
cd Lab[X]/code
pip install -r requirements.txt  # For labs with requirements.txt
# OR
conda env create -f environment.yml  # For labs with conda environment
```

### Running the Labs

Each lab directory contains specific instructions. Generally:

1. **Training**: Run the main training script (e.g., `train.py`, `training_transformer.py`)
2. **Testing**: Use evaluation/test scripts to assess model performance
3. **Inference**: Use inference scripts for generating predictions

## 📊 Results and Models

- Trained models are saved in respective `checkpoints/` or `saved_models/` directories
- Evaluation videos and results are stored in `eval_videos/` and `results_*/` folders
- Reports and documentation are available as PDF files in each lab directory

## 🔧 Key Technologies

- **Deep Learning**: PyTorch, PyTorch Lightning
- **Computer Vision**: Semantic Segmentation, Image Inpainting, Generative Models
- **Reinforcement Learning**: DQN, A2C, PPO
- **Generative Models**: VQGAN, Transformers, DDPM
- **Evaluation**: FID scores, custom metrics

## 📝 Reports

Each lab includes detailed reports in PDF format documenting:
- Methodology and implementation details
- Experimental results and analysis
- Performance comparisons
- Conclusions and insights

## 🎯 Learning Objectives

- Implement state-of-the-art deep learning architectures
- Understand training dynamics and optimization techniques
- Apply deep learning to various domains (CV, RL, Generative AI)
- Develop practical skills in model evaluation and analysis

## 👨‍🎓 Student Information

- **Student ID**: 413551036
- **Name**: 翁祥恩
- **Course**: Deep Learning Practice
- **Semester**: Spring 2025
- **University**: National Yang Ming Chiao Tung University (NYCU)

## 📄 License

None

---

*Note: Some labs may require additional setup steps or dataset downloads. Please refer to individual lab README files or documentation for specific requirements.*
