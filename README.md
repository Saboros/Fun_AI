# Fun_AI

# AI Drowning Detection System - Thesis Prototype

This repository contains a prototype AI system for drowning detection using multi-modal sensor data, developed as part of a thesis project.

## 📋 Overview

The system combines multiple AI models to detect potential drowning incidents by analyzing:
- Heart rate data (using LSTM)
- Motion patterns (using 1D CNN)
- Environmental context (integrated in fusion model)

## ⚠️ Important Disclaimer

⚠️ **This is a research prototype using synthetic datasets for demonstration purposes only.**
- Datasets are artificially generated and do not represent real-world scenarios
- This system is NOT intended for actual drowning prevention or emergency response only for prototyping.
- Always use proper safety equipment and supervision near water

## 🏗️ System Architecture

### Individual Models
- **Heart Rate Analysis**: LSTM-based model (`heartmodel.py`)
- **Motion Classification**: 1D CNN model (`motion_classifier.py`)
- **Fusion/Decision**: Multi-modal fusion model (`model.py`)

### Structure
Each model component includes:
- `model_*.py` - Model architecture and implementation
- `preprocess_*.py` - Data preprocessing pipeline
- `train_*.py` - Training script
- `test_*.py` - Testing and prediction scripts

## 🔄 Refactoring
The fusion model was refactored for improved code quality and maintainability with AI assistance. See `REFACTORING_SUMMARY.md` for detailed changes.

## 📁 Dataset Requirements
- Dataset format must align with the provided synthetic dataset structure
- See `data/` directory for example format

## 🚀 Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Train individual models: `python train_heart.py` and `python train_motion.py`
3. Train fusion model: `python fusion_train.py`
4. Test the system: `python test_fusion.py`

## 🧪 Testing
Each model includes test scripts to verify predictions and tensor outputs.

## 📚 Thesis Context
This prototype serves as a proof-of-concept for multi-modal AI approaches in aquatic safety monitoring research.
