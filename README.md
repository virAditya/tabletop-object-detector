# Tabletop Object Detection & Analysis System

A classical computer vision system for detecting, segmenting, and analyzing objects on a table using fundamental CV techniques—**no deep learning required**. This project demonstrates image formation theory, binary image processing, and segmentation algorithms from first principles.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This project implements a complete classical computer vision pipeline that captures overhead images of objects on a table using a phone camera (via DroidCam), processes them through multiple stages, and outputs detailed geometric analysis including:

- **Object Detection**: Connected component analysis for segmentation
- **Feature Extraction**: Centroid coordinates, area, dimensions, orientation
- **Intelligent Filtering**: Position-based, size-based, and aspect-ratio filtering
- **Structured Logging**: Outputs in JSON, CSV, and TXT formats

### Why Classical CV?

While deep learning dominates modern computer vision, understanding classical techniques provides:
- ✅ **No training data required** - Works immediately
- ✅ **Lightweight & fast** - Runs on any hardware
- ✅ **Mathematically interpretable** - Clear understanding of results
- ✅ **Foundation knowledge** - Essential for CV engineering roles

---

## ✨ Features

### Core Capabilities
- 📸 **Live camera countdown** (3-second preview with adjustment time)
- 🔄 **Automatic preprocessing** (grayscale conversion, Gaussian blur, morphological operations)
- 🎯 **Precise segmentation** (Otsu's thresholding + connected components)
- 📐 **Geometric analysis** (orientation using second-order image moments)
- 🎨 **Visual annotations** (bounding boxes, centroids, orientation lines)
- 📊 **Multi-format logging** (JSON, CSV, TXT)

### Intelligent Filtering
- **Position filtering**: Removes DroidCam overlay artifacts (top 15%)
- **Size filtering**: Configurable minimum area threshold (default: 2000px)
- **Shape filtering**: Removes elongated text-like objects (aspect ratio filtering)

---

## 🖼️ Demo

### Input → Processing → Output

**Original Image** → **Binary Segmentation** → **Annotated Results**


---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.7+ | Core implementation |
| **CV Library** | OpenCV 4.8+ | Image processing |
| **Math** | NumPy 1.24+ | Array operations |
| **Camera** | DroidCam | Phone as webcam |

### Algorithms Used

- **Binarization**: Otsu's automatic thresholding
- **Morphology**: Opening and closing operations
- **Segmentation**: 8-connectivity connected component analysis
- **Orientation**: Second-order central moments (μ₂₀, μ₀₂, μ₁₁)

