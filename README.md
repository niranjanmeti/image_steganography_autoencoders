# 🧠 Image Steganography using Neural Networks and Autoencoders

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This MATLAB project demonstrates *image steganography using convolutional autoencoders*, where one image (the secret) is invisibly embedded into another (the cover). A dedicated extraction network is used to recover the secret image later with high fidelity.

---

## 📌 Project Highlights

- MATLAB implementation with Deep Learning Toolbox  
- Embedding and Extraction CNN-based networks  
- Trains on 1000 pairs of images resized to 256×256  
- Stego images generated and saved to disk  
- Secret images reconstructed with high perceptual quality  
- PSNR and SSIM used for evaluation  
- Includes code, report, and presentation

---

## 📐 Block Diagrams

### 🔒 Embedding Network  
![Embedding Network](images/Embedding_network.jpg)

### 🔓 Extraction Network  
![Extraction Network](images/Extraction_network.jpg)
---

## 🧠 Architecture Overview

- *Embedding Network*  
  Inputs: cover and secret images  
  Layers: parallel conv layers → concatenation → more convs → stego image

- *Extraction Network*  
  Input: stego image  
  Layers: conv stack → relu activations → reconstruct secret

- Input size: *256×256×3*  
- Optimizer: *Adam*, learning rate: 0.001  
- Epochs: 10 (can be tuned)

---

## 🚀 How to Run

1. Launch MATLAB or MATLAB Online  
2. Clone/download this repo  
3. Edit code/steganography_pipeline.m to set correct paths for:
   - cover_dataset/
   - secret_dataset/
   - stego_images_output/
4. Run the script  
5. Output includes:
   - Saved stego images  
   - Extracted secret images  
   - PSNR and SSIM averages

---

## 📊 Sample Results

| Metric | Value |
|--------|-------|
| PSNR   | ~30.5 dB |
| SSIM   | ~0.93   |

---

## 🗂 Repository Structure
---

## 📄 License

This project is licensed under the *MIT License*.
> Full license text is available in the LICENSE file

---

## 👤 Author

*Niranjan Meti*  
Deep learning + embedded systems enthusiast  
📫 [LinkedIn](https://www.linkedin.com/niranjan-meti-69076921a)  
🌐 [GitHub](https://github.com/niranjanmeti)

---

## 🌟 Star this repo if you found it useful!