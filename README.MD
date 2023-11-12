# Phrex - Face Recognition AI Model

## Overview

Phrex is a face recognition AI model designed to verify faces using the Digiface 1M dataset. What sets Phrex apart is its unique approach to achieving better accuracy by leveraging multiple images (where x > 1) for face classification. This multi-image verification system provides a significant boost in accuracy compared to other alternatives.

## Features

- **Multi-Image Verification:** Phrex utilizes more than one image for face verification, enhancing accuracy and reliability in classification.

- **Digiface 1M Dataset:** The model is trained on the Digiface 1M dataset, ensuring a diverse and comprehensive set of facial features for robust recognition.

- **Python Implementation:** Developed in Python, making it accessible and easy to integrate with existing projects or frameworks.

- **Low-End Device Compatibility:** Phrex is designed to be resource-efficient and can be trained on low-end devices with a minimum of 1GB RAM.

## Installation

To use Phrex, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/sh3ldr0id/phrex.git
   ```

2. Navigate to the project directory:

   ```bash
   cd phrex
   ```

3. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Linux or macOS
   .\venv\Scripts\activate       # On Windows
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training:**
   - Run the training script:

     ```bash
     python Model.py
     ```

2. **Verification:**
   - Use the trained model for face verification:

     ```python
     # Not yet implemented!
     ```

## Compatibility

Phrex is designed to be trainable on a wide range of devices, including low-end devices with a minimum of 1GB RAM.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Digiface 1M dataset used in this project.

Feel free to contribute to Phrex by opening issues or submitting pull requests. Happy face recognition!