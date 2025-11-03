# RFI
RFI Detection and Analysis using CNN
Overview: This project focuses on developing a deep learning-based system for automatic detection and classification of Radio Frequency Interference (RFI) in radio astronomy data. Using simulated data derived from FITS files, the system leverages Convolutional Neural Networks (CNNs) to identify interference patterns across time–frequency spectrograms.

Project Workflow
1. Data Generation

Synthetic RFI data is created and stored in FITS (Flexible Image Transport System) format.
Each FITS file contains multiple layers (real, imaginary, magnitude, and binary mask) representing signal information across frequency and time.
Doppler effects are optionally simulated to represent moving sources and dynamic spectrum variations.

2. Preprocessing

The FITS data is read and converted into 2D arrays (NChan × NTime), representing spectrograms of the received signal.
Binary RFI masks are applied to isolate regions of interference.
Normalization and resizing are performed to prepare data for CNN input.

3. Model Design (CNN)

A 2D Convolutional Neural Network is trained to detect and classify RFI regions in the time–frequency domain.
Input: spectrogram images
Output: RFI / non-RFI prediction or segmentation mask
The CNN learns spatial–temporal features of interference patterns, making it adaptable to different noise environments.

4. Evaluation & Visualization

Performance is evaluated using metrics such as accuracy, precision, recall and F1-score.

5. Visualization tools plot:

Original spectrogram
RFI mask
Model predictions and comparison with ground truth
