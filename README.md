# sickle-cell-classification
Sickle cell Disease affects million of children all over the world. This project aims to achieve a breakthrough in making the diagnosis of Sickle Cell Disease easier.
Comparative Deep Learning Analysis for Automated Sickle Cell Diagnosis Using Red Blood Cells
This project develops and evaluates three deep learning models—Custom Convolutional Neural Network (CNN), Fine-Tuned MobileNetV2, and Fine-Tuned EfficientNetB0—for detecting sickle cell disease (SCD) from red blood cell microscopy images using the Florence Tushabe Sickle Cell Dataset (Tushabe et al., 2024). The pipeline, implemented in Python with TensorFlow, supports automated diagnostics for low-resource settings.

Models Compared
Custom CNN (lightweight, designed from scratch)

MobileNetV2 (fine-tuned transfer learning model)

EfficientNetB0 (fine-tuned transfer learning model)

Dataset
The dataset should be structured as follows:

SCD_DATA/ ├── Negative/ # Normal red blood cell images └── Positive/ # Sickle red blood cell images

Dataset Source: Kaggle - Florence Tushabe Sickle Cell Dataset (https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset). Details:
569 images (422 sickle, 147 normal) from hospitals in Soroti and Kumi districts, Uganda.

Licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

Setup Instructions
1. Clone or Download the Repository
bash

git clone https://github.com/your-username/scd-detection.git cd scd-detection

2. Create a Virtual Environment (Recommended)
bash

python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install Required Dependencies
bash

pip install tensorflow==2.4.1 numpy==1.19.5 opencv-python==4.5.5 scikit-learn==0.24.2 matplotlib==3.3.4 seaborn==0.11.2

Note: For GPU support, ensure CUDA and cuDNN are installed (compatible with TensorFlow 2.4.1). See TensorFlow GPU setup.

Running the Code
1. Save the Code
Save the provided Python code as scd_detection.py in the project folder.

2. Modify Dataset Path
In scd_detection.py, update: python

DATASET_PATH = "/path/to/SCD_DATA" # Replace with your dataset folder path

Run the Script bash
python scd_detection.py

Execution Details: The script performs: Data Preprocessing: Loads 569 images, applies augmentation (rotation, flip, zoom), and splits into 80% train, 10% validation, 10% test.

Model Training: Trains Custom CNN, MobileNetV2, and EfficientNetB0 for 50 epochs with early stopping, followed by fine-tuning (MobileNetV2, EfficientNetB0: layers from 100 unfrozen, 20 epochs) and Custom CNN dropout tuning (0.3, 0.5, 0.7).

Evaluation: Computes accuracy, precision, recall, F1-score, and AUC on the test set.

Visualization: Generates confusion matrices, ROC curves, Grad-CAM heatmaps, and training history plots.

Runtime: 10–20 minutes per model on an NVIDIA RTX 3060 GPU (1–2 hours on CPU).

Optimizer: Adam (learning rates: 1e-3 for Custom CNN, 1e-4/1e-5 for transfer learning).

Output
Console Outputs: Model summaries (layer configurations).

Training progress (epoch-wise accuracy/loss).

Evaluation metrics, e.g.:

MobileNetV2 Evaluation on Test Set: Accuracy: 0.9233 Precision: 0.9220 Recall: 0.9941 F1-Score: 0.9567 AUC: 0.9735

Best Custom CNN dropout rate (e.g., 0.5, F1-score: ~0.9199).

Final comparison (F1-score for Custom CNN, AUC for fine-tuned models).

Visual Outputs (displayed, save with plt.savefig()): Confusion Matrices: Heatmaps showing true positives, false positives, etc.

ROC Curves: Plots with AUC values.

Grad-CAM Heatmaps: Visualizations of sickle cell feature focus.

Training History Plots: Accuracy/loss curves over epochs.

Saved Files: best_cnn_model.h5: Best Custom CNN model.

fine_tuned_mobilenetv2_model.h5: Fine-tuned MobileNetV2 model.

fine_tuned_efficientnetb0_model.h5: Fine-tuned EfficientNetB0 model.

Visuals (if saved): Add plt.savefig('figure_name.png') to store plots.

Notes
Hardware: A GPU (e.g., NVIDIA RTX 3060) is recommended for faster training.

Dataset: Ensure Negative and Positive folders contain .jpg or .png images.

Licensing: The Tushabe dataset (CC BY 4.0) and tools (TensorFlow, Apache 2.0; scikit-learn, BSD 3-Clause) require proper citation for research use.

Troubleshooting
Dataset Path Error: Verify DATASET_PATH points to the correct folder with Negative and Positive subfolders.

Check image formats and folder permissions.

GPU Memory Issues: Reduce BATCH_SIZE to 16 in scd_detection.py: python

BATCH_SIZE = 16

Use a CPU or clear GPU memory.

Module Not Found: Reinstall dependencies: pip install -r requirements.txt.

Ensure TensorFlow 2.4.1 compatibility with Python (e.g., 3.8).

Poor Model Performance: Verify dataset integrity (no corrupted images).

Increase EPOCHS to 100 or adjust learning rates.

Grad-CAM Errors: Confirm layer names (conv2d_2, Conv_1, top_conv) using model.summary().

Citation
If using this pipeline, please cite the dataset:

Tushabe, F., et al., 2024. Florence Tushabe Sickle Cell Dataset. Kaggle Dataset. Available at: [https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset] [Accessed 4 May 2025].

Author
This project was developed by Bewaji Lekan as part of a dissertation for the MSc in Data Science at Edge Hill University, focusing on automated SCD detection for equitable healthcare in low-resource settings.

