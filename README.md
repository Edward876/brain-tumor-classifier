
# Brain Tumor Detection Using MRI and CT Images

This project implements a deep learning approach for detecting brain tumors using MRI and CT image data. The project leverages transfer learning with MobileNet to classify healthy and tumor images from multimodal datasets.

## Features

- **Dataset Support:** Processes multimodal MRI and CT image datasets.
- **Deep Learning Models:** Utilizes MobileNet for transfer learning to classify brain tumor images.
- **Preprocessing Pipeline:** Includes image normalization, resizing, and dataset splitting.
- **Prediction:** Combines predictions from MRI and CT models to enhance accuracy.

## Project Structure

- **`main.ipynb`**: The primary notebook for training, evaluating, and testing the models.
- **Datasets**: Downloaded from Kaggle, consisting of MRI and CT brain tumor images.
- **Saved Models**: Pretrained MobileNet models fine-tuned for MRI and CT modalities (`mri_model_mobilenet.h5`, `ct_model_mobilenet.h5`).

## Dataset

The dataset is sourced from Kaggle and contains:

- **MRI Images**
  - Healthy: 1997 images
  - Tumor: 2984 images
- **CT Images**
  - Healthy: 1616 images
  - Tumor: 2984 images

The dataset is preprocessed into a tabular format with the following columns:
- `path`: Path to the image file.
- `label`: Classification label (`0` for healthy, `1` for tumor).
- `modality`: Image modality (`MRI` or `CT`).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and extract it to the specified path in the notebook.

## Training and Evaluation

1. Open the `main.ipynb` notebook.
2. Specify the paths to the MRI and CT datasets.
3. Run the preprocessing, training, and evaluation cells to train MobileNet models for MRI and CT images.
4. The training process outputs the trained models:
   - `mri_model_mobilenet.h5`
   - `ct_model_mobilenet.h5`

## Prediction

Use the `predict_tumor_single` function to classify a single image:

```python
tumor_detected = predict_tumor_single(
    mri_model_path="mri_model_mobilenet.h5",
    ct_model_path="ct_model_mobilenet.h5",
    img_path="path_to_image.png"
)
print(f"Tumor Detected: {tumor_detected}")
```

## Results

Sample accuracy metrics:
- MRI Model: ~97.8% validation accuracy
- CT Model: ~98.1% validation accuracy

## Future Improvements

- Incorporate additional data augmentation techniques.
- Explore ensemble models for improved accuracy.
- Add a web-based interface for easier interaction.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.


