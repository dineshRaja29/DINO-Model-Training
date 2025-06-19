# Fine-Tuning DINO on CIFAR-10 Binary Classification (PyTorch)

This project demonstrates how to fine-tune the [DINO](https://github.com/facebookresearch/dino) (self-supervised ViT) model from Facebook AI using a custom CSV dataset. We perform binary classification using a modified CIFAR-10 dataset, where:

- Class `0` is labelled as `0`
- All other classes (`1-9`) are grouped and labelled as `1`
- The resulting dataset is imbalanced, and we handle this via upsampling during training

## Main Features

- Uses a CSV file to load data (`image_path`, `label`)
- Converts CIFAR-10 into a binary classification task
- Fine-tunes all layers of the pretrained DINO ViT model with an added classification head
- Implements upsampling of the minority class to handle class imbalance during training
- Exposure to Pytorch 


