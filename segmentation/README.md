# Semantic Segmenation with Ships
Created by Mason Pennell, Spencer Lafferty, and Jack de Bruyn

Github: https://github.com/masonhpennell/ship-detection

Dataset: [Classification of Ship Images](https://www.kaggle.com/code/teeyee314/classification-of-ship-images/input)

## Experimenation

During training, the model is shown many examples of images together with their correct masks. For each image, the network processes the pixels through many layers of transformations. Early layers detect simple structures like edges and textures, while deeper layers detect more abstract spatial patterns and shapes. The encoder portion compresses the image into a compact mathematical representation containing the most important visual information. The decoder then expands that compressed representation back into a full-resolution spatial prediction.

The model’s prediction is compared against the true mask pixel by pixel. The difference between the prediction and the true answer is measured using a loss function, in this case binary cross-entropy, which mathematically penalizes incorrect probabilities. If the model predicts high probability where the mask should be 0, or low probability where it should be 1, the loss increases.

Using gradient descent and backpropagation, the program computes derivatives of the loss with respect to every trainable parameter in the network. Those derivatives indicate how each parameter should change to reduce prediction error. The optimizer repeatedly updates the weights in the direction that minimizes the loss over many training iterations.

Over repeated epochs, the network ideally converges toward a set of parameters that minimize prediction error and maximize spatial overlap accuracy on unseen validation images.

## Instructions

- Requires python 3.10+
- Follow instructions for [TensorFlow](https://www.tensorflow.org/install)

### Transfer learning config

Set these constants in `segmentation.py`:

- `EPOCHS`: initial feature-extractor training epochs (frozen backbone)
- `FINE_TUNING`: Determine whether the ImageNet weights change during training.
- `TRANSFER_LEARNING`: every parameter starts random and must learn from the dataset alone.

Assignment 2 Contributions:
- Jack: Created synthetic masks and 99% of pipeline
- Mason: Hand annotated masks, Write-up, Transfer Learning implementation
- Spencer: Evaluation

Assignment 3 Contributions:
- Jack: Handmade training masks, base pipeline code, sam2 generated masks
- Mason: Handmade training masks, Transfer learning, ReadMe
- Spencer: Handmade training masks, Evaluation + Report