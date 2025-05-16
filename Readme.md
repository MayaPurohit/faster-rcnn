# Computer Vision Assignment: Faster R-CNN Implementation

## Overview
This assignment focuses on implementing the Faster R-CNN architecture for object detection from scratch. You will gain hands-on experience with one of the most influential object detection models in computer vision. The implementation will cover all critical components of the Faster R-CNN pipeline, including the backbone network, Region Proposal Network (RPN), and detection head.

## Learning Objectives
- Understand the complete architecture of Faster R-CNN
- Implement a Region Proposal Network (RPN) for generating object proposals
- Create RoI operations for feature extraction from proposed regions
- Build detection heads for classification and bounding box regression
- Train and evaluate an object detection model on standard datasets
- Gain practical experience with PyTorch for complex computer vision tasks

## Dataset
- Pascal VOC dataset (primary dataset)
- MS COCO dataset (optional for extension)

## Timeline
- Release date: April 26, 2025
- Submission deadline: May 20, 2025

## Installation and Setup

### Prerequisites (Can use the previous environment)
- Python 3.8+
- PyTorch 1.10+
- CUDA-enabled GPU (required)

### Setup Environment
You can use the previous environment, and install the uninstalled packages if encountered errors.

### Dataset Preparation
- Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007` folder
- Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007-test` folder

## Project Structure
```
assignment3/
├── checkpoints/            # Directory for saved models
├── coco/                   # COCO dataset (optional)
├── config/                 # Configuration files
│   └── voc.yaml           # VOC dataset configuration
├── dataset/                # Dataset handling code
│   └── voc.py             # Pascal VOC dataset loader
├── src/                    # Source code (to be implemented)
│   └── faster_rcnn.py     # Main Faster R-CNN implementation
├── test/                   # Testing scripts
│   └── test_faster_rcnn.py # Tests for Faster R-CNN implementation
├── train/                  # Training scripts
│   ├── test_implementation.py # Basic implementation tests
│   └── train_faster_rcnn.py   # Training script
├── VOC2007/                # VOC2007 dataset
├── VOC2007-test/           # VOC2007 test dataset
└── README.md
```

## Assignment Tasks

### Part 1: Backbone Network (Not to be implemented)
The backbone network for feature extraction is already provided for you using a pre-trained VGG16 model. You should:

**Requirements:**
- Understand how the backbone extracts features from input images
- Familiarize yourself with the feature maps produced by the backbone
- Note that you don't need to implement this part as it's already included in the starter code

### Part 2: Region Proposal Network (20%)
Implement the RPN that generates region proposals where objects might be located.

**Requirements:**
- Create an anchor generator that produces anchors at multiple scales and aspect ratios
- Implement the RPN head with convolutional layers for objectness classification and box regression
- Create functions to transform between anchor boxes and regression targets
- Implement sampling strategies for selecting positive and negative anchors
- Apply non-maximum suppression (NMS) to filter proposals

### Part 3: RoI Feature Extraction (20%)
Implement the region-of-interest (RoI) feature extraction.

**Requirements:**
- Implement RoI Pooling to extract fixed-size features from proposals
- Create functions to match proposals with ground truth boxes
- Implement positive/negative sampling for RoI training

### Part 4: Detection Head (20%)
Implement the detection head for final classification and bounding box refinement.

**Requirements:**
- Create fully connected layers for the detection head
- Implement multi-class classification of object categories
- Implement bounding box regression for final localization refinement
- Apply NMS to remove duplicate detections

### Part 5: Build the Faster R-CNN Model (20%)
Integrate all components into a complete Faster R-CNN model.
**Requirements:**
- Combine the backbone, RPN, RoI pooling, and detection head into a single model
- Implement the forward pass for the entire model
- Create a training loop that handles both RPN and detection head training

### Part 6: Report (10%)
Write a report on your implementation and compare it with the PyTorch Faster R-CNN model.
Please see the PyTorch demo for the Faster R-CNN model [here](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).

**Requirements:**
- Implement the multi-task loss function (classification + regression)
- Create a training loop that handles both RPN and detection head training
- Implement mAP calculation for evaluation
- Analyze and visualize model performance

## Testing Your Implementation

We have provided test files to help you verify your implementation:

```bash
# Test your implementation components
python train/test_implementation.py

# Test the full Faster R-CNN model
python test/test_faster_rcnn.py
```

Make sure all tests pass before submitting your assignment. These tests will check:
- Anchor generation and transformation
- RPN forward and backward passes
- RoI feature extraction
- Detection head classification and regression
- End-to-end model operation

## Running the Assignment

### Training
```bash
# Train Faster R-CNN on VOC dataset
python train/train_faster_rcnn.py
```

### Evaluation
```bash
# Evaluate Faster R-CNN
python test/test_faster_rcnn.py --checkpoint checkpoints/faster_rcnn_final.pth
```

## Evaluation Criteria
Your assignment will be evaluated based on:

- Correctness of implementation (50%)
- Model performance on test set (30%)
- Code quality and documentation (10%)
- Technical report (10%)

## Submission Requirements
- Full source code with your implementation
- Pre-trained model weights
- Technical report (PDF) containing:
  - Description of your implementation
  - Performance analysis (mAP, examples of detections)
  - Discussion of challenges and solutions
  - Extend the Faster RCNN to Masked RCNN. (20%)
    - Model Correctly Implemented (5%)
    - Trained, and have inference results in report (15%)

## Resources
- Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2015)
- He et al., "Deep Residual Learning for Image Recognition" (2016)
- Pascal VOC dataset documentation
- PyTorch documentation

## Additional Notes
***IMPORTANT: PLEASE BE PREPARED TO TRAIN FOR MORE THAN 24 HOURS.***
- Start by implementing and testing individual components before integrating them
- Use the provided test files to verify your implementation correctness
- Pay special attention to the implementation of RPN and RoI operations
- Debugging object detection models can be challenging; use visualization tools to understand errors

## Questions and Support
If you have questions about the assignment, please post them on the course discussion forum or contact the teaching assistants during office hours.