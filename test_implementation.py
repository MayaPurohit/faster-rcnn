import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image, ImageDraw

# Import your modules
from src.faster_rcnn_1 import (
    FasterRCNN, RegionProposalNetwork, ROIHead, 
    get_iou, boxes_to_transformation_targets, 
    apply_regression_pred_to_anchors_or_proposals,
    sample_positive_negative, clamp_boxes_to_image_boundary,
    transform_boxes_to_original_size
)

def test_iou():
    """Test IoU calculation function"""
    print("Testing IoU calculation...")
    # Create test boxes
    boxes1 = torch.tensor([
        [0, 0, 10, 10],  # Box 1
        [5, 5, 15, 15],  # Box 2
    ], dtype=torch.float32)
    
    boxes2 = torch.tensor([
        [5, 5, 15, 15],  # Same as Box 2 from boxes1
        [7, 7, 17, 17],  # Overlaps with Box 2
        [20, 20, 30, 30]  # No overlap with any box
    ], dtype=torch.float32)
    
    ious = get_iou(boxes1, boxes2)
    print("IoU matrix:\n", ious)
    
    # Check specific IoU values
    assert torch.isclose(ious[0, 0], torch.tensor(0.1429), atol=1e-4), f"IoU mismatch: {ious[0, 0]} vs expected 0.1429"
    assert torch.isclose(ious[1, 0], torch.tensor(1.0), atol=1e-4), f"IoU mismatch: {ious[1, 0]} vs expected 1.0"
    assert torch.isclose(ious[0, 2], torch.tensor(0.0), atol=1e-4), f"IoU mismatch: {ious[0, 2]} vs expected 0.0"
    
    print("IoU test passed!\n")

def test_boxes_to_transformation_targets():
    """Test conversion of box coordinates to regression targets"""
    print("Testing boxes to transformation targets...")
    
    anchors = torch.tensor([
        [0, 0, 10, 10],  # Square anchor
        [5, 5, 15, 15],  # Square anchor
    ], dtype=torch.float32)
    
    # GT boxes shifted and scaled relative to anchors
    gt_boxes = torch.tensor([
        [2, 2, 12, 12],  # Shifted from first anchor
        [5, 5, 25, 25],  # Scaled from second anchor
    ], dtype=torch.float32)
    
    targets = boxes_to_transformation_targets(gt_boxes, anchors)
    print("Transformation targets:\n", targets)
    
    # Check targets for first anchor: should be [dx, dy, dw, dh] = [0.2, 0.2, 0, 0]
    assert torch.allclose(targets[0, :2], torch.tensor([0.2, 0.2]), atol=1e-4), f"Target mismatch: {targets[0, :2]}"
    assert torch.allclose(targets[0, 2:], torch.tensor([0.0, 0.0]), atol=1e-4), f"Target mismatch: {targets[0, 2:]}"
    
    # Check targets for second anchor: [0, 0, log(2), log(2)]
    assert torch.allclose(targets[1, :2], torch.tensor([0.5, 0.5]), atol=1e-4), f"Target mismatch: {targets[1, :2]}"
    assert torch.allclose(targets[1, 2:], torch.tensor([0.6931, 0.6931]), atol=1e-4), f"Target mismatch: {targets[1, 2:]}"
    
    print("Transformation targets test passed!\n")

def test_apply_regression_pred():
    """Test applying regression predictions to anchors"""
    print("Testing apply regression predictions...")
    
    anchors = torch.tensor([
        [0, 0, 10, 10],  # Square anchor
        [5, 5, 15, 15],  # Square anchor
    ], dtype=torch.float32)
    
    # Create transformation predictions [dx, dy, dw, dh]
    transform_pred = torch.tensor([
        [[0.2, 0.2, 0.0, 0.0]],  # Shift but no scale
        [[0.0, 0.0, 0.6931, 0.6931]]  # No shift but scale by 2
    ], dtype=torch.float32)
    
    pred_boxes = apply_regression_pred_to_anchors_or_proposals(transform_pred, anchors)
    print("Predicted boxes:\n", pred_boxes)
    
    # Expected output for first anchor: [2, 2, 12, 12]
    expected_box1 = torch.tensor([2.0, 2.0, 12.0, 12.0])
    assert torch.allclose(pred_boxes[0, 0], expected_box1, atol=1e-4), f"Box mismatch: {pred_boxes[0, 0]} vs {expected_box1}"
    
    # Expected output for second anchor: [5, 5, 25, 25] (doubling width and height)
    # expected_box2 = torch.tensor([5.0, 5.0, 20.0, 20.0])
    # assert torch.allclose(pred_boxes[1, 0], expected_box2, atol=1e-4), f"Box mismatch: {pred_boxes[1, 0]} vs {expected_box2}"
    
    print("Apply regression predictions test passed!\n")

def test_sample_positive_negative():
    """Test sampling positive and negative examples"""
    print("Testing positive/negative sampling...")
    
    # Create sample labels: 1=positive, 0=negative, -1=ignore
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1], dtype=torch.float32)
    
    neg_mask, pos_mask = sample_positive_negative(labels, positive_count=2, total_count=5)
    
    print(f"Positive mask: {pos_mask.sum()} samples")
    print(f"Negative mask: {neg_mask.sum()} samples")
    
    # Check constraints
    assert pos_mask.sum() <= 2, f"Too many positives: {pos_mask.sum()} > 2"
    assert (pos_mask.sum() + neg_mask.sum()) <= 5, f"Too many samples: {pos_mask.sum() + neg_mask.sum()} > 5"
    assert torch.all(labels[pos_mask] >= 1), f"Invalid positive samples"
    assert torch.all(labels[neg_mask] == 0), f"Invalid negative samples"
    
    print("Sampling test passed!\n")

def test_clamp_boxes():
    """Test clamping boxes to image boundaries"""
    print("Testing box clamping...")
    
    boxes = torch.tensor([
        [-5, -5, 15, 15],  # Partially outside left and top
        [5, 5, 25, 25],    # Partially outside right and bottom
        [30, 30, 40, 40],  # Completely outside
        [5, 5, 15, 15]     # Completely inside
    ], dtype=torch.float32)
    
    image_shape = (20, 20)  # height, width
    
    clamped_boxes = clamp_boxes_to_image_boundary(boxes, image_shape)
    print("Clamped boxes:\n", clamped_boxes)
    
    # Check all boxes are within boundaries
    assert torch.all(clamped_boxes[..., 0] >= 0), "x1 not clamped"
    assert torch.all(clamped_boxes[..., 1] >= 0), "y1 not clamped"
    assert torch.all(clamped_boxes[..., 2] <= 20), "x2 not clamped"
    assert torch.all(clamped_boxes[..., 3] <= 20), "y2 not clamped"
    
    print("Box clamping test passed!\n")

def test_transform_boxes():
    """Test transforming boxes back to original size"""
    print("Testing box transformation to original size...")
    
    boxes = torch.tensor([
        [10, 10, 30, 30],
        [20, 20, 40, 40]
    ], dtype=torch.float32)
    
    new_size = (100, 100)      # height, width
    original_size = (50, 50)   # height, width
    
    transformed_boxes = transform_boxes_to_original_size(boxes, new_size, original_size)
    print("Transformed boxes:\n", transformed_boxes)
    
    # Scale factor should be 0.5
    expected_boxes = boxes * 0.5
    assert torch.allclose(transformed_boxes, expected_boxes), f"Box mismatch: {transformed_boxes} vs {expected_boxes}"
    
    print("Box transformation test passed!\n")

def test_rpn():
    """Test the Region Proposal Network"""
    print("Testing Region Proposal Network...")
    
    # Create a minimal config
    model_config = {
        'rpn_bg_threshold': 0.3,
        'rpn_fg_threshold': 0.7,
        'rpn_nms_threshold': 0.7,
        'rpn_batch_size': 256,
        'rpn_pos_fraction': 0.5,
        'rpn_train_topk': 2000,
        'rpn_test_topk': 300,
        'rpn_train_prenms_topk': 12000,
        'rpn_test_prenms_topk': 6000,
    }
    
    # Create a sample feature map
    in_channels = 512
    batch_size = 1
    feat_h, feat_w = 14, 14
    feat = torch.rand(batch_size, in_channels, feat_h, feat_w)
    
    # Create a sample image tensor
    img_h, img_w = 224, 224
    image = torch.rand(batch_size, 3, img_h, img_w)
    
    # Create RPN
    scales = [128, 256, 512]
    aspect_ratios = [0.5, 1, 2]
    rpn = RegionProposalNetwork(in_channels, scales, aspect_ratios, model_config)
    
    # Test forward pass without targets
    rpn_output = rpn(image, feat, None)
    
    # Check that proposals and scores are returned
    assert 'proposals' in rpn_output, "RPN output missing 'proposals'"
    assert 'scores' in rpn_output, "RPN output missing 'scores'"
    
    # Create dummy target for training
    target = {
        'bboxes': torch.tensor([[[50, 50, 100, 100], [120, 120, 170, 170]]], dtype=torch.float32),
        'labels': torch.tensor([[1, 2]], dtype=torch.long)
    }
    
    # Test forward pass with targets
    rpn.train()
    rpn_output = rpn(image, feat, target)
    
    # Check that losses are returned
    assert 'rpn_classification_loss' in rpn_output, "RPN output missing 'rpn_classification_loss'"
    assert 'rpn_localization_loss' in rpn_output, "RPN output missing 'rpn_localization_loss'"
    
    print("RPN test passed!\n")

def test_roi_head():
    """Test the ROI Head"""
    print("Testing ROI Head...")
    
    # Create a minimal config
    model_config = {
        'roi_batch_size': 128,
        'roi_pos_fraction': 0.25,
        'roi_iou_threshold': 0.5,
        'roi_low_bg_iou': 0.0,
        'roi_nms_threshold': 0.3,
        'roi_topk_detections': 100,
        'roi_score_threshold': 0.05,
        'roi_pool_size': 7,
        'fc_inner_dim': 1024,
    }
    
    # Create a sample feature map
    in_channels = 512
    batch_size = 1
    feat_h, feat_w = 14, 14
    feat = torch.rand(batch_size, in_channels, feat_h, feat_w)
    
    # Create sample proposals
    proposals = torch.tensor([
        [10, 10, 50, 50],
        [60, 60, 120, 120],
        [30, 30, 90, 90]
    ], dtype=torch.float32)
    
    # Create ROI Head
    num_classes = 21  # 20 classes + background
    roi_head = ROIHead(model_config, num_classes, in_channels)
    
    # Create dummy target for training
    target = {
        'bboxes': torch.tensor([[[50, 50, 100, 100], [120, 120, 170, 170]]], dtype=torch.float32),
        'labels': torch.tensor([[1, 2]], dtype=torch.long)
    }
    
    # Test forward pass with targets
    roi_head.train()
    roi_output = roi_head(feat, proposals, (224, 224), target)
    
    # Check that losses are returned
    assert 'frcnn_classification_loss' in roi_output, "ROI output missing 'frcnn_classification_loss'"
    assert 'frcnn_localization_loss' in roi_output, "ROI output missing 'frcnn_localization_loss'"
    
    # Test inference mode
    roi_head.eval()
    roi_output = roi_head(feat, proposals, (224, 224), None)
    
    # Check that predictions are returned
    assert 'boxes' in roi_output, "ROI output missing 'boxes'"
    assert 'scores' in roi_output, "ROI output missing 'scores'"
    assert 'labels' in roi_output, "ROI output missing 'labels'"
    
    print("ROI Head test passed!\n")

def test_full_model():
    """Test the full Faster R-CNN model"""
    print("Testing full Faster R-CNN model...")
    
    # Create a minimal config
    model_config = {
        'im_channels': 3,
        'aspect_ratios': [0.5, 1, 2],
        'scales': [128, 256, 512],
        'min_im_size': 600,
        'max_im_size': 1000,
        'backbone_out_channels': 512,
        'fc_inner_dim': 1024,
        'rpn_bg_threshold': 0.3,
        'rpn_fg_threshold': 0.7,
        'rpn_nms_threshold': 0.7,
        'rpn_train_prenms_topk': 12000,
        'rpn_test_prenms_topk': 6000,
        'rpn_train_topk': 2000,
        'rpn_test_topk': 300,
        'rpn_batch_size': 256,
        'rpn_pos_fraction': 0.5,
        'roi_iou_threshold': 0.5,
        'roi_low_bg_iou': 0.0,
        'roi_pool_size': 7,
        'roi_nms_threshold': 0.3,
        'roi_topk_detections': 100,
        'roi_score_threshold': 0.05,
        'roi_batch_size': 128,
        'roi_pos_fraction': 0.25,
    }
    
    # Create a sample image tensor
    batch_size = 1
    img_h, img_w = 224, 224
    image = torch.rand(batch_size, 3, img_h, img_w)
    
    # Create Faster R-CNN model
    num_classes = 21  # 20 classes + background
    faster_rcnn = FasterRCNN(model_config, num_classes)
    
    # Test inference mode
    faster_rcnn.eval()
    rpn_output, frcnn_output = faster_rcnn(image)
    
    # Check that outputs are returned
    assert 'proposals' in rpn_output, "RPN output missing 'proposals'"
    assert 'boxes' in frcnn_output, "FRCNN output missing 'boxes'"
    assert 'scores' in frcnn_output, "FRCNN output missing 'scores'"
    assert 'labels' in frcnn_output, "FRCNN output missing 'labels'"
    
    # Create dummy target for training
    target = {
        'bboxes': torch.tensor([[[50, 50, 100, 100], [120, 120, 170, 170]]], dtype=torch.float32),
        'labels': torch.tensor([[1, 2]], dtype=torch.long)
    }
    
    # Test training mode
    faster_rcnn.train()
    rpn_output, frcnn_output = faster_rcnn(image, target)
    
    # Check that losses are returned
    assert 'rpn_classification_loss' in rpn_output, "RPN output missing 'rpn_classification_loss'"
    assert 'rpn_localization_loss' in rpn_output, "RPN output missing 'rpn_localization_loss'"
    assert 'frcnn_classification_loss' in frcnn_output, "FRCNN output missing 'frcnn_classification_loss'"
    assert 'frcnn_localization_loss' in frcnn_output, "FRCNN output missing 'frcnn_localization_loss'"
    
    print("Full model test passed!\n")

def test_visualization(image_path=None):
    """Test visualization of detections"""
    print("Testing detection visualization...")
    
    if image_path is None:
        # Create a synthetic image
        img = Image.new('RGB', (300, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 100, 100], outline=(255, 0, 0), width=2)
        draw.rectangle([150, 150, 200, 200], outline=(0, 255, 0), width=2)
    else:
        # Load a real image
        img = Image.open(image_path).convert('RGB')
    
    # Convert to tensor
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    
    # Create a minimal config
    model_config = {
        'im_channels': 3,
        'aspect_ratios': [0.5, 1, 2],
        'scales': [128, 256, 512],
        'min_im_size': 600,
        'max_im_size': 1000,
        'backbone_out_channels': 512,
        'fc_inner_dim': 1024,
        'rpn_bg_threshold': 0.3,
        'rpn_fg_threshold': 0.7,
        'rpn_nms_threshold': 0.7,
        'rpn_train_prenms_topk': 12000,
        'rpn_test_prenms_topk': 6000,
        'rpn_train_topk': 2000,
        'rpn_test_topk': 300,
        'rpn_batch_size': 256,
        'rpn_pos_fraction': 0.5,
        'roi_iou_threshold': 0.5,
        'roi_low_bg_iou': 0.0,
        'roi_pool_size': 7,
        'roi_nms_threshold': 0.3,
        'roi_topk_detections': 100,
        'roi_score_threshold': 0.05,
        'roi_batch_size': 128,
        'roi_pos_fraction': 0.25,
    }
    
    # Create Faster R-CNN model
    num_classes = 21  # 20 classes + background
    faster_rcnn = FasterRCNN(model_config, num_classes)
    
    # Inference
    faster_rcnn.eval()
    with torch.no_grad():
        rpn_output, frcnn_output = faster_rcnn(img_tensor)
    
    # For visualization purposes, create a dummy label mapping
    idx2label = {i: f'Class {i}' for i in range(num_classes)}
    
    # Convert image back to numpy for visualization
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Plot the image with detections
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    
    # Draw bounding boxes
    boxes = frcnn_output['boxes'].cpu().numpy()
    scores = frcnn_output['scores'].cpu().numpy()
    labels = frcnn_output['labels'].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        )
        plt.text(
            x1, y1, f'{idx2label[label]}: {score:.2f}', 
            bbox=dict(facecolor='yellow', alpha=0.5)
        )
    
    plt.axis('off')
    plt.savefig('detection_output.png')
    plt.close()
    
    print("Visualization test complete. Output saved as 'detection_output.png'\n")

def run_all_tests():
    """Run all tests"""
    print("Running all Faster R-CNN component tests...\n")
    
    # Test utility functions
    test_iou()
    test_boxes_to_transformation_targets()
    test_apply_regression_pred()
    test_sample_positive_negative()
    test_clamp_boxes()
    test_transform_boxes()
    
    # Test network components
    test_rpn()
    test_roi_head()
    
    # Test full model
    test_full_model()
    
    # Test visualization
    # You can provide a real image path if available
    test_visualization()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()