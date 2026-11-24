"""Create the Faster R-CNN model."""

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import NUM_CLASSES


def create_model(pretrained=True, min_size=640, max_size=1024):
    """Return Faster R-CNN with a ResNet-50/FPN backbone.

    COCO weights give us a useful starting point. The final COCO classifier is
    replaced because BDD100K has 10 object classes plus background.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        # Full detection weights already contain the backbone. Setting this to
        # None also prevents an ImageNet download during evaluation/resume.
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )

    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        input_features,
        NUM_CLASSES,
    )

    return model
