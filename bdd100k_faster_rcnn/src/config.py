"""Names shared by training, evaluation, and prediction."""

# Faster R-CNN reserves class 0 for the background.
CATEGORIES = (
    "person",
    "car",
    "rider",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
)

CLASS_TO_ID = {
    category: class_id
    for class_id, category in enumerate(CATEGORIES, start=1)
}

ID_TO_CLASS = {
    class_id: category
    for category, class_id in CLASS_TO_ID.items()
}

# Ten object classes plus the background class.
NUM_CLASSES = len(CATEGORIES) + 1

