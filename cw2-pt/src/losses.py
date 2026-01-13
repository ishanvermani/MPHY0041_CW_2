import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def build_distance_matrix(num_classes: int, device: torch.device)-> torch.Tensor:
    """
    Builds and returns a [C,C] matrix D where D[true, pred] is the penalty weight
    we use the number of classes to structure the matrix and define its dimensions.
    row index corresponds to the true class, column index to the predicted class.
    the values in the cells represent the penalty based on the distance between classes.

    the matrix represents the anatomy knowledge of the organs turned into numbers
    """

    """"
    D = torch.tensor([
        [0, 1, 1, 1, 1, 3],  # true: background
        [1, 0, 1, 2, 2, 3],  # true: prostate
        [1, 1, 0, 2, 2, 3],  # true: seminal vesicles
        [1, 2, 2, 0, 1, 3],  # true: bladder
        [1, 2, 2, 1, 0, 3],  # true: rectum
        [3, 3, 3, 3, 3, 0],  # true: bone
    ], device=device, dtype=torch.float32)
    """

    # Extended D matrix for 9 classes (rows=true, cols=pred)
    # D = torch.tensor([
    #     [0, 1, 1, 1, 1, 1, 1, 1, 3],  # background
    #     [1, 0, 0, 0, 1, 1, 1, 2, 3],  # prostate 1
    #     [1, 0, 0, 0, 1, 1, 1, 2, 3],  # prostate 2
    #     [1, 0, 0, 0, 1, 1, 1, 2, 3],  # prostate 3
    #     [1, 1, 1, 1, 0, 0, 1, 2, 3],  # muscle
    #     [1, 1, 1, 1, 1, 0, 1, 2, 3],  # seminal vesicles
    #     [1, 2, 2, 2, 1, 1, 0, 1, 3],  # bladder
    #     [1, 2, 2, 2, 1, 1, 1, 0, 3],  # rectum
    #     [3, 3, 3, 3, 3, 3, 3, 3, 0],  # bone
    # ], device=device, dtype=torch.float32)
    
    D = torch.tensor([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # background
        [0.00, 0.00, 0.91, 0.64, 0.45, 0.36, 0.82, 0.64, 0.55], # Bladder
        [0.00, 0.91, 0.00, 0.45, 0.82, 0.91, 1.00, 0.73, 0.73], # Bone 
        [0.00, 0.64, 0.45, 0.00, 0.36, 0.45, 0.55, 0.27, 0.27], # Obturator Internus
        [0.00, 0.45, 0.82, 0.36, 0.00, 0.18, 0.36, 0.18, 0.09], # Transition (pros 1)
        [0.00, 0.36, 0.91, 0.45, 0.18, 0.00, 0.55, 0.36, 0.27], # Central Gland (Pros 2)
        [0.00, 0.82, 1.00, 0.55, 0.36, 0.55, 0.00, 0.27, 0.36], # Rectum
        [0.00, 0.64, 0.73, 0.27, 0.18, 0.36, 0.27, 0.00, 0.09], # Seminal Vesicle
        [0.00, 0.55, 0.73, 0.27, 0.09, 0.27, 0.36, 0.09, 0.00], # Neurovascular Bundle
    ], device=device, dtype=torch.float32)

    if D.shape != (num_classes, num_classes):
        raise ValueError(f"Distance matrix shape {D.shape} does not match num_classes {num_classes}")
    
    return D

# perform MLE for our segmentation model baselinelearnign objective
# standard cross entropy loss implementation
# we have as input the outputs C of the model, (unormalized scores) for each pixel
def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the standard cross-entropy loss between logits and targets.

    Args:
        logits (torch.Tensor): The unnormalized scores from the model of shape [B, C, H, W].
        targets (torch.Tensor): The ground truth labels of shape [B, H, W].

    Returns:
        torch.Tensor: The computed cross-entropy loss. 
        if the model predicts the correct class with high confidence, the loss is low.
        otherwise the loss is high.
    """
    return F.cross_entropy(logits, targets)

# # performs MLE with anatomy-aware penalties
# # D represents the hierarchy distance matrix/ penalty weight
def hierarchical_ce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        D: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:

    """computees the hierarchical cross-entropy loss between logits and targets using distance matrix D."""
    # cross entropy loss per pixel wihtput avearaging
    ce = F.cross_entropy(logits, targets, reduction='none')  

    # ce shape:  torch.Size([16, 192, 192])
    # logits:  torch.Size([16, 9, 192, 192])
    # targets:  torch.Size([16, 192, 192])
    # penalties shape:  torch.Size([16, 192, 192])

    # predict class probabilities per pixel given the logits
    preds = logits.argmax(dim=1)

    # find the penalty weights based on true and predicted classes
    penalties = D[targets, preds]

    # weight the cross entropy loss with the penalties
    weighted_ce = ce * (1 + penalties)

    return weighted_ce.mean()
