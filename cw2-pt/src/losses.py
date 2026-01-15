import torch
import torch.nn.functional as F

def build_distance_matrix(num_classes: int, device: torch.device)-> torch.Tensor:
    """
    Builds and returns a [C,C] matrix D where D[true, pred] is the penalty weight
    we use the number of classes to structure the matrix and define its dimensions.
    row index corresponds to the true class, column index to the predicted class.
    the values in the cells represent the penalty based on the distance between classes.

    the matrix represents the anatomy knowledge of the organs turned into numbers
    """
    
    D = torch.tensor([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 0. background
        [0.00, 0.00, 0.91, 0.64, 0.45, 0.45, 0.82, 0.64, 0.55], # 1. Bladder
        [0.00, 0.91, 0.00, 0.45, 0.82, 0.82, 1.00, 0.73, 0.73], # 2. Bone 
        [0.00, 0.64, 0.45, 0.00, 0.36, 0.36, 0.55, 0.27, 0.27], # 3. Obturator Internus
        [0.00, 0.45, 0.82, 0.36, 0.00, 0.00, 0.36, 0.18, 0.09], # 4. Transition (pros 1)
        [0.00, 0.45, 0.82, 0.36, 0.00, 0.00, 0.36, 0.18, 0.09], # 5. Central Gland (Pros 2)
        [0.00, 0.82, 1.00, 0.55, 0.36, 0.36, 0.00, 0.27, 0.36], # 6. Rectum
        [0.00, 0.64, 0.73, 0.27, 0.18, 0.18, 0.27, 0.00, 0.09], # 7. Seminal Vesicle
        [0.00, 0.55, 0.73, 0.27, 0.09, 0.09, 0.36, 0.09, 0.00], # 8. Neurovascular Bundle
    ], device=device, dtype=torch.float32)

    if D.shape != (num_classes, num_classes):
        raise ValueError(f"Distance matrix shape {D.shape} does not match num_classes {num_classes}")
    
    return D

def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ Computes the standard cross-entropy loss between logits and targets. """
    return F.cross_entropy(logits, targets)

def hierarchical_ce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        D: torch.Tensor,
        alpha: float=10,
        epsilon: float = 1e-8
    ) -> torch.Tensor:

    """ Computes the hierarchical cross-entropy loss between logits and targets using distance matrix D. """
    # cross entropy loss per pixel wihtput avearaging
    ce = F.cross_entropy(logits, targets, reduction='none')  
    
    # predict class probabilities per pixel given the logits
    preds = logits.argmax(dim=1)

    # find the penalty weights based on true and predicted classes
    penalties = D[targets, preds]

    # weight the cross entropy loss with the penalties
    weighted_ce = ce * (1 + alpha*penalties)

    return weighted_ce.mean()
