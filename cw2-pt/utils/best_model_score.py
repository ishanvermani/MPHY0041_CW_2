"""
# ###############################################################
# Choose super_class Dice or general Dice to check best model score !
# ###############################################################
"""
import torch

# Path to your saved model
model_path = "data/processed/best_model.pt"

try:
    # Load the file
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Check if it has the metadata we saved
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("---------------------------------------")
        print(f"Model saved at Epoch:  {checkpoint['epoch']}")
        print(f"Best Validation Loss:  {checkpoint['val_loss']:.4f}")
        print(f"Best Validation Dice:  {checkpoint['val_dice']:.4f}")
        print("---------------------------------------")
    else:
        print("This file contains only the model weights, no score data.")

except FileNotFoundError:
    print("Could not find best_model.pt at that location.")
except Exception as e:
    print(f"Error reading file: {e}")

