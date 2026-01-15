import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def split_nifti_files(data_dir: str, 
                      output_dir: str,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_seed: int = 42) -> Dict[str, List[Tuple[str, str]]]:
    """
    This function finds all *_img.nii files and their corresponding *_mask.nii files,
    shuffles them randomly, and splits them into train/validation/test sets according
    to the specified ratios.
    """

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        'Ratios must sum to 1.0'
    
    random.seed(random_seed)

    data_path = Path(data_dir)
    img_files = sorted([f for f in data_path.glob('*_img.nii')])
    
    # Find corresponding mask files
    pairs = []
    for img_file in img_files:
        mask_file = data_path / img_file.name.replace('_img.nii', '_mask.nii')
        if mask_file.exists():
            pairs.append((str(img_file), str(mask_file)))
        else:
            print(f'Warning: No mask found for {img_file.name}')
    
    print(f'Found {len(pairs)} image-mask pairs')
    
    random.shuffle(pairs)
    
    # Calculate split indices
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split pairs
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    print(f'Train: {len(train_pairs)} pairs ({len(train_pairs)/n_total*100:.1f}%)')
    print(f'Val: {len(val_pairs)} pairs ({len(val_pairs)/n_total*100:.1f}%)')
    print(f'Test: {len(test_pairs)} pairs ({len(test_pairs)/n_total*100:.1f}%)')
    
    # Copy files to output directories
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_pairs in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for images and masks
            img_dir = split_dir / 'images'
            mask_dir = split_dir / 'masks'
            img_dir.mkdir(exist_ok=True)
            mask_dir.mkdir(exist_ok=True)
            
            for img_path, mask_path in split_pairs:
                img_name = Path(img_path).name
                mask_name = Path(mask_path).name
                
                shutil.copy2(img_path, img_dir / img_name)
                shutil.copy2(mask_path, mask_dir / mask_name)
            
            print(f'Copied {len(split_pairs)} pairs to {split_dir}')
    
    return splits

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    splits = split_nifti_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )
    
    print('\n✓ Split completed successfully!')
    if args.output_dir:
        print(f'  Output directory: {args.output_dir}')
    else:
        print('  Note: Files were not copied/moved (use --output_dir to organize files)')
    print(f'  Splits available: {list(splits.keys())}')
    
    return splits

if __name__ == '__main__':
    main()
