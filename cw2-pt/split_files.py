import argparse
import random
import shutil
from pathlib import Path


def split_nifti_files(data_dir: str, 
                      output_dir: str | None = None,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_seed: int = 42,
                      move_files: bool = False) -> dict[str, list[tuple[str, str]]]:
    '''
    
    This function finds all *_img.nii files and their corresponding *_mask.nii files,
    shuffles them randomly, and splits them into train/validation/test sets.
    '''
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        'Ratios must sum to 1.0'
    
    # Set random seed
    random.seed(random_seed)
    
    # Get all image files
    data_path = Path(data_dir)
    # Search recursively in case they are in subfolders, or just in the root
    img_files = sorted([f for f in data_path.rglob('*_img.nii')])
    
    if not img_files:
        # Fallback for .nii.gz if needed
        img_files = sorted([f for f in data_path.rglob('*_img.nii.gz')])

    # Find corresponding mask files
    pairs = []
    for img_file in img_files:
        # specific string replacement to handle both .nii and .nii.gz
        name = img_file.name
        if name.endswith('_img.nii'):
            mask_name = name.replace('_img.nii', '_mask.nii')
        elif name.endswith('_img.nii.gz'):
            mask_name = name.replace('_img.nii.gz', '_mask.nii.gz')
        else:
            continue

        mask_file = img_file.parent / mask_name
        
        if mask_file.exists():
            pairs.append((str(img_file), str(mask_file)))
        else:
            print(f'Warning: No mask found for {img_file.name}')
    
    print(f'Found {len(pairs)} image-mask pairs')
    
    # Shuffle pairs
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
    
    print(f'Train: {len(train_pairs)} pairs')
    print(f'Val: {len(val_pairs)} pairs')
    print(f'Test: {len(test_pairs)} pairs')
    
    # Move/Copy files to output directories
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        action_name = "Moved" if move_files else "Copied"

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
                
                if move_files:
                    shutil.move(img_path, img_dir / img_name)
                    shutil.move(mask_path, mask_dir / mask_name)
                else:
                    shutil.copy2(img_path, img_dir / img_name)
                    shutil.copy2(mask_path, mask_dir / mask_name)
            
            print(f'{action_name} {len(split_pairs)} pairs to {split_dir}')
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Split NIfTI image-mask pairs')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Source directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Destination directory')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--move', action='store_true', help='Move files instead of copying (Saves space)')
        
    args = parser.parse_args()
    
    split_nifti_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        move_files=args.move
    )
    
    print('\n✓ Processing complete!')

if __name__ == '__main__':
    main()