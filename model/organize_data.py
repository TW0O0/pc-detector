import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def organize_dataset(input_dir, output_dir, test_size=0.2, val_size=0.15):
    """
    Organize dataset into train/val/test splits
    
    Args:
        input_dir: Directory with class subfolders
        output_dir: Output directory for organized data
        test_size: Proportion for test set
        val_size: Proportion for validation set
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each class directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Create class directories in each split
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in {class_dir}")
            continue
        
        # Split into train/val/test
        train_val_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
        train_files, val_files = train_test_split(train_val_files, test_size=val_size/(1-test_size), random_state=42)
        
        # Copy files to respective directories
        for file in train_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.copy2(src, dst)
        
        for file in val_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(val_dir, class_name, file)
            shutil.copy2(src, dst)
        
        for file in test_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(test_dir, class_name, file)
            shutil.copy2(src, dst)
        
        print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize dataset into train/val/test splits')
    parser.add_argument('--input', required=True, help='Input directory with class subfolders')
    parser.add_argument('--output', required=True, help='Output directory for organized data')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion for test set')
    parser.add_argument('--val-size', type=float, default=0.15, help='Proportion for validation set')
    
    args = parser.parse_args()
    organize_dataset(args.input, args.output, args.test_size, args.val_size)
