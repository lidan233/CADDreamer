import os
import shutil

def copy_and_rename_files(source_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Get the name of the current directory
        dir_name = os.path.basename(root)
        
        # Check for the specific files
        if "rgb_000_front.png" in files and "normals_000_front.png" in files:
            # Copy and rename rgb file
            src_rgb = os.path.join(root, "rgb_000_front.png")
            dst_rgb = os.path.join(dest_dir, f"{dir_name}_img.png")
            shutil.copy2(src_rgb, dst_rgb)
            
            # Copy and rename normals file
            src_normal = os.path.join(root, "normals_000_front.png")
            dst_normal = os.path.join(dest_dir, f"{dir_name}_normal.png")
            shutil.copy2(src_normal, dst_normal)
            
            print(f"Copied and renamed files from {root}")


if __name__=='__main__':
    
    source_directory = "/metadisk/yuanli/save_training_image_cad"
    destination_directory = "/metadisk/yuanli/training_datasets_dsine"
    copy_and_rename_files(source_directory, destination_directory)