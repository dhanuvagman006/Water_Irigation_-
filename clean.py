import os
import shutil

def remove_pycache(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                print(f"Removing: {full_path}")
                shutil.rmtree(full_path)

if __name__ == "__main__":
    root_directory = "./backend"  
    remove_pycache(root_directory)
    print("Done.")