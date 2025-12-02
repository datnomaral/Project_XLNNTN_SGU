"""
Script tự động download Multi30K dataset
Không cần torchtext!
"""

import os
import urllib.request
import gzip
import shutil

def download_multi30k():
    """Download Multi30K dataset từ GitHub"""
    
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    files = [
        "train.en.gz", "train.fr.gz",
        "val.en.gz", "val.fr.gz",
        "test_2016_flickr.en.gz", "test_2016_flickr.fr.gz"
    ]
    
    # Tạo thư mục
    os.makedirs("data/multi30k", exist_ok=True)
    
    print("Downloading Multi30K dataset...")
    print("=" * 80)
    
    for file in files:
        url = base_url + file
        gz_path = f"data/multi30k/{file}"
        
        # Tên file sau khi giải nén
        if "test_2016" in file:
            out_file = file.replace("test_2016_flickr.", "test.").replace(".gz", "")
        else:
            out_file = file.replace(".gz", "")
        out_path = f"data/multi30k/{out_file}"
        
        # Kiểm tra đã tồn tại chưa
        if os.path.exists(out_path):
            print(f"✓ {out_file} already exists")
            continue
        
        try:
            # Download
            print(f"Downloading {file}...", end=" ")
            urllib.request.urlretrieve(url, gz_path)
            print("✓")
            
            # Giải nén
            print(f"Extracting {out_file}...", end=" ")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("✓")
            
            # Xóa file .gz
            os.remove(gz_path)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    print("=" * 80)
    print("✓ Multi30K dataset downloaded successfully!")
    print(f"Location: data/multi30k/")
    return True


def load_multi30k():
    """Load Multi30K từ file local"""
    def read_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    base_path = 'data/multi30k'
    
    try:
        train_en = read_file(f'{base_path}/train.en')
        train_fr = read_file(f'{base_path}/train.fr')
        val_en = read_file(f'{base_path}/val.en')
        val_fr = read_file(f'{base_path}/val.fr')
        test_en = read_file(f'{base_path}/test.en')
        test_fr = read_file(f'{base_path}/test.fr')
        
        train_data = list(zip(train_en, train_fr))
        val_data = list(zip(val_en, val_fr))
        test_data = list(zip(test_en, test_fr))
        
        print(f"\n✓ Loaded Multi30K dataset:")
        print(f"  - Train: {len(train_data)} samples")
        print(f"  - Validation: {len(val_data)} samples")
        print(f"  - Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
        
    except FileNotFoundError:
        print("❌ Dataset files not found!")
        print("Run: python download_multi30k.py")
        return None, None, None


if __name__ == "__main__":
    download_multi30k()
    
    # Test load
    print("\nTesting load...")
    train_data, val_data, test_data = load_multi30k()
    
    if train_data:
        print("\nExample:")
        en, fr = train_data[0]
        print(f"English: {en}")
        print(f"French: {fr}")
