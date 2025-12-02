import torch

print("=" * 60)
print("KIỂM TRA GPU")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ Bạn có thể sử dụng GPU!")
    print("Notebook sẽ tự động chọn GPU nếu có sẵn.")
else:
    print("\n❌ PyTorch không tìm thấy CUDA/GPU")
    print("\nCó 2 khả năng:")
    print("1. Máy bạn không có GPU NVIDIA")
    print("2. PyTorch được cài đặt phiên bản CPU-only")
    print("\nĐể kiểm tra GPU:")
    print("  - Mở Task Manager > Performance > GPU")
    print("  - Hoặc chạy: nvidia-smi")
    print("\nĐể cài PyTorch với GPU (nếu có NVIDIA GPU):")
    print("  pip uninstall torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("=" * 60)
