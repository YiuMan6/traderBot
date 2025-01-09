import torch

def get_device():
    if torch.cuda.is_available():
        print(f"\n=== GPU 信息 ===")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"初始显存占用: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    else:
        return torch.device('cpu')