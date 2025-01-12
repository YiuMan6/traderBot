import torch

def get_device():
    if torch.cuda.is_available():
        print(f"\n=== GPU 信息 ===")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def clear_gpu_memory():
    """清理GPU显存"""
    torch.cuda.empty_cache()