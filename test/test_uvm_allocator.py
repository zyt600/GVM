import torch
from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator


def replace_default_allocator():
    """
    Replace the default CUDA allocator with a custom one.
    """
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LIB_PATH = os.path.join(ROOT_DIR, "csrc/custom_allocator/uvm_allocator.so")
    # Set the custom allocator
    change_current_allocator(
        CUDAPluggableAllocator(LIB_PATH, "uvm_alloc", "uvm_free"))


if __name__ == "__main__":
    # Replace the default allocator
    replace_default_allocator()

    # Allocate a tensor to test the custom allocator
    tensor = torch.randn(1000, 1000, device='cuda')
    print("Tensor allocated with custom allocator:", tensor)

    # Free the tensor
    del tensor
    print("Tensor freed.")
