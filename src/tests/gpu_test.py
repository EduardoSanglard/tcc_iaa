import torch
import time


def gpu_intensive_task(size=10000):
    # Create large tensors
    tensor_a = torch.randn(size, size).to("cuda")
    tensor_b = torch.randn(size, size).to("cuda")

    # Perform matrix multiplication
    start_time = time.time()
    result = torch.matmul(tensor_a, tensor_b)
    end_time = time.time()

    print(f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
    print(f"Result shape: {result.shape}")

    return result


if __name__ == "__main__":
    # Verify CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Run the GPU-intensive task
    gpu_intensive_task()
