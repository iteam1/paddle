'''
python test_gpu_mem.py
'''
import paddle

# Check if GPU is available
if paddle.is_compiled_with_cuda():
    place = paddle.CUDAPlace(0)  # Use the first GPU (change the index if needed)
else:
    print("GPU support is not available.")
    exit()

# Function to allocate and release GPU memory
def allocate_and_release_memory():
    with paddle.fluid.dygraph.guard(place):
        # Allocate GPU memory by creating a large tensor
        large_tensor = paddle.randn([10000, 10000], dtype='float32')

        # Print GPU memory usage before releasing
        print("GPU Memory Usage Before Release:")
        print(paddle.fluid.dygraph.memory_usage(place))

        # Explicitly release GPU memory by deleting the tensor
        del large_tensor

        # Print GPU memory usage after releasing
        print("GPU Memory Usage After Release:")
        print(paddle.fluid.dygraph.memory_usage(place))

if __name__ == "__main__":
    allocate_and_release_memory()
