'''
python test_gpu_mem.py
'''
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

# Check if GPU is available
if paddle.is_compiled_with_cuda():
    place = paddle.CUDAPlace(0)  # Use the first GPU (change the index if needed)
else:
    print("GPU support is not available.")
    exit()

# Function to allocate and release GPU memory
def allocate_and_release_memory():
    with dygraph.guard(place):
        # Allocate GPU memory by creating a large tensor
        large_tensor = paddle.randn([10000, 10000], dtype='float32')

        # Print GPU memory usage before releasing
        memory_before_release = fluid.dygraph.memory_usage(place)
        print("GPU Memory Usage Before Release:")
        print(memory_before_release)

        # Explicitly release GPU memory by deleting the tensor
        del large_tensor

        # Print GPU memory usage after releasing
        memory_after_release = fluid.dygraph.memory_usage(place)
        print("GPU Memory Usage After Release:")
        print(memory_after_release)

    # Print the difference in memory usage
    memory_diff = memory_after_release[0] - memory_before_release[0]
    print("Memory Difference (bytes):", memory_diff)

if __name__ == "__main__":
    allocate_and_release_memory()

