import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import torch


def occupy_multiple_gpus(mem_ratio=0.9):
    """
    占用所有可用 GPU 的显存，防止其他进程使用。

    参数：
    mem_ratio (float): 占用显存的比例（0.0 到 1.0）。
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未检测到可用的 GPU！")
        return

    print(f"检测到 {num_gpus} 张 GPU，开始占用显存...")

    tensor_lists = []  # 存储占用的张量，防止被回收

    try:
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)  # 切换到当前 GPU
            total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
            target_mem = int(total_mem * mem_ratio)  # 计算目标占用量

            # 预分配大张量，逐步填充显存
            tensor_list = []
            block_size = 256 * 1024 * 1024  # 256MB 块
            num_blocks = target_mem // block_size

            for _ in range(num_blocks):
                tensor_list.append(torch.empty((block_size // 4,), dtype=torch.float32, device=f"cuda:{gpu_id}"))

            tensor_lists.append(tensor_list)  # 记录已分配的张量

            print(f"成功占用 GPU-{gpu_id}: {mem_ratio * 100:.1f}% 的显存 (约 {target_mem / (1024 ** 3):.2f} GB)")

        # 保持进程运行，防止显存被释放
        input("按 Enter 释放显存并退出...")

    except RuntimeError as e:
        print(f"运行时错误：{e}")
        print("可能是显存已满，无法分配更多张量。")
        input("按 Enter 释放显存并退出...")

    finally:
        # 释放所有 GPU 显存
        tensor_lists.clear()
        torch.cuda.empty_cache()
        print("显存已释放，程序退出。")


if __name__ == "__main__":
    occupy_multiple_gpus(mem_ratio=0.9)