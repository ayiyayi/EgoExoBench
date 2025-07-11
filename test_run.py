import os
from vlmeval.dataset import build_dataset
from multiprocessing import Pool

def worker(idx):
    # 直接使用全局的 dataset 对象
    global dataset
    dataset.build_prompt(idx, True)
    
if __name__ == '__main__':    
    os.environ['LMUData'] = '/mnt/petrelfs/heyuping/VLMEvalKit/EgoExoBench/videos'
    dataset = build_dataset('EgoExoBench_MCQ')
    
    for i in range(820, 1700):
        dataset.build_prompt(i, False)
    
    # # 并行处理数量（建议设置为 CPU 核心数）
    # num_processes = 8  # 可根据你的机器配置调整
    # # 使用 Pool 并行执行
    # with Pool(processes=num_processes) as pool:
    #     pool.map(worker, range(7350))


# 在 MMBench_DEV_EN、MME 和 SEEDBench_IMG 上使用 IDEFICS-80B-Instruct 仅进行推理
# python test_run.py --data EgoExoBench_MCQ --model qwen_chat --verbose --mode infer
# srun -p HOD_S1 -N1 --gres=gpu:0 --cpus-per-task 16 --quotatype=reserved python test_run.py