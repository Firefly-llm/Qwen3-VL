import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
import string
import traceback

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample

from qwen2_vl.model import Qwen2VLChat
from qwen_vl_utils import process_vision_info


def run_inference(args):
    """
    功能：
        在 MMMU 数据集上运行模型推理，并保存预测结果。
        该函数涵盖了数据集加载、图像预处理、模型初始化、Prompt 构建（支持 CoT）、推理执行以及中间结果持久化等完整流程。

    参数：
        args (argparse.Namespace): 包含所有运行时参数的对象。
            - dataset (str): 数据集名称。
            - model_path (str): 模型权重存放路径。
            - output_file (str): 结果输出文件路径（通常为 .jsonl 格式）。
            - use_cot (bool): 是否启用思维链（Chain of Thought）提示词。
            - cot_prompt (str, 可选): 自定义的 CoT 提示词内容。

    返回：
        None: 函数执行结果直接保存到磁盘文件。

    示例：
        >>> args = argparse.Namespace(dataset="MMMU_DEV_VAL", model_path="./qwen2-vl", output_file="./results.jsonl", use_cot=True)
        >>> run_inference(args)
        Loading HuggingFace model from ./qwen2-vl...
        Inference completed. Results saved to ./results.jsonl
    """
    # 1> 加载 MMMU 数据集
    data = load_dataset(args.dataset)
    
    # 2> 配置图像存储根目录
    # LMUData 环境变量指向评测数据的统一存放路径
    img_root = os.path.join(os.environ['LMUData'], 'images', 'MMMU')
    os.makedirs(img_root, exist_ok=True)
    
    # 3> 定义图像转储函数
    # 封装 dump_image，方便传递给模型类进行自动图像处理
    def dump_image_func(line):
        return dump_image(line, img_root)
    
    # 4> 确保输出文件所在目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 5> 配置思维链（CoT）提示词
    cot_prompt = ""
    if args.use_cot:
        # 如果启用 CoT 但未提供具体 Prompt，则使用默认的推理增强 Prompt
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        print(f"Using CoT prompt: {cot_prompt}")

    # 6> 初始化多模态模型 (Qwen2-VL)
    print(f"Loading HuggingFace model from {args.model_path}")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        use_custom_prompt=True,
        # 指定图像 Token 的最小和最大分辨率限制
        min_pixels=1280*28*28,
        max_pixels=5120*28*28
    )
    # 关联图像转储逻辑，以便推理时能自动保存 Base64 图片到本地
    model.set_dump_image(dump_image_func)

    # 7> 遍历数据集执行推理
    results = []
    for i in tqdm(range(len(data)), desc="Running inference"):
        # 获取当前行数据
        line = data.iloc[i]
        index = line['index']
        
        # 8> 转换 Pandas Series 为字典，并确保数值类型 JSON 可序列化
        # 处理 NumPy 类型转换为 Python 原生类型
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)
        
        # 9> 构建模型输入的消息格式
        messages = model.build_prompt(line, args.dataset)
        
        # 10> 注入 CoT 提示词（如果启用）
        # 将 CoT 内容追加到最后一条文本消息中
        if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
            messages[-1]['value'] += cot_prompt
            
        # 11> 调用模型生成回答
        response = model.generate(messages)
            
        # 打印调试信息
        print(f"response: {response}")
        print(f"annotation answer: {line['answer']}")
        print('-' * 50)
        
        # 12> 封装推理结果
        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict, # 原始标注信息
            "task": args.dataset,
            "result": {"gen": response}, # 模型生成的内容
            "messages": messages # 最终发送给模型的完整 Prompt 结构
        }
        results.append(result)
        
        # 13> 定期保存中间结果（每 10 条保存一次），防止程序崩溃导致数据丢失
        if i % 10 == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')
            
    # 14> 全部推理完成后保存最终结果
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"Inference completed. Results saved to {args.output_file}")

def run_evaluation(args):
    """
    功能：
        对模型生成的推理结果进行自动化评估，计算准确率。
        该函数支持加载推理 JSONL 文件，并利用“评测模型（Judge Model）”配合规则匹配来判定模型回答是否正确。

    参数：
        args (argparse.Namespace): 包含评测参数的对象。
            - input_file (str): 待评测的推理结果文件（JSONL）。
            - dataset (str): 对应的数据集名称。
            - eval_model (str): 辅助提取/判定答案的评测模型 ID。
            - api_type (str): 评测模型所使用的接口类型（'mit' 或 'dash'）。
            - nproc (int): 并行处理的任务数。
            - output_file (str): 评测结果保存路径（CSV）。

    返回：
        None: 评测统计结果和详细记录直接保存到磁盘（CSV 和 JSON）。

    示例：
        >>> args = argparse.Namespace(input_file="results.jsonl", dataset="MMMU_DEV_VAL", nproc=4, output_file="eval.csv")
        >>> run_evaluation(args)
        len(data): 100
        Accuracy for ...: 0.8500
        Results saved to eval.csv
    """
    # 1> 加载并解析推理结果文件 (JSONL)
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            # 将模型生成的推理内容存入 prediction 字段
            annotation["prediction"] = job["result"]["gen"]
            results.append(annotation)
            
    # 2> 将推理结果转换为 DataFrame，并进行初步清洗
    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index') # 按索引排序
    data['prediction'] = [str(x) for x in data['prediction']]

    # 3> 统一字段名称的大小写逻辑
    # 除非是选项标签（A-Z），否则将所有 DataFrame 键转换为小写
    for k in data.keys():
        # string.ascii_uppercase 是 Python 标准库 string 模块中的一个常量字符串，其值是 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # 4> 加载对应的原始数据集元数据，用于获取标准答案 (GT)
    meta = load_dataset(args.dataset)

    # 5> 验证推理数据与数据集元数据的一致性
    # 确保推理文件中的索引和问题描述在原始数据集中完全匹配
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            f'eval_file should be the same as or a subset of dataset MMMU_DEV_VAL'
        )

    # 6> 获取标准答案并进行预处理
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    # 调用 MMMU_preproc 进行数据集特定的预处理
    data = MMMU_preproc(data)
    # 规范化答案标签：确保答案是单个大写字母，非法答案兜底设为 'A'
    # 示例：answer_map = {'q1': 'B', 'q2': 'invalid'} -> {'q1': 'B', 'q2': 'A'}
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}
    
    # 过滤数据：剔除那些在答案映射表中找不到对应 ID 的预测行
    # 示例：data['index'] = ['q1', 'q3'], answer_map = {'q1': 'B'} -> data['index'] = ['q1']
    # data['index'].isin(answer_map) 返回一个布尔索引，表示 data['index'] 中的每个元素是否在 answer_map 中
    # 具体示例：
    # 假设 data (DataFrame) 包含 3 行数据：
    #      index  prediction
    #   0     q1  "选A"
    #   1     q2  "选B"  <-- 此行将被剔除，因为 ID 'q2' 缺失
    #   2     q3  "选C"
    #
    # 假设 answer_map = {'q1': 'A', 'q3': 'C'} (注意：这里没有 'q2')
    #
    # 过滤过程：
    # 1. data['index'] -> 提取出列 ['q1', 'q2', 'q3']
    # 2. data['index'].isin(answer_map) -> 生成布尔掩码 [True, False, True]
    # 3. data[...] -> 应用掩码，保留 True 的行
    #
    # 最终结果 data：
    #      index  prediction
    #   0     q1  "选A"
    #   2     q3  "选C"
    data = data[data['index'].isin(answer_map)]
    
    # 注入真实答案 (GT)：根据索引从 answer_map 中查表并赋值给新列 'GT'
    # 示例：data['index'] = ['q1'], answer_map = {'q1': 'B'} -> data['GT'] = ['B']
    data['GT'] = [answer_map[idx] for idx in data['index']]
    
    # 7> 构造待评测的任务项列表
    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        items.append(item)

    # 8> 初始化辅助判定的评测模型 (Judge Model)
    model = build_judge(args.eval_model, args.api_type)

    # 9> 封装评测任务，每个任务包含模型对象和单个数据项
    eval_tasks = []
    for item in items:
        eval_tasks.append((model, item))

    # 10> 执行评测过程
    eval_results = []
    # 检查是否开启调试模式（单线程处理前 5 条数据）
    debug = os.environ.get('DEBUG', '').lower() == 'true'
    if debug:
        print("Running in debug mode with first 5 samples...")
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # 正常模式：使用线程池 (ThreadPoolExecutor) 并行处理所有样本
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)
    
    # 11> 计算总体准确率 (Accuracy)
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
    
    # 12> 按数据子集 (Split) 进行分类统计
    results_by_split = {}
    for result in eval_results:
        split = result.get('split', 'unknown')
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)
    
    # 计算并打印每个子集的准确率
    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        split_accuracy = sum(r['hit'] for r in split_results) / len(split_results)
        accuracy_by_split[split] = split_accuracy
        print(f"Accuracy for {split} split: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")
    
    # 13> 保存详细的评测结果到 CSV 文件
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)

    # 14> 保存汇总后的准确率指标到 JSON 文件
    with open(args.output_file.replace('.csv', '_acc.json'), 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split
        }, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="MMMU Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv")
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-3.5-turbo-0125", 
                            choices=["gpt-3.5-turbo-0125","gpt-4-0125-preview"],
                            help="Model to use for evaluation")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type to use for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()

    os.environ['LMUData'] = args.data_dir
    
    if args.mode == "infer":
        run_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
