import json
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")
    
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning. 
    一个用于SFT的、采用惰性（按需）方式加载和处理数据的数据集类
    """

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # Q: 为什么使用128？
            # 128是对一张图片经过编码后所占 Token 数量的粗略预估值（经验值），
            # 目的：用于在未加载实际图片文件之前，快速估算每个训练样本的总长度，
            # 作用：这种估算主要用于训练器的按长度分组（Group by length）策略。通过预估长度，将长短相近的样本分配到同一个 Batch 中，从而减少 Padding（填充）的浪费，提高训练效率。
            # 补充：对于像 Qwen-VL 这样支持动态分辨率的模型，实际的图像 Token 数量是变化的。但在读取数据列表阶段无法得知具体分辨率，因此使用 128 作为一个平均化的占位符来代替。
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self) -> List[int]:
        """
        功能：
            计算数据集中每个样本的模态相关长度列表。
            该属性主要用于数据加载时的采样策略（Sampler），通过正负号区分多模态数据和纯文本数据。
            具体规则如下：
            1. 计算每个样本中所有对话内容的单词总数作为基础长度。
            2. 如果样本包含图像或视频（多模态），保持长度为正数。
            3. 如果样本仅包含文本（纯文本），将长度取反为负数。
            这种机制允许 Sampler 能够感知样本的模态类型，从而在构建 Batch 时更合理地混合或区分不同模态的数据。

        参数：
            self (LazySupervisedDataset): 数据集实例本身，包含数据列表 list_data_dict。

        返回：
            List[int]: 一个整数列表，列表长度等于数据集样本数。
                - 正数：表示该样本为多模态数据（包含图像或视频）。
                - 负数：表示该样本为纯文本数据。
                - 绝对值：表示样本中文本内容的近似单词数量。

        示例：
            >>> # 假设 dataset.list_data_dict 包含两个样本：
            >>> # 样本1: {"image": "path/to/img", "conversations": [{"value": "hello world"}]}
            >>> # 样本2: {"conversations": [{"value": "pure text query"}]}
            >>> lengths = dataset.modality_lengths
            >>> print(lengths)
            [2, -3]
            # 解释：
            # 样本1包含 "image"，文本长度为 2 ("hello", "world") -> 结果为 2
            # 样本2不含媒体，文本长度为 3 ("pure", "text", "query") -> 结果为 -3
        """
        # 1> 初始化长度列表，用于存储每个样本的处理结果
        length_list = []

        # 2> 遍历数据集中的每一个样本
        for sample in self.list_data_dict:
            # 3> 计算当前样本的基础文本长度
            # 遍历 conversations 列表中的每一轮对话，统计 "value" 字段分词后的单词数量并求和
            # 注意：这里使用简单的 split() 进行分词估算，仅作为长度参考
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )

            # 4> 根据模态类型调整长度符号
            # 检查样本字典中是否存在 "image" 或 "video" 键
            # - 如果存在（多模态数据）：保持 cur_len 为正数
            # - 如果不存在（纯文本数据）：将 cur_len 变为负数 (-cur_len)
            # 这一步是为了让 Sampler 能通过正负号快速区分数据模态
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )

            # 5> 将计算结果添加到列表中
            length_list.append(cur_len)
            
        return length_list

    @property
    def pre_calculated_length(self) -> np.ndarray:
        """
        功能：
            获取数据集中预计算的样本长度数组。
            该属性检查数据集的第一个样本是否包含 "num_tokens" 字段。如果存在，则提取所有样本的
            预计算 token 数量并返回 numpy 数组；如果不存在，则打印警告并返回一个全为 1 的数组。
            这通常用于优化数据加载器的采样过程（Sampler），避免实时计算长度。

        参数：
            self (LazySupervisedDataset): 数据集实例本身，包含数据列表 list_data_dict。

        返回：
            np.ndarray: 一个包含整数的 numpy 数组。
                - 如果有预计算长度：数组包含每个样本的 "num_tokens" 值。
                - 如果无预计算长度：数组长度等于样本数，且所有元素均为 1。

        示例：
            >>> # 示例 1: 数据集包含 "num_tokens" 字段
            >>> # dataset.list_data_dict = [{"num_tokens": 100}, {"num_tokens": 200}]
            >>> lengths = dataset.pre_calculated_length
            >>> print(lengths)
            [100 200]

            >>> # 示例 2: 数据集不包含 "num_tokens" 字段
            >>> # dataset.list_data_dict = [{"text": "abc"}, {"text": "def"}]
            >>> lengths = dataset.pre_calculated_length
            >>> print(lengths)
            No pre-calculated length available.
            [1 1]
        """
        # 1> 检查是否存在预计算长度信息
        # 通过查看列表中的第一个样本是否包含 "num_tokens" 键来判断
        if "num_tokens" in self.list_data_dict[0]:
            # 2> 提取所有样本的长度信息
            # 遍历整个数据列表，收集每个样本的 "num_tokens" 值
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            
            # 3> 转换为 Numpy 数组并返回
            return np.array(length_list)
        else:
            # 4> 处理无预计算长度的情况
            # 如果不存在 "num_tokens"，则打印提示信息
            print("No pre-calculated length available.")
            
            # 5> 返回默认长度数组
            # 创建一个长度等于样本总数、所有元素均为 1 的数组作为兜底
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        功能：
            根据索引获取处理后的单个数据样本，包含自动重试和容错机制。
            该方法实现了健壮的数据加载流程：
            1. 首先尝试获取指定索引的样本，如果失败（如网络抖动），会进行有限次重试并短暂休眠。
            2. 如果指定样本持续失败，尝试获取下一个样本作为替代（应对文件损坏情况）。
            3. 如果替代样本也失败，最后再次尝试获取原样本并抛出异常。
            这种机制特别适用于大规模云端数据集加载，能有效减少因个别坏数据或IO波动导致的训练中断。

        参数：
            i (int): 要获取的样本在数据集中的索引。

        返回：
            Dict[str, torch.Tensor]: 处理后的数据字典，包含模型所需的输入张量。
                - input_ids: 输入 token ID 序列
                - labels: 标签序列
                - attention_mask: 注意力掩码
                - pixel_values: 图像像素值（如果有）
                - ... 其他模型特定字段

        示例：
            >>> # 假设 dataset 是 LazySupervisedDataset 的实例
            >>> try:
            >>>     sample = dataset[0]
            >>>     print(sample["input_ids"].shape)
            >>> except Exception as e:
            >>>     print(f"Failed to load sample: {e}")
        """
        # 1> 定义重试次数参数
        num_base_retries = 3  # 基础重试次数
        num_final_retries = 30 # 备用重试参数（当前逻辑中暂未使用）

        # 2> 阶段一：尝试获取当前索引的样本
        # 循环尝试 num_base_retries 次，处理可能的临时性错误（如云存储IO抖动）
        for attempt_idx in range(num_base_retries):
            try:
                # 获取原始数据源
                sources = self.list_data_dict[i]
                # 确保 sources 是列表格式（item_fn 预期输入为列表）
                if isinstance(sources, dict):
                    sources = [sources]
                # 调用处理函数将原始数据转换为模型输入张量
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # 如果发生异常，打印错误日志并休眠 1 秒后重试
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # 3> 阶段二：尝试获取下一个样本（容错处理）
        # 如果当前样本多次尝试均失败（可能是文件损坏），尝试读取下一个样本
        for attempt_idx in range(num_base_retries):
            try:
                # 计算下一个样本的索引，防止越界
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                # 尝试处理下一个样本
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # 如果下一个样本也失败，仅打印错误，不进行休眠
                # 这里使用 pass 继续尝试（虽然逻辑上循环内没有变化，主要是记录日志）
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        # 4> 阶段三：最后一次尝试获取原始样本
        # 如果所有容错手段都失效，最后再试一次原始索引，并允许异常抛出
        # 这样可以让上层调用者（如 DataLoader）感知到明确的错误
        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            # 直接抛出捕获到的异常，终止加载
            raise e
    
    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        """
        功能：
            处理单个数据样本，执行完整的预处理流程，包括图像/视频处理、Token化、位置编码生成等。
            这是数据加载的核心处理函数，负责将原始数据转换为模型可直接输入的 Tensor 格式。主要步骤包括：
            1. 调用 preprocess_qwen_visual 处理多模态输入并进行 Token 化。
            2. 提取并标准化图像和视频的网格尺寸信息 (grid_thw)。
            3. 计算旋转位置编码索引 (RoPE index)。
            4. 构建注意力掩码 (attention_mask)。
            5. 处理标签 (labels) 以符合训练要求（如处理 padding）。

        参数：
            sources (List[Dict]): 包含原始样本数据的列表，通常只包含一个字典元素。
                - 每个字典应包含 "image", "video", "conversations" 等键。

        返回：
            Dict[str, torch.Tensor]: 处理完成的数据字典，可以直接输入到模型中。
                - input_ids: 形状为 (Seq_Len,) 的输入 Token 序列
                - labels: 形状为 (Seq_Len,) 的训练标签
                - position_ids: 形状为 (3, Seq_Len) 的位置编码索引
                - attention_mask: 形状为 (Seq_Len,) 的注意力掩码列表
                - pixel_values: 图像特征张量（如果有）
                - image_grid_thw: 图像网格尺寸信息（如果有）
                - pixel_values_videos: 视频特征张量（如果有）
                - video_grid_thw: 视频网格尺寸信息（如果有）

        示例：
            >>> # 构造一个模拟输入
            >>> sources = [{
            >>>     "image": ["/path/to/img.jpg"],
            >>>     "conversations": [{"from": "user", "value": "<image>Describe this."}]
            >>> }]
            >>> # 调用处理函数
            >>> result = dataset._get_item(sources)
            >>> print(result["input_ids"].shape)
            torch.Size([150])
        """
        # 1> 执行多模态预处理
        # 调用 preprocess_qwen_visual 处理图像/视频并生成 input_ids 和 labels
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
        )

        # 获取序列长度，用于后续生成 mask 和 position_ids
        seq_len = data_dict["input_ids"][0].size(0)

        # 2> 处理图像网格信息 (Image Grid THW)
        # 检查是否存在图像网格信息，并确保其格式为列表
        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None
        
        # 3> 处理视频网格信息 (Video Grid THW)
        # 检查是否存在视频网格信息，并计算每帧的时间跨度
        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            # 计算每个视频网格（Grid）对应的时间跨度（秒） (temporal_patch_size / fps)
            # 这是一个关键的时间位置编码参数，用于告诉模型每个特征块代表现实世界中多长的时间
            # 计算公式：单块时长 = 时间切片大小(帧数) / 帧率(FPS)
            # 例如：temporal_patch_size=2, fps=5.0 -> 2/5.0 = 0.4秒/块
            # 最后乘以 len(video_grid_thw) 是为了给每个网格块都赋予相同的时间属性
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        # 4> 计算旋转位置编码索引 (RoPE Index)
        # 使用专门的 get_rope_index 函数生成 3D 位置索引
        # 需要传入合并尺寸、input_ids 以及图像/视频的网格信息
        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        # 5> 更新数据字典
        # 添加生成的 position_ids 和 attention_mask
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        # 6> （调试/日志用）解码并验证文本内容
        # 将 input_ids 解码回文本（保留特殊 Token）
        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        # 7> 处理标签 (Labels)
        # 将 labels 中的 ignore_index (-100) 替换为 pad_token_id，以便于解码验证
        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        # 解码标签序列，通常用于调试检查生成的标签是否正确
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:
        """
        功能：
            处理并打包（Packing）多个数据样本，将它们拼接成一个长的序列以提高训练效率。
            该方法主要用于 "Sequence Packing" 场景，它会接收多个样本的列表，分别对每个样本调用 _get_item 进行处理，
            然后将它们的所有字段（如 input_ids, labels, pixel_values 等）沿着序列维度拼接起来。
            这样可以减少 padding 带来的计算浪费，特别适用于处理长短不一的数据。

        参数：
            sources (List[Dict] or Dict): 输入数据源。
                - 如果是 List[Dict]：表示一组需要被打包在一起的样本。
                - 如果是 Dict：表示单个样本（会被视为只包含一个样本的包）。

        返回：
            Dict[str, torch.Tensor]: 打包合并后的数据字典。
                - input_ids: 拼接后的 Token 序列，形状为 (1, Total_Seq_Len)
                - labels: 拼接后的标签序列
                - position_ids: 拼接后的位置编码
                - attention_mask: 拼接后的注意力掩码（如果有）
                - pixel_values: 合并后的所有图像特征（如果有）
                - image_grid_thw: 合并后的图像网格尺寸信息（如果有）
                - pixel_values_videos: 合并后的所有视频特征（如果有）
                - video_grid_thw: 合并后的视频网格尺寸信息（如果有）

        示例：
            >>> # 假设 sources 包含两个样本
            >>> sources = [sample1, sample2]
            >>> # 调用打包函数
            >>> packed_batch = dataset._get_packed_item(sources)
            >>> print(packed_batch["input_ids"].shape)
            torch.Size([1, Total_Seq_Len])
        """

        # 1> 处理单个字典输入的边界情况
        # 如果输入是单个字典而不是列表，将其封装为列表并作为单样本处理
        # 这通常发生在非 packing 模式或只有一个样本的包中
        if isinstance(sources, dict):
            if isinstance(source, dict): # FIXME: 这里的 source 变量未定义，可能是原代码逻辑错误，假设 sources 就是当前项
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        # 2> 处理列表输入（标准的 Packing 逻辑）
        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            
            # 3> 逐个处理列表中的每个源样本
            for source in sources:
                # 确保每个 source 都是列表格式以符合 _get_item 的接口要求
                if isinstance(source, dict):
                    source = [source]
                # 断言确保每个 source 确实只包含一个样本
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                
                # 调用 _get_item 处理单个样本，并将结果收集到 data_list 中
                data_list.append(self._get_item(source))

            # 4> 拼接基础文本相关字段
            # 将所有样本的 input_ids, labels, position_ids 沿序列维度拼接
            
            # --- 1. data_list 结构说明 ---
            # data_list 是一个列表，其中每个元素 `d` 都是一个样本字典。
            # 假设 data_list 包含两个样本 [Sample_A, Sample_B]：
            # Sample_A: {
            #     "input_ids": tensor([[1, 2]]),     # Shape: (1, 2)
            #     "labels": tensor([[-100, 2]]),     # Shape: (1, 2)
            #     "position_ids": tensor([[[0, 1]]]) # Shape: (3, 1, 2) (假设第0维为3)
            # }
            # Sample_B: {
            #     "input_ids": tensor([[3, 4, 5]]),    # Shape: (1, 3)
            #     "labels": tensor([[3, 4, -100]]),    # Shape: (1, 3)
            #     "position_ids": tensor([[[0, 1, 2]]]) # Shape: (3, 1, 3)
            # }

            # --- 2. 拼接 input_ids ---
            # 操作：沿着 dim=1 (序列长度维度) 拼接所有样本的 input_ids
            # 过程：[[1, 2]] + [[3, 4, 5]] -> [[1, 2, 3, 4, 5]]
            # 形状变化：(1, 2) + (1, 3) -> (1, 5)
            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            
            # --- 3. 拼接 labels ---
            # 操作：同样沿着 dim=1 拼接所有样本的 labels
            # 过程：[[-100, 2]] + [[3, 4, -100]] -> [[-100, 2, 3, 4, -100]]
            # 形状变化：(1, 2) + (1, 3) -> (1, 5)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            
            # --- 4. 拼接 position_ids ---
            # 操作：沿着 dim=2 (序列长度维度) 拼接。注意 Qwen-VL position_ids 是 3D 的。
            # 原始 Shape 通常为 (3, 1, Seq_Len)，其中 3 代表 (时间, 高度, 宽度) 三个分量。
            # 过程：[[[0, 1]]] + [[[0, 1, 2]]] -> [[[0, 1, 0, 1, 2]]] (在第2个维度连接)
            # 形状变化：(3, 1, 2) + (3, 1, 3) -> (3, 1, 5)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            
            # 拼接 attention_mask（如果存在）
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            
            # 构建新的数据字典
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }
            # 5> 拼接图像相关字段 (如果存在)
            # 检查是否有样本包含图像数据 (pixel_values)
            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        # 将所有样本的图像特征张量沿 batch 维度 (dim=0) 拼接
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        # 将所有样本的图像网格尺寸信息沿 dim=0 拼接
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            
            # 6> 拼接视频相关字段 (如果存在)
            # 检查是否有样本包含视频数据 (pixel_values_videos)
            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        # 将所有样本的视频特征张量沿 batch 维度 (dim=0) 拼接
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        # 将所有样本的视频网格尺寸信息沿 dim=0 拼接
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    """
    功能：
        对一组张量（通常是位置编码 position_ids）进行填充（Padding）和拼接（Concatenation）。
        主要用于 DataCollator 中，处理 Batch 内样本长度不一致的情况。
        它首先找到当前 Batch 中最大的序列长度，然后将所有张量的最后维度填充到该长度（填充值为 1），
        最后沿 dim=1（Batch维度）进行拼接。

    参数：
        tensor_list (List[torch.Tensor]): 输入的张量列表。
            - 预期每个张量的形状为 (3, 1, Seq_Len)，其中 Seq_Len 各不相同。
            - 这是 Qwen-VL 特有的位置编码结构，第0维表示3个分量（T, H, W）。

    返回：
        torch.Tensor: 拼接后的批量张量。
            - 形状为 (3, Batch_Size, Max_Seq_Len)。
            - 所有样本已对齐到相同的最大长度。

    示例：
        >>> # 假设有两个样本的位置编码，长度分别为 2 和 3
        >>> t1 = torch.ones(3, 1, 2)
        >>> t2 = torch.ones(3, 1, 3)
        >>> batch_pos = pad_and_cat([t1, t2])
        >>> print(batch_pos.shape)
        torch.Size([3, 2, 3])
    """
    # 1> 计算最大序列长度
    # 遍历列表中的所有张量，获取它们第 2 维（序列长度维）的最大值
    # 输入张量形状假设：(3, 1, Seq_Len) -> 我们关注 Seq_Len
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    # 2> 对每个张量进行填充
    for tensor in tensor_list:
        # 计算需要填充的长度：最大长度 - 当前长度
        pad_length = max_length - tensor.shape[2]
        
        # 执行填充操作
        # F.pad 参数说明：(0, pad_length) 表示仅在最后一个维度的右侧填充 pad_length 个单位
        # mode="constant", value=1 表示使用常数 1 进行填充（Qwen-VL 位置编码通常用 1 作为 Padding 值）
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    # 3> 拼接张量
    # 沿着 dim=1 进行拼接。
    # 变化过程：List[(3, 1, Max_Len), ...] -> Tensor(3, Batch_Size, Max_Len)
    # 这里的 dim=1 实际上对应于 Batch 维度（因为 dim=0 是 RoPE 的分量维度）
    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        功能：
            将一批（Batch）独立的样本整理成模型所需的批量输入格式。
            主要负责执行 Padding（填充）操作，使 Batch 内所有样本的序列长度对齐，并合并图像和视频特征。
            具体处理流程：
            1. 提取 input_ids、labels 和 position_ids。
            2. 对文本序列进行 Padding，对齐到当前 Batch 的最大长度。
            3. 截断超长序列以适应模型最大上下文长度。
            4. 生成 Attention Mask。
            5. 合并 Batch 内所有的图像和视频特征张量。

        参数：
            instances (Sequence[Dict]): 输入的样本列表，每个样本是一个包含各字段（input_ids, labels 等）的字典。
                - 列表长度通常等于 Batch Size。

        返回：
            Dict[str, torch.Tensor]: 整理后的批量数据字典。
                - input_ids: (Batch_Size, Max_Seq_Len)
                - labels: (Batch_Size, Max_Seq_Len)
                - attention_mask: (Batch_Size, Max_Seq_Len)
                - position_ids: (3, Batch_Size, Max_Seq_Len)
                - pixel_values: (Total_Images, C, H, W) 或 None
                - image_grid_thw: (Total_Images, 3) 或 None
                - pixel_values_videos: (Total_Videos, C, T, H, W) 或 None
                - video_grid_thw: (Total_Videos, 3) 或 None

        示例：
            >>> # 假设 collator 是 DataCollatorForSupervisedDataset 的实例
            >>> # instances 是包含两个样本的列表
            >>> batch = collator(instances)
            >>> print(batch["input_ids"].shape)
            torch.Size([2, 512])  # 假设最大长度被填充或截断到 512
        """
        # 1> 提取基础字段
        # 从每个 instance 中提取 input_ids, labels, position_ids，组成元组列表
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        # 2> 移除 Batch 维度（如果存在）以便后续 Padding
        # 通常 input_ids 形状为 (1, Seq_Len)，需要变为 (Seq_Len,)
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        # 3> 执行序列填充 (Padding)
        # 使用 pad_sequence 将列表中的 Tensor 填充到最大长度，并堆叠为 (Batch, Max_Len)
        # input_ids 使用 tokenizer 的 pad_token_id 填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labels 使用 IGNORE_INDEX (-100) 填充，计算 Loss 时会忽略这些位置
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # 4> 处理位置编码
        # position_ids 需要特殊的 Padding 和拼接逻辑 (参考 pad_and_cat 函数)
        position_ids = pad_and_cat(position_ids)

        # 5> 执行长度截断
        # 确保序列长度不超过模型的最大上下文限制 (model_max_length)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        # 6> 构建基础 Batch 字典
        # 生成 attention_mask：非 Padding 位置为 1，Padding 位置为 0
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 7> 收集图像和视频资源
        # 提取每个样本中的 pixel_values 和 video_pixel_values
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )

        # 8> 合并图像特征 (如果存在)
        if len(images) != 0:
            # 将所有样本的图像特征沿第0维拼接
            concat_images = torch.cat([image for image in images], dim=0)
            # 同样收集并拼接图像网格信息
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        # 9> 合并视频特征 (如果存在)
        if len(videos) != 0:
            # 将所有样本的视频特征沿第0维拼接
            concat_videos = torch.cat([video for video in videos], dim=0)
            # 收集并拼接视频网格信息
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        # 10> 更新 Batch 字典
        # 将处理好的多模态数据和位置编码加入 batch
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """
    类功能：
        专用于"Sequence Packing"（序列打包）模式的数据整理器（Data Collator）。
        它将 Batch 中的多个样本直接拼接成一个超长序列，通过 cumulative sum attention mask 机制来区分不同样本，
        从而极大地提高训练效率，减少 Padding 造成的计算浪费。

    继承关系：
        继承自 DataCollatorForSupervisedDataset。
        重写了 `__call__` 方法以实现 Flatten/Packing 特有的拼接逻辑。

    应用场景：
        1. 当 `data_args.data_packing` 为 True 时，用于替代默认的 Collator。
        2. 在处理大量短文本或长短不一的多模态数据时，通过 Packing 技术填满 Context Window，提升吞吐量。

    使用示例：
        >>> tokenizer = AutoTokenizer.from_pretrained(...)
        >>> collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        >>> # 假设 instances 是已经经过 Packing 预处理的样本列表
        >>> batch = collator(instances)

    数据属性：
        tokenizer: transformers.PreTrainedTokenizer
            用于处理文本的 Tokenizer 实例。
            主要用于访问 `pad_token_id` 等属性（虽然在此类中未直接使用，但继承自父类接口保持一致）。
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        功能：
            执行 Flattened 模式的 Batch 整理逻辑。
            不同于普通 Collator 的 Padding 操作，该方法将所有样本的 input_ids、labels、position_ids 
            沿着序列维度直接拼接（Concatenate）。同时，它通过计算 Attention Mask 的累积和（CuSeL / Cumulative Sequence Length），
            生成 Flash Attention 所需的索引信息，从而在同一个 Batch 中并行训练多个互不干扰的样本。

        参数：
            instances (Sequence[Dict]): 输入的样本列表。
                在 Packing 模式下，这里的每个 "instance" 可能已经是一个包含多个原始样本的 Packed Block。

        返回：
            Dict[str, torch.Tensor]: 整理后的批量数据字典。
                - input_ids: (1, Total_Seq_Len) - 所有样本拼接成的一个长序列。
                - labels: (1, Total_Seq_Len) - 对应的标签序列。
                - attention_mask: (Total_Samples + 1,) - 累积序列长度列表（CuSeL），用于 Flash Attention。
                - position_ids: (3, 1, Total_Seq_Len) - 拼接后的位置编码。
                - pixel_values: (Total_Images, ...) - 所有图像特征的拼接。
                - ... 其他多模态相关字段。

        示例：
            >>> # 假设 instances 包含两个样本，长度分别为 L1, L2
            >>> batch = collator(instances)
            >>> print(batch["input_ids"].shape)
            torch.Size([1, L1 + L2])
            >>> print(batch["attention_mask"]) # [0, L1, L1+L2]
        """
        # 1> 提取并解包基础字段
        # 从每个 instance 中提取关键字段，分别形成列表
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )

        # 2> 构建 Flash Attention 所需的 CuSeL (Cumulative Sequence Length)
        # attention_mask 在这里实际上存储的是每个样本的真实序列长度（List[int]）
        # itertools.chain 将嵌套的长度列表展平
        #
        # === 详细处理流程举例 ===
        # 假设当前 Batch (instances) 包含 2 个样本：
        #   Sample 1: 长度为 3，attention_mask=[3] (或内部包含多个子段如 [1, 2])
        #   Sample 2: 长度为 4，attention_mask=[4]
        # 
        # 步骤 A: 提取 attention_mask 字段
        #   instance["attention_mask"] 可能是一个列表，如 [3]
        #   生成器表达式结果：[[3], [4]]
        #
        # 步骤 B: 使用 itertools.chain 展平
        #   list(itertools.chain(*[[3], [4]])) -> [3, 4]
        #   此时得到的 attention_mask 变量实际内容是：[3, 4] (即每个样本的有效长度列表)
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        
        # 3> 计算累积序列长度
        # 构造 [0, len1, len2, ...] 这样的序列
        # [0] 是起始偏移量
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        # cumsum 计算前缀和：[0, len1, len1+len2, ...]
        # 这个张量后续将被用作 Flash Attention 的 cu_seqlens 参数
        # 
        # === 举例续 ===
        # 接上例，attention_mask 为 [3, 4]
        # 1. 拼接 0: [0] + [3, 4] -> [0, 3, 4]
        # 2. 计算前缀和 (cumsum): 
        #    - index 0: 0
        #    - index 1: 0 + 3 = 3
        #    - index 2: 3 + 4 = 7
        #    结果 cumsum_seq_lens 为 tensor([0, 3, 7])
        # 
        # 物理含义：
        #    - Sample 1 的 Token 范围是 [0, 3)
        #    - Sample 2 的 Token 范围是 [3, 7)
        #    - 这种格式是 Flash Attention 库（如 flash-attn）处理变长序列的标准输入格式 (cu_seqlens)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        # 4> 拼接文本相关张量
        # input_ids: (1, Len1) + (1, Len2) -> (1, Total_Len)
        input_ids = torch.cat(input_ids, dim=1)
        # labels: (1, Len1) + (1, Len2) -> (1, Total_Len)
        labels = torch.cat(labels, dim=1)
        # position_ids: (3, 1, Len1) + (3, 1, Len2) -> (3, 1, Total_Len)
        position_ids = torch.cat(position_ids, dim=2)

        # 5> 构建基础 Batch 字典
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # 注意：这里的 attention_mask 不再是 0/1 矩阵，而是累积长度索引
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )

        # 6> 收集并合并多模态资源 (图像/视频)
        # 这一部分的逻辑与 DataCollatorForSupervisedDataset 相同，只是简单地将所有媒体资源堆叠
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )

        # 7> 合并图像特征 (如果存在)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        # 8> 合并视频特征 (如果存在)
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        # 9> 更新 Batch 字典
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """
    功能：
        构建和配置用于监督微调（Supervised Fine-Tuning, SFT）的数据模块。
        该工厂函数负责实例化数据集对象（LazySupervisedDataset）和相应的数据整理器（Data Collator）。
        它根据 `data_args` 中的配置参数（如是否启用 Packing）来决定使用标准整理器还是扁平化（Packing）整理器。

    参数：
        processor (transformers.ProcessorMixin): 
            模型处理器，通常包含 tokenizer、image_processor 等组件。
            用于传递给数据集和整理器以处理多模态输入。
        data_args (DataArguments): 
            数据配置参数对象，包含训练相关的所有超参数。
            - dataset_use: 指定使用的数据集名称。
            - data_packing (bool): 是否启用序列打包模式。
            - data_flatten (bool): 是否启用扁平化模式（通常等同于 Packing）。
            - 其他图像/视频处理参数。

    返回：
        Dict: 一个包含训练所需关键组件的字典，符合 Hugging Face Trainer 的接口要求。
            - "train_dataset": 训练数据集实例 (LazySupervisedDataset)。
            - "eval_dataset": 验证数据集实例 (通常为 None，除非实现了验证集逻辑)。
            - "data_collator": 数据整理器实例 (DataCollatorForSupervisedDataset 或 FlattenedDataCollatorForSupervisedDataset)。

    示例：
        >>> # 假设已初始化 processor 和 data_args
        >>> data_module = make_supervised_data_module(processor, data_args)
        >>> trainer = Trainer(
        ...     model=model,
        ...     **data_module,  # 自动解包为 train_dataset, data_collator 等
        ...     args=training_args
        ... )
    """
    # 1> 实例化训练数据集
    # 使用 LazySupervisedDataset 进行懒加载，减少内存占用
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)

    # 2> 实例化数据整理器 (Data Collator)
    # 根据配置决定使用哪种整理器
    # - data_flatten 或 data_packing 为 True: 使用支持序列打包的 FlattenedDataCollator
    #   这种模式下，多个样本会被拼接成长序列以提高训练效率
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    
    # - 默认模式: 使用标准的数据整理器，仅进行普通的 Padding
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    
    # 3> 返回数据模块字典
    # 格式符合 HF Trainer 的入参要求
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
