from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from .base import BaseModel
# mixin: 一种代码复用模式：将一组功能“混入”到多个类中，而不作为主继承层级的一部分，它是可组合的功能模块
from .prompt import Qwen2VLPromptMixin
from .util import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    """
    功能：
        根据当前分布式配置与可用 GPU 数量，生成 Qwen2-VL 的分层加载映射表。
        该映射表用于将模型各层与视觉模块分配到不同 GPU 上，以均衡显存占用。

    参数：
        无。

    返回：
        Dict[str, int]: 组件名称到 GPU 索引的映射字典，用于 `device_map` 参数。

    示例：
        >>> device_map = split_model()
        >>> device_map["model.layers.0"] >= 0
        True
        >>> device_map["visual"] >= 0
        True
    """
    # 1> 初始化设备映射字典
    device_map = {}

    # 2> 获取物理 GPU 总数
    total_gpus = torch.cuda.device_count()
    # 3> 获取当前进程 rank 与 world_size（分布式并行信息）
    rank, world_size = get_rank_and_world_size()
    # 4> 计算当前进程可使用的 GPU 数量
    num_gpus = total_gpus // world_size
    # 5> 80 为模型层数，+8 为视觉模块虚拟层的显存占位
    num_layers = 80 + 8
    # 6> 计算每张 GPU 的平均层数（向上取整）
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    # 7> 构造每张 GPU 的层数分配列表
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    # 8> 调整首卡层数，预留更多显存给视觉与嵌入
    num_layers_per_gpu[0] -= 6
    # 9> 调整末卡层数，预留显存给输出层与归一化
    num_layers_per_gpu[-1] -= 2
    # 10> 初始化当前层计数器
    layer_cnt = 0

    # 11> 遍历每张 GPU 的分配层数，生成层到 GPU 的映射
    for i, num_layer in enumerate(num_layers_per_gpu):
        # 12> 在当前 GPU 上依次放置指定数量的层
        for j in range(num_layer):
            # 13> 将第 layer_cnt 层映射到对应 GPU（考虑分布式 rank）
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            # 14> 递增层计数器
            layer_cnt += 1

    # 15> 计算最后一张 GPU 的全局索引
    last_gpu = rank + (num_gpus - 1) * world_size
    # 16> 视觉模块放在首卡，减少跨卡通信
    device_map['visual'] = rank
    # 17> 词嵌入层放在首卡，配合视觉输入构建
    device_map['model.embed_tokens'] = rank
    # 18> 归一化层放在末卡，靠近输出头
    device_map['model.norm'] = last_gpu
    # 19> 旋转位置编码放在末卡
    device_map['model.rotary_emb'] = last_gpu
    # 20> 语言模型输出头放在末卡
    device_map['lm_head'] = last_gpu
    # 21> 返回完整的设备映射表
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    """Qwen2-VL 多模态对话模型封装类，负责加载模型并执行图文/视频对话推理。

    继承关系：
        继承 `Qwen2VLPromptMixin` 与 `BaseModel`，分别提供提示词构建能力与通用输入预处理流程。

    应用场景：
        1. 在评测任务中加载 Qwen2-VL 并生成文本回答或解析视觉内容。
        2. 在服务端推理中处理图像/视频输入并返回对话式响应。

    使用示例：
        >>> model = Qwen2VLChat(
        ...     model_path="Qwen/Qwen2-VL-7B-Instruct",
        ...     max_new_tokens=512,
        ...     top_p=0.8,
        ...     temperature=0.2,
        ... )

        >>> model = Qwen2VLChat(
        ...     model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        ...     system_prompt="你是一个多模态助手",
        ...     post_process=True,
        ... )

    数据属性：
        INSTALL_REQ: bool
            标记该类是否需要额外安装依赖。默认 False。
            约束：仅用于上层逻辑判断，不影响本类运行。

        INTERLEAVE: bool
            是否支持图文交错输入。默认 True。
            约束：配合 BaseModel 的输入检查使用。

        VIDEO_LLM: bool
            是否支持视频输入。默认 True。
            约束：仅当 `type == "video"` 且依赖可用时生效。

        model_path: str
            模型权重与处理器的本地路径或远程仓库标识。
            约束：不能为空；用于选择 Qwen2 或 Qwen2.5 分支。

        processor: Any
            文本/视觉处理器实例，用于构建模型输入。
            默认由 `transformers` 自动加载。

        model: Any
            模型实例，已根据显存与配置分配到设备。
            约束：模型加载失败会抛出异常。

        min_pixels: Optional[int]
            视觉输入最小像素限制，用于图像缩放控制。
            默认 None，表示不强制限制。

        max_pixels: Optional[int]
            视觉输入最大像素限制，用于图像缩放控制。
            默认 None，表示不强制限制。

        generate_kwargs: Dict[str, Any]
            文本生成参数集合（如 top_p、temperature）。
            默认由构造函数参数组合生成。

        system_prompt: Optional[str]
            系统提示词，若不为 None 会插入对话开头。
            约束：为空则不插入 system 消息。

        verbose: bool
            是否打印调试信息。默认 False。

        post_process: bool
            是否对输出进行 \\boxed{} 内容截取。默认 False。
            约束：仅当输出包含 \\boxed{...} 时生效。

        fps: float
            视频抽帧帧率，默认 2.0。
            约束：当不为 None 时优先生效。

        nframe: int
            视频抽帧总数上限，默认 64。
            约束：仅当 fps 为 None 时作为备选策略。

        FRAME_FACTOR: int
            视频帧数对齐因子，默认 2。
            约束：用于帧数下调时的整除对齐。
    """
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
    ):
        """初始化 Qwen2-VL 模型与处理器，并配置推理参数。

        功能：
            1> 解析并保存推理参数与视觉约束配置。
            2> 按模型路径判断 Qwen2/Qwen2.5 分支并加载处理器。
            3> 根据显存与分布式策略选择 device_map，并加载模型到 GPU。

        参数：
            model_path (str): 模型权重或仓库路径，不能为空。
            min_pixels (int | None, optional): 图像最小像素限制，默认 None。
            max_pixels (int | None, optional): 图像最大像素限制，默认 None。
            max_new_tokens (int, optional): 生成最大 token 数，默认 2048。
            top_p (float, optional): nucleus 采样阈值，默认 0.001。
            top_k (int, optional): top-k 采样阈值，默认 1。
            temperature (float, optional): 采样温度，默认 0.01。
            repetition_penalty (float, optional): 重复惩罚系数，默认 1.0。
            use_custom_prompt (bool, optional): 是否使用自定义提示词，默认 True。
            system_prompt (str | None, optional): 系统提示词，默认 None。
            post_process (bool, optional): 是否截取 \\boxed{} 内容，默认 False。
            verbose (bool, optional): 是否输出调试信息，默认 False。

        返回：
            None。

        示例：
            >>> model = Qwen2VLChat(model_path="Qwen/Qwen2-VL-7B-Instruct")
            >>> model = Qwen2VLChat(
            ...     model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            ...     max_new_tokens=512,
            ...     top_p=0.8,
            ...     temperature=0.2,
            ... )
        """
        # 1> 初始化父类（处理自定义提示词配置）
        super().__init__(use_custom_prompt=use_custom_prompt)
        # 2> 保存视觉输入的像素限制配置
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # 3> 组装生成参数，供模型推理调用
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        # 4> 保存系统提示词与调试开关
        self.system_prompt = system_prompt
        self.verbose = verbose
        # 5> 是否开启 \\boxed{} 后处理
        self.post_process = post_process
        # 6> 视频默认抽帧参数
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        # 7> 获取分布式信息（用于 device_map 决策）
        rank, world_size = get_rank_and_world_size()
        # 8> 校验模型路径并保存
        assert model_path is not None
        self.model_path = model_path
        # 9> 初始化模型类占位
        MODEL_CLS = None

        # 10> 判断是否为 Qwen2.5 系列，选择对应处理器与模型类
        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        # 11> 获取 GPU 显存信息并校验可用性
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # 12> 大模型（32B/72B）走手动分层切分策略
        if '72b' in self.model_path.lower() or '32b' in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
            )
            self.model.eval()
        # 13> AUTO_SPLIT 启用时，要求单进程并自动切分
        elif auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
            )
        # 14> 默认策略：先加载到 CPU，再整体转移到 CUDA
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
            )
            self.model.cuda().eval()
        # 15> 释放加载过程中的临时显存
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """将通用输入结构转换为模型期望的多模态内容列表。

        功能：
            1> 遍历输入内容，根据类型构建 image/video/text 结构。
            2> 对图像路径与视频路径进行 URL 规范化。
            3> 根据数据集与参数设置像素或帧率/帧数约束。

        参数：
            inputs (list[dict[str, str]]): 输入内容列表，每项包含 `type` 与 `value`。
            dataset (str | None, optional): 数据集名称，用于特定数据集配置。

        返回：
            list[dict[str, str]]: 模型可直接使用的内容列表。

        示例：
            >>> model._prepare_content([{"type": "text", "value": "hello"}])
            [{'type': 'text', 'text': 'hello'}]
            >>> model._prepare_content([{"type": "image", "value": "/tmp/a.png"}], dataset="mmmu")
            [{'type': 'image', 'image': 'file:///tmp/a.png'}]
        """
        # 1> 初始化内容列表
        content = []
        # 2> 逐条处理输入项
        for s in inputs:
            # 3> 处理图片类型
            if s['type'] == 'image':
                # 3.1> 规范化图片路径为 URL
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                # 3.2> OCRBench 使用固定最小像素配置
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    # 3.3> 若配置了 max_pixels 则补充
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    # 3.4> 通用场景下按配置注入 min/max_pixels
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            # 4> 处理视频类型
            elif s['type'] == 'video':
                # 4.1> 规范化视频路径为 URL
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                # 4.2> 优先使用 fps 作为抽帧策略
                if self.fps is not None:
                    item['fps'] = self.fps
                # 4.3> 否则使用 nframe 作为抽帧策略
                elif self.nframe is not None:
                    import cv2
                    # 4.3.1> 打开视频并读取总帧数
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    # 4.3.2> 若总帧数不足，则按 FRAME_FACTOR 向下对齐
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        # 4.3.3> 否则使用配置的 nframe
                        item['nframes'] = self.nframe
            # 5> 处理文本类型
            elif s['type'] == 'text':
                # 5.1> 直接映射到 text 字段
                item = {'type': 'text', 'text': s['value']}
            else:
                # 6> 未知类型直接抛错
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            # 7> 将处理后的条目加入列表
            content.append(item)
        # 8> 返回规范化后的内容列表
        return content

    def generate_inner(self, message, dataset=None):
        """执行单轮多模态生成并返回文本结果。

        功能：
            1> 构建系统/用户消息并预处理视觉输入。
            2> 调用处理器生成模型输入，再执行推理。
            3> 解码生成结果，并可选进行 \\boxed{} 内容截取。

        参数：
            message (list[dict]): 已规范化的输入内容列表。
            dataset (str | None, optional): 数据集名称，用于视觉参数调整。

        返回：
            str: 模型生成的文本结果。

        示例：
            >>> msg = [{"type": "text", "value": "你好"}]
            >>> model.generate_inner(msg, dataset="mmmu")
            '...'
        """
        # 1> 尝试导入视觉信息处理工具
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            # 依赖缺失时给出明确错误信息
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        # 2> 初始化对话消息列表
        messages = []
        # 若配置了系统提示词则加入 system 消息
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        # 加入用户消息，并预处理多模态内容
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        # 调试模式打印原始消息
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # 输出 messages 便于排查
        print(f"messages: {messages}")
        # 使用模板拼接对话文本，并添加生成提示
        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        # 解析视觉输入为模型可用的 images/videos
        images, videos = process_vision_info([messages])
        # 构建模型输入张量
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        # 将输入移动到 CUDA 设备
        inputs = inputs.to('cuda')

        # 调用模型生成
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        # 去除输入部分的 token，只保留新生成内容
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        # 解码为文本结果
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        # 可选：仅提取最后一个 \\boxed{} 内容
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            # 通过计数器匹配成对的大括号
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            # 截取有效范围作为最终结果
            if end is not None:
                response = resp[:end]

        # 调试模式打印最终响应
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        # 返回生成文本
        return response
