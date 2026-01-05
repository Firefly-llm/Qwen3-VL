import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    """
    类功能：
        用于配置和管理模型相关的参数，包括预训练模型路径及不同组件的微调开关。

    继承关系：
        无显式继承关系（Python 3.7+ @dataclass 装饰器自动生成的类）。

    应用场景：
        1. 在微调脚本启动时，解析命令行参数以确定加载哪个模型。
        2. 控制模型不同部分（LLM、MLP 投影层、Vision Encoder）的训练状态（冻结或微调）。

    使用示例：
        >>> model_args = ModelArguments(
        ...     model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        ...     tune_mm_llm=True,
        ...     tune_mm_vision=False
        ... )

    数据属性：
        model_name_or_path: Optional[str]
            预训练模型的名称或本地路径。
            默认为 "Qwen/Qwen2.5-VL-3B-Instruct"。
            这是加载模型权重的入口。

        tune_mm_llm: bool
            是否微调大语言模型（LLM）的主干部分。
            默认为 False。
            如果为 True，则 LLM 部分参与梯度更新；否则冻结。

        tune_mm_mlp: bool
            是否微调多模态投影层（MLP/Projector）。
            默认为 False。
            通常在视觉-语言对齐训练阶段设置为 True。

        tune_mm_vision: bool
            是否微调视觉编码器（Vision Encoder/Vision Tower）。
            默认为 False。
            如果需要适应特定领域的图像特征，可设置为 True。
    """
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    """
    类功能：
        用于配置数据处理、加载及增强相关的参数，特别是针对多模态（图像、视频）数据的约束。

    继承关系：
        无显式继承关系。

    应用场景：
        1. 数据预处理阶段，控制图像的分辨率范围和视频的帧数采样。
        2. 设置数据集的加载方式（如是否使用 Flatten 模式或 Packing 策略）。

    使用示例：
        >>> data_args = DataArguments(
        ...     dataset_use="coco_caption",
        ...     max_pixels=1024*1024,
        ...     video_max_frames=16
        ... )

    数据属性：
        dataset_use: str
            指定使用的数据集名称或标识符。
            默认为空字符串。

        data_flatten: bool
            是否将多模态数据展平（Flatten）处理。
            默认为 False。
            通常用于特定的 Attention 计算优化或数据对齐格式。

        data_packing: bool
            是否启用数据打包（Sequence Packing）策略。
            默认为 False。
            启用后可将多个短序列打包成一个长序列以提高训练效率。

        base_interval: int
            基础采样间隔或相关的时间/空间基准参数。
            默认为 2。

        max_pixels: int
            图像处理允许的最大像素数。
            默认为 28 * 28 * 576 (约 45万像素)。
            超过此分辨率的图像可能会被下采样或裁剪。

        min_pixels: int
            图像处理允许的最小像素数。
            默认为 28 * 28 * 16 (约 1.2万像素)。

        video_max_frames: Optional[int]
            视频数据处理时保留的最大帧数。
            默认为 8。
            超过此帧数的视频会被均匀采样或截断。

        video_min_frames: Optional[int]
            视频数据处理时要求的最小帧数。
            默认为 4。

        video_max_pixels: int
            视频帧处理允许的总最大像素限制（或单帧限制，视具体逻辑而定）。
            默认为 1024 * 28 * 28。

        video_min_pixels: int
            视频帧处理允许的最小像素限制。
            默认为 256 * 28 * 28。

        video_fps: float
            视频处理采样的目标帧率（FPS）。
            默认为 2.0。
    """
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    类功能：
        继承自 HuggingFace 的 TrainingArguments，用于扩展和自定义训练过程中的超参数。
        增加了针对多模态组件的学习率控制及 LoRA（Low-Rank Adaptation）微调配置。

    继承关系：
        继承自 `transformers.TrainingArguments`。

    应用场景：
        1. 传递给 HuggingFace Trainer 以控制训练流程（如 batch size, learning rate 等）。
        2. 为不同的模型组件（如 Projector, Vision Tower）设置差异化的学习率。
        3. 配置 LoRA 适配器参数以进行高效微调。

    使用示例：
        >>> training_args = TrainingArguments(
        ...     output_dir="./output",
        ...     learning_rate=2e-5,
        ...     mm_projector_lr=1e-4,
        ...     lora_enable=True,
        ...     lora_r=128
        ... )

    数据属性：
        cache_dir: Optional[str]
            模型缓存目录。
            默认为 None。

        optim: str
            使用的优化器类型。
            默认为 "adamw_torch"。

        model_max_length: int
            模型的最大序列长度。
            默认为 512。
            序列将被右侧填充（Padding）或截断（Truncation）。

        mm_projector_lr: Optional[float]
            多模态投影层（Projector）的特定学习率。
            默认为 None。
            如果设置，Projector 将使用此学习率而非全局学习率。

        vision_tower_lr: Optional[float]
            视觉编码器（Vision Tower）的特定学习率。
            默认为 None。
            如果设置，Vision Tower 将使用此学习率。

        lora_enable: bool
            是否启用 LoRA 微调。
            默认为 False。

        lora_r: int
            LoRA 的秩（Rank），决定了适配器的参数量。
            默认为 64。

        lora_alpha: int
            LoRA 的缩放系数（Scaling Factor）。
            默认为 128。

        lora_dropout: float
            LoRA 层的 Dropout 概率。
            默认为 0.0。
    """
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
