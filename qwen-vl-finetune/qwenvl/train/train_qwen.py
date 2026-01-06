# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    功能：
        安全地保存 Hugging Face Trainer 的模型状态。
        该函数兼容 DeepSpeed 分布式训练环境和普通训练环境：
        1. 在 DeepSpeed 模式下，同步 CUDA 流并调用 Trainer 原生保存方法。
        2. 在非 DeepSpeed 模式下，显式将模型参数移动到 CPU 后再保存，防止显存溢出，并仅在主进程保存。

    参数：
        trainer (transformers.Trainer): Hugging Face 的 Trainer 实例，包含待保存的模型对象和训练配置参数。
        output_dir (str): 模型权重和配置文件保存的目标目录路径。

    返回：
        None: 该函数没有返回值。

    示例：
        >>> trainer = Trainer(model=model, args=training_args, ...)
        >>> safe_save_model_for_hf_trainer(trainer, output_dir="./output/checkpoint-final")
    """
    # 1> 检查是否使用了 DeepSpeed 分布式训练框架
    if trainer.deepspeed:
        # 同步所有 GPU 设备的 CUDA 流，确保训练计算已完成
        torch.cuda.synchronize()
        # 使用 DeepSpeed 优化的保存方法（它会自动处理 ZeRO 分片参数的收集）
        trainer.save_model(output_dir)
        return

    # 2> 获取模型的当前状态字典（参数权重）
    state_dict = trainer.model.state_dict()

    # 3> 检查当前进程是否具备保存权限（通常仅 Rank 0 进程执行保存）
    if trainer.args.should_save:
        # 4> 将所有参数权重移动到 CPU 内存
        # 这一步是为了防止在保存大模型时占用过多显存导致 OOM (Out Of Memory)
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        
        # 删除原始状态字典引用，辅助垃圾回收释放显存
        del state_dict
        
        # 5> 调用 Trainer 内部的 _save 方法将 CPU 上的状态字典写入磁盘
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """
    功能：
        根据参数配置，动态设置模型各部分的梯度计算状态（冻结或微调）。
        该函数控制三个主要模块的训练状态：Vision Tower（视觉编码器）、Merger（模态对齐层）和 LLM（语言模型）。

    参数：
        model_args (ModelArguments): 模型配置参数对象，包含以下关键开关：
            - tune_mm_vision: 是否微调 Vision Tower。
            - tune_mm_mlp: 是否微调 Projector/Merger 层。
            - tune_mm_llm: 是否微调语言模型主体。
        model (transformers.PreTrainedModel): 待配置的 Qwen-VL 系列模型实例。

    返回：
        None: 该函数直接修改 model 实例的 requires_grad 属性，无返回值。

    示例：
        >>> model = Qwen2VLForConditionalGeneration.from_pretrained(...)
        >>> model_args = ModelArguments(tune_mm_vision=True, tune_mm_llm=False, ...)
        >>> set_model(model_args, model)
    """
    # 1> 配置 Vision Tower (视觉编码器) 的训练状态
    if model_args.tune_mm_vision:
        # 如果开启微调，遍历 Vision Tower 所有参数并设为可训练
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        # 否则，冻结 Vision Tower 所有参数（通常作为预训练特征提取器）
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # 2> 配置 Merger/Projector (模态对齐层) 的训练状态
    # 注意：在 Qwen-VL 架构中，连接 Vision 和 LLM 的层通常称为 merger 或 projector
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # 3> 配置 LLM (语言模型主体) 的训练状态
    if model_args.tune_mm_llm:
        # 如果开启微调，解冻语言模型的所有层
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        # 显式确保 LM Head (输出层) 可训练
        model.lm_head.requires_grad = True
    else:
        # 否则，冻结语言模型（例如在进行 LoRA 微调或仅训练 Projector 时）
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    """
    功能：
        模型微调的主入口函数，负责协调完整的训练流程。
        该函数执行以下核心任务：
        1. 解析命令行参数（模型、数据、训练配置）。
        2. 自动根据模型路径加载对应的 Qwen-VL 系列模型（支持 Qwen2-VL, Qwen2.5-VL, Qwen3-VL 及其 MoE 版本）。
        3. 初始化数据处理器 (Processor) 和分词器 (Tokenizer)。
        4. 根据配置应用优化策略（如 Flash Attention 猴子补丁、梯度检查点）。
        5. 配置参数微调策略（全量微调、部分冻结或 LoRA 微调）。
        6. 构建数据模块并初始化 Hugging Face Trainer。
        7. 执行训练循环（支持从断点恢复）并保存最终模型。

    参数：
        attn_implementation (str, optional): 注意力机制的实现方式，默认为 "flash_attention_2"。
                                             可选项通常包括 "eager", "sdpa", "flash_attention_2"。

    返回：
        None: 函数执行完毕后直接退出，训练产物保存至磁盘。

    示例：
        # 在命令行中调用（通常通过 launch 脚本）：
        # python train_qwen.py --model_name_or_path Qwen/Qwen2-VL-7B ...

        # 在代码中直接调用：
        >>> if __name__ == "__main__":
        ...     train(attn_implementation="flash_attention_2")
    """
    global local_rank

    # 1> 解析命令行参数
    # 将参数解析为 ModelArguments, DataArguments, TrainingArguments 三个数据类
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2> 初始化环境配置
    # 设置本地 rank（用于分布式训练）并创建输出目录
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 3> 加载预训练模型
    # 根据模型名称中的关键字自动选择对应的模型类 (Qwen3-VL MoE / Qwen3-VL / Qwen2.5-VL / Qwen2-VL)
    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        # 加载 Qwen3-VL MoE (Mixture of Experts) 版本
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        # 加载 Qwen3-VL 标准版本
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        # 加载 Qwen2.5-VL 版本
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        # 默认回退到 Qwen2-VL 版本
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    
    # 4> 初始化数据处理器 (Processor)
    # 用于处理图像和文本输入
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    # 5> 应用 Flash Attention 优化
    # 如果开启了数据扁平化 (flatten) 或打包 (packing) 策略，需要替换 Attention 实现以支持变长序列
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    
    # 训练期间禁用 KV Cache（生成时才需要）
    model.config.use_cache = False

    # 6> 配置梯度检查点 (Gradient Checkpointing)
    # 通过以计算换内存的方式降低显存占用
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # 兼容性处理：如果模型没有原生支持，注册 hook 强制输入需要梯度
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 7> 初始化分词器 (Tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 8> 配置 LoRA 或全量微调
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        # 冻结所有原有参数
        for p in model.parameters():
            p.requires_grad = False

        # 配置 LoRA 参数（秩、Alpha、Dropout、目标层）
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对 Attention 的线性层应用 LoRA
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # 将模型包装为 PeftModel
        model = get_peft_model(model, lora_config)
    else:
        # 如果不是 LoRA，则根据 arguments 配置部分或全量微调
        set_model(model_args, model)

        # 在主进程打印可训练参数的统计信息
        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    # 9> 准备数据模块
    data_module = make_supervised_data_module(processor, data_args=data_args)
    
    # 10> 初始化 Hugging Face Trainer
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    # 11> 执行训练
    # 检查是否存在之前的 checkpoint，如果有则恢复训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # 保存最终训练状态
    trainer.save_state()

    # 12> 恢复模型配置并保存
    # 重新启用 KV Cache（为了保存后的推理）
    model.config.use_cache = True

    # 使用自定义的安全保存函数保存模型权重
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # 保存处理器配置
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
