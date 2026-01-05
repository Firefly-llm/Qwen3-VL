from typing import Dict, List, Optional, Sequence, Tuple, Callable

from fsspec.caching import P

import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    功能：
        包装 Flash Attention 的 varlen 函数执行高效的自注意力计算。
        该函数处理输入张量的转置、维度压缩以及精度转换，以适配 flash_attn_varlen_func 的接口要求。
        特别适用于处理经过 Packing（打包）后的变长序列数据。

    参数：
        module (torch.nn.Module): 
            调用该函数的模型模块，主要用于访问配置（如 quantization 配置）以确定目标数据类型。
        query (torch.Tensor): 
            查询张量，初始形状为 (Batch_Size, Num_Heads, Seq_Len, Head_Dim)。
        key (torch.Tensor): 
            键张量，初始形状为 (Batch_Size, Num_Heads, Seq_Len, Head_Dim)。
        value (torch.Tensor): 
            值张量 ，初始形状为 (Batch_Size, Num_Heads, Seq_Len, Head_Dim)。
        attention_mask (Optional[torch.Tensor]): 
            在此处作为 cu_seqlens (Cumulative Sequence Lengths) 使用。
            形状为 (Total_Sequences + 1,)，指示了 packed 序列中每个子序列的起始和结束位置。
        dropout (float, optional): 
            注意力机制的 Dropout 概率。默认为 0.0。
        scaling (Optional[float], optional): 
            缩放因子。如果为 None，默认使用 1 / sqrt(head_dim)。
        sliding_window (Optional[int], optional): 
            滑动窗口大小。在此 varlen 实现中通常不直接使用或由内部处理。
        softcap (Optional[float], optional): 
            Softcap 值（用于限制 logits 的范围）。默认为 None。
        **kwargs: 
            其他可选参数（如 output_attentions 等）。

    返回：
        tuple[torch.Tensor, None]: 
            - attn_output: 注意力输出张量，形状恢复为 (Batch_Size, Seq_Len, Hidden_Dim)。
            - None: Flash Attention 通常不返回注意力权重矩阵。

    示例：
        >>> # 假设 batch_size=1 (packed 模式), num_heads=16, seq_len=20, head_dim=64
        >>> q = torch.randn(1, 16, 20, 64, device='cuda', dtype=torch.bfloat16)
        >>> k = torch.randn(1, 16, 20, 64, device='cuda', dtype=torch.bfloat16)
        >>> v = torch.randn(1, 16, 20, 64, device='cuda', dtype=torch.bfloat16)
        >>> # 定义两个序列，长度分别为 10, 10
        >>> cu_seqlens = torch.tensor([0, 10, 20], device='cuda', dtype=torch.int32)
        >>> output, _ = flash_attention_forward(model, q, k, v, cu_seqlens)
    """
    # 1> 检查不支持的参数配置
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    
    # 2> 获取序列长度（转置前，维度 2 是 Seq_Len）
    # input shape: (Batch, Head, Seq_Len, Dim)
    seq_len = query.shape[2]

    # 3> 检查输入张量是否存在 0 维度（无效形状）
    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    
    # 4> 转置输入张量以适配 Flash Attention 格式
    # input: (Batch, Num_Heads, Seq_Len, Head_Dim) -> output: (Batch, Seq_Len, Num_Heads, Head_Dim)
    # Flash Attention varlen 接口期望输入的 Token 在连续维度上
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    # 5> 处理数据类型转换（针对 PEFT/Quantization 的稳定性兼容）
    # 如果 LayerNorm 层被强制转为 fp32（常见于 PEFT），输入可能是 fp32。
    # 为了避免性能下降，这里尝试将其转换回模型原本的精度（如 bf16/fp16）。
    # 注意：下面的 target_dtype 计算逻辑在原始代码中存在，但未显式执行 tensor.to(target_dtype)，保留原样。
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # 6> 压缩 Batch 维度
    # varlen 接口期望输入为 (Total_Tokens, Num_Heads, Head_Dim)
    # 假设 Batch=1（Packed 数据），squeeze(0) 将 (1, Seq_Len, Head, Dim) -> (Seq_Len, Head, Dim)
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)

    # 7> 准备 Cumulative Sequence Lengths (cu_seqlens)
    # attention_mask 在此上下文中被复用为 cu_seqlens，用于指示 packed 序列的边界
    cu_seqlens = attention_mask

    # 8> 计算最大序列长度
    # 用于 Flash Attention 内部优化，遍历 cu_seqlens 计算最长子序列
    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    # 9> 调用 Flash Attention Varlen 函数
    # 执行实际的自注意力计算
    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True, # 启用因果掩码 (Causal Masking)
    )

    # 10> 恢复 Batch 维度
    # (Seq_Len, Head, Dim) -> (1, Seq_Len, Head, Dim)
    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


# 处理参数重命名兼容性：将旧参数名 "past_key_value" 映射为新参数名 "past_key_values"
# 如果用户使用了旧参数名，会发出弃用警告，提示在 4.58 版本后变更
@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """
    功能：
        Qwen2-VL 注意力层的核心前向传播函数。
        该函数整合了 Q、K、V 投影、多模态旋转位置编码 (RoPE)、KV 缓存更新以及高效的 Flash Attention 计算。
        特别优化了多模态数据（视觉+文本）的混合处理流程。

    注意：之所以第一个参数命名为 self，是因为这个函数是被设计用来进行 Monkey Patching（动态替换/猴子补丁） 的。基于动态绑定机制替换类的属性。

    参数：
        self: Qwen2VLAttention
            注意力模块实例，包含 q_proj, k_proj, v_proj 等层。
        hidden_states (torch.Tensor): 
            输入隐藏状态，形状为 (Batch, Seq_Len, Hidden_Dim)。
        attention_mask (Optional[torch.Tensor]): 
            注意力掩码（在 Flash Attention 模式下通常作为 cu_seqlens 使用）。
        position_ids (Optional[torch.LongTensor]): 
            位置 ID，用于计算或检索位置编码。
        past_key_values (Optional[Cache]): 
            用于推理加速的 KV 缓存对象。
        output_attentions (bool): 
            是否返回注意力权重（Flash Attention 模式下通常不支持，需为 False）。
        use_cache (bool): 
            是否使用 KV 缓存。
        cache_position (Optional[torch.LongTensor]): 
            缓存位置索引，用于更新 StaticCache。
        position_embeddings (Optional[tuple]): 
            预计算的位置编码 (cos, sin) 元组。
        **kwargs: 
            其他传递给 Flash Attention 的参数。

    返回：
        tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
            - attn_output: 注意力输出，形状 (Batch, Seq_Len, Hidden_Dim)。
            - attn_weights: 注意力权重（在 Flash Attention 下通常为 None）。
            - past_key_values: 更新后的 KV 缓存（如果 use_cache=True）。

    示例：
        >>> # 假设 hidden_states 形状为 (1, 512, 4096)
        >>> output, _, _ = qwen2vl_forward(
        ...     self=attn_layer,
        ...     hidden_states=hidden_states,
        ...     position_embeddings=(cos, sin),
        ...     attention_mask=cu_seqlens
        ... )
    """
    # 1> 获取输入维度信息
    bsz, q_len, _ = hidden_states.size()

    # 2> 投影计算 Q, K, V
    # 将输入隐藏状态映射到查询、键、值空间
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 3> 重塑张量形状以适配多头注意力机制
    # (Batch, Seq, Hidden) -> (Batch, Seq, Head, Head_Dim) -> (Batch, Head, Seq, Head_Dim)
    # 转置是为了方便后续的处理，尽管 Flash Attention 内部会再次转置
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # 4> 应用多模态旋转位置编码 (RoPE)
    # 使用预计算的 cos/sin 和特定的 mrope 配置对 Q 和 K 进行旋转嵌入
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # 5> 更新 KV 缓存 (如果启用)
    # 用于自回归生成过程中的加速
    if past_key_values is not None:
        # 传递 sin/cos 和缓存位置信息
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 6> 执行 Flash Attention 计算
    # 调用优化的 Flash Attention 前向函数
    # 注意：这里传递了 position_ids，这是 FA2 可能需要的辅助信息
    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    # 7> 重塑输出并进行最终投影
    # 恢复形状: (Batch, Head, Seq, Head_Dim) -> (Batch, Seq, Hidden)
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # 输出线性投影
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights



@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    功能：
        Qwen3-VL 注意力层的核心前向传播函数。
        负责执行查询/键/值的归一化（QK-Norm）和投影、旋转位置编码 (RoPE) 的应用、KV 缓存的维护以及 Flash Attention 的调用。
        相比 Qwen2-VL，它在投影前增加了 QK 归一化层，增强了训练稳定性。

    参数：
        self: Qwen3VLAttention
            注意力模块实例，包含 q_proj, k_proj, v_proj, q_norm, k_norm 等层。
        hidden_states (torch.Tensor): 
            输入隐藏状态，形状通常为 (Batch, Seq_Len, Hidden_Dim)。
        position_embeddings (tuple[torch.Tensor, torch.Tensor]): 
            位置编码元组 (cos, sin)，用于旋转嵌入。
        attention_mask (Optional[torch.Tensor]): 
            注意力掩码（在 Flash Attention 中作为 cu_seqlens 使用）。
        past_key_values (Optional[Cache]): 
            KV 缓存对象，用于推理加速。
        cache_position (Optional[torch.LongTensor]): 
            缓存位置索引。
        **kwargs: 
            其他传递给 Flash Attention 的参数（如 dropout, scaling 等）。

    返回：
        tuple[torch.Tensor, Optional[torch.Tensor]]:
            - attn_output: 注意力输出张量，形状与输入一致。
            - attn_weights: 注意力权重（Flash Attention 模式下通常为 None）。

    示例：
        >>> # 假设输入形状为 (1, 256, 4096)
        >>> output, _ = qwen3vl_forward(
        ...     self=attn_module,
        ...     hidden_states=x,
        ...     position_embeddings=(cos, sin),
        ...     attention_mask=mask
        ... )
    """
    # 1> 获取输入张量的形状信息
    # input_shape: (Batch, Seq_Len)
    input_shape = hidden_states.shape[:-1]
    # hidden_shape: (Batch, Seq_Len, -1, Head_Dim) -> (Batch, Seq_Len, Num_Heads, Head_Dim)
    # 用于 view 操作以分离多头维度
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 2> 投影并进行 QK 归一化 (Qwen3 特性)
    # Qwen3 在 Q 和 K 投影后引入了 RMSNorm (q_norm, k_norm)
    # 流程：Input -> Proj -> View -> Norm -> Transpose
    # 最终形状：(Batch, Num_Heads, Seq_Len, Head_Dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    # V 不需要归一化
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 3> 应用旋转位置编码 (RoPE)
    # 使用传入的 cos/sin 对 Q 和 K 进行旋转变换
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 4> 更新 KV 缓存 (如果启用)
    if past_key_values is not None:
        # sin 和 cos 是 RoPE 模型特有的；cache_position 用于静态缓存索引
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 5> 执行 Flash Attention 计算
    # 调用封装好的 flash_attention_forward 函数
    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # 6> 重塑输出并进行最终投影
    # 恢复形状: (Batch, Num_Heads, Seq_Len, Head_Dim) -> (Batch, Seq_Len, Hidden_Dim)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # 输出层线性投影
    attn_output = self.o_proj(attn_output)
    
    return attn_output, attn_weights


def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    """
    功能：
        辅助函数，用于覆盖 Transformers 默认的掩码（Mask）创建逻辑。
        该函数直接返回传入的 `attention_mask`，而不进行额外的处理（如创建因果掩码或扩展维度）。
        在使用 Flash Attention 的 Packing 模式（VarLen）时，`attention_mask` 已经被处理为 `cu_seqlens`，因此不需要默认的掩码生成逻辑。

    参数：
        config: 
            模型配置对象。
        input_embeds: 
            输入嵌入张量。
        attention_mask: 
            传入的注意力掩码（在此上下文中通常是 `cu_seqlens` 或已处理好的掩码）。
        cache_position: 
            缓存位置索引。
        past_key_values: 
            KV 缓存对象。
        position_ids: 
            位置 ID。
        **kwargs: 
            其他可能传入的参数。

    返回：
        torch.Tensor: 原样返回传入的 `attention_mask`。

    示例：
        >>> # 在替换类方法时使用
        >>> transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = return_mask
    """
    # 1> 直接返回输入的 attention_mask
    # 这个函数的主要目的是绕过 transformers 库中默认复杂的掩码生成逻辑
    # 因为在 Flash Attention Varlen 模式下，mask 处理逻辑由外部（DataCollator）和 FA 内核接管
    return attention_mask


def replace_qwen2_vl_attention_class():
    """
    功能：
        动态替换（Monkey Patch）Qwen 系列模型（Qwen2-VL, Qwen2.5-VL, Qwen3-VL 及其 MoE 变体）的 Attention 类方法。
        将原本的 `forward` 方法替换为支持 Flash Attention Varlen（Packing）模式的自定义实现 (`qwen2vl_forward` 或 `qwen3vl_forward`)。
        同时，将 `create_causal_mask` 和 `create_sliding_window_causal_mask` 方法替换为 `return_mask`，以绕过默认的掩码生成逻辑。

    参数：
        无。

    返回：
        None: 该函数直接修改导入的模块，无返回值。

    示例：
        >>> # 在模型初始化之前调用
        >>> replace_qwen2_vl_attention_class()
        >>> model = Qwen2VLForConditionalGeneration.from_pretrained(...)
    """
    import transformers
    import transformers.modeling_flash_attention_utils

    # 1> 替换 Qwen2-VL 系列的 Attention 方法
    # 替换 forward 方法为自定义的 qwen2vl_forward，以支持 Flash Attention Packing
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = (
        qwen2vl_forward
    )
    # 替换 Mask 生成逻辑，直接返回 cu_seqlens
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = (
        return_mask
    )    
    
    # 2> 替换 Qwen2.5-VL 系列的 Attention 方法
    # Qwen2.5-VL 使用与 Qwen2-VL 相同的 Attention 结构
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = (
        return_mask
    )

    # 3> 替换 Qwen3-VL 系列的 Attention 方法
    # Qwen3-VL 引入了 QK-Norm，因此使用 qwen3vl_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = (
        return_mask
    )
    
    # 4> 替换 Qwen3-VL MoE 系列的 Attention 方法
    # MoE 版本同样使用 qwen3vl_forward
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = (
        return_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    功能：
        打印并检查视觉模型组件（Vision Tower）的可训练状态。
        该函数遍历视觉模型的各个模块（Attention Blocks 和 Merger Module），输出哪些部分参与梯度更新（Trainable），哪些部分被冻结（Non-Trainable）。
        主要用于调试和确认微调策略（如是否仅微调部分层）是否生效。

    参数：
        self: Qwen2VisionTransformerPretrainedModel 或其子类实例
            视觉模型对象，包含 `blocks`（注意力层列表）和 `merger`（特征融合层）等属性。

    返回：
        None: 该函数直接将检查结果打印到标准输出，无返回值。

    示例：
        >>> # 假设 visual_model 是一个 Qwen2-VL 的视觉编码器实例
        >>> print_trainable_parameters_visual(visual_model)
        Vision Module - Attention Blocks:
        Trainable Block Indices: [0, 1, 2, ..., 31]
        Non-Trainable Block Indices: None
        Merger Module Trainable: True
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # 1> 检查每个 Vision Attention Block 的可训练状态
    # 遍历视觉模型的所有块（通常是 Transformer Encoder Layers）
    for block_idx, block in enumerate(self.blocks):
        # 只要该块中有一个参数需要梯度，则认为该块是可训练的
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # 2> 检查 Merger Module（特征融合/投影层）的可训练状态
    # Merger 通常负责将视觉特征维度映射到 LLM 的维度
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # 3> 打印统计结果
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    功能：
        打印并检查大语言模型（LLM）部分的可训练状态。
        该函数遍历 LLM 的核心组件（Embedding 层和 Decoder Layers），输出哪些部分参与梯度更新，哪些部分被冻结。
        与 `print_trainable_parameters_visual` 配合使用，可全面了解多模态模型的训练配置。

    参数：
        self: Qwen2VLModel 或其子类实例
            LLM 模型对象，包含 `language_model` 属性（内部结构通常包含 `embed_tokens` 和 `layers`）。

    返回：
        None: 该函数直接将检查结果打印到标准输出，无返回值。

    示例：
        >>> # 假设 model 是一个完整的多模态模型实例
        >>> print_trainable_parameters(model)
        LLM Module - Embed Tokens Trainable: False
        LLM Module - Trainable Layer Indices: [28, 29, 30, 31]
        LLM Module - Non-Trainable Layer Indices: [0, 1, ..., 27]
    """
    # 1> 检查词嵌入层 (Embed Tokens) 的可训练状态
    # 通常在微调过程中，为了节省显存，词嵌入层会被冻结
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # 2> 检查每个 Decoder Layer 的可训练状态
    trainable_layers = []
    non_trainable_layers = []

    # 遍历 LLM 的每一层
    for layer_idx, layer in enumerate(self.language_model.layers):
        # 只要该层中有一个参数需要梯度，则认为该层是可训练的
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # 3> 打印统计结果
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):
    """
    功能：
        创建并初始化优化器（Optimizer）。
        该函数覆盖了 HuggingFace Trainer 的默认实现，支持为模型不同的组件（如 Vision Tower, Projector, LLM Backbone）设置不同的学习率。
        这对于多模态模型的微调至关重要，因为通常视觉编码器需要较小的学习率（或冻结），而 Projector 可能需要较大的学习率。

    参数：
        self: Trainer
            Trainer 实例，包含 `args`（训练参数）、`model`（模型）等属性。

    返回：
        torch.optim.Optimizer: 初始化后的 PyTorch 优化器实例。

    示例：
        >>> # 这是一个 Trainer 内部调用的方法，通常不需要手动调用。
        >>> # 在 TrainingArguments 中设置：
        >>> # mm_projector_lr = 1e-4
        >>> # vision_tower_lr = 2e-6
        >>> # learning_rate = 2e-5 (LLM default)
        >>> trainer.create_optimizer()
    """
    opt_model = self.model

    if self.optimizer is None:
        # 1> 获取需要应用权重衰减（Weight Decay）的参数名称列表
        # 通常排除 Bias 和 LayerNorm 层
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        # 2> 检查是否配置了 Projector 的特定学习率
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            # 筛选出 Projector (Merger) 相关的参数
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            
            # 3> 进一步检查是否配置了 Vision Tower 的特定学习率
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                # 筛选出 Vision Tower 相关的参数
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                
                # 3.1> 构建参数组 (6组): LLM(Decay/NoDecay) + Vision(Decay/NoDecay) + Projector(Decay/NoDecay)
                optimizer_grouped_parameters = [
                    # Group 1: LLM 主干部分 - 应用 Weight Decay
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    # Group 2: Vision Tower - 应用 Weight Decay，使用 vision_tower_lr
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    # Group 3: LLM 主干部分 - 不应用 Weight Decay (如 Bias, LayerNorm)
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    # Group 4: Vision Tower - 不应用 Weight Decay，使用 vision_tower_lr
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    # Group 5: Projector - 应用 Weight Decay，使用 mm_projector_lr
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    # Group 6: Projector - 不应用 Weight Decay，使用 mm_projector_lr
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                # 3.2> 仅配置了 Projector LR (Vision Tower 使用全局 LR 或冻结)
                # 构建参数组 (4组): 其他所有(Decay/NoDecay) + Projector(Decay/NoDecay)
                optimizer_grouped_parameters = [
                    # Group 1: 非 Projector 部分 (LLM + Vision) - 应用 Weight Decay
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    # Group 2: 非 Projector 部分 - 不应用 Weight Decay
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    # Group 3: Projector - 应用 Weight Decay，使用 mm_projector_lr
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    # Group 4: Projector - 不应用 Weight Decay，使用 mm_projector_lr
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            # 4> 默认情况：不区分组件，仅根据 Weight Decay 分组
            # 适用于标准的 LLM 微调或所有组件使用相同学习率
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        # 5> 实例化优化器
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# Apply monkey patches
Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen3VLVisionModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters