import torch
from typing import Dict, Optional, Sequence, List, Tuple


def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：
        为 Qwen3-VL 模型生成 3D 旋转位置编码 (RoPE) 索引。
        该函数专门处理多模态输入（图像、视频、文本），通过分别计算 Temporal（时间）、Height（高度）、Width（宽度）三个维度的位置索引，
        以适应模型对时空信息的编码需求。与 Qwen2-VL 不同，Qwen3-VL 使用时间戳 (timestamps) 而非绝对时间位置 ID 来区分视频帧。

    参数：
        spatial_merge_size (int, optional): 
            空间合并大小，用于降低视觉特征的分辨率。默认为 2。
            例如：H x W 的网格会被处理为 (H//merge) x (W//merge)。
        input_ids (torch.LongTensor, optional): 
            输入 Token 序列，形状为 (Batch_Size, Seq_Len)。
            包含了文本和特殊的视觉占位符 Token。
        image_grid_thw (torch.LongTensor, optional): 
            图像的 Temporal-Height-Width 网格信息，形状为 (Num_Images, 3)。
            每一行 [t, h, w] 描述了一张图片在特征图中的维度。
        video_grid_thw (torch.LongTensor, optional): 
            视频的 Temporal-Height-Width 网格信息，形状为 (Num_Videos, 3)。
        second_per_grid_ts (torch.Tensor, optional): 
            (Qwen3-VL 中未使用) 每个时间网格对应的秒数。
        attention_mask (torch.Tensor, optional): 
            注意力掩码，形状为 (Batch_Size, Seq_Len)。
            用于区分有效 Token 和 Padding Token。

    返回：
        Tuple[torch.Tensor, torch.Tensor]:
            - position_ids: 生成的 3D 位置索引，形状为 (3, Batch_Size, Seq_Len)。
              第 0 维对应 Temporal，第 1 维对应 Height，第 2 维对应 Width。
            - mrope_position_deltas: 多模态 RoPE 的位置增量，用于后续计算，形状为 (Batch_Size, 1)。

    示例：
        >>> # 假设有一个包含一张图片的输入
        >>> input_ids = torch.tensor([[101, 151652, 151655, 102]]) # [CLS, <vision_start>, <image>, SEP]
        >>> image_grid = torch.tensor([[1, 14, 14]]) # T=1, H=14, W=14
        >>> pos_ids, deltas = get_rope_index_3(
        ...     input_ids=input_ids,
        ...     image_grid_thw=image_grid
        ... )
        >>> print(pos_ids.shape)
        torch.Size([3, 1, 4])
    """

    # 1> 预处理视频网格信息
    # Qwen3-VL 使用时间戳分隔视频，因此需要将 video_grid_thw 按照时间维度 (T) 拆分
    # 例如：如果一个视频有 3 帧 (T=3)，则将其拆分为 3 个 T=1 的独立网格
    if video_grid_thw is not None:
        # --- torch.repeat_interleave 方法说明 ---
        # 功能：沿指定维度重复张量的元素，每个元素重复的次数可以不同（由 repeats 参数控制）
        # 函数原型：torch.repeat_interleave(input, repeats, dim=None, output_size=None)
        #   - input: 输入张量
        #   - repeats: 每个元素的重复次数，可以是标量（所有元素重复相同次数）或张量（每个元素重复不同次数）
        #   - dim: 沿哪个维度重复（如果为 None，则先展平张量）
        # 示例：
        #   >>> x = torch.tensor([[1, 2], [3, 4]])
        #   >>> repeats = torch.tensor([2, 3])  # 第一行重复2次，第二行重复3次
        #   >>> result = torch.repeat_interleave(x, repeats, dim=0)
        #   >>> print(result)
        #   tensor([[1, 2],
        #           [1, 2],  # 第一行重复了2次
        #           [3, 4],
        #           [3, 4],
        #           [3, 4]]) # 第二行重复了3次
        
        # --- 当前代码的具体执行过程举例 ---
        # 假设输入的 video_grid_thw 为：
        #   tensor([[3, 14, 14],  # 第一个视频：3帧，每帧14x14网格
        #           [2, 28, 28]]) # 第二个视频：2帧，每帧28x28网格
        #
        # 执行过程：
        # 步骤 1: 提取重复次数 video_grid_thw[:, 0] = tensor([3, 2])
        #         这表示：第一个视频要重复3次，第二个视频要重复2次
        #
        # 步骤 2: 执行 torch.repeat_interleave(video_grid_thw, [3, 2], dim=0)
        #         沿着 dim=0 (行维度) 重复，结果为：
        #         tensor([[3, 14, 14],  # 第1个视频的第1份副本
        #                 [3, 14, 14],  # 第1个视频的第2份副本
        #                 [3, 14, 14],  # 第1个视频的第3份副本
        #                 [2, 28, 28],  # 第2个视频的第1份副本
        #                 [2, 28, 28]]) # 第2个视频的第2份副本
        #         形状从 (2, 3) 变为 (5, 3)，即 3+2=5 行
        #
        # 步骤 3: 将第一列（时间维度 T）全部设置为 1
        #         video_grid_thw[:, 0] = 1 执行后：
        #         tensor([[1, 14, 14],  # T 从 3 改为 1
        #                 [1, 14, 14],  # T 从 3 改为 1
        #                 [1, 14, 14],  # T 从 3 改为 1
        #                 [1, 28, 28],  # T 从 2 改为 1
        #                 [1, 28, 28]]) # T 从 2 改为 1
        #
        # 最终效果：原本一个 T=3 的视频网格被拆分为 3 个 T=1 的独立时间片段网格。
        #          这符合 Qwen3-VL 使用时间戳 (timestamps) 而非绝对时间位置的设计理念。
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    
    # 2> 定义特殊 Token ID
    image_token_id = 151655        # <image>
    video_token_id = 151656        # <video>
    vision_start_token_id = 151652 # <vision_start>
    
    mrope_position_deltas = []

    # 3> 检查是否需要进行多模态位置编码计算
    # 只有当 input_ids 存在且包含图像或视频网格信息时才进行复杂计算
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        
        # 初始化 attention_mask，如果未提供则默认为全 1 (即所有 Token 都有效)
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
            
        # 初始化 position_ids 张量，形状为 (3, Batch, Seq)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        # 4> 逐样本遍历 Batch
        for i, input_ids in enumerate(total_input_ids):
            # 过滤掉 Padding Token，只处理有效部分
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            
            # 5> 统计当前样本中的图像和视频数量
            # 查找 <vision_start> 标记的位置
            
            # --- torch.argwhere 方法说明 ---
            # 功能：返回张量中所有非零元素（或满足条件的元素）的索引位置
            # 函数原型：torch.argwhere(input) -> LongTensor
            #   - input: 输入张量（通常是布尔张量，即条件判断的结果）
            #   - 返回值: 一个二维张量，每一行是一个满足条件的元素的完整索引 (多维坐标)
            # 
            # 示例：
            #   >>> x = torch.tensor([10, 20, 30, 20, 10])
            #   >>> indices = torch.argwhere(x == 20)
            #   >>> print(indices)
            #   tensor([[1],  # 第1个位置的值是20
            #           [3]]) # 第3个位置的值也是20
            #   >>> # 对于多维张量
            #   >>> y = torch.tensor([[1, 0, 3], [0, 5, 0]])
            #   >>> indices = torch.argwhere(y > 2)
            #   >>> print(indices)
            #   tensor([[0, 2],  # 位置 [0, 2] 的值是 3
            #           [1, 1]]) # 位置 [1, 1] 的值是 5
            
            # --- 当前代码的具体执行过程举例 ---
            # 注意：此时的 input_ids 已经被过滤过（第142行：input_ids = input_ids[attention_mask[i] == 1]）
            # 因此这里处理的是单个样本的一维序列
            #
            # 假设原始 total_input_ids 的形状为 (batch_size=2, seq_len=7)：
            #   tensor([[101, 151652, 151655, 102, 151652, 151656, 103],  # Batch 0
            #           [101, 151652, 151655, 200, 300, 400, 102]])        # Batch 1
            #
            # 当前循环处理第 i=0 个样本（即 Batch 0），经过 attention_mask 过滤后，
            # input_ids 变为一维张量：
            #   tensor([101, 151652, 151655, 102, 151652, 151656, 103])
            #   解释：[CLS, <vision_start>, <image>, SEP, <vision_start>, <video>, END]
            #   其中 vision_start_token_id = 151652
            #
            # 执行过程：
            # 步骤 1: 执行条件判断 input_ids == vision_start_token_id
            #         生成布尔张量：
            #         tensor([False, True, False, False, True, False, False])
            #         表示索引 1 和 4 的位置是 <vision_start>
            #
            # 步骤 2: 执行 torch.argwhere()
            #         找到所有 True 的位置索引：
            #         tensor([[1],  # 第一个 <vision_start> 在索引 1
            #                 [4]]) # 第二个 <vision_start> 在索引 4
            #         注意：返回的是 2D 张量，形状为 (2, 1)
            #
            # 步骤 3: 执行 .squeeze(1)
            #         压缩第 1 维度（列维度），将 2D 张量变为 1D：
            #         tensor([1, 4])
            #         形状从 (2, 1) 变为 (2,)
            #
            # 最终结果：vision_start_indices = tensor([1, 4])
            #          这个一维张量存储了当前样本中所有 <vision_start> Token 在序列中的位置
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            # 获取 <vision_start> 后面紧跟的 Token，判断是 image 还是 video
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0 # 当前处理的起始位置索引
            remain_images, remain_videos = image_nums, video_nums

            # 6> 遍历处理所有视觉片段 (Image/Video)
            # 按在序列中出现的顺序依次处理
            for _ in range(image_nums + video_nums):
                # 寻找下一个 <image> 和 <video> 的位置
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1 # 设置为无穷大
                
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1 # 设置为无穷大

                # 7> 判断当前遇到的是图片还是视频，并获取对应的 THW 网格信息
                if ed_image < ed_video: # 当前是图片
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image # 结束位置更新为图片位置

                else: # 当前是视频
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video # 结束位置更新为视频位置
                
                # 计算在 LLM 特征图中的实际尺寸 (考虑空间合并)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                
                # 8> 处理视觉片段前的文本部分
                # text_len: 当前视觉片段之前的文本长度
                text_len = ed - st
                
                # 计算起始索引 st_idx，基于前一个片段的最大位置 ID 累加
                # --- 当前代码的具体执行过程举例 ---
                # st_idx 的作用：确保位置 ID 连续递增，不重复
                #
                # 假设在处理一个序列时，llm_pos_ids_list 逐步增长：
                #
                # 第 1 次循环（处理第一段文本，长度为 2）：
                #   - len(llm_pos_ids_list) = 0 (列表为空)
                #   - st_idx = 0 (因为列表为空，使用默认值 0)
                #   - 生成位置 ID: torch.arange(2) + 0 = tensor([0, 1])
                #   - llm_pos_ids_list = [tensor([[0, 1], [0, 1], [0, 1]])]  # shape: (3, 2)
                #
                # 第 2 次循环（处理第一个图像，网格大小 2x2=4）：
                #   - len(llm_pos_ids_list) = 1
                #   - llm_pos_ids_list[-1].max() = 1 (上一个片段的最大 ID)
                #   - st_idx = 1 + 1 = 2
                #   - 生成视觉位置 ID: 
                #     t_index = [0, 0, 0, 0] + 2 = [2, 2, 2, 2]
                #     h_index = [0, 0, 1, 1] + 2 = [2, 2, 3, 3]
                #     w_index = [0, 1, 0, 1] + 2 = [2, 3, 2, 3]
                #   - llm_pos_ids_list 增加新元素，现在有 2 个元素
                #
                # 第 3 次循环（处理第二段文本，长度为 3）：
                #   - len(llm_pos_ids_list) = 2
                #   - llm_pos_ids_list[-1].max() = 3 (上一个视觉片段的最大 ID)
                #   - st_idx = 3 + 1 = 4
                #   - 生成位置 ID: torch.arange(3) + 4 = tensor([4, 5, 6])
                #   - llm_pos_ids_list 继续增长
                #
                # 关键点：st_idx 确保每个新片段的位置 ID 从上一个片段的最大值 +1 开始，
                #        从而保证整个序列的位置编码是连续的、不重叠的
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                
                # 为文本生成位置 ID：
                # 文本是 1D 的，所以在 T, H, W 三个维度上位置 ID 相同，且线性增长
                # shape: (3, text_len)
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 9> 为视觉片段生成 3D 位置 ID
                # 这一步是 RoPE 3D 的核心：分别为 T, H, W 维度生成网格索引
                
                # t_index: 时间维度索引。由于 Qwen3 使用 timestamp，这里通常全为 0 (llm_grid_t=1)
                # 假设网格尺寸为：llm_grid_t=2, llm_grid_h=2, llm_grid_w=3
                # 目标：为 2×2×3=12 个视觉 Token 生成时间维度的位置索引
                #
                # 执行过程：
                # 步骤 1: torch.arange(llm_grid_t) 生成时间索引序列
                #         torch.arange(2) = tensor([0, 1])
                #         shape: (2,)
                #
                # 步骤 2: .view(-1, 1) 变为列向量
                #         tensor([[0],
                #                 [1]])
                #         shape: (2, 1)
                #
                # 步骤 3: .expand(-1, llm_grid_h * llm_grid_w) 
                #         沿第2维度复制，每个时间索引重复 h×w 次
                #         expand(-1, 2*3) = expand(-1, 6)
                #         tensor([[0, 0, 0, 0, 0, 0],  # 时间片 0 重复 6 次
                #                 [1, 1, 1, 1, 1, 1]]) # 时间片 1 重复 6 次
                #         shape: (2, 6)
                #
                # 步骤 4: .flatten() 展平为一维向量
                #         tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
                #         shape: (12,)
                #
                # 最终结果：t_index 为所有 12 个 Token 分配时间维度的位置编码
                #          前 6 个 Token 属于时间片 0，后 6 个属于时间片 1
                #
                # 注意：在 Qwen3-VL 中，由于使用时间戳机制，llm_grid_t 通常为 1，
                #      因此 t_index 通常全为 0: tensor([0, 0, 0, 0, 0, 0])
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                
                # h_index: 高度维度索引
                # view(1, -1, 1) -> expand -> flatten 构造出类似 [0,0,.., 1,1,..] 的网格
                # 继续使用相同网格尺寸：llm_grid_t=2, llm_grid_h=2, llm_grid_w=3
                # 目标：为 2×2×3=12 个视觉 Token 生成高度维度的位置索引
                #
                # 执行过程：
                # 步骤 1: torch.arange(llm_grid_h) 生成高度索引序列
                #         torch.arange(2) = tensor([0, 1])
                #         shape: (2,)
                #
                # 步骤 2: .view(1, -1, 1) 变为 3D 张量
                #         tensor([[[0],
                #                  [1]]])
                #         shape: (1, 2, 1)
                #         解释：第0维=时间(1), 第1维=高度(2), 第2维=宽度(1)
                #
                # 步骤 3: .expand(llm_grid_t, -1, llm_grid_w)
                #         沿时间维度和宽度维度扩展
                #         expand(2, -1, 3) 表示：时间扩展到2，高度保持2，宽度扩展到3
                #         tensor([[[0, 0, 0],  # 时间片0: 高度0重复3次(对应3个宽度)
                #                  [1, 1, 1]], # 时间片0: 高度1重复3次
                #                 [[0, 0, 0],  # 时间片1: 高度0重复3次
                #                  [1, 1, 1]]])# 时间片1: 高度1重复3次
                #         shape: (2, 2, 3)
                #
                # 步骤 4: .flatten() 展平为一维向量
                #         按照行优先顺序（时间->高度->宽度）展平
                #         tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
                #         shape: (12,)
                #         解释：前6个元素是时间片0(0,0,0,1,1,1)，后6个是时间片1
                #
                # 最终结果：h_index 为所有 12 个 Token 分配高度位置编码
                #          每连续3个Token共享相同的高度索引（因为宽度方向有3个位置）
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()

                # w_index: 宽度维度索引
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                
                # 将三个维度的索引堆叠，并加上之前的长度偏移
                # shape: (3, grid_t * grid_h * grid_w)
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                # 更新下一个起始位置 st
                # 视觉部分在输入序列中只占 1 个 Token 位置 (<image> 或 <video>)，但在位置编码中展开为 grid 大小
                # 因此这里 st 更新加上了 grid 的总大小
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # 10> 处理剩余的尾部文本
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # 11> 拼接当前样本的所有位置 ID 并填入 position_ids 张量
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # 将生成的 3D 位置 ID 填入对应的有效区域 (attention_mask == 1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)

            # 计算位置增量 delta，用于后续长度修正
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    
    else:
        # 12> 处理纯文本或其他无需 3D 编码的情况
        if attention_mask is not None:
            # 使用简单的累积求和生成 1D 位置 ID
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 扩展为 3D 格式 (3, Batch, Seq)，三个维度值相同
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            
            # 计算 delta
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            # 如果连 mask 都没有，生成最简单的 range 序列
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：
        为 Qwen2.5-VL 模型生成 3D 旋转位置编码 (RoPE) 索引。
        该函数根据图像和视频的时空结构（Temporal, Height, Width）计算对应的位置 ID。
        与 Qwen3-VL 不同，Qwen2.5-VL 的视频位置编码通常使用基于时间间隔（second_per_grid_t）的计算方式。
        具体规则如下：
        1. 文本部分：使用 1D 位置编码，并在 T、H、W 三个维度上共享。
        2. 视觉部分：使用 3D 位置编码，分别计算 T、H、W 索引。
        3. 拼接机制：文本起始位置基于前一个视觉片段的最大位置 ID 加 1。

    参数：
        spatial_merge_size (int, optional): 空间合并因子，默认为 2。
        input_ids (torch.LongTensor, optional): 输入 Token ID 序列，形状为 (Batch, Seq_Len)。
        image_grid_thw (torch.LongTensor, optional): 图像的 THW 网格信息，形状为 (Num_Images, 3)。
        video_grid_thw (torch.LongTensor, optional): 视频的 THW 网格信息，形状为 (Num_Videos, 3)。
        second_per_grid_ts (torch.Tensor, optional): 视频每个网格的时间跨度，形状为 (Num_Videos,)。
        attention_mask (torch.Tensor, optional): 注意力掩码，形状为 (Batch, Seq_Len)。

    返回：
        Tuple[torch.Tensor, torch.Tensor]: 
            - position_ids: (3, Batch, Seq_Len) 形状的位置 ID。
            - mrope_position_deltas: (Batch, 1) 形状的位置增量。

    示例：
        >>> # 假设有一个 1 帧 14x14 的视频
        >>> input_ids = torch.tensor([[151652, 151656, 102]]) # <vision_start>, <video>, text
        >>> video_grid = torch.tensor([[1, 14, 14]])
        >>> pos_ids, deltas = get_rope_index_25(input_ids=input_ids, video_grid_thw=video_grid)
    """
    # 1> 初始化特殊 Token ID 和位置增量列表
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    # 2> 检查是否存在多模态输入
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        
        # 初始化 3D position_ids (3, Batch, Seq)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        # 3> 遍历 Batch 中的每个样本
        for i, input_ids in enumerate(total_input_ids):
            # 过滤 Padding
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            
            # 4> 统计样本中的视觉片段数量及位置
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # 5> 遍历并处理每个视觉片段及其间的文本
            for _ in range(image_nums + video_nums):
                # 寻找下一个图片或视频标记的位置
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # 6> 获取片段的时空参数 (T, H, W)
                if ed_image < ed_video: # 处理图片
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0 # 图片时间间隔为 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else: # 处理视频
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    # 获取该视频对应的时间步长，默认为 1.0
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # 计算在 LLM 特征空间中的网格尺寸
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                
                # 7> 计算文本部分的位置 ID
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                # 文本 1D 映射到 3D (T, H, W 值相同)
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 8> 计算视觉部分的位置 ID (3D RoPE)
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                # 计算时间位置编码：基于时间步长的跳跃计算
                # 示例：t=2, second_per_grid_t=50 -> [0, 100]
                time_tensor = expanded_range * second_per_grid_t * 2
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                # 高度和宽度索引（网格平铺）
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )

                # 叠加偏移并添加至列表
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                # 更新处理进度
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # 9> 处理尾部文本
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # 10> 拼接所有片段并回填至结果张量
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            # 记录位置增量
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    
    else:
        # 11> 处理纯文本或无视觉数据的情况 (1D -> 3D 简单映射)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_rope_index_2(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：
        为 Qwen2-VL 模型生成 3D 旋转位置编码 (RoPE) 索引。
        该函数是 Qwen 系列模型处理多模态位置信息的早期版本逻辑。
        它通过分析 Token 序列中的视觉标记（<image>, <video>），为不同模态分配对应的 3D 坐标。
        与 Qwen2.5/Qwen3 不同，它的时间维度处理相对简单，主要依赖网格的物理结构进行平铺。

    参数：
        spatial_merge_size (int, optional): 空间维度合并比例，默认为 2。
        input_ids (torch.LongTensor, optional): 输入的 Token ID 序列，形状 (Batch, SeqLen)。
        image_grid_thw (torch.LongTensor, optional): 图像特征的 T, H, W 网格信息。
        video_grid_thw (torch.LongTensor, optional): 视频特征的 T, H, W 网格信息。
        second_per_grid_ts (torch.Tensor, optional): (当前逻辑未直接使用) 时间步长信息。
        attention_mask (torch.Tensor, optional): 注意力掩码。

    返回：
        Tuple[torch.Tensor, torch.Tensor]:
            - position_ids: (3, Batch, Seq) 形状的 3D 位置索引。
            - mrope_position_deltas: (Batch, 1) 形状的位置偏移量。

    示例：
        >>> # 典型调用方式
        >>> pos_ids, deltas = get_rope_index_2(
        ...     spatial_merge_size=2,
        ...     input_ids=ids,
        ...     image_grid_thw=img_thw
        ... )
    """
    # 1> 定义特殊 Token 常量
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    # 2> 检查是否存在需要处理的视觉数据
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        
        # 初始化返回的 3D position_ids 张量
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0

        # 3> 逐样本处理 Batch 数据
        for i, input_ids in enumerate(total_input_ids):
            # 获取有效 Token (排除 Padding)
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            
            # 4> 定位视觉片段的起始点并统计数量
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # 5> 遍历并处理每个视觉片段及其间的文本
            for _ in range(image_nums + video_nums):
                # 寻找下一个 <image> 或 <video> 的位置
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # 6> 提取当前片段的 THW 网格参数
                if ed_image < ed_video: # 处理图像
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else: # 处理视频
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # 计算 LLM 内部使用的网格尺寸 (考虑空间合并)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                
                # 7> 生成视觉片段前的文本位置 ID
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                # 文本 1D 映射到 3 个维度（值相同）
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 8> 生成视觉部分的 3D 网格位置索引
                # t_index: 时间维度线性平铺
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                # h_index: 高度维度线性平铺
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                # w_index: 宽度维度线性平铺
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                
                # 堆叠 T, H, W 索引并加上之前的偏移
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                # 更新处理起始点
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # 9> 处理末尾剩余文本
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # 10> 合并所有片段的位置信息并回填
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            # 计算 delta（物理长度与 token 长度的差异）
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        # 11> 将 delta 转换为 Tensor 格式返回
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    
    else:
        # 12> 兜底逻辑：纯文本或简单输入的情况
        if attention_mask is not None:
            # 基于 attention mask 的累积和生成 1D 位置索引并扩展到 3D
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            # 最基础的 arange 生成
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
