import json
import os
import numpy as np
from PIL import Image
from copy import deepcopy
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from torchcodec.decoders import VideoDecoder
import binpacking
from tqdm import tqdm
import concurrent.futures
import time


def read_data(file_path):
    """Read JSON or JSONL file"""
    if file_path.endswith(('.json', '.jsonl')):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            return [json.loads(line) for line in f]
    raise ValueError('Please provide a .json or .jsonl file')


def write_data(file_path, data):
    """Write data to JSON or JSONL file"""
    with open(file_path, 'w') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=4)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item) + '\n')


class DataArguments:
    def __init__(self):
        self.max_pixels = 2048 * 28 * 28
        self.min_pixels = 32 * 28 * 28
        self.video_max_frame_pixels = 576 * 28 * 28
        self.video_min_frame_pixels = 144 * 28 * 28
        self.base_interval = 4
        self.video_min_frames = 4
        self.video_max_frames = 8
        self.data_path = ''


class MultimodalProcessor:
    """多模态数据处理器类

    类功能：
        封装了图像和视频的预处理逻辑，用于计算多模态输入在模型中的 Token 占用数量。

    继承关系：
        无显式继承关系，独立工具类。

    应用场景：
        1. 在数据打包（Data Packing）前，预估每个样本的 Token 长度，以便进行高效的长度分组。
        2. 动态调整图像和视频的预处理参数（如分辨率、帧率），以适应不同的硬件限制。

    使用示例：
        >>> data_args = DataArguments()
        >>> base_processor = Qwen2VLImageProcessor.from_pretrained(...)
        >>> processor = MultimodalProcessor(data_args, base_processor, device='cpu')
        >>> tokens = processor.process_image("example.jpg")

    数据属性：
        data_args: DataArguments
            包含预处理配置参数的对象，如最大/最小像素数、视频帧数采样策略等。
        
        base_processor: Qwen2VLImageProcessor
            基础的 Hugging Face 图像处理器，作为克隆模板使用。
        
        device: str
            执行视频解码运算的设备（'cpu' 或 'cuda'），默认为 'cpu'。
    """
    def __init__(self, data_args, base_processor, device='cpu'):
        self.data_args = data_args
        self.base_processor = base_processor
        self.device = device

    def _configure_processor(self, max_val, min_val):
        """配置处理器参数

        功能：
            创建基础处理器的深拷贝，并根据给定的最大/最小像素值动态更新其配置。
            这确保了每次处理（图像或视频）时都使用独立的、针对该任务配置的处理器实例。

        参数：
            max_val (int): 允许的最大像素数（高度 x 宽度）。
            min_val (int): 允许的最小像素数。

        返回：
            Qwen2VLImageProcessor: 配置好的新处理器实例。

        示例：
            >>> proc = self._configure_processor(2048*28*28, 32*28*28)
        """
        # 1> 深拷贝基础处理器，避免修改原始对象
        processor = deepcopy(self.base_processor)
        
        # 2> 设置像素限制参数
        processor.max_pixels = max_val
        processor.min_pixels = min_val
        
        # 3> 设置具体的尺寸约束（最长边和最短边）
        # 注意：这里的逻辑简化为使用总像素限制来指导尺寸，具体行为依赖于 processor 内部实现
        processor.size = {'longest_edge': max_val, 'shortest_edge': min_val}
        return processor

    def process_image(self, image_file):
        """处理单张图像

        功能：
            加载图像文件，进行预处理，并计算其在 Qwen-VL 模型中对应的视觉 Token 数量。

        参数：
            image_file (str): 图像文件的相对路径（相对于 data_args.data_path）。

        返回：
            int: 该图像产生的视觉 Token 数量。如果文件不存在，返回 0。

        示例：
            >>> count = processor.process_image("images/cat.jpg")
        """
        # 1> 拼接完整图像路径
        image_path = os.path.join(self.data_args.data_path, image_file)
        
        # 2> 检查文件是否存在
        if not os.path.exists(image_path):
            print(f'Image file does not exist: {image_path}')
            return 0
            
        # 3> 配置图像专用的处理器
        processor = self._configure_processor(self.data_args.max_pixels, self.data_args.min_pixels)
        
        # 4> 加载并转换图像格式
        image = Image.open(image_path).convert('RGB')
        
        # 5> 执行预处理，获取模型输入张量
        visual_processed = processor.preprocess(images=image, return_tensors='pt')
        
        # 6> 计算 Token 数量
        # 原理：Qwen-VL 将图像编码为 grid_t * grid_h * grid_w 的特征块
        # 并通过池化或其他机制压缩（此处除以 4 表示每 2x2 个特征块对应 1 个 Token 的某种关系，或者具体的编码压缩比）
        return visual_processed['image_grid_thw'].prod() // 4

    def process_video(self, video_file):
        """处理单个视频

        功能：
            加载视频文件，按策略采样帧，进行预处理，并计算其在 Qwen-VL 模型中对应的视觉 Token 数量。

        参数：
            video_file (str): 视频文件的相对路径。

        返回：
            int: 该视频产生的视觉 Token 数量。

        示例：
            >>> count = processor.process_video("videos/demo.mp4")
        """
        # 1> 拼接完整视频路径
        video_path = os.path.join(self.data_args.data_path, video_file)
        
        # 2> 配置视频专用的处理器（使用视频特定的像素限制）
        processor = self._configure_processor(self.data_args.video_max_frame_pixels, self.data_args.video_min_frame_pixels)
        
        # 3> 初始化视频解码器
        decoder = VideoDecoder(video_path, device=self.device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        
        # 4> 计算采样帧数
        # 策略：根据视频时长和基础间隔 (base_interval) 计算理论帧数，并限制在 [min_frames, max_frames] 范围内
        interval = self.data_args.base_interval
        num_frames_to_sample = round(video_length / interval)
        target_frames = min(max(num_frames_to_sample, self.data_args.video_min_frames), self.data_args.video_max_frames)

        # 5> 生成均匀采样的帧索引
        frame_idx = np.unique(np.linspace(0, total_frames - 1, target_frames, dtype=int)).tolist()

        # 6> 解码指定帧并转换为 Numpy 数组
        frame_batch = decoder.get_frames_at(indices=frame_idx)
        video_frames_numpy = frame_batch.data.cpu().numpy()
        
        # 7> 执行预处理
        # 注意：这里调用 preprocess 时传入的是 videos 参数
        visual_processed = processor.preprocess(images=None, videos=video_frames_numpy, return_tensors='pt')
        
        # 8> 计算 Token 数量
        return visual_processed['video_grid_thw'].prod() // 4


def calculate_tokens(conversation, processor, tokenizer):
    """
    功能：
        计算单条多模态对话数据所需的总 Token 数量。
        该函数累加了三部分的 Token：
        1. 基础系统开销（固定的起始/特殊 Token）。
        2. 对话文本内容经过分词器编码后的长度。
        3. 图像或视频文件经过处理器处理后占据的视觉 Token 数量。

    参数：
        conversation (dict): 包含对话内容和媒体信息的字典，通常包含 'conversations'、'image' 或 'video' 字段。
        processor (MultimodalProcessor): 用于处理图像和视频并计算视觉 Token 的处理器实例。
        tokenizer (transformers.PreTrainedTokenizer): 用于将文本转换为 Token ID 的分词器。

    返回：
        int: 该条数据对应的总 Token 数量。

    示例：
        >>> conv = {'conversations': [...], 'image': 'cat.jpg'}
        >>> total = calculate_tokens(conv, processor, tokenizer)
        >>> print(total)
        # 输出: 1536
    """
    # 1> 初始化基础 Token 计数
    # 21 是经验值，可能包含 BOS, EOS 以及对话模板中固定的特殊标记开销
    total_tokens = 21
    roles = {'human': 'user', 'gpt': 'assistant'}
    
    # 2> 计算文本部分的 Token 数量
    for message in conversation['conversations']:
        role = message['from']
        text = message['value']
        
        # 将数据转换为 chat_template 接受的格式
        conv = [{'role': roles[role], 'content': text}]
        
        # 使用 tokenizer 应用模版并编码
        # add_generation_prompt=False 表示仅计算当前消息，不添加回复引导
        encode_id = tokenizer.apply_chat_template(conv, return_tensors='pt', add_generation_prompt=False)[0]
        
        # 累加编码后的 Token ID 长度
        total_tokens += len(encode_id)

    # 3> 计算视觉部分的 Token 数量（图像或视频）
    if 'image' in conversation:
        # 兼容单张图片（str）和多张图片（list）的情况
        images = conversation['image'] if isinstance(conversation['image'], list) else [conversation['image']]
        for image_file in images:
            # 调用 processor 处理图像并获取 Token 数
            total_tokens += processor.process_image(image_file)
    elif 'video' in conversation:
        # 兼容单个视频（str）和多个视频（list）的情况
        videos = conversation['video'] if isinstance(conversation['video'], list) else [conversation['video']]
        for video_file in videos:
            # 调用 processor 处理视频并获取 Token 数
            total_tokens += processor.process_video(video_file)

    return total_tokens


def pack_data(data_list, pack_length):
    """
    功能：
        使用装箱算法（Bin Packing）对数据样本进行分组打包。
        该函数旨在将多个短样本组合成一个长序列（Group），使得每个 Group 的总 Token 数量尽可能接近但不超过 `pack_length`。
        这样做可以显著减少训练时的 Padding 浪费，提高计算效率（即 Sequence Packing / Sample Packing）。

    参数：
        data_list (List[dict]): 包含原始数据样本的列表，每个字典必须包含预计算好的 'num_tokens' 字段。
        pack_length (int): 设定的目标序列长度（通常等于模型的最大上下文长度，如 2048, 4096 等）。

    返回：
        List[List[dict]]: 打包后的数据列表。
                          外层列表的每个元素代表一个 Group（即一个 Batch 中实际的一行数据），
                          内层列表包含该 Group 中聚合的多个原始样本。

    示例：
        >>> samples = [{'id': 1, 'num_tokens': 100}, {'id': 2, 'num_tokens': 50}]
        >>> packed = pack_data(samples, pack_length=150)
        >>> print(len(packed)) # 可能输出 1，表示两个样本被打包进了一个 Group
    """
    # 1> 提取所有样本的长度信息
    # 从数据字典中获取预先计算好的 num_tokens
    lengths = [data["num_tokens"] for data in data_list]
    
    # 2> 执行装箱算法（Bin Packing）
    # 使用 binpacking 库的 to_constant_volume 方法
    # 输入: list(enumerate(lengths)) -> [(0, len0), (1, len1), ...]
    # pack_length: 每个箱子的最大容量
    # weight_pos=1: 指定元组中第 2 个元素（即长度）作为权重
    # 返回: grouped_indices 是一组 list，每个 list 包含被分到同一组的 (index, length) 元组
    grouped_indices = binpacking.to_constant_volume(
        list(enumerate(lengths)),  # 显式转换为 list 兼容不同版本
        pack_length,
        weight_pos=1
    )
    
    # 3> 根据分组索引重组数据
    packed_data = []
    for group in grouped_indices:
        group_data = []
        for index, _ in group:
            # 复制原始数据，避免修改原列表
            new_data = data_list[index].copy()
            
            # 4> 移除辅助字段
            # 'num_tokens' 仅用于打包计算，实际训练数据中不再需要
            new_data.pop("num_tokens", None)
            
            group_data.append(new_data)
        packed_data.append(group_data)
        
    return packed_data


# 1> 配置数据集与路径信息：字典键为数据集名称，值为包含数据文件路径和注解文件路径的配置字典
datasets = {
    'dummy_dataset': {
        'data_path': '',  # 数据集根目录
        'annotation_path': 'path/to/your/annotation.json'  # 标注文件路径
    }
}

# 2> 初始化全局配置与模型组件
data_args = DataArguments()
model_path = 'path/to/your/model'  # 预训练模型路径

# 加载分词器，并自定义 chat_template 以适配 Qwen-VL 的对话格式
# 模板逻辑：遍历消息 -> 拼接 role 和 content -> 添加 im_start/im_end 标记
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# 加载基础图像处理器
base_image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
print(f'Successfully loaded model components from {model_path}')

# 实例化多模态处理器
processor = MultimodalProcessor(data_args, base_image_processor, device='cpu')

# 3> 遍历处理每个数据集
for dataset_name, config in datasets.items():
    # 4> 设置当前数据集的路径参数
    processor.data_args.data_path = config['data_path']
    annotation_path = os.path.join(processor.data_args.data_path, config['annotation_path'])
    
    print(f'\n--- Processing dataset: {dataset_name} ---')
    print(f'Annotation file path: {annotation_path}')
    print(f'Image configuration: max_pixels={data_args.max_pixels}, min_pixels={data_args.min_pixels}')
    print(f'Video frame configuration: video_max_frame_pixels={data_args.video_max_frame_pixels}, video_min_frame_pixels={data_args.video_min_frame_pixels}')
    
    if not os.path.exists(annotation_path):
        print(f'Annotation file not found: {annotation_path}')
        continue
    
    # 读取原始数据
    data = read_data(annotation_path)

    # 5> 获取或计算 Token 数量
    # 检查是否已存在包含 Token 计数的中间文件，避免重复计算
    count_file_path = annotation_path.replace('.jsonl', '_count.json').replace('.json', '_count.json')
    if os.path.exists(count_file_path):
        print(f"Found pre - calculated token counts, loading data from {count_file_path}.")
        data_with_tokens = read_data(count_file_path)
    else:
        # 定义单条数据的处理函数
        def calculate_and_update(item):
            item['num_tokens'] = calculate_tokens(item, processor, tokenizer)
            return item
        
        # 使用线程池并行计算 Token 数量，提高 I/O 密集型任务（如读取图片）的效率
        with concurrent.futures.ThreadPoolExecutor() as executor:
            data_with_tokens = list(tqdm(executor.map(calculate_and_update, data), total=len(data), desc=f"Processing {dataset_name} data"))

        # 保存带有 Token 计数的结果，方便下次直接加载
        write_data(count_file_path, data_with_tokens)
        print(f"Token counts saved to: {count_file_path}")

    # 6> 执行数据打包 (Bin Packing)
    # 设定目标序列长度（通常与模型最大上下文长度一致）
    # Assume the packing length is 4096
    pack_length = 4096
    
    # 设定批处理大小，分批进行打包算法运算，防止内存溢出
    # Define the batch size
    batch_size = 256
    all_packed_results = []

    # 记录打包算法的执行时间
    # Record the start time of binpacking
    start_time = time.time()

    # 分批次执行打包
    for i in range(0, len(data_with_tokens), batch_size):
        batch_data = data_with_tokens[i: i + batch_size]
        # 对当前批次数据进行装箱打包
        batch_packed_result = pack_data(batch_data, pack_length)
        all_packed_results.extend(batch_packed_result)
        
    # Record the end time of binpacking
    end_time = time.time()

    # Calculate the time spent on binpacking
    binpack_time = end_time - start_time
    print(f"Time spent on binpacking: {binpack_time:.4f} seconds")

    # 7> 保存最终打包结果
    # Save the packed results as a JSON file
    pack_output_path = annotation_path.replace('.jsonl', '_pack.json').replace('.json', '_pack.json')
    with open(pack_output_path, 'w', encoding='utf-8') as file:
        json.dump(all_packed_results, file, indent=2)
    print(f"Packed results saved to: {pack_output_path}")