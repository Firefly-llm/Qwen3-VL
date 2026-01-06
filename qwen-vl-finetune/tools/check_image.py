import json
import os
from tqdm import tqdm
from datasets import load_dataset


def validate_data(json_file_path, media_folder_path):
    """
    功能：
        验证多模态数据集 JSON 文件的完整性和一致性。
        主要执行两项核心检查：
        1. 检查数据条目中引用的所有媒体文件（图片、视频）是否在指定文件夹中实际存在。
        2. 检查对话文本（conversations）中的媒体占位符（如 <image>, <video>）数量是否与元数据中声明的媒体文件数量一致。
        最后将验证合格的数据和有问题的数据分别保存到不同的 JSON 文件中，并打印统计摘要。

    参数：
        json_file_path (str): 数据集标注文件的路径，支持 .json 或 .jsonl 格式。
        media_folder_path (str): 存放媒体文件（图片、视频）的根目录路径，用于拼接绝对路径进行存在性检查。

    返回：
        None: 该函数没有返回值，验证结果直接写入磁盘文件（*_valid.json 和 *_problems.json）。

    示例：
        >>> json_path = "/data/dataset/train.json"
        >>> media_root = "/data/dataset/images"
        >>> validate_data(json_path, media_root)
        # 输出：Processing 1000 entries...
        # Validation Summary: ...
    """
    # 1> 校验输入文件格式
    # 仅支持 .json 或 .jsonl 后缀的文件
    if not json_file_path.endswith((".json", ".jsonl")):
        print("Invalid file format. Please provide a .json or .jsonl file.")
        return
    
    # 2> 准备输出文件路径
    # 分离文件名和扩展名，生成 *_valid.json 和 *_problems.json 的路径
    base_path = os.path.splitext(json_file_path)[0]
    valid_file_path = f"{base_path}_valid.json"
    problem_file_path = f"{base_path}_problems.json"

    # 3> 加载数据集
    # 使用 datasets 库加载数据，支持大数据集高效加载
    try:
        data = load_dataset("json", data_files=json_file_path)["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 4> 初始化统计计数器和结果容器
    valid_data = []
    problem_data = []
    stats = {
        'total_entries': 0,        # 总条目数
        'valid_entries': 0,        # 合格条目数
        'missing_media': 0,        # 媒体文件缺失的条目数
        'token_mismatches': 0,     # Token 数量不匹配的条目数
        'gpt_media_tokens': 0,     # GPT 回复中包含媒体 Token 的条目数
        'missing_files': [],       # 所有缺失文件的路径列表
        'media_types': {           # 媒体类型分布统计
            'image': 0,
            'video': 0,
            'mixed': 0
        }
    }

    print(f"Processing {len(data)} entries...")

    # 5> 遍历每一条数据进行验证
    for item in tqdm(data):
        stats['total_entries'] += 1
        problems = []
        
        # 6> 提取媒体文件列表
        # 兼容单数 ("image", "video") 和复数 ("images", "videos") 字段
        media_info = {
            'image': item.get("image", item.get("images", [])),
            'video': item.get("video", item.get("videos", []))
        }
        
        # 7> 规范化媒体字段为列表格式
        # 确保无论是字符串还是列表，最终都统一处理为 list
        for media_type in media_info:
            if isinstance(media_info[media_type], str):
                media_info[media_type] = [media_info[media_type]]
            elif not isinstance(media_info[media_type], list):
                media_info[media_type] = []
        
        # 8> 统计媒体类型分布
        # 计算该条数据中包含的媒体数量，并更新全局统计
        media_counts = {k: len(v) for k, v in media_info.items()}
        active_media = [k for k, v in media_counts.items() if v > 0]

        if len(active_media) > 1:
            stats['media_types']['mixed'] += 1
        elif len(active_media) == 1:
            stats['media_types'][active_media[0]] += 1
        
        # 9> 验证物理文件是否存在
        # 遍历所有引用的媒体文件，检查磁盘上是否存在
        missing_files = []
        for media_type, files in media_info.items():
            for media_file in files:
                # 拼接完整路径：媒体根目录 + 相对路径
                media_path = os.path.join(media_folder_path, media_file)
                if not os.path.exists(media_path):
                    missing_files.append(media_path)
        
        # 如果发现文件缺失，记录错误信息
        if missing_files:
            stats['missing_media'] += 1
            stats['missing_files'].extend(missing_files)
            problems.append({
                'type': 'missing_files',
                'files': missing_files,
                'message': f"Missing media files: {missing_files}"
            })
        
        # 10> 验证 Token 一致性
        # 统计对话文本（human 和 gpt）中的 <image>/<video> 标签数量
        conversations = item.get("conversations", [])
        expected_counts = {
            'image': media_counts['image'],
            'video': media_counts['video']
        }
        
        actual_counts = {
            'image': 0,
            'video': 0
        }
        gpt_has_media_token = False

        for conv in conversations:
            if conv.get("from") == "human":
                # 仅统计 human 发送的媒体 token
                actual_counts['image'] += conv.get("value", "").count("<image>")
                actual_counts['video'] += conv.get("value", "").count("<video>")
            elif conv.get("from") == "gpt":
                # 检查 GPT 回复中是否错误地包含了媒体 token（通常不应包含）
                if "<image>" in conv.get("value", "") or "<video>" in conv.get("value", ""):
                    gpt_has_media_token = True
        
        # 11> 对比期望数量与实际 Token 数量
        for media_type in ['image', 'video']:
            if actual_counts[media_type] != expected_counts[media_type]:
                stats['token_mismatches'] += 1
                problems.append({
                    'type': 'token_mismatch',
                    'media_type': media_type,
                    'expected': expected_counts[media_type],
                    'actual': actual_counts[media_type],
                    'message': f"Expected {expected_counts[media_type]} <{media_type}> tokens, found {actual_counts[media_type]}"
                })
                break  # 只要有一种类型不匹配，就标记为不匹配
        
        # 12> 检查 GPT 回复异常
        if gpt_has_media_token:
            stats['gpt_media_tokens'] += 1
            problems.append({
                'type': 'gpt_media_token',
                'message': "GPT response contains media token (<image> or <video>)"
            })
        
        # 13> 数据分流
        # 根据是否存在问题，将数据分流到 valid_data 或 problem_data
        if not problems:
            stats['valid_entries'] += 1
            valid_data.append(item)
        else:
            problem_item = item.copy()
            problem_item['validation_problems'] = problems
            problem_data.append(problem_item)
    
    # 14> 保存验证结果
    with open(valid_file_path, 'w') as f:
        json.dump(valid_data, f, indent=2)
    
    with open(problem_file_path, 'w') as f:
        json.dump(problem_data, f, indent=2)
    
    # 15> 打印最终统计摘要
    print("\nValidation Summary:")
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Valid entries: {stats['valid_entries']} ({stats['valid_entries']/stats['total_entries']:.1%})")
    print(f"Media type distribution:")
    print(f"  - Image only: {stats['media_types']['image']}")
    print(f"  - Video only: {stats['media_types']['video']}")
    print(f"  - Mixed media: {stats['media_types']['mixed']}")
    print(f"Entries with missing media: {stats['missing_media']}")
    print(f"Entries with token mismatches: {stats['token_mismatches']}")
    print(f"Entries with GPT media tokens: {stats['gpt_media_tokens']}")
    
    if stats['missing_files']:
        print("\nSample missing files (max 5):")
        for f in stats['missing_files'][:5]:
            print(f"  - {f}")

# Example usage
if __name__ == "__main__":
    json_file_path = "example.json"  # Replace with your JSON file path
    media_folder_path = "media"      # Replace with your media folder path
    validate_data(json_file_path, media_folder_path)