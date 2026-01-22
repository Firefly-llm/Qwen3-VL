import os
from re import A
import pandas as pd
import numpy as np
from typing import Dict, Any
from common_utils import download_file, md5, toliststr, decode_base64_to_image_file

MMMU_DATASET_URL = 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv'
MMMU_DATASET_MD5 = '521afc0f3bf341e6654327792781644d'

def load_dataset(dataset_name='MMMU_DEV_VAL'):
    """
    功能：
        加载 MMMU 数据集并进行初步的清洗与处理。
        支持自动下载（如果本地不存在或校验失败）、解析 TSV 文件、以及处理图像数据字段。

    参数：
        dataset_name (str, optional): 数据集的名称。默认为 'MMMU_DEV_VAL'。

    返回：
        pd.DataFrame: 处理后的数据集内容。

    示例：
        >>> df = load_dataset('MMMU_DEV_VAL')
        >>> print(df.head(2))
           index question  options  answer  image  ...
        0      0     ...      ...     ...    ...
        1      1     ...      ...     ...    ...
    """
    # 1> 确定数据存放的根目录
    # 从环境变量 'LMUData' 获取路径，并确保目录存在
    data_root = os.path.join(os.environ['LMUData'])
    os.makedirs(data_root, exist_ok=True)

    # 2> 构造数据文件的完整路径
    file_name = f"{dataset_name}.tsv"
    data_path = os.path.join(data_root, file_name)
    
    # 3> 检查数据文件是否有效
    # 如果文件不存在，或者 MD5 校验码与预期不符，则触发下载
    if not os.path.exists(data_path) or md5(data_path) != MMMU_DATASET_MD5:
        print(f"Downloading {dataset_name} dataset...")
        download_file(MMMU_DATASET_URL, data_path)
    
    # 4> 读取数据集内容
    # 使用 pandas 以制表符为分隔符读取，目前仅选取前 8 行进行处理（可能是为了快速验证）
    data = pd.read_csv(data_path, sep='\t').iloc[:8]
    
    # 5> 标准化索引字段
    # 将索引列强制转换为字符串类型，确保一致性
    data['index'] = [str(x) for x in data['index']]
    
    # 6> 处理图像数据列
    if 'image' in data:
        # 将图像内容（可能是 base64 字符串）转换为字符串格式
        data['image'] = [str(x) for x in data['image']]
        # 构建索引到图像内容的映射关系，用于处理可能的引用逻辑
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            # 如果内容长度过短（<= 64），可能只是一个重定向到其他索引的 ID
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                # 确保被引用的索引确实存在且包含实际的图像数据
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        # 格式化图像字段，确保返回的是列表或单个元素
        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]

    # 7> 处理图像路径列
    if 'image_path' in data:
        # 统一将路径转换为字符串列表，并提取单路径情况下的元素
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # 8> 转换索引类型（如果可能）
    # 如果索引全是数字，则转换为整数类型，方便后续排序或索引操作
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    # 9> 返回处理完成的 DataFrame
    return data

def dump_image(line, img_root):
    """
    功能：
        将数据集中的图像数据（Base64 编码）保存到本地磁盘，并返回保存后的文件路径列表。
        支持单图像和多图像处理，并能根据索引或指定文件名自动命名。

    参数：
        line (Dict[str, Any]): 包含数据行信息的字典，通常包含 'image' (Base64 数据)、'image_path' 或 'index'。
        img_root (str): 图像保存的根目录路径。

    返回：
        List[str]: 保存后的本地图像绝对路径列表。

    示例：
        >>> line = {'index': '0', 'image': 'base64_data...'}
        >>> paths = dump_image(line, './images')
        >>> print(paths)
        ['./images/0.jpg']
    """
    # 1> 确保图像保存的根目录存在
    os.makedirs(img_root, exist_ok=True)
    
    # 2> 处理包含图像数据（Base64）的情况
    if 'image' in line:
        if isinstance(line['image'], list):
            # 场景 A: 存在多张图像
            tgt_path = []
            # 确保存在对应的文件名列表 'image_path'
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = os.path.join(img_root, im_name)
                # 3> 如果本地文件不存在，则解码并保存
                if not os.path.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            # 场景 B: 只有单张图像
            # 使用索引 ID 作为文件名
            tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
            # 4> 解码 Base64 字符串并保存为 JPEG 文件
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            # 统一转换为列表格式返回
            tgt_path = [tgt_path]
    else:
        # 场景 C: 只有图像路径信息，没有原始图像数据
        # 此时直接返回路径列表（假设图像已存在或在别处处理）
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
    
    # 5> 返回所有处理后的图像路径
    return tgt_path

def MMMU_preproc(data):
    """
    功能：
        预处理 MMMU 数据集，将开放式问题重新格式化为多选题形式。
        该处理逻辑确保数据集中的所有条目都具备统一的选择题结构（如 A, B 选项），以便于后续的评测。

    参数：
        data (pd.DataFrame): 原始的 MMMU 数据集，包含 'A', 'B' 和 'answer' 等列。

    返回：
        pd.DataFrame: 格式化后的数据集。

    示例：
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [None], 'B': [None], 'answer': ['Answer X']})
        >>> processed_df = MMMU_preproc(df)
        >>> print(processed_df.iloc[0]['A'])
        Answer X
    """
    print("Preprocessing MMMU dataset...")
    # 1> 初始化计数器，用于统计转换了多少个开放式问题
    cnt = 0

    # 2> 提取关键列数据为列表格式，方便快速遍历
    # A, B 是选项列，Ans 是标准答案列
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)

    # 3> 遍历数据集并执行格式转换
    for i in range(lt):
        # 4> 判断是否为开放式问题
        # 在 MMMU 中，如果选项 A 为空（NaN），则通常被视为开放式问题
        if pd.isna(As[i]):
            # 5> 将标准答案填入选项 A，并将选项 B 设为干扰项 "Other Answers"
            # 这样就把一个填空/开放式问题转换成了一个二选一的单选题
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    
    # 6> 打印预处理统计结果
    print(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones.')
    
    # 7> 将处理后的列表回填到 DataFrame 中
    data['A'] = As
    data['B'] = Bs
    
    # 8> 返回更新后的数据集
    return data