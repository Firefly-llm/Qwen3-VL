import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional, Union, Tuple, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8

MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
logger = logging.getLogger(__name__)


def round_by_factor(number: int, factor: int) -> int:
    """
    功能：
        将给定的整数 number 按照指定的因子 factor 进行四舍五入取整。
        结果是最接近 number 且能被 factor 整除的整数。
        常用于调整图像或视频尺寸，使其符合模型对输入维度的对齐要求（如 Patch Size 的倍数）。

    参数：
        number (int): 需要被取整的原始数值。
        factor (int): 取整的基准因子（步长）。

    返回：
        int: 四舍五入后能被 factor 整除的最近整数。

    示例：
        >>> round_by_factor(28, 14)
        28
        >>> round_by_factor(30, 14)
        28
        >>> round_by_factor(36, 14)
        42
    """
    # number / factor 得到浮点数比例
    # round() 将其四舍五入为最近的整数
    # 最后乘以 factor 还原为对齐后的数值
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """
    功能：
        将给定的整数 number 按照指定的因子 factor 进行向上取整（Ceiling）。
        结果是大于或等于 number 且能被 factor 整除的最小整数。
        常用于确保图像或视频尺寸满足最小对齐要求，避免因尺寸过小而无法被切分成完整的 Patch。

    参数：
        number (int): 需要被取整的原始数值。
        factor (int): 取整的基准因子（步长）。

    返回：
        int: 向上取整后能被 factor 整除的最小整数。

    示例：
        >>> ceil_by_factor(28, 14)
        28
        >>> ceil_by_factor(30, 14)
        42
        >>> ceil_by_factor(1, 14)
        14
    """
    # number / factor 得到浮点数比例
    # math.ceil() 将其向上取整为最近的整数
    # 最后乘以 factor 还原为对齐后的数值
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """
    功能：
        将给定的整数 number 按照指定的因子 factor 进行向下取整（Floor）。
        结果是小于或等于 number 且能被 factor 整除的最大整数。
        常用于裁剪图像或视频尺寸，使其不超过特定的边界限制，同时保持对齐。

    参数：
        number (int): 需要被取整的原始数值。
        factor (int): 取整的基准因子（步长）。

    返回：
        int: 向下取整后能被 factor 整除的最大整数。

    示例：
        >>> floor_by_factor(28, 14)
        28
        >>> floor_by_factor(30, 14)
        28
        >>> floor_by_factor(13, 14)
        0
    """
    # number / factor 得到浮点数比例
    # math.floor() 将其向下取整为最近的整数
    # 最后乘以 factor 还原为对齐后的数值
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    功能：
        智能调整图像尺寸，使其满足多模态模型输入的特定约束条件。
        该算法会寻找一个最优的新尺寸 (h_bar, w_bar)，使得：
        1. 尺寸整除性：新高度和宽度都能被 `factor` 整除（通常对应 Patch Size）。
        2. 像素数量限制：总像素数落在 `[min_pixels, max_pixels]` 区间内。
        3. 宽高比保持：尽可能保持原始图像的宽高比不变。

    参数：
        height (int): 原始图像的高度。
        width (int): 原始图像的宽度。
        factor (int): 尺寸对齐的基准因子（通常为 patch_size * merge_size）。
        min_pixels (int, optional): 允许的最小像素总数。如果为 None，默认为 IMAGE_MIN_TOKEN_NUM * factor^2。
        max_pixels (int, optional): 允许的最大像素总数。如果为 None，默认为 IMAGE_MAX_TOKEN_NUM * factor^2。

    返回：
        Tuple[int, int]: 调整后的目标高度和宽度 (h_bar, w_bar)。

    示例：
        >>> h, w = smart_resize(1000, 1000, factor=28, min_pixels=28*28, max_pixels=1024*28*28)
        >>> print(h, w)
        (1008, 1008)  # 1008 是 28 的倍数，且接近 1000
    """
    # 1> 设置默认的像素限制范围
    # 如果未提供，使用预设的最小/最大 Token 数乘以 Patch 面积作为默认值
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)

    # 2> 校验参数合法性
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    
    # 3> 检查宽高比是否过于极端
    # 如果长边与短边之比超过阈值 (MAX_RATIO=200)，则抛出异常，因为这可能导致模型处理异常
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    
    # 4> 初始尺寸调整
    # 将高宽分别四舍五入到最近的 factor 倍数，并确保至少为 factor 大小
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # 5> 根据像素总数限制进行二次调整
    # 情况 A: 调整后的像素总数超过最大限制
    if h_bar * w_bar > max_pixels:
        # 计算缩放系数 beta，使得缩放后的面积等于 max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 缩小尺寸并向下取整，确保不超过 max_pixels
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)

    # 情况 B: 调整后的像素总数低于最小限制
    elif h_bar * w_bar < min_pixels:
        # 计算缩放系数 beta，使得缩放后的面积等于 min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 放大尺寸并向上取整，确保不低于 min_pixels
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
        
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """
    功能：
        将输入的 PIL 图像转换为标准的 RGB 模式。
        特别处理 RGBA 模式的图像：通过创建一个纯白背景，将 RGBA 图像的 Alpha 通道作为掩码粘贴上去，
        从而正确处理透明度（将透明部分转为白色），避免直接转换导致的黑色背景问题。

    参数：
        pil_image (Image.Image): 输入的 PIL 图像对象，可能是 RGB、RGBA、L 等各种模式。

    返回：
        Image.Image: 转换后的 RGB 模式图像。

    示例：
        >>> img_rgba = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        >>> img_rgb = to_rgb(img_rgba)
        >>> print(img_rgb.mode)
        'RGB'
    """
    # 1> 检查图像模式是否为 RGBA（带透明通道）
    if pil_image.mode == 'RGBA':
        # 2> 创建一个纯白色的 RGB 背景图像，尺寸与原图一致
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        
        # 3> 将原图粘贴到白色背景上
        # 使用原图的 Alpha 通道 (split()[3]) 作为掩码 (mask)
        # 这样透明部分会露出底下的白色，不透明部分保留原色
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        # 4> 对于非 RGBA 模式，直接调用 convert 进行转换
        return pil_image.convert("RGB")


def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
    """
    功能：
        从多种输入源（本地路径、URL、Base64、PIL对象）获取图像，并进行预处理（格式统一、尺寸调整）。
        该函数是多模态数据加载的核心入口，负责将异构的输入标准化为模型可接受的 PIL Image 对象。

    参数：
        ele (dict): 包含图像信息的字典。支持以下键：
            - 'image' 或 'image_url': 图像源，可以是本地路径、HTTP(S) URL、Base64 字符串或 PIL Image 对象。
            - 'resized_height', 'resized_width' (optional): 强制指定的调整尺寸。
            - 'min_pixels', 'max_pixels' (optional): 智能缩放时的像素限制范围。
        image_patch_size (int, optional): 模型使用的 Patch 尺寸，默认为 14。用于计算对齐因子。

    返回：
        Image.Image: 处理完成（已转 RGB、已 Resize）的 PIL 图像对象。

    示例：
        >>> info = {'image': 'https://example.com/cat.jpg'}
        >>> img = fetch_image(info)
        >>> print(img.size)
    """
    # 1> 提取图像源
    # 兼容 'image' 和 'image_url' 两个字段名
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    # 计算对齐因子：patch_size * 2 (SPATIAL_MERGE_SIZE=2)
    # 例如 14 * 2 = 28，图像尺寸必须是 28 的倍数
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    
    # 2> 根据输入类型加载图像
    if isinstance(image, Image.Image):
        # A: 如果已经是 PIL 对象，直接使用
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # B: 如果是网络 URL，下载图像
        # stream=True 减少内存占用
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            # 使用 BytesIO 读取内存中的二进制数据，并 deepcopy 确保资源独立
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        # C: 如果是 file:// 协议，截取路径部分并打开
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        # D: 如果是 Base64 编码，解码并打开
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        # E: 默认为本地文件路径
        image_obj = Image.open(image)
    
    # 检查是否成功加载
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    
    # 3> 统一转换为 RGB 格式
    # 处理透明背景等边缘情况
    image = to_rgb(image_obj)

    # 4> 执行尺寸调整 (Resize)
    if "resized_height" in ele and "resized_width" in ele:
        # 路径 A: 如果指定了明确的目标尺寸，使用 smart_resize 进行对齐调整
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
        )
    else:
        # 路径 B: 否则根据像素限制进行智能缩放
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
    # 5> 应用最终尺寸
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """
    功能：
        智能计算视频输入到模型时需要提取的帧数。
        根据配置（指定帧数或指定FPS）动态确定，并确保满足模型的对齐要求（FRAME_FACTOR 的倍数）。
        支持两种模式：直接指定帧数 (`nframes`) 或根据帧率 (`fps`) 自动计算，并支持最大/最小帧数限制。

    参数：
        ele (dict): 包含视频配置信息的字典。必须包含 `nframes` 或 `fps` 其中之一（互斥）。
            支持的键：
            - 'nframes': 直接指定要提取的帧数。
            - 'fps': 指定提取的帧率，结合视频时长计算帧数。
            - 'min_frames' (optional): 最小帧数限制（仅在 fps 模式下生效），默认为 FPS_MIN_FRAMES。
            - 'max_frames' (optional): 最大帧数限制（仅在 fps 模式下生效），默认为 FPS_MAX_FRAMES。
        total_frames (int): 视频原始的总帧数。
        video_fps (Union[int, float]): 视频原始的帧率。

    返回：
        int: 计算得出的用于模型输入的最终帧数。

    示例：
        >>> # 模式 1: 直接指定帧数
        >>> conf1 = {'nframes': 16}
        >>> n1 = smart_nframes(conf1, total_frames=100, video_fps=25.0)
        >>> print(n1)
        16

        >>> # 模式 2: 指定采样帧率
        >>> conf2 = {'fps': 1.0, 'min_frames': 4, 'max_frames': 64}
        >>> # 视频时长 10s (250/25)，按 1fps 采样应为 10 帧
        >>> n2 = smart_nframes(conf2, total_frames=250, video_fps=25.0)
        >>> print(n2)
        10
    """
    # 1> 检查参数互斥性
    # 确保 'fps' 和 'nframes' 不同时存在，避免配置冲突
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"

    if "nframes" in ele:
        # 2> 模式 A: 直接指定帧数
        # 将指定帧数四舍五入到 FRAME_FACTOR 的倍数，以满足对齐要求
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        # 3> 模式 B: 根据帧率计算帧数
        fps = ele.get("fps", FPS)

        # 计算并对齐最小/最大帧数限制
        # min_frames: 向上取整对齐
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        # max_frames: 向下取整对齐，且不能超过视频总帧数
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)

        # 根据时长和目标 FPS 计算理论帧数
        # duration = total_frames / video_fps
        # target_frames = duration * target_fps
        nframes = total_frames / video_fps * fps

        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")

        # 4> 应用范围限制
        # 逻辑：clamp(nframes, min_frames, max_frames) 且不超过 total_frames
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        
        # 向下取整到 FRAME_FACTOR 的倍数
        nframes = floor_by_factor(nframes, FRAME_FACTOR)

    # 5> 最终有效性检查
    # 确保最终帧数在 [FRAME_FACTOR, total_frames] 范围内
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """
    功能：
        使用 torchvision.io.read_video 后端读取视频数据。
        支持从本地路径或 URL 读取，处理视频截取（start/end），
        并根据配置（nframes/fps）进行帧采样。

    参数：
        ele (dict): 包含视频配置信息的字典。支持以下键：
            - 'video': 视频路径，支持 "file://", "http://", "https://" 及本地路径。
            - 'video_start' (optional): 视频读取起始时间点（秒）。
            - 'video_end' (optional): 视频读取结束时间点（秒）。
            - 'nframes' / 'fps': 帧数或帧率控制（传递给 smart_nframes）。

    返回：
        Tuple[torch.Tensor, Dict[str, Any], float]:
            1. video: 采样后的视频张量，形状为 (T, C, H, W)。
            2. video_metadata: 包含 fps, frames_indices, total_num_frames 等元数据的字典。
            3. sample_fps: 实际采样帧率。
    """
    video_path = ele["video"]
    
    # 1> 检查 torchvision 版本兼容性
    # torchvision < 0.19.0 不支持 HTTP/HTTPS 协议，需发出警告
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        # 如果是 file:// 协议，去除前缀
        if "file://" in video_path:
            video_path = video_path[7:]
            
    st = time.time()
    # 2> 读取视频文件
    # output_format="TCHW" 直接返回 (Time, Channel, Height, Width) 格式
    # pts_unit="sec" 指定时间单位为秒
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    # 3> 计算需要采样的帧数
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    # 4> 生成采样索引并执行采样
    # 使用 linspace 生成均匀分布的索引，round().long() 取整
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    # 计算实际的采样帧率
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    # 根据索引提取帧
    video = video[idx]

    # 5> 构建元数据并返回
    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchvision",
    )
    return video, video_metadata, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> Tuple[int, int, int]:
    """
    功能：
        根据给定的时间范围计算视频的起始帧索引、结束帧索引以及该范围内的总帧数。
        该函数常用于视频剪辑或局部采样场景，支持通过秒数指定开始和结束位置，并自动进行边界对齐。

    参数：
        ele (Dict[str, Any]): 包含视频配置的字典。
            - video_start (float, 可选): 视频开始时间（秒）。如果未提供，默认为 0.0。
            - video_end (float, 可选): 视频结束时间（秒）。如果未提供，默认为视频总长度。
        total_frames (int): 视频的总帧数。
        video_fps (float): 视频的帧率（FPS）。

    返回：
        Tuple[int, int, int]: 包含 (起始帧索引, 结束帧索引, 范围内的总帧数) 的元组。

    示例：
        >>> ele = {"video_start": 1.5, "video_end": 4.0}
        >>> start, end, count = calculate_video_frame_range(ele, total_frames=300, video_fps=30.0)
        >>> print(start, end, count)
        45 120 76
        >>> # 说明：1.5s * 30fps = 45帧; 4.0s * 30fps = 120帧; 120 - 45 + 1 = 76帧
    """
    # 1> 验证核心参数的有效性
    # 确保帧率和总帧数必须为正数，否则后续的时间计算将失去物理意义
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # 2> 获取配置中的起始和结束时间（单位：秒）
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)

    # 3> 处理未指定时间范围的情况
    # 如果起始和结束时间均未定义，则默认选取完整视频范围
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    # 4> 计算视频的最大时长，用于后续的时间边界裁剪（Clamping）
    max_duration = total_frames / video_fps

    # 5> 计算起始帧索引 (start_frame)
    if video_start is not None:
        # 将输入时间限制在 [0, 视频总时长] 范围内，防止索引越界
        video_start_clamped = max(0.0, min(video_start, max_duration))
        # 使用 ceil (向上取整) 确保从指定时间点后的第一帧开始
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        # 如果未指定开始时间，则从第 0 帧开始
        start_frame = 0

    # 6> 计算结束帧索引 (end_frame)
    if video_end is not None:
        # 同样对结束时间进行边界裁剪
        video_end_clamped = max(0.0, min(video_end, max_duration))
        # 使用 floor (向下取整) 确保不超过指定的时间截点
        end_frame = math.floor(video_end_clamped * video_fps)
        # NOTE: 保索引不会超过视频最后一帧的下标 (total_frames - 1)
        end_frame = min(end_frame, total_frames - 1)
    else:
        # 如果未指定结束时间，则默认为最后一帧
        end_frame = total_frames - 1

    # 7> 验证帧序列的逻辑顺序
    # 如果起始帧大于或等于结束帧，说明给定的时间区间无效（如 video_start > video_end）
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    # 8> 打印日志信息并返回计算结果
    # 返回值包括起始下标、结束下标以及实际包含的帧总数（闭区间计算：end - start + 1）
    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """
    功能：
        使用 decord 库作为后端高效读取视频文件，并根据配置进行采样。
        该函数支持从本地或远程路径（http/https）加载视频，并处理指定的时间范围和采样帧数。

    参数：
        ele (Dict[str, Any]): 包含视频配置的字典。
            - video (str): 视频文件的路径，支持 "file://", "http://", "https://" 或本地系统路径。
            - video_start (float, 可选): 视频开始时间（秒）。
            - video_end (float, 可选): 视频结束时间（秒）。
            - fps (float, 可选): 期望的采样频率。
            - nframes (int, 可选): 期望采样的总帧数。

    返回：
        Tuple[torch.Tensor, Dict[str, Any], float]:
            - video (torch.Tensor): 采样后的视频张量，形状为 (T, C, H, W)。
            - video_metadata (Dict[str, Any]): 包含视频元数据（如原始 FPS、采样索引等）的字典。
            - sample_fps (float): 实际采样的等效帧率。

    示例：
        >>> ele = {"video": "sample.mp4", "video_start": 2.0, "video_end": 5.0, "nframes": 10}
        >>> video, meta, fps = _read_video_decord(ele)
        >>> print(video.shape)
        torch.Size([10, 3, 1080, 1920])
    """
    # 1> 导入 decord 库并获取视频路径
    import decord
    video_path = ele["video"]
    st = time.time()  # 记录开始时间以统计耗时

    # 2> 初始化 VideoReader 以获取视频元数据
    # decord 会快速扫描文件头，获取总帧数和平均帧率
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    # 3> 计算有效的时间范围和对应的帧索引区间
    # 调用 calculate_video_frame_range 处理 video_start 和 video_end
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )

    # 4> 计算目标采样帧数
    # 根据用户配置（如 fps 或 nframes）决定在这个范围内抽取多少帧
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    # 5> 生成均匀分布的采样索引并执行读取
    # torch.linspace 在 [start_frame, end_frame] 之间生成 nframes 个均匀分布的点
    # round().long().tolist() 将浮点索引转换为整数列表，以便 decord 读取
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    # vr.get_batch(idx) 能够一次性高效提取多帧，返回形状为 (T, H, W, C) 的数据
    video = vr.get_batch(idx).asnumpy()

    # 6> 转换为 PyTorch 张量并调整维度顺序
    # shape: (T, H, W, C) -> (T, C, H, W)
    # T: 帧数, C: 通道(RGB), H: 高度, W: 宽度
    video = torch.tensor(video).permute(0, 3, 1, 2)
    
    # 7> 打印日志信息
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    # 8> 计算实际采样的等效帧率
    # 基于采样后的总帧数相对于原始范围长度的比例
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # 9> 构建元数据字典并返回结果
    video_metadata = dict(
        fps=video_fps,                 # 原始视频 FPS
        frames_indices=idx,            # 采样的帧下标列表
        total_num_frames=total_frames, # 指定范围内的总帧数
        video_backend="decord",        # 标记所选后端为 decord
    )
    return video, video_metadata, sample_fps


def is_torchcodec_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None


def _read_video_torchcodec(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """
    功能：
        使用 PyTorch 官方的 torchcodec 库作为后端读取视频文件。
        该后端基于 FFmpeg，旨在提供比传统库更快的解码速度和更好的硬件加速支持。

    参数：
        ele (Dict[str, Any]): 包含视频配置的字典。
            - video (str): 视频文件的路径（本地路径）。
            - video_start (float, 可选): 视频开始时间（秒）。
            - video_end (float, 可选): 视频结束时间（秒）。
            - fps (float, 可选): 期望的采样频率。
            - nframes (int, 可选): 期望采样的总帧数。

    返回：
        Tuple[torch.Tensor, Dict[str, Any], float]:
            - video (torch.Tensor): 解码并采样后的视频张量，形状通常为 (T, C, H, W)。
            - video_metadata (Dict[str, Any]): 包含视频元数据（如 FPS、采样索引、后端名称等）的字典。
            - sample_fps (float): 实际采样的等效帧率。

    示例：
        >>> ele = {"video": "demo.mp4", "fps": 1.0}
        >>> video, meta, fps = _read_video_torchcodec(ele)
        >>> print(meta["video_backend"])
        torchcodec
    """
    # 1> 导入 torchcodec 相关的解码器类
    from torchcodec.decoders import VideoDecoder

    # 2> 设置解码线程数
    # 从环境变量 TORCHCODEC_NUM_THREADS 获取线程数，默认为 8
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")

    # 3> 初始化解码器并记录元数据
    video_path = ele["video"]
    st = time.time()  # 记录开始时间以统计耗时
    # 创建 VideoDecoder 实例，指定 FFmpeg 线程数以平衡速度和资源消耗
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    # 4> 计算有效的时间范围和帧索引区间
    # 考虑 video_start 和 video_end 配置，确定实际需要解码的帧范围
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )

    # 5> 计算目标采样帧数
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    # 6> 生成均匀采样索引
    # 在指定范围内生成 nframes 个均匀分布的索引点
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()

    # 7> 计算实际采样的等效帧率
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # 8> 执行解码操作提取指定帧
    # decoder.get_frames_at 会直接根据索引返回对应的帧数据，通常返回的是 TCHW 格式的 Tensor
    video = decoder.get_frames_at(indices=idx).data

    # 9> 打印日志信息
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    # 10> 构建元数据并返回结果
    video_metadata = dict(
        fps=video_fps,                 # 视频原始帧率
        frames_indices=idx,            # 实际采样的帧索引列表
        total_num_frames=total_frames, # 有效范围内的总帧数
        video_backend="torchcodec",    # 指定后端名称
    )
    return video, video_metadata, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    """
    功能：
        自动检测并选择最合适的视频读取后端。
        该函数会根据库的可用性和环境变量配置，按优先级返回推荐的后端名称。
        使用 lru_cache 确保在同一次运行中只执行一次检测逻辑，提高效率。

    参数：
        无参数。

    返回：
        str: 选定的视频读取后端名称，可能的值包括 "torchcodec", "decord" 或 "torchvision"。

    示例：
        >>> backend = get_video_reader_backend()
        >>> print(f"Current backend: {backend}")
        qwen-vl-utils using torchcodec to read video.
        Current backend: torchcodec
    """
    # 1> 优先检查是否通过环境变量强制指定了后端
    # 用户可以通过设置 FORCE_QWENVL_VIDEO_READER 环境变量来覆盖自动检测逻辑
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    
    # 2> 尝试探测 torchcodec 是否可用（第一优先级）
    # torchcodec 是基于 FFmpeg 的最新高性能解码后端
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    
    # 3> 尝试探测 decord 是否可用（第二优先级）
    # decord 是广泛使用的视频处理库，具有良好的随机访问性能
    elif is_decord_available():
        video_reader_backend = "decord"
    
    # 4> 最终回退方案：使用 torchvision（第三优先级/兜底）
    # torchvision 通常作为深度学习环境的基础库，是性能最稳定但功能相对基础的备份
    else:
        video_reader_backend = "torchvision"

    # 5> 将选定的后端信息输出到标准错误流（sys.stderr），方便用户了解加载状态
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    
    # 6> 返回最终选定的后端名称
    return video_reader_backend


def fetch_video(
    ele: Dict[str, Any],
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float], Tuple[torch.Tensor, Dict[str, Any]], Tuple[Tuple[torch.Tensor, Dict[str, Any]], float]]:
    """
    功能：
        获取并预处理视频数据，支持从视频文件或帧列表加载。
        该函数会执行读取、采样、缩放（Resize）以及归一化准备工作，是视频理解流水线的核心环节。

    参数：
        ele (Dict[str, Any]): 包含视频信息的字典。
            - video (str | List): 视频路径或 PIL 图像帧列表。
            - video_start/video_end (float): 采样的时间区间。
            - fps/nframes (float/int): 采样频率或采样总帧数。
            - min_pixels/max_pixels/total_pixels (int): 控制缩放大小的像素限制。
        image_patch_size (int, optional): 图像补丁大小，默认为 14。
        return_video_sample_fps (bool, optional): 是否返回实际采样帧率，默认为 False。
        return_video_metadata (bool, optional): 是否返回视频元数据，默认为 False。

    返回：
        Union[torch.Tensor, Tuple]: 根据参数组合返回视频张量及可选的元数据和采样率。
            - 默认：返回 (T, C, H, W) 形状的 torch.Tensor。
            - 若 return_video_metadata 为 True，返回 (Tensor, Metadata_Dict)。
            - 若 return_video_sample_fps 为 True，则会在上述结果的基础上增加 sample_fps 构成元组。

    示例：
        >>> ele = {"video": "test.mp4", "nframes": 4}
        >>> video = fetch_video(ele)
        >>> print(video.shape[0])  # 帧数
        4
    """
    # 1> 初始化像素计算相关的常数
    # image_factor 用于确保缩放后的尺寸能被 Patch 大小整除
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    # 2> 根据输入类型选择读取逻辑
    if isinstance(ele["video"], str):
        # 场景 A: 输入是视频文件路径
        video_reader_backend = get_video_reader_backend()
        try:
            # 尝试使用首选后端读取
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            # 若失败，则回退到最通用的 torchvision 后端
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
    else:
        # 场景 B: 输入是已有的图像帧列表 (PIL.Image 对象列表)
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)

        # 3> 并行处理图像帧
        # 使用线程池加速 PIL 图像到 Tensor 的转换及初步处理
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_image, {"image": video_element, **process_info}, image_factor)
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        # 4> 确保帧数符合因子要求（如 FRAME_FACTOR 补齐）
        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            # 如果帧数不足，使用最后一帧进行填充
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        # 5> 构建视频张量
        sample_fps = ele.get("sample_fps", 2.0)
        # 将 PIL 列表转换为 (T, C, H, W) 形状的 Tensor
        video = torch.stack([
            torch.from_numpy(np.array(image).transpose(2, 0, 1))
            for image in image_list
        ])

        # 6> 伪造元数据以保持接口一致性
        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=[i for i in range(len(video))],
            total_num_frames=(nframes / sample_fps) * raw_fps,
        )

    # 7> 动态计算 Resize 的目标尺寸
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    # total_pixels 是基于模型序列长度和帧数计算出的全局上限
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    # 计算单帧允许的最大像素，兼顾安全性限制和总 Token 约束
    max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)

    # 8> 执行 Resize 逻辑
    if "resized_height" in ele and "resized_width" in ele:
        # 如果用户指定了具体宽高
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        # 否则根据像素约束自动计算最优宽高
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    # 9> 执行图像重采样 (Resampling)
    # 使用双三次插值 (Bicubic) 并开启抗锯齿，转换为浮点型以供模型输入
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    # 10> 根据配置组装返回值
    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def extract_vision_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """
    功能：
        从对话历史中提取所有包含多媒体信息（图像或视频）的内容项。
        该函数支持单条对话或多条对话批量处理，并能够识别多种格式定义的视觉输入。

    参数：
        conversations (Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]): 对话数据。
            - 可以是单条对话列表：[{"role": "user", "content": [...]}, ...]
            - 也可以是多条对话组成的批量列表：[[{"role": "user", "content": [...]}, ...], ...]

    返回：
        List[Dict[str, Any]]: 提取出的所有视觉信息项（字典）列表。

    示例：
        >>> conv = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"image": "a.jpg"}]}]
        >>> info = extract_vision_info(conv)
        >>> print(len(info))
        1
        >>> print(info[0]["image"])
        a.jpg
    """
    # 1> 初始化用于存储提取结果的列表
    vision_infos = []

    # 2> 统一数据格式
    # 如果输入是单条对话（即列表的第一个元素是字典），则将其包装成批量格式以便后续统一循环处理
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    # 3> 遍历对话结构提取信息
    for conversation in conversations:
        # 遍历对话中的每一条消息（通常包含 role 和 content）
        for message in conversation:
            # 只处理内容（content）为列表格式的消息
            if isinstance(message["content"], list):
                # 遍历消息内容中的每一个具体元素（元素可以是文本、图像、视频等）
                for ele in message["content"]:
                    # 4> 识别并过滤视觉元素
                    # 检查元素是否显式包含 "image", "image_url", "video" 键
                    # 或者其 "type" 字段明确标记为视觉类型
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        # 将符合条件的视觉信息项存入结果列表
                        vision_infos.append(ele)

    # 5> 返回所有提取到的视觉信息
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:
    """
    功能：
        统一处理对话中的所有视觉信息（图像和视频），将其转换为模型可直接使用的输入格式。
        该函数是多模态数据预处理的顶层接口，负责协调图像抓取、视频读取、缩放及元数据管理。

    参数：
        conversations (Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]): 原始对话数据。
        return_video_kwargs (bool, optional): 是否返回包含采样帧率等信息的字典，默认为 False。
        return_video_metadata (bool, optional): 是否让 fetch_video 返回视频元数据，默认为 False。
        image_patch_size (int, optional): 图像补丁大小，影响缩放后的对齐尺寸，默认为 14。

    返回：
        Tuple[Optional[List[Image.Image]], Optional[List[Tensor/List]], Optional[Dict]]:
            - image_inputs: 处理后的 PIL 图像列表，如果没有图像则为 None。
            - video_inputs: 处理后的视频张量列表，如果没有视频则为 None。
            - video_kwargs: (可选) 包含 'do_sample_frames' 和 'fps' 列表的字典。

    示例：
        >>> conv = [{"role": "user", "content": [{"image": "dog.png"}, {"text": "What is this?"}]}]
        >>> images, videos, _ = process_vision_info(conv)
        >>> print(len(images))
        1
    """
    # 1> 从对话中提取所有视觉信息项
    # 调用 extract_vision_info 过滤出包含 image 或 video 的元素
    vision_infos = extract_vision_info(conversations)

    # 2> 初始化数据容器
    image_inputs = []          # 存储处理后的图像
    video_inputs = []          # 存储处理后的视频张量
    video_sample_fps_list = [] # 记录每个视频的实际采样帧率

    # 3> 遍历并处理每个视觉项
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            # 处理图像：调用 fetch_image 将路径/URL 转换为 PIL 图像并执行初步 Resize
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            # 处理视频：调用 fetch_video 执行读取、采样和 Resize
            # 根据配置决定是否返回采样率和元数据
            video_input, video_sample_fps = fetch_video(
                vision_info, 
                return_video_sample_fps=True,
                image_patch_size=image_patch_size, 
                return_video_metadata=return_video_metadata
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            # 异常处理：如果 vision_info 中不包含预期的视觉键值
            raise ValueError("image, image_url or video should in content.")

    # 4> 边界情况处理：如果没有处理到任何数据，将列表置为 None 以符合返回类型注解
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    # 5> 构建视频相关的附加参数字典
    # 'do_sample_frames': False 表示模型端不需要再进行额外的二次采样
    video_kwargs = {'do_sample_frames': False}
    # 向后兼容处理：如果不需要完整元数据，则将采样 FPS 列表放入字典（用于 Qwen2.5-VL 等模型）
    if not return_video_metadata:
        video_kwargs.update({'fps': video_sample_fps_list})

    # 6> 根据配置返回结果
    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs