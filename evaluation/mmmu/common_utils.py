import os
import requests
import base64
import hashlib
import io
from PIL import Image
from typing import List, Union


def encode_image_to_base64(image, target_size=None) -> str:
    """
    功能：
        将 PIL 图像对象转换为 Base64 编码的字符串，支持可选的等比例缩放。
        常用于将图像数据嵌入 JSON 请求或在 Web 传输中直接传递二进制数据。

    参数：
        image (PIL.Image.Image): 需要处理的 PIL 图像对象。
        target_size (int, optional): 目标尺寸（长边长度）。如果指定，将对图像进行等比例缩放，
                                     使图像的长边等于该值。默认为 None，即不缩放。

    返回：
        str: 编码后的 Base64 字符串，格式为纯文本。

    示例：
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 200), color='red')
        >>> b64_str = encode_image_to_base64(img, target_size=128)
        >>> print(b64_str[:10])
        /9j/4AAQSk
    """
    # 1> 检查是否需要对图像进行缩放处理
    if target_size is not None:
        # 获取原始图像的宽度和高度
        width, height = image.size
        
        # 2> 计算缩放后的尺寸，同时保持原始宽高比
        if width > height:
            # 如果宽度是长边，则将宽度设为目标尺寸，并按比例计算高度
            new_width = target_size
            new_height = int(height * new_width / width)
        else:
            # 如果高度是长边（或宽高相等），则将高度设为目标尺寸，并按比例计算宽度
            new_height = target_size
            new_width = int(width * new_height / height)
        
        # 3> 执行缩放操作
        # 使用 resize 方法更新 image 对象
        image = image.resize((new_width, new_height))
    
    # 4> 将图像保存到内存缓冲区
    # 使用 io.BytesIO 创建一个二进制流对象，模拟文件操作
    buffer = io.BytesIO()
    # 以 JPEG 格式将图像数据写入缓冲区
    image.save(buffer, format="JPEG")
    
    # 5> 执行 Base64 编码并转换为字符串
    # getvalue() 获取缓冲区的全部二进制数据
    # base64.b64encode 执行编码，得到 bytes 类型
    # .decode('utf-8') 将编码后的 bytes 转换为标准的 Python 字符串
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_base64_to_image(base64_string) -> Image.Image:
    """Decode a base64 string to an image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def decode_base64_to_image_file(base64_string, output_path):
    """Decode a base64 string and save it to a file."""
    image = decode_base64_to_image(base64_string)
    image.save(output_path)

def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def md5(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError