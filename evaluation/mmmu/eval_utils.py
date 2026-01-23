import os
import requests
import time
import random
import string
import copy
import traceback
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Any
from common_utils import encode_image_to_base64


class OpenAIWrapper:
    """OpenAI API 访问包装类
    
    类功能：
        该类用于封装对 OpenAI 兼容接口的 HTTP 调用，集成了多模态消息格式化、图像 Base64 编码处理以及自动重试机制。
    
    继承关系：
        独立类，无显式父类。
    
    应用场景：
        1. 在评估脚本中调用部署在 vLLM 或 OpenAI 上的视觉语言模型。
        2. 处理包含文本和本地图片路径的混合输入，并将其转换为标准的 API 请求格式。
    
    使用示例：
        >>> wrapper = OpenAIWrapper(
        ...     model="gpt-4o",
        ...     api_base="https://api.openai.com/v1/chat/completions",
        ...     api_key="sk-xxx",
        ...     timeout=60,
        ...     retry=3
        ... )
    
    数据属性：
        model: str
            目标模型的名称或标识符（如 "gpt-4-vision-preview"）。
        
        api_base: str
            API 的基础请求 URL 地址。
        
        api_key: str
            用于身份验证的 API 密钥。
        
        timeout: int
            单次 HTTP 请求的超时时间（秒），默认为 60。
        
        retry: int
            请求失败时的最大重试次数，默认为 5。
        
        wait: int
            两次重试之间的等待间隔时间（秒），默认为 5。
        
        fail_msg: str
            当所有重试尝试均失败后返回的固定错误提示字符串。
    """
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        """初始化 OpenAI 访问配置
        
        功能：
            设置 API 访问所需的模型参数、端点、密钥以及重试策略。
            
        参数：
            model (str): 模型 ID。
            api_base (str): 接口 URL。
            api_key (str): 授权密钥。
            timeout (int, 可选): 超时秒数，默认 60。
            retry (int, 可选): 重试次数，默认 5。
            wait (int, 可选): 重试间隔，默认 5。
            
        返回：
            None
            
        示例：
            >>> wrapper = OpenAIWrapper("model-id", "http://url", "key")
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """生成模型响应内容
        
        功能：
            1. 将原始消息列表（包含文本和图像路径）转换为 OpenAI 兼容的 JSON 格式。
            2. 自动读取并对本地图片进行 Base64 编码。
            3. 执行带有指数退避（此处为固定间隔）重试逻辑的 POST 请求。
            4. 解析并提取 API 返回的文本结果。

        参数：
            messages (List[Dict]): 消息列表，每个字典需包含 'type' ("text" 或 "image") 和 'value' (文本内容或图片本地路径)。

        返回：
            str: 模型生成的文本回答，如果失败则返回预定义的失败信息。

        示例：
            >>> msgs = [{"type": "text", "value": "这张图里有什么？"}, {"type": "image", "value": "cat.jpg"}]
            >>> result = wrapper.generate(msgs)
        """
        # 1> 构造 HTTP 请求头，包含 Content-Type 和 Bearer Token 认证
        # bearer: 持有者 / 携带者 / 承载者
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}

        # 2> 转换消息格式为 API 要求的结构
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                # 处理纯文本消息
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # 3> 处理图像消息：读取图片 -> Base64 编码 -> 封装为 Data URL
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        # 4> 组装请求负载 (Payload)
        # 固定参数：max_tokens=4096, temperature=0 以获得确定性结果
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 4096,
            "temperature": 0
        }

        # 5> 开始执行请求并包含重试逻辑
        for i in range(self.retry):
            try:
                # 6> 发送 POST 请求到 API 节点
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # 7> 检查响应状态码，200 表示成功
                if response.status_code == 200:
                    resp_json = response.json()
                    # 提取第一条回复的内容并去除首尾空格
                    return resp_json['choices'][0]['message']['content'].strip()
                
                # 若状态码非 200，记录日志或等待重试
                time.sleep(self.wait)
            except Exception as e:
                # 捕捉网络异常、连接超时等错误
                print(f"API error: {e}")
                time.sleep(self.wait)

        # 8> 如果所有重试都已耗尽仍未成功，返回错误占位符
        return self.fail_msg

class DashScopeWrapper:
    """DashScope (灵积) API 访问包装类
    
    背景说明：
        DashScope 是阿里云推出的大模型统一调用与管理平台（大模型 API 平台）。
        它主要用于 调用、管理和部署各类大模型能力，包括文本、多模态等。

    类功能：
        该类用于封装对阿里云 DashScope 兼容接口的 HTTP 调用，专门针对通义千问等模型的消息格式和响应规范进行处理。
    
    继承关系：
        独立类，无显式父类。
    
    应用场景：
        1. 在评估脚本中调用部署在阿里云 DashScope 平台上的多模态模型（如 Qwen-VL）。
        2. 将包含图像和文本的混合输入格式化为 DashScope 要求的 payload。
    
    使用示例：
        >>> wrapper = DashScopeWrapper(
        ...     model="qwen-vl-plus",
        ...     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ...     api_key="sk-xxx",
        ...     timeout=120,
        ...     retry=5
        ... )
    
    数据属性：
        model: str
            目标模型的名称（如 "qwen-vl-max" 或 "qwen-vl-plus"）。
        
        api_base: str
            DashScope API 的基础请求 URL 地址。
        
        api_key: str
            阿里云 DashScope 的 API 密钥。
        
        timeout: int
            单次 HTTP 请求的超时时间（秒），由于图像处理较慢，建议设置较大的值。
        
        retry: int
            请求失败或由于并发限制导致失败时的最大重试次数。
        
        wait: int
            两次重试之间的等待间隔时间（秒）。
        
        fail_msg: str
            当所有重试尝试均失败后返回的固定错误提示。
    """
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        """初始化 DashScope 访问配置
        
        功能：
            设置 DashScope API 访问所需的各项参数，包括模型 ID、访问端点及重试策略。
            
        参数：
            model (str): 模型 ID。
            api_base (str): 接口 URL。
            api_key (str): 授权密钥。
            timeout (int, 可选): 超时秒数，默认 60。
            retry (int, 可选): 重试次数，默认 5。
            wait (int, 可选): 重试间隔，默认 5。
            
        返回：
            None
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """生成 DashScope 模型响应
        
        功能：
            1. 构建符合 DashScope 规范的 HTTP 请求头（使用 Bearer Token 认证）。
            2. 遍历并转换输入消息：读取本地图片、转换为 Base64 编码并组装成 Data URL 格式。
            3. 配置请求负载：包含特定的 max_completion_tokens 和 temperature 参数。
            4. 执行带有异常捕获和重试逻辑的 POST 请求，并对响应状态和完成原因进行检查。

        参数：
            messages (List[Dict]): 消息列表，每个条目应包含 'type' ("text" 或 "image") 和 'value' (内容)。

        返回：
            str: 模型生成的文本回答，若失败则返回预定义的错误提示信息。

        示例：
            >>> msgs = [{"type": "text", "value": "描述这张图片内容"}, {"type": "image", "value": "demo.png"}]
            >>> result = wrapper.generate(msgs)
        """
        # 1> 设置请求头，采用 JSON 格式和 Bearer Token 认证方式
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}

        # 2> 转换消息列表为 API 兼容的格式
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                # 文本类型处理
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # 图像类型处理：读取图片、Base64 编码、组装 Data URL
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        # 3> 组装 API 请求负载
        # 注意：DashScope 使用 max_completion_tokens 作为参数名
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_completion_tokens": 4096,
            "n": 1,
            "temperature": 0,
            "stream": False
        }

        # 4> 执行重试循环逻辑
        for i in range(self.retry):
            try:
                # 5> 发送网络请求
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # 6> 处理响应状态码 200 (成功)
                if response.status_code == 200:
                    resp_json = response.json()
                    
                    # 7> 检查生成完成原因 (finish_reason)
                    # 确保回答没有因为非预期原因（如长度超限或敏感内容）被截断
                    for output in resp_json['choices']:
                        if output['finish_reason'] not in ['stop', 'function_call']:
                            print(f"DashScope finished with error: {resp_json}")
                            time.sleep(self.wait)
                            continue
                    
                    # 返回模型生成的文本结果
                    return resp_json['choices'][0]['message']['content']
                else:
                    # 8> 处理 HTTP 错误状态码，尝试打印具体的 API 报错信息
                    print(f"DashScope API error: HTTP {response.status_code}")
                    try:
                        error_content = response.json()
                        print(f"Error details: {error_content}")
                    except:
                        # 如果无法解析为 JSON，则以文本形式打印原始响应
                        print(f"Raw error content: {response.content.decode('utf-8', errors='replace')}")
                
                # 等待重试间隔
                time.sleep(self.wait)
            except requests.exceptions.ConnectionError as conn_err:
                # 连接异常捕获
                print(f"DashScope: Connection error occurred: {conn_err}")
                time.sleep(self.wait)
            except requests.exceptions.Timeout as timeout_err:
                # 超时异常捕获
                print(f"DashScope: Timeout error occurred: {timeout_err}")
                time.sleep(self.wait)
            except requests.exceptions.RequestException as req_err:
                # 通用请求异常捕获
                print(f"DashScope: Request exception occurred: {req_err}")
                time.sleep(self.wait)
            except Exception as e:
                # 其他未知异常捕获及堆栈追踪打印
                print(f"DashScope: An error occurred: {e}")
                print(traceback.format_exc())
                time.sleep(self.wait)
        
        # 9> 若重试全部失败，返回默认失败提示
        return self.fail_msg

def build_judge(model, api_type):
    """
    功能：
        根据指定的 API 类型和模型名称构建对应的评测模型（Judge Model）实例。
        该函数负责从环境变量中自动读取 API 密钥和基础 URL，并根据类型实例化相应的 API 包装类。

    参数：
        model (str): 目标模型的标识符（如 "gpt-4o" 或 "qwen-vl-plus"）。
        api_type (str): 访问接口的类型。目前支持：
                        - 'mit': 对应 OpenAI 兼容格式接口。
                        - 'dash': 对应阿里云 DashScope (灵积) 接口。

    返回：
        Union[OpenAIWrapper, DashScopeWrapper]: 构造好的模型包装类对象。

    示例：
        >>> # 构建一个基于 OpenAI 兼容接口的评测器
        >>> judge = build_judge("gpt-4o", "mit")
        >>> # 构建一个基于 DashScope 接口的评测器
        >>> judge = build_judge("qwen-vl-max", "dash")
    """
    # 1> 逻辑分支：处理 MIT 类型的 API (通常用于 OpenAI 兼容的第三方转发或官方接口)
    if api_type == 'mit':
        # 从环境变量中获取预定义的 Token 和接口地址
        api_key = os.environ.get('MIT_SPIDER_TOKEN', '')
        api_base = os.environ.get('MIT_SPIDER_URL', '')
        # 实例化并返回 OpenAI 包装类
        return OpenAIWrapper(model, api_base, api_key)
    
    # 2> 逻辑分支：处理 Dash 类型的 API (专用于阿里云 DashScope 平台)
    elif api_type == 'dash':
        # 从环境变量中获取 DashScope 专属的 API Key 和端点地址
        api_key = os.environ.get('CHATGPT_DASHSCOPE_API_KEY', '')
        api_base = os.environ.get('DASHSCOPE_API_BASE', '')
        # 实例化并返回 DashScope 包装类
        return DashScopeWrapper(model, api_base, api_key)
    
    # 3> 容错处理：如果传入了未定义的 API 类型，则抛出 ValueError
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def can_infer_option(answer, choices):
    """
    功能：
        基于规则从模型的文本回答中提取多选题的选项（如 'A', 'B', 'C'）。
        该方法通过清洗标点、分词并匹配候选选项，试图识别模型最终选择的答案标签。

    参数：
        answer (str): 模型生成的原始文本回答。
        choices (Iterable[str]): 候选选项列表，通常为 ['A', 'B', 'C', 'D']。

    返回：
        Union[str, bool]: 
            - 如果成功提取到唯一选项，返回该选项字符（如 'A'）。
            - 如果模型拒绝回答，返回 'Z'。
            - 如果提取失败或存在歧义，返回 False。

    示例：
        >>> choices = ['A', 'B', 'C', 'D']
        >>> can_infer_option("The answer is A.", choices)
        'A'
        >>> can_infer_option("I am sorry, I cannot see the image.", choices)
        'Z'
    """
    # 1> 检查是否为 API 调用失败的提示
    if 'Failed to obtain answer via API' in answer:
        return False

    # 2> 定义拒绝回答的常见关键词/短语
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    # 如果回答中包含拒绝短语，统一返回 'Z'（表示无效/无法回答）
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    # 3> 内部辅助函数：统计分词结果中匹配候选选项的个数
    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    # 4> 对原始文本进行预处理：将标点符号替换为空格以便分词
    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')
    
    # 5> 将清洗后的文本按空格分割成单词列表
    splits = [x.strip() for x in answer_mod.split()]

    # 6> 统计分词列表中出现了多少个候选选项标签
    count = count_choice(splits, choices)

    # 7> 核心判断逻辑：如果只出现了一个选项
    if count == 1:
        for ch in choices:
            # 特殊规则：如果提取到的是 'A'，但句子较长（splits > 3），
            # 那么 'A' 极有可能是英文不定冠词（a cat），为了保险起见返回 False。
            if 'A' in splits and len(splits) > 3:
                # print(f'A might be a quantifier in the string: {answer}.')
                return False
            # 返回唯一匹配到的选项
            if ch in splits:
                return ch
    # 8> 如果未找到常规选项，但匹配到了预设的 'Z' 标识
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    
    # 9> 其他情况（多选、错选或未提取到）均视为提取失败
    return False


def can_infer_text(answer, choices):
    """
    功能：
        通过直接匹配选项的文本内容来推断模型选择的答案。
        当选项标签（如 'A'）未在回答中出现，但回答中包含某个选项的具体描述文本时，该方法非常有用。

    参数：
        answer (str): 模型生成的原始文本回答。
        choices (Dict[str, Any]): 选项字典，键为选项标签（大写字母），值为选项对应的描述文本。

    返回：
        Union[str, bool]: 
            - 如果在回答中匹配到且仅匹配到一个选项的描述文本，返回该选项标签。
            - 否则返回 False。

    示例：
        >>> choices = {'A': 'Cat', 'B': 'Dog'}
        >>> can_infer_text("The picture shows a small cat.", choices)
        'A'
        >>> can_infer_text("I can see both a cat and a dog.", choices)
        False
    """
    # 1> 将回答内容统一转换为小写，以实现大小写无关的匹配
    answer = answer.lower()
    
    # 2> 验证输入合法性并预处理选项文本
    assert isinstance(choices, dict)
    for k in choices:
        # 确保字典键为大写英文字母
        assert k in string.ascii_uppercase
        # 将选项的具体描述文本也转换为小写
        choices[k] = str(choices[k]).lower()

    # 3> 遍历所有选项，检查其描述文本是否出现在回答中
    cands = []
    for k in choices:
        if choices[k] in answer:
            # 如果匹配成功，将对应的选项标签（键）加入候选列表
            cands.append(k)
    
    # 4> 核心判断逻辑：只有当匹配到的候选选项唯一时，才认为推断成功
    if len(cands) == 1:
        return cands[0]
    
    # 5> 如果匹配到多个选项或一个都没匹配到，则返回失败
    return False

def can_infer(answer, choices):
    """Combined approach to infer answer choice."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret

def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s

def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

def extract_answer_from_item(model, item, wait=5):
    """
    功能：
        从模型的原始预测结果中提取最终的选项答案。
        该方法采用“先规则、后模型”的混合策略：首先尝试使用正则表达式或字符串匹配进行规则提取，
        如果失败，则调用指定的“评测模型”（Judge Model）辅助进行语义提取。

    参数：
        model (OpenAIWrapper | DashScopeWrapper): 用于辅助提取答案的评测模型实例。
        item (Dict[str, Any]): 包含数据条目信息的字典，需包含 'question'、'prediction'（模型原始输出）
                               以及各个选项（'A', 'B' 等）。
        wait (int, 可选): 接口调用失败或提取失败时的平均等待时间（秒），用于重试逻辑。

    返回：
        Dict[str, Any]: 包含以下字段的字典：
            - opt (str): 提取出的最终选项（如 'A', 'B' 等）。
            - log (str): 处理过程中的日志信息。
            - extract_model (str): 标识最终完成提取的方法（'rule' 或模型名称）。
            - extract_flag (bool): 标识是否成功从内容中提取出有效答案。

    示例：
        >>> item = {'question': '...', 'prediction': 'The answer is A', 'A': '...', 'B': '...'}
        >>> result = extract_answer_from_item(judge_model, item)
        >>> print(result['opt'])
        'A'
    """
    # 1> 预处理选项信息并构建用于模型辅助提取的 Prompt
    # 提取 item 中存在的有效选项（A, B, C...）
    choices = build_choices(item)
    # 将选项转换为字符串格式（如 "A. cat B. dog"）
    option_str = build_option_str(choices)
    # 构建最终发给评测模型的提示词，包含原问题、选项和模型预测结果
    prompt = build_prompt(item['question'], option_str, item['prediction'])

    # 2> 第一阶段：尝试基于规则的快速提取
    prediction = item['prediction']
    # 使用 can_infer 尝试从原始预测文本中解析选项
    ret = can_infer(prediction, choices)
    
    # 3> 如果规则提取成功，直接返回结果
    if ret:
        if ret == 'Z':
            # 'Z' 通常表示规则判断模型拒绝回答或无法匹配任何选项
            extract_flag = False
            log = f"Rule extract failed with rule result: {ret} prediction: {prediction}"
        else:
            extract_flag = True
            log = f"Rule extract success with rule result: {ret} prediction: {prediction}"
        return dict(opt=ret, log=log, extract_model='rule', extract_flag=extract_flag)
    
    # 4> 第二阶段：规则提取失败，进入模型辅助提取逻辑
    print(f"Rule extract failed. Use model-based extraction.")
    if model is None:
       assert model is not None, 'Judge model is None for MMMU_DEV_VAL !!!'

    # 5> 设置最大重试次数，循环调用评测模型进行解析
    retry = 25
    while retry:
        # 调用 API 生成提取结果
        ans = model.generate([{"type": "text", "value": prompt}])

        # 6> 检查 API 返回是否有效
        if 'Failed to obtain answer via API' in ans:
            print('API failed to answer.')
        else:
            # 尝试从评测模型的回答中再次进行规则提取
            ret = can_infer(ans, choices)
            # 如果模型给出了明确的选项且不是 'Z'，则视为提取成功
            if ret and ret != 'Z':
                log = f'{model.model} extract Succeed. {model.model}:{ans}\n'
                return dict(opt=ret, log=log, extract_model=model.model, extract_flag=True)
            else:
                # 打印解析失败的情况，可能包含多个字母或无法识别的内容
                print(f'Output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        
        # 7> 递减重试次数并进行随机退避等待，避免高频请求导致的限制
        retry -= 1
        # random.random() 用于生成一个 [0.0, 1.0) 区间内的随机浮点数
        T = random.random() * wait * 2
        time.sleep(T)

        # 8> 如果达到最大重试次数仍未成功提取，则进行兜底处理
        if retry == 0:
            # 在所有有效选项和 'Z' 中随机选择一个，作为最后的容错
            options = list(choices) + ['Z'] if 'Z' not in choices else list(choices)
            log = f'{model.model} extract failed. randomly generate one. {model.model} response:{ans}\n'
            return dict(opt=random.choice(options), log=log, extract_model=model.model, extract_flag=False)

def eval_single_sample(args):
    """Evaluate a single sample."""
    model, item = args
        
    # Extract answer using the combined approach
    result = extract_answer_from_item(model, item)
    
    # Determine if the answer is correct
    hit = 1 if result['opt'] == item['GT'] else 0
    
    return {
        "index": item['index'],
        "split": item['split'],
        "question": item['question'],
        "prediction": item['prediction'],
        "extracted_answer": result['opt'],
        "extraction_method": result['extract_model'],
        "extraction_success": result['extract_flag'],
        "extraction_log": result['log'],
        "gt": item['GT'],
        "hit": hit
    }