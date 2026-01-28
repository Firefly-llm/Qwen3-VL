from .util import *
from abc import abstractmethod


class BaseModel:
    """多模态模型抽象基类，统一输入规范化与生成/对话流程。

    继承关系：
        无显式继承关系；作为基类被具体模型类继承并实现抽象接口。

    应用场景：
        1. 评测管线中对多模态输入进行统一预处理与校验。
        2. API/本地模型封装中复用生成与多轮对话入口逻辑。

    使用示例：
        >>> model = MyModel()
        >>> model.set_dump_image(lambda line: "/tmp/sample.png")
        >>> result = model.generate("请描述这张图", dataset="mmmu")

        >>> model = MyModel()
        >>> messages = [{"role": "user", "content": "你好"}]
        >>> reply = model.chat(messages, dataset="mmmu")

    数据属性：
        INTERLEAVE: bool
            是否允许图文交错输入。默认 False，表示按内容块处理，不保证交错顺序被保留。
            约束：由子类按需覆盖；仅影响上层调用策略，不在本类内强制校验。

        allowed_types: List[str]
            允许的内容类型白名单，用于 `generate` 入参校验。
            默认值为 ['text', 'image', 'video']。
            约束：不能为空；新增类型需确保 `parse_file` 能识别并在下游支持。

        dump_image_func: Optional[Callable]
            将样本行转换为图片文件路径的回调函数。
            默认 None，需通过 `set_dump_image` 设置后再调用 `dump_image`。
            约束：允许为 None，但调用 `dump_image` 前必须设置，否则会触发异常。
    """

    INTERLEAVE = False
    allowed_types = ['text', 'image', 'video']

    def __init__(self):
        """初始化基类运行时状态。

        功能：
            初始化可选的图片导出回调函数，占位等待外部注入。

        参数：
            无。

        返回：
            None。

        示例：
            >>> model = BaseModel()
            >>> model.dump_image_func is None
            True
        """
        # 初始化图片导出回调，默认未设置
        self.dump_image_func = None

    def use_custom_prompt(self, dataset):
        """判断指定数据集是否启用自定义提示词构建。

        功能：
            根据数据集名称决定是否走 `build_prompt` 的自定义构建路径。
            默认返回 False，表示使用通用提示词策略。

        参数：
            dataset (str): 数据集名称，用于区分提示词策略。

        返回：
            bool: 是否启用自定义提示词。为 True 时将调用 `build_prompt`。

        示例：
            >>> model = BaseModel()
            >>> model.use_custom_prompt("mmmu")
            False
        """
        # 1> 默认不启用自定义提示词
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        """构建指定数据集的自定义提示词。

        功能：
            当 `use_custom_prompt` 返回 True 时被调用，用于把原始样本行转换为模型输入。

        参数：
            line: pandas DataFrame 的一行，包含原始样本字段。
            dataset (str): 数据集名称，用于选择提示词模板。

        返回：
            str: 构建后的文本提示词。

        示例：
            >>> prompt = model.build_prompt(line={"question": "Q?"}, dataset="mmmu")
            >>> isinstance(prompt, str)
            True
        """
        # 1> 由子类实现具体构建逻辑
        raise NotImplementedError

    def set_dump_image(self, dump_image_func):
        """设置图片导出回调函数。

        功能：
            注入将样本行转换为图片文件路径的函数，供 `dump_image` 调用。

        参数：
            dump_image_func (Callable): 接受样本行并返回图片路径的函数。

        返回：
            None。

        示例：
            >>> model.set_dump_image(lambda line: "/tmp/a.png")
        """
        # 1> 保存外部注入的图片导出函数
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        """根据样本行导出图片并返回路径。

        功能：
            调用已注入的 `dump_image_func` 生成图片路径。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称（当前未使用，保留扩展）。

        返回：
            str: 图片文件路径。

        示例：
            >>> model.set_dump_image(lambda line: "/tmp/a.png")
            >>> path = model.dump_image(line={"id": 1}, dataset="mmmu")
        """
        # 1> 直接调用回调函数并返回路径
        return self.dump_image_func(line)

    @abstractmethod
    def generate_inner(self, message, dataset=None):
        """子类实现的生成逻辑入口。

        功能：
            接收预处理后的 message 列表，执行模型推理并返回文本结果。

        参数：
            message (list[dict]): 已规范化的输入内容列表。
            dataset (str, optional): 数据集名称，用于子类策略分支。

        返回：
            str: 生成的文本结果。

        示例：
            >>> output = model.generate_inner([{"type": "text", "value": "你好"}])
        """
        # 1> 由子类实现具体生成流程
        raise NotImplementedError

    def check_content(self, msgs):
        """判断输入消息的结构类型。

        功能：
            对输入进行结构检查并归类为 str、dict、liststr 或 listdict。
            当输入不满足预期结构时返回 'unknown'。

        参数：
            msgs: 原始输入消息，可为字符串、字典或列表。

        返回：
            str: 结构类型标识（'str'/'dict'/'liststr'/'listdict'/'unknown'）。

        示例：
            >>> model.check_content("hello")
            'str'
            >>> model.check_content([{"type": "text", "value": "hi"}])
            'listdict'
        """
        # 1> 单条字符串消息
        if isinstance(msgs, str):
            return 'str'
        # 2> 单条结构化消息
        if isinstance(msgs, dict):
            return 'dict'
        # 3> 列表消息，递归检查每个元素
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]  # 对列表元素逐个判型
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        # 4> 其他结构视为未知
        return 'unknown'

    def preproc_content(self, inputs):
        """将原始输入消息规范化为统一的字典列表结构。

        功能：
            1> 识别输入类型（字符串/字典/列表）。
            2> 对文本与文件路径进行解析，统一转换为 `{'type', 'value'}` 结构。
            3> 校验类型一致性并对可解析文件路径进行归一化。

        参数：
            inputs: 原始输入消息，支持 str、dict、list[str]、list[dict]。

        返回：
            list(dict) | None: 规范化后的消息列表；若无法解析则返回 None。

        示例：
            >>> model.preproc_content("hi")
            [{'type': 'text', 'value': 'hi'}]
            >>> model.preproc_content(["/tmp/a.png", "hello"])
            [{'type': 'image', 'value': '/tmp/a.png'}, {'type': 'text', 'value': 'hello'}]
        """
        # 1> 输入为单条文本时，直接封装为文本类型
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        # 2> 输入为单条字典时，校验结构并包裹为列表
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs  # 必须包含 type/value 键
            return [inputs]
        # 3> 输入为字符串列表时，逐个解析文件类型或文本
        elif self.check_content(inputs) == 'liststr':
            res = []  # 保存规范化后的结果
            for s in inputs:
                mime, pth = parse_file(s)  # 解析文件类型与规范化路径
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))  # 无法解析则视为文本
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))  # 取主类型作为 type
            return res
        # 4> 输入为字典列表时，校验并归一化 value
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item  # 每项必须含 type/value
                mime, s = parse_file(item['value'])  # 解析 value 是否为文件路径
                if mime is None:
                    assert item['type'] == 'text'  # 无 mime 只能是文本类型
                else:
                    assert mime.split('/')[0] == item['type']  # 文件主类型需匹配
                    item['value'] = s  # 使用解析后的规范化路径
            return inputs
        # 5> 其他结构无法处理
        else:
            return None

    def generate(self, message, dataset=None):
        """生成单轮回复的统一入口。

        功能：
            1> 校验输入结构类型。
            2> 规范化内容为 list[dict]。
            3> 按允许类型白名单进行逐项校验。
            4> 调用子类 `generate_inner` 完成推理。

        参数：
            message: 原始输入消息。
            dataset (str, optional): 数据集名称，用于子类策略。

        返回：
            str: 模型生成的文本结果。

        示例：
            >>> model.generate("你好", dataset="mmmu")
            '...'
            >>> model.generate([{"type": "text", "value": "hi"}])
            '...'
        """
        # 1> 校验输入结构类型是否符合约定
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        # 2> 进行输入规范化，统一为 listdict
        message = self.preproc_content(message)
        # 3> 确保预处理成功且类型为 listdict
        assert message is not None and self.check_content(message) == 'listdict'
        # 4> 按白名单检查每一条内容的类型
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
        # 5> 调用子类实现的生成逻辑
        return self.generate_inner(message, dataset)

    def chat(self, messages, dataset=None):
        """多轮对话入口，负责消息规范化与容错重试。

        功能：
            1> 校验 `chat_inner` 是否实现。
            2> 校验并规范化每条消息的 content 结构。
            3> 调用 `chat_inner` 执行对话；失败时通过裁剪历史进行重试。

        参数：
            messages (list[dict]): 多轮消息列表，每条需包含 role 与 content。
            dataset (str, optional): 数据集名称。

        返回：
            str: 对话生成结果；若多次失败则返回失败信息。

        示例：
            >>> messages = [{"role": "user", "content": "你好"}]
            >>> model.chat(messages, dataset="mmmu")
            '...'
        """
        # 1> 要求子类实现 chat_inner
        assert hasattr(self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        # 2> 逐条校验消息结构并预处理 content
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg  # 必须含 role/content
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg  # 校验结构
            msg['content'] = self.preproc_content(msg['content'])  # 统一为 listdict

        # 3> 尝试调用 chat_inner，失败则裁剪历史重试
        while len(messages):
            try:
                return self.chat_inner(messages, dataset=dataset)
            except Exception as e:
                print(f'{type(e)}: {e}')  # 输出异常信息便于排查
                messages = messages[1:]  # 移除最早消息
                while len(messages) and messages[0]['role'] != 'user':
                    messages = messages[1:]  # 确保从用户消息开始
                continue
        # 4> 所有重试失败后的兜底返回
        return 'Chat Mode: Failed with all possible conversation turns.'
    