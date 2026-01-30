from __future__ import annotations


class Qwen2VLPromptMixin:
    """Qwen2-VL 数据集提示词构建的 Mixin 类。

    类功能：
        为不同数据集提供统一的提示词构建与多模态消息组装逻辑。

    继承关系：
        无显式继承关系（Mixin 组件类），通常被对话模型类多继承使用。

    应用场景：
        1. 在多模态评测任务中按数据集规则拼接提示词与图像/视频输入。
        2. 在推理服务中复用数据集特定的提示词构建策略。

    使用示例：
        >>> class MyModel(Qwen2VLPromptMixin):
        ...     def dump_image(self, line, dataset):
        ...         return "/tmp/a.png"
        >>> model = MyModel(use_custom_prompt=True)

        >>> class MyModel2(Qwen2VLPromptMixin):
        ...     def dump_image(self, line, dataset):
        ...         return ["/tmp/a.png", "/tmp/b.png"]
        >>> model = MyModel2(use_custom_prompt=False)

    数据属性：
        _use_custom_prompt: bool
            是否启用自定义提示词构建的内部开关。
            默认值来自构造参数 `use_custom_prompt`。
            约束：当前实现的 `use_custom_prompt` 固定返回 True，
            该字段主要用于保留扩展逻辑。

        dump_image_func: Optional[Callable]
            将样本行转换为图片路径的回调函数。
            默认未设置，需要通过 `set_dump_image` 注入后使用。
            约束：未设置时调用 `dump_image` 会触发异常。
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        """初始化 Mixin 并保存提示词构建开关。

        功能：
            初始化父类并记录是否启用自定义提示词构建。

        参数：
            *args: 透传给父类构造函数的可变参数。
            use_custom_prompt (bool, optional): 是否启用自定义提示词逻辑，默认 True。
            **kwargs: 透传给父类构造函数的关键字参数。

        返回：
            None。

        示例：
            >>> model = Qwen2VLPromptMixin(use_custom_prompt=True)
            >>> model = Qwen2VLPromptMixin(use_custom_prompt=False)
        """
        # 初始化父类，确保多继承链正常构建
        super().__init__(*args, **kwargs)
        # 记录是否启用自定义提示词构建
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        """设置图片导出回调函数。

        功能：
            注入将样本行转换为图片路径的函数，供 `dump_image` 调用。

        参数：
            dump_image_func (Callable): 接收样本行并返回图片路径或路径列表的函数。

        返回：
            None。

        示例：
            >>> model.set_dump_image(lambda line: "/tmp/a.png")
        """
        # 保存回调函数以供后续使用
        self.dump_image_func = dump_image_func
    
    def dump_image(self, line, dataset):
        """将样本行转换为图片路径或路径列表。

        功能：
            调用已注入的 `dump_image_func` 生成图片路径。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称（当前未使用，保留扩展）。

        返回：
            str | list[str]: 图片路径或图片路径列表。

        示例：
            >>> model.set_dump_image(lambda line: "/tmp/a.png")
            >>> model.dump_image({"id": 1}, "mmmu")
            '/tmp/a.png'
        """
        # 调用回调函数并返回路径结果
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        """判断是否启用自定义提示词构建。

        功能：
            当前实现固定返回 True，始终使用自定义提示词逻辑。
            注释中的逻辑保留了按数据集类型启用的扩展方案。

        参数：
            dataset (str): 数据集名称。

        返回：
            bool: 是否使用自定义提示词构建。

        示例：
            >>> model.use_custom_prompt("MMMU_DEV_VAL")
            True
        """
        # 当前版本强制启用自定义提示词
        return True
        # from vlmeval.dataset import DATASET_TYPE
        # dataset_type = DATASET_TYPE(dataset, default=None)

        # if not self._use_custom_prompt:
        #     return False
        # if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
        #     return True
        # if dataset_type == 'MCQ':
        #     return True
        # if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
        #     return True
        # if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
        #     return True
        # return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """构建指定数据集的多模态提示词消息列表。

        功能：
            将样本行转换为模型可用的多模态消息结构。
            当前默认走 MMMU 的提示词构建逻辑。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称。

        返回：
            list[dict[str, str]]: 按模型规范组织的多模态消息列表。

        示例：
            >>> msgs = model.build_prompt(line={"question": "Q?"}, dataset="MMMU_DEV_VAL")
        """
        # 默认使用 MMMU 的提示词构建策略
        return self._build_mmmu_prompt(line, dataset)

    def split_MMMU(self, msgs):
        """将 MMMU 的文本占位符拆分为图文交错的消息列表。

        功能：
            识别文本中的 `<image n>` 占位符，并按顺序插入对应图片，生成交错消息列表。

        参数：
            msgs (list[dict]): 含图片与文本的原始消息列表。

        返回：
            list[dict]: 按 `<image n>` 拆分后的交错消息列表。

        示例：
            >>> msgs = [{"type": "image", "value": "a.png"}, {"type": "text", "value": "see <image 1> here"}]
            >>> model.split_MMMU(msgs)
        """
        # 1> 收集文本与图片列表
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None  # 保证只有一段文本
                text = s['value']
        # 2> 按 `<image n>` 占位符切分文本
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs

        # 3> 交错重建图文消息序列
        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            assert seg[0].isdigit() and seg[1] == '>'
            image_idx = int(seg[0]) - 1

            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """构建 MMMU 数据集的提示词与多模态消息。

        功能：
            1> 解析问题、选项与提示信息。
            2> 生成文本提示词并将图片放置在消息开头。
            3> 支持将 `<image n>` 占位符拆分为交错结构。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称（用于调用 `dump_image`）。

        返回：
            list[dict[str, str]]: MMMU 专用的多模态消息列表。

        示例：
            >>> msgs = model._build_mmmu_prompt(line={"question": "Q?"}, dataset="MMMU_DEV_VAL")
        """

        import string

        import pandas as pd

        # 1> 设置图像像素范围，用于输入图像大小控制
        MIN_PIXELS = 1280 * 28 * 28
        MAX_PIXELS = 5120 * 28 * 28

        # 2> 提取问题、选项与提示信息
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        question_prompt = f'Question: {question}'
        # string.ascii_uppercase: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

        # 3> 组织最终文本提示词
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()

        # 4> 构建消息列表，确保图片置于文本前
        msgs = []
        # msgs.append(dict(type='text', value=question_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)]
        msgs.append(dict(type='text', value=prompt))

        # 5> 如果文本包含占位符，则拆分为交错结构
        msgs_new = self.split_MMMU(msgs)
        return msgs_new

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """构建 MCQ 数据集的提示词与多模态消息。

        功能：
            1> 判断题干是否包含中文，选择中/英文提示语。
            2> 生成问题与选项文本，并组合图片输入。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称（用于调用 `dump_image`）。

        返回：
            list[dict[str, str]]: MCQ 专用的多模态消息列表。

        示例：
            >>> msgs = model._build_mcq_prompt(line={"question": "Q?"}, dataset="MCQ")
        """
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string
        import pandas as pd

        def cn_string(s):
            import re
            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        # 1> 提取问题、选项与提示信息
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

        # 2> 组织提示词，并根据中文检测选择提示语
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()

        # 3> 组装图片与文本消息
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """构建 Y/N（是/否）数据集的提示词与多模态消息。

        功能：
            1> 拼接问题文本并追加是/否回答要求。
            2> 组合图片与文本消息结构。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称（用于调用 `dump_image`）。

        返回：
            list[dict[str, str]]: Y/N 专用的多模态消息列表。

        示例：
            >>> msgs = model._build_yorn_prompt(line={"question": "Is it red?"}, dataset="Y/N")
        """
        YORN_PROMPT = ' Please answer yes or no.'

        # 1> 获取图片路径与问题文本
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        # 2> 组装图片消息
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        # 3> 追加文本消息并拼接是/否提示
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """构建 VQA 数据集的提示词与多模态消息。

        功能：
            1> 拼接问题文本并附加简短回答提示。
            2> 组合图片与文本消息结构。

        参数：
            line: 数据集中单行样本数据。
            dataset (str): 数据集名称（用于调用 `dump_image`）。

        返回：
            list[dict[str, str]]: VQA 专用的多模态消息列表。

        示例：
            >>> msgs = model._build_vqa_prompt(line={"question": "What is shown?"}, dataset="VQA")
        """
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        # 1> 获取图片路径与问题文本
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        # 2> 组装图片消息
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        # 3> 追加文本消息并拼接提示语
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs
