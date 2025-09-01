from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. 配置本地模型路径（替换为你下载的 opus-mt-en-zh 文件夹路径）
LOCAL_MODEL_PATH = "E:/笔记/ai模型/opus-mt-en-zh"

# 2. 加载本地模型和分词器（首次加载会占用一定内存，后续调用更快）
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)

# 3. 定义翻译函数（英文转中文，可自定义参数优化效果）
def en2zh_translate(english_text, max_new_tokens=200):
    """
    英文转中文函数
    :param english_text: 输入的英文文本（字符串）
    :param max_new_tokens: 最大生成文本长度（避免输出过长）
    :return: 翻译后的中文文本
    """
    # 对输入文本进行编码（转为模型可识别的格式）
    inputs = tokenizer(
        english_text,
        return_tensors="pt",  # 返回 PyTorch 张量（CPU/GPU 通用）
        padding="max_length",  # 自动填充到模型支持的最大长度
        truncation=True,       # 过长文本自动截断
        max_length=128
    )
    
    # 模型生成中文翻译结果
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,  #  beam search 策略，提升翻译准确性（数值越大越慢但效果越好）
        early_stopping=True,  # 生成结束符后停止
        no_repeat_ngram_size=2  # 避免重复短语
    )
    
    # 解码结果（去除特殊符号，如 <pad>、</s>）
    chinese_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return chinese_text

# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 测试单句翻译
    # test_english = "The quick brown fox jumps over the lazy dog."
    # test_chinese = en2zh_translate(test_english)
    # print(f"英文原文：{test_english}")
    # print(f"中文翻译：{test_chinese}")  # 预期输出：敏捷的棕狐狸跳过了懒狗。

    # 测试长文本翻译
    long_english = "Most organizations in the 21st century are critically dependent on information technology (IT) in their dailyoperations. In most private and public, commercial and non- commercial organizations IT systems became an essentialinfrastructure required to execute day-to-day businessactivities. Even small organizations cannot operate in the modern competitive environment without leveraging the support provided by information systems, while large organizationsare often running and maintaining thousands of various IT systems enabling their businesses. Information systems help run business processes, store business data and facilitateinternal communication within organizations."
    long_chinese = en2zh_translate(long_english)
    print("\n英文长文本：", long_english)
    print("中文翻译：", long_chinese)