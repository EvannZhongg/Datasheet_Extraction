import os
import json
from dotenv import load_dotenv
from dashscope import Generation

# 加载 .env 文件中的环境变量
load_dotenv()


def load_prompt_from_file(prompt_file):
    """
    从指定文件中加载 Prompt 内容。
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt = file.read().strip()
        return prompt
    except Exception as e:
        print("加载 Prompt 文件失败:", e)
        return None


def extract_image_info(markdown_text, prompt_file):
    """
    使用通义千问 API 提取器件类型、器件名称和参数。
    """
    # 加载 Prompt
    prompt_template = load_prompt_from_file(prompt_file)
    if not prompt_template:
        raise ValueError("无法加载 Prompt，请检查 image_prompt.txt 文件是否存在且内容正确。")

    # 将输入文本插入到 Prompt 模板中
    prompt = prompt_template.format(markdown_text=markdown_text)

    # 调用通义千问 API
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 确保传递了 API Key
        model="qwen-max",  # 推荐使用 qwen-max 模型
        messages=[{'role': 'user', 'content': prompt}],  # 使用 messages 参数代替 prompt
        result_format='text',  # 设置返回结果格式
        max_tokens=2048,  # 最大生成长度
        temperature=0.7  # 控制输出随机性
    )

    # 解析 API 响应
    if response.status_code == 200:
        result = response.output["text"]

        # 清理返回结果，移除多余的代码块标记（如 ```json）
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]  # 移除开头的 ```json
        if result.endswith("```"):
            result = result[:-3]  # 移除结尾的 ```
        result = result.strip()  # 去掉多余的空格或换行符

        try:
            # 验证 JSON 格式是否正确
            extracted_data = json.loads(result)  # 将结果解析为 JSON
            return extracted_data
        except json.JSONDecodeError as e:
            print("解析 JSON 失败:", e)
            print("原始响应:", result)
            return None
    else:
        print("API 调用失败:", response.message)
        return None


def save_to_txt(data, output_file):
    """
    将提取的数据保存为 TXT 文件。
    """
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            # 将数据转换为字符串并写入文件
            file.write(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"数据已成功保存到 {output_file}")
    except Exception as e:
        print("保存文件失败:", e)


# 示例：读取 Markdown 文件并提取信息
if __name__ == "__main__":
    # 从 .env 文件中读取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件是否正确配置。")

    # 定义 Prompt 文件路径
    prompt_file = "prompt/image_prompt.txt"  # 替换为您的 Prompt 文件路径

    # 读取 Markdown 文件内容
    input_file = "image_test.md"  # 替换为您的输入文件路径
    with open(input_file, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    # 调用函数提取信息
    extracted_info = extract_image_info(markdown_content, prompt_file)

    # 如果提取成功，保存为 TXT 文件
    if extracted_info:
        output_file = "image_output.txt"  # 输出文件路径
        save_to_txt(extracted_info, output_file)
