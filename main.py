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


def extract_handbook_nodes(handbook_content, prompt_file, output_table_dir, output_image_dir):
    """
    提取数据手册的节点信息，并结合已有结果进行纠错与融合。
    """
    # 加载 Prompt 模板
    prompt_template = load_prompt_from_file(prompt_file)
    if not prompt_template:
        raise ValueError("无法加载 Prompt，请检查 full_prompt.txt 文件是否存在且内容正确。")

    # 加载已有结果
    existing_results = []
    for folder in [output_table_dir, output_image_dir]:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            existing_results.append(data)
                    except Exception as e:
                        print(f"加载文件 {file_path} 失败: {e}")

    # 将输入内容和已有结果插入到 Prompt 模板中
    context = {
        "handbook_content": handbook_content,
        "existing_results": json.dumps(existing_results, indent=2, ensure_ascii=False),
    }
    prompt = prompt_template.format(**context)

    # 调用通义千问 API
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        messages=[{'role': 'user', 'content': prompt}],
        result_format='text',
        max_tokens=8192,
        temperature=0.6
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


def save_to_json(data, output_file):
    """
    将提取的数据保存为 JSON 文件。
    """
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"数据已成功保存到 {output_file}")
    except Exception as e:
        print("保存文件失败:", e)


if __name__ == "__main__":
    # 从 .env 文件中读取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件是否正确配置。")

    # 定义 Prompt 文件路径
    prompt_file = "prompt/full_prompt.txt"

    # 定义输出文件夹路径
    output_table_dir = "output_table"
    output_image_dir = "output_image"

    # 读取数据手册内容（假设数据手册是一个 Markdown 文件）
    handbook_file = "APD_Series_203250D.md"
    with open(handbook_file, "r", encoding="utf-8") as file:
        handbook_content = file.read()

    # 提取数据手册的节点信息
    extracted_info = extract_handbook_nodes(handbook_content, prompt_file, output_table_dir, output_image_dir)

    # 如果提取成功，保存为 JSON 文件
    if extracted_info:
        output_file = "handbook_nodes.json"
        save_to_json(extracted_info, output_file)
