import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import google.generativeai as genai
import fitz  # PyMuPDF 用于 PDF
from docx import Document  # 用于 DOCX

# 配置 Gemini API Key
genai.configure(api_key="AIzaSyD8ldcBX9Ugn36sG11gm1xWvEyV7vi7JTs")  # 替换为你的 API Key

# 加载 Gemini 模型
model = genai.GenerativeModel("gemini-1.5-flash")


# 1. 文件识别函数
def recognize_file(file_path):
    file_extension = file_path.lower().split('.')[-1]

    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
        # 图像识别
        with open(file_path, "rb") as img_file:
            image = Image.open(io.BytesIO(img_file.read()))
        response = model.generate_content([image, "这张图片包含什么内容？"])
        return response.text

    elif file_extension == 'pdf':
        # PDF 识别
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        response = model.generate_content([text, "这段文本包含什么内容？"])
        return response.text

    elif file_extension == 'docx':
        # DOCX 识别
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        response = model.generate_content([text, "这段文本包含什么内容？"])
        return response.text

    elif file_extension == 'txt':
        # TXT 识别
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        response = model.generate_content([text, "这段文本包含什么内容？"])
        return response.text

    elif file_extension == 'json':
        # JSON 识别
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = json.dumps(data, ensure_ascii=False)
        response = model.generate_content([text, "这段 JSON 包含什么内容？"])
        return response.text

    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")


# 2. 加载知识库
def load_knowledge_base(mapping_file, index_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        text_mapping = json.load(f)
    index = faiss.read_index(index_file)
    return text_mapping, index


# 加载两个知识库
code_standards_mapping, code_standards_index = load_knowledge_base(
    'code_standards_output_text_mapping.json', 'code_standards_output_faiss_index.index'
)
copybook_mapping, copybook_index = load_knowledge_base(
    'copybook_output_text_mapping.json', 'copybook_output_faiss_index.index'
)

# 3. 初始化嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# 4. 检索函数
def retrieve_from_index(index, text_mapping, query_vector, top_k=3):
    distances, indices = index.search(np.array(query_vector), top_k)
    retrieved_docs = [text_mapping[str(idx)] for idx in indices[0]]
    return retrieved_docs


# 5. 主流程
def generate_cobol_code(file_path):
    # 识别文件内容
    query_text = recognize_file(file_path)
    print(f"Gemini 识别的内容: {query_text}")

    # 将查询文本向量化
    query_vector = embedding_model.encode([query_text])

    # 从知识库检索
    code_standards_docs = retrieve_from_index(code_standards_index, code_standards_mapping, query_vector)
    copybook_docs = retrieve_from_index(copybook_index, copybook_mapping, query_vector)

    # 构造提示词
    prompt_template = """
    你是一个 COBOL 编程专家。请根据以下信息生成符合规范的 COBOL 代码：
    1. 代码规范要求：
    {standards}
    2. Copybook 定义：
    {copybook}
    3. 需求描述（从文件识别结果提取）：
    {query}
    请生成对应的 COBOL 代码。
    """
    prompt = prompt_template.format(
        standards="\n".join(code_standards_docs),
        copybook="\n".join(copybook_docs),
        query=query_text
    )
    print(f"生成的提示词:\n{prompt}")

    # 模拟生成 COBOL 代码（实际可将 prompt 输入给我）
    cobol_code = """
    IDENTIFICATION DIVISION.
    PROGRAM-ID. GENERATED-PROG.
    ENVIRONMENT DIVISION.
    DATA DIVISION.
    WORKING-STORAGE SECTION.
    COPY {copybook_name}.
    PROCEDURE DIVISION.
        DISPLAY 'Generated COBOL Code'.
        STOP RUN.
    """
    print(f"生成的 COBOL 代码:\n{cobol_code}")
    return cobol_code


# 调用示例
file_path = "C:/Users/ADMIN/Desktop/微信图片_20240501150423.jpg"  # 可替换为 PDF/DOCX/TXT/JSON 文件
generate_cobol_code(file_path)
