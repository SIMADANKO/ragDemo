import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import google.generativeai as genai
import fitz  # PyMuPDF 用于 PDF
from docx import Document  # 用于 DOCX
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置 Gemini API Key
genai.configure(api_key="AIzaSyD8ldcBX9Ugn36sG11gm1xWvEyV7vi7JTs")  # 替换为你的 API Key
model = genai.GenerativeModel("gemini-1.5-flash")

# 1. 文件识别函数
def recognize_file(file_path):
    try:
        file_extension = file_path.lower().split('.')[-1]
        logger.info(f"正在识别文件: {file_path}")

        if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
            with open(file_path, "rb") as img_file:
                image = Image.open(io.BytesIO(img_file.read()))
            response = model.generate_content([image, "将包含的内容整理成一份详细的设计书完整中文打印出来"])
            return response.text

        elif file_extension == 'pdf':
            doc = fitz.open(file_path)
            text = "".join([page.get_text() for page in doc])
            response = model.generate_content([text, "将包含的内容整理成一份详细的设计书完整中文打印出来"])
            return response.text

        elif file_extension == 'docx':
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            response = model.generate_content([text, "将包含的内容整理成一份详细的设计书完整中文打印出来"])
            return response.text

        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            response = model.generate_content([text, "将包含的内容整理成一份详细的设计书完整中文打印出来"])
            return response.text

        elif file_extension == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)
            response = model.generate_content([text, "将包含的内容整理成一份详细的设计书完整中文打印出来"])
            return response.text

        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")
    except Exception as e:
        logger.error(f"文件识别失败: {str(e)}")
        raise

# 2. 加载知识库
def load_knowledge_base(mapping_file, index_file):
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            text_mapping = json.load(f)
        index = faiss.read_index(index_file)
        logger.info(f"成功加载知识库: {mapping_file}")
        return text_mapping, index
    except Exception as e:
        logger.error(f"加载知识库失败: {str(e)}")
        raise

# 3. 检索函数
def retrieve_from_index(index, text_mapping, query_vector, top_k=3, threshold=0.8):
    try:
        distances, indices = index.search(np.array(query_vector), top_k)
        retrieved_docs = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist < threshold:  # 距离阈值筛选
                doc = text_mapping.get(str(idx), {})
                retrieved_docs.append(doc.get('description', ''))
        return retrieved_docs if retrieved_docs else ["未检索到相关内容"]
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        return ["检索失败"]

# 4. 调用 Gemini 生成 COBOL 代码
def generate_code_with_gemini(prompt):
    try:
        response = model.generate_content([prompt])
        logger.info("成功调用 Gemini 生成 COBOL 代码")
        return response.text
    except Exception as e:
        logger.error(f"Gemini 生成代码失败: {str(e)}")
        return None

# 5. 主流程
def generate_cobol_code(file_path):
    try:
        # 初始化嵌入模型
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # 加载知识库
        code_standards_mapping, code_standards_index = load_knowledge_base(
            'C:/Users/ADMIN/Desktop/code_standard_text_mapping.json',
            'C:/Users/ADMIN/Desktop/code_standard_faiss_index.index'
        )
        copybook_mapping, copybook_index = load_knowledge_base(
            'C:/Users/ADMIN/Desktop/copybook_text_mapping.json',
            'C:/Users/ADMIN/Desktop/copybook_faiss_index.index'
        )

        # 识别文件内容
        query_text = recognize_file(file_path)
        logger.info(f"Gemini 识别的内容: {query_text}")

        # 将查询文本向量化
        query_vector = embedding_model.encode([query_text])

        # 从知识库检索
        code_standards_docs = retrieve_from_index(code_standards_index, code_standards_mapping, query_vector)
        copybook_docs = retrieve_from_index(copybook_index, copybook_mapping, query_vector)

        # 动态提取 Copybook 名称
        copybook_name = copybook_docs[0].get('name', 'DEFAULT-COPYBOOK') if isinstance(copybook_docs[0], dict) else 'DEFAULT-COPYBOOK'
        code_standards_text = "\n".join(code_standards_docs)
        copybook_text = "\n".join(copybook_docs)

        # 构造提示词
        prompt_template = """
        你是一个 COBOL 编程专家。请根据以下信息生成符合规范的 COBOL 代码：
        1. 代码规范要求（请严格遵守）：
        {standards}
        2. Copybook 定义（使用以下定义）：
        {copybook}
        3. 需求描述（从文件识别结果提取，请完整实现）：
        {query}
        请生成：
        - 符合 IBM COBOL 标准的代码
        - 包含详细注释
        - 能完整实现需求的逻辑
        - 使用 Copybook 名称: {copybook_name}
        """
        prompt = prompt_template.format(
            standards=code_standards_text,
            copybook=copybook_text,
            query=query_text,
            copybook_name=copybook_name
        )
        logger.info(f"生成的提示词:\n{prompt}")

        # 调用 Gemini 生成 COBOL 代码
        cobol_code = generate_code_with_gemini(prompt)
        if cobol_code:
            logger.info(f"生成的 COBOL 代码:\n{cobol_code}")
        else:
            logger.warning("未能生成 COBOL 代码")
        return cobol_code

    except Exception as e:
        logger.error(f"生成 COBOL 代码失败: {str(e)}")
        return None

# 调用示例
if __name__ == "__main__":
    file_path = "C:/Users/ADMIN/Desktop/式样设计书.jpg"
    cobol_code = generate_cobol_code(file_path)