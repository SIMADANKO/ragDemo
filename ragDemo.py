import os
import re
import logging
from typing import Optional, List
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRAG:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, top_k: int = 10,
                 gemini_api_key: Optional[str] = None):
        """
        初始化RAG系统
        """
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"加载嵌入模型时发生错误: {e}")
            raise

        self.chunk_size = chunk_size
        self.top_k = top_k
        self.client = None
        self.cobol_splitter = self.CobolCodeSplitter(chunk_size)

        if gemini_api_key:
            self._init_gemini_client(gemini_api_key)

    def _init_gemini_client(self, gemini_api_key: str):
        """初始化 Gemini API 客户端"""
        try:
            genai.configure(api_key=gemini_api_key)
            logger.info("Gemini API Key 初始化成功")
        except Exception as e:
            logger.error(f"初始化 Gemini 客户端失败: {e}")
            raise

    def predict(self, prompt: str, model_name: str) -> str:
        """
        使用指定模型生成预测结果
        """
        try:
            client = genai.GenerativeModel(model_name)
            response = client.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                logger.error("Gemini API 返回格式不正确")
                return "AI未能生成有效回答"
        except Exception as e:
            logger.error(f"发生预测错误: {e}")
            return f"发生错误: {str(e)}"

    def read_file(self, file_path: str) -> str:
        """读取文件内容"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf':
                return self._read_pdf(file_path)
            elif file_ext == '.docx':
                return self._read_docx(file_path)
            elif file_ext == '.txt':
                return self._read_txt(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时发生错误: {e}")
            raise

    def _read_pdf(self, file_path: str) -> str:
        try:
            pdf_reader = PdfReader(file_path)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            return text
        except Exception as e:
            logger.error(f"读取PDF文件 {file_path} 时发生错误: {e}")
            raise

    def _read_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            logger.error(f"读取DOCX文件 {file_path} 时发生错误: {e}")
            raise

    def _read_txt(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取TXT文件 {file_path} 时发生错误: {e}")
            raise

    class CobolCodeSplitter:
        def __init__(self, chunk_size: int):
            self.chunk_size = chunk_size

        def get_text_chunks_by_code_block(self, document_text: str) -> List[str]:
            """
            按完整 COBOL 代码块分割文本，确保代码和对应的文本解释在一起。
            """
            chunks = []
            division_pattern = r'(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\.'
            divisions = list(re.finditer(division_pattern, document_text, re.IGNORECASE))

            if not divisions:
                return self._split_non_code_text(document_text)

            last_end = 0
            for i, match in enumerate(divisions):
                start = match.start()
                if start > last_end:
                    non_code_text = document_text[last_end:start].strip()
                    if non_code_text:
                        chunks.extend(self._split_non_code_text(non_code_text))

                end = divisions[i + 1].start() if i + 1 < len(divisions) else len(document_text)
                code_block = document_text[start:end].strip()
                if code_block:
                    chunks.extend(self._split_code_block_with_explanation(code_block))
                last_end = end

            if last_end < len(document_text):
                remaining_text = document_text[last_end:].strip()
                if remaining_text:
                    chunks.extend(self._split_non_code_text(remaining_text))

            return [chunk for chunk in chunks if chunk]

        def _split_non_code_text(self, text: str) -> List[str]:
            """分割非代码文本"""
            paragraphs = text.split('\n')
            chunks = []
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                while len(para) > self.chunk_size:
                    chunks.append(para[:self.chunk_size])
                    para = para[self.chunk_size:]
                if para:
                    chunks.append(para)
            return chunks

        def _split_code_block_with_explanation(self, code_block: str) -> List[str]:
            """处理代码块与其解释文本，确保它们在一起"""
            lines = code_block.split('\n')
            chunks = []
            current_chunk = ""
            section_pattern = r'([A-Z][A-Z0-9-]*)\s+SECTION\.'

            for line in lines:
                line = line.strip()
                is_comment = line.startswith('*') or line.startswith('/')
                is_code = not is_comment and line.strip() != ""

                if is_code or is_comment:
                    if len(current_chunk) + len(line) + 1 <= self.chunk_size:
                        current_chunk += "\n" + line if current_chunk else line
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = line
                elif re.match(section_pattern, line):
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line

            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

    def get_text_chunks_by_code_block(self, document_text: str) -> List[str]:
        """外部接口调用COBOL分块"""
        return self.cobol_splitter.get_text_chunks_by_code_block(document_text)

    def text_to_embeddings(self, chunks: List[str]) -> np.ndarray:
        """将文本块转换为嵌入向量"""
        try:
            embeddings = self.embedding_model.encode(chunks)
            return embeddings
        except Exception as e:
            logger.error(f"生成嵌入时发生错误: {e}")
            return np.array([])

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """创建FAISS向量索引"""
        if embeddings is None or len(embeddings) == 0:
          raise
        ValueError("嵌入向量为空，无法创建索引")
        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve_relevant_text(self, query: str, chunks: List[str], index: faiss.IndexFlatL2) -> List[str]:
        """根据查询检索相关文本"""
        try:
            query_embedding = self.text_to_embeddings([query])
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            distances, indices = index.search(query_embedding, self.top_k)
            return [chunks[i] for i in indices[0].tolist()]
        except Exception as e:
            logger.error(f"检索相关文本时发生错误: {e}")
            return []

    def ask_gemini_rag(self, query: str, chunks: List[str], index: faiss.IndexFlatL2, model_name: str) -> Optional[str]:
        """使用Gemini结合检索结果回答问题"""
        try:
            relevant_texts = self.retrieve_relevant_text(query, chunks, index)
            context = "\n".join(relevant_texts)
            prompt = (
                f"你是一个COBOL编程专家，以下是从文档中检索到的相关信息：\n{context}\n"
                f"请基于这些信息回答问题：{query}\n"
                f"要求：\n"
                f"1. 生成一个完整的COBOL程序，包括IDENTIFICATION、ENVIRONMENT、DATA和PROCEDURE DIVISION。\n"
                f"2. 程序应使用WORKING-STORAGE SECTION定义必要的变量。\n"
                f"3. 在PROCEDURE DIVISION中实现逻辑，并使用段落（PARAGRAPH）组织代码。\n"
                f"4. 在回答末尾，列出具体参考了知识库中的哪些代码片段，并完整打印这些片段。\n"
                f"5. 如果知识库中没有直接相关的信息，基于COBOL最佳实践设计程序，并在回答中说明这一点。"
            )
            response = self.predict(prompt, model_name)
            return response
        except Exception as e:
            logger.error(f"使用Gemini生成回答时发生错误: {e}")
            return f"发生错误: {str(e)}"

def main():
    try:
        gemini_api_key = "AIzaSyD8ldcBX9Ugn36sG11gm1xWvEyV7vi7JTs"  # 请替换为实际的API密钥
        rag_system = DocumentRAG(embedding_model_name="all-MiniLM-L6-v2", gemini_api_key=gemini_api_key)

        # 假设这些COBOL代码已保存在ragdemo.txt中
        file_path = "C:/Users/ADMIN/Desktop/test/ragdemo.txt"
        document_text = rag_system.read_file(file_path)
        logger.info(f"文件加载成功，文本长度: {len(document_text)}")

        chunks = rag_system.get_text_chunks_by_code_block(document_text)
        logger.info(f"分割后的文本块数量: {len(chunks)}")

        for i, chunk in enumerate(chunks[:5]):
            logger.debug(f"Chunk {i}: {chunk[:100]}... (长度: {len(chunk)})")

        embeddings = rag_system.text_to_embeddings(chunks)
        logger.info(f"嵌入生成完成，嵌入数量: {len(embeddings)}")

        index = rag_system.create_faiss_index(embeddings)
        logger.info("FAISS索引创建完成")

        model_name = "gemini-1.5-pro"
        query = (
            "参考知识库，用COBOL设计一个程序，处理下一个发票的日期计算。"
            "其目标是根据当前月份和年份，计算出下一个发票日期（假设每个月都有发票日期），"
            "并特别处理二月份的天数（考虑闰年和非闰年）。最后请给出参照了知识库中的哪些信息而生成的，"
            "打印那些信息的完整代码"
        )

        answer = rag_system.ask_gemini_rag(query, chunks, index, model_name)
        print(f"AI的回答:\n{answer}")

    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()