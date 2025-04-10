import os
import json
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量数据库系统
        """
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"加载嵌入模型时发生错误: {e}")
            raise

    def read_json_file(self, file_path: str) -> dict:
        """读取JSON文件内容"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件 {file_path} 时发生错误: {e}")
            raise

    def get_text_chunks(self, document: dict) -> list:
        """
        从 JSON 字典中提取所有字段的 description 文本，作为 chunk 列表返回。
        针对类似 coding_guidelines 结构的 JSON 格式进行处理，并提取描述字段。
        """
        chunks = []
        try:
            # 处理 general 中的 commenting 和 structure 部分
            general = document.get("coding_guidelines", {}).get("general", {})

            # 提取 commenting 部分
            commenting = general.get("commenting", {})
            if commenting:
                style = commenting.get("style", "")
                examples = commenting.get("examples", [])
                if style:
                    chunks.append({
                        "category": "general.commenting.style",
                        "description": style
                    })
                for example in examples:
                    chunks.append({
                        "category": "general.commenting.examples",
                        "description": example
                    })

            # 提取 structure 部分
            structure = general.get("structure", {})
            if structure:
                for key, value in structure.items():
                    chunks.append({
                        "category": f"general.structure.{key}",
                        "description": value
                    })

            # 处理 copy_statements 部分
            copy_statements = document.get("coding_guidelines", {}).get("copy_statements", {})
            if copy_statements:
                usage = copy_statements.get("usage", "")
                examples = copy_statements.get("examples", [])
                best_practices = copy_statements.get("best_practices", [])

                if usage:
                    chunks.append({
                        "category": "copy_statements.usage",
                        "description": usage
                    })
                for example in examples:
                    copy_statement = example.get("copy_statement", "")
                    explanation = example.get("explanation", "")
                    if copy_statement and explanation:
                        chunks.append({
                            "category": "copy_statements.examples",
                            "description": f"{copy_statement} - {explanation}"
                        })
                for practice in best_practices:
                    chunks.append({
                        "category": "copy_statements.best_practices",
                        "description": practice
                    })

            # 处理 error_handling 部分
            error_handling = document.get("coding_guidelines", {}).get("error_handling", {})
            if error_handling:
                guideline = error_handling.get("guideline", "")
                examples = error_handling.get("examples", [])
                if guideline:
                    chunks.append({
                        "category": "error_handling.guideline",
                        "description": guideline
                    })
                for example in examples:
                    chunks.append({
                        "category": "error_handling.examples",
                        "description": example
                    })

        except Exception as e:
            logger.error(f"获取文本块时发生错误: {e}")

        return chunks

    def text_to_embeddings(self, chunks: list) -> np.ndarray:
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
            raise ValueError("嵌入向量为空，无法创建索引")
        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def save_faiss_index(self, index: faiss.IndexFlatL2, file_path: str):
        """保存FAISS索引到文件"""
        try:
            faiss.write_index(index, file_path)
            logger.info(f"FAISS索引已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存FAISS索引时发生错误: {e}")
            raise

    def save_text_mapping(self, mapping: dict, file_path: str):
        """保存文本与索引映射关系"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=4)
            logger.info(f"文本映射已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存文本映射时发生错误: {e}")
            raise

    def process_and_save(self, json_file_path: str, output_index_path: str, output_mapping_path: str):
        """读取JSON文件，分割文本，生成向量并保存索引和映射"""
        try:
            # 读取JSON文件
            document = self.read_json_file(json_file_path)
            logger.info(f"文件 {json_file_path} 加载成功")

            # 获取文本块
            chunks = self.get_text_chunks(document)
            logger.info(f"文本块分割完成，块数量: {len(chunks)}")

            # 打印检查文本块
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i + 1}:\n{chunk}\n{'-' * 40}")

            # 将文本块转换为嵌入向量
            embeddings = self.text_to_embeddings(chunks)
            if embeddings.size == 0:
                logger.error("嵌入生成失败，无法继续")
                return

            # 创建FAISS索引
            index = self.create_faiss_index(embeddings)
            logger.info("FAISS索引创建完成")

            # 保存FAISS索引到本地文件
            self.save_faiss_index(index, output_index_path)

            # 保存文本与索引的映射关系
            text_mapping = {i: chunk for i, chunk in enumerate(chunks)}
            self.save_text_mapping(text_mapping, output_mapping_path)

        except Exception as e:
            logger.error(f"处理和保存过程发生错误: {e}")
            raise

def main():
    try:
        # 设置JSON文件路径和输出索引文件路径
        json_file_path = "C:/Users/ADMIN/Desktop/code_standard.json"  # 请替换为实际的文件路径
        output_index_path = "C:/Users/ADMIN/Desktop/code_standard_faiss_index.index"  # FAISS索引文件的保存路径
        output_mapping_path = "C:/Users/ADMIN/Desktop/code_standard_text_mapping.json"  # 文本映射文件保存路径

        # 初始化向量数据库系统并处理文件
        vector_db = VectorDatabase()
        vector_db.process_and_save(json_file_path, output_index_path, output_mapping_path)

    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()