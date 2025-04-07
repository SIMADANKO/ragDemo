import os
import base64
import google.generativeai as genai
from PyPDF2 import PdfReader
from PIL import Image
import io
import tempfile
import fitz  # PyMuPDF

# 设置 Google API 密钥
os.environ["GOOGLE_API_KEY"] = "AIzaSyD8ldcBX9Ugn36sG11gm1xWvEyV7vi7JTs"  # 替换为你的 API 密钥
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本内容"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_images_from_pdf(pdf_path):
    """从PDF中提取图片"""
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # 将图像字节转换为PIL图像
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    return images


def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def analyze_pdf_with_gemini(pdf_path):
    """使用Gemini-2.0-Flash分析PDF内容"""
    # 提取文本
    text_content = extract_text_from_pdf(pdf_path)

    # 提取图片
    images = extract_images_from_pdf(pdf_path)

    # 初始化Gemini模型
    model = genai.GenerativeModel('gemini-2.0-flash')

    # 分析文本内容
    text_analysis = model.generate_content(
        f"分析以下PDF文档内容，提供文档的主题、关键点和结构概述:\n\n{text_content[:10000]}"  # 限制长度以防止超出token限制
    )

    # 分析图片内容
    image_analysis_results = []
    for i, img in enumerate(images[:5]):  # 限制处理的图片数量
        img_base64 = image_to_base64(img)

        image_parts = [
            {
                "mime_type": "image/png",
                "data": img_base64
            }
        ]

        image_prompt = "描述并分析这张从PDF中提取的图片。它展示了什么内容？有什么关键信息？"
        image_analysis = model.generate_content(
            contents=[image_prompt, *image_parts]
        )
        image_analysis_results.append(f"图片 {i + 1} 分析: {image_analysis.text}")

    # 整合所有分析结果
    full_analysis = {
        "文本内容摘要": text_analysis.text,
        "图片分析": image_analysis_results,
        "文档总体评估": model.generate_content(
            f"基于以下文本内容和图片分析，总结这份PDF文档的整体内容和目的:\n\n文本内容:{text_content[:2000]}\n\n图片分析:{image_analysis_results}"
        ).text
    }

    return full_analysis


def main():
    pdf_path = input("请输入PDF文件路径: ")

    if not os.path.exists(pdf_path):
        print(f"错误：文件 '{pdf_path}' 不存在")
        return

    print("正在分析PDF文件，请稍候...")

    try:
        results = analyze_pdf_with_gemini(pdf_path)

        print("\n===== PDF分析结果 =====")
        print("\n文本内容摘要:")
        print(results["文本内容摘要"])

        print("\n图片分析:")
        for img_analysis in results["图片分析"]:
            print(img_analysis)
            print("-" * 40)

        print("\n文档总体评估:")
        print(results["文档总体评估"])

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()