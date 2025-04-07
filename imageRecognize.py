import google.generativeai as genai
from PIL import Image
import io

# 设置 API Key（请替换为你的 API Key）
genai.configure(api_key="AIzaSyD8ldcBX9Ugn36sG11gm1xWvEyV7vi7JTs")

# 加载 Gemini Pro Vision 模型
model = genai.GenerativeModel("gemini-1.5-flash")

# 读取图片并进行推理
def recognize_image(image_path):
    with open(image_path, "rb") as img_file:
        image = Image.open(io.BytesIO(img_file.read()))

    response = model.generate_content([image, "这张图片包含什么内容？"])
    print(response.text)



# 调用示例
recognize_image("C:/Users/ADMIN/Desktop/微信图片_20240501150423.jpg")
