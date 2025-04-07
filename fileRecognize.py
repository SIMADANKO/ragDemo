from google import genai
from google.genai import types

import PIL.Image

image = PIL.Image.open('C:/Users/ADMIN/PycharmProjects/PythonProject2/gemini-native-image.png')

client = genai.Client(api_key="AIzaSyD8ldcBX9Ugn36sG11gm1xWvEyV7vi7JTs")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What is this image?", image])

print(response.text)