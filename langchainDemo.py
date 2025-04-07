import os
import subprocess
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_vertexai import ChatVertexAI

# 设置 Google Cloud 认证环境变量
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\ADMIN\\Downloads\\static-forest-454701-a1-0040cf5d0dfe.json"


# 初始化 Gemini AI
llm = ChatVertexAI(model_name="gemini-1.5-pro", temperature=0.3)

# 设计生成修改提示的 Prompt
modify_prompt = PromptTemplate(
    input_variables=["java_code", "error_message"],
    template=""" 
你是一个专业的 Java 开发助手，能够根据编译错误信息修正代码。
当前编译失败，错误信息如下：
{error_message}

以下是编译失败的 Java 代码：
{java_code}

根据这些错误信息，给出详细的修改提示：
- 请提出代码中存在的问题。
- 提供如何修改代码的建议。
- 如果有必要，指出代码中需要更改或添加的部分。

请只给出修改的建议，不要重新生成整个代码。
"""
)

# 设计 Java 代码生成的 Prompt
java_generator_prompt = PromptTemplate(
    input_variables=["description", "modification_suggestion"],
    template=""" 
你是一个 Java 开发助手，根据以下描述和修改建议生成 Java 代码：

描述：{description}
修改建议：{modification_suggestion}

请生成符合 Java 规范、能够通过编译的 Java 代码，并确保包含 package 语句。
仅需要生成代码，不需要任何的说明和注释。
"""
)

# LangChain 生成修改提示词的链
modify_chain = modify_prompt | llm

# LangChain 生成修改后的 Java 代码的链
java_generator = java_generator_prompt | llm


def clean_code(java_code):
    """清除代码块标记（如 ```java 和 ```）"""
    return re.sub(r'```java\s*|\s*```', '', java_code)


def extract_class_name(java_code):
    """从 Java 代码中提取类名"""
    match = re.search(r'class (\w+)', java_code)
    if match:
        return match.group(1)
    return "GeneratedCode"  # 默认类名


def extract_package_name(java_code):
    """从 Java 代码中提取 package 名称"""
    match = re.search(r'package\s+([\w.]+);', java_code)
    if match:
        return match.group(1)
    return "com.generated"  # 默认 package 名称




def generate_code_with_error_fix(description):
    attempt = 1

    # 生成初始 Java 代码
    java_code_response = java_generator.invoke({"description": description, "modification_suggestion": ""})
    java_code = java_code_response.content.strip()  # 使用 .content 来访问文本
    java_code = clean_code(java_code)

    while True:
        print(f"尝试第 {attempt} 次生成和编译代码...")

        # 打印生成的 Java 代码
        print(f"🔧 生成的 Java 代码：\n{java_code}")

        # 运行代码并获取执行结果
        success, output = run_code(java_code)

        if success:

            break  # 成功则跳出循环

        # 代码运行失败，传递错误信息给 LLM 修正
        print(f"❌ 运行失败，错误信息：\n{output}")

        modify_response = modify_chain.invoke({
            "java_code": java_code,
            "error_message": output  # 直接使用 run_code() 返回的错误信息
        })

        # 获取 LLM 的修改建议
        modification_suggestion = modify_response.content.strip()
        print(f"📝 LLM 提供的修改建议：\n{modification_suggestion}")

        # 生成修改后的 Java 代码
        java_code_response = java_generator.invoke({
            "description": description,
            "modification_suggestion": modification_suggestion
        })
        java_code = java_code_response.content.strip()
        java_code = clean_code(java_code)

        # print(f"🔄 修改后的代码：\n{java_code}")

        # 增加尝试次数
        attempt += 1



    return java_code  # 返回最终修正并成功运行的代码

def run_code(code):
    """ 运行 Java 代码，并返回 (成功与否, 输出或错误信息) """
    # 提取 package 和类名
    package_name = extract_package_name(code)
    class_name = extract_class_name(code)
    package_path = package_name.replace(".", "/")
    full_dir_path = os.path.join("generated_code", package_path)

    # 确保目录存在
    os.makedirs(full_dir_path, exist_ok=True)

    # 生成 Java 文件完整路径
    java_file_path = os.path.join(full_dir_path, f"{class_name}.java")

    # 写入 Java 代码到文件
    with open(java_file_path, "w", encoding="utf-8") as f:
        f.write(code)

    # 编译 Java 代码
    compile_result = subprocess.run(["javac", "-d", "generated_code", java_file_path], capture_output=True, text=True)

    if compile_result.returncode == 0:
        print("✅ 编译成功！")

        # 运行 Java 程序并捕获运行时错误
        run_result = subprocess.run(["java", "-cp", "generated_code", f"{package_name}.{class_name}"],
                                    capture_output=True, text=True)
        if run_result.returncode == 0:
            print("✅ 程序运行成功！")
            # 打印 Java 程序的输出
            print(f"程序输出:\n{run_result.stdout}")
            return True, run_result.stdout  # 运行成功，返回标准输出
        else:
            print(f"❌ 程序运行失败")
            return False, run_result.stderr  # 运行失败，返回错误信息

    else:
        print(f"❌ 编译失败，错误信息：\n{compile_result.stderr}")
        return False, compile_result.stderr  # 编译失败，返回错误信息
# 示例：用户描述
description = """"树可以看成是一个连通且 无环 的 无向 图。

给定往一棵edges =
[[1,2],[1,3],[2,3]]
的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。

请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回并打印数组 edges 中最后出现的那个。"""

# 调用生成和修复流程
final_code = generate_code_with_error_fix(description)

