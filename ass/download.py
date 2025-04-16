import yt_dlp

def download_bilibili_video(video_url, output_dir='C:/Users/34743/Videos'):
    ydl_opts = {
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',  # 输出目录和文件名
        'format': 'bestvideo+bestaudio/best',  # 下载最佳视频+音频组合，或最佳可用格式
        'merge_output_format': 'mp4',  # 确保输出为 mp4 格式
        'quiet': False,  # 显示下载进度信息
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

if __name__ == '__main__':
    url = input("请输入 B站视频 URL：")
    download_bilibili_video(url)