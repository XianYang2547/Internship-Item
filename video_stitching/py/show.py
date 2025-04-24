import asyncio
import cv2
import numpy as np
import subprocess
from asyncio import Queue

async def read_frame_from_rtsp(rtsp_url: str, frame_queue: Queue, name: str):
    """异步读取RTSP流，优化低延迟"""
    command = [
        'ffmpeg',
        '-i', rtsp_url,  # RTSP流地址
        '-f', 'image2pipe',  # 使用管道输出
        '-pix_fmt', 'bgr24',  # 设置图像格式为bgr24
        '-vcodec', 'rawvideo',  # 使用原始视频流格式
        '-an', '-sn', '-dn',  # 禁用音频、字幕和数据流
        '-loglevel', 'error',  # 降低ffmpeg的日志输出级别
        '-'
    ]
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    width, height = 1920, 1080
    frame_size = width * height * 3
    
    try:
        while True:
            raw_frame = await process.stdout.readexactly(frame_size)
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            if frame_queue.full():
                await frame_queue.get()  # 丢弃旧帧
            await frame_queue.put(frame)
    except Exception as e:
        print(f"{name} 错误: {e}")
    finally:
        process.terminate()

async def main():
    # 球机
    # rtsp_url_1 = "rtsp://admin:yfzh123456@172.16.20.3:554/Streaming/Channels/201"
    # rtsp_url_2 = "rtsp://admin:yfzh123456@172.16.20.103:554/Streaming/Channels/201"#(黑色的)
    # 枪机
    rtsp_url_1 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/101"
    rtsp_url_2 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/201"

    frame_queue_1 = Queue(maxsize=5) 
    frame_queue_2 = Queue(maxsize=5)

    asyncio.create_task(read_frame_from_rtsp(rtsp_url_1, frame_queue_1, "Stream1"))
    asyncio.create_task(read_frame_from_rtsp(rtsp_url_2, frame_queue_2, "Stream2"))

    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # 同步获取两帧，避免时间差
            frame1, frame2 = await asyncio.gather(
                frame_queue_1.get(),
                frame_queue_2.get()
            )
            # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
            result = np.hstack((frame1, frame2))
            cv2.imshow("Video Stream", result)
            
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())

