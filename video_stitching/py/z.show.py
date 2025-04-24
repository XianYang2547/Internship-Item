import asyncio
import cv2
import numpy as np
import subprocess
import time
from asyncio import Queue


# ========= 用户自定义参数 =========
STITCH_RESIZE = (960, 540)    # 拼接前图像缩放大小
DISPLAY_RESIZE = (960, 540)   # 显示窗口分辨率
# ==================================


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
    rtsp_url_1 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/101"
    rtsp_url_2 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/201"

    frame_queue_1 = Queue(maxsize=5)
    frame_queue_2 = Queue(maxsize=5)

    asyncio.create_task(read_frame_from_rtsp(rtsp_url_1, frame_queue_1, "Stream1"))
    asyncio.create_task(read_frame_from_rtsp(rtsp_url_2, frame_queue_2, "Stream2"))

    cv2.namedWindow("Stitched Image", cv2.WINDOW_NORMAL)

    stitcher = cv2.Stitcher_create()
    stitcher.setPanoConfidenceThresh(0.3)
    black_img = np.zeros((960, 540), dtype=np.uint8)

    try:
        while True:
            t_total_start = time.time()
            frame1, frame2 = await asyncio.gather(frame_queue_1.get(), frame_queue_2.get())

            frame1 = cv2.resize(frame1, STITCH_RESIZE)
            frame2 = cv2.resize(frame2, STITCH_RESIZE)

            t5 = time.time()
            try:
                status, stitched = stitcher.stitch([frame1, frame2])
            except cv2.error as e:
                print(f"[错误] 图像拼接失败: {e}")
                cv2.imshow("Stitched Image", black_img)
                continue
            t6 = time.time()
            print(f"[耗时] 图像拼接: {(t6 - t5)*1000:.2f} ms")


            if status == 0:
                stitched_resized = cv2.resize(stitched, (960, 540))
                cv2.imshow("Stitched Image", stitched_resized)
            else:
                print("拼接失败，错误码：{}".format(status))
                cv2.imshow("Stitched Image", black_img)

            t_total_end = time.time()
            print(f"[耗时] 总耗时: {(t_total_end - t_total_start)*1000:.2f} ms\n")

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())

