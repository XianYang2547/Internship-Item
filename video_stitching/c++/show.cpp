#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

std::queue<cv::Mat> frame_queue_1;
std::queue<cv::Mat> frame_queue_2;
std::mutex mtx_1, mtx_2;
std::condition_variable cv_1, cv_2;

void read_frame_from_rtsp(const std::string& rtsp_url, std::queue<cv::Mat>& frame_queue, std::mutex& mtx, std::condition_variable& cv, const std::string& name) {
    // 使用 FFmpeg 解码 RTSP 流
    cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);  // 强制使用 FFmpeg 解码器
    if (!cap.isOpened()) {
        std::cerr << name << " 错误: 无法打开 RTSP 流" << std::endl;
        return;
    }

    int width = 1920, height = 1080;
    cv::Mat frame;

    while (true) {
        cap >> frame;  // 从 RTSP 流中读取帧
        if (frame.empty()) {
            std::cerr << name << " 错误: 读取帧失败" << std::endl;
            break;
        }

        // 确保队列不溢出，丢弃旧帧
        std::unique_lock<std::mutex> lock(mtx);
        if (frame_queue.size() >= 5) {
            frame_queue.pop();  // 丢弃旧帧
        }
        frame_queue.push(frame);
        cv.notify_one();  // 通知主线程新帧可用
    }
}

void display_video() {
    cv::namedWindow("Video Stream", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame1, frame2;

        // 从两个队列中获取帧
        {
            std::unique_lock<std::mutex> lock1(mtx_1);
            cv_1.wait(lock1, [](){ return !frame_queue_1.empty(); });
            frame1 = frame_queue_1.front();
            frame_queue_1.pop();
        }

        {
            std::unique_lock<std::mutex> lock2(mtx_2);
            cv_2.wait(lock2, [](){ return !frame_queue_2.empty(); });
            frame2 = frame_queue_2.front();
            frame_queue_2.pop();
        }

        // 合并两个视频流
        cv::Mat result;
        //黑色摄像头画面翻转
        // cv::Mat frame2_rotated;
        // cv::rotate(frame2, frame2_rotated, cv::ROTATE_180);
        cv::hconcat(frame1, frame2, result);
        cv::imshow("Video Stream", result);

        if (cv::waitKey(1) == 'q' || cv::waitKey(1) == 27) {
            break;  // 按 'q' 或 'Esc' 键退出
        }
    }

    cv::destroyAllWindows();
}

int main() {
    std::string rtsp_url_1 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/101";
    std::string rtsp_url_2 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/201";

    // 启动线程读取视频流
    std::thread rtsp_thread_1(read_frame_from_rtsp, rtsp_url_1, std::ref(frame_queue_1), std::ref(mtx_1), std::ref(cv_1), "Stream1");
    std::thread rtsp_thread_2(read_frame_from_rtsp, rtsp_url_2, std::ref(frame_queue_2), std::ref(mtx_2), std::ref(cv_2), "Stream2");

    // 启动显示视频流的线程
    display_video();

    rtsp_thread_1.join();
    rtsp_thread_2.join();

    return 0;
}

// g++ -o show show.cpp `pkg-config --cflags --libs opencv4`