#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <opencv2/stitching.hpp> 
#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>

std::queue<cv::Mat> frame_queue_1;
std::queue<cv::Mat> frame_queue_2;
std::mutex mtx_1, mtx_2;
std::condition_variable cv_1, cv_2;


void configure_stitcher(cv::Ptr<cv::Stitcher>& stitcher) {
    using namespace cv::detail;
    using namespace cv;
    // 设置特征匹配器（BestOf2NearestMatcher）
    stitcher->setFeaturesMatcher(makePtr<BestOf2NearestMatcher>(false, 0.3f));
    // 设置曝光补偿（GAIN_BLOCKS 是默认值，可以不设）
    stitcher->setExposureCompensator(ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS));
    // 设置接缝查找器（GraphCutSeamFinder）
    stitcher->setSeamFinder(makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR));
    // 设置混合器（MULTI_BAND 混合）
    stitcher->setBlender(Blender::createDefault(Blender::MULTI_BAND, false));
    // 设置全景置信度阈值（默认 1.0，这里设为 0.3 以放宽匹配要求）
    stitcher->setPanoConfidenceThresh(0.3);
    // 启用波形校正（默认 true，WAVE_CORRECT_HORIZ 是默认值）
    stitcher->setWaveCorrection(true);
    stitcher->setWaveCorrectKind(WAVE_CORRECT_HORIZ);  
}

void read_frame_from_rtsp(const std::string& rtsp_url, std::queue<cv::Mat>& frame_queue, 
    std::mutex& mtx, std::condition_variable& cv, const std::string& name) {
    // 使用 FFmpeg 解码 RTSP 流
    cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << name << " E: 无法打开 RTSP 流" << std::endl;
        return;
    }

    int width = 1920, height = 1080;
    cv::Mat frame;

    while (true) {
        cap >> frame; 
        if (frame.empty()) {
            std::cerr << name << " E: 读取帧失败" << std::endl;
            break;
        }

        // 确保队列不溢出，丢弃旧帧
        std::unique_lock<std::mutex> lock(mtx);
        if (frame_queue.size() >= 5) {
            frame_queue.pop(); 
        }
        frame_queue.push(frame);
        cv.notify_one();
    }
}

void my_rtsp(const cv::Mat& frame, const std::string& rtsp_url, int width, int height, int fps=30) {
    static FILE* ffmpeg_pipe = nullptr;
    static pid_t ffmpeg_pid = -1;

    if (frame.empty() || frame.cols != width || frame.rows != height || frame.type() != CV_8UC3) {
        std::cerr << "Frame must be CV_8UC3 with size " << width << "x" << height << std::endl;
        return;
    }

    if (!ffmpeg_pipe) {
        int pipe_fd[2];
        if (pipe(pipe_fd) == -1) {
            std::cerr << "Failed to create pipe!" << std::endl;
            return;
        }

        ffmpeg_pid = fork();
        if (ffmpeg_pid == -1) {
            std::cerr << "Failed to fork!" << std::endl;
            return;
        }

        if (ffmpeg_pid == 0) {
            close(pipe_fd[1]);
            dup2(pipe_fd[0], STDIN_FILENO);
            close(pipe_fd[0]);

            execlp("ffmpeg", "ffmpeg",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", (std::to_string(width) + "x" + std::to_string(height)).c_str(),
                "-r", std::to_string(fps).c_str(),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-tune", "zerolatency",
                "-g", "30",
                "-f", "rtsp",
                rtsp_url.c_str(),
                nullptr);

            std::cerr << "Failed to execute FFmpeg!" << std::endl;
            exit(1);
        } else {
            close(pipe_fd[0]);
            ffmpeg_pipe = fdopen(pipe_fd[1], "wb");
        }
    }

    if (ffmpeg_pipe) {
        fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpeg_pipe);
        fflush(ffmpeg_pipe);
    }
}

void display_video() {
    bool is_first_frame = true;
    bool is_init_success = false;

    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    configure_stitcher(stitcher);

    while (true) {
        cv::Mat frame1, frame2;
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

        cv::resize(frame1, frame1, cv::Size(960, 540));
        cv::resize(frame2, frame2, cv::Size(960, 540));

        std::vector<cv::Mat> images = { frame1, frame2 };
        cv::Mat result;

        try {
            auto start = std::chrono::high_resolution_clock::now();

            if (is_first_frame) {
                cv::Stitcher::Status status = stitcher->estimateTransform(images);
                if (status != cv::Stitcher::OK) {
                    std::cerr << "E: estimateTransform 失败，状态码: " << int(status) << std::endl;
                    continue;
                }
                is_first_frame = false;
                is_init_success = true;
                std::cout << "初始化成功，拼接参数已锁定。" << std::endl;
            }

            if (is_init_success) {
                stitcher->composePanorama(images, result);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "cost: " << duration.count() << " ms" << std::endl;

            if (result.empty()) {
                std::cerr << "E: 拼接结果为空。" << std::endl;
                cv::imshow("Stitched Image", cv::Mat::zeros(cv::Size(960, 540), CV_8UC3));
                continue;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "E: 图像拼接失败: " << e.what() << std::endl;
            cv::imshow("Stitched Image", cv::Mat::zeros(cv::Size(960, 540), CV_8UC3));
            continue;
        }

        cv::Mat resized_result;
        cv::resize(result, resized_result, cv::Size(960, 540));
        cv::imshow("Stitched Image", resized_result);

        my_rtsp(resized_result, "rtsp://172.16.20.231:8554/mystream", 960, 540, 30);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cv::destroyAllWindows();
}

int main() {
    std::string rtsp_url_1 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/101";
    std::string rtsp_url_2 = "rtsp://admin:yfzh123456@172.16.30.242:554/Streaming/Channels/201";

    std::thread rtsp_thread_1(read_frame_from_rtsp, rtsp_url_1, std::ref(frame_queue_1), std::ref(mtx_1), std::ref(cv_1), "Stream1");
    std::thread rtsp_thread_2(read_frame_from_rtsp, rtsp_url_2, std::ref(frame_queue_2), std::ref(mtx_2), std::ref(cv_2), "Stream2");

    display_video();

    rtsp_thread_1.join();
    rtsp_thread_2.join();

    return 0;
}

// ffplay -rtsp_flags prefer_tcp -fflags nobuffer rtsp://172.16.20.231:8554/mystream
// g++ -o stitching stitching.cpp `pkg-config --cflags --libs opencv4`

