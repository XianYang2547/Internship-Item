#include <iostream>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <opencv2/stitching.hpp> 
#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>

// ---------- 全局内容 ----------
std::queue<cv::Mat> frame_queue_1;
std::queue<cv::Mat> frame_queue_2;
std::mutex mtx_1, mtx_2;
std::condition_variable cv_1, cv_2;
std::atomic<bool> exit_flag = false;

// ---------- 相机参数结构 ----------
struct UndistortParams {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat newCameraMatrix;
    cv::Rect roi;
    cv::Size targetSize = cv::Size(960, 540);
    cv::Size initSize;
    bool crop = true;
};

// ---------- 初始化函数 ----------
void initUndistortParams(const cv::Mat& frame, UndistortParams& params) {
    params.initSize = frame.size();
    params.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        1400, 0, frame.cols / 2.0,
        0, 1400, frame.rows / 2.0,
        0, 0, 1);
    params.distCoeffs = (cv::Mat_<double>(1, 5) << -0.35, 0.15, 0.0005, 0.0005, -0.05);
    params.newCameraMatrix = cv::getOptimalNewCameraMatrix(params.cameraMatrix, params.distCoeffs, frame.size(), 1, frame.size(), &params.roi);
    params.targetSize = cv::Size(960, 540);
    params.crop = true;
}

// ---------- 去畸变 + resize ----------
void undistortAndResize(cv::Mat& frame, const UndistortParams& params) {
    cv::Mat undistorted;
    cv::undistort(frame, undistorted, params.cameraMatrix, params.distCoeffs, params.newCameraMatrix);
    if (params.crop) {
        undistorted = undistorted(params.roi);
    }
    cv::resize(undistorted, frame, params.targetSize);
}

// ---------- stitcher设置 ----------
void configure_stitcher(cv::Ptr<cv::Stitcher>& stitcher) {
    using namespace cv::detail;
    stitcher->setFeaturesMatcher(cv::makePtr<BestOf2NearestMatcher>(false, 0.3f));
    stitcher->setExposureCompensator(ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS));
    stitcher->setSeamFinder(cv::makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR));
    stitcher->setBlender(Blender::createDefault(Blender::MULTI_BAND, false));
    stitcher->setPanoConfidenceThresh(0.3);
    stitcher->setWaveCorrection(true);
    stitcher->setWaveCorrectKind(WAVE_CORRECT_HORIZ);  
}

// ---------- 读取rtsp流 ----------
void read_frame_from_rtsp(const std::string& rtsp_url, std::queue<cv::Mat>& frame_queue, 
    std::mutex& mtx, std::condition_variable& cv, const std::string& name) {
    cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << name << " E: 无法打开 RTSP 流" << std::endl;
        return;
    }

    int width = 1920, height = 1080;
    cv::Mat frame;

    while (!exit_flag) {
        cap >> frame; 
        if (frame.empty()) {
            std::cerr << name << " E: 读取帧失败" << std::endl;
            break;
        }

        std::unique_lock<std::mutex> lock(mtx);
        if (frame_queue.size() >= 5) {
            frame_queue.pop(); 
        }
        frame_queue.push(frame);
        cv.notify_one();
    }
}

// ---------- 推送图像流 ----------
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
    // 相机1 & 相机2
    bool is_camera1_initialized = false;
    bool is_camera2_initialized = false;

    UndistortParams undistortParams1;
    UndistortParams undistortParams2;

    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    configure_stitcher(stitcher);

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        // 获取帧
        cv::Mat frame1, frame2;
        {
            std::unique_lock<std::mutex> lock1(mtx_1);
            cv_1.wait(lock1, []() { return !frame_queue_1.empty() || exit_flag.load(); });
            if (exit_flag && frame_queue_1.empty()) break;
            frame1 = frame_queue_1.front();
            frame_queue_1.pop();
        }

        {
            std::unique_lock<std::mutex> lock2(mtx_2);
            cv_2.wait(lock2, [](){ return !frame_queue_2.empty() || exit_flag.load(); });
            if (exit_flag && frame_queue_2.empty()) break;
            frame2 = frame_queue_2.front();
            frame_queue_2.pop();
        }

        if (frame1.empty() || frame2.empty()) {
            std::cerr << "one of the frame is empty. Skip..." << std::endl;
            continue;
        }
        // 相机各自初始化
        if (!is_camera1_initialized) {
            initUndistortParams(frame1, undistortParams1);
            is_camera1_initialized = true;
        }
        if (!is_camera2_initialized) {
            initUndistortParams(frame2, undistortParams2);
            is_camera2_initialized = true;
        }
        // 帧尺寸变化时判断并重新初始化(一般不会)
        if (frame1.size() != undistortParams1.initSize) {
            initUndistortParams(frame1, undistortParams1);
        }
        if (frame2.size() != undistortParams2.initSize) {
            initUndistortParams(frame2, undistortParams2);
        }
        
        // 去畸变+resize
        undistortAndResize(frame1, undistortParams1);
        undistortAndResize(frame2, undistortParams2);

        // 拼接准备
        std::vector<cv::Mat> images = { frame1, frame2 };
        cv::Mat result;

        try {
            auto start1 = std::chrono::high_resolution_clock::now();
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
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << "stitcher cost: " << duration.count() << " ms" << std::endl;
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

        // 显示
        cv::Mat resized_result;
        cv::resize(result, resized_result, cv::Size(960, 540));
        cv::imshow("Stitched Image", resized_result);

        // 推流
        my_rtsp(resized_result, "rtsp://172.16.20.231:8554/mystream", 960, 540, 30);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "cost: " << duration.count() << " ms" << std::endl;
        
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            exit_flag = true;
            cv_1.notify_all();
            cv_2.notify_all();
            break;
        }
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
// g++ -o stitching_old stitching_old.cpp `pkg-config --cflags --libs opencv4`

