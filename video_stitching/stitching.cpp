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
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

// ---------- 全局内容 ----------
std::map<int, std::queue<cv::Mat>> frame_queues;
std::map<int, std::mutex> queue_mutexes;
std::map<int, std::condition_variable> queue_conds;
std::atomic<bool> exit_flag = false;


// ---------- 流配置结构 ----------
struct CameraConfig {
    std::string url;
    int queue_id;
    bool undistort;
};

// ---------- 图像校准结构 ----------
struct UndistortParams {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat newCameraMatrix;
    cv::Rect roi;
    cv::Size targetSize = cv::Size(960, 540);
    cv::Size initSize;
    bool crop = true;
};

// ---------- 自定义的校准参数 ----------
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
void read_frame_from_rtsp(const CameraConfig& cfg) {
    while (!exit_flag) {
        cv::VideoCapture cap(cfg.url, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            std::cerr << "E: 无法打开 RTSP 流: " << cfg.url << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        cv::Mat frame;
        bool read_success = true;
        
        while (!exit_flag && read_success) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "E: 读取帧失败: " << cfg.url << "，3秒后重试..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(3));
                read_success = false;
                break;
            }

            std::unique_lock<std::mutex> lock(queue_mutexes[cfg.queue_id]);
            if (frame_queues[cfg.queue_id].size() > 5) {
                frame_queues[cfg.queue_id].pop();
            }
            frame_queues[cfg.queue_id].push(frame.clone());  // 使用clone确保帧数据安全
            queue_conds[cfg.queue_id].notify_one();
        }

        // 释放VideoCapture资源
        cap.release();
        
        if (!read_success) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
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

// ---------- 读取摄像头地址列表 ----------
std::vector<CameraConfig> load_camera_configs(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        auto nodes = config["cameras"];
        if (!nodes || !nodes.IsSequence() || nodes.size() < 2)
            throw std::runtime_error("至少需要两个摄像头配置");

        std::vector<CameraConfig> configs;
        for (const auto& node : nodes) {
            CameraConfig cfg;
            cfg.url = node["url"].as<std::string>();
            cfg.queue_id = node["queue_id"].as<int>();
            cfg.undistort = node["undistort"].as<bool>();
            configs.push_back(cfg);
        }
        return configs;
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML解析失败: " + std::string(e.what()));
    }
}

void display_video(const std::vector<CameraConfig>& camera_configs) {
    bool is_first_frame = true;
    bool is_init_success = false;

    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    std::map<int, UndistortParams> undistort_params; //图像校准结构体
    std::map<int, bool> undistort_initialized; // 
    configure_stitcher(stitcher);

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        // 获取帧
        std::vector<cv::Mat> input_images;
        for (const auto& cam : camera_configs) {
            cv::Mat frame;

            // 读取对应 queue_id 的图像帧
            {
                std::unique_lock<std::mutex> lock(queue_mutexes[cam.queue_id]);
                queue_conds[cam.queue_id].wait(lock, [&]() {
                    return !frame_queues[cam.queue_id].empty() || exit_flag;
                });

                if (exit_flag && frame_queues[cam.queue_id].empty())
                    return;

                frame = frame_queues[cam.queue_id].front();
                frame_queues[cam.queue_id].pop();
            }

            // 如果配置了去畸变
            if (cam.undistort) {
                if (!undistort_initialized[cam.queue_id]) {
                    initUndistortParams(frame, undistort_params[cam.queue_id]);
                    undistort_initialized[cam.queue_id] = true;
                }
                undistortAndResize(frame, undistort_params[cam.queue_id]);
            } else {
                cv::resize(frame, frame, cv::Size(960, 540));
            }

            input_images.push_back(frame);
        }

        // 拼接准备
        cv::Mat result;
        try {
            auto start1 = std::chrono::high_resolution_clock::now();
            if (is_first_frame) {
                cv::Stitcher::Status status = stitcher->estimateTransform(input_images);
                if (status != cv::Stitcher::OK) {
                    std::cerr << "E: estimateTransform 失败，状态码: " << int(status) << std::endl;
                    continue;
                }
                is_first_frame = false;
                is_init_success = true;
                std::cout << "初始化成功，拼接参数已锁定。" << std::endl;
            }

            if (is_init_success) {
                stitcher->composePanorama(input_images, result);
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
            break;
        }
    }

    cv::destroyAllWindows();
}

int main() {
    std::string config_path = "config.yaml";
    std::vector<CameraConfig> camera_configs;

    try {
        camera_configs = load_camera_configs(config_path);
    } catch (const std::exception& ex) {
        std::cerr << "配置加载失败: " << ex.what() << std::endl;
        return -1;
    }

    // 初始化用于每个队列的 mutex、cond 和 queue
    for (const auto& cam : camera_configs) {
        int qid = cam.queue_id;
        if (frame_queues.find(qid) == frame_queues.end()) {
            frame_queues[qid] = std::queue<cv::Mat>();
            queue_mutexes[qid];  
            queue_conds[qid];
        }
    }

    // 启动摄像头线程
    std::vector<std::thread> threads;
    for (const auto& cam : camera_configs) {
        threads.emplace_back(read_frame_from_rtsp, cam);
    }

    // 启动拼接显示线程
    display_video(camera_configs);

    // 等待线程结束
    for (auto& t : threads) {
        if (t.joinable())
            t.join();
    }

    return 0;
}


// ffplay -rtsp_flags prefer_tcp -fflags nobuffer rtsp://172.16.20.231:8554/mystream
// g++ -o stitching stitching.cpp `pkg-config --cflags --libs opencv4 yaml-cpp`

