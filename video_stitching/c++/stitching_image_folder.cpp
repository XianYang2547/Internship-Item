#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <folder_path>" << endl;
        return EXIT_FAILURE;
    }

    string folderPath = argv[1];
    vector<string> files;
    glob(folderPath, files);
    
    // 检查是否找到图像文件
    if (files.empty()) {
        cout << "No images found in the folder: " << folderPath << endl;
        return EXIT_FAILURE;
    }

    vector<Mat> images;
    for (int i = 0; i < files.size(); i++) {
        printf("image file : %s \n", files[i].c_str());
        Mat img = imread(files[i]);
        if (img.empty()) {
            cout << "Failed to load image: " << files[i] << endl;
            continue;
        }
        images.push_back(img);
    }

    // 检查是否有有效的图像被加载
    if (images.empty()) {
        cout << "No valid images to process" << endl;
        return EXIT_FAILURE;
    }

    // 设置拼接模式与参数
    Mat result;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Ptr<Stitcher> stitcher = Stitcher::create(mode);

    // 拼接方式-多通道融合
    auto blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND);
    stitcher->setBlender(blender);
    stitcher->setPanoConfidenceThresh(0.3);

    // 拼接
    Stitcher::Status status = stitcher->stitch(images, result);

    if (status != Stitcher::OK) {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return EXIT_FAILURE;
    }

    string outputFile = "./output/result.png";
    imwrite(outputFile, result);
    cout << "saved to: " << outputFile << endl;

    return 0;
}

// g++ -o stitching_image_folder stitching_image_folder.cpp `pkg-config --cflags --libs opencv4`