#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 打开YAML文件
    FileStorage fs("/home/melo/work/new_stereo/intrinsics.yml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open file." << endl;
        return -1;
    }

    // 读取内参矩阵和畸变系数
    Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;
    fs["M1"] >> cameraMatrix1;
    fs["D1"] >> distCoeffs1;
    fs["M2"] >> cameraMatrix2;
    fs["D2"] >> distCoeffs2;
    fs.release(); // 关闭文件

    // 打开外参文件
    fs.open("/home/melo/work/new_stereo/extrinsics.yml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open extrinsics file." << endl;
        return -1;
    }

    Mat R, T;
    fs["R"] >> R;
    fs["T"] >> T;
    fs.release(); // 关闭文件

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    // 进行立体校正
    stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, Size(1280, 720), R, T, R1, R2, P1, P2, Q, 
                  CALIB_ZERO_DISPARITY, -1, Size(1280, 720), &validRoi[0], &validRoi[1]);

    // 计算校正映射
    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, Size(1280, 720), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, Size(1280, 720), CV_32FC1, map2x, map2y);

    // 图片路径列表
    // vector<string> leftImagePaths = {"/home/melo/work/new_stereo/20240702-154440/"};
    // vector<string> rightImagePaths = {"/home/melo/work/new_stereo/20240702-154440/"};

    for (size_t i = 0; i < 10; ++i) {
        string leftImagePaths = "/home/melo/work/new_stereo/20240702-161210/left_" + to_string(i) + ".jpg";
        string rightImagePaths = "/home/melo/work/new_stereo/20240702-161210/right_" + to_string(i) + ".jpg";
        // 加载图像
        Mat img1 = imread((leftImagePaths), IMREAD_COLOR);
        Mat img2 = imread((rightImagePaths), IMREAD_COLOR);
        if (img1.empty() || img2.empty()) {
            cerr << "Failed to load images: " << leftImagePaths[i] << " and/or " << rightImagePaths[i] << endl;
            continue; // 继续处理下一对图像
        }

        // 去畸变
        Mat undistortedImg1, undistortedImg2;
        undistort(img1, undistortedImg1, cameraMatrix1, distCoeffs1);
        undistort(img2, undistortedImg2, cameraMatrix2, distCoeffs2);

        // 保存去畸变后的图像
        string undistortedLeftPath = "/home/melo/work/new_stereo/undistort_left/left_" + to_string(i) + ".jpg";
        string undistortedRightPath = "/home/melo/work/new_stereo/undistort_right/right_" + to_string(i) + ".jpg";
        imwrite(undistortedLeftPath, undistortedImg1);
        imwrite(undistortedRightPath, undistortedImg2);

        // 立体校正
        Mat rectifiedImg1, rectifiedImg2;
        remap(undistortedImg1, rectifiedImg1, map1x, map1y, INTER_LINEAR);
        remap(undistortedImg2, rectifiedImg2, map2x, map2y, INTER_LINEAR);

        string rectifiedLeftPath = "/home/melo/work/new_stereo/rectified_left/left_" + to_string(i) + ".jpg";
        string rectifiedRightPath = "/home/melo/work/new_stereo/rectified_right/right_" + to_string(i) + ".jpg";
        imwrite(rectifiedLeftPath, rectifiedImg1);
        imwrite(rectifiedRightPath, rectifiedImg2);

        // 查找相同特征点（示例中使用ORB特征检测）
        Ptr<ORB> orb = ORB::create(100); // 增加特征点数量
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;

        orb->detectAndCompute(rectifiedImg1, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(rectifiedImg2, noArray(), keypoints2, descriptors2);

        BFMatcher matcher(NORM_HAMMING);
        vector<vector<DMatch>> knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        // 使用比率测试来过滤匹配
        const float ratioThresh = 0.75f;
        vector<DMatch> goodMatches;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }

        // 使用RANSAC过滤匹配
        vector<Point2f> points1, points2;
        for (size_t i = 0; i < goodMatches.size(); i++) {
            points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
            points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
        }

        Mat mask;
        Mat F = findFundamentalMat(points1, points2, RANSAC, 3, 0.99, mask);

        vector<DMatch> inliers;
        for (size_t i = 0; i < goodMatches.size(); i++) {
            if (mask.at<uchar>(i)) {
                inliers.push_back(goodMatches[i]);
            }
        }

        // 绘制匹配结果
        Mat imgMatches;
        drawMatches(rectifiedImg1, keypoints1, rectifiedImg2, keypoints2, inliers, imgMatches);
        string matchPath = "/home/melo/work/new_stereo/matches_" + to_string(i) + ".jpg";
        imwrite(matchPath, imgMatches);

        // imshow("Matches", imgMatches);
        // waitKey(0);
    }

    return 0;
}
