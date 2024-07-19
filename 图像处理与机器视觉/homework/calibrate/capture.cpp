#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <ctime>
#include <sys/stat.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>

using namespace cv;

static void print_help(char** argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: %s <left_image> <right_image> [--algorithm=bm|sgbm|hh|hh4|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [--color] [-o=<disparity_image>] [-p=<point_cloud_file>]\n", argv[0]);
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

const int CAM_INDEX = 0;
const cv::Size FRAME_SIZE(2560, 720);
const cv::Size show_size(1280, 480);
std::string save_dir="/home/melo/work/new_stereo/pictures";

// void init_dir() {
//     std::string current_path = "./";
//     if (!std::experimental::filesystem::exists(current_path + "/save_img")) {
//         std::experimental::filesystem::create_directory(current_path + "/save_img");
//     }

//     save_dir = current_path + "/save_img/" + time;
//     if (!std::experimental::filesystem::exists(save_dir)) {
//         std::experimental::filesystem::create_directory(save_dir);
//     }
// }

void print_info() {
    std::cout << "====================== INFO ======================" << std::endl;
    std::cout << "VIDEOCAPTURE INDEX:      " << CAM_INDEX << std::endl;
    std::cout << "DUAL CAMERA IMAGE SIZE:  [W: " << FRAME_SIZE.width << ", H: " << FRAME_SIZE.height << "]" << std::endl;
    std::cout << "DISPLAY IMAGE SIZE:      [W: " << show_size.width << ", H: " << show_size.height << "]" << std::endl;
    std::cout << "IMAGE SAVE PATH:         " << save_dir << std::endl;
    std::cout << "" << std::endl;
    std::cout << "PRESS 'S' SAVE IMAGE" << std::endl;
    std::cout << "PRESS 'ESC' QUIT" << std::endl;
    std::cout << "====================== INFO ======================" << std::endl;
}

static int stereo_match(int argc, char** argv)
{
    std::string img1_filename = "/home/melo/work/new_stereo/picutres/left.jpg";
    std::string img2_filename = "/home/melo/work/new_stereo/pictures/right.jpg";
    std::string intrinsic_filename = "/home/melo/work/new_stereo/intrinsics.yml";
    std::string extrinsic_filename = "/home/melo/work/new_stereo/extrinsics.yml";
    std::string disparity_filename = "/home/melo/work/new_stereo/disparity.png";
    std::string point_cloud_filename = "/home/melo/work/new_stereo/cloudpoint.ply";

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4, STEREO_HH4=5 };
    int alg = STEREO_SGBM;
    int SADWindowSize, numberOfDisparities;
    bool no_display;
    bool color_display;
    float scale;

    Ptr<StereoBM> bm = StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    cv::CommandLineParser parser(argc, argv,
        "{@arg1|/home/melo/work/new_stereo/pictures/left.jpg|}{@arg2|/home/melo/work/new_stereo/pictures/right.jpg|}{help h||}{algorithm|sgbm|}{max-disparity|256|}{blocksize|+3|}{no-display||}{color|1|}{scale|1|}{i|/home/melo/work/stereo/intrinsics.yml|}{e|/home/melo/work/stereo/extrinsics.yml|}{o||}{p||}");
    if(parser.has("help"))
    {
        print_help(argv);
        return 0;
    }
    // img1_filename = "/home/melo/work/new_stereo/pictures/left.jpg";
    // img2_filename = "/home/melo/work/new_stereo/pictures/right.jpg"
    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = _alg == "bm" ? STEREO_BM :
            _alg == "sgbm" ? STEREO_SGBM :
            _alg == "hh" ? STEREO_HH :
            _alg == "var" ? STEREO_VAR :
            _alg == "hh4" ? STEREO_HH4 :
            _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    color_display = parser.has("color");
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");
    if( parser.has("o") )
        disparity_filename = parser.get<std::string>("o");
    if( parser.has("p") )
        point_cloud_filename = parser.get<std::string>("p");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if( alg < 0 )
    {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        print_help(argv);
        return -1;
    }
    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        print_help(argv);
        return -1;
    }
    if (scale < 0)
    {
        printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        return -1;
    }
    if( img1_filename.empty() || img2_filename.empty() )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
    if( (!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if( extrinsic_filename.empty() && !point_cloud_filename.empty() )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if( !intrinsic_filename.empty() )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_HH4)
        sgbm->setMode(StereoSGBM::MODE_HH4);
    else if(alg==STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    float disparity_multiplier = 1.0f;
    if( alg == STEREO_BM )
    {
        bm->compute(img1, img2, disp);
        if (disp.type() == CV_16S)
            disparity_multiplier = 16.0f;
    }
    else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_HH4 || alg == STEREO_3WAY )
    {
        sgbm->compute(img1, img2, disp);
        if (disp.type() == CV_16S)
            disparity_multiplier = 16.0f;
    }
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);

    Mat disp8_3c;
    if (color_display)
        cv::applyColorMap(disp8, disp8_3c, COLORMAP_TURBO);

    if(!disparity_filename.empty())
        imwrite(disparity_filename, color_display ? disp8_3c : disp8);
        

    if(!point_cloud_filename.empty())
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        Mat floatDisp;
        disp.convertTo(floatDisp, CV_32F, 1.0f / disparity_multiplier);
        reprojectImageTo3D(floatDisp, xyz, Q, true);
        saveXYZ(point_cloud_filename.c_str(), xyz);
        printf("\n");
    }

    if( !no_display )
    {
        std::ostringstream oss;
        oss << "disparity  " << (alg==STEREO_BM ? "bm" :
                                 alg==STEREO_SGBM ? "sgbm" :
                                 alg==STEREO_HH ? "hh" :
                                 alg==STEREO_VAR ? "var" :
                                 alg==STEREO_HH4 ? "hh4" :
                                 alg==STEREO_3WAY ? "sgbm3way" : "");
        oss << "  blocksize:" << (alg==STEREO_BM ? SADWindowSize : sgbmWinSize);
        oss << "  max-disparity:" << numberOfDisparities;
        std::string disp_name = oss.str();

        // namedWindow("left", cv::WINDOW_NORMAL);
        // imshow("left", img1);
        // namedWindow("right", cv::WINDOW_NORMAL);
        // imshow("right", img2);
        // cv::Size window_size(1280, 480);
        namedWindow(disp_name, cv::WINDOW_AUTOSIZE);
        cv::resizeWindow(disp_name, 640, 240);
        imshow(disp_name, color_display ? disp8_3c : disp8);

        // printf("press ESC key or CTRL+C to close...");
        // fflush(stdout);
        // printf("\n");
        // while(1)
        // {
        //     if(waitKey() == 27) //ESC (prevents closing on actions like taking screenshots)
        //         break;
        // }
    }

    return 0;
}

int main(int argc, char** argv) {
    std::cout << "============== DUAL CAMERA SHUTTER ==============" << std::endl;
    std::cout << "LOADING DUAL CAMERA PLEASE WAITING..." << std::endl;

    cv::VideoCapture cap(CAM_INDEX);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_SIZE.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_SIZE.height);

    // init_dir();
    print_info();

    int save_count = 0;

    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            std::cout << "Empty frame" << std::endl;
            break;
        }

        cv::Mat resize_frame;
        cv::resize(frame, resize_frame, show_size);
        cv::imshow("dual_camera", resize_frame);

        char key = cv::waitKey(1);
        if (key == 27) {
            break;
        }

        if (key == 's') {
            std::string left_img_path = save_dir + "/left" +  ".jpg";
            std::string right_img_path = save_dir + "/right"  + ".jpg";

            cv::Mat left_img = frame(cv::Rect(0, 0, FRAME_SIZE.width / 2, FRAME_SIZE.height));
            cv::Mat right_img = frame(cv::Rect(FRAME_SIZE.width / 2, 0, FRAME_SIZE.width / 2, FRAME_SIZE.height));

            cv::imwrite(left_img_path, left_img);
            cv::imwrite(right_img_path, right_img);
            save_count++;

            std::cout << "[INFO]left  img save: " << left_img_path << std::endl;
            std::cout << "[INFO]right img save: " << right_img_path << std::endl;
            std::cout << "" << std::endl;
            stereo_match(argc, argv);
        }
        // std::string left_img_path = save_dir + "/left" +  ".jpg";
        // std::string right_img_path = save_dir + "/right"  + ".jpg";

        // cv::Mat left_img = frame(cv::Rect(0, 0, FRAME_SIZE.width / 2, FRAME_SIZE.height));
        // cv::Mat right_img = frame(cv::Rect(FRAME_SIZE.width / 2, 0, FRAME_SIZE.width / 2, FRAME_SIZE.height));

        // cv::imwrite(left_img_path, left_img);
        // cv::imwrite(right_img_path, right_img);
        // save_count++;

        // std::cout << "[INFO]left  img save: " << left_img_path << std::endl;
        // std::cout << "[INFO]right img save: " << right_img_path << std::endl;
        // std::cout << "" << std::endl;
        // stereo_match(argc, argv);
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
