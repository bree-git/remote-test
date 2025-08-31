#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <numeric>  // 用于代价计算中的累加操作

using namespace cv;
using namespace std;

// 系统配置参数
const int FPS_SET = 30;
const int MORPH_KERNEL_SIZE = 50;
const int MIN_LED_AREA = 20;
const int MAX_LED_AREA = 600;
const int MIN_SUB_COMPONENTS = 5;
const int SMOOTH_WINDOW_SIZE = 5;
const double COORD_FONT_SCALE = 0.7;

// 相机标定参数存储
Mat cameraMatrix1, distCoeffs1;
Mat cameraMatrix2, distCoeffs2;
Mat R, T, R1, R2, P1, P2, Q;
Mat map1x, map1y, map2x, map2y;

// 实际相机分辨率（需与标定参数匹配）
const Size CAM_RESOLUTION(640, 480);

// 相机初始化函数
bool setupCameras(VideoCapture& cap1, VideoCapture& cap2) {
#ifdef _WIN32
    int cam1_index = 0, cam2_index = 2;
#else
    int cam1_index = 0;
    int cam2_index = 2;
#endif

    cap1.open(cam1_index);
    cap2.open(cam2_index);

    if (!cap1.isOpened() || !cap2.isOpened()) {
        cerr << "错误：相机初始化失败！请检查：" << endl;
        cerr << "左相机索引: " << cam1_index << endl;
        cerr << "右相机索引: " << cam2_index << endl;
        return false;
    }

    auto configureCamera = [](VideoCapture& cap) {
        cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION.width);
        cap.set(CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION.height);
        cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);
        cap.set(CAP_PROP_EXPOSURE, -11);
        cap.set(CAP_PROP_FPS, FPS_SET);
        };

    configureCamera(cap1);
    configureCamera(cap2);
    return true;
}

// 加载标定数据
bool loadCalibrationData() {
    FileStorage fs("calibration.yml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Calibration file missing!" << endl;
        return false;
    }

    fs["cameraMatrix1"] >> cameraMatrix1;
    fs["distCoeffs1"] >> distCoeffs1;
    fs["cameraMatrix2"] >> cameraMatrix2;
    fs["distCoeffs2"] >> distCoeffs2;
    fs["R"] >> R;
    fs["T"] >> T;

    stereoRectify(cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2, CAM_RESOLUTION,
        R, T, R1, R2, P1, P2, Q);

    // 新增Q矩阵输出
    cout << "\n========== Q Matrix ==========\n";
    cout << Q << endl;
    cout << "=============================\n" << endl;

    initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1,
        CAM_RESOLUTION, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2,
        CAM_RESOLUTION, CV_32FC1, map2x, map2y);
    return true;
}

// =========================================================
// LED代价计算函数（从第二个代码中引入）
// =========================================================

// 亮度差代价：计算左右相机对应LED点的亮度差异
float brightnessCost(const Mat& gray1, const Mat& gray2,
    const vector<Point2f>& pts1, const vector<Point2f>& pts2) {
    if (pts1.size() != 4 || pts2.size() != 4) return 1e5;

    float cost = 0;
    for (int i = 0; i < 4; ++i) {
        // 确保点在图像范围内
        if (pts1[i].x >= 0 && pts1[i].y >= 0 &&
            pts1[i].x < gray1.cols && pts1[i].y < gray1.rows &&
            pts2[i].x >= 0 && pts2[i].y >= 0 &&
            pts2[i].x < gray2.cols && pts2[i].y < gray2.rows) {
            int b1 = gray1.at<uchar>(pts1[i]);
            int b2 = gray2.at<uchar>(pts2[i]);
            cost += abs(b1 - b2);
        }
    }
    return cost / 4.0f;
}

// 视差一致性代价：计算4个LED点视差的方差
float disparityConsistency(const vector<Point2f>& pts1, const vector<Point2f>& pts2) {
    if (pts1.size() != 4 || pts2.size() != 4) return 1e5;

    vector<float> disparities;
    for (int i = 0; i < 4; ++i) {
        disparities.push_back(abs(pts1[i].x - pts2[i].x));
    }
    float mean_disp = accumulate(disparities.begin(), disparities.end(), 0.0f) / 4.0f;
    float var = 0;
    for (float d : disparities) var += pow(d - mean_disp, 2);
    return var;
}

// 结构一致性代价：评估LED阵列的几何结构规则性
float structureConsistency(const vector<Point2f>& pts) {
    if (pts.size() != 4) return 1e5;
    float d1 = norm(pts[0] - pts[1]);  // 上边两点距离
    float d2 = norm(pts[2] - pts[3]);  // 下边两点距离
    float d3 = norm(pts[0] - pts[2]);  // 左边两点距离
    float d4 = norm(pts[1] - pts[3]);  // 右边两点距离
    float avg_h = (d1 + d2) / 2;      // 平均水平间距
    float avg_v = (d3 + d4) / 2;      // 平均垂直间距
    return abs(d1 - d2) + abs(d3 - d4) + abs(avg_h - avg_v);
}

// =========================================================
// 修改后的LED阵列检测函数
// =========================================================
struct LEDDetectionResult {
    Point2f center;
    vector<Point2f> led_points;
};

LEDDetectionResult findLEDArray(Mat gray_frame, Mat& processed_img) {
    Mat processFrame;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE));

    GaussianBlur(gray_frame, processFrame, Size(5, 5), 0);
    morphologyEx(processFrame, processFrame, MORPH_DILATE, kernel);

    Mat labels, stats, centroids;
    int nccomps = connectedComponentsWithStats(processFrame, labels, stats, centroids);

    Point2f img_center(gray_frame.cols / 2, gray_frame.rows / 2);
    double min_dist = numeric_limits<double>::max();
    int best_idx = -1;

    for (int i = 1; i < nccomps; ++i) {
        if (i >= stats.rows || i >= centroids.rows) {
            cerr << "警告：组件索引越界" << endl;
            continue;
        }

        Rect component_rect(
            stats.at<int>(i, CC_STAT_LEFT),
            stats.at<int>(i, CC_STAT_TOP),
            stats.at<int>(i, CC_STAT_WIDTH),
            stats.at<int>(i, CC_STAT_HEIGHT)
        );

        if (component_rect.x < 0 || component_rect.y < 0 ||
            component_rect.br().x > gray_frame.cols ||
            component_rect.br().y > gray_frame.rows) {
            continue;
        }

        Mat roi = gray_frame(component_rect).clone();
        Mat sub_labels, sub_stats, sub_centroids;
        int sub_comps = connectedComponentsWithStats(roi, sub_labels, sub_stats, sub_centroids);

        // 提取所有有效LED点
        vector<Point2f> valid_leds;
        for (int j = 1; j < sub_comps; ++j) {
            if (j >= sub_stats.rows) break;

            int area = sub_stats.at<int>(j, CC_STAT_AREA);
            if (area >= MIN_LED_AREA && area <= MAX_LED_AREA) {
                Point2f led_center(
                    sub_centroids.at<double>(j, 0) + component_rect.x,
                    sub_centroids.at<double>(j, 1) + component_rect.y
                );
                valid_leds.push_back(led_center);
            }
        }

        // 只处理找到4个LED点的情况
        if (valid_leds.size() == 4) {
            // 排序LED点: 先按y坐标排序，然后按x坐标分组
            sort(valid_leds.begin(), valid_leds.end(),
                [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

            // 分为上下两行
            vector<Point2f> top_row(valid_leds.begin(), valid_leds.begin() + 2);
            vector<Point2f> bottom_row(valid_leds.begin() + 2, valid_leds.end());

            // 每行按x坐标排序
            sort(top_row.begin(), top_row.end(),
                [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
            sort(bottom_row.begin(), bottom_row.end(),
                [](const Point2f& a, const Point2f& b) { return a.x < b.x; });

            // 合并排序后的点
            vector<Point2f> sorted_leds;
            sorted_leds.push_back(top_row[0]);  // 左上
            sorted_leds.push_back(top_row[1]);  // 右上
            sorted_leds.push_back(bottom_row[0]); // 左下
            sorted_leds.push_back(bottom_row[1]); // 右下

            // 计算中心点
            Point2f center(0, 0);
            for (const auto& pt : sorted_leds) center += pt;
            center /= 4.0f;

            double dist = norm(center - img_center);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;

                // 保存LED点和中心点
                LEDDetectionResult result;
                result.center = center;
                result.led_points = sorted_leds;

                // 创建处理后的图像
                processed_img = Mat::zeros(gray_frame.size(), CV_8UC1);
                gray_frame(component_rect).copyTo(processed_img(component_rect));
                threshold(processed_img, processed_img, 1, 255, THRESH_BINARY);

                return result;
            }
        }
    }

    // 如果没有找到有效的LED阵列
    processed_img = Mat::zeros(gray_frame.size(), CV_8UC1);
    LEDDetectionResult result;
    result.center = Point2f(-1, -1);
    return result;
}

Point3f calculate3DCoordinate(Point2f pt1, Point2f pt2) {
    if (pt1.x < 0 || pt1.y < 0 || pt2.x < 0 || pt2.y < 0)
        return Point3f(-1, -1, -1);

    try {
        vector<Point2d> pts1 = { Point2d(pt1) }, pts2 = { Point2d(pt2) };
        Mat pt4d;

        triangulatePoints(P1, P2, pts1, pts2, pt4d);

        if (pt4d.rows != 4 || pt4d.cols != 1) {
            throw runtime_error("Invalid triangulation result");
        }

        Mat pt3d;
        convertPointsFromHomogeneous(pt4d.reshape(4, 1).t(), pt3d);

        return Point3f(
            static_cast<float>(pt3d.at<double>(0, 0)),
            static_cast<float>(pt3d.at<double>(0, 1)),
            static_cast<float>(pt3d.at<double>(0, 2))
        );
    }
    catch (const Exception& e) {
        cerr << "OpenCV Error: " << e.what() << endl;
        return Point3f(-1, -1, -1);
    }
}

void processAndDisplayFrames(VideoCapture& cap1, VideoCapture& cap2) {
    deque<Point2f> cam1_history, cam2_history;

    namedWindow("Camera1 Processed", WINDOW_NORMAL);
    namedWindow("Camera2 Processed", WINDOW_NORMAL);
    resizeWindow("Camera1 Processed", 680, 480);
    resizeWindow("Camera2 Processed", 680, 480);

    while (true) {
        Mat frame1, frame2;
        if (!cap1.read(frame1) || !cap2.read(frame2)) {
            cerr << "帧读取失败！" << endl;
            break;
        }

        Mat frame1_rect, frame2_rect;
        remap(frame1, frame1_rect, map1x, map1y, INTER_LINEAR);
        remap(frame2, frame2_rect, map2x, map2y, INTER_LINEAR);

        Mat gray1, gray2, bin1, bin2;
        cvtColor(frame1_rect, gray1, COLOR_BGR2GRAY);
        cvtColor(frame2_rect, gray2, COLOR_BGR2GRAY);

        // 1. 先高斯模糊
        GaussianBlur(gray1, gray1, Size(5, 5), 0);
        GaussianBlur(gray2, gray2, Size(5, 5), 0);

        // 2. 使用大津法
        threshold(gray1, bin1, 0, 255, THRESH_BINARY | THRESH_OTSU);
        threshold(gray2, bin2, 0, 255, THRESH_BINARY | THRESH_OTSU);

        Mat proc1, proc2;
        LEDDetectionResult result1 = findLEDArray(bin1, proc1);
        LEDDetectionResult result2 = findLEDArray(bin2, proc2);

        Point2f center1 = result1.center;
        Point2f center2 = result2.center;
        vector<Point2f> leds1 = result1.led_points;
        vector<Point2f> leds2 = result2.led_points;

        // 计算代价（仅在检测到完整LED阵列时）
        float bCost = 1e5, dCost = 1e5, sCost = 1e5;
        if (leds1.size() == 4 && leds2.size() == 4) {
            bCost = brightnessCost(gray1, gray2, leds1, leds2);
            dCost = disparityConsistency(leds1, leds2);
            sCost = structureConsistency(leds1);
        }

        auto updateHistory = [](deque<Point2f>& history, Point2f new_point) {
            if (new_point.x > 0 && new_point.y > 0) {
                history.push_back(new_point);
                if (history.size() > SMOOTH_WINDOW_SIZE)
                    history.pop_front();
            }
            else if (!history.empty()) {
                history.pop_front();
            }
            };

        updateHistory(cam1_history, center1);
        updateHistory(cam2_history, center2);

        auto getSmoothed = [](const deque<Point2f>& history) {
            if (history.empty()) return Point2f(-1, -1);
            Point2f sum(0, 0);
            for (const auto& pt : history) sum += pt;
            return sum / (float)history.size();
            };

        Point2f smooth1 = getSmoothed(cam1_history);
        Point2f smooth2 = getSmoothed(cam2_history);

        Point3f world_coord = calculate3DCoordinate(smooth1, smooth2);

        auto drawResults = [](Mat& display, const Point2f& pt2D, const Point3f& pt3D,
            const vector<Point2f>& leds, float bCost, float dCost, float sCost) {
                Mat color_display;
                cvtColor(display, color_display, COLOR_GRAY2BGR);

                // 绘制LED点
                if (!leds.empty()) {
                    for (const auto& pt : leds) {
                        circle(color_display, pt, 5, Scalar(0, 255, 255), -1);
                    }
                }

                if (pt2D.x > 0 && pt2D.y > 0) {
                    string coord2D = format("2D: (%.1f, %.1f)", pt2D.x, pt2D.y);
                    putText(color_display, coord2D, Point(20, 40),
                        FONT_HERSHEY_SIMPLEX, COORD_FONT_SCALE,
                        Scalar(0, 255, 0), 2);

                    if (pt3D.z > 0) {
                        string coord3D = format("3D: (%.0f, %.0f, %.0f)cm",
                            pt3D.x * 0.1, pt3D.y * 0.1, pt3D.z * 0.1);
                        putText(color_display, coord3D, Point(20, 80),
                            FONT_HERSHEY_SIMPLEX, COORD_FONT_SCALE,
                            Scalar(0, 255, 255), 2);
                    }

                    // 显示代价信息
                    string costInfo = format("Cost: B:%.1f D:%.1f S:%.1f", bCost, dCost, sCost);
                    putText(color_display, costInfo, Point(20, 120),
                        FONT_HERSHEY_SIMPLEX, COORD_FONT_SCALE,
                        Scalar(200, 200, 0), 2);
                }
                return color_display;
            };

        Mat display1 = drawResults(proc1, smooth1, world_coord, leds1, bCost, dCost, sCost);
        Mat display2 = drawResults(proc2, smooth2, world_coord, leds2, bCost, dCost, sCost);

        imshow("Original Camera1", frame1);
        imshow("Original Camera2", frame2);

        imshow("Rectified Camera1", frame1_rect);
        imshow("Rectified Camera2", frame2_rect);

        if (!display1.empty()) imshow("Camera1 Processed", display1);
        if (!display2.empty()) imshow("Camera2 Processed", display2);

        if (waitKey(10) == 'q') break;
    }
}


int main() {
    VideoCapture cap1, cap2;
    if (!setupCameras(cap1, cap2)) return -1;
    if (!loadCalibrationData()) return -1;

    try {
        processAndDisplayFrames(cap1, cap2);
    }
    catch (const Exception& e) {
        cerr << "OpenCV异常: " << e.what() << endl;
    }
    catch (const exception& e) {
        cerr << "标准异常: " << e.what() << endl;
    }
    catch (...) {
        cerr << "未知异常发生！" << endl;
    }

    cap1.release();
    cap2.release();
    destroyAllWindows();
    return 0;
}