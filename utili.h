#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern vector<Mat> warped_imgs, masks;
extern map<pair<int, int>, Mat> overlaps;
extern vector<vector<int>> ref_seq;
extern map<pair<int, int>, vector<pair<Point2f, Point2f>>> Corr_vec;

// super parameters
extern int height, width;
extern bool ignore_non_corrected, pano;
extern float sigma1, sigma2;
extern float alpha;

struct ImgPack
{
public:
    vector<float> PDF;
    vector<float> CDF;
};

namespace Utils
{
    void single_img_correction(pair<int, int> tar_ref);

    // boarder wavefront
    Mat computeWavefront(Mat& overlap);

    // 新增: border擴散計算函數
    Mat computeBorderExpansion(const vector<Point>& border_pixel, int expansion_radius, const Mat& mask);

    //Fecker func
    ImgPack CalDF(Mat& src, int channel, Mat& overlap);
    Mat HM_Fecker(Mat& ref, Mat& tar, Mat& overlap);

    vector<pair<int, int>> getCorrectImgPair();
    int findFirstImg();

    //Proposed fusion color correction method
    vector<int> EC();

    //Corr based func
    Mat JBI(Mat& ref, Mat& tar, Mat& overlap, vector<pair<Point2f, Point2f>>& correspondences);

    Mat NonOverlapJBI(Mat& ori_img, Mat& correct_img, vector<Point>& border_pixel, Mat& overlap, Mat& mask);
    void nonOverlapColorCorrect(Mat& warped_tar_img, Mat& cite_range, Mat& Fecker_comp, Mat& JBI_comp, Mat& overlap, Mat& mask, vector<Point>& border_pixel);
    Mat getFusionNonOverlap(Mat& warped_tar_img, Mat& cite_range, Mat& Fecker_comp, Mat& JBI_comp, Mat& overlap, Mat& mask, vector<Point>& border_pixel);
    void getNonOverlapColorCorrectImg(Mat& warped_tar_img, Mat& fusion_comp, Mat& overlap, Mat& mask);

    //ref interval
    vector<Point> getBorderPixel(Mat& wavefrontMap);
    Mat build_range_map_with_side_addition(Mat& tar_img, vector<Point>& border_pixel, Mat& overlap, Mat& mask);

    //error map
    Mat errorMap(Mat& ref, Mat& tar, Mat& overlap, Mat& comp);

    //fusion
    void HE_JBI_fusion(Mat& warped_tar_img, Mat& Fecker_comp, Mat& Color_diff_comp,
        Mat& Fecker_error_map, Mat& Color_diff_error_map, Mat& overlap);
}

namespace Inits
{
    // load data.
    vector<string> getImgFilenameList(string& addr);
    vector<string> getCorrFilenameList(string& addr);
    vector<string> getHFilenameList(string& addr);
    vector<Mat> LoadImage(string& addr, int flag);
    map<pair<int, int>, Mat> LoadMask(string& addr, int N);
    vector<Mat> LoadOrigin(string& addr, int flag);
    map<pair<int, int>, vector<pair<Point2f, Point2f>>> LoadCorr(string& addr, int N);
    vector<Mat> LoadImgMask(string& addr, int N);
    void loadAll(string& warpDir, string& overlapDir, string& corrDir, string& maskDir);
}

namespace Tools
{
    Mat createResult(const std::vector<cv::Mat>& warped_imgs, const vector<cv::Mat>& masks);
}