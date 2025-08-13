#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <numeric>
#include <cstring>

#include "utili.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace filesystem;

vector<string> Inits::getImgFilenameList(string& addr)
{
    vector<string> imgs;
    for (auto& entry : filesystem::directory_iterator(addr)) {
        if (entry.path().u8string().find(".png") != string::npos || entry.path().u8string().find(".JPG") != string::npos)
            imgs.push_back(entry.path().u8string());
    }
    return imgs;
}

vector<string> Inits::getCorrFilenameList(string& addr)
{
    vector<string> csvs;
    for (auto& entry : filesystem::directory_iterator(addr)) {
        if (entry.path().u8string().find(".csv") != string::npos)
            csvs.push_back(entry.path().u8string());
    }
    return csvs;
}

vector<string> Inits::getHFilenameList(string& addr)
{
    vector<string> csvs;
    for (auto& entry : filesystem::directory_iterator(addr)) {
        if (entry.path().u8string().find(".txt") != string::npos)
            csvs.push_back(entry.path().u8string());
    }
    return csvs;
}

vector<Mat> Inits::LoadImage(string& addr, int flag)
{
    vector<string> img_names = Inits::getImgFilenameList(addr);
    vector<Mat> imgs;
    imgs.reserve(img_names.size());
    for (const string& file_name : img_names)
        imgs.push_back(imread(file_name, flag));

    return imgs;
}

map<pair<int, int>, Mat> Inits::LoadMask(string& addr, int N)
{
    vector<vector<int>> seq(N);
    map<pair<int, int>, Mat> masks;
    vector<string> img_names = Inits::getImgFilenameList(addr);

    for (const string& file_name : img_names) {
        size_t start = file_name.find_last_of("/\\") + 1;
        string filename_only = file_name.substr(start);

        int first_num = stoi(filename_only.substr(0, 2));
        int second_num = stoi(filename_only.substr(4, 2));
        seq[first_num].push_back(stoi(file_name.substr(addr.size() + 4, 2)));
        masks[make_pair(first_num, second_num)] = imread(file_name, IMREAD_GRAYSCALE);
    }

    ref_seq = seq;
    return masks;
}

vector<Mat> Inits::LoadOrigin(string& addr, int flag)
{
    vector<string> img_names = Inits::getImgFilenameList(addr);
    vector<Mat> imgs;
    imgs.reserve(img_names.size());
    for (const string& file_name : img_names)
        imgs.push_back(imread(file_name, cv::IMREAD_COLOR));
    return imgs;
}

vector<pair<Point2f, Point2f>> readCSV(const std::string& filePath) {
    vector<pair<Point2f, Point2f>> correspondences;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return correspondences;
    }

    string line;
    while (std::getline(file, line)) {
        istringstream ss(line);
        string token;
        float src_x, src_y, tgt_x, tgt_y;

        getline(ss, token, ',');
        src_x = stof(token);
        getline(ss, token, ',');
        src_y = stof(token);
        getline(ss, token, ',');
        tgt_x = stof(token);
        getline(ss, token, ',');
        tgt_y = stof(token);

        correspondences.emplace_back(Point2f(src_x, src_y), Point2f(tgt_x, tgt_y));
    }

    file.close();
    return correspondences;
}

map<pair<int, int>, vector<pair<Point2f, Point2f>>> Inits::LoadCorr(string& addr, int N) {
    map<pair<int, int>, vector<pair<Point2f, Point2f>>> Corr_map;
    vector<string> csv_names = Inits::getCorrFilenameList(addr);

    for (const string& file_name : csv_names) {
        size_t start = file_name.find_last_of("/\\") + 1;
        string filename_only = file_name.substr(start);

        int first_num = stoi(filename_only.substr(0, 2));
        int second_num = stoi(filename_only.substr(4, 2));

        vector<pair<Point2f, Point2f>> correspondences = readCSV(file_name);

        Corr_map[make_pair(first_num, second_num)] = correspondences;
    }

    return Corr_map;
}

vector<Mat> Inits::LoadImgMask(string& addr, int N)
{
    vector<Mat> masks;
    vector<string> img_names = Inits::getImgFilenameList(addr);

    for (const string& file_name : img_names) {
        masks.push_back(imread(file_name, IMREAD_GRAYSCALE));
    }
    return masks;
}

void Inits::loadAll(string& warpDir, string& overlapDir, string& corrDir, string& maskDir)
{
    cout << "load image." << endl;
    warped_imgs = Inits::LoadImage(warpDir, IMREAD_COLOR);
    int N = (int)warped_imgs.size();

    cout << "load overlap." << endl;
    overlaps = Inits::LoadMask(overlapDir, N);

    cout << "load correspondence." << endl;
    Corr_vec = Inits::LoadCorr(corrDir, N);
    
    cout << "load img masks." << endl;
    masks = Inits::LoadImgMask(maskDir, N);
}

Mat Tools::createResult(const std::vector<cv::Mat>& warped_imgs, const vector<cv::Mat>& masks) {
    if (warped_imgs.empty() || masks.empty() || warped_imgs.size() != masks.size()) {
        cout << "Error: Invalid input images or masks" << std::endl;
        return Mat();
    }

    // Initialize output images and count matrix
    Mat result = Mat::zeros(height, width, CV_32FC3);
    Mat countMatrix = Mat::zeros(height, width, CV_32S);

    // Process each image in sequence
    for (size_t idx = 0; idx < warped_imgs.size(); ++idx) {
        // Convert image to float32
        Mat img;
        warped_imgs[idx].convertTo(img, CV_32F);

        Mat mask;
        if (masks[idx].channels() == 3) {
            cvtColor(masks[idx], mask, COLOR_BGR2GRAY);
        }
        else {
            masks[idx].copyTo(mask);
        }

        Mat binaryMask;
        threshold(mask, binaryMask, 128, 1, THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_32F);

        // Add image values to panorama where mask is 1
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float maskVal = binaryMask.at<float>(y, x);
                if (maskVal > 0) {
                    result.at<Vec3f>(y, x) += img.at<Vec3f>(y, x) * maskVal;
                    countMatrix.at<int>(y, x) += 1;
                }
            }
        }
    }

    // Average the pixel values where count > 0
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int count = countMatrix.at<int>(y, x);
            if (count > 0) {
                result.at<Vec3f>(y, x) /= count;
            }
        }
    }

    result.convertTo(result, CV_8U);
    return result;
}

//Fecker func
ImgPack Utils::CalDF(Mat& src, int channel, Mat& overlap)
{
    ImgPack src_DF;
    src_DF.PDF.assign(256, 0);
    src_DF.CDF.assign(256, 0);

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            if (overlap.at<bool>(i, j))
                src_DF.PDF[(int)src.at<Vec3b>(i, j)[channel]]++;

    src_DF.CDF[0] = src_DF.PDF[0];

    for (int i = 1; i < src_DF.PDF.size(); i++)
        src_DF.CDF[i] = src_DF.PDF[i] + src_DF.CDF[i - 1];

    return src_DF;
}


Mat Utils::HM_Fecker(Mat& ref, Mat& tar, Mat& overlap)
{
    vector<vector<int>> mapping_Func(3, vector<int>(256, 0));
    for (int channel = 0; channel < 3; channel++) {
        ImgPack ref_DF, tar_DF;
        ref_DF = CalDF(ref, channel, overlap);
        tar_DF = CalDF(tar, channel, overlap);

        bool flag = false;
        int temp_x = -100, temp_y = -100;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                if (ref_DF.CDF[j] > tar_DF.CDF[i]) {
                    if (!flag && (j - 1) >= 0)
                    {
                        flag = true;
                        temp_x = i;
                        temp_y = j;
                    }

                    mapping_Func[channel][i] = (int)saturate_cast<uchar>(j);
                    break;
                }

                if (i) mapping_Func[channel][i] = mapping_Func[channel][i - 1];
            }
        }

        for (int i = temp_x; i >= 0; i--)
            mapping_Func[channel][i] = temp_y;

        float sum1 = 0, sum2 = 0;
        for (int i = 0; i <= mapping_Func[channel][0]; i++)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * (float)i;
        }
        mapping_Func[channel][0] = (int)(sum2 / sum1);

        sum1 = 0, sum2 = 0;
        for (int i = 255; i >= mapping_Func[channel][255]; i--)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * (float)i;
        }
        mapping_Func[channel][255] = (int)(sum2 / sum1);
    }

    Mat output_comp = Mat(ref.size(), ref.type());
    for (int channel = 0; channel < 3; channel++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                bool sw = tar.at<Vec3b>(i, j)[0] != 0 || tar.at<Vec3b>(i, j)[1] != 0 || tar.at<Vec3b>(i, j)[2] != 0;
                if (sw)
                    output_comp.at<Vec3b>(i, j)[channel] = mapping_Func[channel][tar.at<Vec3b>(i, j)[channel]];
                else
                    output_comp.at<Vec3b>(i, j)[channel] = tar.at<Vec3b>(i, j)[channel];
            }
        }
    }

    tar.convertTo(tar, CV_64FC3);
    output_comp.convertTo(output_comp, CV_64FC3);
    output_comp -= tar;
    tar.convertTo(tar, CV_8UC3);

    return output_comp;
}

double normalized_error(double& error) {
    double exp_value = exp(-abs(error) / 255);
    double min_value = exp(-255 / 255);
    double max_value = exp(-0 / 255);

    return (exp_value - min_value) / (max_value - min_value);
}

double normalized_wave(int& wave, double& maxexpWave, double& minexpWave) {
    double dwave = static_cast<double>(wave);
    return (exp(1 / dwave) - minexpWave) / (maxexpWave - minexpWave);
}

int Utils::findFirstImg() {
    int n = warped_imgs.size();
    vector<double> edges(n - 1);

    for (const auto& overlap : overlaps) {
        int img1_id = overlap.first.first;
        int img2_id = overlap.first.second;

        if (img1_id < img2_id) {
            continue;
        }

        const Mat& overlap_mask = overlap.second;
        vector<Point> overlap_indices;
        findNonZero(overlap_mask, overlap_indices);

        double cd = 0.0;
        vector<vector<int>> img1_hist(3, vector<int>(256, 0));
        vector<vector<int>> img2_hist(3, vector<int>(256, 0));

        for (const Point& p : overlap_indices) {
            for (int ch = 0; ch < 3; ch++) {
                img1_hist[ch][warped_imgs[img1_id].at<Vec3b>(p)[ch]]++;
                img2_hist[ch][warped_imgs[img2_id].at<Vec3b>(p)[ch]]++;
            }
        }

        for (int ch = 0; ch < 3; ch++) {
            for (int val = 0; val < 256; val++) {
                cd += abs(img1_hist[ch][val] - img2_hist[ch][val]);
            }
        }
        cd /= (double)overlap_indices.size();

        int edge_idx = min(img1_id, img2_id);
        edges[edge_idx] = cd;
    }

    int min_edge_idx = 0;
    double min_mcd = edges[0];
    for (int i = 1; i < n - 1; i++) {
        if (edges[i] < min_mcd) {
            min_mcd = edges[i];
            min_edge_idx = i;
        }
    }

    int vertex1 = min_edge_idx;
    int vertex2 = min_edge_idx + 1;
    
    int mid = n / 2;
    int dist1 = abs(vertex1 - mid);
    int dist2 = abs(vertex2 - mid);

    return (dist1 <= dist2) ? vertex1 : vertex2;
}

vector<pair<int, int>> Utils::getCorrectImgPair() {
    vector<pair<int, int>> seq;
    int center = Utils::findFirstImg();
    for (int i = center; i - 1 >= 0; i--) {
        seq.push_back(make_pair(i - 1, i));
    }
    for (int i = center; i + 1 < warped_imgs.size(); i++) {
        seq.push_back(make_pair(i + 1, i));
    }

    return seq;
}

vector<int> Utils::EC()
{
    vector<bool> is_corrected(warped_imgs.size(), false);
    vector<pair<int, int>> correct_seq = Utils::getCorrectImgPair();

    vector<int> final_sequence;
    cout << "starting correct." << endl;
    for (int i = 0; i < correct_seq.size(); i++) {
        cout << "pair(" << i + 1 << "/" << correct_seq.size() << ")  ";
        pair<int, int> tar_ref = correct_seq[i];
        final_sequence.push_back(tar_ref.first);

        Utils::single_img_correction(tar_ref);
        is_corrected[tar_ref.first] = true;
        
    }
    
    return final_sequence;
}

Vec3f getInterpolatedColor(const Mat& image, const Point2f& pt) {
    int x = floor(pt.x);
    int y = floor(pt.y);

    float alpha = pt.x - x;
    float beta = pt.y - y;

    x = std::max(0, std::min(x, image.cols - 2));
    y = std::max(0, std::min(y, image.rows - 2));

    Vec3f color = (1 - alpha) * (1 - beta) * image.at<Vec3b>(y, x) +
        alpha * (1 - beta) * image.at<Vec3b>(y, x + 1) +
        (1 - alpha) * beta * image.at<Vec3b>(y + 1, x) +
        alpha * beta * image.at<Vec3b>(y + 1, x + 1);

    return color;
}

double calculateJBIWeight(const cv::Vec3f& color1, const cv::Vec3f& color2, double& spatialDistance, double sigma1, double sigma2, int M) {
    double colorDistance = cv::norm(color1, color2);

    double weight = std::exp(-(colorDistance * colorDistance) / (sigma1 * sigma1)) * std::exp(-(spatialDistance * spatialDistance) / (M * sigma2 * sigma2));
    if (!std::isfinite(weight) || weight < std::numeric_limits<double>::min()) {
        weight = std::numeric_limits<double>::min();
    }
    return weight;
}

Mat Utils::JBI(Mat& ref, Mat& tar, Mat& overlap, vector<pair<Point2f, Point2f>>& correspondences) {
    // input
    vector<Vec3d> color_diff(correspondences.size());

    // output
    Mat result = Mat::zeros(tar.size(), CV_64FC3);

    Mat ref_64FC3, tar_64FC3;
    ref.convertTo(ref_64FC3, CV_64FC3);
    tar.convertTo(tar_64FC3, CV_64FC3);

    for (size_t i = 0; i < correspondences.size(); ++i) {
        Point2f pt1 = correspondences[i].first;
        Point2f pt2 = correspondences[i].second;

        if (pt1.x >= 0 && pt1.y >= 0 && pt1.x < ref.cols && pt1.y < ref.rows &&
            pt2.x >= 0 && pt2.y >= 0 && pt2.x < tar.cols && pt2.y < tar.rows) {
            Vec3d color1 = ref_64FC3.at<Vec3d>(pt1);
            Vec3d color2 = tar_64FC3.at<Vec3d>(pt2);
            //ref RGB - tar RGB
            color_diff[i] = color1 - color2;
        }
    }


#pragma omp parallel for collapse(2)
    for (int i = 0; i < tar.rows; ++i) {
        for (int j = 0; j < tar.cols; ++j) {
            if (overlap.at<uchar>(i, j) != 0) {
                Point2f curr_pt(j, i);
                double sumWeights = 0.0;
                Vec3d weightedColorDiff(0, 0, 0);
                vector<double> weightVec(correspondences.size());

                for (int k = 0; k < correspondences.size(); ++k) {
                    Point2f src_pt = correspondences[k].first;
                    if (curr_pt.x >= 0 && curr_pt.y >= 0 && curr_pt.x < tar.cols && curr_pt.y < tar.rows &&
                        src_pt.x >= 0 && src_pt.y >= 0 && src_pt.x < tar.cols && src_pt.y < tar.rows) {
                        Vec3d color_curr = tar_64FC3.at<Vec3d>(curr_pt);
                        Vec3d color_src = ref_64FC3.at<Vec3d>(src_pt);

                        // JBI
                        double spatialDistance = cv::norm(curr_pt - src_pt);
                        double weight = calculateJBIWeight(color_curr, color_src, spatialDistance, 0.3, 0.7, tar.cols);

                        weightVec[k] = weight;

                        sumWeights += weight;
                    }
                }

                if (sumWeights > 0) {
                    double logSumWeight = std::log(sumWeights);
                    for (int idx = 0; idx < correspondences.size(); idx++) {
                        for (int c = 0; c < 3; c++) {
                            double logWeight = std::log(weightVec[idx]);
                            double expWeight = exp(logWeight - logSumWeight);
                            if (color_diff[idx][c] != 0.0) {
                                weightedColorDiff[c] += expWeight * color_diff[idx][c];
                            }
                        }
                    }
                }
                result.at<Vec3d>(i, j) = weightedColorDiff;
            }
        }
    }
    return result;
}


Mat Utils::NonOverlapJBI(Mat& ori_img, Mat& correct_img, vector<Point>& border_pixel, Mat& overlap, Mat& mask) {
    vector<Vec3d> color_diff(border_pixel.size());
    Mat result = Mat::zeros(ori_img.size(), CV_64FC3);
    Mat ref_64FC3, tar_64FC3;
    correct_img.convertTo(ref_64FC3, CV_64FC3);
    ori_img.convertTo(tar_64FC3, CV_64FC3);

#pragma omp parallel for
    for (int i = 0; i < border_pixel.size(); ++i) {
        Point2f pt = border_pixel[i];
        if (pt.x >= 0 && pt.y >= 0 && pt.x < ori_img.cols && pt.y < ori_img.rows) {
            Vec3d color1 = ref_64FC3.at<Vec3d>(pt);
            Vec3d color2 = tar_64FC3.at<Vec3d>(pt);
            color_diff[i] = color1 - color2;
        }
    }

    vector<Point> valid_pixels;

#pragma omp parallel
    {
        vector<Point> local_pixels;

#pragma omp for collapse(2) nowait
        for (int i = 0; i < ori_img.rows; ++i) {
            for (int j = 0; j < ori_img.cols; ++j) {
                if (!overlap.at<uchar>(i, j) && mask.at<uchar>(i, j)) {
                    local_pixels.push_back(Point(j, i));
                }
            }
        }

#pragma omp critical
        {
            valid_pixels.insert(valid_pixels.end(), local_pixels.begin(), local_pixels.end());
        }
    }

#pragma omp parallel for schedule(dynamic, 50)
    for (int idx = 0; idx < valid_pixels.size(); ++idx) {
        Point pixel = valid_pixels[idx];
        int i = pixel.y;
        int j = pixel.x;

        Point2f curr_pt(j, i);
        double sumWeights = 0.0;
        Vec3d weightedColorDiff(0, 0, 0);
        vector<double> weightVec(border_pixel.size());

        for (int k = 0; k < border_pixel.size(); ++k) {
            Point2f src_pt = border_pixel[k];
            if (curr_pt.x >= 0 && curr_pt.y >= 0 && curr_pt.x < ori_img.cols && curr_pt.y < ori_img.rows &&
                src_pt.x >= 0 && src_pt.y >= 0 && src_pt.x < ori_img.cols && src_pt.y < ori_img.rows) {
                Vec3d color_curr = tar_64FC3.at<Vec3d>(curr_pt);
                Vec3d color_src = ref_64FC3.at<Vec3d>(src_pt);
                // JBI
                double spatialDistance = cv::norm(curr_pt - src_pt);
                double weight = calculateJBIWeight(color_curr, color_src, spatialDistance, 0.3, 0.7, ori_img.cols);
                weightVec[k] = weight;
                sumWeights += weight;
            }
        }

        // weight
        if (sumWeights > 0) {
            double logSumWeight = std::log(sumWeights);
            for (int k = 0; k < border_pixel.size(); k++) {
                for (int c = 0; c < 3; c++) {
                    double logWeight = std::log(weightVec[k]);
                    double expWeight = exp(logWeight - logSumWeight);
                    if (color_diff[k][c] != 0.0) {
                        weightedColorDiff[c] += expWeight * color_diff[k][c];
                    }
                }
            }
        }
        result.at<Vec3d>(i, j) = weightedColorDiff;
    }

    return result;
}

void Utils::HE_JBI_fusion(Mat& warped_tar_img, Mat& Fecker_comp, Mat& Color_diff_comp,
    Mat& Fecker_error_map, Mat& Color_diff_error_map, Mat& overlap) {

    warped_tar_img.convertTo(warped_tar_img, CV_64FC3);

    // HE + JBI fusion term
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Point p(x, y);
            if (overlap.at<bool>(p)) {
                for (int c = 0; c < 3; c++) {
                    double total_error = abs(Fecker_error_map.at<Vec3d>(p)[c]) + abs(Color_diff_error_map.at<Vec3d>(p)[c]);
                    double rate = total_error == 0.0 ? 0.0 : abs(Fecker_error_map.at<Vec3d>(p)[c]) / total_error;
                    double single_color_correct = Color_diff_comp.at<Vec3d>(p)[c] * rate + Fecker_comp.at<Vec3d>(p)[c] * (1 - rate);
                    warped_tar_img.at<Vec3d>(p)[c] += single_color_correct;
                    if (warped_tar_img.at<Vec3d>(p)[c] < 0) {
                        warped_tar_img.at<Vec3d>(p)[c] = 0;
                    }
                    else if (warped_tar_img.at<Vec3d>(p)[c] > 255) {
                        warped_tar_img.at<Vec3d>(p)[c] = 255;
                    }
                }
            }
        }
    }

    warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
    return;
}

Mat Utils::errorMap(Mat& ref, Mat& tar, Mat& overlap, Mat& comp) {

    Mat result = Mat::zeros(comp.size(), CV_64FC3);

    Mat ref_64f, tar_64f;
    ref.convertTo(ref_64f, CV_64FC3);
    tar.convertTo(tar_64f, CV_64FC3);

#pragma omp parallel for
    for (int i = 0; i < comp.rows; ++i) {
        for (int j = 0; j < comp.cols; ++j) {
            if (i < tar_64f.rows && j < tar_64f.cols &&
                i < ref_64f.rows && j < ref_64f.cols &&
                overlap.at<uchar>(i, j) != 0) { // 检查 overlap 是否为非零
                Vec3d tar_pixel = tar_64f.at<Vec3d>(i, j);
                Vec3d ref_pixel = ref_64f.at<Vec3d>(i, j);
                Vec3d comp_pixel = comp.at<Vec3d>(i, j);

                Vec3d error_value = (tar_pixel + comp_pixel) - ref_pixel;

                result.at<Vec3d>(i, j) = error_value;
            }
        }
    }

    return result;
}

Mat Utils::computeWavefront(Mat& overlap)
{
    Mat binary;
    threshold(overlap, binary, 128, 255, THRESH_BINARY);
    Mat wavefront_map = Mat::zeros(overlap.size(), CV_32SC1);
    queue<Point> q;

    const int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
    const int dy[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };

    // 初始化：找到所有邊界像素
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (binary.at<uchar>(y, x) == 255) { // 白色像素
                bool isBoundary = false;
                for (int k = 0; k < 8; ++k) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (ny >= 0 && ny < binary.rows && nx >= 0 && nx < binary.cols) {
                        if (binary.at<uchar>(ny, nx) == 0) { // 黑色鄰居
                            isBoundary = true;
                            break;
                        }
                    }
                }
                if (isBoundary) {
                    wavefront_map.at<int>(y, x) = 1;
                    q.push(Point(x, y));
                }
            }
        }
    }

    // BFS 賦值: white pixel
    int current_value = 1;
    while (!q.empty()) {
        int level_size = q.size();
        current_value++;
        for (int i = 0; i < level_size; ++i) {
            Point p = q.front();
            q.pop();
            for (int k = 0; k < 8; ++k) {
                int nx = p.x + dx[k], ny = p.y + dy[k];
                if (ny >= 0 && ny < binary.rows && nx >= 0 && nx < binary.cols) {
                    if (wavefront_map.at<int>(ny, nx) == 0) {  // 確保不是邊界像素
                        wavefront_map.at<int>(ny, nx) = current_value;
                        q.push(Point(nx, ny));
                    }
                }
            }
        }
    }

    return wavefront_map;
}

vector<Point> Utils::getBorderPixel(Mat& wavefrontMap) {
    vector<Point> border_pixel;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Point p(x, y);
            if (wavefrontMap.at<int>(p) == 1) {
                border_pixel.push_back(p);
            }
        }
    }
    return border_pixel;
}

Mat Utils::build_range_map_with_side_addition(Mat& tar_img, vector<Point>& border_pixel, Mat& overlap, Mat& mask)
{
    Mat range_count_map = Mat::zeros(height, width, CV_32FC1);
    Mat discovered_time_stamp_map = Mat::zeros(height, width, CV_16U);
    Mat range_map = Mat::zeros(height, width, CV_16UC2);
    Mat discovered_map = Mat::zeros(height, width, CV_8UC1);

    for (int idx = 0; idx < border_pixel.size(); idx++)
    {
        range_map.at<Vec2w>(border_pixel[idx])[0] = idx;
        range_map.at<Vec2w>(border_pixel[idx])[1] = idx;
    }

    int max_cite_range = (int)border_pixel.size() - 1;
    int time_stamp = 1;
    static uint8_t gray = 255;
    static int x_offset[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    static int y_offset[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    bool flag_early_termination = false;
    vector<Point> next_wavefront;
    next_wavefront.reserve(10000);
    vector<Point> current_wavefront = border_pixel;

    //Mat discovered_time_stamp_map(height, width, CV_16U);
    for (Point point : border_pixel)
        discovered_map.at<uchar>(point) = 255;
    for (Point point : border_pixel)
        discovered_time_stamp_map.at<ushort>(point) = time_stamp;

    // march through whole target image.
    while (true)
    {
        // find next wavefront
        time_stamp++;
        next_wavefront.clear();
        for (Point point : current_wavefront)
        {
            for (int i = 0; i < 8; i++)
            {
                int y_n = point.y + y_offset[i];
                int x_n = point.x + x_offset[i];
                if (x_n == -1 || y_n == -1 || x_n == width || y_n == height)
                    continue;
                if (!overlap.at<uchar>(y_n, x_n) && mask.at<uchar>(y_n, x_n)) {
                    if (discovered_map.at<uchar>(y_n, x_n) == 0) {
                        next_wavefront.emplace_back(x_n, y_n);
                        discovered_map.at<uchar>(y_n, x_n) = 255;
                        discovered_time_stamp_map.at<ushort>(y_n, x_n) = time_stamp;
                    }
                }

            }
        }

        // break from the while loop if there is no next wavefront.
        if (next_wavefront.empty())
            break;
        int reference_all_range_cnt = 0;
        // propagate the citation range from previous wavefront to current wavefront.
        for (Point point : next_wavefront)
        {
            int min_range = std::numeric_limits<int>::max();
            int max_range = std::numeric_limits<int>::min();

            if (!flag_early_termination)
            {
                for (int i = 0; i < 8; i++)
                {
                    int y_n = point.y + y_offset[i];
                    int x_n = point.x + x_offset[i];
                    if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height)
                        continue;
                    if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp - 1)
                    {
                        if (min_range > range_map.at<Vec2w>(y_n, x_n)[0])
                            min_range = range_map.at<Vec2w>(y_n, x_n)[0];
                        if (max_range < range_map.at<Vec2w>(y_n, x_n)[1])
                            max_range = range_map.at<Vec2w>(y_n, x_n)[1];
                    }
                }

                min_range = max(min_range - 2, 0);
                max_range = min(max_range + 2, max_cite_range);
                range_map.at<Vec2w>(point)[0] = min_range;
                range_map.at<Vec2w>(point)[1] = max_range;


                if (range_map.at<Vec2w>(point)[0] == 0 && range_map.at<Vec2w>(point)[1] == max_cite_range)
                {
                    // Fecker_mask.at<uchar>(point) = 255;
                    reference_all_range_cnt++;
                }

                range_count_map.at<int>(point) = max_range - min_range + 1;
            }
            else
            {
                range_map.at<Vec2w>(point)[0] = 0;
                range_map.at<Vec2w>(point)[1] = max_cite_range;
                range_count_map.at<int>(point) = max_range - min_range + 1;
            }
        }
        if (next_wavefront.size() == reference_all_range_cnt)
            flag_early_termination = true;

        vector<int> pending_update_min_ranges, pending_update_max_ranges;

        for (Point point : next_wavefront)
        {
            int min_range = std::numeric_limits<int>::max();
            int max_range = std::numeric_limits<int>::min();

            for (int i = 0; i < 8; i++)
            {
                int y_n = point.y + y_offset[i];
                int x_n = point.x + x_offset[i];
                if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height)
                    continue;
                if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp)
                {
                    if (min_range > range_map.at<Vec2w>(y_n, x_n)[0])
                        min_range = range_map.at<Vec2w>(y_n, x_n)[0];
                    if (max_range < range_map.at<Vec2w>(y_n, x_n)[1])
                        max_range = range_map.at<Vec2w>(y_n, x_n)[1];
                }
            }
            pending_update_min_ranges.push_back(min_range);
            pending_update_max_ranges.push_back(max_range);
        }

        for (int idx = 0; idx < next_wavefront.size(); idx++)
        {
            range_map.at<Vec2w>(next_wavefront[idx])[0] = pending_update_min_ranges[idx];
            range_map.at<Vec2w>(next_wavefront[idx])[1] = pending_update_max_ranges[idx];
        }

        gray -= 10;

        current_wavefront = next_wavefront;
    }

    //discovered_time_stamp_map.copyTo(dts_map);
    return range_count_map;
}

void Utils::nonOverlapColorCorrect(Mat& warped_tar_img, Mat& cite_range, Mat& Fecker_comp, Mat& JBI_comp, Mat& overlap, Mat& mask, vector<Point>& border_pixel)
{
    if (cite_range.empty()) return;

    warped_tar_img.convertTo(warped_tar_img, CV_64FC3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Point p(x, y);
            if (!overlap.at<uchar>(p) && mask.at<uchar>(p)) {
                for (int c = 0; c < 3; c++) {
                    double rate = 1.0 * cite_range.at<int>(p) / border_pixel.size();
                    rate = rate > 1.0 ? 1.0 : rate < 0.0 ? 0.0 : rate;
                    warped_tar_img.at<Vec3d>(p)[c] += Fecker_comp.at<Vec3d>(p)[c] * rate + JBI_comp.at<Vec3d>(p)[c] * (1.0 - rate);
                }
            }
        }
    }
    warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
}

Mat Utils::getFusionNonOverlap(Mat& warped_tar_img, Mat& cite_range, Mat& Fecker_comp, Mat& JBI_comp, Mat& overlap, Mat& mask, vector<Point>& border_pixel) {

    Mat result = Mat::zeros(warped_tar_img.size(), CV_64FC3);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Point p(x, y);
            if (!overlap.at<uchar>(p) && mask.at<uchar>(p)) {
                for (int c = 0; c < 3; c++) {
                    double rate = 1.0 * cite_range.at<int>(p) / border_pixel.size();
                    double weighted_rate = rate * 0.2;
                    weighted_rate = weighted_rate > 1.0 ? 1.0 : weighted_rate < 0.0 ? 0.0 : weighted_rate;
                    result.at<Vec3d>(p)[c] = Fecker_comp.at<Vec3d>(p)[c] * (1.0 - weighted_rate) + JBI_comp.at<Vec3d>(p)[c] * weighted_rate;
                }
            }
        }
    }
    return result;
}

void Utils::getNonOverlapColorCorrectImg(Mat& warped_tar_img, Mat& fusion_comp, Mat& overlap, Mat& mask)
{
    warped_tar_img.convertTo(warped_tar_img, CV_64FC3);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Point p(x, y);
            if (!overlap.at<bool>(p) && mask.at<bool>(p)) {
                for (int c = 0; c < 3; c++) {
                    warped_tar_img.at<Vec3d>(p)[c] += fusion_comp.at<Vec3d>(p)[c];
                    if (warped_tar_img.at<Vec3d>(p)[c] < 0) {
                        warped_tar_img.at<Vec3d>(p)[c] = 0;
                    }
                    else if (warped_tar_img.at<Vec3d>(p)[c] > 255) {
                        warped_tar_img.at<Vec3d>(p)[c] = 255;
                    }
                }
            }
        }
    }
    warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
}

void Utils::single_img_correction(pair<int, int> tar_ref)
{
    int tar = tar_ref.first;
    int ref = tar_ref.second;

    Mat NonCorrectTarImg = warped_imgs[tar].clone();

    Mat overlap = overlaps[make_pair(tar, ref)];
    vector<pair<Point2f, Point2f>> corr = Corr_vec[make_pair(tar, ref)];

    Mat wavefront_map = computeWavefront(overlap);

    cout << "[overlap]-->";
    //HM-based
    Mat Fecker_comp = Utils::HM_Fecker(warped_imgs[ref], warped_imgs[tar], overlap);
    Mat Fecker_map = Utils::errorMap(warped_imgs[ref], warped_imgs[tar], overlap, Fecker_comp);

    //correspondence-based
    Mat JBI_comp = Utils::JBI(warped_imgs[ref], warped_imgs[tar], overlap, corr);
    Mat JBI_map = Utils::errorMap(warped_imgs[ref], warped_imgs[tar], overlap, JBI_comp);

    cout << "[fusion]-->";
    Utils::HE_JBI_fusion(warped_imgs[tar], Fecker_comp, JBI_comp, Fecker_map, JBI_map, overlap);

    vector<Point> border_pixel = Utils::getBorderPixel(wavefront_map);
    Mat cite_range = Utils::build_range_map_with_side_addition(warped_imgs[tar], border_pixel, overlap, masks[tar]);

    cout << "[non-overlap]-->";
    Mat non_Fecker_comp = Utils::HM_Fecker(warped_imgs[tar], NonCorrectTarImg, overlap);
    
    Mat non_JBI_comp = Utils::NonOverlapJBI(NonCorrectTarImg, warped_imgs[tar], border_pixel, overlap, masks[tar]);
    
    cout << "[fusion]" << endl;
    Mat nonOverlap_comp = Utils::getFusionNonOverlap(warped_imgs[tar], cite_range, non_Fecker_comp, non_JBI_comp, overlap, masks[tar], border_pixel);

    Utils::getNonOverlapColorCorrectImg(warped_imgs[tar], nonOverlap_comp, overlap, masks[tar]);
}
