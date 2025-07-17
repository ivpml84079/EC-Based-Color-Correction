#include "utili.h"
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>

namespace fs = std::filesystem;

vector<Mat> warped_imgs, masks;
map<pair<int, int>, Mat> overlaps;
vector<vector<int>> ref_seq;
map<pair<int, int>, vector<pair<Point2f, Point2f>>> Corr_vec;

// parameters
int height, width;
bool ignore_non_corrected;
float sigma1 = 0.3, sigma2 = 0.7, alpha = 0.2;

// 時間記錄結構體
struct TimingResult {
    std::string dataset_name;
    double loading_time;     // 資料載入時間 (秒)
    double correction_time;  // 色彩校正時間 (秒)
    double total_time;       // 總時間 (秒)
    int num_images;          // 圖片數量
};

// 提取資料集名稱的輔助函數
std::string extractDatasetName(const std::string& path) {
    fs::path p(path);
    std::string name = p.filename().string();
    // 移除尾部的斜線
    if (name.empty() && p.has_parent_path()) {
        name = p.parent_path().filename().string();
    }
    return name;
}

// 輸出CSV結果的函數
void exportTimingResults(const std::vector<TimingResult>& results,
    const std::string& method,
    float sigma1, float sigma2, float alpha) {

    // 建立檔案名稱
    std::ostringstream filename;
    filename << "timing_results_" << method << ".csv";

    std::ofstream file(filename.str());

    if (!file.is_open()) {
        std::cerr << "Error: Cannot create output file " << filename.str() << std::endl;
        return;
    }

    // 寫入CSV標頭
    file << "Dataset Name,Number of Images,Loading Time (s),Color Correction Time (s),Total Time (s)\n";

    // 統計變數
    double total_loading_time = 0.0;
    double total_correction_time = 0.0;
    double total_overall_time = 0.0;
    int total_images = 0;

    // 寫入每個資料集的結果
    for (const auto& result : results) {
        file << result.dataset_name << ","
            << result.num_images << ","
            << std::fixed << std::setprecision(3) << result.loading_time << ","
            << std::fixed << std::setprecision(3) << result.correction_time << ","
            << std::fixed << std::setprecision(3) << result.total_time << "\n";

        total_loading_time += result.loading_time;
        total_correction_time += result.correction_time;
        total_overall_time += result.total_time;
        total_images += result.num_images;
    }

    // 計算平均值
    int num_datasets = results.size();
    double avg_loading_time = total_loading_time / num_datasets;
    double avg_correction_time = total_correction_time / num_datasets;
    double avg_total_time = total_overall_time / num_datasets;
    double avg_images = (double)total_images / num_datasets;

    // 寫入分隔線和統計資訊
    file << "\n";
    file << "STATISTICS\n";
    file << "Total Datasets," << num_datasets << "\n";
    file << "Total Images," << total_images << "\n";
    file << "Average Images per Dataset," << std::fixed << std::setprecision(1) << avg_images << "\n";
    file << "\n";
    file << "TIMING SUMMARY\n";
    file << "Total Loading Time (s)," << std::fixed << std::setprecision(3) << total_loading_time << "\n";
    file << "Total Correction Time (s)," << std::fixed << std::setprecision(3) << total_correction_time << "\n";
    file << "Total Overall Time (s)," << std::fixed << std::setprecision(3) << total_overall_time << "\n";
    file << "\n";
    file << "AVERAGE TIMES\n";
    file << "Average Loading Time (s)," << std::fixed << std::setprecision(3) << avg_loading_time << "\n";
    file << "Average Correction Time (s)," << std::fixed << std::setprecision(3) << avg_correction_time << "\n";
    file << "Average Total Time (s)," << std::fixed << std::setprecision(3) << avg_total_time << "\n";
    file << "\n";
    file << "PARAMETERS\n";
    file << "sigma1," << sigma1 << "\n";
    file << "sigma2," << sigma2 << "\n";
    file << "alpha," << alpha << "\n";

    file.close();

    std::cout << "\n=== TIMING RESULTS EXPORTED TO: " << filename.str() << " ===\n";
    std::cout << "Total datasets processed: " << num_datasets << std::endl;
    std::cout << "Average correction time: " << std::fixed << std::setprecision(3)
        << avg_correction_time << " seconds" << std::endl;
    std::cout << "Total processing time: " << std::fixed << std::setprecision(3)
        << total_overall_time << " seconds" << std::endl;
}

int main(int argc, char* argv[])
{
    // 解析命令列參數
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--sigma1" && i + 1 < argc) {
            sigma1 = std::stof(argv[i + 1]);
            i++;
        }
        else if (arg == "--sigma2" && i + 1 < argc) {
            sigma2 = std::stof(argv[i + 1]);
            i++;
        }
        else if (arg == "--alpha" && i + 1 < argc) {
            alpha = std::stof(argv[i + 1]);
            i++;
        }
    }

    // print parameters
    std::cout << "Parameters:" << std::endl;
    std::cout << "sigma1: " << sigma1 << std::endl;
    std::cout << "sigma2: " << sigma2 << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "ignore_non_corrected: " << (ignore_non_corrected ? "true" : "false") << std::endl;

    // parameters setting
    ignore_non_corrected = false;

    std::ostringstream oss;
    oss << "ErrorMap_" << std::fixed << std::setprecision(1) << sigma1 << "_" << sigma2 << "_" << std::setprecision(1) << alpha << "_delaunay_5pixel";

    std::string method = oss.str();
    std::cout << "Method: " << method << std::endl;

    vector<string> data_num_set = {
         "D:/panorama_dataset_DW/agrowing_1_DW/",
        "D:/panorama_dataset_DW/agrowing_3_DW/",
        "D:/panorama_dataset_DW/agrowing_4_DW/",
        "D:/panorama_dataset_DW/agrowing_5_DW/",
        "D:/panorama_dataset_DW/Belair_M1_2_DW/",
        "D:/panorama_dataset_DW/Belair_M1_4_DW/",
        "D:/panorama_dataset_DW/Belair_M1_5_DW/",
        "D:/panorama_dataset_DW/Belair_M1_8_DW/",
        "D:/panorama_dataset_DW/Belair_M2_1_DW/",
        "D:/panorama_dataset_DW/Belair_M2_2_DW/",
        "D:/panorama_dataset_DW/drone_beach_1_DW/",
        "D:/panorama_dataset_DW/drone_beach_2_DW/",
        "D:/panorama_dataset_DW/phalaborwa_lg_ol_1/",
        "D:/panorama_dataset_DW/phalaborwa_lg_ol_2/",
        "D:/panorama_dataset_DW/phalaborwa_lg_ol_3/",
        "D:/panorama_dataset_DW/phalaborwa_lg_ol_5/",
        "D:/panorama_dataset_DW/phalaborwa_lg_ol_7/",
        "D:/panorama_dataset_DW/phalaborwa_sm_ol_1/",
        "D:/panorama_dataset_DW/phalaborwa_sm_ol_2/",
        "D:/panorama_dataset_DW/phalaborwa_sm_ol_4/",
        "D:/panorama_dataset_DW/sheffield_cross_1/",
        "D:/panorama_dataset_DW/sheffield_cross_3/",
        "D:/panorama_dataset_DW/sheffield_cross_5/",
        "D:/panorama_dataset_DW/sheffield_cross_6/",
        "D:/panorama_dataset_DW/sheffield_cross_7/",
        "D:/panorama_dataset_DW/sheffield_cross_8/",
        "D:/panorama_dataset_DW/sheffield_cross_9/",
        "D:/panorama_dataset_DW/sheffield_cross_10/",
        "D:/panorama_dataset_DW/sheffield_cross_11/",
        "D:/panorama_dataset_DW/sheffield_cross_12/"

    };

    // 儲存時間結果的向量
    std::vector<TimingResult> timing_results;

    // 整體開始時間
    auto overall_start = std::chrono::high_resolution_clock::now();

    for (const string& data_num : data_num_set) {
        TimingResult current_result;
        current_result.dataset_name = extractDatasetName(data_num);

        cout << "\n=== Processing: " << current_result.dataset_name << " ===" << endl;

        // 建立輸出資料夾
        fs::path new_folder = fs::path(data_num) / method;
        try {
            if (!fs::exists(new_folder)) {
                fs::create_directory(new_folder);
                std::cout << "Created folder: " << new_folder << std::endl;
            }
        }
        catch (fs::filesystem_error& e) {
            std::cerr << "Error creating folder " << new_folder << ": " << e.what() << std::endl;
        }

        string warpDir = data_num + "multiview_warp_result/";
        string overlapDir = data_num + "overlap/";
        string corrDir = data_num + "global/";
        string maskDir = data_num + "img_masks/";

        // 測量資料載入時間
        auto loading_start = std::chrono::high_resolution_clock::now();

        Inits::loadAll(warpDir, overlapDir, corrDir, maskDir);
        height = warped_imgs[0].rows;
        width = warped_imgs[0].cols;
        current_result.num_images = warped_imgs.size();

        auto loading_end = std::chrono::high_resolution_clock::now();
        current_result.loading_time = std::chrono::duration<double>(loading_end - loading_start).count();

        cout << "Loaded " << current_result.num_images << " images in "
            << std::fixed << std::setprecision(3) << current_result.loading_time << " seconds" << endl;
        cout << "sigma1: " << sigma1 << ", sigma2: " << sigma2 << endl;

        // 測量色彩校正時間
        auto correction_start = std::chrono::high_resolution_clock::now();

        vector<int> exe_seq = Utils::EC();

        auto correction_end = std::chrono::high_resolution_clock::now();
        current_result.correction_time = std::chrono::duration<double>(correction_end - correction_start).count();

        cout << "Color correction completed in "
            << std::fixed << std::setprecision(3) << current_result.correction_time << " seconds" << endl;

        // 計算總時間
        current_result.total_time = current_result.loading_time + current_result.correction_time;

        // 儲存結果
        for (int i = 0; i < warped_imgs.size(); i++)
            imwrite(data_num + method + "/" + to_string(i) + "__warp.png", warped_imgs[i]);

        Mat result = Tools::createResult(warped_imgs, masks);
        imwrite(data_num + "/panorama/" + method + ".png", result);

        // 添加到結果向量
        timing_results.push_back(current_result);

        cout << current_result.dataset_name << " --> Processing finished. Total time: "
            << std::fixed << std::setprecision(3) << current_result.total_time << " seconds" << endl;

        // 清理記憶體
        warped_imgs.clear();
        masks.clear();
        overlaps.clear();
        ref_seq.clear();
        Corr_vec.clear();
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    double total_program_time = std::chrono::duration<double>(overall_end - overall_start).count();

    // 輸出結果到CSV檔案
    exportTimingResults(timing_results, method, sigma1, sigma2, alpha);

    std::cout << "\n=== FINAL SUMMARY ===" << std::endl;
    std::cout << "Total program execution time: " << std::fixed << std::setprecision(3)
        << total_program_time << " seconds (" << total_program_time / 60.0 << " minutes)" << std::endl;
    std::cout << "All " << timing_results.size() << " datasets processed successfully!" << std::endl;

    return 0;
}