#include "utili.h"
#include<filesystem>

namespace fs = std::filesystem;

vector<Mat> warped_imgs, masks;
map<pair<int, int>, Mat> overlaps;
vector<vector<int>> ref_seq;
map<pair<int, int>, vector<pair<Point2f, Point2f>>> Corr_vec;


// parameters
int height, width;
bool ignore_non_corrected;


int main()
{
    //cout << "Current working directory: " << fs::current_path() << endl;
    // parameters setting
    ignore_non_corrected = false;

    string method = "Ours";

    string data_num = "../dataset/01/";
    cout << data_num << endl;

    string warpDir = data_num + "aligned_result/";
    string overlapDir = data_num + "overlap/";
    string corrDir = data_num + "correspondence/";
    string maskDir = data_num + "img_masks/";

    Inits::loadAll(warpDir, overlapDir, corrDir, maskDir);
    height = warped_imgs[0].rows;
    width = warped_imgs[0].cols;

    vector<int> exe_seq;
    if (method == "Ours") {
        exe_seq = Utils::EC();
    }
    else {
        method = "original";
        cout << "Non-existing method, output original warped result." << endl;
    }

    //save result
    for (int i = 0; i < warped_imgs.size(); i++)
        imwrite(data_num + method + "/" + to_string(i) + "__warp.png", warped_imgs[i]);

    Mat result = Tools::createResult(warped_imgs, masks);
    imwrite(data_num + "/result/" + method + ".png", result);
    
    cout << data_num << " --> Color correction finished." << endl;

    return 0;
}