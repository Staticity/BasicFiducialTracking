#include <opencv.hpp>

#include "Util.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "MultiView.hpp"

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc != 3 - 1)
    {
        cout << " <calibration_filepath>";
        cout << " <image_1_filepath>";
        cout << " <image_2_filepath>";
        cout << endl;
    }

    string calibration_filepath = argv[1];
    string image_1_filepath     = argv[2];
    string image_2_filepath     = argv[3];

    // Find calibration file and load images
    Camera camera(calibration_filepath);
    Mat im1 = imread(image_1_filepath);
    Mat im2 = imread(image_2_filepath);

    // Images must be of the same size for correspondences
    assert(im1.size() == im2.size());

    // Resize the camera matrix to the image sizes taken
    camera.resize(im1.size());   
    
    // Find the point matches between the two frames
    vector<KeyPoint> feat1, feat2;
    Mat desc1, desc2;
    vector<DMatch> matches;
    Features::findMatches(im1, im2, feat1, feat2, desc1, desc2, matches);


}
