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
    if (argc != 5)
    {
        cout << "<video_camera_index>";
        cout << " <calibration_filepath>";
        cout << " <video_width>";
        cout << " <video_height>";
        cout << endl;
    }

    int video_camera_index      = atoi(argv[1]);
    string calibration_filepath = argv[2];
    int video_width             = atoi(argv[3]);
    int video_height            = atoi(argv[4]);

    Camera camera(calibration_filepath);
    camera.resize(Size(video_width, video_height));

    VideoCapture vc(video_camera_index);
    vc.set(CV_CAP_PROP_FRAME_WIDTH, video_width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, video_height);

    if (!vc.isOpened()) return 0;

}
