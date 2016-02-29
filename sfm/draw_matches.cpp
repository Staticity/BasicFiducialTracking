#include <opencv.hpp>

#include "Util.hpp"
#include "Features.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    VideoCapture vc(0);

    int count = 0;
    Mat image;
    Mat im[2];
    namedWindow("source");
    while (vc.isOpened() && count < 2)
    {
        vc >> image;
        imshow("source", image);

        char c = waitKey(20);
        if (c == ' ')
        {
            im[count] = image.clone();
            ++count;
        }
        else if (c == 27)
        {
            return 0;
        }
    }
    destroyWindow("source");

    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    vector<DMatch> matches;
    Features::findMatches(im[0], im[1], kp1, kp2, desc1, desc2, matches);

    cout << kp1.size() << " " << kp2.size() << endl;
    Mat drawing;
    drawMatches(im[0], kp1, im[1], kp2, matches, drawing);

    imshow("drawing", drawing);
    waitKey();
}
