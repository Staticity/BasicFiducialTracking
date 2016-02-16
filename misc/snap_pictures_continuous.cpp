#include <opencv.hpp>

#include <string>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 4)
        return 0;

    string directory = argv[1];
    int count        = atoi(argv[2]);
    int delay        = atoi(argv[3]);

    if (directory[directory.size() - 1] != '/')
        directory += '/';

    Mat image;
    VideoCapture vc(0);

    if (!vc.isOpened()) return 0;

    for (int i = 0; i < count && vc.isOpened(); ++i)
    {
        vc >> image;
        imshow("image", image);
        imwrite(directory + to_string(i) + ".jpg", image);
        waitKey(delay);
    }
}
