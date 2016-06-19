#include <iostream>
#include <opencv.hpp>
#include <imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

void harrisCornerScore(
    const Mat& image,
    Mat& dest,
    int block_size,
    int aperture_size,
    float k_scale)
{
    assert(image.type() == CV_8UC1);
    assert(aperture_size > 0);

    Mat cov;
    cov = Mat::zeros(image.size(), CV_32FC3);
    dest = Mat::zeros(image.size(), CV_32FC1);

    float scale = 1.0 / (255.0 * (1 << (aperture_size - 1)) * block_size);

    Mat Dx, Dy;
    Sobel(image, Dx, CV_32FC1, 1, 0, aperture_size, scale);
    Sobel(image, Dy, CV_32FC1, 0, 1, aperture_size, scale);

    // imshow("dx", Dx);
    // imshow("dy", Dy);
    // waitKey();
    // destroyWindow("dx");
    // destroyWindow("dy");

    for (int i = 0; i < image.rows; ++i)
    {
        float* cov_data = cov.ptr<float>(i);
        const float* dx_data = Dx.ptr<float>(i);
        const float* dy_data = Dy.ptr<float>(i);

        for (int j = 0; j < image.cols; ++j)
        {
            float dx = dx_data[j];
            float dy = dy_data[j];

            int k = j * 3;
            cov_data[k + 0] = dx * dx;
            cov_data[k + 1] = dx * dy;
            cov_data[k + 2] = dy * dy;
        }
    }

    // imshow("covariance", cov);

    boxFilter(
        cov,
        cov,
        cov.depth(),
        Size(block_size, block_size),
        Point(-1, -1),
        false); // Don't normalize

    // imshow("covariance blur", cov);
    // waitKey();
    // destroyWindow("covariance");
    // destroyWindow("covariance blur");

    for (int i = 0; i < image.rows; ++i)
    {
        const float* cov_data = cov.ptr<float>(i);
        float* corner_score = dest.ptr<float>(i);

        for (int j = 0; j < image.cols; ++j)
        {
            int k = j * 3;
            float a = cov_data[k + 0];
            float b = cov_data[k + 1];
            float c = cov_data[k + 2];

            corner_score[j] = (a * c) - (b * b) - k_scale * (a + c) * (a + c);
        }
    }
}

int main(int argc, char** argv)
{
    Mat image = imread(argv[1]);
    Mat dest, dest_cv;

    cvtColor(image, image, CV_BGR2GRAY);

    harrisCornerScore(image, dest, 2, 5, 0.04);
    // cornerHarris(image, dest_cv, 2, 3, 0.04, BORDER_DEFAULT);

    normalize(dest, dest, 0, 1.0, NORM_MINMAX, CV_32FC1, Mat());
    // normalize(dest_cv, dest_cv, 0, 1.0, NORM_MINMAX, CV_32FC1, Mat());

    imshow("Original", image);
    // imshow("Harris of OpenCV", dest_cv);
    waitKey();
}
