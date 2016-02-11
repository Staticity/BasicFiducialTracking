#include <opencv.hpp>

#include <cstdio>
#include <string>

using namespace cv;

double find_correspondences(const Mat& im1, const Mat& im2,
                            std::vector<Point>& pts1, std::vector<Point>& pts2)
{
    // printf("=== [find_correspondences] ===\n");
    double t = getTickCount();
    // std::string feature_type  = "SIFT";
    // std::string detector_type = "SIFT";
    std::string matcher_type  = "BruteForce";

    Ptr<FeatureDetector> detector = ORB::create();// FeatureDetector::create(feature_type);
    std::vector<KeyPoint> feat1, feat2;
    detector->detect(im1, feat1);
    detector->detect(im2, feat2);

    // printf("Num features: %lu, %lu\n", feat1.size(), feat2.size());

    Ptr<DescriptorExtractor> extractor = ORB::create();// DescriptorExtractor::create(detector_type);
    Mat descriptor1, descriptor2;
    extractor->compute(im1, feat1, descriptor1);
    extractor->compute(im2, feat2, descriptor2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_type);
    std::vector<std::vector<DMatch> > matches;
    matcher->knnMatch(descriptor1, descriptor2, matches, 2);

    // printf("Num matches: %lu\n", matches.size() * 2);

    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches[i].size() >= 2)
        {
            const float ratio = 0.8;
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                pts1.push_back(feat1[matches[i][0].queryIdx].pt);
                pts2.push_back(feat2[matches[i][0].trainIdx].pt);
            }
        }
    }

    double time_spent = (((double)getTickCount()) - t) / getTickFrequency();
    // printf("[find_correspondences]: %0.4f seconds\n", time_spent);

    return time_spent;
}

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    int width = 1280;
    int height = 720;
    VideoCapture vc(index);

    if (!vc.isOpened()) return 0;

    vc.set(CV_CAP_PROP_FRAME_WIDTH, width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    Mat im1, im2;

    namedWindow("im1");
    namedWindow("im2");

    vc >> im1;
    waitKey();
    vc >> im2;
    waitKey();

    std::vector<Point> pts1, pts2;

    int its = 10;
    assert(its > 0);
    double total_time = 0.0;
    for (int i = 0; i < its; ++i)
        total_time += find_correspondences(im1, im2, pts1, pts2);

    printf("[avg_find_correspondences]: %0.4f seconds\n", total_time / its);

    assert(pts1.size() == pts2.size());

    {
        RNG rng(0);
        int n = pts1.size();
        for (int i = 0; i < n; ++i)
        {
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            circle(im1, pts1[i], 10, color, 3);
            circle(im2, pts2[i], 10, color, 3);
        }

        Mat joined_image  = Mat::zeros(max(im1.rows, im2.rows), im1.cols + im2.cols, im1.type());
        Rect left_roi  = Rect(0, 0, im1.cols, im1.rows);
        Rect right_roi = Rect(im1.cols, 0, im2.cols, im2.rows);

        Mat left_portion  = Mat(joined_image, left_roi);
        Mat right_portion = Mat(joined_image, right_roi);

        im1.copyTo(left_portion);
        im2.copyTo(right_portion);

        imshow("im", joined_image);
    }

    waitKey();

}
