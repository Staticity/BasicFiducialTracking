#include <opencv.hpp>
#include <xfeatures2d.hpp>
#include <iostream>
#include <string>
#include <map>

using namespace cv;
using namespace std;

// 1. Find features for each frame
// 2. Find correspondences between frames
// 3. Use RANSAC to find best match between frames
// 4. Merge images

struct ImageMatch
{
    Point2d pt1;
    Point2d pt2;
};

Scalar randomColor(RNG& rng)
{
    int icolor = (unsigned) rng;
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void findPutativeMatches(
    const vector<Mat>& images,
    vector<vector<ImageMatch> >& matches)
{
    assert(images.size() >= 2);

    Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(/*400*/);
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    BFMatcher matcher = BFMatcher::BFMatcher(NORM_L2, true);

    vector<Mat> descriptors(images.size());
    vector<vector<KeyPoint> > features(images.size());
    matches = vector<vector<ImageMatch> >(images.size() - 1);

    for (int i = 0; i < images.size(); ++i)
    {
        assert(!images[i].empty());

        Mat grayImage;
        cvtColor(images[i], grayImage, CV_BGR2GRAY);

        detector->detect(grayImage, features[i]);
        assert(!features[i].empty());

        extractor->compute(grayImage, features[i], descriptors[i]);
        assert(!descriptors[i].empty());

        if (i > 0)
        {
            vector<DMatch> dmatches;
            matcher.match(descriptors[i - 1], descriptors[i], dmatches);

            assert(matches.size() > i - 1);
            matches[i - 1].resize(dmatches.size());

            Rect rect1(Point(), images[i - 1].size());
            Rect rect2(Point(), images[i].size());
            for (int j = 0; j < dmatches.size(); ++j)
            {
                matches[i - 1][j].pt1 = features[i - 1][dmatches[j].queryIdx].pt;
                matches[i - 1][j].pt2 = features[i][dmatches[j].trainIdx].pt;

                assert(rect1.contains(matches[i - 1][j].pt1));
                assert(rect2.contains(matches[i - 1][j].pt2));
            }

            assert(!matches[i - 1].empty());            
        }
    }
}

void visualizeMatches(
    const vector<Mat>& images,
    const vector<vector<ImageMatch> >& matches,
    const vector<vector<uchar> >& masks = vector<vector<uchar> >())
{
    assert(images.size() == matches.size() + 1);
    assert(masks.empty() || masks.size() == matches.size());

    for (int i = 0; i < matches.size(); ++i)
    {
        RNG rng(i);
        const vector<ImageMatch>& image_matches = matches[i];

        int rows1 = images[i].rows;
        int cols1 = images[i].cols;
        int rows2 = images[i + 1].rows;
        int cols2 = images[i + 1].cols;
        Mat drawing = Mat::zeros(rows1, cols1 + cols2, images[i].type());
        images[i].copyTo(drawing(Rect(0, 0, cols1, rows1)));
        images[i + 1].copyTo(drawing(Rect(cols1, 0, cols2, rows2)));

        Point2d offset(cols1, 0);
        for (int j = 0; j < image_matches.size(); ++j)
        {
            if (!masks.empty() && !masks[i][j])
                continue;

            Scalar color = randomColor(rng);
            Point2d start = image_matches[j].pt1;
            Point2d end = image_matches[j].pt2 + offset;
            circle(drawing, start, 5, color, 2);
            circle(drawing, end, 5, color, 2);
            line(drawing, start, end, color);
        }

        string window1 = "matches_" + to_string(i) + ".jpg";
        imshow(window1, drawing);
        imwrite(window1, drawing);
        waitKey();
        destroyWindow(window1);
    }    
}

template <class T>
void randomElements(int k, vector<T>& v)
{
    RNG rng(getTickCount());
    for (int i = 0; i < k; ++i)
    {
        int j = rng.uniform(i, v.size() - 1);
        T temp = v[i];
        v[i] = v[j];
        v[j] = temp;
    }
}

Mat createHomographyMatrix(const vector<ImageMatch>& matches)
{
    Mat A = Mat::zeros(matches.size() * 2, 9, CV_64F);

    for (int i = 0; i < A.rows; i += 2)
    {
        int index = i / 2;
        double x1 = matches[index].pt1.x;
        double y1 = matches[index].pt1.y;
        double x2 = matches[index].pt2.x;
        double y2 = matches[index].pt2.y;

        Mat row1 = (Mat_<double>(1, 9) << 0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2);
        Mat row2 = (Mat_<double>(1, 9) << -x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2);
        row1.copyTo(A(Rect(0,   i  , 9, 1)));
        row2.copyTo(A(Rect(0, i + 1, 9, 1)));
    }

    return A;
}

Mat estimateHomography(
    const vector<ImageMatch>& matches,
    vector<uchar>& inliers)
{
    double inlier_threshold = 3.0; // Need to change
    int iterations = 2000; // Need to change

    Mat best_homography;
    vector<int> best_inliers;

    vector<ImageMatch> matchesCopy(matches);

    int max_inliers = 0;
    while (iterations--)
    {
        randomElements(4, matchesCopy);
        vector<ImageMatch> subsetMatches(matchesCopy.begin(), matchesCopy.begin() + 4);
        assert(subsetMatches.size() == 4);

        Mat A = createHomographyMatrix(subsetMatches);

        Mat u, w, vt;
        SVD::compute(A, u, w, vt);

        Mat_<double> X = vt.row(vt.rows - 1);
        assert(X.rows * X.cols == 9);

        Mat H = (Mat_<double>(3, 3) << X(0), X(1), X(2),
                                       X(3), X(4), X(5),
                                       X(6), X(7), X(8));
 
        vector<int> inlier_indices;
        for (int i = 0; i < matches.size(); ++i)
        {
            double x1 = matches[i].pt1.x;
            double y1 = matches[i].pt1.y;    
            double x2 = matches[i].pt2.x;
            double y2 = matches[i].pt2.y;            
            
            Mat_<double> sourcePt = (Mat_<double>(3, 1) << x1, y1, 1);
            Mat_<double> destPt = H * sourcePt;

            assert(destPt(2) != 0.0);
            double hx2 = destPt(0) / destPt(2);
            double hy2 = destPt(1) / destPt(2);

            double error = sqrt((hx2 - x2) * (hx2 - x2) + (hy2 - y2) * (hy2 - y2));

            if (error <= inlier_threshold)
            {
                inlier_indices.push_back(i);
            }
        }

        // somehow mapping linear least squares didn't work?
        // this is rare... but idfk how it happens
        if (inlier_indices.size() < 4)
        {
            continue;
        }

        if (inlier_indices.size() > max_inliers)
        {
            max_inliers = inlier_indices.size();
            best_homography = H;
            best_inliers = inlier_indices;
        }
    }

    assert(!best_inliers.empty());
    inliers.clear();
    inliers.resize(matches.size());
    for (int i = 0; i < best_inliers.size(); ++i)
    {
        inliers[best_inliers[i]] = 1;
    }

    vector<ImageMatch> inlierMatches(best_inliers.size());
    for (int i = 0; i < best_inliers.size(); ++i)
    {
        inlierMatches[i] = matches[best_inliers[i]];
    }


    Mat A = createHomographyMatrix(inlierMatches);

    Mat u, w, vt;
    SVD::compute(A, u, w, vt);
    Mat_<double> X = vt.row(vt.rows - 1);
    best_homography = (Mat_<double>(3, 3) << X(0), X(1), X(2),
                                             X(3), X(4), X(5),
                                             X(6), X(7), X(8));

    return best_homography;
}

Mat mergeImages(
    const Mat& im1,
    const Mat& im2,
    const Mat& homography,
    Mat& translation)
{
    assert(im1.type() == im2.type());

    double minX, minY, maxX, maxY;

    // shift to 0-index for index warping
    double cols1 = im1.cols - 1.0;
    double rows1 = im1.rows - 1.0;

    Mat corners = (Mat_<double>(4, 3) <<     0,     0, 1,
                                         cols1,     0, 1,
                                             0, rows1, 1,
                                         cols1, rows1, 1);

    Mat warped_corners = homography * corners.t();

    for (int i = 0; i < 4; ++i)
    {
        Mat_<double> warped = warped_corners.col(i);

        assert(warped(2) != 0.0);
        double x = warped(0) / warped(2);
        double y = warped(1) / warped(2);

        minX = min(minX, x);
        minY = min(minY, y);
        maxX = max(maxX, x);
        maxY = max(maxY, y);

        if (i == 0)
        {
            minX = x;
            maxX = x;
            minY = y;
            maxY = y;
        }
        else
        {
            minX = min(x, minX);
            minY = min(y, minY);
            maxX = max(x, maxX);
            maxY = max(y, maxY);
        }
    }

    int start_row = minY;
    int start_col = minX;
    int warped_rows = (int)(ceil(maxY) - floor(minY));
    int warped_cols = (int)(ceil(maxX) - floor(minX));

    minX = min(0.0, minX);
    minY = min(0.0, minY);
    maxX = max((double)im2.cols, maxX);
    maxY = max((double)im2.rows, maxY);

    start_row -= minY;
    start_col -= minX;

    int rows = (int)(ceil(maxY) - floor(minY));
    int cols = (int)(ceil(maxX) - floor(minX));

    translation = (Mat_<double>(3, 3) << 1, 0, -minX,
                                         0, 1, -minY,
                                         0, 0,     1);

    Mat inv_homography = (translation * homography).inv();

    Mat merged = Mat::zeros(rows, cols, im1.type());
    im2.copyTo(merged(Rect(-minX, -minY, im2.cols, im2.rows)));

    // update to range of im2
    for (int i = start_row; i < start_row + warped_rows; ++i)
    // for (int i = 0; i < merged.rows; ++i)
    {
        for (int j = start_col; j < start_col + warped_cols; ++j)
        // for (int j = 0; j < merged.cols; ++j)
        {
            Mat pt = (Mat_<double>(3, 1) << j, i, 1);
            Mat_<double> inv_pt = inv_homography * pt;

            assert(inv_pt(2) != 0.0);
            double x = (inv_pt(0) / inv_pt(2));
            double y = (inv_pt(1) / inv_pt(2));

            if (x >= 0 && y >= 0 && x < im1.cols && y < im1.rows)
            {
                merged.at<Vec3b>(i, j) = im1.at<Vec3b>(y, x);
            }
        }
    }

    return merged;
}

Mat mergeImages(const vector<Mat>& images)
{
    assert(images.size() >= 2);

    vector<vector<ImageMatch> > matches;
    findPutativeMatches(images, matches);

    vector<vector<uchar> > inliers(matches.size());
    vector<Mat> homographies;
    for (int i = 0; i < images.size() - 1; ++i)
    {
        homographies.push_back(estimateHomography(matches[i], inliers[i]));
    }

    Mat merged = images.back();
    Mat homography_comp = Mat::eye(3, 3, CV_64F);
    Mat translation_comp = Mat::eye(3, 3, CV_64F);

    Mat translation;
    for (int i = images.size() - 2; i >= 0; --i)
    {
        homography_comp *= homographies[i];
        Mat homography = translation_comp * homography_comp;

        merged = mergeImages(images[i], merged, homography, translation);
        translation_comp *= translation;
    }

    return merged;
}

int main(int argc, char** argv)
{
    assert(argc >= 3);

    vector<Mat> images;
    for (int i = 1; i < argc; ++i)
    {
        images.push_back(imread(argv[i]));
    }

    Mat merged = mergeImages(images);
    // imshow("merged", merged);
    // waitKey();
    imwrite("panorama.jpg", merged);
}
