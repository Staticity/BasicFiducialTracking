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

    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(/*400*/);
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    BFMatcher matcher = BFMatcher::BFMatcher(NORM_L2, true);

    vector<Mat> descriptors(images.size());
    vector<vector<KeyPoint> > features(images.size());
    matches = vector<vector<ImageMatch> >(images.size() - 1);

    for (int i = 0; i < images.size(); ++i)
    {
        assert(!images[i].empty());

        detector->detect(images[i], features[i]);
        assert(!features[i].empty());

        extractor->compute(images[i], features[i], descriptors[i]);
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
    const vector<uchar>& mask = vector<uchar>())
{
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
            if (!mask.empty() && !mask[j])
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

void randomIndices(int k, int n, vector<int>& indices)
{
    RNG rng(getTickCount());
    indices.clear();
    indices.resize(k);

    for (int i = 0; i < k; ++i)
    {
        indices[i] = i;
    }

    for (int i = k; i < n; ++i)
    {
        int j = rng.uniform(0, i + 1);
        if (j < k)
            indices[j] = i;
    }

    assert(indices.size() == k);
    for (int i = 0; i < indices.size(); ++i)
    {
        assert(indices[i] >= 0 && indices[i] < n);
    }
}

Mat createHomographyMatrix(const vector<ImageMatch>& matches)
{
    Mat A = Mat::zeros(matches.size() * 2, 8, CV_64F);

    for (int i = 0; i < A.rows; i += 2)
    {
        int index = i / 2;
        double x1 = matches[index].pt1.x;
        double y1 = matches[index].pt1.y;
        double x2 = matches[index].pt2.x;
        double y2 = matches[index].pt2.y;

        Mat xrow = (Mat_<double>(1, 8) << x1, y1, 1,  0,  0, 0, -x1 * x2, -y1 * x2);
        Mat yrow = (Mat_<double>(1, 8) <<  0,  0, 0, x1, y1, 1, -x1 * y2, -y1 * y2);
        xrow.copyTo(A(Rect(0,   i  , 8, 1)));
        yrow.copyTo(A(Rect(0, i + 1, 8, 1)));
    }

    return A;
}

Mat estimateHomography(
    const vector<ImageMatch>& matches,
    vector<uchar>& inliers)
{
    double inlier_threshold = 1.0; // Need to change
    double update_ratio = 0.9; // Need to change
    int iterations = 1000; // Need to change

    Mat best_homography;
    vector<int> best_inliers;
    double best_avg_error = -1.0;

    int max_inliers = 0;
    while (iterations--)
    {
        // choose first 4 elements
        vector<int> indices;
        randomIndices(4, matches.size(), indices);

        vector<ImageMatch> subsetMatches;
        for (int i = 0; i < indices.size(); ++i)
        {
            assert(indices[i] < matches.size());
            subsetMatches.push_back(matches[indices[i]]);

            assert(matches[indices[i]].pt1 == subsetMatches[i].pt1);
            assert(matches[indices[i]].pt2 == subsetMatches[i].pt2);
        }

        // Set up matrix Ax = b, where x contains homography elements
        Mat A = createHomographyMatrix(subsetMatches);
        Mat B = Mat::zeros(subsetMatches.size() * 2, 1, CV_64F);

        for (int i = 0; i < B.rows; i += 2)
        {
            int index = i / 2;
            B.at<double>(i) = subsetMatches[index].pt2.x;
            B.at<double>(i + 1) = subsetMatches[index].pt2.y;
        }

        // Solve for homography elements
        Mat_<double> X;

        // Skip collinear points
        if (!solve(A, B, X, DECOMP_LU))
        {
            continue;
        }

        Mat H = (Mat_<double>(3, 3) << X(0), X(1), X(2),
                                      X(3), X(4), X(5),
                                      X(6), X(7),   1);

        vector<int> inlier_indices;
        for (int i = 0; i < matches.size(); ++i)
        {
            double x1 = matches[i].pt1.x;
            double y1 = matches[i].pt1.y;    
            double x2 = matches[i].pt2.x;
            double y2 = matches[i].pt2.y;            
            
            Mat_<double> sourcePt = (Mat_<double>(3, 1) << x1, y1, 1);
            Mat_<double> destPt = H * sourcePt;

            if (destPt(2) == 0.0)
            {
                continue;
            }
            // assert(destPt(2) != 0.0);
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
        if (inlier_indices.size() < indices.size())
        {
            continue;
        }

        // Potentially better match, solve for inlier solution
        if (inlier_indices.size() >= max_inliers * update_ratio)
        {
            subsetMatches.clear();
            subsetMatches.resize(inlier_indices.size());
            for (int i = 0; i < inlier_indices.size(); ++i)
            {
                subsetMatches[i] = matches[inlier_indices[i]];
            }

            A = createHomographyMatrix(subsetMatches);
            Mat B = Mat::zeros(subsetMatches.size() * 2, 1, CV_64F);

            for (int i = 0; i < B.rows; i += 2)
            {
                int index = i / 2;
                B.at<double>(i) = subsetMatches[index].pt2.x;
                B.at<double>(i + 1) = subsetMatches[index].pt2.y;
            }

            assert(solve(A, B, X, DECOMP_NORMAL));

            H = (Mat_<double>(3, 3) << X(0), X(1), X(2),
                                       X(3), X(4), X(5),
                                       X(6), X(7),   1);

            inlier_indices.clear();
            double total_error = 0.0f;
            for (int i = 0; i < matches.size(); ++i)
            {
                double x1 = matches[i].pt1.x;
                double y1 = matches[i].pt1.y;    
                double x2 = matches[i].pt2.x;
                double y2 = matches[i].pt2.y;            
                
                Mat_<double> sourcePt = (Mat_<double>(3, 1) << x1, y1, 1);
                Mat_<double> destPt = H * sourcePt;

                assert(destPt(2) != 0.0f);
                double hx2 = destPt(0) / destPt(2);
                double hy2 = destPt(1) / destPt(2);

                double error = sqrt((hx2 - x2) * (hx2 - x2) + (hy2 - y2) * (hy2 - y2));
                total_error += error;

                if (error <= inlier_threshold)
                {
                    inlier_indices.push_back(i);
                }
            }

            double avg_error = total_error / matches.size();
            if (avg_error < best_avg_error || best_avg_error < 0)
            {
                if (inlier_indices.size() > max_inliers)
                {
                    max_inliers = inlier_indices.size();
                }

                best_avg_error = avg_error;
                best_homography = H;
                best_inliers = inlier_indices;
            }
        }
    }

    inliers.clear();
    inliers.resize(matches.size());
    for (int i = 0; i < best_inliers.size(); ++i)
    {
        inliers[best_inliers[i]] = 1;
    }

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
    int warped_rows = start_row + (int)(ceil(maxY) - floor(minY));
    int warped_cols = start_col + (int)(ceil(maxX) - floor(minX));

    minX = min(0.0, minX);
    minY = min(0.0, minY);
    maxX = max((double)im2.cols, maxX);
    maxY = max((double)im2.rows, maxY);

    int rows = (int)(ceil(maxY) - floor(minY));
    int cols = (int)(ceil(maxX) - floor(minX));

    translation = (Mat_<double>(3, 3) << 1, 0, -minX,
                                         0, 1, -minY,
                                         0, 0,     1);

    Mat inv_homography = (translation * homography).inv();

    Mat merged = Mat::zeros(rows, cols, im1.type());
    im2.copyTo(merged(Rect(-minX, -minY, im2.cols, im2.rows)));

    // update to range of im2
    // for (int i = p; i < warped_rows; ++i)
    for (int i = 0; i < merged.rows; ++i)
    {
        // for (int j = start_col; j < warped_cols; ++j)
        for (int j = 0; j < merged.cols; ++j)
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

    vector<uchar> inliers;
    vector<Mat> homographies;
    for (int i = 0; i < images.size() - 1; ++i)
    {
        homographies.push_back(estimateHomography(matches[i], inliers));
    }

    Mat merged = images.back();
    Mat homography = Mat::eye(3, 3, CV_64F);
    Mat translation = Mat::eye(3, 3, CV_64F);
    for (int i = images.size() - 2; i >= 0; --i)
    {
        homography = translation * homographies[i] * homography;
        merged = mergeImages(images[i], merged, homography, translation);
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
    imwrite("panorama.jpg", merged);
}
