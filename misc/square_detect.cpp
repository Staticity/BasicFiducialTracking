#include <opencv.hpp>

#include <string>
#include <iostream>
#include <map>
#include <cmath>
#include <ctime>
#include <utility>
#include <assert.h>
#include <cstdio>

using namespace cv;
using namespace std;

bool poly_in_poly(const vector<Point>& container, const vector<Point>& query)
{
    for (int i = 0; i < query.size(); ++i)
    {
        if (pointPolygonTest(container, query[i], false) < -0.5f)
            return false;
    }

    return true;
}

Point centroid(const vector<Point>& poly)
{
    Point2f center = Point2f(0, 0);
    for (int i = 0; i < poly.size(); ++i)
        center += Point2f(poly[i]);
    return center / std::max(1, (int) poly.size());
}

void draw_poly(const Mat& im, const vector<Point>& poly, Scalar color)
{
    int n = poly.size();
    for (int i = 0; i <n; ++ i)
    {
        int i1 = i % n;
        int i2 = (i + 1) % n;
        line(im, poly[i1], poly[i2], color, 2);
    }
}

bool find_black_quad(Mat& image, vector<Point>& quad)
{
    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    Mat adapt;
    adaptiveThreshold(
        gray,
        adapt,
        255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY,
        7,
        10
    );

    // imshow("adapt", adapt);

    int erosion_size = 3;
    int square_size  = erosion_size * 2 + 1;
    Mat element = getStructuringElement(
        MORPH_RECT,
        Size(square_size, square_size),
        Point(erosion_size, erosion_size)
    );

    erode(adapt, adapt, element);
    // imshow("eroded", adapt);

    Mat labels;
    connectedComponents(adapt, labels, 4);

    Mat thresh;
    threshold(
        gray,
        thresh,
        100,
        255,
        THRESH_BINARY
    );

    // imshow("thresh", thresh);

    double maxLabel;
    minMaxLoc(labels, NULL, &maxLabel);

    int labelCount = (int)(maxLabel) + 1;
    vector<int> blackCount(labelCount);
    vector<vector<Point> > points(labelCount);

    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label        = labels.at<int>(i, j);
            int thresh_value = thresh.at<uchar>(i, j);

            blackCount[label] += thresh_value == 0;  
            points[label].push_back(Point(j, i));
        }
    }

    float percent_acceptance = 0.90;
    int min_size_black_acc = 500;
    int min_size_white_acc = min_size_black_acc / 10;

    vector<int> candidates_s1;
    vector<int> white_candidates;
    for (int i = 0; i < labelCount; ++i)
    {
        int numPixels = points[i].size();
        float percent_black = ((float)blackCount[i]) / max(1, numPixels);
        float percent_white = 1.0 - percent_black;

        if (percent_black >= percent_acceptance)
        {
            if (numPixels >= min_size_black_acc)
                candidates_s1.push_back(i);
        }
        else if (percent_white >= percent_acceptance)
        {
            if (numPixels >= min_size_white_acc)
                white_candidates.push_back(i);
        }
    }

    float max_growth_percent = 1.001f;
    vector<Point> white_poly_centroids;
    for (int i = 0; i < white_candidates.size(); ++i)
    {
        int index = white_candidates[i];

        vector<Point> hull;
        convexHull(Mat(points[index]), hull, true);
        int original_size = points[index].size();
        float    new_size = contourArea(hull);

        float inc_in_size = new_size / max(1, original_size);
        if (inc_in_size > max_growth_percent)
            continue;

        approxPolyDP(hull, hull, 3, true);

        white_poly_centroids.push_back(centroid(hull));
        // circle(image, white_poly_centroids.back(), 10, Scalar(0, 255, 0), 2);
    }

    max_growth_percent = 1.25f;
    vector<int> candidates_s2;
    vector<vector<Point> > polys;
    for (int i = 0; i < candidates_s1.size(); ++i)
    {
        int index = candidates_s1[i];
        vector<Point> hull;
        convexHull(Mat(points[index]), hull, true);
        int original_size = points[index].size();
        float    new_size = contourArea(hull);

        float inc_in_size = new_size / max(1, original_size);

        if (inc_in_size > max_growth_percent)
            continue;

        approxPolyDP(hull, hull, 3, true);


        if (hull.size() == 4)
        {
            // draw_poly(image, hull, Scalar(0, 0, 255));
            bool contains_white_poly = false;
            for (int j = 0; j < white_poly_centroids.size(); ++j)
            {
                if (pointPolygonTest(hull, white_poly_centroids[j], false) > -0.5f)
                {
                    contains_white_poly = true;
                    break;
                }
            }

            if (contains_white_poly)
            {
                candidates_s2.push_back(index);
                polys.push_back(hull);
            }
        }
    }

    int best_index = -1;
    int best_size = -1;
    for (int i = 0; i < candidates_s2.size(); ++i)
    {
        int index = candidates_s2[i];
        int numPixels = points[index].size();
        if (numPixels > best_size)
        {
            best_index = i;
            best_size = numPixels;
        }
    }
    
    if (best_index == -1)
        return false;

    quad = polys[best_index];

    return true;
}

void sort_quad_corners(Mat& image,
                       const vector<Point>& quad,
                       vector<Point>& sorted_quad)
{
    assert(quad.size() == 4);
    assert(&quad != &sorted_quad);

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    Mat thresh;
    threshold(
        gray,
        thresh,
        155,
        255,
        THRESH_BINARY
    );

    Rect r = boundingRect(quad);
    Point2f white_centroid = Point(0, 0);
    int num_white_points = 0;
    for (int i = 0; i < r.height; ++i)
    {
        for (int j = 0; j < r.width; ++j)
        {
            int x = j + r.x;
            int y = i + r.y;

            // if (x, y) is in the quad
            if (pointPolygonTest(quad, Point(x, y), false) > -0.5)
            {
                if (thresh.at<uchar>(y, x) != 0)
                {
                    white_centroid.x += x;
                    white_centroid.y += y;
                    ++num_white_points;
                }
            }
        }
    }

    // center of white pixels
    white_centroid /= max(1, num_white_points);

    // circle(image, white_centroid, 5, Scalar(255, 0, 255));

    // find quad corner closes to white centroid
    int white_corner_index = 0;
    float white_sq_distance = -1.0f;
    for (int i = 0; i < 4; ++i)
    {
        Point2f dv = white_centroid - Point2f(quad[i]);
        float sq_distance = dv.x * dv.x + dv.y * dv.y;
        
        if (sq_distance <= white_sq_distance ||
            white_sq_distance < 0.0f)
        {
            white_corner_index = i;
            white_sq_distance = sq_distance;
        }
    }

    Point2f quad_centroid = (quad[0] + quad[1] + quad[2] + quad[3]) / 4.0f;
    vector<pair<double, int> > point_angle_tuples(4);

    // circle(image, quad_centroid, 5, Scalar(255, 255, 0));

    // sort points in clockwise order
    Point2f white_vector = Point2f(quad[white_corner_index]) - quad_centroid;
    for (int i = 0; i < 4; ++i)
    {
        Point2f corner_vector = Point2f(quad[i]) - quad_centroid;
        double dot = white_vector.ddot(corner_vector);
        double det = white_vector.cross(corner_vector);
        double ang = atan2(det, dot);
        while (ang < 0.0)
            ang += 360.0;
        point_angle_tuples[i] = make_pair(ang, i);
    }

    sort(point_angle_tuples.begin(), point_angle_tuples.end());

    sorted_quad = vector<Point>(4);
    for (int i = 0; i < 4; ++i)
    {
        int sorted_index = point_angle_tuples[i].second;
        sorted_quad[i] = quad[sorted_index];
    }
}

void draw_sorted_corners(Mat& image,
                         vector<Point>& quad,
                         Scalar line_color=Scalar(128, 128, 128))
{
    draw_poly(image, quad, line_color);

    Scalar colors[4] = 
    {
        Scalar(0, 0, 255),
        Scalar(0, 255, 0),
        Scalar(255, 0, 0),
        Scalar(255, 255, 255)
    };

    for (int i = 0; i < 4; ++i)
    {
        circle(image, quad[i], 7, colors[i], 2);
    }
}

Mat draw_image_in_quad(Mat& image, vector<Point>& quad, const Mat& imposed, Mat homography = Mat())
{
    vector<Point> src_points;
    src_points.push_back(Point(0, 0));
    src_points.push_back(Point(imposed.cols, 0));
    src_points.push_back(Point(imposed.cols, imposed.rows));
    src_points.push_back(Point(0, imposed.rows));

    if (homography.rows == 0 || homography.cols == 0)
    {
        homography = findHomography(src_points, quad);
    }

    Mat homography_inv = homography.inv();
    Rect r = boundingRect(quad);

    for (int i = 0; i < r.height; ++i)
    {
        for (int j = 0; j < r.width; ++j)
        {
            int x = j + r.x;
            int y = i + r.y;

            // if (x, y) is in the quad
            if (pointPolygonTest(quad, Point(x, y), false) > -0.5)
            {
                // lookup pixel value
                // H * src == dest ->
                // src ~= H^-1 * dest

                Mat p = Mat::zeros(3, 1, CV_64F);
                p.at<double>(0, 0) = x;
                p.at<double>(1, 0) = y;
                p.at<double>(2, 0) = 1;

                Mat q = homography_inv * p;
                double wx = q.at<double>(0, 0);
                double wy = q.at<double>(1, 0);
                double w  = q.at<double>(2, 0);

                assert(w != 0.0f);

                int qx = (int)(round(wx / w));
                int qy = (int)(round(wy / w));

                if (qx >= 0 && qy >= 0 && qx < imposed.cols && qy < imposed.rows)
                {
                    // printf("(%d, %d) -> (%d, %d)\n", y, x, qy, qx);
                    image.at<Vec3b>(y, x) = imposed.at<Vec3b>(qy, qx);
                }
                else
                {
                    // cout << qx << ", " << qy << endl;
                }
            }
        }
    }

    return homography;
}

// Kind of slow way to do it. Should update quad to represent new region
void draw_image_in_quad_rec(Mat& image, vector<Point> quad, int its=3)
{
    assert(quad.size() == 4);

    Mat homography = Mat();
    for (int i = 1; i < its; ++i)
    {
        Mat image_clone = image.clone();
        homography = draw_image_in_quad(image, quad, image_clone, homography);

        for (int j = 0; j < quad.size(); ++j)
        {
            Mat p = Mat(3, 1, CV_64F);
            p.at<double>(0, 0) = quad[j].x;
            p.at<double>(1, 0) = quad[j].y;
            p.at<double>(2, 0) = 1;

            Mat q = homography * p;
            float wx = q.at<double>(0, 0);
            float wy = q.at<double>(1, 0);
            float w  = q.at<double>(2, 0);

            assert(w != 0.0);

            quad[j] = Point(wx / w, wy / w);
        }
    }
}

void get_square_pose(const vector<Point>& quad, 
                     const Mat& camera_matrix,
                     const Mat& distortion_coeff,
                     Mat& rot, Mat& trans)
{
    assert(quad.size() == 4);

    double object_points[4][3] =
    {
        {-1.0,  1.0,  0.0f},
        { 1.0,  1.0,  0.0f},
        { 1.0, -1.0,  0.0f},
        {-1.0, -1.0,  0.0f},
    };

    Mat object_mat = Mat(quad.size(), 3, CV_64F, object_points);
    Mat quad_mat = Mat(quad.size(), 2, CV_64F);

    for (int i = 0; i < quad.size(); ++i)
    {
        quad_mat.at<double>(i, 0) = quad[i].x;
        quad_mat.at<double>(i, 1) = quad[i].y;
    }

    solvePnP(object_mat, quad_mat, camera_matrix, distortion_coeff, rot, trans);
}

void draw_cube_with_pose(Mat& image,
                         const Mat& rot, const Mat& trans,
                         const Mat& camera_matrix,
                         const Mat& distortion_coeff)
{
    double cube_points[8][3] =
    {
        {-1.0,  1.0,  0.0f},
        { 1.0,  1.0,  0.0f},
        { 1.0, -1.0,  0.0f},
        {-1.0, -1.0,  0.0f},
        {-1.0,  1.0,  1.0f},
        { 1.0,  1.0,  1.0f},
        { 1.0, -1.0,  1.0f},
        {-1.0, -1.0,  1.0f},
    };

    Mat cube_object_coordinates = Mat(8, 3, CV_64F, cube_points);

    Mat projected_points;
    projectPoints(
        cube_object_coordinates,
        rot,
        trans,
        camera_matrix,
        distortion_coeff,
        projected_points);

    vector<Point> cube_projected_points(8);
    for (int i = 0; i < 8; ++i)
    {
        cube_projected_points[i] = Point2f(0.0, 0.0);
        cube_projected_points[i].x = projected_points.at<double>(i, 0);
        cube_projected_points[i].y = projected_points.at<double>(i, 1);
    }
    
    Scalar color = Scalar(0, 255, 0);
    for (int i = 0; i < 4; ++i)
    {
        int i1 = i;
        int i2 = (i + 1) % 4;
        int i3 = i1 + 4;
        int i4 = i2 + 4;
        line(image, cube_projected_points[i1], cube_projected_points[i2], color, 3);
        line(image, cube_projected_points[i3], cube_projected_points[i4], color, 3);
        line(image, cube_projected_points[i1], cube_projected_points[i3], color, 3);
    }
}

// assumes no skew
void update_camera_matrix_resize(Mat& camera_matrix, Size old_size, Size new_size)
{
    assert(old_size.width * new_size.height == old_size.height * new_size.width);
    assert(old_size.width != 0 && old_size.height != 0);

    double width_ratio = ((double) new_size.width) / old_size.width;
    double height_ratio = ((double) new_size.height) / old_size.height;

    camera_matrix.at<double>(0, 0) *= width_ratio;
    camera_matrix.at<double>(1, 1) *= height_ratio;
    camera_matrix.at<double>(0, 2) = new_size.width * 0.5;
    camera_matrix.at<double>(1, 2) = new_size.height * 0.5;
}

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    string camera_calib_filename = "../calibration_data/out_camera_data_1.xml";
    FileStorage fs(camera_calib_filename, FileStorage::READ);

    Mat camera_matrix, distortion_coeff;
    int camera_width, camera_height;
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coeff;
    fs["image_width"] >> camera_width;
    fs["image_height"] >> camera_height;

    int width = 1280;
    int height = 720;

    update_camera_matrix_resize(
        camera_matrix,
        Size(camera_width, camera_height),
        Size(width, height));

    fs.release();

    VideoCapture vc(index);
    vc.set(CV_CAP_PROP_FRAME_WIDTH, width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    if (!vc.isOpened()) return 0;

    Mat image;
    vc >> image;

    namedWindow("outlined_source", CV_WINDOW_NORMAL);
    vector<Point> quad;
    while (vc.isOpened())
    {
        vc >> image;
        bool found = find_black_quad(image, quad);

        if (quad.size() == 4)
        {
            if (found)
            {
                vector<Point> sorted_quad;
                sort_quad_corners(image, quad, sorted_quad);
                quad = sorted_quad;

                // draw_sorted_corners(image, quad);
                // draw_image_in_quad(image, quad, image.clone());
                // draw_image_in_quad_rec(image, quad, 3);

                Mat rot, trans;
                get_square_pose(quad, camera_matrix, distortion_coeff, rot, trans);
                // cout << "Rotation\n" << rot << endl;
                // cout << "Translation\n" << trans << endl << endl;

                draw_cube_with_pose(image, rot, trans, camera_matrix, distortion_coeff);
            }
        }

        // Mat resized_image;
        // resize(image, resized_image, Size(width, height));
        imshow("outlined_source", image);

        if (waitKey(20) == 27)
            break;
    }
}
