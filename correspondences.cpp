#include <opencv.hpp>

#include <cstdio>
#include <string>

using namespace cv;

struct CloudPoint
{
    cv::Point3d pt;
    cv::Point2d img_pt;
    double reprojection_error;
};

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

bool get_rotation_and_translation_from_e(Mat E,
                                         std::vector<Mat>& rots,
                                         std::vector<Mat>& trans)
{
    SVD svd(E, SVD::MODIFY_A);

    double s1 = svd.w.at<double>(0);
    double s2 = svd.w.at<double>(1);

    assert(s1 != 0.0 && s2 != 0.0);

    double ratio_of_singular_values = fabs(s1 / s2);
    if (ratio_of_singular_values > 1.0)
        ratio_of_singular_values = 1.0 / ratio_of_singular_values;

    const double max_percent_diff = 0.7;
    if (ratio_of_singular_values < max_percent_diff)
    {
        printf("Singluar values of Essential matrix too far. Received %0.4f%% change.", ratio_of_singular_values);
        printf("Singular values: %0.2f & %0.2f\n", s1, s2);
        return false;
    }

    double w_data[3][3] =
    {
        { 0.0, -1.0,  0.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  0.0,  1.0}
    };

    Mat W = Mat(3, 3, CV_64F, w_data);

    rots.clear();
    rots.push_back(svd.u *   W   * svd.vt);
    rots.push_back(svd.u * W.t() * svd.vt);
    trans.clear();
    trans.push_back( svd.u.col(2));
    trans.push_back(-svd.u.col(2));

    return true;
}

bool get_rotation_and_translation(const std::vector<Point>& pts1,
                                  const std::vector<Point>& pts2,
                                  const Mat& camera_matrix_left,
                                  const Mat& camera_matrix_right,
                                  const Mat& distortion_coeffs,
                                  Mat& rotation,
                                  Mat& translation,
                                  std::vector<CloudPoint>& cloud)
{
    assert(pts1.size() == pts2.size());
    assert(camera_matrix_right.rows == 3);
    assert(camera_matrix_left.rows == 3);

    int n_points = pts1.size();
    std::vector<uchar> is_inlier(n_points);
    Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3., 0.99, is_inlier);
    Mat E = camera_matrix_right.t() * F * camera_matrix_left;

    printf("FUCK YEAH\n");

    float det_e = determinant(E);
    if (fabs(det_e) > 1e-07)
    {
        printf("Expected approximately 0 determinant of E. Received: %0.2f\n", det_e);
        return false;
    }

    std::vector<Mat> rots, trans;
    if (!get_rotation_and_translation_from_e(E, rots, trans))
        return false;

    assert(rots.size() == 2 && trans.size() == 2);

    double det_r1 = determinant(rots[0]);
    if (fabs(det_r1 - 1.0f) < 1e-09)
    {
        E = -E;
        if (!get_rotation_and_translation_from_e(E, rots, trans))
            return false;

        assert(rots.size() == 2 && trans.size() == 2);
        det_r1 = determinant(rots[0]);
    }

    if (fabs(det_r1) - 1.0f > 1e-07)
    {
        printf("Invalid rotation matrix. Expected -1 or 1 determinant, received: %0.4f\n", det_r1);
        return false;
    }

    std::vector<Point> inlier_points1;
    std::vector<Point> inlier_points2;
    for (int i = 0; i < n_points; ++i)
    {
        if (is_inlier[i])
        {
            inlier_points1.push_back(pts1[i]);
            inlier_points2.push_back(pts2[i]);
        }
    }
    printf("FUCK YEAH x2\n");

    int n_inlier_points = inlier_points1.size();

    Mat camera_matrix_left_inv = camera_matrix_left.inv();
    Mat camera_matrix_right_inv = camera_matrix_right.inv();

    double p_left_data[3][4] =
    {
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
    };

    double p_right_data[3][4] =
    {
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
    };

    Mat Pl = Mat(3, 4, CV_64F, p_left_data);

    // redundant assert for clarity
    assert(rots.size() * trans.size() == 4);
    std::vector<std::vector<CloudPoint> > clouds(4);
    std::vector<double> reprojection_errors_max(4);
    std::vector<bool> is_candidate(4);

    int index = 0;
    for (int i = 0; i < rots.size(); ++i)
    {
        for (int j = 0; j < trans.size(); ++j)
        {
            // printf("FUCK YEAH IN THE LOOP: %d, %d\n", i, j);
            // store the projection matrices
            for (int x = 0; x < 3; ++x)
                for (int y = 0; y < 3; ++y)
                    p_right_data[x][y] = rots[i].at<double>(x, y);
            for (int x = 0; x < 3; ++x)
                p_right_data[x][3] = trans[j].at<double>(x);

            Mat Pr = Mat(3, 4, CV_64F, p_right_data);
            // std::cout << Pr << std::endl;
            // std::cout << "Rotation: " << rots[i] << std::endl;
            // std::cout << "Translation: " << trans[j] << std::endl;

            double in_front_left  = 0;
            double in_front_right = 0;

            // Determine re-projection error
            double total_error_left  = 0.0;
            double total_error_right = 0.0;
            for (int k = 0; k < n_inlier_points; ++k)
            {
                // printf("HOLY SHIT I'M AT THAT %dth of %d INLIERS!!!\n", k, n_inlier_points);
                double  left_x = inlier_points1[k].x;
                double  left_y = inlier_points1[k].y;
                double right_x = inlier_points2[k].x;
                double right_y = inlier_points2[k].y;

                Mat homog_left  = (Mat_<double>(3, 1) <<  left_x,  left_y, 1.0);
                Mat homog_right = (Mat_<double>(3, 1) << right_x, right_y, 1.0);

                std::cout << homog_left << std::endl;

                Mat  left_3d = camera_matrix_left_inv  * homog_left;
                Mat right_3d = camera_matrix_right_inv * homog_right;

                in_front_left  += ( left_3d.at<double>(3) >= 0.0);
                in_front_right += (right_3d.at<double>(3) >= 0.0);

                // project points
                double lx = left_3d.at<double>(0);
                double ly = left_3d.at<double>(1);
                double lz = left_3d.at<double>(2);

                double rx = right_3d.at<double>(0);
                double ry = right_3d.at<double>(1);
                double rz = right_3d.at<double>(2);

                Mat  homog_left_3d = (Mat_<double>(4, 1) << lx, ly, lz, 1.0);
                Mat homog_right_3d = (Mat_<double>(4, 1) << rx, ry, rz, 1.0);

                Mat  reprojected_left = camera_matrix_left  * Pl * homog_left_3d;
                Mat reprojected_right = camera_matrix_right * Pr * homog_right_3d;

                double xw_l = reprojected_left.at<double>(0);
                double yw_l = reprojected_left.at<double>(1);
                double w_l  = reprojected_left.at<double>(2);
                assert(w_l != 0.0);

                double xw_r = reprojected_right.at<double>(0);
                double yw_r = reprojected_right.at<double>(1);
                double w_r  = reprojected_right.at<double>(2);
                assert(w_r != 0.0);

                double x_l = xw_l / w_l;
                double y_l = yw_l / w_l;

                // printf("[%d] Left 2d point: %0.2f %0.2f\n", k, left_x, left_y);
                // printf("[%d] Left 2d  proj: %0.2f %0.2f\n", k, x_l, y_l);
                // printf("[%d] Left 3d point: %0.2f %0.2f %0.2f\n", k, lx, ly, lz);
                // printf("\n");

                double x_r = xw_r / w_r;
                double y_r = yw_r / w_r;

                double re_l = ( left_x - x_l) * ( left_x - x_l) + ( left_y - y_l) * ( left_y - y_l);
                double re_r = (right_x - x_r) * (right_x - x_r) + (right_y - y_r) * (right_y - y_r);

                total_error_left  += re_l;
                total_error_right += re_r;

                // printf("Did this fuck up? :(\n");

                CloudPoint cloud_pt_left;
                cloud_pt_left.pt = Point3d(left_3d);
                cloud_pt_left.img_pt = Point2d(left_x, left_y);
                cloud_pt_left.reprojection_error = re_l;

                assert(index >= 0 && index < clouds.size());
                // printf("Cloud size: %d of %lu", index, clouds.size());
                clouds[index].push_back(cloud_pt_left);

                /*
                CloudPoint cloud_pt_right;
                cloud_pt_right.pt = Point3d(right_3d);
                cloud_pt_right.img_pt = Point2d(right_x, right_y);
                cloud_pt_right.reprojection_error = re_r;
                */
            }

            const double min_acc_in_front = 0.75;
            const double percent_in_front_l = ((float) in_front_left) / n_inlier_points;
            const double percent_in_front_r = ((float)in_front_right) / n_inlier_points;
            reprojection_errors_max[index] = max(total_error_left, total_error_right);

            printf("left: %0.2f\tright:%0.2f\terror:%0.2f\n", percent_in_front_l, percent_in_front_r, reprojection_errors_max[index]);
            is_candidate[index] = percent_in_front_l >= min_acc_in_front &&
                                  percent_in_front_r >= min_acc_in_front &&
                                  reprojection_errors_max[index] < 10e2;

            ++index;
        }
    }

    int best_index = (is_candidate[0]) ? 0 : -1;
    double min_error = reprojection_errors_max[0];
    for (int i = 1; i < 4; ++i)
    {
        if (is_candidate[index])
        {
            if (reprojection_errors_max[i] < min_error)
            {
                best_index = i;
                min_error = reprojection_errors_max[i];
            }
        }
    }

    // 0: R1 T1
    // 1: R1 T2
    // 2: R2 T1
    // 3: R2 T1
    // n: /2 %2
    if (best_index != -1)
    {
        cloud       = clouds[best_index];
        rotation    = rots[best_index / 2];
        translation = trans[best_index % 2];
    }

    return best_index != -1;
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

    std::string camera_calib_filename = "calibration_data/out_camera_data_1.xml";
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
    find_correspondences(im1, im2, pts1, pts2);
    assert(pts1.size() == pts2.size());

    Mat rotation, translation;
    std::vector<CloudPoint> cloud;
    bool found = get_rotation_and_translation(
        pts1,
        pts2,
        camera_matrix,
        camera_matrix,
        distortion_coeff,
        rotation,
        translation,
        cloud);

    printf("Found cloud: %d\n", found);

    if (0)
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

    // waitKey();
}
