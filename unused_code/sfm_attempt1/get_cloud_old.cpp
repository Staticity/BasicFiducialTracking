cv::Mat_<double> VanillaTracker::_triangulatePoint(const cv::Point3d& u, const cv::Matx34d& P1,
                                                   const cv::Point3d& v, const cv::Matx34d& P2) const
{
    // std::cout << "start" << std::endl;
    cv::Matx44d A;

    for (int i = 0; i < 4; ++i)
    {
        A(0, i) = u.x * P1(2, i) - P1(0, i);
        A(1, i) = u.y * P1(2, i) - P1(1, i);
        A(2, i) = v.x * P2(2, i) - P2(0, i);
        A(3, i) = v.y * P2(2, i) - P2(1, i);
    }

    cv::Mat_<double> X_hom;
    cv::SVD::solveZ(A, X_hom);

    return X_hom;
}

bool VanillaTracker::_getCloud(const Input& in, Args& args, Output& out) const
{
    cv::Mat P1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0);
    static cv::Rect RotationROI     = cv::Rect(0, 0, 3, 3);
    static cv::Rect TranslationROI  = cv::Rect(3, 0, 1, 3);

    int best_index = -1;
    // int best_in_front = 0;
    double min_error = 0.0;
    for (int i = 0; i < args.rotations.size(); ++i)
    {
        cv::Mat rotation = args.rotations[i];
        for (int j = 0; j < args.translations.size(); ++j)
        {
            cv::Mat translation = args.translations[j];

            // Verified correct copy
            cv::Mat P2 = cv::Mat::zeros(3, 4, P1.type());
            rotation.copyTo(P2(RotationROI));
            translation.copyTo(P2(TranslationROI));

            double err = 0.0;
            int front = 0;
            std::vector<CloudPoint> temp_cloud;
            for (int k = 0; k < args.inliers1.size(); ++k)
            {
                cv::Point2d pt1 = args.inliers1[k];
                cv::Point2d pt2 = args.inliers2[k];

                cv::Point3d u = cv::Point3d(pt1.x, pt1.y, 1.0);
                cv::Point3d v = cv::Point3d(pt2.x, pt2.y, 1.0);

                // std::cout << "booga" << std::endl;

                cv::Mat_<double> x_hom  = _triangulatePoint(u, P1, v, P2);

                double ptx = x_hom(0) / x_hom(3);
                double pty = x_hom(1) / x_hom(3);
                double ptz = x_hom(2) / x_hom(3);

                cv::Mat_<double> x = (cv::Mat_<double>(3, 1) << ptx, pty, ptz);
                cv::Mat_<double> xp = rotation * x + translation;
                // std::cout << "booga 1" << std::endl;
                double z1 = x(2);
                double z2 = xp(2);

                if (z1 > 0.0 && z2 > 0.0)
                {
                // std::cout << "booga 2" << std::endl;
                    std::cout << in.camera.matrix().size() << " " << P1.size() << " " << x.size() << std::endl;
                    cv::Mat_<double> proj_pt1 = in.camera.matrix() * P1 * ;
                    cv::Mat_<double> proj_pt2 = in.camera.matrix() * P2 * xp;

                    if (Util::feq(proj_pt1(2), 0.0) || Util::feq(proj_pt1(2), 0.0)) continue;
                    cv::Point2d img_pt1(proj_pt1(0) / proj_pt1(2), proj_pt1(1) / proj_pt1(2));
                    cv::Point2d img_pt2(proj_pt2(0) / proj_pt2(2), proj_pt2(1) / proj_pt2(2));

                    double err1 = norm(img_pt1 - pt1);
                    double err2 = norm(img_pt2 - pt2);
                // std::cout << "booga 3" << std::endl;

                    err += err1 + err2;

                    ++front;
                    int index = temp_cloud.size();
                    temp_cloud.push_back(CloudPoint());
                    temp_cloud[index].pt = cv::Point3d(x(0), x(1), x(2));
                    temp_cloud[index].index = args.indices[k];
                }
            }

            err /= std::max(1, 2 * (int)args.inliers1.size());

            double percent = ((double) front) / std::max(1, (int)args.inliers1.size());;
            printf("%d %d : %0.6f\n", i, j, percent);

            if (percent > 0.75 && (err < min_error || out.points.empty()))
            {
                // std::cout << best_cloud.size() << " ";
                out.rotation = args.rotations[i];
                out.translation = args.translations[j];
                out.points = temp_cloud;
                // std::cout << " to " << out.points.size() << std::endl;

                out.visible_percent = percent;
                out.avg_reprojection_error = err;
            }
        }
    }

    return !out.points.empty();
}