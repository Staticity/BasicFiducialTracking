            cv::Mat pts_homog_3d(1, num_points, CV_64FC4);
            std::vector<cv::Point3d> pts_3d;
            cv::triangulatePoints(P1, P2, args.undistortedPts1.reshape(1, 2), args.undistortedPts2.reshape(1, 2), pts_homog_3d);
            cv::convertPointsFromHomogeneous(pts_homog_3d.reshape(4, 1), pts_3d);


            // NOTE:
            // This assumes that no points have w = 0, since
            // as of 2/13/2016 convertPointsFromHomogenous
            // doesn't zero out the vector if w = 0.
            std::vector<unsigned char> status(num_points);
            for (int k = 0; k < num_points; ++k)
            {
                status[k] = (pts_3d[k].z > 0.0);
            }

            int num_in_front  = cv::countNonZero(status);
            double front_percent = ((double)(num_in_front)) / std::max(1, num_points);

            // std::cout << front_percent << std::endl;

            // Not enough points in front
            if (front_percent < min_percent) continue;

            cv::Vec3d rvec, tvec;
            Rodrigues(P1(RotationROI), rvec);
            tvec = P1(TranslationROI);

            std::vector<cv::Point2d> reprojected_pts1;
            cv::projectPoints(pts_3d, rvec, tvec, in.camera.matrix(), in.camera.distortion(), reprojected_pts1);

            const double error_total = cv::norm(cv::Mat(reprojected_pts1), cv::Mat(args.inliers1), cv::NORM_L2); 
            const double error_avg   = error_total / std::max((int)reprojected_pts1.size(), 1);

            if (error_avg > min_error) continue;

            for (int k = 0; k < num_points; ++k)
            {
                const double error = cv::norm(args.inliers1[k] - reprojected_pts1[k]);
                status[k] &= (error < 2 * min_error);
            }

            out.rotation               = rotation;
            out.translation            = translation;
            out.visible_percent        = front_percent;
            out.avg_reprojection_error = error_avg;

            out.points.clear();
            for (int k = 0; k < num_points; ++k)
            {
                if (status[k])
                {
                    out.points.push_back(CloudPoint());
                    int out_index = out.points.size() - 1;
                    out.points[out_index].pt = pts_3d[k];
                    out.points[out_index].index = args.indices[k];
                }
            }