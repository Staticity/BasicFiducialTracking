
bool find_square(Mat grayImage)
{
    if (grayImage.channels() != 1)
    {
        std::cout << "[find_square]: expected grayscale image, returning false" << std::endl;
        return false;
    }

    Mat adapt;
    adaptiveThreshold(
        grayImage,
        adapt,
        255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY,
        7,
        10
    );

    int erosion_size = 1;
    int square_size  = erosion_size * 2 + 1;
    Mat element = getStructuringElement(
        MORPH_RECT,
        Size(square_size, square_size),
        Point(erosion_size, erosion_size)
    );

    erode(adapt, adapt, element);

    
    disjoint_set ds;
    Mat labels = Mat::zeros(grayImage.size(), CV_32SC1);

    clock_t t = clock();
    // do label based connected components with a
    // disjoint set data structure.
    int label = 0;
    for (int i = 0; i < adapt.rows; ++i)
    {
        for (int j = 0; j < adapt.cols; ++j)
        {

            uchar   cur_pixel = adapt.at<uchar>(i, j);
            int&    cur_label = labels.at<int>(i, j);
            int      up_label = (i == 0) ? -1 : labels.at< int >(i - 1, j    );
            int    left_label = (j == 0) ? -1 : labels.at< int >(i    , j - 1);
            uchar    up_pixel = (i == 0) ? -1 :  adapt.at<uchar>(i - 1, j    );
            uchar  left_pixel = (j == 0) ? -1 :  adapt.at<uchar>(i    , j - 1);

            bool same_left = left_pixel == cur_pixel;
            bool same_up   =   up_pixel == cur_pixel;

            if (same_left)
            {
                if (same_up)
                {
                    cur_label = min(up_label, left_label);
                    ds.join(up_label, left_label);
                }
                else
                {
                    cur_label = left_label;
                }
            }
            else if (same_up)
            {
                cur_label = up_label;
            }
            else
            {
                ++label;
                labels.at<uint>(i, j) = label;
                ds.add(label);
            }
        }
    }

    if (0)
    {
        RNG rng(0);
        map<int, Vec3b> colorMap;
        for (int i = 1; i <= label; ++i)
        {
            colorMap[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }

        Mat assignment = Mat::zeros(grayImage.size(), CV_8UC3);
        for (int i = 0; i < assignment.rows; ++i)
        {
            for (int j = 0; j < assignment.cols; ++j)
            {
                int label_value = ds.find(labels.at<int>(i, j));
                assignment.at<Vec3b>(i, j) = colorMap[label_value];
            }
        }

        imshow("Connections", assignment);
        waitKey();
    }

    t = clock() - t;
    cout << ((float) t) / CLOCKS_PER_SEC << endl;
    
    // imshow("Find", adapt);
    return true;
}