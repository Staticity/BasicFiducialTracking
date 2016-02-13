#include <opencv.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    cout << (Size(3, 3) == Size(2 + 1, 3)) << endl;
}
