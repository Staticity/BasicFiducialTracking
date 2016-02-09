#include "disjoint_set.hpp"
#include <iostream>

using namespace std;

int main()
{
    disjoint_set ds;
    ds.add(2);
    ds.add(3);
    ds.add(4);
    ds.join(2, 4);
    cout << ds << endl;
}