#ifndef DISJOINT_SET
#define DISJOINT_SET

#include <map>
#include <iostream>

// Simple, inefficient disjoint set implementation
// Should probably template it...
class disjoint_set
{
private:
    std::map<int, int> parent;

public:

    disjoint_set() {}

    void add(int a)
    {
        parent[a] = a;
    }

    void join(int a, int b)
    {
        parent[find(b)] = find(a);
    }

    int find(int a)
    {
        return a == parent[a] ? a : parent[a] = find(parent[a]);
    }

    bool check(int a, int b)
    {
        return find(a) == find(b);
    }

    friend std::ostream& operator<<(std::ostream& os, disjoint_set& ds);
};

std::ostream& operator<<(std::ostream& os, disjoint_set& ds)
{
    std::map<int, int>::iterator it;
    for (it = ds.parent.begin(); it != ds.parent.end(); ++it)
        os << it->first << ": " << it->second << '\n';
    return os; 
}

#endif