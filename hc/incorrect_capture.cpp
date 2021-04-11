#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <hc.hpp>

namespace {
    class array_my
    {
    public:
        int val;
    };
}

int main(int argc, char ** argv)
{
    int size = 50;
    hc::accelerator default_acc;
    hc::accelerator_view acc_view = default_acc.get_default_view();
    hc::array<int> device_data(hc::extent<1>(size), acc_view);
    std::wcout << "Using device: " << default_acc.get_description() << std::endl;
    array_my arr;
    arr.val = 5;
  
    hc::parallel_for_each(hc::extent<1>(size),
           [&device_data, arr](hc::index<1> id) [[hc]]
            {
                device_data[id] = arr.val;
            }
    ); 

    int * vals = new int[size];
    hc::copy(device_data, vals);
    std::for_each(vals, vals + size, [](int & v){ assert(v == 5); });
    delete[] vals;

    return 0;
}
