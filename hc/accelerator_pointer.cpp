#include <cstdlib>
#include <iostream>
#include <hc.hpp>

int main(int argc, char ** argv)
{
    int size = 50;
    hc::accelerator default_acc;
    hc::accelerator_view acc_view = default_acc.get_default_view();
    hc::array<int> device_data(hc::extent<1>(size), acc_view);
    std::wcout << "Using device: " << default_acc.get_description() << std::endl;
   
    int * ptr = device_data.accelerator_pointer();
    // Now create an array view enforcing a change of queue
    // in rw_info, named 'curr'
    hc::array_view<int> data_view(device_data);
    data_view[0] = 0; 
    // Now pointer will point to CPU data cached in array_view
    int * second_ptr = device_data.accelerator_pointer();
    assert(second_ptr == data_view.data());

    // This will not fail, because accelerator pointer
    // in GPU array has changed
    assert(second_ptr != ptr);

    return 0;
}
