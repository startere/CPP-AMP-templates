#include <iostream>

#include <hc.hpp>

namespace amp = hc;

struct param
{
    int some_value;
    float other_val;
};

struct specific_data
{
    specific_data() restrict(amp, cpu)
    {
        x_ = 2.0;
    }

    specific_data(int val, const std::string &) restrict(amp, cpu)
    {
        x_ = 1.0;
    }

    [[hc]] specific_data(int val, const param & param_value) restrict(amp,cpu)
    {
        x_ = val + param_value.some_value; 
    }  

    [[hc]] ~specific_data() restrict(amp, cpu)
    {

    }

    [[hc, cpu]] int& x() restrict(amp, cpu)
    {
        return x_;
    }

private:
    int x_;
};

template<typename T>
struct wrapper
{
    amp::array<T> & buffer;
};


template<typename F, typename... Args>
void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, const Args &... args)
{
    amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
        [=](amp::tiled_index<1> idx) [[hc]] {
            f(idx.global[0], args...);
        }
    );
}

template<typename F, typename... Args>
void launch(amp::accelerator_view & acc_view, int threads, F && f, const Args &... args)
{
    amp::parallel_for_each(acc_view, amp::extent<1>(threads),
        [=](amp::index<1> idx) [[hc]] {
            f(idx[0], args...);
        }
    );
}

template<typename T>
void destruct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer)
{
    launch(acc_view, threads,5, 
        [](const int idx, wrapper<T> p) [[hc]] {
            (p.buffer)[ idx ].~T(); 
        }, pointer);
}

template<typename T, typename... Args>
void construct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer, const Args &... args)
{
    launch(acc_view, threads, 
        [](const int idx, wrapper<T> p, Args const &... args) [[hc]] {
            new (&p.buffer[ idx ]) T(args...);
            (p.buffer)[ idx ].x()++;
        }, pointer, args...);
}

int main(int argc, char ** argv)
{
	const int size = 50;
	amp::accelerator default_acc;
	amp::accelerator_view acc_view = default_acc.get_default_view();
	amp::array<specific_data> device_data(amp::extent<1>(size), acc_view);
	std::wcout << "Using device: " << default_acc.get_description() << std::endl;

    int val = 3.0;
    std::string s = "abc";
    param r{1, 3.0};
    
    wrapper<specific_data> p{device_data};

    construct(acc_view, size, p, val, r);
    
    acc_view.wait();

    amp::array_view<specific_data> data_view(device_data);
    for(int i = 0; i < size; ++i)
        assert(data_view[i].x() == val + r.some_value + 1);

    destruct(acc_view,size, p);	
    acc_view.wait(); 

    return 0;
}
