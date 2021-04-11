#include <iostream>
#include <cstdlib>

#include <amp.h>

namespace amp = Concurrency;

template<typename T>
struct copy_accessor
{
    copy_accessor(amp::array<T> & data, const uint32_t access_size) : 
        data(data), access_size(access_size), read_vals( new T[access_size] )
    {}

    void read_write(const uint32_t pos, T * vals)
    {
        write(pos, vals);
        read(pos);
#if defined(DEBUG)
        for(int i = 0; i < access_size; ++i) 
	    assert( vals[i] == read_vals[i] );
#endif
    }

    void write(const uint32_t pos, T * vals)
    {
        auto dest = data.section( amp::index<1>(pos), amp::extent<1>(access_size) );
        auto fut = amp::copy_async(vals, vals + access_size, dest);
        fut.get();
    }

    void read(const uint32_t pos)
    {
        auto src = data.section( amp::index<1>(pos), amp::extent<1>(access_size) );
        auto fut = amp::copy_async(src, read_vals);
        fut.get();
    }

private:
    const int32_t access_size;
    T * read_vals;
    amp::array<T> & data;
};

template<typename data_type>
struct view_accessor
{

    view_accessor(amp::array<data_type> & data, const uint32_t access_size) : 
	data(data), data_view(data), access_size(access_size),
	read_vals(new data_type[access_size])
    {}

    ~view_accessor()
    {
	delete[] read_vals;
    }

    void read_write(const uint32_t pos, data_type * vals)
    {
        write(pos, vals);
        read(pos);
#if defined(DEBUG)
        for(int i = 0; i < access_size; ++i) 
	    assert( vals[i] == read_vals[i] );
#endif
    }

    void write(const uint32_t pos, data_type * vals)
    {
        for(int i = 0; i < access_size; ++i) 
	    data_view[pos + i] = vals[i];
        data_view.synchronize();
    }

    void read(const uint32_t pos)
    {
        for(int i = 0; i < access_size; ++i)
	    read_vals[i] = data_view[pos + i];
    }

private:
    const int32_t access_size;
    data_type * read_vals;
    amp::array<data_type> & data;
    amp::array_view<data_type> data_view;
};

template<typename data_type, typename accessor_type>
void measure_time(const int32_t data_size, const int32_t access_size, const uint32_t seed, accessor_type & accessor)
{
    std::srand(seed);
    data_type * vals = new data_type[access_size];
    for(int i = 0; i < access_size; ++i) {
	vals[i] = std::rand();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < data_size; ++i) {
        int pos = std::rand() % (data_size - access_size);
        accessor.read_write(pos, vals);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;
    delete[] vals;
}

int main(int argc, char ** argv)
{
    assert(argc > 2);
    const int32_t size = atoi(argv[1]);
    const int32_t access_size = atoi(argv[2]);
    const uint32_t seed = (unsigned int)std::time(0);
	amp::accelerator default_acc;
	amp::accelerator_view acc_view = default_acc.get_default_view();
	std::wcout << "Using device: " << default_acc.get_description() << std::endl;
    std::cout << "Using seed: " << seed << std::endl;
    
    amp::array<int> device_data(amp::extent<1>(size), acc_view);
    amp::parallel_for_each(device_data.get_extent(),
        [=, &device_data](amp::index<1> idx) restrict(amp) {
            device_data[ idx[0] ] = 0;
        }
    );
    acc_view.wait();
    
    {
        view_accessor<int> acc(device_data, access_size);
        measure_time<int>(size, access_size, seed, acc);
    }

    {
        copy_accessor<int> acc(device_data, access_size);
        measure_time<int>(size, access_size, seed, acc);
    }

    return 0;
}
