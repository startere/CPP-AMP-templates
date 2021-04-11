#include <iostream>

#include <amp.h>

namespace amp = Concurrency;


int main(int argc, char ** argv)
{
	const int size = 50, buffer = 1;
	amp::accelerator default_acc;
	amp::accelerator_view acc_view = default_acc.get_default_view();
	amp::array<int> device_data(amp::extent<1>(size*2 + 2*buffer), acc_view);
	std::wcout << "Using device: " << default_acc.get_description() << std::endl;

	int * host_data = new int[size*2 + 2*buffer];
	int n = 0;
	std::generate(host_data + buffer, host_data + size + buffer, [&]() { return n++; });

	// Copy from host to device
	auto data_view = device_data.section(amp::index<1>(buffer), amp::extent<1>(size));
	amp::copy(host_data + buffer, host_data + size + buffer, data_view);

	// Copy on device between
	amp::parallel_for_each(amp::extent<1>(size), 
		[=, &device_data](amp::index<1> idx) restrict(amp)
		{
			device_data[idx[0] + size + 2*buffer] = device_data[idx[0] + buffer];
		}
	);
	acc_view.wait();

	// Copy from device to host
	data_view = device_data.section(amp::index<1>(size + buffer*2), amp::extent<1>(size));
	amp::copy(data_view, host_data + size + buffer*2);

	// Check correctness
	for(int i = 0; i < size; ++i) {
		assert(host_data[i + buffer] == host_data[i + size + 2*buffer]);
	}

    delete[] host_data;

	return 0;
}
