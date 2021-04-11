#include <iostream>

#include <hc.hpp>

namespace amp = hc;

static const amp::array<int> null_array(amp::extent<1>(1));

struct param
{
    int some_value;
    float other_val;
};

struct specific_data
{
    [[hc]] specific_data() restrict(amp, cpu)
    {
        x_ = 2;
    }

    [[hc]] specific_data(int val, const std::string &) restrict(amp, cpu)
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
struct wrapper_second
{
    amp::array<T> & buffer;
    T & operator[](uint64_t pos_)
    {
        return buffer[pos_];
    }
};

template<typename T>
struct wrapper
{
    wrapper_second<T> * buffer;
    //uint64_t pos;
    T & operator[](uint64_t pos_)
    {
        return (*buffer)[pos_];
    }
};

template<typename T>
struct wrapper_val
{
    amp::array<T> buffer;
//    uint64_t pos;
    //template<typename U>
    //wrapper_val(const wrapper_val<U> & obj) : buffer(obj.buffer) {}
    //wrapper_val(amp::array<T> & buffer) : buffer(buffer){}
    //wrapper_val() : buffer(null_array) {}
};

template<typename T>
const T & make_value(const T & t)
{
   return t;
} 

template<typename T>
wrapper_val<T> make_value(const wrapper<T> & t)
{
   return wrapper_val<T>(*t.buffer);
} 

    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args &... args)
    {
        //launch_(acc_view, threads, local_threads, std::forward<F>(f), args...);//make_value(args)...);
    }

/*
 * Pass an array_view stored in struct B,
 * kept in struct A as a reference.
 *
 * Doesn't work, dmesg shows GPU memory faults.
 */
namespace pass_as_view_wrapper_ref
{
    template<typename T>
    struct array_wrapper
    {
        // Object to capture on GPU
        amp::array_view<T> ptr;
        
        array_wrapper(amp::array<T> * ptr) : ptr(*ptr) {}

        T & operator[](uint64_t pos_)
        {
            return ptr[pos_];
        }
    };
    
    template<typename T>
    struct wrapper
    {
        array_wrapper<T> & ptr;
        
        wrapper(array_wrapper<T> * ptr) : 
            ptr(*ptr) {}

        T & operator[](uint64_t pos_) const
        {
            return ptr[pos_];
        }
    };
    
    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
            [=](amp::tiled_index<1> idx) [[hc]] {
                f(idx.global[0], args...);
            }
        );
    }
    
    template<typename T>
    void destruct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer)
    {
        pass_as_view_wrapper_ref::launch(acc_view, threads,5, 
            [](const int idx, wrapper<T> p) [[hc]] {
                p[ idx ].~T();
            }, pointer);
    }

    template<typename T, typename... Args>
    void construct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer, const Args &... args)
    {
        pass_as_view_wrapper_ref::launch(acc_view, threads, 1,
            [](const int idx, wrapper<T> p, Args const &... args) [[hc]] {
                new (&p[ idx ]) T(args...);
                p[ idx ].x()++;
            }, pointer, args...);
    }
}

/*
 * Pass INDIRECTLY an array_view stored in struct B, 
 * kept in struct A as a pointer. Process variadic args
 * before capturing in kernel - dereference struct B and
 * copy array_view to launch function.
 *
 * In the gpu code recreate wrapper - works like a pointer
 * and we want to save the position for future usage.
 *
 * Works!
 */
namespace pass_as_view_wrapper_ptr_dereference
{
    template<typename T>
    struct array_wrapper
    {
        amp::array_view<T> ptr;
        
        array_wrapper(amp::array<T> * ptr) : ptr(*ptr) {}

        T & operator[](uint64_t pos_)
        {
            return ptr[pos_];
        }
    };
    
    template<typename T>
    struct wrapper
    {
        array_wrapper<T> * ptr;
        uint64_t pos; 
        wrapper(array_wrapper<T> * ptr) : 
            ptr(ptr), pos(0) {}
        
        T & operator[](uint64_t pos_) const
        {
            return (*ptr)[pos_];
        }
    };
    
    template<typename T>
    const T & dereference_view(const T & t)
    {
       return t;
    } 

    template<typename T>
    std::tuple<amp::array_view<T>, uint64_t>  dereference_view(const wrapper<T> & t)
    {
       return std::make_tuple(t.ptr->ptr, t.pos);
    } 
    
    template<typename T>
    const T & reference_view(const T & t) [[hc]]
    {
       return t;
    } 

    template<typename T>
    const amp::array_view<T> & reference_view(const std::tuple<amp::array_view<T>, uint64_t> & t) [[hc]]
    {
        return std::get<0>(t);
    } 
    
    template<typename F, typename... Args>
    void launch_(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
            [=](amp::tiled_index<1> idx) [[hc]] {
                f(idx.global[0], reference_view(args)...);
            }
        );
    }
    
    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        launch_(acc_view, threads, local_threads, std::forward<F>(f), dereference_view(args)...); 
    }
    
    template<typename T>
    void destruct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer)
    {
        pass_as_view_wrapper_ptr_dereference::launch(acc_view, threads,5, 
            [](const int idx, const hc::array_view<T> & p) [[hc]] {
                p[ idx ].~T();
            }, pointer);
    }

    template<typename T, typename... Args>
    void construct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer, const Args &... args)
    {
        pass_as_view_wrapper_ptr_dereference::launch(acc_view, threads, 1,
            [](const int idx, const hc::array_view<T> & p, Args const &... args) [[hc]] {
                new (&p[ idx ]) T(args...);
                p[ idx ].x()++;
            }, pointer, args...);
    }
}

/*
 * Pass an array_view stored in struct B, 
 * kept in struct A as a pointer.
 *
 * Compiler complains about ambigous call to wrapper in kernel,
 * can't decide between the declared one, implicit copy and implicit move.
 */
namespace pass_as_view_wrapper_ptr
{
    template<typename T>
    struct array_wrapper
    {
        amp::array_view<T> ptr;
        
        array_wrapper(amp::array<T> * ptr) : ptr(*ptr) {}

        T & operator[](uint64_t pos_)
        {
            return ptr[pos_];
        }
    };
    
    template<typename T>
    struct wrapper
    {
        array_wrapper<T> * ptr;
        
        wrapper(array_wrapper<T> * ptr) : 
            ptr(ptr) {}
        
        T & operator[](uint64_t pos_)
        {
            return (*ptr)[pos_];
        }
    };

    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
            [=](amp::tiled_index<1> idx) [[hc]] {
                f(idx.global[0], args...);
            }
        );
    }
    
    template<typename T>
    void destruct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer)
    {
        pass_as_view_wrapper_ptr::launch(acc_view, threads,5, 
            [](const int idx, wrapper<T> p) [[hc]] {
                p[ idx ].~T();
            }, pointer);
    }

    template<typename T, typename... Args>
    void construct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer, const Args &... args)
    {
        pass_as_view_wrapper_ptr::launch(acc_view, threads, 1,
            [](const int idx, wrapper<T> p, Args const &... args) [[hc]] {
                new (&p[ idx ]) T(args...);
                p[ idx ].x()++;
            }, pointer, args...);
    }
}

namespace pass_as_view_wrapper_val
{ 
    template<typename T>
    struct array_wrapper
    {
        amp::array_view<T> ptr;
        
        array_wrapper(amp::array<T> * ptr) : ptr(*ptr) {}

        T & operator[](uint64_t pos_)
        {
            return ptr[pos_];
        }
    };

    template<typename T>
    struct wrapper
    {
        array_wrapper<T> ptr;
        
        wrapper(array_wrapper<T> * ptr) : ptr(*ptr) {}

        T & operator[](uint64_t pos_)
        {
            return ptr[pos_];
        }
    };
    
    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
            [=](amp::tiled_index<1> idx) [[hc]] {
                f(idx.global[0], args...);
            }
        );
    }
    
    template<typename T>
    void destruct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer)
    {
        pass_as_view_wrapper_val::launch(acc_view, threads,5, 
            [](const int idx, wrapper<T> p) [[hc]] {
                p[ idx ].~T();
            }, pointer);
    }

    template<typename T, typename... Args>
    void construct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer, const Args &... args)
    {
        pass_as_view_wrapper_val::launch(acc_view, threads, 1,
            [](const int idx, wrapper<T> p, Args const &... args) [[hc]] {
                new (&p[ idx ]) T(args...);
                p[ idx ].x()++;
            }, pointer, args...);
    }
}

namespace pass_as_pointer
{    
    template<typename T>
    struct array_wrapper
    {
        amp::array<T> * ptr;

        T & operator[](uint64_t pos_)
        {
            return (*ptr)[pos_];
        }
    };
    
    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
            [=](amp::tiled_index<1> idx) [[hc]] {
                f(idx.global[0], args...);
            }
        );
    }
    //template<typename F, typename... Args>
    //void launch(amp::accelerator_view & acc_view, int threads, F && f, const Args &... args)
    //{
    //    amp::parallel_for_each(acc_view, amp::extent<1>(threads),
    //        [=](amp::index<1> idx) [[hc]] {
    //            f(idx[0], args...);
    //        }
    //    );
    //}

    template<typename T>
    void destruct(amp::accelerator_view & acc_view, int threads, array_wrapper<T> pointer)
    {
        pass_as_pointer::launch(acc_view, threads,5, 
            [](const int idx, array_wrapper<T> p) [[hc]] {
                p[ idx ].~T();
            }, pointer);
    }

    template<typename T, typename... Args>
    void construct(amp::accelerator_view & acc_view, int threads, array_wrapper<T> pointer, const Args &... args)
    {
        pass_as_pointer::launch(acc_view, threads, 1,
            [](const int idx, array_wrapper<T> p, Args const &... args) [[hc]] {
                new (&p[ idx ]) T(args...);
                p[ idx ].x()++;
            }, pointer, args...);
    }
}

namespace two_wrappers
{

    template<typename F, typename... Args>
    void launch(amp::accelerator_view & acc_view, int threads, int local_threads, F && f, Args const &... args)
    {
        amp::parallel_for_each(acc_view, amp::extent<1>(threads).tile(local_threads),
            [=](amp::tiled_index<1> idx) [[hc]] {
                f(idx.global[0], args...);
            }
        );
    }
    //template<typename F, typename... Args>
    //void launch(amp::accelerator_view & acc_view, int threads, F && f, const Args &... args)
    //{
    //    amp::parallel_for_each(acc_view, amp::extent<1>(threads),
    //        [=](amp::index<1> idx) [[hc]] {
    //            f(idx[0], args...);
    //        }
    //    );
    //}

    template<typename T>
    void destruct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer)
    {
        two_wrappers::launch(acc_view, threads,5, 
            [](const int idx, wrapper<T> p) [[hc]] {
                p[ idx ].~T();
            }, pointer);
    }

    template<typename T, typename... Args>
    void construct(amp::accelerator_view & acc_view, int threads, wrapper<T> pointer, const Args &... args)
    {
        two_wrappers::launch(acc_view, threads, 1,
            [](const int idx, wrapper<T> p, Args const &... args) [[hc]] {
                new (&p[ idx ]) T(args...);
                p[ idx ].x()++;
            }, pointer, args...);
    }
}

int main(int argc, char ** argv)
{
	const int size = 50;
	amp::accelerator default_acc;
	amp::accelerator_view acc_view = default_acc.get_default_view();
	amp::array<specific_data> * device_data = 
        new amp::array<specific_data>(amp::extent<1>(size), acc_view);
	std::wcout << "Using device: " << default_acc.get_description() << std::endl;

    int val = 3.0;
    std::string s = "abc";
    param r{1, 3.0};

    {   
        wrapper_second<specific_data> data{*device_data};
        wrapper<specific_data> p{&data};
        /**
         * Two wrappers - works with memory errors
         * Doesn't work due to GPU memory errors.
         */
        //two_wrappers::construct(acc_view, size, p, val, r);
        //two_wrappers::construct(acc_view, size, p);
    }
   
    //{
    //    pass_as_pointer::array_wrapper<specific_data> p{device_data};
    //    pass_as_pointer::construct(acc_view, size, p);
    //} 
    
    //{
    //    pass_as_view_wrapper_ref::wrapper<specific_data> p(device_data);
    //    pass_as_view_wrapper_ref::construct(acc_view, size, p, val, r);

    //    acc_view.wait();
   
    //    amp::array_view<specific_data> data_view(*device_data);
    //    for(int i = 0; i < size; ++i)
    //        assert(data_view[i].x() == val + r.some_value + 1); 
    //        
    //    pass_as_view_wrapper_ref::destruct(acc_view,size, p);
    //    acc_view.wait(); 
    //} 
    namespace test_implementation = pass_as_view_wrapper_ptr_dereference;
    {
        test_implementation::array_wrapper<specific_data> wrapper(device_data); 
        test_implementation::wrapper<specific_data> p(&wrapper);
        test_implementation::construct(acc_view, size, p, val, r);

        acc_view.wait();
   
        amp::array_view<specific_data> data_view(*device_data);
        for(int i = 0; i < size; ++i)
            assert(data_view[i].x() == val + r.some_value + 1); 
            
        test_implementation::destruct(acc_view,size, p);
        acc_view.wait(); 
    } 

    delete device_data;
    return 0;
}
