//#if defined(DACE_USE_GPU_ATOMICS)
// Implementation of Complex64 summation reduction .
// Special case for Complex64
template <>
struct _wcr_fixed<ReductionType::Sum, dace::complex64>
{
    // static DACE_HDFI void reduce(dace::complex64 *ptr, const dace::complex64 &value)
    // {
    //     float *real_ptr = reinterpret_cast<float *>(ptr);
    //     float *imag_ptr = real_ptr + 1;

    //     _wcr_fixed<ReductionType::Sum, float>::reduce(real_ptr, value.real());
    //     _wcr_fixed<ReductionType::Sum, float>::reduce(imag_ptr, value.imag());
    // }

    static DACE_HDFI void reduce_atomic(dace::complex64 *ptr, const dace::complex64 &value)
    {
        float *real_ptr = reinterpret_cast<float *>(ptr);
        float *imag_ptr = real_ptr + 1;

        _wcr_fixed<ReductionType::Sum, float>::reduce_atomic(real_ptr, value.real());
        _wcr_fixed<ReductionType::Sum, float>::reduce_atomic(imag_ptr, value.imag());
    }

    DACE_HDFI dace::complex64 operator()(const dace::complex64 &a, const dace::complex64 &b) const
    {
        return _wcr_fixed<ReductionType::Sum, dace::complex64>()(a, b);
    }
};

// Enables template based on if T is a complex number or not .
template <typename T>
using EnableIfcomplex64 = typename std::enable_if<std::is_same<T, dace::complex64>::value>::type;

// When atomics are supported , use _wcr_fixed normally
template <ReductionType REDTYPE, typename T>
struct wcr_fixed<REDTYPE, T, EnableIfcomplex64<T>>
{
    static DACE_HDFI void reduce(T *ptr, const T &value)
    {
        _wcr_fixed<REDTYPE, T>::reduce(ptr, value);
    }

    static DACE_HDFI void reduce_atomic(T *ptr, const T &value)
    {
        _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, value);
    }

    DACE_HDFI T operator()(const T &a, const T &b) const
    {
        return _wcr_fixed<REDTYPE, T>()(a, b);
    }
};

// Implementation of Complex128 summation reduction .
// Special case for Complex128
template <>
struct _wcr_fixed<ReductionType::Sum, dace::complex128>
{
    // static DACE_HDFI void reduce(dace::complex128 *ptr, const dace::complex128 &value)
    // {
    //     double *real_ptr = reinterpret_cast<double *>(ptr);
    //     double *imag_ptr = real_ptr + 1;

    //     _wcr_fixed<ReductionType::Sum, double>::reduce(real_ptr, value.real());
    //     _wcr_fixed<ReductionType::Sum, double>::reduce(imag_ptr, value.imag());
    // }

    static DACE_HDFI void reduce_atomic(dace::complex128 *ptr, const dace::complex128 &value)
    {
        double *real_ptr = reinterpret_cast<double *>(ptr);
        double *imag_ptr = real_ptr + 1;

        _wcr_fixed<ReductionType::Sum, double>::reduce_atomic(real_ptr, value.real());
        _wcr_fixed<ReductionType::Sum, double>::reduce_atomic(imag_ptr, value.imag());
    }

    DACE_HDFI dace::complex128 operator()(const dace::complex128 &a, const dace::complex128 &b) const
    {
        return _wcr_fixed<ReductionType::Sum, dace::complex128>()(a, b);
    }
};

// Enables template based on if T is a complex number or not .
template <typename T>
using EnableIfComplex128 = typename std::enable_if<std::is_same<T, dace::complex128>::value>::type;

// When atomics are supported , use _wcr_fixed normally
template <ReductionType REDTYPE, typename T>
struct wcr_fixed<REDTYPE, T, EnableIfComplex128<T>>
{
    static DACE_HDFI void reduce(T *ptr, const T &value)
    {
        _wcr_fixed<REDTYPE, T>::reduce(ptr, value);
    }

    static DACE_HDFI void reduce_atomic(T *ptr, const T &value)
    {
        _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, value);
    }

    DACE_HDFI T operator()(const T &a, const T &b) const
    {
        return _wcr_fixed<REDTYPE, T>()(a, b);
    }
};
//#endif