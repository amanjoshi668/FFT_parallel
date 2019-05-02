#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

typedef complex<float> base;
typedef float2 Complex;

template <typename T>
ostream &operator<<(ostream &o, vector<T> v)
{
    if (v.size() > 0)
        o << v[0];
    for (unsigned i = 1; i < v.size(); i++)
        o << " " << v[i];
    return o << endl;
}
static __device__ __host__ inline Complex Add(Complex A, Complex B)
{
    Complex C;
    C.x = A.x + B.x;
    C.y = A.y + B.y;
    return C;
}

/**
 *  Inverse of Complex Number
 */
static __device__ __host__ inline Complex Inverse(Complex A)
{
    Complex C;
    C.x = -A.x;
    C.y = -A.y;
    return C;
}

/**
 *  Multipication of Complex Numbers
 */
static __device__ __host__ inline Complex Multiply(Complex A, Complex B)
{
    Complex C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.y * B.x + A.x * B.y;
    return C;
}

/**
* Parallel Functions for performing various tasks
*/

/**
*  Dividing by constant for inverse fft transform
*/
__global__ void inplace_divide_invert(Complex *A, int n, int threads)
{
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n)
    {
        // printf("in divide");
        A[i].x /= n;
        A[i].y /= n;
    }
    else
    {
        // printf("else in divide");
        // printf("i=%d, n=%d", i, n);
    }
}

/**
* Reorders array by bit-reversing the indexes.
*/
__global__ void bitrev_reorder(Complex *__restrict__ r, Complex *__restrict__ d, int s, size_t nthr, int n)
{
    int id = blockIdx.x * nthr + threadIdx.x;
    //r[id].x = -1;
    if (id < n and __brev(id) >> (32 - s) < n)
        r[__brev(id) >> (32 - s)] = d[id];
}

/**
* Inner part of the for loop
*/
__device__ void inplace_fft_inner(Complex *__restrict__ A, int i, int j, int len, int n, bool invert)
{
    if (i + j + len / 2 < n and j < len / 2)
    {
        Complex u, v;

        float angle = (2 * M_PI * j) / (len * (invert ? -1.0 : 1.0));
        v.x = cos(angle);
        v.y = sin(angle);

        u = A[i + j];
        v = Multiply(A[i + j + len / 2], v);
        // printf("i:%d j:%d u_x:%f u_y:%f    v_x:%f v_y:%f\n", i, j, u.x, u.y, v.x, v.y);
        A[i + j] = Add(u, v);
        A[i + j + len / 2] = Add(u, Inverse(v));
    }
}

/**
* FFT if number of threads are sufficient.
*/
__global__ void inplace_fft(Complex *__restrict__ A, int i, int len, int n, int threads, bool invert)
{
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A, i, j, len, n, invert);
}

/**
* FFt if number of threads are not sufficient.
*/
__global__ void inplace_fft_outer(Complex *__restrict__ A, int len, int n, int threads, bool invert)
{
    int i = (blockIdx.x * threads + threadIdx.x);
    for (int j = 0; j < len / 2; j++)
    {
        inplace_fft_inner(A, i, j, len, n, invert);
    }
}

/**
* parallel FFT transform and inverse transform
* Arguments vector of complex numbers, invert, balance, number of threads
* Perform inplace transform
*/
void fft(vector<base> &a, bool invert, int balance = 10, int threads = 32)
{
    // Creating array from vector
    int n = (int)a.size();
    int data_size = n * sizeof(Complex);
    Complex *data_array = (Complex *)malloc(data_size);
    for (int i = 0; i < n; i++)
    {
        data_array[i].x = a[i].real();
        data_array[i].y = a[i].imag();
    }
    
    // Copying data to GPU
    Complex *A, *dn;
    cudaMalloc((void **)&A, data_size);
    cudaMalloc((void **)&dn, data_size);
    cudaMemcpy(dn, data_array, data_size, cudaMemcpyHostToDevice);
    // Bit reversal reordering
    int s = log2(n);

    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);

    
    // Synchronize
    cudaDeviceSynchronize();
    // Iterative FFT with loop parallelism balancing
    for (int len = 2; len <= n; len <<= 1)
    {
        if (n / len > balance)
        {

            inplace_fft_outer<<<ceil((float)n / threads), threads>>>(A, len, n, threads, invert);
        }
        else
        {
            for (int i = 0; i < n; i += len)
            {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A, i, len, n, threads, invert);
            }
        }
    }
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    // Copy data from GPU
    Complex *result;
    result = (Complex *)malloc(data_size);
    cudaMemcpy(result, A, data_size, cudaMemcpyDeviceToHost);
    
    // Saving data to vector<complex> in input.
    for (int i = 0; i < n; i++)
    {
        a[i] = base(result[i].x, result[i].y);
    }
    // Free the memory blocks
    free(data_array);
    cudaFree(A);
    cudaFree(dn);
    return;
}

/**
* Performs 2D FFT 
* takes vector of complex vectors, invert and verbose as argument
* performs inplace FFT transform on input vector
*/
void fft2D(vector<vector<base>> &a, bool invert, int verbose = 0)
{
    auto matrix = a;
    // Transform the rows
    if (verbose > 0)
        cout << "Transforming Rows" << endl;

    for (auto i = 0; i < matrix.size(); i++)
    {
        //cout<<i<<endl;
        fft(matrix[i], invert);
    }

    // preparing for transforming columns

    if (verbose > 0)
        cout << "Converting Rows to Columns" << endl;

    a = matrix;
    matrix.resize(a[0].siz*e());
    for (int i = 0; i < matrix.size(); i++)
        matrix[i].resize(a.size());

    // Transposing matrix
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            matrix[j][i] = a[i][j];
        }
    }
    if (verbose > 0)
        cout << "Transforming Columns" << endl;

    // Transform the columns
    for (auto i = 0; i < matrix.size(); i++)
        fft(matrix[i], invert);

    if (verbose > 0)
        cout << "Storing the result" << endl;

    // Storing the result after transposing
    // [j][i] is getting value of [i][j]
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            a[j][i] = matrix[i][j];
        }
    }
}

/**
* Function to multiply two polynomial
* takes two polynomials represented as vectors as input
* return the product of two vectors
*/
vector<int> mult(vector<int> a, vector<int> b)
{
    // Creating complex vector from input vectors
    vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());

    // Padding with zero to make their size equal to power of 2
    size_t n = 1;
    while (n < max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;

    fa.resize(n), fb.resize(n);

    // Transforming both a and b
    // Converting to points form
    fft(fa, false), fft(fb, false);
    cout << fa << endl;
    cout << endl;
    cout << fb << endl;
    cout << endl;
    // performing point wise multipication of points
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];

    // Performing Inverse transform
    fft(fa, true);

    // Saving the real part as it will be the result
    vector<int> res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i)
        res[i] = int(fa[i].real() + 0.5);

    return res;
}
int main()
{
    vector<int> a; //= {1, 1}; //{3,4,-5,2};
    vector<int> b; // = {2, 1}; //{2,1,1,-9};
    for (int i = 0; i < 4; i++)
    {
        a.push_back(i);

        b.push_back(i);
    }

    vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    fft(fa, false);
    cout << "###################################" << endl;
    cout << fa << endl;
    cout << endl;
    cout << endl;
    fft(fa, true);
    cout << endl;

    for (int i = 0; i < 4; i++)
        b[i] = fa[i].real();
    if (b == a)
    {
        cout << "Yes" << endl;
    }
    cout << b << endl;
    // auto fft = FFT();
    //cout << mult(a, b);
    return 0;
}
