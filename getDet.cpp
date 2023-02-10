#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
using namespace std;

template<typename T> static void
mixChannels_( const T** src, const int* sdelta,
              T** dst, const int* ddelta,
              int len, int npairs )
{
    int i, k;
    for( k = 0; k < npairs; k++ )
    {
        const T* s = src[k];
        T* d = dst[k];
        int ds = sdelta[k], dd = ddelta[k];
        if( s )
        {
            for( i = 0; i <= len - 2; i += 2, s += ds*2, d += dd*2 )
            {
                T t0 = s[0], t1 = s[ds];
                d[0] = t0; d[dd] = t1;
            }
            if( i < len )
                d[0] = s[0];
        }
        else
        {
            for( i = 0; i <= len - 2; i += 2, d += dd*2 )
                d[0] = d[dd] = 0;
            if( i < len )
                d[0] = 0;
        }
    }
}

template<typename _Tp> static inline int
LUImpl(_Tp* A, size_t astep, int m, _Tp* b, size_t bstep, int n, _Tp eps)
{
    int i, j, k, p = 1;
    astep /= sizeof(A[0]);
    bstep /= sizeof(b[0]);

    for( i = 0; i < m; i++ )
    {
        k = i;

        for( j = i+1; j < m; j++ )
            if( std::abs(A[j*astep + i]) > std::abs(A[k*astep + i]) )
                k = j;

        if( std::abs(A[k*astep + i]) < eps )
            return 0;

        if( k != i )
        {
            for( j = i; j < m; j++ )
                std::swap(A[i*astep + j], A[k*astep + j]);
            if( b )
                for( j = 0; j < n; j++ )
                    std::swap(b[i*bstep + j], b[k*bstep + j]);
            p = -p;
        }

        _Tp d = -1/A[i*astep + i];

        for( j = i+1; j < m; j++ )
        {
            _Tp alpha = A[j*astep + i]*d;

            for( k = i+1; k < m; k++ )
                A[j*astep + k] += alpha*A[i*astep + k];

            if( b )
                for( k = 0; k < n; k++ )
                    b[j*bstep + k] += alpha*b[i*bstep + k];
        }
    }

    if( b )
    {
        for( i = m-1; i >= 0; i-- )
            for( j = 0; j < n; j++ )
            {
                _Tp s = b[i*bstep + j];
                for( k = i+1; k < m; k++ )
                    s -= A[i*astep + k]*b[k*bstep + j];
                b[i*bstep + j] = s/A[i*astep + i];
            }
    }

    return p;
}

#define det2(m)   ((double)m(0,0)*m(1,1) - (double)m(0,1)*m(1,0))
#define det3(m)   (m(0,0)*((double)m(1,1)*m(2,2) - (double)m(1,2)*m(2,1)) -  \
                   m(0,1)*((double)m(1,0)*m(2,2) - (double)m(1,2)*m(2,0)) +  \
                   m(0,2)*((double)m(1,0)*m(2,1) - (double)m(1,1)*m(2,0)))

double myDeterminant( InputArray _mat )
{
    Mat mat = _mat.getMat();
    double result = 0;
    int type = mat.type(), rows = mat.rows;
    size_t step = mat.step;
    cout<<"sst: "<<step<<" "<<mat.step1()<<endl;
    const uchar* m = mat.ptr();

    CV_Assert( !mat.empty() );
    CV_Assert( mat.rows == mat.cols && (type == CV_32F || type == CV_64F));

#define Mf(y, x) ((float*)(m + y*step))[x]
#define Md(y, x) ((double*)(m + y*step))[x]

    if( type == CV_32F )
    {
        if( rows == 2 )
            result = det2(Mf);
        else if( rows == 3 )
            result = det3(Mf);
        else if( rows == 1 )
            result = Mf(0,0);
        else
        {
            size_t bufSize = rows*rows*sizeof(float);
            AutoBuffer<uchar> buffer(bufSize);
            Mat a(rows, rows, CV_32F, buffer.data());
            mat.copyTo(a);

            result = LUImpl<float>(a.ptr<float>(), a.step, rows, 0, 0, 0, FLT_EPSILON*10);

            if( result )
            {
                for( int i = 0; i < rows; i++ )
                    result *= a.at<float>(i,i);
            }
        }
    }
    else
    {
        if( rows == 2 )
            result = det2(Md);
        else if( rows == 3 )
            result = det3(Md);
        else if( rows == 1 )
            result = Md(0,0);
        else
        {
            size_t bufSize = rows*rows*sizeof(double);
            AutoBuffer<uchar> buffer(bufSize);
            Mat a(rows, rows, CV_64F, buffer.data());
            mat.copyTo(a);

//            result = hal::LU64f(a.ptr<double>(), a.step, rows, 0, 0, 0);
            if( result )
            {
                for( int i = 0; i < rows; i++ )
                    result *= a.at<double>(i,i);
            }
        }
    }

#undef Mf
#undef Md
    return result;
}

int mainDet(){
     Mat det(4,4,CV_32FC1);
     randu(det,0,1);
     cout<<"det: "<<det<<endl;
     auto dt = myDeterminant(det);
     cout<<dt<<endl;
}

int main() {
  
/**
  ******************************************************************************
  * @author         : oswin
  * @brief          : input an image with three channels, eg.:Scalar(7,11,13),
  * fromto = {0,1,1,0,2,2} denotes the sequences of channels,
  * return an image with three channels, Scalar(11,7,13).
  ******************************************************************************
  */  
  
    int nsrcs = 1, ndsts = 1, npairs =3;
    int i,j,k;
    int fromTo[] = {1,0,0,1,2,2};
    Mat oriI = Mat(5,5,CV_8UC3,Scalar(7,11,13));
    Mat dstI(oriI.size(),CV_8UC3);//todo output img: 11,7,13
    Mat src[] = {oriI};
    Mat dst[] = {dstI};
    int depth(oriI.depth()),esz1(1);
    int BLOCK_SIZE = 1024;

    AutoBuffer<uchar> buf((nsrcs + ndsts + 1)*(sizeof(Mat*) + sizeof(uchar*)) + npairs*(sizeof(uchar*)*2 + sizeof(int)*6));
    const Mat** arrays = (const Mat**)(uchar*)buf.data();
    uchar** ptrs = (uchar**)(arrays + nsrcs + ndsts);
    const uchar** srcs = (const uchar**)(ptrs + nsrcs + ndsts + 1);
    uchar** dsts = (uchar**)(srcs + npairs);
    int* tab = (int*)(dsts + npairs);
    int *sdelta = (int*)(tab + npairs*4), *ddelta = sdelta + npairs;

    for( i = 0; i < nsrcs; i++)
        arrays[i] = &src[i];
    for( i = 0; i < ndsts; i++ )
        arrays[i + nsrcs] = &dst[i];
    ptrs[nsrcs + ndsts] = 0;

    for( i = 0; i < npairs; i++ )
    {
        int i0 = fromTo[i*2], i1 = fromTo[i*2+1];
        if( i0 >= 0 )
        {
            for( j = 0; j < nsrcs; i0 -= src[j].channels(), j++ )
                if( i0 < src[j].channels() )
                    break;
            CV_Assert(j < nsrcs && src[j].depth() == depth);
            tab[i*4] = (int)j; tab[i*4+1] = (int)(i0*esz1);
            sdelta[i] = src[j].channels();

            cout<<"tab: "<<tab[i*4]<<" "<<tab[i*4+1]<<endl;
            cout<< "sdelta[i]: "<<sdelta[i]<<endl;
        }
        else
        {
            tab[i*4] = (int)(nsrcs + ndsts); tab[i*4+1] = 0;
            sdelta[i] = 0;
        }

        for( j = 0; j < ndsts; i1 -= dst[j].channels(), j++ )
            if( i1 < dst[j].channels() )
                break;
        CV_Assert(i1 >= 0 && j < ndsts && dst[j].depth() == depth);
        tab[i*4+2] = (int)(j + nsrcs); tab[i*4+3] = (int)(i1*esz1);
        ddelta[i] = dst[j].channels();

        cout<<"tab[i*4+2]: "<<tab[i*4+2]<<endl;
        cout<< "ddelta[i]: "<<ddelta[i]<<endl;
    }


    NAryMatIterator it(arrays, ptrs, (int)(nsrcs + ndsts));


    int total = (int)it.size, blocksize = std::min(total, (int)((BLOCK_SIZE + esz1-1)/esz1));
    cout<<"total: "<<total<<endl;

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( k = 0; k < npairs; k++ )
        {
            srcs[k] = ptrs[tab[k*4]] + tab[k*4+1];
            dsts[k] = ptrs[tab[k*4+2]] + tab[k*4+3];
        }

        for( int t = 0; t < total; t += blocksize )
        {
            int bsz = std::min(total - t, blocksize);
            mixChannels_( srcs, sdelta, dsts, ddelta, bsz, (int)npairs );

            if( t + blocksize < total )
                for( k = 0; k < npairs; k++ )
                {
                    srcs[k] += blocksize*sdelta[k]*esz1;
                    dsts[k] += blocksize*ddelta[k]*esz1;
                }

            cout<<"k "<<blocksize*sdelta[k]*esz1<<" "<<blocksize*ddelta[k]*esz1<<endl;
        }
    }

    cout<<"dst: \n"<<dst[0]<<endl;
}
