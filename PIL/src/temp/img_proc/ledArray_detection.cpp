#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <valarray>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>

#include <rtk_utils.h>
#include <rtk_debug.h>
#include <rtk_cv.h>
#include <rtk_math.h>

#include "ledArray_detection.h"


using namespace std;
using namespace cv;
using namespace Eigen;


namespace rtk {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline void drawApproxCurve ( Mat &in, vector<Point> &contour, Scalar color )
{
    for ( int i=0; i<contour.size(); i++ ) {
        cv::line( in,contour[i], contour[ (i+1)%contour.size() ], color, 2);
    }
}

inline int perimeter ( vector<Point2f> &a )
{
    int sum = 0;

    for ( int i=0; i<a.size(); i++ ) {
        int i2 = (i+1)%a.size();
        sum += sqrt ( ( a[i].x-a[i2].x ) * ( a[i].x-a[i2].x ) +
                      ( a[i].y-a[i2].y ) * ( a[i].y-a[i2].y ) ) ;
    }

    return sum;
}

inline bool warpRect(Mat &in, Mat &out, Size size, vector<Point2f> points)
{
    if ( points.size() != 4 ) {
        dbg_pe("Input rectangle is wrong! nPoints = %d\n", points.size());
        return false;
    }

    //obtain the perspective transform
    Point2f  pointsRes[4], pointsIn[4];

    for ( int i=0;i<4;i++ ) pointsIn[i]=points[i];
    pointsRes[0]= ( Point2f ( 0,0 ) );
    pointsRes[1]= Point2f ( size.width-1,0 );
    pointsRes[2]= Point2f ( size.width-1,size.height-1 );
    pointsRes[3]= Point2f ( 0,size.height-1 );

    Mat M = getPerspectiveTransform ( pointsIn, pointsRes );
    cv::warpPerspective( in, out, M, size, cv::INTER_CUBIC );

    return true;
}

inline int matrix_swapLR(int *m, int s)
{
    int     i, j;
    int     n = s*s;
    int     *t;

    t = new int[n];

    for(j=0; j<s; j++) {
        for(i=0; i<s; i++) {
            t[j*s + i] = m[j*s + s-1-i];
        }
    }

    for(i=0; i<n; i++) m[i] = t[i];

    delete t;

    return 0;
}

inline int matrix_swapUD(int *m, int s)
{
    int     i, j;
    int     n = s*s;
    int     *t;

    t = new int[n];

    for(j=0; j<s; j++) {
        for(i=0; i<s; i++) {
            t[j*s + i] = m[(s-1-j)*s + i];
        }
    }

    for(i=0; i<n; i++) m[i] = t[i];

    delete t;

    return 0;
}

inline int matrix_rotate90(int *m, int s)
{
    int     i, j;
    int     n = s*s;
    int     *t;

    t = new int[n];

    for(j=0; j<s; j++) {
        for(i=0; i<s; i++) {
            t[j*s + i] = m[(s-1-i)*s + j];
        }
    }

    for(i=0; i<n; i++) m[i] = t[i];

    delete t;

    return 0;
}

inline int matrix_rotate180(int *m, int s)
{
    int     i, j;
    int     n = s*s;
    int     *t;

    t = new int[n];

    for(j=0; j<s; j++) {
        for(i=0; i<s; i++) {
            t[j*s + i] = m[(s-1-j)*s + (s-1-i)];
        }
    }

    for(i=0; i<n; i++) m[i] = t[i];

    delete t;

    return 0;
}

inline int matrix_rotate270(int *m, int s)
{
    int     i, j;
    int     n = s*s;
    int     *t;

    t = new int[n];

    for(j=0; j<s; j++) {
        for(i=0; i<s; i++) {
            t[j*s + i] = m[i*s + (s-1-j)];
        }
    }

    for(i=0; i<n; i++) m[i] = t[i];

    delete t;

    return 0;
}

inline float rectMeanValue(Mat &img)
{
    int     w, h, i, j;
    float   sum = 0.0;
    ru8     *p;

    w = img.cols;
    h = img.rows;
    p = img.data;

    for(j=0; j<h; j++) {
        for(i=0; i<w; i++) {
            sum += p[j*w + i];
        }
    }

    return sum / (w*h);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


RLEDArray_Detection::RLEDArray_Detection()
{
    initParam();
}

RLEDArray_Detection::~RLEDArray_Detection()
{

}


void RLEDArray_Detection::initParam(void)
{
    // set default values
    _minSize = 0.04;                // contour detection min size
    _maxSize = 0.5;                 // contour detection max size

    minLeng_t = 10;                 // rectangle edge min length threshold
    edgeLengRatio = 1.5;            // maxEdge/minEdge ratio

    arr_size = 8;                   // array size (block size in each direction)
    roi_size = 2;                   // detection ROI size
    rect_pixel_size = 160;          // rectangle size

    drawRes = 1;
}


int RLEDArray_Detection::detect(cv::Mat &img_in, RLEDArray_Marker_List &arrM)
{
    int correctN = 0;

    // clear output list
    arrM.clear();


    // copy raw image
    img_in.copyTo(m_imgRaw);


    // detect candicate rectanges
    vector<CandidateRect>  arrCR;
    detectRectangs(img_in, arrCR);


    // wrap each of candicate rectangles
    vector<Mat> rectImgs;

    for(int i=0; i<arrCR.size(); i++) {
        // wrap rects
        Mat rectImg;
        warpRect(img_in, rectImg,
                 Size(rect_pixel_size, rect_pixel_size),
                 arrCR[i].contour);
        rectImgs.push_back(rectImg);
    }


    // detect markers
    for(int i=0; i<arrCR.size(); i++) {
        RLEDArray_Marker rla_m(this);

        rla_m.m_cr = arrCR[i];

        if( rla_m.detect(rectImgs[i]) == 0 ) {
            correctN++;
        }

        arrM.push_back(rla_m);
    }

    if( drawRes ) {
        imshow("detectRects", m_imgRes);
    }

    return correctN;
}

int RLEDArray_Detection::detectRectangs(cv::Mat &img_in, vector<CandidateRect> &arrCR)
{
    // clear output list
    arrCR.clear();

    // convert color image to gray-scale image
    cv::cvtColor(img_in, m_imgGray, CV_BGR2GRAY);

    // bin treshold input image
    cv::Mat bin_img;
    cv::adaptiveThreshold( m_imgGray, bin_img, 255,
                           ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,
                           7, 7);

    // find contours
    std::vector<std::vector<cv::Point> > contours2;
    std::vector<cv::Vec4i> hierarchy2;
    vector<Point> approxCurve;

    int minSize=_minSize*std::max(img_in.cols,img_in.rows)*4;
    int maxSize=_maxSize*std::max(img_in.cols,img_in.rows)*4;

    cv::findContours ( bin_img , contours2, hierarchy2,
                       CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

    // detect candicate rectangles
    vector<CandidateRect> MarkerCanditates;

    for ( unsigned int i=0;i<contours2.size();i++ ) {
        //check it is a possible element by first checking is has enough points
        if ( minSize< contours2[i].size() && contours2[i].size()<maxSize  ) {
            //approximate to a poligon
            approxPolyDP( contours2[i], approxCurve,
                          double(contours2[i].size())*0.05 , true );

            //check that the poligon has 4 points & convex
            if ( approxCurve.size() == 4 && isContourConvex(Mat(approxCurve)) ) {

                //ensure that the   distace between consecutive points is large enough
                float minDist=1e10, maxDist=-1e10;

                for ( int j=0;j<4;j++ ) {
                    float d= std::sqrt ( ( float ) ( approxCurve[j].x-approxCurve[ ( j+1 ) %4].x ) * ( approxCurve[j].x-approxCurve[ ( j+1 ) %4].x ) +
                            ( approxCurve[j].y-approxCurve[ ( j+1 ) %4].y ) * ( approxCurve[j].y-approxCurve[ ( j+1 ) %4].y ) );


                    if ( d<minDist ) minDist = d;
                    if ( d>maxDist ) maxDist = d;
                }


                // check that distance is not very small & edge length ratio is proper
                if ( minDist > minLeng_t && maxDist/minDist < edgeLengRatio ) {

                    CandidateRect cr;

                    cr.idx = i;
                    for ( int j=0;j<4;j++ ) {
                        cr.contour.push_back(Point2f ( approxCurve[j].x, approxCurve[j].y ));
                    }

                    // add to candicat rectangle list
                    MarkerCanditates.push_back(cr);
                }
            }
        }
    }

    // sort the points in anti-clockwise order
    valarray<bool> swapped(false, MarkerCanditates.size());
    for( unsigned int i=0; i<MarkerCanditates.size(); i++ ) {

        //trace a line between the first and second point.
        //if the thrid point is at the right side, then the points are anti-clockwise
        double dx1 = MarkerCanditates[i].contour[1].x - MarkerCanditates[i].contour[0].x;
        double dy1 = MarkerCanditates[i].contour[1].y - MarkerCanditates[i].contour[0].y;
        double dx2 = MarkerCanditates[i].contour[2].x - MarkerCanditates[i].contour[0].x;
        double dy2 = MarkerCanditates[i].contour[2].y - MarkerCanditates[i].contour[0].y;
        double o = ( dx1*dy2 )- ( dy1*dx2 );

        //if the third point is in the left side, then sort in anti-clockwise order
        if ( o  < 0.0 ) {
            std::swap ( MarkerCanditates[i].contour[1], MarkerCanditates[i].contour[3] );
            swapped[i]=true;

            //sort the contour points
            //reverse(MarkerCanditates[i].contour.begin(),MarkerCanditates[i].contour.end());
        }
    }

    // remove these elements whise corners are too close to each other
    //      first detect candidates

    vector< pair<int,int> > TooNearCandidates;
    for( unsigned int i=0; i<MarkerCanditates.size(); i++ ) {
        //calculate the average distance of each corner to the nearest corner of the other marker candidate
        for ( unsigned int j=i+1; j<MarkerCanditates.size(); j++ ) {
            float dist=0;
            for ( int c=0; c<4; c++ )
                dist+= sqrt ( ( MarkerCanditates[i].contour[c].x-MarkerCanditates[j].contour[c].x ) *
                              ( MarkerCanditates[i].contour[c].x-MarkerCanditates[j].contour[c].x ) +
                              ( MarkerCanditates[i].contour[c].y-MarkerCanditates[j].contour[c].y ) *
                              ( MarkerCanditates[i].contour[c].y-MarkerCanditates[j].contour[c].y ) );
            dist /= 4;

            //if distance is too small
            if ( dist < minLeng_t ) {
                TooNearCandidates.push_back ( pair<int,int> ( i,j ) );
            }
        }
    }

    // mark for removal the element of  the pair with smaller perimeter
    valarray<bool> toRemove ( false, MarkerCanditates.size() );
    for ( unsigned int i=0;i<TooNearCandidates.size();i++ ) {
        if ( perimeter ( MarkerCanditates[TooNearCandidates[i].first ].contour ) >
             perimeter ( MarkerCanditates[ TooNearCandidates[i].second].contour ) )
            toRemove[TooNearCandidates[i].second]=true;
        else
            toRemove[TooNearCandidates[i].first]=true;
    }

    //remove the invalid ones
    //removeElements ( MarkerCanditates,toRemove );
    //finally, assign to the remaining candidates the contour
    bool _enableCylinderWarp = false;

    arrCR.reserve(MarkerCanditates.size());
    for (size_t i=0;i<MarkerCanditates.size();i++) {
        if ( !toRemove[i] ) {
            arrCR.push_back(MarkerCanditates[i]);

            // if the corners where swapped, it is required to
            //  reverse here the points so that they are in the same order
            if (swapped[i] && _enableCylinderWarp )
                reverse(arrCR.back().contour.begin(), arrCR.back().contour.end());//????
        }
    }

    // draw detected rectangles
    if( drawRes ) {
        img_in.copyTo(m_imgRes);

        // draw detected rectangles to work image
        vector<Point> c;

        for(int i=0; i<MarkerCanditates.size(); i++) {
            c.clear();
            for(int j=0; j<4; j++)
                c.push_back(Point(MarkerCanditates[i].contour[j].x, MarkerCanditates[i].contour[j].y));
            drawApproxCurve(m_imgRes, c, Scalar(255, 0, 255));
        }

        for(int i=0; i<arrCR.size(); i++) {
            c.clear();
            for(int j=0; j<4; j++)
                c.push_back(Point(arrCR[i].contour[j].x, arrCR[i].contour[j].y));
            drawApproxCurve(m_imgRes, c, Scalar(0,255,255));
        }

        fmt::print("candidateRect = {0}, Out_candidateRect = {1}\n",
               MarkerCanditates.size(), arrCR.size());
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


RLEDArray_Marker::RLEDArray_Marker()
{
    init();
}

RLEDArray_Marker::RLEDArray_Marker(RLEDArray_Detection *la_d)
{
    init();
    m_la_det = la_d;
}

RLEDArray_Marker::~RLEDArray_Marker()
{

}

void RLEDArray_Marker::init(void)
{
    m_la_det = NULL;

    marker_c = 0;
    tm_m1 = 0;  tm_m2 = 0;  tm_m = 0; tm_m1_c = 0; tm_m2_c = 0;
    tm_s1 = 0;  tm_s2 = 0;  tm_s = 0; tm_s1_c = 0; tm_s2_c = 0;
    tm_sf = 0;  tm_sf_c = 0;
}

int RLEDArray_Marker::detect(cv::Mat &imgMarker)
{
    // copy raw image
    imgMarker.copyTo(m_imgMarker);

    // extract binary code
    extractCode(imgMarker, m_codeRaw);

    // parse code
    return parseCode(m_codeRaw);
}

int RLEDArray_Marker::extractCode(cv::Mat &imgMarker, vector<int> &c)
{
    int     arrSize, arrN;
    int     x, y, w, h;
    int     r_size, r_full_size;

    Mat     ri_gray, ri_draw;

    // load parameters
    arrSize = m_la_det->arr_size;
    arrN = arrSize*arrSize;

    r_size = m_la_det->roi_size;
    r_full_size = m_la_det->rect_pixel_size / arrSize;
    w = r_size * 2;
    h = r_size * 2;

    // convert to gray image
    cv::cvtColor(imgMarker, ri_gray, CV_BGR2GRAY);

    ri_gray.copyTo(ri_draw);


    // detect gray-scale values
    float   ri_matrix[arrN];
    float   ri_v;

    for(int i=0; i<arrN; i++)  ri_matrix[i] = 0.0;

    for(int j=0; j<arrSize; j++) {
        y = j*r_full_size + r_full_size/2 - r_size;
        for(int k=0; k<arrSize; k++) {
            x = k*r_full_size + r_full_size/2 - r_size;

            Rect r_roi(x, y, w, h);

            // draw roi rectangle
            rectangle(ri_draw, r_roi, Scalar(0xFF, 0xFF, 0xFF), 1);

            // get roi mean color
            Mat img_roi = ri_gray(r_roi);
            ri_v = rectMeanValue(img_roi);

            ri_matrix[j*arrSize+k] = ri_v;
        }
    }


    // k-means to seperate two color
    Mat cl_samples(arrN, 1, CV_32F);
    int clusterCount = 2, attempts = 5;
    Mat labels, centers;
    float cl_1, cl_2;

    for(int j=0; j<arrN; j++) {
        cl_samples.at<float>(j) = ri_matrix[j];
    }

    kmeans(cl_samples, clusterCount, labels,
            TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
            attempts, KMEANS_PP_CENTERS,
            centers );
    cl_1 = centers.at<float>(0);
    cl_2 = centers.at<float>(1);


    float v_mean1 = 0.0, v_mean2;
    for(int j=0; j<arrN; j++) v_mean1 += ri_matrix[j];
    v_mean1 = v_mean1 / arrN;
    v_mean2 = (cl_1 + cl_2)/2.0;

    // get binary threshold
    m_grayThreshold = (v_mean1+v_mean2)/2.0;

    // convert to binary code
    int *p;

    c.resize(arrN);
    p = c.data();

    for(int j=0; j<arrN; j++) {
        if( ri_matrix[j] >= m_grayThreshold )  p[j] = 1;
        else                                   p[j] = 0;
    }

    if( m_la_det->drawRes ) {
        printf(">>> v_mean = %f, %f (%f, %f), threshold = %f\n",
               v_mean1, v_mean2, cl_1, cl_2,
               m_grayThreshold);

        for(int j=0; j<arrSize; j++) {
            for(int k=0; k<arrSize; k++) {
                printf("%d ", p[j*8+k]);
            }
            printf("\n");
        }
    }

    return 0;
}

int RLEDArray_Marker::parseCode(vector<int> &c)
{
    int     *p, *pi2;
    int     arrSize, arrN;

    // load parameters
    arrSize = m_la_det->arr_size;
    arrN    = arrSize*arrSize;

    // get bin array pointer
    p = c.data();

    // check board is correct or not
    int nBoard = 0, nBoardT;

    for(int i=0; i<arrSize; i++) nBoard += p[i];
    for(int i=0; i<arrSize; i++) nBoard += p[(arrSize -1)*arrSize + i];
    for(int i=0; i<arrSize; i++) nBoard += p[i*arrSize];
    for(int i=0; i<arrSize; i++) nBoard += p[i*arrSize + arrSize -1];

    nBoard -= 4;
    nBoardT = (int)(0.8*arrSize*4);

    if( nBoard < nBoardT ) {
        marker_c = 0;
        return -1;
    }

    // check corner mark
    int nCorner = 0;
    int arrC[4];

    arrC[0] = p[1*arrSize + 1];
    arrC[1] = p[1*arrSize + arrSize-2];
    arrC[2] = p[(arrSize-2)*arrSize + 1];
    arrC[3] = p[(arrSize-2)*arrSize + (arrSize-2)];

    nCorner = arrC[0] + arrC[1] + arrC[2] + arrC[3];
    if( nCorner != 3 ) {
        marker_c = 0;
        return -1;
    }

    marker_c = 1;   // set marker correct flag

    // process bin matrix to coorect order
    m_code.resize(arrN);
    pi2 = m_code.data();
    for(int i=0; i<arrN; i++) pi2[i] = p[i];
    p = pi2;

    if( arrC[0] == 0 ) {
        matrix_rotate180(p, arrSize);
    } else if ( arrC[1] == 0 ) {
        matrix_rotate90(p, arrSize);
    } else if ( arrC[2] == 0 ) {
        matrix_rotate270(p, arrSize);
    }

    // get time value
    tm_m1 = p[2*8+2] << 3 | p[2*8+3] << 2 | p[2*8+4] << 1 | p[2*8+5];
    tm_m2 = p[3*8+2] << 3 | p[3*8+3] << 2 | p[3*8+4] << 1 | p[3*8+5];
    tm_s1 = p[4*8+2] << 3 | p[4*8+3] << 2 | p[4*8+4] << 1 | p[4*8+5];
    tm_s2 = p[5*8+2] << 3 | p[5*8+3] << 2 | p[5*8+4] << 1 | p[5*8+5];

    tm_m = tm_m1*10 + tm_m2;
    tm_s = tm_s1*10 + tm_s2;

    // check time correct or not
    if( tm_m1 > 6 ) tm_m1_c = 0;
    if( tm_s1 > 6 ) tm_s1_c = 0;

    int p1, p2;

    p2 = p[2*8+1]; p1 = p[2*8+6];
    if( tm_m1 % 2 == 0 ) {
        if( p1 == 1 && p2 == 0 ) tm_m1_c = 1;
        else                     tm_m1_c = 0;
    } else {
        if( p1 == 0 && p2 == 1 ) tm_m1_c = 1;
        else                     tm_m1_c = 0;
    }

    p2 = p[3*8+1]; p1 = p[3*8+6];
    if( tm_m2 % 2 == 0 ) {
        if( p1 == 1 && p2 == 0 ) tm_m2_c = 1;
        else                     tm_m2_c = 0;
    } else {
        if( p1 == 0 && p2 == 1 ) tm_m2_c = 1;
        else                     tm_m2_c = 0;
    }

    p2 = p[4*8+1]; p1 = p[4*8+6];
    if( tm_s1 % 2 == 0 ) {
        if( p1 == 1 && p2 == 0 ) tm_s1_c = 1;
        else                     tm_s1_c = 0;
    } else {
        if( p1 == 0 && p2 == 1 ) tm_s1_c = 1;
        else                     tm_s1_c = 0;
    }

    p2 = p[5*8+1]; p1 = p[5*8+6];
    if( tm_s2 % 2 == 0 ) {
        if( p1 == 1 && p2 == 0 ) tm_s2_c = 1;
        else                     tm_s2_c = 0;
    } else {
        if( p1 == 0 && p2 == 1 ) tm_s2_c = 1;
        else                     tm_s2_c = 0;
    }

    // detect second flash
    int nSecFlash = 0;
    nSecFlash = p[1*8+2] + p[1*8+3] + p[6*8+4] + p[6*8+5];
    if( nSecFlash >= 2 ) tm_sf = 1;
    else                 tm_sf = 0;
    tm_sf_c = 1;

    // print results
    if( m_la_det->drawRes ) {
        printf(">>> ");
        print();
        printf("\n");

        float   fx, fy;
        int     x, y;
        char    buf[200];

        sprintf(buf, "Time: %02d:%02d (%d) (%2d)", tm_m, tm_s, tm_sf, isCorrect());

        fx = m_cr.contour[0].x;
        fy = m_cr.contour[0].y;
        for(int i=1; i<4; i++) {
            if( m_cr.contour[i].x > fx ) fx = m_cr.contour[i].x;
            if( m_cr.contour[i].y > fy ) fy = m_cr.contour[i].y;
        }

        x = fx + 10;
        y = fy + 10;

        putText(m_la_det->m_imgRes, buf, Point(x, y),
                FONT_HERSHEY_PLAIN, 1.5,
                Scalar(0xFF, 0, 0), 2);
    }

    if( tm_m1_c && tm_m2_c && tm_s1_c && tm_s2_c )
        return 0;
    else
        return -2;
}

int RLEDArray_Marker::isCorrect(void)
{
    if( !marker_c )
        return -1;

    if( tm_m1_c && tm_m2_c && tm_s1_c && tm_s2_c )
        return 0;
    else
        return -2;
}

void RLEDArray_Marker::print(void)
{
    fmt::print_colored(fmt::BLUE, "Time: {0:02d}:{1:02d} ({2}) ",
                       tm_m, tm_s, tm_sf);
    fmt::print_colored(fmt::RED, "({0}, {1}, {2}, {3}, {4}, {5})\n",
                       marker_c,
                       tm_m1_c, tm_m2_c, tm_s1_c, tm_s2_c,
                       tm_sf_c);
}


} // end of namespace rtk
