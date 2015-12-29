#ifndef __LEDARRAY_DETECTION_H__
#define __LEDARRAY_DETECTION_H__

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;

namespace rtk {

class RLEDArray_Marker;
typedef std::vector<RLEDArray_Marker> RLEDArray_Marker_List;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


///
/// \brief The CandidateRect class
///
class CandidateRect
{
public:
    CandidateRect() {}

    CandidateRect(const  CandidateRect &M) {
        contour = M.contour;
        idx     = M.idx;
    }
    CandidateRect & operator = (const  CandidateRect &M) {
        contour = M.contour;
        idx     = M.idx;
    }


public:
    vector<cv::Point2f> contour;    //all the points of its contour
    int                 idx;        //index position in the global contour list
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


///
/// \brief The RLEDArray_Detection class
///
class RLEDArray_Detection
{
public:
    RLEDArray_Detection();
    ~RLEDArray_Detection();

    ///
    /// \brief init parameters
    ///
    void initParam(void);

    ///
    /// \brief detect led array marker
    /// \param img_in - input image
    /// \param arrM   - output marker list
    /// \return
    ///
    int detect(cv::Mat &img_in, RLEDArray_Marker_List &arrM);

    ///
    /// \brief detectRectangs
    /// \param img_in
    /// \param arrCR
    /// \return
    ///
    int detectRectangs(cv::Mat &img_in, vector<CandidateRect> &arrCR);

public:
    double  _minSize;               ///< contour detection min size
    double  _maxSize;               ///< contour detection max size

    int     minLeng_t;              ///< rectangle edge min length threshold
    double  edgeLengRatio;          ///< maxEdge/minEdge ratio

    int     arr_size;               ///< array size (block size in each direction)
    int     roi_size;               ///< detection ROI size
    int     rect_pixel_size;        ///< rectangle size

    int     drawRes;                ///< draw results

    cv::Mat m_imgRaw, m_imgGray, m_imgRes;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


///
/// \brief The RLEDArray_Marker class
///
class RLEDArray_Marker
{
public:
    RLEDArray_Marker();
    RLEDArray_Marker(RLEDArray_Detection *la_d);
    ~RLEDArray_Marker();

    ///
    /// \brief init variables
    ///
    void init(void);

    ///
    /// \brief detect marker
    /// \param img_mark - input marker image
    /// \return 0 - success, -1 boarder error, -2 - time error
    ///
    int detect(cv::Mat &imgMarker);

    ///
    /// \brief extract code from image
    /// \param imgMarker - marker image
    /// \param c         - binary code
    /// \return
    ///
    int extractCode(cv::Mat &imgMarker, vector<int> &c);

    ///
    /// \brief parse code from marker
    /// \param c - 2D array of int
    /// \return
    ///
    int parseCode(vector<int> &c);

    ///
    /// \brief detected marker is correct or not
    /// \return 0 - correct, -1 - wrong
    ///
    int isCorrect(void);

    ///
    /// \brief print results
    ///
    void print(void);


public:
    cv::Mat         m_imgMarker;
    CandidateRect   m_cr;

    vector<int>     m_codeRaw, m_code;

    double          m_grayThreshold;

    int     marker_c;
    int     tm_m1, tm_m2, tm_m, tm_m1_c, tm_m2_c;
    int     tm_s1, tm_s2, tm_s, tm_s1_c, tm_s2_c;
    int     tm_sf, tm_sf_c;

    RLEDArray_Detection *m_la_det;
};


} // end of namespace rtk

#endif // end of __LEDARRAY_DETECTION_H__
