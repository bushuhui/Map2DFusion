#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include <base/types/SIM3.h>

//KeyFrame* mpKF1;
//KeyFrame* mpKF2;
//std::vector<MapPoint*> mvpMapPoints1;
//std::vector<MapPoint*> mvpMapPoints2;
//std::vector<MapPoint*> mvpMatches12;
//std::vector<size_t> mvSigmaSquare1;
//std::vector<size_t> mvSigmaSquare2;

struct Sim3Match
{
    cv::Point3f p1,p2;//
    float  sigma1,sigma2;
};

class Sim3Solver
{
public:

    Sim3Solver();

    void SetData(std::vector<Sim3Match>& matches,cv::Mat k);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float   GetEstimatedScale();

    static int getSim3Fast(std::vector<pi::Point3f> TrackPoints,
                           std::vector<pi::Point3f> GpsPoints,
                           pi::SIM3f& sim3);
    static int getSim3Fast(std::vector<pi::Point3d> TrackPoints,
                           std::vector<pi::Point3d> GpsPoints,
                           pi::SIM3d& sim3);
    void computeT(cv::Mat &P1, cv::Mat &P2);

protected:

    bool Refine();

    void centroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);


    void CheckInliers();

    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches

    std::vector<cv::Mat> mvX3Dc1;
    std::vector<cv::Mat> mvX3Dc2;

    std::vector<size_t> mvnIndices1;
    std::vector<size_t> mvnMaxError1;
    std::vector<size_t> mvnMaxError2;

    int N;
    int mN1;

    // Current Estimation
    cv::Mat mR12i;
    cv::Mat mt12i;
    float ms12i;
    cv::Mat mT12i;
    cv::Mat mT21i;
    std::vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    cv::Mat mBestT12;
    cv::Mat mBestRotation;
    cv::Mat mBestTranslation;
    float mBestScale;

    // Refined
    cv::Mat mRefinedT12;
    std::vector<bool> mvbRefinedInliers;
    int mnRefinedInliers;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    // Projections
    std::vector<cv::Mat> mvP1im1;
    std::vector<cv::Mat> mvP2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1;
    cv::Mat mK2;

};


#endif // SIM3SOLVER_H
