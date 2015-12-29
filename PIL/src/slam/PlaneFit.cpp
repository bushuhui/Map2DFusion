#include "PlaneFit.h"
#include <vector>
#include <TooN/wls.h>
#include <TooN/SymEigen.h>

using namespace std;

namespace pi{

int Plane_Fit::fitPlaneRansac(std::vector<pi::Point3d>& points,pi::SE3d& plane,
                              std::vector<int>& outliers,double thresholdZ)
{
    unsigned int nPoints = points.size();
    if(nPoints < 10)
    {
        cout << "CalcPlane: too few points to calc plane." << endl;
        return -1;
    }
    int nRansacs =100;// GV2.GetInt("MapMaker.PlaneAlignerRansacs", 100, HIDDEN|SILENT);
    pi::Point3d v3BestMean;
    pi::Point3d v3BestNormal;
    double dBestDistSquared = 9999999999999999.9;

    for(int i=0; i<nRansacs; i++)
    {
        int nA = rand()%nPoints;
        int nB = nA;
        int nC = nA;
        while(nB == nA)
            nB = rand()%nPoints;
        while(nC == nA || nC==nB)
            nC = rand()%nPoints;

        pi::Point3d v3Mean = 0.33333333 * (points[nA] +
                                           points[nB] +
                                           points[nC]);

        pi::Point3d v3CA = points[nC]  - points[nA];
        pi::Point3d v3BA = points[nB]  - points[nA];
        pi::Point3d v3Normal = v3CA ^ v3BA;
        if(v3Normal * v3Normal  == 0)
            continue;
        v3Normal=v3Normal.normalize();

        double dSumError = 0.0;
        for(unsigned int i=0; i<nPoints; i++)
        {
            pi::Point3d v3Diff = points[i] - v3Mean;
            double dDistSq = v3Diff * v3Diff;
            if(dDistSq == 0.0)
                continue;
            double dNormDist = fabs(v3Diff * v3Normal);

            if(dNormDist > thresholdZ)
                dNormDist = thresholdZ;
            dSumError += dNormDist;
        }
        if(dSumError < dBestDistSquared)
        {
            dBestDistSquared = dSumError;
            v3BestMean = v3Mean;
            v3BestNormal = v3Normal;
        }
    }

    // Done the ransacs, now collect the supposed inlier set
    vector<pi::Point3d > vv3Inliers;
    outliers.clear();
    outliers.reserve(nPoints);
    vv3Inliers.reserve(nPoints);

    for(unsigned int i=0; i<nPoints; i++)
    {
        pi::Point3d v3Diff = points[i] - v3BestMean;
        double dDistSq = v3Diff * v3Diff;
        if(dDistSq == 0.0)
            continue;
        double dNormDist = fabs(v3Diff * v3BestNormal);
        if(dNormDist < thresholdZ)
        {
            vv3Inliers.push_back(points[i]);
        }
        else
            outliers.push_back(i);
    }

    // With these inliers, calculate mean and cov
    pi::Point3d v3MeanOfInliers(0,0,0);
    for(unsigned int i=0; i<vv3Inliers.size(); i++)
        v3MeanOfInliers=v3MeanOfInliers+vv3Inliers[i];
    v3MeanOfInliers =v3MeanOfInliers*(1.0 / vv3Inliers.size());

    TooN::Matrix<3> m3Cov = TooN::Zeros;
    for(unsigned int i=0; i<vv3Inliers.size(); i++)
    {
        pi::Point3d v3Diff_ZY = vv3Inliers[i] - v3MeanOfInliers;
        TooN::Vector<3>& v3Diff=*(TooN::Vector<3>*)&v3Diff_ZY;
        m3Cov += v3Diff.as_col() * v3Diff.as_row();
    };

    // Find the principal component with the minimal variance: this is the plane normal
    TooN::SymEigen<3> sym(m3Cov);
    TooN::Vector<3> v3Normal = sym.get_evectors()[0];

    if(v3Normal[2] < 0)
        v3Normal = -v3Normal ;

    TooN::Matrix<3> m3Rot = TooN::Identity;
    m3Rot[2] = v3Normal;//z
    m3Rot[0] = m3Rot[0] - (v3Normal * (m3Rot[0] * v3Normal));//x
    TooN::normalize(m3Rot[0]);
    m3Rot[1] = m3Rot[2] ^ m3Rot[0];//y

    SE3d se3Aligner;
    se3Aligner.get_rotation().fromMatrixUnsafe(m3Rot);
    se3Aligner=se3Aligner.inverse();

    se3Aligner.get_translation() = v3MeanOfInliers;

    plane=se3Aligner;
    return 0;
}

}
