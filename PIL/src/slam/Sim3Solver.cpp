#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv/cv.h>

//#include "ORBmatcher.h"

#include <base/types/Random.h>

using namespace std;

Sim3Solver::Sim3Solver():
    mnIterations(0), mnBestInliers(0)
{

}

//return the inliers number
int Sim3Solver::getSim3Fast(std::vector<pi::Point3f> TrackPoints,
                            std::vector<pi::Point3f> GpsPoints,
                            pi::SIM3f& sim3)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    /// 1. Compute the centre of two point set and translate points to centre

    pi::Point3f centre_Track(0,0,0);
    pi::Point3f centre_GPS(0,0,0);
    size_t Num=TrackPoints.size();
    if(GpsPoints.size()!=Num) return -1;

    for(size_t i=0;i<Num;i++)
    {
        centre_Track=centre_Track+TrackPoints[i];
        centre_GPS  =centre_GPS+GpsPoints[i];
    }
    centre_Track=centre_Track/Num;
    centre_GPS=centre_GPS/Num;
    cout<<"centre_GPS:"<<centre_GPS<<endl;
    cout<<"centre_Track:"<<centre_Track<<endl;

    cv::Mat Pr1(3,Num,CV_32F); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(3,Num,CV_32F); // Relative coordinates to centroid (set 2)
    for(size_t i=0;i<Num;i++)
    {
        pi::Point3f& pt1=TrackPoints[i];
        pi::Point3f& pt2=GpsPoints[i];
        pt1=pt1-centre_Track;
        pt2=pt2-centre_GPS;
        cv::Mat(3,1,CV_32F,&pt1).copyTo(Pr1.col(i));
        cv::Mat(3,1,CV_32F,&pt2).copyTo(Pr2.col(i));
    }
    cout<<"Pr1:"<<Pr1<<endl;
    cout<<"Pr2:"<<Pr2<<endl;

    /// 2. Compute M ,N matrix

    cv::Mat M = Pr2*Pr1.t();
    cout<<M<<endl;

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,Pr1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    /// 3. Get rotation from eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec);
    cout<<"Eigen:"<<evec<<endl;
    pi::SO3f so3(evec.at<float>(0,1),
                 evec.at<float>(0,2),
                 evec.at<float>(0,3),
                 evec.at<float>(0,0));
//    so3=so3.inv();

    cv::Mat mR12i(3,3,CV_32F);
    pi::Point3f vec_p=so3.inv().ln();
    cout<<"ln:"<<vec_p<<endl;
    cv::Mat vec(1,3,CV_32F,&vec_p);
    cv::Rodrigues(vec,mR12i);
    cout<<"R:"<<mR12i<<endl;

    /// 4: Rotate set 2 and compute scale

    cv::Mat P3 = mR12i*Pr2;
    cout<<"P3"<<P3<<endl;

    double nom = Pr1.dot(P3);
    cv::Mat aux_P3(P3.size(),P3.type());
    aux_P3=P3;
    cv::pow(P3,2,aux_P3);
    double den = 0;

    for(int i=0; i<aux_P3.rows; i++)
    {
        for(int j=0; j<aux_P3.cols; j++)
        {
            den+=aux_P3.at<float>(i,j);
        }
    }

    float scale = den/nom;
    cout<<"Scale:"<<scale<<endl;

    /// 5. Compute translation and get SIM3

    cout<<"SO3:"<<so3<<endl;

    pi::Point3f translation = centre_GPS - (so3*centre_Track)*scale;
    cout<<"Translation:"<<translation<<endl;

    sim3=pi::SIM3f(so3,translation,scale);
}

int Sim3Solver::getSim3Fast(std::vector<pi::Point3d> TrackPoints,
                            std::vector<pi::Point3d> GpsPoints,
                            pi::SIM3d& sim3)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    /// 1. Compute the centre of two point set and translate points to centre

    pi::Point3d centre_Track(0,0,0);
    pi::Point3d centre_GPS(0,0,0);
    size_t Num=TrackPoints.size();
    if(GpsPoints.size()!=Num) return -1;

    for(size_t i=0;i<Num;i++)
    {
        centre_Track=centre_Track+TrackPoints[i];
        centre_GPS  =centre_GPS+GpsPoints[i];
    }
    centre_Track=centre_Track/(double)Num;
    centre_GPS=centre_GPS/(double)Num;
//    cout<<"centre_GPS:"<<centre_GPS<<endl;
//    cout<<"centre_Track:"<<centre_Track<<endl;

    cv::Mat Pr1(3,Num,CV_64F); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(3,Num,CV_64F); // Relative coordinates to centroid (set 2)
    for(size_t i=0;i<Num;i++)
    {
        pi::Point3d& pt1=TrackPoints[i];
        pi::Point3d& pt2=GpsPoints[i];
        pt1=pt1-centre_Track;
        pt2=pt2-centre_GPS;
        cv::Mat(3,1,Pr1.type(),&pt1).copyTo(Pr1.col(i));
        cv::Mat(3,1,Pr1.type(),&pt2).copyTo(Pr2.col(i));
    }
//    cout<<"Pr1:"<<Pr1<<endl;
//    cout<<"Pr2:"<<Pr2<<endl;

    /// 2. Compute M ,N matrix

    cv::Mat M = Pr2*Pr1.t();
//    cout<<M<<endl;

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,Pr1.type());

    N11 = M.at<double>(0,0)+M.at<double>(1,1)+M.at<double>(2,2);
    N12 = M.at<double>(1,2)-M.at<double>(2,1);
    N13 = M.at<double>(2,0)-M.at<double>(0,2);
    N14 = M.at<double>(0,1)-M.at<double>(1,0);
    N22 = M.at<double>(0,0)-M.at<double>(1,1)-M.at<double>(2,2);
    N23 = M.at<double>(0,1)+M.at<double>(1,0);
    N24 = M.at<double>(2,0)+M.at<double>(0,2);
    N33 = -M.at<double>(0,0)+M.at<double>(1,1)-M.at<double>(2,2);
    N34 = M.at<double>(1,2)+M.at<double>(2,1);
    N44 = -M.at<double>(0,0)-M.at<double>(1,1)+M.at<double>(2,2);

    N = (cv::Mat_<double>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    /// 3. Get rotation from eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec);
//    cout<<"Eigen:"<<evec<<endl;
    pi::SO3d so3(evec.at<double>(0,1),
                 evec.at<double>(0,2),
                 evec.at<double>(0,3),
                 evec.at<double>(0,0));
//    so3=so3.inv();

    cv::Mat mR12i(3,3,CV_64F);
    pi::Point3d vec_p=so3.inv().ln();
//    cout<<"ln:"<<vec_p<<endl;
    cv::Mat vec(1,3,CV_64F,&vec_p);
    cv::Rodrigues(vec,mR12i);
//    cout<<"R:"<<mR12i<<endl;

    /// 4: Rotate set 2 and compute scale

    cv::Mat P3 = mR12i*Pr2;
//    cout<<"P3"<<P3<<endl;

    double nom = Pr1.dot(P3);
    cv::Mat aux_P3(P3.size(),P3.type());
    aux_P3=P3;
    cv::pow(P3,2,aux_P3);
    double den = 0;

    for(int i=0; i<aux_P3.rows; i++)
    {
        for(int j=0; j<aux_P3.cols; j++)
        {
            den+=aux_P3.at<double>(i,j);
        }
    }

    double scale = den/nom;
//    cout<<"Scale:"<<scale<<endl;

    /// 5. Compute translation and get SIM3

//    cout<<"SO3:"<<so3<<endl;

    pi::Point3d translation = centre_GPS - (so3*centre_Track)*scale;
//    cout<<"Translation:"<<translation<<endl;

    sim3=pi::SIM3d(so3,translation,scale);
}

void Sim3Solver::SetData(std::vector<Sim3Match>& matches,cv::Mat k)
{
    N = matches.size();

//    mvnIndices1.reserve(N);
    mvX3Dc1.resize(N);
    mvX3Dc2.resize(N);

    mvAllIndices.reserve(N);

    for(int i=0; i<N; i++)
    {
        Sim3Match& match=matches[i];
        mvnMaxError1.push_back(9.210*match.sigma1);
        mvnMaxError2.push_back(9.210*match.sigma2);
        mvX3Dc1[i]=cv::Mat(3,1,CV_32F,&match.p1).clone();
        mvX3Dc2[i]=cv::Mat(3,1,CV_32F,&match.p2).clone();
    }

    mK1 = k.clone();
    mK2 = k.clone();

    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);
    SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;


    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mnIterations = 0;
}

cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);
    nInliers=0;

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < 3; ++i)
        {
            int randi = pi::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            vAvailableIndices[idx] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        computeT(P3Dc1i,P3Dc2i);

        CheckInliers();

        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[i] = true;
                return mBestT12;
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

void Sim3Solver::centroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    C = C/P.cols;

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;
    }
}

void Sim3Solver::computeT(cv::Mat &P1, cv::Mat &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    centroid(P1,Pr1,O1);
    centroid(P2,Pr2,O2);
    cout<<"Pr1:"<<Pr1<<endl;
    cout<<"Pr2:"<<Pr2<<endl;

    // Step 2: Compute M matrix

    cv::Mat M = Pr2*Pr1.t();
    cout<<"M"<<M<<endl;

    // Step 3: Compute N matrix

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);
    cout<<"N:"<<N<<endl;


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation
    cout<<"Eigen:"<<evec<<endl;

    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)
    cout<<"Vec:"<<vec<<endl;

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    double ang=atan2(norm(vec),evec.at<float>(0,0));

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

    mR12i.create(3,3,P1.type());

    cout<<"ln:"<<vec<<endl;
    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale

    double nom = Pr1.dot(P3);
    cv::Mat aux_P3(P3.size(),P3.type());
    aux_P3=P3;
    cv::pow(P3,2,aux_P3);
    double den = 0;

    for(int i=0; i<aux_P3.rows; i++)
    {
        for(int j=0; j<aux_P3.cols; j++)
        {
            den+=aux_P3.at<float>(i,j);
        }
    }

    ms12i = nom/den;

    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;

    // Step 8: Transformation

    // Step 8.1 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
    cout<<"Sim3:"<<mT21i<<endl;
}


void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);
    Project(mvX3Dc1,vP1im2,mT21i,mK2);

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        float err1 = dist1.dot(dist1);
        float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}


cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    float fx = K.at<float>(0,0);
    float fy = K.at<float>(1,1);
    float cx = K.at<float>(0,2);
    float cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        float invz = 1/(P3Dc.at<float>(2));
        float x = P3Dc.at<float>(0)*invz;
        float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    float fx = K.at<float>(0,0);
    float fy = K.at<float>(1,1);
    float cx = K.at<float>(0,2);
    float cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        float invz = 1/(vP3Dc[i].at<float>(2));
        float x = vP3Dc[i].at<float>(0)*invz;
        float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}
