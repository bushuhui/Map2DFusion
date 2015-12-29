#include "Optimizer.h"
#include "EdgeSE3GPSXYZ.h"
#include "TypeSE3_GPS.h"

#include <hardware/Camera/Camera.h>
#include <hardware/Gps/PathTable.h>
#include <base/Svar/Svar.h>
#include <base/time/Global_Timer.h>


#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/dense/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/sim3/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#ifdef HasPBA
#include <pba/pba.h>
#include <pba/DataInterface.h>
#include <pba/pba_util.h>
#endif

using namespace std;

typedef pi::Point3d Point3Type;

g2o::SE3Quat toQuat(const pi::SE3<double>& se3)
{
    const pi::Point3d& t=se3.get_translation();
    const pi::SO3d &    r=se3.get_rotation();
    g2o::SE3Quat result;
    result.setTranslation(*((g2o::Vector3d *)&t));
    result.setRotation(Eigen::Quaterniond(r.w,r.x,r.y,r.z));
    return result;
}

pi::SE3<double> fromQuat(const g2o::SE3Quat& se3)
{
    pi::SO3d r;
    r.x=se3.rotation().coeffs()[0];
    r.y=se3.rotation().coeffs()[1];
    r.z=se3.rotation().coeffs()[2];
    r.w=se3.rotation().coeffs()[3];
    const g2o::Vector3d& t=se3.translation();
    return pi::SE3<double>(r,pi::Point3d(t[0],t[1],t[2]));
}

int Optimizer::PoseOptimizationG2O(vector<pi::Point3d>& mappoints,vector<pi::Point2d>& observes,vector<float>& invSigma2,Camera* cam,pi::SE3d& pose)
{
    if(mappoints.size()!=observes.size()||mappoints.size()<10) return -1;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    int nInitialCorrespondences=0;

    // SET FRAME VERTEX
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(toQuat(pose));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // SET MAP POINT VERTICES
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vector<g2o::VertexSBAPointXYZ*> vVertices;
    vector<float>& vInvSigmas2=invSigma2;
    vector<size_t> vnIndexEdge;

    const int N = mappoints.size();
    vpEdges.reserve(N);
    vVertices.reserve(N);
    vnIndexEdge.reserve(N);
    vector<bool> mvbOutlier(N,false);

    const float delta = sqrt(5.991);
    Eigen::Matrix<double,2,1> obs;

    for(int i=0; i<N; i++)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(*((g2o::Vector3d*)&mappoints[i]));
        vPoint->setId(i+1);
        vPoint->setFixed(true);
        optimizer.addVertex(vPoint);
        vVertices.push_back(vPoint);

        nInitialCorrespondences++;
        //            mvbOutlier[i] = false;

        //SET EDGE
        obs(0,0)=observes[i].x;
        obs(1,0)=observes[i].y;

        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e->setMeasurement(obs);
        e->setInformation(Eigen::Matrix2d::Identity()*vInvSigmas2[i]);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(delta);

        e->fx = cam->Fx();
        e->fy = cam->Fy();
        e->cx = cam->Cx();
        e->cy = cam->Cy();

        optimizer.addEdge(e);

        vpEdges.push_back(e);
        vnIndexEdge.push_back(i);

    }

    // We perform 4 optimizations, decreasing the inlier region
    // From second to final optimization we include only inliers in the optimization
    // At the end of each optimization we check which points are inliers
    const float chi2[4]={9.210,7.378,5.991,5.991};
    const int its[4]={10,10,7,5};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization();
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdges.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

            const size_t idx = vnIndexEdge[i];

            if(mvbOutlier[idx])
            {
                e->setInformation(Eigen::Matrix2d::Identity()*vInvSigmas2[i]);
                e->computeError();
            }

            if(e->chi2()>chi2[it])
            {
                mvbOutlier[idx]=true;
                e->setInformation(Eigen::Matrix2d::Identity()*1e-10);
                nBad++;
            }
            else// if(e->chi2()<=chi2[it])
            {
                mvbOutlier[idx]=false;
            }
        }

        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    pose = fromQuat(SE3quat_recov);

    for(int i=0;i<N;i++)
        if(mvbOutlier[i]) vInvSigmas2[i]=-1;

    return nInitialCorrespondences-nBad;
}

std::vector<uint> Optimizer::BundleAdjustmentG2O
(std::vector<pi::SE3d>& KeyFrames,std::vector<pi::Point3d>& Points,
 std::vector<BundleObserve>& Observes,bool* pbStopFlag,int unFixKFNum,Camera* cam,
 const std::vector<GPSObserve>& GpsObs)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // SET KEYFRAME VERTICES
    for(std::vector<pi::SE3d>::iterator it=KeyFrames.begin(),iend=KeyFrames.end();it!=iend;it++)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(toQuat(*it));
        vSE3->setFixed(maxKFid>=unFixKFNum);
        vSE3->setId(maxKFid++);
        optimizer.addVertex(vSE3);
    }

    // SET MAP POINT VERTICES
    long unsigned int pointStartId=maxKFid;
    for(vector<pi::Point3d>::iterator it=Points.begin(),iend=Points.end();it!=iend;it++)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(*((g2o::Vector3d*)&*it));
        vPoint->setId(maxKFid++);
//        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
    }
    // SET UP EDGES

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vector<uint> outliers;
    vpEdges.reserve(Observes.size());

    const float thHuber = sqrt(5.991);

    double fx=cam->Fx();
    double fy=cam->Fy();
    double cx=cam->Cx();
    double cy=cam->Cy();
    for(vector<BundleObserve>::iterator it=Observes.begin(),iend=Observes.end(); it!=iend; it++)
    {
        BundleObserve& observe=*it;
        Eigen::Matrix<double,2,1> obs;
        obs(0,0)=observe.p_img.x;
        obs(1,0)=observe.p_img.y;

        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(observe.point_id+pointStartId)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(observe.frame_id)));
        e->setMeasurement(obs);
        e->setInformation(Eigen::Matrix2d::Identity()*observe.invSigma2);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuber);

        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;

        optimizer.addEdge(e);
        vpEdges.push_back(e);
    }
    // GPS VERTICES AND EDGES
    for(size_t i=0,iend=GpsObs.size();i<iend;i++)
    {
        const GPSObserve& gps_obs=GpsObs[i];
        // GPS VERTICE
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(*((g2o::Vector3d*)&gps_obs.gps_pose));
        vPoint->setId(maxKFid);
        vPoint->setMarginalized(true);
        vPoint->setFixed(true);
        optimizer.addVertex(vPoint);

        // EDGE
        g2o::EdgeSE3GPSXYZ* e=new g2o::EdgeSE3GPSXYZ();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(maxKFid++)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(gps_obs.frame_id)));
        e->setMeasurement(*((g2o::Vector3d*)&gps_obs.measure));
        e->setInformation(Eigen::Matrix3d::Identity()*gps_obs.invSigma2);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inlier observations
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            optimizer.removeEdge(e);
            vpEdges[i]=NULL;
            outliers.push_back(i);
        }
    }

    // Optimize again without the outliers

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Check inlier observations
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

        if(!e)
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            outliers.push_back(i);
        }
    }

    // Recover optimized data

    //Keyframes
    for(size_t i=0,iend=KeyFrames.size();i<iend;i++)
    {
        pi::SE3d& kf=KeyFrames[i];
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        kf=fromQuat(SE3quat);
    }

    //Points
    for(size_t i=0,iend=Points.size();i<iend;i++)
    {
        pi::Point3d& pt=Points[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pointStartId+i));
        g2o::Vector3d vec3=vPoint->estimate();
        pt=*((pi::Point3d*)&vec3);
    }
    return outliers;
}

using namespace pba;

std::vector<uint> Optimizer::BundleAdjustmentPBA
(std::vector<pi::SE3d>& KeyFrames,std::vector<pi::Point3d>& Points,
 std::vector<BundleObserve>& Observes,bool* pbStopFlag,int unFixKFNum,Camera* cam)
{
#ifdef HasPBA
    // CameraT, Point3D, Point2D are defined in pba/DataInterface.h
    vector<CameraT>         camera_data;    //camera (input/ouput)
    vector<PBAPoint3>       point_data;     //3D point(iput/output)
    vector<PBAPoint2f>      measurements;   //measurment/projection vector
    vector<int>             camidx,ptidx;  //index of camera/point for each projection

    float   f, cx, cy,fy,
            d[2],
            q[9], c[3];
    f =cam->Fx();
    fy=cam->Fy();
    cx=cam->Cx();
    cy=cam->Cy();
    float fx_fy=f/fy;
    d[0]=d[1]=0;

    /// 1.insert cameras
    int id=0;
    CameraT cameraT;
    cameraT.SetFocalLength(f);
    cameraT.SetNormalizedMeasurementDistortion(d[0]);    // FIXME: why d[0]?
    cameraT.SetFixedIntrinsic();
    camera_data.resize(KeyFrames.size(),cameraT);
    for(int i=0,iend=KeyFrames.size();i<iend;i++)
    {
        pi::SE3f se3=KeyFrames[i];//.inverse();
        se3.get_rotation().getMatrix(q);

//        cout<<"Camera "<<i<<":"<<se3<<endl;
//        for(int j=0;j<9;j++) cout<<q[j]<<" ";
//        cout<<endl;

        pi::Point3f t=se3.get_translation();
        c[0] = t[0]; c[1] = t[1];c[2] = t[2];

        camera_data[i].SetMatrixRotation(q);
        camera_data[i].SetTranslation(c);
        camera_data[i].SetFocalLength(f);
        camera_data[i].SetNormalizedMeasurementDistortion(d[0]);    // FIXME: why d[0]?
        //            camera_data[id].SetFixedIntrinsic();

        camera_data[i].SetVariableCamera();

        if(i>=unFixKFNum)
            camera_data[i].SetConstantCamera();
    }

    ///2.insert points
    id=0;
    point_data.resize(Points.size());
    for(int i=0,iend=Points.size();i<iend;i++)
    {
        pi::Point3d& pt=Points[i];
        point_data[i].SetPoint(pt.x,pt.y,pt.z);
    }

    ///3.insert observes
    measurements.resize(Observes.size());
    camidx.resize(Observes.size());
    ptidx.resize(Observes.size());
    for(int i=0,iend=Observes.size();i<iend;i++)
    {
        const BundleObserve& obs=Observes[i];
        measurements[i]=PBAPoint2f(obs.p_img.x-cx,(obs.p_img.y-cy)*fx_fy);
        camidx[i]      =obs.frame_id;
        ptidx[i]       =obs.point_id;
    }

    int N=camera_data.size();
    int M=point_data.size();
    int K=measurements.size();

    printf("\nadjustBundle_pba: \n");
    printf("  N (cams) = %d, M (points) = %d, K (measurements) = %d\n", N, M, K);

    //////////////////////////////////////////////////////////
    /// begin PBA
    //////////////////////////////////////////////////////////
    string inputFile=svar.GetString("PBA.InputFile","NoFile");
    if(inputFile!="NoFile")
    {
        vector<int>    ptc;  //point color
        vector<string> names;//keyframe names
        SaveNVM(inputFile.c_str(),camera_data,point_data,measurements,
                ptidx, camidx,
                names,
                ptc);
    }

    ParallelBA::DeviceT device = ParallelBA::PBA_CUDA_DEVICE_DEFAULT;

    ParallelBA pba(device);

    pba.SetFixedIntrinsics(true);

    pba.SetCameraData(camera_data.size(),  &camera_data[0]);    //set camera parameters
    pba.SetPointData(point_data.size(),    &point_data[0]);     //set 3D point data
    pba.SetProjection(measurements.size(), &measurements[0],    //set the projections
                        &ptidx[0], &camidx[0]);

    pba.RunBundleAdjustment();

    string outputFile=svar.GetString("PBA.OutputFile","NoFile");
    if(outputFile!="NoFile")
    {
        vector<int>    ptc;  //index of camera/point for each projection
        vector<string> names;
        SaveNVM(outputFile.c_str(),camera_data,point_data,measurements,ptidx,camidx,names,ptc);
    }

    //////////////////////////////////////////////////////////
    /// copy data back
    //////////////////////////////////////////////////////////
    ///
    for(int j=0; j<M; j++)
    {
        Point3Type& point=Points[j];
        point[0]=point_data[j].xyz[0];
        point[1]=point_data[j].xyz[1];
        point[2]=point_data[j].xyz[2];
    }

    pi::SE3f se3;
    for(int i=0;i<N;i++)
    {
        camera_data[i].GetMatrixRotation(q);
        camera_data[i].GetTranslation(c);
        se3.get_rotation().fromMatrix(q);
        se3.get_translation()=Point3Type(c[0],c[1],c[2]);
        KeyFrames[i]=se3;//.inverse();
    }
    return std::vector<uint>();
#else
    return BundleAdjustmentG2O(KeyFrames,Points,Observes,pbStopFlag,unFixKFNum,cam);
#endif

}

#if 0
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

cv::Vec3d LocalCoord2G(cv::Matx34d &mse3)
{
    using namespace Eigen;
    Matx33d R;
    Vec3d   t, tg;

    int     i, j;

    for(j=0; j<3; j++) {
        for(i=0; i<3; i++) {
            R(j, i) = mse3(j, i);
        }

        t(j) = mse3(j, 3);
    }

    tg = -R.t() * t;
    return tg;
}

cv::Matx34d GlobalCoord2L(cv::Vec3d &p)
{
    Matx34d mse3;
    Matx33d R(1, 0, 0, 0, 1, 0, 0, 0, 1);
    Vec3d   t;
    int     i, j;

    t = -p;

    for(j=0; j<3; j++) {
        for(i=0; i<3; i++)
            mse3(j, i) = R(j, i);

        mse3(j, 3) = t(j);
    }

    return mse3;
}

double CalcCameraDistance(cv::Matx34d &cam1, cv::Matx34d &cam2)
{
    Vec3d   c1, c2, vd;

    c1 = LocalCoord2G(cam1);
    c2 = LocalCoord2G(cam2);
    vd = c1 - c2;

    return sqrt(vd(0)*vd(0) + vd(1)*vd(1) + vd(2)*vd(2));
}

int CalcGround2CameraDis(cv::Matx34d &gp, vector<cv::Matx34d> &cam, vector<double> &d)
{
    Vec3d   gc, gn, cc, v;
    double  _d;

    // gound plane center & norm
    gc = LocalCoord2G(gp);
    gn(0) = gp(2, 0); gn(1) = gp(2, 1); gn(2) = gp(2, 2);

    // for each camera
    d.reserve(cam.size());

    for(int i=0; i<cam.size(); i++) {
        cc = LocalCoord2G(cam[i]);
        v = gc - cc;
        _d = fabs(v.dot(gn));
        d.push_back(_d);
    }

    return 0;
}
int CameraGPS_Fitting_lm(std::vector<cv::Matx34d> &cams, std::vector<cv::Vec3d> &gps,
                      cv::Matx34d &trans)
{
    using namespace Eigen;

    int             i, j, it;
    int             Ndata, Nparams;

    MatrixXd        cam_p, gps_p;
    MatrixXd        x_est, x_lm;
    MatrixXd        y_est, y_est_lm;
    MatrixXd        d, d_lm, dp;
    MatrixXd        J, H, H_lm, H_I;

    double          e, e_lm, e_lm_old;

    double          s, t, tx, ty, xc, yc;

    int             n_iters = 100;
    double          e_delta = 1e-3;
    double          lambda  = 0.01;
    int             updateJ = 1;


    // data & parameter size
    Ndata   = cams.size();
    Nparams = 4;

    // alloc matrix
    cam_p.resize(2, Ndata);
    gps_p.resize(2, Ndata);

    d.resize(2*Ndata, 1);
    d_lm.resize(2*Ndata, 1);
    dp.resize(Nparams, 1);

    J.resize(2*Ndata, Nparams);
    H.resize(Nparams, Nparams);
    H_lm.resize(Nparams, Nparams);
    H_I.resize(Nparams, Nparams);
    for(i=0; i<Nparams; i++) {
        for(j=0; j<Nparams; j++) {
            if( i==j ) H_I(i, j) = 1.0;
            else       H_I(i, j) = 0.0;
        }
    }

    x_est.resize(Nparams, 1);
    x_lm.resize(Nparams, 1);

    y_est.resize(2*Ndata, 1);
    y_est_lm.resize(2*Ndata, 1);


    // set initial gauss
    x_est << 1.0, 0, 0, 0;

    // convert camera & GPS positions
    for(i=0; i<cams.size(); i++) {
        Vec3d p = LocalCoord2G(cams[i]);

        cam_p(0, i) = p(0);
        cam_p(1, i) = p(1);

        gps_p(0, i) = gps[i](0);
        gps_p(1, i) = gps[i](1);
    }

    // begin iteration
    for(it=0, e_lm_old = 9e99; it<n_iters; it++) {
        if( updateJ == 1 ) {
            for(i=0; i<Ndata; i++) {
                s  = x_est(0);
                t  = x_est(1);
                xc = cam_p(0, i);
                yc = cam_p(1, i);

                J(i*2+0, 0) = xc*cos(t) - yc*sin(t);
                J(i*2+0, 1) = -s*xc*sin(t) - s*yc*cos(t);
                J(i*2+0, 2) = 1.0;
                J(i*2+0, 3) = 0.0;

                J(i*2+1, 0) = xc*sin(t) + yc*cos(t);
                J(i*2+1, 1) = s*xc*cos(t) - s*yc*sin(t);
                J(i*2+1, 2) = 0.0;
                J(i*2+1, 3) = 1.0;
            }

            // evaluate y by current x_est
            s  = x_est(0);
            t  = x_est(1);
            tx = x_est(2);
            ty = x_est(3);
            for(i=0; i<Ndata; i++) {
                xc = cam_p(0, i);
                yc = cam_p(1, i);

                y_est(i*2+0) = s*xc*cos(t) - s*yc*sin(t) + tx;
                y_est(i*2+1) = s*xc*sin(t) + s*yc*cos(t) + ty;

                d(i*2+0) = gps_p(0, i) - y_est(i*2+0);
                d(i*2+1) = gps_p(1, i) - y_est(i*2+1);
            }

            // calculate Hessian matrix
            H = J.transpose() * J;

            // calcualte error
            if( it == 0 ) {
                for(i=0, e = 0.0; i<Ndata; i++)
                    e += sqr(d(i*2+0)) + sqr(d(i*2+1));
            }
        }

        // calculate H_lm
        H_lm = H + lambda*H_I;

        // calculate step size
        dp = H_lm.inverse() * (J.transpose() * d);
        x_lm = x_est + dp;

        // calculate new error and resdiual
        s  = x_lm(0);
        t  = x_lm(1);
        tx = x_lm(2);
        ty = x_lm(3);

        for(i=0, e_lm=0.0; i<Ndata; i++) {
            xc = cam_p(0, i);
            yc = cam_p(1, i);

            y_est_lm(i*2+0) = s*xc*cos(t) - s*yc*sin(t) + tx;
            y_est_lm(i*2+1) = s*xc*sin(t) + s*yc*cos(t) + ty;

            d_lm(i*2+0) = gps_p(0, i) - y_est_lm(i*2+0);
            d_lm(i*2+1) = gps_p(1, i) - y_est_lm(i*2+1);

            e_lm += sqr(d_lm(i*2+0)) + sqr(d_lm(i*2+1));
        }

        printf("[%4d] e, e_lm = %12g, %12g, lambda = %12g\n",
               it, e, e_lm, lambda);

        // decide update by error
        if( e_lm < e ) {
            lambda = lambda/10.0;
            x_est = x_lm;
            e = e_lm;
            updateJ = 1;
        } else {
            updateJ = 0;
            lambda = lambda*10;
        }

        // check exit iteration
        if( fabs(e_lm - e_lm_old) < e_delta )
            break;

        e_lm_old = e_lm;
    }

    // output transformation matrix
    s  = x_est(0);
    t  = x_est(1);
    tx = x_est(2);
    ty = x_est(3);

    printf("s = %12g, t = %12g, tx, ty = %12g, %12g\n", s, t, tx, ty);

    // convert to Mat se(3)
    trans = Matx34d(s*cos(t), -s*sin(t), 0, tx,
                    s*sin(t),  s*cos(t), 0, ty,
                    0,         0,        s, 0);

    return 0;
}
#endif

double Optimizer::FitGPSSIM3WithTime(std::vector<std::pair<double,pi::SE3d> >& KeyFrames,
                                   FastPathTable* pathTable,pi::SIM3d& sim3,double& timeDiff,
                                   double invSigma)
{
    pi::timer.enter("Optimizer::FitGPSSIM3WithTime");
    if(KeyFrames.size()<4) return -1;

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(20);
    optimizer.setAlgorithm(solver);


    // SET GPS VERTICES
    g2o::VertexSBAGPSSIM3Time* gps=new g2o::VertexSBAGPSSIM3Time();
    gps->gps=pathTable;
    g2o::SIM3WithTime sim3WithTime;
    sim3WithTime.time=timeDiff;
    sim3WithTime.sim3.rotation()=g2o::Quaterniond(sim3.get_rotation().w,
                                                  sim3.get_rotation().x,
                                                  sim3.get_rotation().y,
                                                  sim3.get_rotation().z);
    sim3WithTime.sim3.translation()=*(g2o::Vector3d*)&sim3.get_translation();
    sim3WithTime.sim3.scale()=sim3.get_scale();
    gps->setEstimate(sim3WithTime);
    gps->setId(0);
    optimizer.addVertex(gps);

    // SET KEYFRAME VERTICES
    for(int i=0;i<KeyFrames.size();i++)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(toQuat(KeyFrames[i].second));
        vSE3->setId(i+1);
        if(invSigma<=0||i==0||i==KeyFrames.size()-1)
            vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        // set up gps edges
        {
            g2o::EdgeSE3_GPS* e = new g2o::EdgeSE3_GPS();
            e->frameTimestamp=KeyFrames[i].first;
            e->setVertex(1,vSE3);
            e->setVertex(0,gps);
            if(invSigma<=0)
                e->setInformation(Eigen::Matrix3d::Identity());
            else
                e->setInformation(Eigen::Matrix3d::Identity()*invSigma);
            optimizer.addEdge(e);
        }
    }

    // SET KEYFRAME EDGES
    if(invSigma>0)
    for(int i=1;i<KeyFrames.size();i++)
    {
        g2o::EdgeSE3Expmap* e=new g2o::EdgeSE3Expmap();
        e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
        e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
        e->setMeasurement(toQuat(KeyFrames[i].second*KeyFrames[i-1].second.inverse()));
        e->setInformation(Eigen::Matrix<double,6,6>::Identity()*100);
        optimizer.addEdge(e);
    }

//    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(30);


    // Recover optimized data
    double diffTemp=timeDiff;
    pi::SIM3d sim3Temp=sim3;
    diffTemp=gps->estimate().time;
    g2o::Sim3 sim_result=gps->estimate().sim3;
    sim3Temp.get_rotation()=pi::SO3d(sim_result.rotation().x(),
                                 sim_result.rotation().y(),
                                 sim_result.rotation().z(),
                                 sim_result.rotation().w());
    sim3Temp.get_translation()=*(pi::Point3d*)&sim_result.translation();
    sim3Temp.get_scale()=sim_result.scale();

    // Compute everaged squeare error
    double error=0;
    int    num=0;
    pi::Point3d gps_pose,fit_pose;
    for(size_t i=0,iend=KeyFrames.size();i<iend;i++)
    {
        if(pathTable->Get(KeyFrames[i].first+diffTemp,gps_pose))
        {
            fit_pose= gps_pose - sim3Temp*KeyFrames[i].second.inverse().get_translation();
//            cout<<"Error:"<<fit_pose<<endl;
            error  += fit_pose*fit_pose;
            num++;
        }
    }
    error =sqrt(error/num);

//    cout<<"Fitting everage error is "<<error<<endl;
//    if(error>10)
//    {
//        pi::timer.leave("Optimizer::FitGPSSIM3WithTime");
//        return error;
//    }

    if(invSigma>0)
    for(size_t i=0,iend=KeyFrames.size();i<iend;i++)
    {
        pi::SE3d& kf=KeyFrames[i].second;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i+1));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        kf=fromQuat(SE3quat);
    }

    sim3=sim3Temp;
    timeDiff=diffTemp;
    pi::timer.leave("Optimizer::FitGPSSIM3WithTime");
    return error;
}
