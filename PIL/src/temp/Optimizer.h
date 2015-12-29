#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>

#include <Eigen/StdVector>

#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"


#include <opengl/Win3D.h>

#include <TooN/se3.h>

#include "edge_se3_gps.h"
namespace pi {


class Optimizer:public Draw_Opengl,public EventHandle
{
public:
    Optimizer();
    void setCameraParaments(double fx, double fy, double cx, double cy, double tx=1);

    g2o::SparseOptimizer& Opt(){return optimizer;}

    void insertPose(std::vector<g2o::VertexSCam*> &pose){/*poses=pose;*/}
    void insertPose(g2o::VertexSCam* pose){poses.push_back(pose);}
    void insertPoint(std::vector<g2o::VertexSBAPointXYZ*> &point){points=point;}
    void insertPoint(g2o::VertexSBAPointXYZ* point){points.push_back(point);}
    void insertObserve(std::vector<g2o::Edge_XYZ_VSC*> &obs){observe=obs;}
    void insertObserve(g2o::Edge_XYZ_VSC* obs){observe.push_back(obs);}

    void insertPoint(Eigen::Vector3d p,int id_point,bool fixed=false);
    void insertPose(Eigen::Isometry3d pose,int ID=-1,bool fixed=false);
    void insertObserve(Eigen::Vector3d z,int id_cam,int id_point);


    bool loadFromNVMFile(const string& filename);
    bool save2NVMFile(const string& filename);

    Eigen::Vector3d getPoint(int id_point);
    Eigen::Isometry3d getPose(int id_cam);


    void compute(int i=10);
    void sim();
    void GPS_Fitting();
    void GPS_Fitting(vector<TooN::SE3<> >& frames,vector<TooN::Vector<3> >&gps_poses);
protected:
    virtual void Draw_Something();
    virtual bool KeyPressHandle(void *);

public:
    Win3D  *win3d;

protected:
    g2o::SparseOptimizer                     optimizer;
    g2o::BlockSolver_6_3 *                   solver_ptr;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    g2o::OptimizationAlgorithmLevenberg*     solver;

    std::vector<g2o::VertexSE3*>            poses;
    std::vector<g2o::VertexSBAPointXYZ*>    points;
    std::vector<g2o::Edge_XYZ_VSC*>         observe;
    std::vector<g2o::EdgeSE3*>              odoment;
    std::vector<g2o::VertexPointXYZ*>       gps_points;
    std::vector<g2o::EdgeSE3GPS*>           edge_gps;

    int id;
    float gl_pointsize,axis_length;
};

}//end of namespace pi
#endif // OPTIMIZER_H
