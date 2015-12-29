#include "Optimizer.h"

#include <unordered_set>

#include <iostream>
#include <stdint.h>
#include <base/Svar/Svar_Inc.h>
#include <base/time/Global_Timer.h>

namespace pi {

using namespace Eigen;
using namespace std;
using namespace g2o;

class Sample
{
public:
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr)
{
    return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma)
{
    double x, y, r2;
    do {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to)
{
    return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform()
{
    return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma)
{
    return gauss_rand(0., sigma);
}

Optimizer::Optimizer()
{
    win3d=NULL;

    id=0;
    //setup optimizer
    optimizer.setVerbose(false);
    string solver_name=GV2.GetString("Solver","Csparse");
    if ("Dense"==solver_name)
    {
        linearSolver= new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        cerr << "Using DENSE" << endl;
    }
    else
    {
        if("Cholmod"==solver_name)
        {
            cerr << "Using CHOLMOD" << endl;
            linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
        }
        else
        {
            linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
            cerr << "Using CSPARSE" << endl;
        }
    }

    solver_ptr= new g2o::BlockSolver_6_3(linearSolver);
    solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);


    optimizer.setAlgorithm(solver);
    gl_pointsize=GV2.GetDouble("PointSize",2.5);
    axis_length=GV2.GetDouble("AxisLength",2);

}

bool Optimizer::loadFromNVMFile(const string& filename)
{
#if 0
    int rotation_parameter_num = 4;
    bool format_r9t = false;
    string token;
    if(in.peek() == 'N')
    {
        in >> token; //file header
        if(strstr(token.c_str(), "R9T"))
        {
            rotation_parameter_num = 9;    //rotation as 3x3 matrix
            format_r9t = true;
        }
    }

    int ncam = 0, npoint = 0, nproj = 0;
    // read # of cameras
    in >> ncam;  if(ncam <= 1) return false;

    //read the camera parameters
//    poses.resize(ncam); // allocate the camera data
//    names.resize(ncam);
    Eigen::Isometry3d pose;
    Eigen:: Quaterniond quat;
    for(int i = 0; i < ncam; ++i)
    {

        g2o::VertexSCam * v_se3
                = new g2o::VertexSCam();

        double f, q[9], c[3], d[2];
        in >> token >> f ;

        for(int j = 0; j < rotation_parameter_num; ++j) in >> q[j];
        in >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];


        v_se3->setKcam(f,f,0,0,0);

        if(format_r9t)
        {
            Matrix4d& m=pose.matrix();
        }
        else
        {
            //older format for compability
            camera_data[i].SetQuaternionRotation(q);        //quaternion from the file
            camera_data[i].SetCameraCenterAfterRotation(c); //camera center from the file
        }
//        camera_data[i].SetNormalizedMeasurementDistortion(d[0]);
//        names[i] = token;


        v_se3->setId(i);
        v_se3->setEstimate(pose);
        v_se3->setAll();            // set aux transforms
        poses.push_back(v_se3);
        optimizer.addVertex(v_se3);

    }

    //////////////////////////////////////
    in >> npoint;   if(npoint <= 0) return false;

    //read image projections and 3D points.
    point_data.resize(npoint);
    for(int i = 0; i < npoint; ++i)
    {
        float pt[3]; int cc[3], npj;
        in  >> pt[0] >> pt[1] >> pt[2]
                >> cc[0] >> cc[1] >> cc[2] >> npj;
        for(int j = 0; j < npj; ++j)
        {
            int cidx, fidx; float imx, imy;
            in >> cidx >> fidx >> imx >> imy;

            camidx.push_back(cidx);    //camera index
            ptidx.push_back(i);        //point index

            //add a measurment to the vector
            measurements.push_back(Point2D(imx, imy));
            nproj ++;
        }
        point_data[i].SetPoint(pt);
        ptc.insert(ptc.end(), cc, cc + 3);
    }
    ///////////////////////////////////////////////////////////////////////////////
    std::cout << ncam << " cameras; " << npoint << " 3D points; " << nproj << " projections\n";
#endif
    return true;
}

bool Optimizer::save2NVMFile(const string& filename)
{
#if 0
    std::cout << "Saving model to " << filename << "\n";
    ofstream out(filename);

    out << "NVM_V3_R9T\n" << poses.size() << '\n' << std::setprecision(12);
    if(names.size() < camera_data.size()) names.resize(camera_data.size(),string("unknown"));
    if(ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

    ////////////////////////////////////
    for(size_t i = 0; i < camera_data.size(); ++i)
    {
        CameraT& cam = camera_data[i];
        out << names[i] << ' ' << cam.GetFocalLength() << ' ';
        for(int j  = 0; j < 9; ++j) out << cam.m[0][j] << ' ';
        out << cam.t[0] << ' ' << cam.t[1] << ' ' << cam.t[2] << ' '
            << cam.GetNormalizedMeasurementDistortion() << " 0\n";
    }

    out << point_data.size() << '\n';

    for(size_t i = 0, j = 0; i < point_data.size(); ++i)
    {
        Point3D& pt = point_data[i];
        int * pc = &ptc[i * 3];
        out << pt.xyz[0] << ' ' << pt.xyz[1] << ' ' << pt.xyz[2] << ' '
            << pc[0] << ' ' << pc[1] << ' ' << pc[2] << ' ';

        size_t je = j;
        while(je < ptidx.size() && ptidx[je] == (int) i) je++;

        out << (je - j) << ' ';

        for(; j < je; ++j)    out << camidx[j] << ' ' << " 0 " << measurements[j].x << ' ' << measurements[j].y << ' ';

        out << '\n';
    }
#endif
}

void Optimizer::setCameraParaments(double fx, double fy, double cx, double cy, double tx)
{
    g2o::VertexSCam::setKcam(fx,fy,cx,cy,tx);
}

void Optimizer::insertPose(Eigen::Isometry3d pose,int ID,bool fixed)
{
//    cout<<"inserting pose,id="<<ID<<endl;
    g2o::VertexSCam * v_se3
            = new g2o::VertexSCam();

    if(ID<0)
    {
        v_se3->setId(id);
        id++;
    }
    else
        v_se3->setId(ID);
    v_se3->setEstimate(pose);
    v_se3->setAll();            // set aux transforms
//    v_se3->setKcam();

    if (fixed)
        v_se3->setFixed(true);
    poses.push_back(v_se3);
    optimizer.addVertex(v_se3);
}

void Optimizer::insertPoint(Eigen::Vector3d p,int id_point,bool fixed)
{
//    cout<<"Insert Point "<<p.transpose()<<"Id="<<id_point<<endl;
    g2o::VertexSBAPointXYZ * v_p
            = new g2o::VertexSBAPointXYZ();
    v_p->setId(id_point);
    v_p->setMarginalized(true);
    v_p->setEstimate(p);
    optimizer.addVertex(v_p);
    points.push_back(v_p);
}

void Optimizer::insertObserve(Eigen::Vector3d z,int id_cam,int id_point)
{

    g2o::Edge_XYZ_VSC * e
            = new g2o::Edge_XYZ_VSC();


//    cout<<"insertObserve.1\n";
    e->vertices()[0]
            = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(id_point)->second);

//    cout<<"insertObserve.2\n";
    e->vertices()[1]
            = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(id_cam)->second);

//    cout<<"insertObserve.3\n";
    z[2]=1;
    e->setMeasurement(z);
//    cout<<"insertObserve.4\n";
    //e->inverseMeasurement() = -z;
    Matrix3d m;
    m<<1,0,0,
            0,1,0,
            0,0,0.0001;

    e->information() = m;

//    cout<<"insertObserve.5\n";
    optimizer.addEdge(e);

//    cout<<"insertObserve.6\n";
    observe.push_back(e);
}

Eigen::Vector3d Optimizer::getPoint(int id_point)
{
    const VertexSBAPointXYZ* v1 = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertices().find(id_point)->second);
    return v1->estimate();
}

Eigen::Isometry3d Optimizer::getPose(int id_cam)
{
    const VertexSCam* v1 = dynamic_cast<g2o::VertexSCam*>(optimizer.vertices().find(id_cam)->second);
    return v1->estimate();
}

void Optimizer::compute(int i)
{

    optimizer.initializeOptimization();

    optimizer.setVerbose(false);
//    cout<<"start optimizing...\n";
    optimizer.optimize(i);
}

void Optimizer::Draw_Something()
{
    glDisable(GL_LIGHTING);
    //    cout<<"Drawing...";
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    //Draw colorful points
    glPointSize(gl_pointsize);

    glBegin(GL_POINTS);
    glNormal3d(0, 0, 1);
    glColor3ub(255,255,255);
    for(int i=0;i<points.size();i++)
    {
        g2o::VertexSBAPointXYZ * v_p=points[i];
        Vector3d p=10*v_p->estimate();
        glVertex3d(p[0],p[1],p[2]);
    }

    glColor3ub(255,255,0);
    for(int i=0;i<gps_points.size();i++)
    {
        g2o::VertexPointXYZ * v_p=gps_points[i];
        Vector3d p=10*v_p->estimate();
        glVertex3d(p[0],p[1],p[2]);
    }
    glEnd();

    //Draw camera axis
    double length=axis_length;
    for(int i=0;i<poses.size();i++)
    {
        g2o::VertexSE3* cam=poses[i];
        Eigen::Isometry3d pose=cam->estimate();
        //        Vector3d trans=pose.translate();
        Matrix4d m=pose.matrix();
        Vector3d trans=pose.translation();//m.block<3,3>(0,0)*m.block<3,1>(0,3);
        trans*=10;
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3ub(255,0,0);
        glVertex3d(trans[0],trans[1],trans[2]);
        glVertex3d(trans[0]+length*m(0,0),trans[1]+length*m(1,0),trans[2]+length*m(2,0));
        glColor3ub(0,255,0);
        glVertex3d(trans[0],trans[1],trans[2]);
        glVertex3d(trans[0]+length*m(0,1),trans[1]+length*m(1,1),trans[2]+length*m(2,1));
        glColor3ub(0,0,255);
        glVertex3d(trans[0],trans[1],trans[2]);
        glVertex3d(trans[0]+length*m(0,2),trans[1]+length*m(1,2),trans[2]+length*m(2,2));
        glEnd();
    }
    //Draw edges
    glColor3ub(0,255,255);
    glLineWidth(1);
    for(int i=0;i<edge_gps.size();i++)
    {
        glBegin(GL_LINES);
        g2o::EdgeSE3GPS* edge=edge_gps[i];

        const VertexSE3* v1 = static_cast<const VertexSE3*>(edge->vertices()[0]);
        Vector3D t1=10*v1->estimate().translation();
        const VertexPointXYZ* v2 = static_cast<const VertexPointXYZ*>(edge->vertices()[1]);
        Vector3D t2=10*v2->estimate();
        glVertex3d(t1[0],t1[1],t1[2]);
        glVertex3d(t2[0],t2[1],t2[2]);
        glEnd();
    }

    glColor3ub(255,0,255);
    glLineWidth(1);
    for(int i=0;i<odoment.size();i++)
    {
        glBegin(GL_LINES);
        g2o::EdgeSE3* edge=odoment[i];

        const VertexSE3* v1 = poses[i];//static_cast<const VertexSE3*>(edge->vertices()[0]);
        Vector3D t1=10*v1->estimate().translation();
        const VertexSE3* v2 = poses[i+1];//static_cast<const VertexSE3*>(edge->vertices()[1]);
        Vector3D t2=10*v2->estimate().translation();
        glVertex3d(t1[0],t1[1],t1[2]);
        glVertex3d(t2[0],t2[1],t2[2]);
        glEnd();
    }
    glPopMatrix();
}

bool Optimizer::KeyPressHandle(void *arg)
{
    QKeyEvent* e=(QKeyEvent*)arg;
    switch(e->key())
    {
    case Qt::Key_O:
        optimizer.optimize(2);
        if(win3d)
            win3d->update();
        return true;
    case Qt::Key_S:
        optimizer.save("result.g2o");
        return true;
    default:
        return false;
    }
}


void Optimizer::sim()
{

    // set up 500 points
    vector<Vector3d> true_points;
    int pointnum=svar.GetInt("PointNum",500);
    for (size_t i=0;i<pointnum; ++i)
    {
        true_points.push_back(Vector3d((Sample::uniform()-0.5)*10,
                                       Sample::uniform()-0.5,
                                       Sample::uniform()+10));
    }


    Vector2d focal_length(500,500); // pixels
    Vector2d principal_point(320,240); // 640x480 image
    double baseline = 0.075;      // 7.5 cm baseline


    vector<Eigen::Isometry3d,
            aligned_allocator<Eigen::Isometry3d> > true_poses;

    // set up camera params
    g2o::VertexSCam::setKcam(focal_length[0],focal_length[1],
            principal_point[0],principal_point[1],
            baseline);

    // set up 5 vertices, first 2 fixed
    int vertex_id = 0;
    int pose_num=svar.GetInt("PoseNum",10);
    for (size_t i=0; i<pose_num; ++i)
    {


        Vector3d trans(i*0.04-1.,0,0);

        Eigen:: Quaterniond q;
        q.setIdentity();
        Eigen::Isometry3d pose;
        pose = q;
        pose.translation() = trans;

//        g2o::VertexSCam * v_se3
//                = new g2o::VertexSCam();

//        v_se3->setId(vertex_id);
//        v_se3->setEstimate(pose);
//        v_se3->setAll();            // set aux transforms

//        if (i<1)
//            v_se3->setFixed(true);

//        optimizer.addVertex(v_se3);
//        poses.push_back(v_se3);
        if(i<2)
            insertPose(pose,i,true);
        else
            insertPose(pose,i,false);
        true_poses.push_back(pose);
        vertex_id++;
    }

    int point_id=vertex_id;
    int point_num = 0;
    double sum_diff2 = 0;

    cout << endl;
    unordered_map<int,int> pointid_2_trueid;
    unordered_set<int> inliers;

    // add point projections to this vertex
    for (size_t i=0; i<true_points.size(); ++i)
    {

        int num_obs = 0;

        for (size_t j=0; j<true_poses.size(); ++j)
        {
            Vector3d z;
            dynamic_cast<g2o::VertexSCam*>
                    (optimizer.vertices().find(j)->second)
                    ->mapPoint(z,true_points.at(i));

            if (z[0]>=0 && z[1]>=0 && z[0]<640 && z[1]<480)
            {
                ++num_obs;
            }
        }

        if (num_obs>=2)
        {
//            g2o::VertexSBAPointXYZ * v_p
//                    = new g2o::VertexSBAPointXYZ();


//            v_p->setId(point_id);
//            v_p->setMarginalized(true);
//            v_p->setEstimate(true_points.at(i)
//                             + Vector3d(Sample::gaussian(1),
//                                        Sample::gaussian(1),
//                                        Sample::gaussian(1)));
//            optimizer.addVertex(v_p);
//            points.push_back(v_p);
            insertPoint(true_points.at(i)
                        + Vector3d(Sample::gaussian(1),
                                   Sample::gaussian(1),
                                   Sample::gaussian(1)),point_id,false);
            bool inlier = true;
            Vector3d z;
            for (size_t j=0; j<true_poses.size(); ++j)
            {
                dynamic_cast<g2o::VertexSCam*>
                        (optimizer.vertices().find(j)->second)
                        ->mapPoint(z,true_points.at(i));

                if (z[0]>=0 && z[1]>=0 && z[0]<640 && z[1]<480)
                {
                    double sam = Sample::uniform();
                    if (sam<0)
                    {
                        z = Vector3d(Sample::uniform(64,640),
                                     Sample::uniform(0,480),
                                     Sample::uniform(0,64)); // disparity
                        z(2) = z(0) - z(2); // px' now

                        inlier= false;
                    }
                    int PIXEL_NOISE=1;
                    z += Vector3d(Sample::gaussian(PIXEL_NOISE),
                                  Sample::gaussian(PIXEL_NOISE),
                                  Sample::gaussian(PIXEL_NOISE/16.0));

//                    g2o::Edge_XYZ_VSC * e
//                            = new g2o::Edge_XYZ_VSC();


//                    e->vertices()[0]
//                            = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);

//                    e->vertices()[1]
//                            = dynamic_cast<g2o::OptimizableGraph::Vertex*>
//                            (optimizer.vertices().find(j)->second);

//                    e->setMeasurement(z);
//                    //e->inverseMeasurement() = -z;
//                    e->information() = Matrix3d::Identity();

//                    if (0) {
//                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                        e->setRobustKernel(rk);
//                    }

//                    optimizer.addEdge(e);
                    insertObserve(z,j,point_id);
//                    observe.push_back(e);


                }

            }

            if (inlier)
            {
                inliers.insert(point_id);
                Vector3d diff = z - true_points[i];

                sum_diff2 += diff.dot(diff);
            }
            // else
            //   cout << "Point: " << point_id <<  "has at least one spurious observation" <<endl;

            pointid_2_trueid.insert(make_pair(point_id,i));

            ++point_id;
            ++point_num;
        }

    }
    cout<<"1\n";
    cout<<"2\n";
    //    compute();

    optimizer.initializeOptimization();

    optimizer.setVerbose(true);
    //    win3d->update();
    while(1)
    {
        optimizer.optimize(1);
        win3d->update();
    }
}

Eigen::Isometry3d se32Iso(const TooN::SE3<>& se3)
{
   Eigen::Isometry3d iso;
   TooN::Vector<3> t=se3.get_translation();
//   TooN::Matrix<3,3> r=se3.get_rotation().inverse().get_matrix();
   TooN::Matrix<3,3> r=se3.get_rotation().get_matrix();
   Matrix4d& m=iso.matrix();/*
   m(0,0)=r[0][0];m(0,1)=r[0][1];m(0,2)=r[0][2];
   m(1,0)=r[1][0];m(1,1)=r[1][1];m(1,2)=r[1][2];
   m(2,0)=r[2][0];m(2,1)=r[2][1];m(2,2)=r[2][2];*/
   m(0,0)=r[0][0];m(0,1)=r[0][1];m(0,2)=r[0][2];
   m(1,0)=r[1][0];m(1,1)=r[1][1];m(1,2)=r[1][2];
   m(2,0)=r[2][0];m(2,1)=r[2][1];m(2,2)=r[2][2];
   iso.translation()=Vector3d(t[0],t[1],t[2]);
   return iso;
}

TooN::SE3<> Iso2se3(const Eigen::Isometry3d iso)
{
   TooN::SE3<> se3;
   TooN::Matrix<3,3> r;
   Matrix4d m=iso.matrix();
   r[0][0]=m(0,0);r[0][1]=m(0,1);r[0][2]=m(0,2);
   r[1][0]=m(1,0);r[1][1]=m(1,1);r[1][2]=m(1,2);
   r[2][0]=m(2,0);r[2][1]=m(2,1);r[2][2]=m(2,2);
   se3.get_rotation()=r;
   se3.get_rotation()=se3.get_rotation();

   Vector3d t=iso.translation();
   se3.get_translation()=TooN::makeVector(t[0],t[1],t[2]);
   return se3;
}

void Optimizer::GPS_Fitting(vector<TooN::SE3<> >& frames,vector<TooN::Vector<3> >&gps_poses)
{
    if(frames.size()!=gps_poses.size()) return;
    //set camera poses
    for(int i=0;i<frames.size();i++)
    {
        //        set camera poses
        g2o::VertexSCam * v_cam
                = new g2o::VertexSCam();
        g2o::VertexSE3 * v_se3
                = v_cam;//new g2o::VertexSE3;
        {

            Eigen::Isometry3d pose;
            pose=se32Iso(frames[i]);
            v_se3->setId(id++);
            v_se3->setEstimate(pose);
            //      v_se3->setAll();            // set aux transforms
            optimizer.addVertex(v_se3);
            poses.push_back(v_se3);
        }

        //set the gps poses
        g2o::VertexPointXYZ* gps_point=new g2o::VertexPointXYZ;
        {
        TooN::Vector<3> a=gps_poses[i];
        //      gps_point->setEstimate(a+Vector3d(Sample::uniform(),Sample::uniform(),Sample::uniform()+10)/4);
        //      gps_point->setEstimate(a*2+Vector3d(1,4,10));
        gps_point->setEstimate(Vector3d(a[0],a[1],a[2]));
        gps_point->setId(id++);
        gps_point->setFixed(true);
        optimizer.addVertex(gps_point);
        gps_points.push_back(gps_point);
        }

        //set the edge of camera gps
        {
        g2o::EdgeSE3GPS*   edgeGps=new g2o::EdgeSE3GPS;
        edgeGps->vertices()[0] = optimizer.vertex(v_se3->id());

        edgeGps->vertices()[1] = optimizer.vertex(gps_point->id());

        edgeGps->setMeasurement(Vector3d(0,0,0));
        //e->inverseMeasurement() = -z;
        edgeGps->information() = Matrix3d::Identity();

        optimizer.addEdge(edgeGps);
        edge_gps.push_back(edgeGps);
        }

        //set the edge of odoment
        if(GV2.GetInt("UseOdoment",1)&&id>2)
        {
            g2o::EdgeSE3*  edge_se3=new g2o::EdgeSE3;
            edge_se3->vertices()[0]=optimizer.vertex(v_se3->id()-2);
            edge_se3->vertices()[1]=optimizer.vertex(v_se3->id());

//            TooN::SE3<> se3;
//            se3.get_translation()[2]=1;
            Eigen::Isometry3d step=poses[i]->estimate()*poses[i-1]->estimate().inverse();
//            cout<<step<<endl;
//            TooN::SE3<> se3=step.inverse()
            edge_se3->setMeasurement(step);
            edge_se3->information()=Eigen::Matrix<double,6,6>::Identity()*10000;
            optimizer.addEdge(edge_se3);
            odoment.push_back(edge_se3);
        }
    }

    optimizer.initializeOptimization();

    optimizer.setVerbose(true);
    compute(svar.GetInt("GPS_Fitting.Iter",10));
    for(int i=0;i<frames.size();i++)
    {
        g2o::VertexSE3* cam=poses[i];
        frames[i]=Iso2se3(cam->estimate());
    }
}

void Optimizer::GPS_Fitting()
{


    int pose_num=svar.GetInt("PoseNum",10);
    vector<TooN::SE3<> > frames;
    vector<TooN::Vector<3> > gps_poses;
    TooN::SE3<> se3;
    frames.push_back(se3);
    TooN::Matrix<3,3> r=TooN::Data(0.999,-0.0185854 ,-0.0381214,
                        0.0180004 ,0.999716 ,-0.0156316 ,
                        0.0384011 ,0.0149314, 0.999151);
    se3.get_rotation()=r;
    se3.get_translation()=TooN::makeVector(12.5351,-27.6328,24.2509);
    frames.push_back(se3);
    gps_poses.push_back(TooN::makeVector(-30.014 ,-20.051, 70.797));
    gps_poses.push_back(TooN::makeVector(-60.362, -40.433, 90.298));
    GPS_Fitting(frames,gps_poses);

    return ;
    for(int i=0;i<pose_num;i++)
    {
        TooN::SE3<> se3;
        se3.get_translation()[0]=i;
        frames.push_back(se3);
        gps_poses.push_back(TooN::makeVector(0,i,0));
    }
    GPS_Fitting(frames,gps_poses);
    return ;
    cout<<"entering GPS_Fitting...\n";
    Vector3d trans(0,0,0);

    Eigen:: Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d pose;
    pose = q;
    pose.translation() = trans;

    Eigen::Isometry3d step;
    step=q;
    step.translation()=Vector3d(1,0,0);

    for (size_t i=0; i<pose_num; ++i)
    {
//        set camera poses
                g2o::VertexSCam * v_cam
                    = new g2o::VertexSCam();
        g2o::VertexSE3 * v_se3
                = v_cam;//new g2o::VertexSE3;
        v_se3->setId(id);
        v_se3->setEstimate(pose);
        //      v_se3->setAll();            // set aux transforms

        pose=step*pose;

        optimizer.addVertex(v_se3);
        poses.push_back(v_se3);
        id++;

        //set the gps poses
        g2o::VertexPointXYZ* gps_point=new g2o::VertexPointXYZ;
        Vector3d a=pose.translation();
        //      gps_point->setEstimate(a+Vector3d(Sample::uniform(),Sample::uniform(),Sample::uniform()+10)/4);
        //      gps_point->setEstimate(a*2+Vector3d(1,4,10));
        gps_point->setEstimate(Vector3d(a[2],a[1],a[0])*2);
        gps_point->setId(id);
        gps_point->setFixed(true);
        optimizer.addVertex(gps_point);
        gps_points.push_back(gps_point);
        id++;

        //set the edge of camera gps
        g2o::EdgeSE3GPS*   edgeGps=new g2o::EdgeSE3GPS;
        edgeGps->vertices()[0] = optimizer.vertex(v_se3->id());

        edgeGps->vertices()[1] = optimizer.vertex(gps_point->id());

        edgeGps->setMeasurement(Vector3d(0,0,0));
        //e->inverseMeasurement() = -z;
        edgeGps->information() = Matrix3d::Identity();

        optimizer.addEdge(edgeGps);
        edge_gps.push_back(edgeGps);

        //set the edge of odoment
        if(GV2.GetInt("UseOdoment",0)&&id>2)
        {
            g2o::EdgeSE3*  edge_se3=new g2o::EdgeSE3;
            edge_se3->vertices()[0]=optimizer.vertex(v_se3->id()-2);
            edge_se3->vertices()[1]=optimizer.vertex(v_se3->id());
            edge_se3->setMeasurement(step);
            edge_se3->information()=Eigen::Matrix<double,6,6>::Identity()*1000;
            optimizer.addEdge(edge_se3);
            odoment.push_back(edge_se3);
        }
    }

    optimizer.initializeOptimization();

    optimizer.setVerbose(true);
}



void GPS_Fitting(vector<TooN::SE3<> >& trackPoses,vector<TooN::Vector<3> >& gps_poses)
{
    timer.enter("GPS_Fitting");
    Optimizer opt;
    opt.GPS_Fitting(trackPoses,gps_poses);
    timer.leave("GPS_Fitting");
}

}
