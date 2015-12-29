#include "TypeSE3_GPS.h"
#include <hardware/Gps/PathTable.h>

namespace g2o {


VertexSBAGPSSIM3Time::VertexSBAGPSSIM3Time()
    : BaseVertex<8,SIM3WithTime>(),gps(NULL),fixTime(false),fixScale(false)
{

}

bool VertexSBAGPSSIM3Time::read(std::istream& is)
{
    Vector7d cam2world;
    double   time;
    is >>time;
    for (int i=0; i<6; i++){
      is >> cam2world[i];
    }
    is >> cam2world[6];

    setEstimate(SIM3WithTime(Sim3(cam2world).inverse(),time));
    return true;
}

bool VertexSBAGPSSIM3Time::write(std::ostream& os) const
{
    Sim3 cam2world(estimate().sim3.inverse());
    Vector7d lv=cam2world.log();
    os<<estimate().time;
    for (int i=0; i<7; i++){
      os << lv[i] << " ";
    }

    return os.good();
}

void EdgeSE3_GPS::computeError()
{
    calledCount++;
    using namespace std;
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAGPSSIM3Time* v2 = static_cast<const VertexSBAGPSSIM3Time*>(_vertices[0]);

    SIM3WithTime est_gps=v2->estimate();
    pi::Point3d gps_pose;
    if(v2->gps->Get(frameTimestamp+est_gps.time,gps_pose))
    {
//        cout<<"GPSDiff:"<<est_gps.time<<",";
//        cout<<"Error of Edge "<<frameTimestamp<<" is "<<error()<<endl;
        _error=(*(Vector3d*)&gps_pose)-est_gps.sim3*v1->estimate().inverse().translation();
    }
    else
    {
        cout<<"GPSDiff:"<<est_gps.time<<",";
        cout<<"Error of Edge "<<frameTimestamp<<" is "<<error()<<endl;
        _error=Vector3d(10,10,10);
    }
}

bool EdgeSE3_GPS::read(std::istream& is){
    is>>frameTimestamp;

    for (int i=0; i<3; i++){
        is >> _measurement[i];
    }
    for (int i=0; i<3; i++)
        for (int j=i; j<3; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
        }
    return true;
}

bool EdgeSE3_GPS::write(std::ostream& os) const {

    os<<frameTimestamp;

    for (int i=0; i<3; i++){
        os << measurement()[i] << " ";
    }

    for (int i=0; i<3; i++)
        for (int j=i; j<3; j++){
            os << " " <<  information()(i,j);
        }
    return os.good();
}

}
