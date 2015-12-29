#ifndef TYPESE3_GPS_H
#define TYPESE3_GPS_H

#include <Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/types/sim3/sim3.h>
#include <base/Svar/Svar.h>

class FastPathTable;

namespace g2o {
typedef Matrix<double, 8, 1>  Vector8d;

struct SIM3WithTime
{
    SIM3WithTime():sim3(Sim3()),time(0){}
    SIM3WithTime(const Sim3& s,const double& t):sim3(s),time(t){}

    Sim3 sim3;
    double time;
};

/**
 * \brief GPS vertex, estimate is sim3 and timeDiff betweeen videoTimestamp to gpsTimestamp
 * (x,y,z,qw,qx,qy,qz,timeDiff)
 */
 class G2O_TYPES_SBA_API VertexSBAGPSSIM3Time : public BaseVertex<8, SIM3WithTime>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSBAGPSSIM3Time();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
      _estimate=SIM3WithTime();
    }

    virtual void oplusImpl(const double* update_)
    {
        using namespace std;
        Eigen::Map<Vector8d> update(const_cast<double*>(update_));
        Eigen::Map<Vector7d> sim3(const_cast<double*>(update_));

        if(fixScale)
            sim3[6]   = 0;
        if(fixTime)
            update[7] = 0;
//        if(update[7])
//            cout<<"Time update:"<<update[7]<<endl;
        Sim3 s(sim3);
        setEstimate(SIM3WithTime(s*estimate().sim3,estimate().time+update[7]*1e6));
    }

//    virtual bool setEstimateDataImpl(const double* est){
//      Map<const Vector8d> _est(est);
//      _estimate = _est;
//      return true;
//    }

//    virtual bool getEstimateData(double* est) const{
//      Map<Vector8d> _est(est);
//      _est = _estimate;
//      return true;
//    }

    virtual int estimateDimension() const {
      return 8;
    }

    FastPathTable*        gps;
    bool fixTime,fixScale;
};

class G2O_TYPES_SBA_API EdgeSE3_GPS: public  BaseBinaryEdge<3, Vector3d, VertexSBAGPSSIM3Time, VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3_GPS():calledCount(0){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    virtual void computeError() ;

    double frameTimestamp;
    int    calledCount;
};


}
#endif
