#ifndef EDGESE3GPSXYZ_H
#define EDGESE3GPSXYZ_H

#include <Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h>

namespace g2o {

class G2O_TYPES_SBA_API EdgeSE3GPSXYZ: public  BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3GPSXYZ(){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
        const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
        Vector3d obs(_measurement);
        _error = obs+v2->estimate()-v1->estimate().inverse().translation();
    }

    bool ignoreThis;
};


}
#endif // EDGESE3GPSXYZ_H
