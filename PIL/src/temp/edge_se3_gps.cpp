#include "edge_se3_gps.h"

namespace g2o {

  EdgeSE3GPS::EdgeSE3GPS() :
    BaseBinaryEdge<3, Vector3D,VertexSE3, VertexPointXYZ>()
  {
    _information.setIdentity();
    _error.setZero();
  }

  bool EdgeSE3GPS::read(std::istream& is)
  {
    Vector3D p;
    is >> p[0] >> p[1] >> p[2];
    setMeasurement(p);
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        is >> information()(i, j);
        if (i != j)
          information()(j, i) = information()(i, j);
      }
    return true;
  }

  bool EdgeSE3GPS::write(std::ostream& os) const
  {
    Vector3D p = measurement();
    os << p.x() << " " << p.y() << " " << p.z();
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        os << " " << information()(i, j);
    return os.good();
  }


//#ifndef NUMERIC_JACOBIAN_THREE_D_TYPES
//  void EdgeSE3GPS::linearizeOplus()
//  {
////    _jacobianOplusXi=-Matrix3D::Identity();
//    _jacobianOplusXj= Matrix3D::Identity();
//  }
//#endif

} // end namespace
