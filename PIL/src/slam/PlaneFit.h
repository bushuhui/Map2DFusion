#ifndef PLANEFIT_H
#define PLANEFIT_H

#include <base/types/SE3.h>

namespace pi{

class Plane_Fit
{
public:
    static int fitPlaneRansac(std::vector<pi::Point3d>& points,pi::SE3d& plane,
                              std::vector<int>& outliers,double thresholdZ);
};

}
#endif // PLANEFIT_H
