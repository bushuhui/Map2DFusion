#ifndef GPS_FIT_H
#define GPS_FIT_H
#include <TooN/se3.h>
#include <vector>
void GPS_Fitting(std::vector<TooN::SE3<> >& trackPoses,std::vector<TooN::Vector<3> >& gps_poses);
#endif // GPS_FIT_H
