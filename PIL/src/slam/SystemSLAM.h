#ifndef __SYSTEMSLAM_H__
#define __SYSTEMSLAM_H__

#include <string>

namespace pi {


///
/// \brief The SLAM system interface
///
class SystemSLAM
{
public:
    SystemSLAM() {}
    virtual ~SystemSLAM() {}

    ///
    /// \brief begin the SLAM system
    /// \return
    ///
    virtual int run(void) = 0;

    ///
    /// \brief stop the SLAM system
    /// \return
    ///
    virtual int stop(void) = 0;

    ///
    /// \brief reset SLAM system to initial state
    /// \return
    ///
    virtual int reset(void) = 0;

    virtual int saveKeyFrameData(const std::string &fname) { return -1; }
};

} // end of namespace pi

#endif // end of __SYSTEMSLAM_H__
