#ifndef __VIRTUALUAV_QUAD_H__
#define __VIRTUALUAV_QUAD_H__


#include "VirtualUAV.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class VirtualUAV_Quad : public VirtualUAV
{
public:
    VirtualUAV_Quad();
    virtual ~VirtualUAV_Quad();

    int init();
    int release();

    virtual int timerFunction(void *arg);
    virtual int simulation(pi::JS_Val *jsv);
    virtual int toFlightGear(FGNetFDM *fgData);

protected:
    double          sim_dragK;
};


#endif // end of __VIRTUALUAV_QUAD_H__
