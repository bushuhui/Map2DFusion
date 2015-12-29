#ifndef __HAL_JOYSTICK_H__
#define __HAL_JOYSTICK_H__

#include "base/osa/osa++.h"


namespace pi {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#define JS_AXIS_MAX_NUM     15
#define JS_BUTTON_MAX_NUM   30

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct JS_Val {
    float       AXIS[JS_AXIS_MAX_NUM];
    double      BUTTON[JS_BUTTON_MAX_NUM ];
    int         dataUpdated;
};

typedef struct _axes_t {
    int x;
    int y;
} AXES_T;



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class HAL_JoyStick: public RThread
{
public:
    HAL_JoyStick() {
        m_devID = -1;
        m_bOpened = 0;
    }
    HAL_JoyStick(int dev) {
        m_devID = dev;
        m_bOpened = 0;
    }
    virtual ~HAL_JoyStick() {
        close();
    }

    virtual int thread_func(void *arg);

    int open(int devID = 0);
    int close(void);
    int read(JS_Val *jsv);

public:
    int         m_devType;                  // 0:joystick 1:control
    int         m_devID;
    int         m_devFD;
    char        number_of_axes;
    char        number_of_btns;

protected:
    RMutex      m_mutex;
    JS_Val      m_JSVal;
    int         m_bOpened;
};

} // end of namespace pi

#endif // end of __HAL_JOYSTICK_H__
