#ifndef __VIRTUALUAV_H__
#define __VIRTUALUAV_H__

#include <stdio.h>
#include <stdint.h>

#include <string>
#include <vector>
#include <map>

#include <mavlink/v1.0/common/mavlink.h>

#include <base/utils/utils.h>
#include <base/osa/osa++.h>
#include <hardware/UART/UART.h>
#include <hardware/Joystick/HAL_Joystick.h>


#include "UAS_types.h"
#include "FlightGear_Interface.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class VirtualUAV_Manager;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class VirtualUAV
{
public:
    VirtualUAV();
    virtual ~VirtualUAV();

    int init(void);
    int release(void);

public:
    uint8_t             ID, compID;             ///< mavlink ID number
    uint8_t             gcsID, gcsCompID;       ///< GCS ID & Component ID

    uint8_t             baseMode;               ///< base mode
    uint8_t             customMode;             ///< custom mode
    uint8_t             uavType;                ///< UAV type
    uint8_t             autopilotType;          ///< Autopilot type
    uint8_t             systemStatus;           ///< System Status
    uint8_t             mavlinkVersion;         ///< mavlink version
    int                 mavlinkChan;            ///< mavlink channel

    int                 severity;               ///< Severity of status. Relies on the definitions
                                                ///<    within RFC-5424. See enum MAV_SEVERITY.
    char                statusText[256];        ///< system status text

    float               cpuLoad;                ///< CPU load
    float               battVolt;               ///< battery voltage
    float               battCurrent;            ///< battery current
    float               battRemaining;          ///< remaining battery
    float               commDropRate;           ///< communication drop rate

    uint64_t            systimeUnix;            ///< Timestamp of the master clock in microseconds since UNIX epoch.
    uint32_t            bootTime;               ///< Timestamp of the component clock since boot time in milliseconds.
    pi::DateTime        dateTime;               ///< Date-time
    uint32_t            dateTime_bootTime;      ///< Date-time bootTime


    // navigation data
    uint64_t            gpsTime;                ///< GPS time (us)
    double              lat, lon, alt;          ///< GPS raw position
    double              HDOP_h, HDOP_v;         ///< HDOP
    double              gpsGroundSpeed;         ///< ground speed
    int                 gpsFixType;             ///< GPS fix type
    int                 nSat;                   ///< visable satellite

    double              gpLat, gpLon;           ///< fused position
    double              gpAlt, gpH;             ///< altitude / height
    double              gpVx, gpVy, gpVz;       ///< speed
    double              gpHeading;              ///< heading
    pi::DateTime        gpDateTime;             ///< fused position Date/time

    float               roll, pitch, yaw;       ///< attitude
    float               rollSpd,
                        pitchSpd,
                        yawSpd;

    double              homeLat, homeLng,       ///< Home position
                        homeAlt, homeH;         ///< Home altitude & height
    int                 homeSetCount;           ///< Home set count
    int                 homeSetted;             ///< Home position is setted
    int                 homePosReaded;          ///< Home position readed

    uint64_t            lastTM;
    double              lastAlt,
                        lastLat, lastLon;
    double              velV;                   ///< velocity in vertical
    double              velH;                   ///< velocity in horiz
    double              dis, disLOS;            ///< distance

    // raw sensor data
    int                 Ax, Ay, Az;
    int                 Gx, Gy, Gz;
    int                 Mx, My, Mz;
    int                 Ax_raw, Ay_raw, Az_raw;
    int                 Gx_raw, Gy_raw, Gz_raw;
    int                 Mx_raw, My_raw, Mz_raw;

    // RC data
    int                 rcRSSI;
    int                 rcRaw[8];
    int                 rcAll[16];
    int                 rcRaw_port;
    int                 rcAll_channels;

    // Stream request frequence
    int                 m_bStreamRequested;     ///< stream frequency has setted
    int                 *m_frqStream[16];       ///< request stream frequency


    VirtualUAV_Manager  *m_vuavManager;         ///< Virtual UAV manager

public:
    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int sendMavlinkMsg(mavlink_message_t *msg);
    virtual int generateMavlinkMsg(uint8_t msgID, mavlink_message_t *msg);

    virtual int timerFunction(void *arg);

    virtual int simulation(pi::JS_Val *jsv);
    virtual int toFlightGear(FGNetFDM *fgData);

    int initUAV(double _lat, double _lng, double _alt, double _H);

    int updatePOS(double _lat, double _lng, double _alt, double _H,
                  double _yaw, double _pitch, double _roll);
    int updateTime(double tsNow);

protected:
    uint64_t        m_timerCount;               ///< timer counter
    double          m_tmStart;                  ///< start time (boot)

    double          sim_m ;                     ///< UAV's weight (kg)
    double          sim_tLast, sim_tNow;        ///< timestamp last, now

    double          sim_Ax, sim_Ay, sim_Az;     ///< accelate
    double          sim_Vx, sim_Vy, sim_Vz;     ///< velocity
    double          sim_totalThrust;            ///< total thrust

    FGNetFDM        m_flightGearData;           ///< flightgear data
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef std::vector<VirtualUAV*>    VUAV_Array;
typedef std::map<int, VirtualUAV*>  VUAV_IDMap;


class VirtualUAV_Manager
{
public:
    VirtualUAV_Manager();
    virtual ~VirtualUAV_Manager();

    virtual void init(void);
    virtual void release(void);
    virtual void reset(void);

    int startTimer(void);
    int stopTimer(void);
    virtual int timerFunction(void *arg);

    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int sendMavlinkMsg(mavlink_message_t *msg, VirtualUAV *u);

    int addUAV(VirtualUAV *u);
    int removeUAV(int id);

    int         getUAVs(VUAV_Array &uavArr);
    VirtualUAV* getUAV(int id);
    int         setActiveUAV(int id);
    VirtualUAV* getActiveUAV(void);


    int writeMsgBuf(uint8_t *buf, int len);
    int readMsgBuf(uint8_t *buf, int *len);

    void setJoystick(pi::HAL_JoyStick *js) { m_joystick = js; }
    void setUART(pi::VirtualUART *uart) { m_vUART = uart; }

protected:
    VUAV_IDMap              m_mapUAV;

    VirtualUAV              *m_activeUAV;

    pi::OSA_HANDLE          m_timer;
    uint64_t                m_timerCount;
    pi::RMutex              m_mutexMsgWrite;
    std::vector<uint8_t>    m_msgBuffer;

    int                     m_uartBufMaxSize;
    uint8_t                 *m_uartBuf;

    int                     mavlinkChan;            ///< mavlink channel

    pi::HAL_JoyStick        *m_joystick;            ///< joystick
    pi::VirtualUART         *m_vUART;               ///< virtual UART
};


#endif // end of __VIRTUALUAV_H__
