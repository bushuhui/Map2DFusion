#ifndef __UAS_H__
#define __UAS_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <map>
#include <list>
#include <string>
#include <tr1/functional>

#include <base/time/DateTime.h>
#include <hardware/Gps/POS_reader.h>

#include "UAS_types.h"
#include "utils_mavlink.h"



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class UAS_Manager;

// Mavlink message call back function
typedef std::tr1::function<int (mavlink_message_t*)>    Mavlink_Message_Handle;

typedef std::map<std::string, std::string> String2StringMap;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class UAS_Base
{
public:
    UAS_Base();
    virtual ~UAS_Base();

    virtual void init(void);
    virtual void release(void);

public:
    // basic information
    UAS_Type            uasType;                ///< UAS type


    uint8_t             ID, compID;             ///< mavlink ID number
    uint8_t             gcsID, gcsCompID;       ///< GCS ID & Component ID

    uint8_t             baseMode;               ///< base mode
    uint8_t             customMode;             ///< custom mode (flight mode)
    string              szFlightMode;           ///< flight mode string
    uint8_t             motorArmed;             ///< motor armed or not
    string              szMotorArmed;           ///< motor armed (string)

    uint8_t             uavType;                ///< UAV type
    uint8_t             autopilotType;          ///< Autopilot type
    uint8_t             systemStatus;           ///< System Status
    string              szSystemStatus;         ///< System Status string
    uint8_t             mavlinkVersion;         ///< mavlink version
    int                 mavlinkChan;            ///< mavlink channel

    int                 severity;               ///< Severity of status. Relies on the definitions within RFC-5424. See enum MAV_SEVERITY.
    char                statusText[256];        ///< system status text

    float               cpuLoad;                ///< CPU load
    float               battVolt;               ///< battery voltage
    float               battCurrent;            ///< battery current
    float               battRemaining;          ///< remaining battery
    float               commDropRate;           ///< communication drop rate

    uint64_t            systimeUnix;            ///< Timestamp of the master clock in microseconds since UNIX epoch.
    uint64_t            bootTime;               ///< Timestamp of the component clock since boot time in milliseconds.
    pi::DateTime        dateTime;               ///< Date-time
    uint64_t            dateTime_bootTime;      ///< Date-time bootTime


    // navigation data
    uint64_t            gpsTime;                ///< GPS time
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

    // parameters
    AP_ParamArray       m_paramArray;           ///< MAV parameter array

    // waypoints
    AP_WPArray          m_waypoints;            ///< waypoints
    int                 currMission;            ///< current mission

    // Stream request frequence
    int                 m_bStreamRequested;     ///< stream frequency has setted
    int                 *m_frqStream[16];       ///< request stream frequency


    // UAS manager
    UAS_Manager         *m_uasManager;

    // POS data manager
    pi::POS_DataManager m_posData;

    // Mavlink message handle list
    typedef std::map<std::string, Mavlink_Message_Handle> MavlinkMessageHandleMap;
    MavlinkMessageHandleMap     m_mavlinkMsgHandleMap;

public:
    int link_connected(void);
    int clear_home(void);

    int requireParameters(int clearOld=0);
    int requireParameter(int index);
    int requireParameter(char *id);

    int updateParameters(void);
    int updateParameter(int index);
    int updateParameter(char *id);


    int writeWaypoints(AP_WPArray &wpa);
    int readWaypoints(void);
    int clearWaypoints(void);
    int setCurrentWaypoint(int idx);

    int writeWaypointsNum(void);
    int readWaypointsNum(void);

    int getStreamFrequency(MAV_DATA_STREAM streamID);
    int setStreamFrequency(MAV_DATA_STREAM streamID, int freq=-1);


    int executeCommand(MAV_CMD command, int confirmation,
                       float param1, float param2, float param3, float param4, float param5, float param6, float param7,
                       int component);
    int executeCommandAck(int num, bool success);

    int _MAV_CMD_DO_SET_SERVO(int ch, int pwm);


    int registMavlinkMessageHandle(const std::string &handleName, Mavlink_Message_Handle &msgHandle);
    int unregistMavlinkMessageHandle(const std::string &handleName);

    virtual int startTimer(void);
    virtual int stopTimer(void);

    virtual int gen_listmap_important(String2StringMap &lm);
    virtual int gen_listmap_all(String2StringMap &lm);

    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int sendMavlinkMsg(mavlink_message_t *msg);
    virtual int timerFunction(void *arg);

protected:
    virtual int setPosData(void);

protected:
    pi::OSA_HANDLE                  m_timer;
    pi::RMutex                      m_mutexMessageHandle;

    uint64_t                        m_timerCount;
    int                             m_connectTime;
    int                             m_paramAutoLoaded;

    int                             m_bLinkConnected;                   /// datalink ok?
    int                             m_pkgLost, m_pkgLastID;
    int                             m_recvMessageInSec;
    int                             m_statusMsgTime;

    int                             m_bCmdConfirmed;
    char                            m_sLastCommand[256];

    ValueAverager<float>            m_avgBatV;

    int                             m_tsValid_Year;
    uint64_t                        m_tsLastPOS;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class UAS_MAV : public UAS_Base
{
public:
    UAS_MAV();
    virtual ~UAS_MAV();

    virtual void init(void);
    virtual void release(void);

    virtual int moveTo(double Lat, double Lng);
    virtual int changeHeight(double newH);

    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int gen_listmap_important(String2StringMap &lm);
    virtual int gen_listmap_all(String2StringMap &lm);

    virtual int timerFunction(void *arg);
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class UAS_GCS : public UAS_Base
{
public:
    UAS_GCS();
    virtual ~UAS_GCS();

    virtual void init(void);
    virtual void release(void);

    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int gen_listmap_important(String2StringMap &lm);
    virtual int gen_listmap_all(String2StringMap &lm);

    virtual int timerFunction(void *arg);


    int send_ATA_yawOffset(void);

public:
    double          ataYawOffset;
    int             ataYawOffset_setted;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class UAS_Telem : public UAS_Base
{
public:
    UAS_Telem();
    virtual ~UAS_Telem();

    virtual void init(void);
    virtual void release(void);

    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int gen_listmap_important(String2StringMap &lm);
    virtual int gen_listmap_all(String2StringMap &lm);

    virtual int timerFunction(void *arg);

public:
    // Telemetry data
    int                 radioRSSI;
    int                 radioRSSI_remote;

    float               radioRSSI_per, radioRSSI_remote_per;

    int                 radioRX_errors;
    int                 radioFixed;
    int                 radioTXBuf;
    int                 radioNoise;
    int                 radioNoise_remote;
};



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef std::vector<UAS_Base*>   UAS_Array;
typedef std::map<int, UAS_Base*> UAS_IDMap;

class UAS_Manager
{
public:
    UAS_Manager();
    virtual ~UAS_Manager();

    virtual void init(void);
    virtual void release(void);
    virtual void reset(void);

    int startTimer(void);
    int stopTimer(void);
    virtual int timerFunction(void *arg);

    virtual int parseMavlinkMsg(mavlink_message_t *msg);
    virtual int sendMavlinkMsg(mavlink_message_t *msg);

    virtual int gen_listmap_important(String2StringMap &lm);
    virtual int gen_listmap_all(String2StringMap &lm);

    UAS_Array*      get_uas(void);
    UAS_Base*       get_uas(int id);
    int             get_uas(UAS_Type t, UAS_Array &uasArr);

    int             get_mav(UAS_Array &uasArr);
    int             get_gcs(UAS_Array &uasArr);
    int             get_telem(UAS_Array &uasArr);

    UAS_MAV*        get_active_mav(void);
    UAS_GCS*        get_active_gcs(void);
    UAS_Telem*      get_active_telem(void);

    UAS_MAV*        set_active_mav(int id);

    int put_msg_buff(uint8_t *buf, int len);
    int get_msg_buff(uint8_t *buf, int *len);

protected:
    UAS_Array               m_arrUAS;
    UAS_IDMap               m_mapUAS;

    pi::OSA_HANDLE          m_timer;
    uint64_t                m_timerCount;
    pi::RMutex              *m_mutexMsgWrite;
    std::vector<uint8_t>    m_msgBuffer;

    float                   bpsIn, bpsOut;
    uint64_t                m_nReadLast, m_nWriteLast, m_tLast;

    UAS_Base                *m_activeMAV;
    UAS_Base                *m_activeGCS;
    UAS_Base                *m_activeTelem;

public:
    uint8_t                 gcsID, gcsCompID;       ///< GCS ID & Component ID
    int                     mavlinkChan;            ///< mavlink channel
};


#endif // end of __UAS_H__
