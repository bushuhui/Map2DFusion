#ifndef __UAS_TYPES_H__
#define __UAS_TYPES_H__


#include <stdio.h>
#include <stdint.h>

#include <string>
#include <vector>
#include <map>
#include <deque>

#include <mavlink/v1.0/common/mavlink.h>
#include <mavlink/v1.0/ardupilotmega/ardupilotmega.h>

#include <base/utils/utils.h>
#include <base/osa/osa++.h>


////////////////////////////////////////////////////////////////////////////////
/// predefine class
////////////////////////////////////////////////////////////////////////////////
class UAS_Base;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief The UAS Type enum
 *
 *  FIXME: parse MAVLINK message based on sysid
 *      0 ~  49: MAV
 *     50 ~ 249: Telemetry
 *    250 ~ 255: GCS
 */
enum UAS_Type
{
    UAS_TYPE_MAV     = 0,
    UAS_TYPE_GCS     = 1,
    UAS_TYPE_TELEM   = 2
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class AP_MessageQueue
{
public:
    AP_MessageQueue();
    ~AP_MessageQueue();

    int push(mavlink_message_t &m);
    int pop(mavlink_message_t &m);
    int size(void);

protected:
    std::deque<mavlink_message_t>   m_msgQueue;
    pi::RMutex                      *m_mutex;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct AP_ParamItem
{
public:
    int             index;                      ///< index number
    char            id[17];                     ///< ID name
    MAV_PARAM_TYPE  type;                       ///< value type
    float           value;                      ///< value

    int             modified;                   ///< modified flag

public:
    int8_t      toInt8(void);
    uint8_t     toUint8(void);
    int16_t     toInt16(void);
    uint16_t    toUint16(void);
    int32_t     toInt32(void);
    uint32_t    toUint32(void);
    float       toFloat(void);

    void        fromInt8(int8_t v);
    void        fromUint8(uint8_t v);
    void        fromInt16(int16_t v);
    void        fromUint16(uint16_t v);
    void        fromInt32(int32_t v);
    void        fromUint32(uint32_t v);
    void        fromFloat(float v);

public:
    AP_ParamItem();
    virtual ~AP_ParamItem();

    void init();
    void release();
};

typedef std::map<int, AP_ParamItem*>            AP_ParamIndexMap;
typedef std::map<std::string, AP_ParamItem*>    AP_ParamIDMap;
typedef std::vector<AP_ParamItem*>              AP_ParamVector;


class AP_ParamArray
{
public:
    enum PARAM_RW_STATUS {
        IDLE,                           ///< idle state
        READING,                        ///< read single item
        READING_ALL,                    ///< read all items one by one
        READING_BATCH,                  ///< batch read all items (UAS auto send parameters)
        WRITING,                        ///< write single item
        WRITING_ALL                     ///< write all items
    };

public:
    AP_ParamArray();
    ~AP_ParamArray();

    void init(void);
    void release(void);

    int size(void);
    int reserve(int n);
    int clear(void);

    int lock(void);
    int unlock(void);

    int requireParameters(int clearOld = 0);
    int requireParameter(int index);
    int requireParameter(char *id);
    int _requireParameter(int index);
    int _requireParameter(char *id);

    int updateParameters(void);
    int updateParameters(pi::StringArray &lstParams);
    int updateParameter(int index);
    int updateParameter(char *id);
    int _updateParameter(AP_ParamItem *pi);

    int set(AP_ParamArray &pa);
    int set(AP_ParamItem &item);
    AP_ParamItem* get(int idx);
    AP_ParamItem* get(std::string id);

    AP_ParamVector* get_allParam(void);
    AP_ParamIndexMap* get_paramIndexMap(void);

    int set_paramN(int nParam);
    int get_paramN(void);
    int get_currIdx(void);

    int collectUnreadedItems(void);
    int getUnreadedItems(int n, std::vector<int> &lst);

    int isLoaded(void);
    int setLoaded(int bl);

    int setStatus(PARAM_RW_STATUS st);
    PARAM_RW_STATUS getStatus(int &nParamRW, int &idxCurr);
    PARAM_RW_STATUS getStatus(void);

    int get_tm_lastRead(uint64_t &t);

    int save(std::string fname);
    int load(std::string fname);

    virtual int timerFunction(void *arg);
    virtual int parseMavlinkMsg(mavlink_message_t *msg);

    void setUAS(UAS_Base *u) { m_uas = u; }

protected:
    virtual int startTimer(void);
    virtual int stopTimer(void);

protected:
    AP_ParamVector              m_lstAll;
    AP_ParamIndexMap            m_mapIndex;
    AP_ParamIDMap               m_mapID;
    AP_ParamVector              m_lstWrite;
    std::deque<int>             m_lstRead;

    int                         m_nParam;               ///< parameter total number
    int                         m_nParamRW;             ///< parameter number for r/w
    int                         m_idxReading;           ///< parameter index for reading
    int                         m_idxWritting;          ///< parameter index for writting
    int                         m_idxCurrent;           ///< current r/w index
    int                         m_bLoaded;              ///< full parameters loaded or not

    PARAM_RW_STATUS             m_stParamReading;       ///< parameter reading status
    uint64_t                    m_tLastReading;         ///< last read/write time

    pi::RMutex                  *m_mutex;               ///< reading mutex
    pi::OSA_HANDLE              m_timer;                ///< timer

    UAS_Base                    *m_uas;                 ///< UAS
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct AP_Mission_CMD_Types
{
    MAV_CMD     cmd;
    char        cmd_name[64];
};

int     AP_Mission_CMD_getAll(AP_Mission_CMD_Types *allTypes);
char*   AP_Mission_CMD_cmd(MAV_CMD cmd);
MAV_CMD AP_Mission_CMD_cmd(char *cmd);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


class AP_WayPoint
{
public:
    AP_WayPoint() { init(); }
    ~AP_WayPoint() { release(); }

    void init(void);
    void release(void);

    void set(double lat_, double lng_, double alt_);
    void setPos(double lat_, double lng_);
    void setAlt(double alt_);
    void setHeading(double heading_);

    void get(double &lat_, double &lng_, double &alt_);
    void getPos(double &lat_, double &lng_);
    void getAlt(double &alt_);
    void getHeading(double &heading_);

    int to_mission_item(mavlink_mission_item_t *mi);
    int from_mission_item(mavlink_mission_item_t *mi);

    void setWriteConfirm(int c) { writeConfirmed = c; }
    int  getWriteConfirm(void) { return writeConfirmed; }

    int compare(AP_WayPoint *wp);

public:
    int         idx;                    ///< index number

    MAV_CMD     cmd;                    ///< mission command

    double      lat, lng;               ///< latitdue & longtitude
    double      alt;                    ///< altitude from home
    double      heading;                ///< heading (0:North, 90: Est, 180: South, 270: West)

    float       param1;                 ///< PARAM1, see MAV_CMD enum
    float       param2;                 ///< PARAM2, see MAV_CMD enum
    float       param3;                 ///< PARAM3, see MAV_CMD enum
    float       param4;                 ///< PARAM4, see MAV_CMD enum

    int         current;                ///< current mission (1:True, 0:False)

    MAV_FRAME   frame;                  ///< The coordinate system of the MISSION. see MAV_FRAME in mavlink_types.h
    uint8_t     autocontinue;           ///< autocontinue to next wp

    int         writeConfirmed;         ///< write confirmed
};


typedef std::map<int, AP_WayPoint*>     AP_WayPointMap;
typedef std::vector<AP_WayPoint*>       AP_WayPointVector;


class AP_WPArray
{
public:
    enum WP_RW_STATUS {
        IDLE,                           ///< idle state

        READING_NUM,                    ///< request mission number
        READING_ITEM,                   ///< reading item

        WRITTING_NUM,                   ///< write mission number
        WRITTING_ITEM,                  ///< write item
        WRITTING_CONFIRM_NUM,           ///< confirm number
        WRITTING_CONFIRM_ITEM,          ///< confirm item

        CLEAR_WP                        ///< clear waypoints
    };

public:
    AP_WPArray();
    ~AP_WPArray();

    void init(void);
    void release(void);

    int set(AP_WPArray *wpa);
    int set(AP_WayPoint &wp);
    int get(AP_WayPoint &wp);
    AP_WayPoint* get(int idx);
    int getAll(AP_WayPointVector &wps);
    AP_WayPointMap* getAll();

    int remove(int idx);
    int size(void);
    int clear(void);

    int save(std::string fname);
    int load(std::string fname);

    void setUAS(UAS_Base *u) { m_uas = u; }
    UAS_Base* getUAS(void) { return m_uas; }

    int writeWaypoints(void);
    int readWaypoints(void);
    int writeWaypointsNum(void);
    int readWaypointsNum(void);

    int clearWaypoints(void);
    int setCurrentWaypoint(int idx);
    int getCurrentWaypoint(void) { return m_currWaypoint; }

    WP_RW_STATUS getStatus(int &iRW, int &nRW) {
        iRW = m_iRW;
        nRW = m_nRW;
        return m_status;
    }

    virtual int timerFunction(void *arg);
    virtual int parseMavlinkMsg(mavlink_message_t *msg);

protected:
    virtual int startTimer(void);
    virtual int stopTimer(void);

    int _readWaypointsNum(void);
    int _readWaypoint(int idx);
    int _writeWaypoint(int idx);

protected:
    AP_WayPointMap      m_arrWP;        ///< waypoints map
    pi::OSA_HANDLE      m_timer;        ///< timer

    UAS_Base            *m_uas;         ///< UAS obj

    int                 m_iRW;          ///< current R/W waypoint
    int                 m_nRW;          ///< readed/written waypoints
    WP_RW_STATUS        m_status;       ///< status
    uint64_t            m_tLast;        ///< Last R/W timestamp (in ms)
    int                 m_currWaypoint; ///< current mission
    int                 m_nReadTry;     ///< Read try number
};

#endif // end of __UAS_TYPES_H__
