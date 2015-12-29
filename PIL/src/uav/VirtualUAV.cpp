
#include "VirtualUAV.h"

using namespace std;
using namespace pi;



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

VirtualUAV::VirtualUAV()
{
    init();
}

VirtualUAV::~VirtualUAV()
{
    release();
}

int VirtualUAV::init(void)
{
    m_timerCount    = 0;
    m_tmStart       = tm_getTimeStamp();

    // system information
    ID              = 1;
    compID          = 1;

    gcsID           = 254;
    gcsCompID       = 1;

    customMode      = 0;
    baseMode        = 0;
    systemStatus    = 0;
    mavlinkChan     = 0;

    uavType         = MAV_TYPE_QUADROTOR;

    cpuLoad         = 0;
    battVolt        = 0;
    battCurrent     = 0;
    battRemaining   = 0;
    commDropRate    = 0;

    systimeUnix     = 0;
    bootTime        = 0;
    dateTime.toLocalTime();
    dateTime_bootTime = 0;

    severity        = 1;
    statusText[0]   = 0;

    // navigation
    gpsTime         = 0;
    lat             = -9999;
    lon             = -9999;
    alt             = -0;
    HDOP_h          = 9999;
    HDOP_v          = 9999;
    gpsGroundSpeed  = 0;
    gpsFixType      = 0;
    nSat            = 0;

    gpLat           = -9999;
    gpLon           = -9999;
    gpAlt           = 0;
    gpH             = 0;
    gpVx            = 0;
    gpVy            = 0;
    gpVz            = 0;
    gpHeading       = 0;
    gpDateTime.toLocalTime();

    lastTM          = 0;
    lastAlt         = 0;
    lastLat         = 0;
    lastLon         = 0;
    velV            = 0.0;
    velH            = 0.0;

    dis             = 0.0;
    disLOS          = 0.0;

    homeLat         = -9999;
    homeLng         = -9999;
    homeAlt         = 0;
    homeH           = 0;
    homeSetCount    = 10;
    homeSetted      = 0;
    homePosReaded   = 0;

    // raw sensor
    Ax              = 0;
    Ay              = 0;
    Az              = 0;
    Gx              = 0;
    Gy              = 0;
    Gz              = 0;
    Mx              = 0;
    My              = 0;
    Mz              = 0;

    Ax_raw          = 0;
    Ay_raw          = 0;
    Az_raw          = 0;
    Gx_raw          = 0;
    Gy_raw          = 0;
    Gz_raw          = 0;
    Mx_raw          = 0;
    My_raw          = 0;
    Mz_raw          = 0;

    // attitude
    roll            = 0;
    pitch           = 0;
    yaw             = 0;
    rollSpd         = 0;
    pitchSpd        = 0;
    yawSpd          = 0;

    return 0;
}

int VirtualUAV::release(void)
{
    return 0;
}

int VirtualUAV::parseMavlinkMsg(mavlink_message_t *msg)
{
    return 0;
}

int VirtualUAV::sendMavlinkMsg(mavlink_message_t *msg)
{
    return m_vuavManager->sendMavlinkMsg(msg, this);
}

int VirtualUAV::generateMavlinkMsg(uint8_t msgID, mavlink_message_t *msg)
{
    int ret = -1;

    switch(msgID) {
    case MAVLINK_MSG_ID_HEARTBEAT:
    {
        mavlink_heartbeat_t msg_hb;

        msg->sysid              = ID;
        msg->compid             = compID;

        msg_hb.custom_mode      = customMode;
        msg_hb.type             = uavType;
        msg_hb.autopilot        = autopilotType;
        msg_hb.base_mode        = baseMode;
        msg_hb.system_status    = systemStatus;
        msg_hb.mavlink_version  = mavlinkVersion;

        mavlink_msg_heartbeat_encode(ID, compID, msg, &msg_hb);
        ret = 0;
        break;
    }

    case MAVLINK_MSG_ID_SYS_STATUS:
    {
        mavlink_sys_status_t msg_ss;

        msg_ss.load             = cpuLoad * 10.0;
        msg_ss.voltage_battery  = battVolt * 1000.0;
        msg_ss.current_battery  = battCurrent * 100.0;
        msg_ss.battery_remaining= battRemaining;
        msg_ss.drop_rate_comm   = commDropRate * 100.0;

        mavlink_msg_sys_status_encode(ID, compID, msg, &msg_ss);
        ret = 0;
        break;
    }

    case MAVLINK_MSG_ID_STATUSTEXT:
    {
        mavlink_statustext_t st;

        if( strlen(statusText) == 0 ) return -1;

        st.severity = severity;
        strcpy(st.text, statusText);

        mavlink_msg_statustext_encode(ID, compID, msg, &st);
        ret = 0;
        break;
    }

    case MAVLINK_MSG_ID_SYSTEM_TIME:
    {
        mavlink_system_time_t msg_st;

        msg_st.time_unix_usec   = systimeUnix;
        msg_st.time_boot_ms     = bootTime;

        mavlink_msg_system_time_encode(ID, compID, msg, &msg_st);
        ret = 0;
        break;
    }


    case MAVLINK_MSG_ID_GPS_RAW_INT:
    {
        mavlink_gps_raw_int_t msg_gps_raw;

        msg_gps_raw.time_usec               = gpsTime;
        msg_gps_raw.lat                     = lat * 1e7;
        msg_gps_raw.lon                     = lon * 1e7;
        msg_gps_raw.alt                     = alt * 1000.0;
        msg_gps_raw.eph                     = HDOP_h * 100.0;
        msg_gps_raw.epv                     = HDOP_v * 100.0;
        msg_gps_raw.vel                     = gpsGroundSpeed * 100.0;
        msg_gps_raw.fix_type                = gpsFixType;
        msg_gps_raw.satellites_visible      = nSat;

        mavlink_msg_gps_raw_int_encode(ID, compID, msg, &msg_gps_raw);
        ret = 0;
        break;
    }

    case MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
    {
        mavlink_global_position_int_t msg_gp;

        msg_gp.time_boot_ms     = bootTime;
        msg_gp.lat              = gpLat * 1e7;
        msg_gp.lon              = gpLon * 1e7;
        msg_gp.alt              = gpAlt * 1e3;
        msg_gp.relative_alt     = gpH * 1e3;
        msg_gp.vx               = gpVx * 100.0;
        msg_gp.vy               = gpVy * 100.0;
        msg_gp.vz               = gpVz * 100.0;
        msg_gp.hdg              = gpHeading * 100.0;

        mavlink_msg_global_position_int_encode(ID, compID, msg, &msg_gp);
        ret = 0;
        break;
    }

    case MAVLINK_MSG_ID_ATTITUDE:
    {
        mavlink_attitude_t msg_att;

        msg_att.time_boot_ms    = bootTime;
        msg_att.roll            = roll * M_PI/180.0;
        msg_att.pitch           = pitch * M_PI/180.0;
        msg_att.yaw             = yaw * M_PI/180.0;
        msg_att.rollspeed       = rollSpd;
        msg_att.pitchspeed      = pitchSpd;
        msg_att.yawspeed        = yawSpd;

        mavlink_msg_attitude_encode(ID, compID, msg, &msg_att);
        ret = 0;
        break;
    }
    } // end of switch

    return ret;
}

int VirtualUAV::timerFunction(void *arg)
{
    mavlink_message_t msg;

    m_timerCount ++;

    if( m_timerCount % 2 == 0 ) {
        if( generateMavlinkMsg(MAVLINK_MSG_ID_SYS_STATUS, &msg) == 0 )
            sendMavlinkMsg(&msg);

        if( generateMavlinkMsg(MAVLINK_MSG_ID_SYSTEM_TIME, &msg) == 0 )
            sendMavlinkMsg(&msg);

        if( generateMavlinkMsg(MAVLINK_MSG_ID_GLOBAL_POSITION_INT, &msg) == 0 )
            sendMavlinkMsg(&msg);
    } else {
        if( generateMavlinkMsg(MAVLINK_MSG_ID_STATUSTEXT, &msg) == 0 )
            sendMavlinkMsg(&msg);

        if( generateMavlinkMsg(MAVLINK_MSG_ID_GPS_RAW_INT, &msg) == 0 )
            sendMavlinkMsg(&msg);

        if( generateMavlinkMsg(MAVLINK_MSG_ID_ATTITUDE, &msg) == 0 )
            sendMavlinkMsg(&msg);
    }

    return 0;
}

int VirtualUAV::simulation(pi::JS_Val *jsv)
{
    return 0;
}

int VirtualUAV::toFlightGear(FGNetFDM *fgData)
{
    return 0;
}

int VirtualUAV::initUAV(double _lat, double _lng, double _alt, double _H)
{
    homeLat         = _lat;
    homeLng         = _lng;
    homeAlt         = _alt;
    homeH           = _H;

    updatePOS(_lat, _lng, _alt, _H, 0, 0, 0);

    return 0;
}

int VirtualUAV::updatePOS(double _lat, double _lng, double _alt, double _H,
                          double _yaw, double _pitch, double _roll)
{
    lat             = _lat;
    lon             = _lng;
    alt             = _alt;

    gpLat           = _lat;
    gpLon           = _lng;
    gpAlt           = _alt;
    gpH             = _H;

    yaw             = _yaw;
    pitch           = _pitch;
    roll            = _roll;
    gpHeading       = yaw;

    return 0;
}

int VirtualUAV::updateTime(double tsNow)
{
    gpsTime         = tsNow * 1e6;
    systimeUnix     = tsNow * 1e6;
    bootTime        = (tsNow - m_tmStart) * 1000;

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void VirtualUAV_Manager_timerFunc(void *arg)
{
    VirtualUAV_Manager *u = (VirtualUAV_Manager*) arg;

    u->timerFunction(arg);
}

VirtualUAV_Manager::VirtualUAV_Manager()
{
    init();

    // create timer
    m_timerCount = 0;

    // Mavlink channel
    mavlinkChan = MAVLINK_COMM_1;

    // UART buffer
    m_uartBufMaxSize = 8192;
    m_uartBuf = new uint8_t[m_uartBufMaxSize];

    // Joystick & UART
    m_joystick = NULL;
    m_vUART = NULL;

    // timer
    m_timer = 0;
    startTimer();
}

VirtualUAV_Manager::~VirtualUAV_Manager()
{
    release();

    stopTimer();

    m_timer = 0;
    m_timerCount = 0;

    delete [] m_uartBuf;
    m_uartBuf = NULL;

    m_joystick = NULL;
    m_vUART = NULL;
}

void VirtualUAV_Manager::init(void)
{
    m_mapUAV.clear();

    // init active objs
    m_activeUAV   = NULL;


    // output buffer
    m_msgBuffer.reserve(1024);
}

void VirtualUAV_Manager::release(void)
{
    // set IO devices
    m_joystick = NULL;
    m_vUART = NULL;

    // set active objs
    m_activeUAV = NULL;

    // clear UAS array & map
    if( m_mapUAV.size() > 0 ) {
        VUAV_IDMap::iterator it;
        VirtualUAV *u;

        for(it=m_mapUAV.begin(); it!=m_mapUAV.end(); it++) {
            u = it->second;
            delete u;
        }

        m_mapUAV.clear();
    }

    // clear msg buffer
    m_msgBuffer.clear();
}

void VirtualUAV_Manager::reset(void)
{
    release();
}

int VirtualUAV_Manager::startTimer(void)
{
    if( m_timer == 0 ) {
        osa_tm_create(&m_timer, 50, VirtualUAV_Manager_timerFunc, this);
    }

    return 0;
}

int VirtualUAV_Manager::stopTimer(void)
{
    if( m_timer != 0 ) {
        osa_tm_delete(m_timer);

        m_timer = 0;
    }

    return 0;
}

int VirtualUAV_Manager::timerFunction(void *arg)
{
    m_timerCount ++;

    // flight simulation
    if( m_activeUAV != NULL && m_joystick != NULL ) {
        JS_Val jsv;
        int r = m_joystick->read(&jsv);

        if( r == 0 ) {
            m_activeUAV->simulation(&jsv);

            // send simulation data to FlightGear
            int useFG = svar.GetInt("FlightGearTrans.UseFG", 1);
            FlightGear_Transfer *fgTrans = SvarWithType<FlightGear_Transfer*>::instance()["FlightGear_Transfer.ptr"];
            if( useFG && fgTrans != NULL ) {
                FGNetFDM fdm;

                m_activeUAV->toFlightGear(&fdm);
                fgTrans->trans(&fdm);
            }
        }
    }

    // for every second
    if( m_timerCount % 20 == 0 ) {
        // send heartbeat
        mavlink_message_t msg_hb;

        for(VUAV_IDMap::iterator it=m_mapUAV.begin(); it!=m_mapUAV.end(); it++) {
            VirtualUAV *u = it->second;

            if( 0 == u->generateMavlinkMsg(MAVLINK_MSG_ID_HEARTBEAT, &msg_hb) )
                sendMavlinkMsg(&msg_hb, u);
        }
    }

    // call each uas's timerFunction
    for(VUAV_IDMap::iterator it=m_mapUAV.begin(); it!=m_mapUAV.end(); it++) {
        VirtualUAV *u = it->second;

        u->timerFunction(u);
    }

    // process in/out stream
    if( m_vUART != NULL ) {
        int bufLen = m_uartBufMaxSize;

        // parse input data
        bufLen = m_vUART->read(m_uartBuf, bufLen, 0);
        if( bufLen > 0 ) {
            mavlink_message_t   msg;
            mavlink_status_t    status;

            for(int i=0; i<bufLen; i++) {
                if( mavlink_parse_char(mavlinkChan, m_uartBuf[i], &msg, &status) ) {
                    parseMavlinkMsg(&msg);
                }
            }
        }

        // send message
        bufLen = m_uartBufMaxSize;
        readMsgBuf(m_uartBuf, &bufLen);
        if( bufLen > 0 ) m_vUART->write(m_uartBuf, bufLen, 0);
    }

    return 0;
}

int VirtualUAV_Manager::parseMavlinkMsg(mavlink_message_t *msg)
{
    // call each uas's parseMavlinkMsg
    for(VUAV_IDMap::iterator it=m_mapUAV.begin(); it!=m_mapUAV.end(); it++) {
        VirtualUAV *u = it->second;

        u->parseMavlinkMsg(msg);
    }

    return 0;
}

int VirtualUAV_Manager::sendMavlinkMsg(mavlink_message_t *msg, VirtualUAV *u)
{
    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];

    RMutex m(&m_mutexMsgWrite);

    // Write message into buffer, prepending start sign
    static uint8_t messageKeys[256] = MAVLINK_MESSAGE_CRCS;
    mavlink_finalize_message_chan(msg,
                                  u->ID, u->compID,
                                  mavlinkChan,
                                  msg->len, messageKeys[msg->msgid]);

    int len = mavlink_msg_to_send_buffer(buffer, msg);

    for(int i=0; i<len; i++) m_msgBuffer.push_back(buffer[i]);

    return 0;
}

int VirtualUAV_Manager::addUAV(VirtualUAV *u)
{
    // check given UAV exist
    VirtualUAV *uu = getUAV(u->ID);

    if( uu != NULL ) {
        dbg_pe("Add UAV (%d) exist!", u->ID);
        return -1;
    }

    // add to list & map
    u->m_vuavManager = this;
    m_mapUAV.insert(std::make_pair(u->ID, u));

    // set active UAV to new one
    m_activeUAV = u;

    return 0;
}

int VirtualUAV_Manager::removeUAV(int id)
{
    // check given UAV exist
    VirtualUAV *uu = getUAV(id);

    if( uu == NULL ) {
        dbg_pe("Can not find UAV [%d]!", id);
        return -1;
    }

    // get the UAV
    VUAV_IDMap::iterator it = m_mapUAV.find(id);
    VirtualUAV *u = it->second;

    if( m_activeUAV == u ) m_activeUAV = NULL;
    m_mapUAV.erase(it);

    delete u;

    return 0;
}


int VirtualUAV_Manager::getUAVs(VUAV_Array &uavArr)
{
    uavArr.clear();

    for(VUAV_IDMap::iterator it=m_mapUAV.begin(); it!=m_mapUAV.end(); it++) {
        VirtualUAV *u = it->second;

        uavArr.push_back(u);
    }

    return 0;
}

VirtualUAV* VirtualUAV_Manager::getUAV(int id)
{
    VUAV_IDMap::iterator it;

    it = m_mapUAV.find(id);
    if( it != m_mapUAV.end() ) {
        return it->second;
    } else {
        return NULL;
    }
}

int VirtualUAV_Manager::setActiveUAV(int id)
{
    m_activeUAV = getUAV(id);

    return 0;
}

VirtualUAV* VirtualUAV_Manager::getActiveUAV(void)
{
    return m_activeUAV;
}


int VirtualUAV_Manager::writeMsgBuf(uint8_t *buf, int len)
{
    RMutex m(&m_mutexMsgWrite);

    for(int i=0; i<len; i++) m_msgBuffer.push_back(buf[i]);

    return 0;
}

int VirtualUAV_Manager::readMsgBuf(uint8_t *buf, int *len)
{
    RMutex m(&m_mutexMsgWrite);

    int l = m_msgBuffer.size();
    if( *len > l ) *len = l;

    for(int i=0; i<*len; i++) buf[i] = m_msgBuffer[i];
    m_msgBuffer.clear();

    return 0;
}
