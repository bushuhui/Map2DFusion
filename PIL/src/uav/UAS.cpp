

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "base/utils/utils.h"
#include "hardware/Gps/utils_GPS.h"

#include "UAS.h"

using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void UAS_timerFunc(void *arg)
{
    UAS_Base *u = (UAS_Base*) arg;

    u->timerFunction(arg);
}


UAS_Base::UAS_Base()
{
    // initialize parameters
    init();

    // clear home
    clear_home();

    // set parameter array's UAS obj
    m_paramArray.setUAS(this);
    m_waypoints.setUAS(this);
    currMission = 1;

    // initial UAS manager
    m_uasManager = NULL;

    // initialize timer & mutex for msg wirting
    m_timer = 0;
    m_timerCount = 0;
    m_connectTime = 0;
    m_paramAutoLoaded = 0;

    // status message time
    m_statusMsgTime = -1;
    statusText[0] = 0;
    severity = 1;

    // package lost stastic
    m_pkgLost = 0;
    m_pkgLastID = -1;

    // MAVLINK connected or not
    m_bLinkConnected = 0;

    // received message in one second
    m_recvMessageInSec = 0;

    // action confirmed
    m_bCmdConfirmed = 0;
    m_sLastCommand[0] = 0;

    // UAV stream requested
    //  see GCS_MAVLINK::data_stream_send(void)
    m_bStreamRequested = 0;

    // requrest information frequence
    //  STREAM_RAW_SENSORS:
    //      MSG_RAW_IMU1
    //      MSG_RAW_IMU2
    //      MSG_RAW_IMU3
    //  STREAM_EXTENDED_STATUS:
    //      MSG_EXTENDED_STATUS1
    //      MSG_EXTENDED_STATUS2
    //      MSG_CURRENT_WAYPOINT
    //      MSG_GPS_RAW
    //      MSG_NAV_CONTROLLER_OUTPUT
    //      MSG_LIMITS_STATUS
    //  STREAM_RC_CHANNELS:
    //      MSG_RADIO_OUT
    //      MSG_RADIO_IN
    //  STREAM_RAW_CONTROLLER:
    //      MSG_SERVO_OUT
    //  STREAM_POSITION:
    //      MSG_LOCATION
    //  STREAM_EXTRA1:
    //      MSG_ATTITUDE
    //      MSG_SIMSTATE
    //  STREAM_EXTRA2
    //      MSG_VFR_HUD
    //  STREAM_EXTRA3:
    //      MSG_AHRS
    //      MSG_HWSTATUS
    //      MSG_SYSTEM_TIME
    //      MSG_RANGEFINDER
    //      MSG_TERRAIN
    //      MSG_MOUNT_STATUS
    //      MSG_OPTICAL_FLOW
    //      MSG_GIMBAL_REPORT

    for(int i=0; i<16; i++) m_frqStream[i] = NULL;

    m_frqStream[MAV_DATA_STREAM_RAW_SENSORS]        = &svar.GetInt("Mavlink.frqStreamRawSensors",    2);
    m_frqStream[MAV_DATA_STREAM_EXTENDED_STATUS]    = &svar.GetInt("Mavlink.frqStreamExtStatus",     5);
    m_frqStream[MAV_DATA_STREAM_RC_CHANNELS]        = &svar.GetInt("Mavlink.frqStreamRCChannels",    5);
    m_frqStream[MAV_DATA_STREAM_RAW_CONTROLLER]     = &svar.GetInt("Mavlink.frqStreamRawController", 2);
    m_frqStream[MAV_DATA_STREAM_POSITION]           = &svar.GetInt("Mavlink.frqStreamPos",           5);
    m_frqStream[MAV_DATA_STREAM_EXTRA1]             = &svar.GetInt("Mavlink.frqStreamExtr1",         5);
    m_frqStream[MAV_DATA_STREAM_EXTRA2]             = &svar.GetInt("Mavlink.frqStreamExtr2",         2);
    m_frqStream[MAV_DATA_STREAM_EXTRA3]             = &svar.GetInt("Mavlink.frqStreamExtr3",         2);

    // valid date/time
    m_tsValid_Year = svar.GetInt("FastGCS.tsValid_Year");
    m_tsLastPOS = 0;
}

UAS_Base::~UAS_Base()
{
    stopTimer();

    release();
}

void UAS_Base::init(void)
{
    // system information
    ID              = 1;
    compID          = 1;

    gcsID           = 254;
    gcsCompID       = 1;

    uasType         = UAS_TYPE_MAV;
    customMode      = 0;
    szFlightMode    = "STABILIZE";
    motorArmed      = 0;
    szMotorArmed    = "Disarm";
    baseMode        = 0;
    motorArmed      = 0;
    systemStatus    = 0;
    szSystemStatus  = "STANDBY";
    mavlinkChan     = 0;

    cpuLoad         = 0;
    battVolt        = 0;
    battCurrent     = 0;
    battRemaining   = 0;
    commDropRate    = 0;

    systimeUnix     = 0;
    bootTime        = 0;
    dateTime.toLocalTime();
    dateTime_bootTime = 0;

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
}

void UAS_Base::release(void)
{
    return;
}



int UAS_Base::gen_listmap_important(String2StringMap &lm)
{
    return 0;
}

int UAS_Base::gen_listmap_all(String2StringMap &lm)
{
    return 0;
}

int UAS_Base::link_connected(void)
{
    return m_bLinkConnected;
}

int UAS_Base::clear_home(void)
{
    homeLat = -9999;
    homeLng = -9999;
    homeAlt = 0;
    homeH   = 0;

    homeSetted = 0;
    homeSetCount = 10;
    homePosReaded = 0;
}


int UAS_Base::startTimer(void)
{

    // create timer
    if( 0 != osa_tm_create(&m_timer, 100, UAS_timerFunc, this) ) {
        dbg_pe("Can not creat timer");
    }

    return 0;
}

int UAS_Base::stopTimer(void)
{
    if( m_timer != 0 )
        osa_tm_delete(m_timer);
    m_timer = 0;

    return 0;
}


///
/// \brief UAS_Base::timerFunction
///        call every 100 ms
///
/// \param arg
/// \return
///
int UAS_Base::timerFunction(void *arg)
{
    // count
    m_timerCount ++;

    // call hook timer
    m_paramArray.timerFunction(arg);
    m_waypoints.timerFunction(arg);

    // one second loop
    if( m_timerCount % 10 == 0 ) {
        // check connection
        if( m_recvMessageInSec < 1 ) {
            m_bLinkConnected = 0;
            m_connectTime = 0;
        } else {
            m_bLinkConnected = 1;
            m_connectTime ++;
        }

        // auto set data stream frequency
        if( !m_bStreamRequested &&
            svar.GetInt("Mavlink.AutoSetStreamFreq", 1) &&
            m_connectTime > svar.GetInt("Mavlink.AutoSetStreamFreq_StartTime", 5) ) {

            // FIXME: only set MAV's data stream frequency
            if( uasType == UAS_TYPE_MAV )
                setStreamFrequency(MAV_DATA_STREAM_ALL);

            m_bStreamRequested = 1;
        }

        // auto load parameters
        if(  !m_paramAutoLoaded &&
             svar.GetInt("Mavlink.AutoLoadParameters", 1) &&
             m_connectTime > svar.GetInt("Mavlink.AutoLoadParameters_StartTime", 10) ) {
            m_paramArray.requireParameters();
            m_paramAutoLoaded = 1;
        }

        // clear one-second count
        m_recvMessageInSec = 0;

        // auto clean status message
        if( m_statusMsgTime >= 0 ) {
            m_statusMsgTime ++;

            if( m_statusMsgTime > svar.GetInt("Mavlink.StatusMessage_CleanTimeout", 30) ) {
                m_statusMsgTime = -1;
                statusText[0] = 0;
                severity = 0;
            }
        }
    }

    return 0;
}


int UAS_Base::parseMavlinkMsg(mavlink_message_t *msg)
{
    // count received message in one second
    m_recvMessageInSec ++;

    // FIXME: calculate lost packages
    if( m_pkgLastID >= 0 ) {
        if( msg->seq < m_pkgLastID )
            m_pkgLost += (255+msg->seq) - m_pkgLastID;
        else
            m_pkgLost += msg->seq - m_pkgLastID - 1;
    }
    m_pkgLastID = msg->seq;

    // process hook message
    if( m_paramArray.parseMavlinkMsg(msg) ) return 0;
    if( m_waypoints.parseMavlinkMsg(msg) )  return 0;

    // callback functions
    {
        RMutex m(&m_mutexMessageHandle);

        MavlinkMessageHandleMap::iterator it;

        //printf("m_mavlinkMsgHandleMap.size = %d\n", m_mavlinkMsgHandleMap.size());
        for(it=m_mavlinkMsgHandleMap.begin(); it!=m_mavlinkMsgHandleMap.end(); it++) {
            it->second(msg);
        }
    }

    // for each message type
    switch( msg->msgid ) {
    case MAVLINK_MSG_ID_HEARTBEAT:
    {
        mavlink_heartbeat_t msg_hb;
        mavlink_msg_heartbeat_decode(msg, &msg_hb);

        ID                  = msg->sysid;
        compID              = msg->compid;

        customMode          = msg_hb.custom_mode;
        uavType             = msg_hb.type;
        autopilotType       = msg_hb.autopilot;
        baseMode            = msg_hb.base_mode;
        systemStatus        = msg_hb.system_status;
        mavlinkVersion      = msg_hb.mavlink_version;

        // motor armed or not?
        if( baseMode & MAV_MODE_FLAG_SAFETY_ARMED ) {
            motorArmed = 1;
            szMotorArmed = "Armed";
        } else {
            motorArmed = 0;
            szMotorArmed = "Disarm";
        }

        // get system status
        if( systemStatus == MAV_STATE_STANDBY )         szSystemStatus = "STANDBY";
        else if( systemStatus == MAV_STATE_ACTIVE )     szSystemStatus = "ACTIVE";
        else if( systemStatus == MAV_STATE_CRITICAL )   szSystemStatus = "CRITICAL";

        // get flight mode
        char *fm;
        mavlink_mav_custommode_getName(customMode, &fm);
        szFlightMode = fm;

        break;
    }

    case MAVLINK_MSG_ID_SYS_STATUS:
    {
        mavlink_sys_status_t msg_ss;
        mavlink_msg_sys_status_decode(msg, &msg_ss);

        //sensorsPresent      = msg_ss.onboard_control_sensors_present;
        //sensorsEnabled      = msg_ss.onboard_control_sensors_enabled;
        //sensorsHealth       = msg_ss.onboard_control_sensors_health;
        cpuLoad             = msg_ss.load * 1.0 / 10.0;
        battVolt            = m_avgBatV.push(msg_ss.voltage_battery * 1.0 / 1000.0);
        battCurrent         = msg_ss.current_battery * 1.0 / 100.0;
        battRemaining       = msg_ss.battery_remaining;
        commDropRate        = msg_ss.drop_rate_comm * 1.0 / 100.0;

        //mavlink_sys_status_sensor_getIDs(sensorsPresent, sensorsPresentList);
        //mavlink_sys_status_sensor_getIDs(sensorsEnabled, sensorsEnabledList);
        //mavlink_sys_status_sensor_getIDs(sensorsHealth, sensorsHealthList);
        //mavlink_sys_status_sensor_getID_Difference(sensorsPresentList, sensorsHealthList,
        //                                           sensorsUnhealthList);

        break;
    }

    case MAVLINK_MSG_ID_STATUSTEXT:
    {
        mavlink_statustext_t st;
        mavlink_msg_statustext_decode(msg, &st);

        severity = st.severity;
        strcpy(statusText, st.text);

        m_statusMsgTime = 0;

        break;
    }

    case MAVLINK_MSG_ID_SYSTEM_TIME:
    {
        mavlink_system_time_t msg_st;
        mavlink_msg_system_time_decode(msg, &msg_st);

        systimeUnix         = msg_st.time_unix_usec;
        bootTime            = msg_st.time_boot_ms;

        dateTime.fromTimeStamp(systimeUnix);
        dateTime_bootTime = bootTime;

        //dbg_pt("systimeUnix = %lld, bootTime = %lldd\n", systimeUnix, bootTime);
        //dateTime.print(); printf("\n\n");

        break;
    }


    case MAVLINK_MSG_ID_GPS_RAW_INT:
    {
        mavlink_gps_raw_int_t msg_gps_raw;
        mavlink_msg_gps_raw_int_decode(msg, &msg_gps_raw);

        gpsTime             = msg_gps_raw.time_usec;
        lat                 = msg_gps_raw.lat * 1.0 / 1e7;
        lon                 = msg_gps_raw.lon * 1.0 / 1e7;
        alt                 = msg_gps_raw.alt * 1.0 / 1000.0;
        HDOP_h              = msg_gps_raw.eph * 1.0 / 100.0;
        HDOP_v              = msg_gps_raw.epv * 1.0 / 100.0;
        gpsGroundSpeed      = msg_gps_raw.vel * 1.0 / 100.0;
        gpsFixType          = msg_gps_raw.fix_type;
        nSat                = msg_gps_raw.satellites_visible;

        // FIXME: fix HDOP values
        if( fabs(HDOP_h) < 0.001 ) HDOP_h = 9999;
        if( fabs(HDOP_v) < 0.001 ) HDOP_v = 9999;

        // FIXME: get home position
        if( gpsFixType >= 3 && homeSetCount > 0 ) {
            homeLat = lat;
            homeLng = lon;
            homeAlt = alt;
            homeH   = gpH;

            if( homeSetCount > 0 )
                homeSetCount --;

            if( homeSetCount == 0 ) {
                homeSetted = 1;
                homePosReaded = 0;
            }
        }

        // calculate vertical & horiz speed
        if( lastTM == 0 ) lastTM = gpsTime;

        uint64_t dt = gpsTime - lastTM;
        if( dt > 1000000 ) {
            velV = (alt - lastAlt)/(1.0*dt/1000000.0);

            double dx, dy;

            calcLngLatDistance(lastLon, lastLat, lon, lat, dx, dy);
            velH = sqrt(dx*dx + dy*dy) / (1.0*dt/1000000.0);

            lastTM  = gpsTime;
            lastAlt = alt;
            lastLat = lat;
            lastLon = lon;
        }

        // calculate distance
        if( homeSetted ) {
            double dx, dy, dh;

            dh = alt - homeAlt;
            calcLngLatDistance(homeLng, homeLat, lon, lat, dx, dy);
            dis = sqrt(dx*dx + dy*dy);
            disLOS = sqrt(dis*dis + dh*dh);
        }

        break;
    }

    case MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
    {
        mavlink_global_position_int_t msg_gp;
        mavlink_msg_global_position_int_decode(msg, &msg_gp);

        bootTime            = msg_gp.time_boot_ms;
        gpLat               = msg_gp.lat * 1.0 / 1e7;
        gpLon               = msg_gp.lon * 1.0 / 1e7;
        gpAlt               = msg_gp.alt * 1.0 / 1e3;
        gpH                 = msg_gp.relative_alt * 1.0 / 1e3;
        gpVx                = msg_gp.vx * 1.0 / 100.0;
        gpVy                = msg_gp.vy * 1.0 / 100.0;
        gpVz                = msg_gp.vz * 1.0 / 100.0;
        gpHeading           = msg_gp.hdg * 1.0 / 100.0;

        // set POS data
        int64_t ts = systimeUnix + (bootTime - dateTime_bootTime)*1000;
        gpDateTime.fromTimeStamp(ts);
        setPosData();

        break;
    }

    case MAVLINK_MSG_ID_RAW_IMU:
    {
        mavlink_raw_imu_t msg_imu_raw;
        mavlink_msg_raw_imu_decode(msg, &msg_imu_raw);

        Ax_raw              = msg_imu_raw.xacc;
        Ay_raw              = msg_imu_raw.yacc;
        Az_raw              = msg_imu_raw.zacc;
        Gx_raw              = msg_imu_raw.xgyro;
        Gy_raw              = msg_imu_raw.ygyro;
        Gz_raw              = msg_imu_raw.zgyro;
        Mx_raw              = msg_imu_raw.xmag;
        My_raw              = msg_imu_raw.ymag;
        Mz_raw              = msg_imu_raw.zmag;

        break;
    }

    case MAVLINK_MSG_ID_SCALED_IMU2:
    {
        mavlink_scaled_imu2_t msg_imu;
        mavlink_msg_scaled_imu2_decode(msg, &msg_imu);

        Ax                  = msg_imu.xacc;
        Ay                  = msg_imu.yacc;
        Az                  = msg_imu.zacc;
        Gx                  = msg_imu.xgyro;
        Gy                  = msg_imu.ygyro;
        Gz                  = msg_imu.zgyro;
        Mx                  = msg_imu.xmag;
        My                  = msg_imu.ymag;
        Mz                  = msg_imu.zmag;

        break;
    }

    case MAVLINK_MSG_ID_ATTITUDE:
    {
        mavlink_attitude_t msg_att;
        mavlink_msg_attitude_decode(msg, &msg_att);

        bootTime            = msg_att.time_boot_ms;
        roll                = msg_att.roll  * 180.0 / M_PI;
        pitch               = msg_att.pitch * 180.0 / M_PI;
        yaw                 = msg_att.yaw   * 180.0 / M_PI;
        rollSpd             = msg_att.rollspeed;
        pitchSpd            = msg_att.pitchspeed;
        yawSpd              = msg_att.yawspeed;

        break;
    }


    case MAVLINK_MSG_ID_RC_CHANNELS_RAW:
    {
        mavlink_rc_channels_raw_t msg_rc_raw;
        mavlink_msg_rc_channels_raw_decode(msg, &msg_rc_raw);

        rcRaw[0]            = msg_rc_raw.chan1_raw;
        rcRaw[1]            = msg_rc_raw.chan2_raw;
        rcRaw[2]            = msg_rc_raw.chan3_raw;
        rcRaw[3]            = msg_rc_raw.chan4_raw;
        rcRaw[4]            = msg_rc_raw.chan5_raw;
        rcRaw[5]            = msg_rc_raw.chan6_raw;
        rcRaw[6]            = msg_rc_raw.chan7_raw;
        rcRaw[7]            = msg_rc_raw.chan8_raw;
        rcRaw_port          = msg_rc_raw.port;
        rcRSSI              = msg_rc_raw.rssi;

        /*
        printf("rcRaw (%5d): ", rcRSSI);
        for(int i=0; i<8; i++) printf("%4d ", rcRaw[i]);
        printf("\n");
        */

        break;
    }

    case MAVLINK_MSG_ID_RC_CHANNELS:
    {
        mavlink_rc_channels_t msg_rc_all;
        mavlink_msg_rc_channels_decode(msg, &msg_rc_all);

        rcAll[0]            = msg_rc_all.chan1_raw;
        rcAll[1]            = msg_rc_all.chan2_raw;
        rcAll[2]            = msg_rc_all.chan3_raw;
        rcAll[3]            = msg_rc_all.chan4_raw;
        rcAll[4]            = msg_rc_all.chan5_raw;
        rcAll[5]            = msg_rc_all.chan6_raw;
        rcAll[6]            = msg_rc_all.chan7_raw;
        rcAll[7]            = msg_rc_all.chan8_raw;
        rcAll[8]            = msg_rc_all.chan9_raw;
        rcAll[9]            = msg_rc_all.chan10_raw;
        rcAll[10]           = msg_rc_all.chan11_raw;
        rcAll[11]           = msg_rc_all.chan12_raw;
        rcAll[12]           = msg_rc_all.chan13_raw;
        rcAll[13]           = msg_rc_all.chan14_raw;
        rcAll[14]           = msg_rc_all.chan15_raw;
        rcAll[15]           = msg_rc_all.chan16_raw;
        rcAll[16]           = msg_rc_all.chan17_raw;
        rcAll[17]           = msg_rc_all.chan18_raw;
        rcAll_channels      = msg_rc_all.rssi;

        //printf("rcAll (%5d): ", rcRSSI);
        //for(int i=0; i<16; i++) printf("%4d ", rcAll[i]);
        //printf("\n");

        break;
    }

    case MAVLINK_MSG_ID_RC_CHANNELS_OVERRIDE:
    {
        mavlink_rc_channels_override_t pack;
        mavlink_msg_rc_channels_override_decode(msg, &pack);

        /*
        printf("RC_OVERRIDE: %3d %3d-%3d, %4d %4d %4d %4d %4d %4d %4d %4d\n",
               msg.sysid, pack.target_system, pack.target_component,
               pack.chan1_raw, pack.chan2_raw, pack.chan3_raw, pack.chan4_raw,
               pack.chan5_raw, pack.chan6_raw, pack.chan7_raw, pack.chan8_raw);
        */

        break;
    }


    case MAVLINK_MSG_ID_COMMAND_ACK:
    {
        mavlink_command_ack_t pack;
        mavlink_msg_command_ack_decode(msg, &pack);

        severity = 0;
        sprintf(statusText, "Command Ack: %s (CMD: %d, RES: %d)",
                m_sLastCommand, pack.command, pack.result);

        m_statusMsgTime = 0;
        m_bCmdConfirmed = 0;

        break;
    }


    } // end of switch(msg->msgid)

    return 0;
}


int UAS_Base::sendMavlinkMsg(mavlink_message_t *msg)
{
    if( m_uasManager != NULL )
        m_uasManager->sendMavlinkMsg(msg);

    return 0;
}

int UAS_Base::requireParameters(int clearOld)
{
    return m_paramArray.requireParameters(clearOld);
}

int UAS_Base::requireParameter(int index)
{
    return m_paramArray.requireParameter(index);
}

int UAS_Base::requireParameter(char *id)
{
    return m_paramArray.requireParameter(id);
}

int UAS_Base::updateParameters(void)
{
    return m_paramArray.updateParameters();
}

int UAS_Base::updateParameter(int index)
{
    return m_paramArray.updateParameter(index);
}

int UAS_Base::updateParameter(char *id)
{
    return m_paramArray.updateParameter(id);
}



int UAS_Base::writeWaypoints(AP_WPArray &wpa)
{
    // check waypoint number
    if( wpa.size() <= 1 ) {
        dbg_pe("Waypoint number (%d) is not enough!\n", wpa.size());
        return -1;
    }

    // copy wp array
    m_waypoints.set(&wpa);

    // send waypoints number
    return m_waypoints.writeWaypoints();
}

int UAS_Base::readWaypoints(void)
{
    return m_waypoints.readWaypoints();
}

int UAS_Base::clearWaypoints(void)
{
    return m_waypoints.clearWaypoints();
}

int UAS_Base::setCurrentWaypoint(int idx)
{    
    return m_waypoints.setCurrentWaypoint(idx);
}


int UAS_Base::getStreamFrequency(MAV_DATA_STREAM streamID)
{
    vector<int> streamIDs = mavlink_getStreamIDs();
    vector<int>::iterator it;
    int i;

    for(it = streamIDs.begin(); it != streamIDs.end(); it++) {
        i = *it;
        if( i == streamID ) {
            if( m_frqStream[i] == NULL ) return -1;
            else                         return *m_frqStream[i];
        }
    }

    return -1;
}

int UAS_Base::setStreamFrequency(MAV_DATA_STREAM streamID, int freq)
{
    mavlink_request_data_stream_t packet;
    mavlink_message_t msg;

    vector<int> streamIDs = mavlink_getStreamIDs();
    vector<int>::iterator it;
    int i;

    packet.target_system = ID;
    packet.target_component = compID;

    for(it = streamIDs.begin(); it != streamIDs.end(); it++) {
        i = *it;
        if( streamID == MAV_DATA_STREAM_ALL || i == streamID ) {
            packet.req_stream_id = (MAV_DATA_STREAM) i;

            if( m_frqStream[i] == NULL ) continue;

            if( freq >= 0 ) *m_frqStream[i] = freq;

            if( *m_frqStream[i] > 0 ) packet.start_stop = 1;
            else                      packet.start_stop = 0;

            packet.req_message_rate = *m_frqStream[i];

            mavlink_msg_request_data_stream_encode(gcsID, gcsCompID, &msg, &packet);
            sendMavlinkMsg(&msg);
        }
    }

    return 0;
}


int UAS_Base::executeCommand(MAV_CMD command, int confirmation,
                             float param1, float param2, float param3, float param4, float param5, float param6, float param7,
                             int component)
{
    mavlink_message_t msg;
    mavlink_command_long_t cmd;

    cmd.command = (uint16_t)command;
    cmd.confirmation = confirmation;
    cmd.param1 = param1;
    cmd.param2 = param2;
    cmd.param3 = param3;
    cmd.param4 = param4;
    cmd.param5 = param5;
    cmd.param6 = param6;
    cmd.param7 = param7;
    cmd.target_system = ID;
    cmd.target_component = component;

    mavlink_msg_command_long_encode(gcsID, gcsCompID, &msg, &cmd);
    sendMavlinkMsg(&msg);

    return 0;
}

int UAS_Base::executeCommandAck(int num, bool success)
{
    mavlink_message_t msg;
    mavlink_command_ack_t ack;

    ack.command = num;
    ack.result = (success ? 1 : 0);

    mavlink_msg_command_ack_encode(gcsID, gcsCompID, &msg, &ack);
    sendMavlinkMsg(&msg);

    return 0;
}

int UAS_Base::_MAV_CMD_DO_SET_SERVO(int ch, int pwm)
{
    mavlink_command_long_t packet;
    mavlink_message_t msg;

    packet.param1           = ch;
    packet.param2           = pwm;
    packet.command          = MAV_CMD_DO_SET_SERVO;
    packet.target_system    = ID;
    packet.target_component = compID;

    //dbg_pt("packet.confirmation = %d\n", packet.confirmation);

    mavlink_msg_command_long_encode(gcsID, gcsCompID, &msg, &packet);
    sendMavlinkMsg(&msg);

    sprintf(m_sLastCommand, "MAV_CMD_DO_SET_SERVO: CH=%d, PWM = %d", ch, pwm);

    return 0;
}



int UAS_Base::registMavlinkMessageHandle(const std::string &handleName,
                                         Mavlink_Message_Handle &msgHandle)
{
    RMutex m(&m_mutexMessageHandle);

    MavlinkMessageHandleMap::iterator it;

    it = m_mavlinkMsgHandleMap.find(handleName);
    if( it == m_mavlinkMsgHandleMap.end() ) {
        m_mavlinkMsgHandleMap.insert(make_pair(handleName, msgHandle));
    } else {
        it->second = msgHandle;
    }

    return 0;
}

int UAS_Base::unregistMavlinkMessageHandle(const std::string &handleName)
{
    RMutex m(&m_mutexMessageHandle);

    MavlinkMessageHandleMap::iterator it;

    it = m_mavlinkMsgHandleMap.find(handleName);
    if( it == m_mavlinkMsgHandleMap.end() ) {
        return -1;
    } else {
        m_mavlinkMsgHandleMap.erase(it);
    }

    return 0;
}


int UAS_Base::setPosData(void)
{
    POS_Data    pos;

    // if GPS not fixed then return
    if( gpsFixType < 3 ) return -1;

    pos.time        = gpDateTime;

    pos.lat         = gpLat;
    pos.lng         = gpLon;
    pos.altitude    = gpAlt;
    pos.h           = gpH;
    pos.vx          = gpVx;
    pos.vy          = gpVy;
    pos.vz          = gpVz;

    pos.HDOP        = HDOP_h;
    pos.nSat        = nSat;
    pos.fixQuality  = gpsFixType;

    pos.ahrs.yaw    = yaw;
    pos.ahrs.roll   = roll;
    pos.ahrs.pitch  = pitch;

    // calculate positon from home point
    if( homeSetted ) {
        double dx, dy, dh;

        dh = gpAlt - homeAlt;
        calcLngLatDistance(homeLng, homeLat, gpLon, gpLat, dx, dy);

        pos.x = dx;
        pos.y = dy;
        pos.z = dh;
    }

    pos.correct = 1;
    pos.posAvaiable = 1;

    // insert to array
    int64_t tsNow = pos.time.toTimeStamp();
    if( tsNow - m_tsLastPOS > 1000 ) {
        if( gpDateTime.year >= m_tsValid_Year ) {
            m_posData.addData(pos);
            m_tsLastPOS = tsNow;
        }
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

UAS_MAV::UAS_MAV() : UAS_Base()
{

}

UAS_MAV::~UAS_MAV()
{

}

void UAS_MAV::init(void)
{
    UAS_Base::init();
}

void UAS_MAV::release(void)
{
    UAS_Base::release();
}


int UAS_MAV::moveTo(double Lat, double Lng)
{
    mavlink_message_t msg;
    mavlink_mission_item_t packet;

    double &wpRadius = svar.GetDouble("MAVLINK.MAV_CMD_NAV_WAYPOINT.Radisu", 5.0);

    packet.param1           = 0;                        /* float param1 : hold time in seconds */
    packet.param2           = wpRadius;                 /* float param2 : acceptance radius in meters */
    packet.param3           = 0;                        /* float param3 : pass through waypoint */
    packet.param4           = 0;                        /* float param4 : desired yaw angle at waypoint */

    packet.x                = Lat;                      /* float x : lat degrees */
    packet.y                = Lng;                      /* float y : lon degrees */
    packet.z                = gpH;                      /* float z : alt meters */

    packet.seq              = 0;                        /* uint16_t seq: always 0, unknown why. */
    packet.current          = 2;                        /* uint8_t current: 2 indicates guided mode waypoint */
    packet.command          = MAV_CMD_NAV_WAYPOINT;     /* uint16_t command: arducopter specific */
    packet.frame            = MAV_FRAME_GLOBAL;         /* uint8_t frame: arducopter uninterpreted */
    packet.autocontinue     = 0;                        /* uint8_t autocontinue: always 0 */

    packet.target_system    = ID;
    packet.target_component = compID;

    mavlink_msg_mission_item_encode(gcsID, gcsCompID, &msg, &packet);
    return sendMavlinkMsg(&msg);
}

int UAS_MAV::changeHeight(double newH)
{
    mavlink_message_t msg;
    mavlink_mission_item_t packet;

    double &wpRadius = svar.GetDouble("MAVLINK.MAV_CMD_NAV_WAYPOINT.Radisu", 5.0);

    packet.param1           = 0;                        /* float param1 : hold time in seconds */
    packet.param2           = wpRadius;                 /* float param2 : acceptance radius in meters */
    packet.param3           = 0;                        /* float param3 : pass through waypoint */
    packet.param4           = 0;                        /* float param4 : desired yaw angle at waypoint */

    packet.x                = gpLat;                    /* float x : lat degrees */
    packet.y                = gpLon;                    /* float y : lon degrees */
    packet.z                = newH;                     /* float z : alt meters */

    packet.seq              = 0;                        /* uint16_t seq: always 0, unknown why. */
    packet.current          = 2;                        /* uint8_t current: 2 indicates guided mode waypoint */
    packet.command          = MAV_CMD_NAV_WAYPOINT;     /* uint16_t command: arducopter specific */
    packet.frame            = MAV_FRAME_GLOBAL;         /* uint8_t frame: arducopter uninterpreted */
    packet.autocontinue     = 0;                        /* uint8_t autocontinue: always 0 */

    packet.target_system    = ID;
    packet.target_component = compID;

    mavlink_msg_mission_item_encode(gcsID, gcsCompID, &msg, &packet);
    return sendMavlinkMsg(&msg);
}


int UAS_MAV::parseMavlinkMsg(mavlink_message_t *msg)
{
    // perform basic parsing
    UAS_Base::parseMavlinkMsg(msg);

    // print status message
    if( msg->msgid == MAVLINK_MSG_ID_STATUSTEXT ) {
        fmt::print_colored(fmt::RED, "STATUS_MAV[{0}] = {1}\n",
                           severity, statusText);
    }

    return 0;
}

int UAS_MAV::gen_listmap_important(String2StringMap &lm)
{
    // boot time
    int bt_min, bt_sec, bt_msec;

    bt_msec = bootTime % 1000;
    bt_sec  = bootTime/1000;
    bt_min  = bt_sec / 60;
    bt_sec  = bt_sec % 60;
    lm["UAV_bTime"] = trim(fmt::sprintf("%d:%02d.%03d", bt_min, bt_sec, bt_msec));

    //char *name;
    //mavlink_mav_type_name(uavType, &name);
    //lm["UAV_uavType"]   = name;
    //mavlink_autopilot_name(uavAutopilot, &name);
    //lm["UAV_AP"]        = name;
    //mavlink_mav_state_name(uavSystemStatus, &name);
    //lm["UAV_state"]     = name;

    if( strlen(statusText) > 0 ) {
        lm["UAV_status"] = trim(fmt::sprintf("[%d] %s", severity, statusText));
    } else {
        lm["UAV_status"] = "";
    }

    //lm["UAV_pkgLost"] = trim(fmt::sprintf("%d", m_pkgLost));

    /*
    mavlink_mav_mode_name(uavBaseMode, nl);
    for(int i=0; i<nl.size(); i++) {
        if( i == 0 ) nl_all = nl_all + nl[i];
        else         nl_all = nl_all + ", " + nl[i];
    }
    lm["UAV_mode"]      = nl_all;
    */


    //lm["UAV_ID"]     = trim(fmt::sprintf("%d", ID));

    lm["UAV_bat"]     = trim(fmt::sprintf("%4.2fV, %3.1fA, %4.1f%%", battVolt, battCurrent, battRemaining));
    //lm["UAV_CPU"]      = trim(fmt::sprintf("%6.2f%%", cpuLoad));

    //lm["UAV_roll"]      = trim(fmt::sprintf("%12f", roll));
    //lm["UAV_pitch"]     = trim(fmt::sprintf("%12f", pitch));
    //lm["UAV_yaw"]       = trim(fmt::sprintf("%12f", yaw));

    lm["UAV_Alt"]        = trim(fmt::sprintf("%4.2f, %4.2f", gpAlt, gpH));

    string gpsFixed = "Lost";
    switch( gpsFixType ) {
    case 0:
    case 1:
        gpsFixed  = "Lost";
        break;

    case 2:
        gpsFixed  = "2D-fixed";
        break;

    case 3:
        gpsFixed  = "3D-fixed";
        break;

    case 4:
        gpsFixed  = "DGPS";
        break;

    case 5:
        gpsFixed  = "RTK";
        break;
    }

    lm["UAV_GPS"]       = trim(fmt::sprintf("%d, %s, %3.1f", nSat, gpsFixed, HDOP_h));
    //lm["UAV_HDOP_V"]  = trim(fmt::sprintf("%8.2f", HDOP_v));
    lm["UAV_heading"]   = trim(fmt::sprintf("%6.1f", gpHeading));


    lm["UAV_currWP"]    = trim(fmt::sprintf("%d", currMission));
    lm["UAV_Vel"]       = trim(fmt::sprintf("%4.2f, %4.2f", velH, velV));
    lm["UAV_Dis"]       = trim(fmt::sprintf("%3.1f (%3.1f)", dis, disLOS));

    lm["UAV_Mode"]      = trim(fmt::sprintf("%s, %s, %s", szFlightMode, szMotorArmed, szSystemStatus));

    return 0;
}

int UAS_MAV::gen_listmap_all(String2StringMap &lm)
{
    return 0;
}

int UAS_MAV::timerFunction(void *arg)
{
    // call basic timer function
    UAS_Base::timerFunction(arg);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

UAS_GCS::UAS_GCS() : UAS_Base()
{

}

UAS_GCS::~UAS_GCS()
{

}

void UAS_GCS::init(void)
{
    UAS_Base::init();

    // ATA yaw offset
    ataYawOffset = 0.0;
    ataYawOffset_setted = 0;
}

void UAS_GCS::release(void)
{
    UAS_Base::release();
}

int UAS_GCS::parseMavlinkMsg(mavlink_message_t *msg)
{
    // parse basic information
    UAS_Base::parseMavlinkMsg(msg);

    // get arguments
    int gcsShowRC = 0;
    gcsShowRC = svar.GetInt("FastGCS.gcsShowRC", gcsShowRC);

    // output status message
    if( msg->msgid == MAVLINK_MSG_ID_STATUSTEXT ) {
        fmt::print_colored(fmt::GREEN, "STATUS_GCS[{0}] = {1}\n",
                           severity, statusText);
    }

    // output RC values
    if( msg->msgid == MAVLINK_MSG_ID_RC_CHANNELS_RAW && gcsShowRC ) {
        printf("rcRaw (%5d): ", rcRSSI);
        for(int i=0; i<8; i++) printf("%4d ", rcRaw[i]);
        printf("\n");
    }

    return 0;
}

int UAS_GCS::gen_listmap_important(String2StringMap &lm)
{
    // GCS
    if( strlen(statusText) > 0 ) {
        lm["GCS_status"] = trim(fmt::sprintf("[%d] %s", severity, statusText));
    } else {
        lm["GCS_status"] = "";
    }

    lm["GCS_bat"]       = trim(fmt::sprintf("%6.2f", battVolt));
    //lm["GCS_CPU"]     = trim(fmt::sprintf("%6.2f%%", cpuLoad));
    lm["GCS_Alt"]       = trim(fmt::sprintf("%4.2f, %4.2f", alt, gpH));

    string gpsFixed = "Lost";
    switch( gpsFixType ) {
    case 0:
    case 1:
        gpsFixed  = "Lost";
        break;

    case 2:
        gpsFixed  = "2D-fixed";
        break;

    case 3:
        gpsFixed  = "3D-fixed";
        break;

    case 4:
        gpsFixed  = "DGPS";
        break;

    case 5:
        gpsFixed  = "RTK";
        break;
    }

    lm["GCS_GPS"]       = trim(fmt::sprintf("%d, %s, %4.2f", nSat, gpsFixed, HDOP_h));
    lm["GCS_heading"]   = trim(fmt::sprintf("%6.1f", gpHeading));

    return 0;
}

int UAS_GCS::gen_listmap_all(String2StringMap &lm)
{
    return 0;
}

int UAS_GCS::timerFunction(void *arg)
{
    // call basic timer function
    UAS_Base::timerFunction(arg);
}


int UAS_GCS::send_ATA_yawOffset(void)
{
    mavlink_gps_raw_int_t  msg_gps_raw;
    mavlink_message_t      msg_mavlink;

    // FIXME: use GPS_RAW package to send yaw offset
    msg_gps_raw.time_usec           = tm_get_us();
    msg_gps_raw.lat                 = (int)( ataYawOffset*100 );
    msg_gps_raw.lon                 = 0;
    msg_gps_raw.alt                 = 0;
    msg_gps_raw.eph                 = 0;
    msg_gps_raw.epv                 = 0;
    msg_gps_raw.vel                 = 0;
    msg_gps_raw.fix_type            = 2;
    msg_gps_raw.satellites_visible  = 0;

    mavlink_msg_gps_raw_int_encode(gcsID, gcsCompID,
                                   &msg_mavlink, &msg_gps_raw);
    m_uasManager->sendMavlinkMsg(&msg_mavlink);

    ataYawOffset_setted = 1;

    printf("Set ATA yaw offset = %f\n", ataYawOffset);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

UAS_Telem::UAS_Telem() : UAS_Base()
{
    // Telemetry
    radioRX_errors      = 0;
    radioFixed          = 0;
    radioRSSI           = 0;
    radioRSSI_remote    = 0;
    radioTXBuf          = 0;
    radioNoise          = 0;
    radioNoise_remote   = 0;
}

UAS_Telem::~UAS_Telem()
{

}

void UAS_Telem::init(void)
{
    UAS_Base::init();
}

void UAS_Telem::release(void)
{
    UAS_Base::release();
}

int UAS_Telem::parseMavlinkMsg(mavlink_message_t *msg)
{
    // FIXME: RSSI min/max need load from configure file
    int rssi_min = 90, rssi_max = 220;

    // for each message type
    switch( msg->msgid ) {
    case MAVLINK_MSG_ID_RADIO_STATUS:
        mavlink_radio_status_t rs;

        mavlink_msg_radio_status_decode(msg, &rs);
        radioRX_errors      = rs.rxerrors;
        radioFixed          = rs.fixed;
        radioRSSI           = rs.rssi;
        radioRSSI_remote    = rs.remrssi;
        radioTXBuf          = rs.txbuf;
        radioNoise          = rs.noise;
        radioNoise_remote   = rs.remnoise;

        // Telemetry RSSI percent
        radioRSSI_per = (radioRSSI - rssi_min)*1.0 / (rssi_max - rssi_min) * 100.0;
        if( radioRSSI_per > 100.0 ) radioRSSI_per = 100.0;
        if( radioRSSI_per < 0.0 )   radioRSSI_per = 0.0;

        radioRSSI_remote_per = (radioRSSI_remote - rssi_min)*1.0 / (rssi_max - rssi_min) * 100.0;
        if( radioRSSI_remote_per > 100.0 ) radioRSSI_remote_per = 100.0;
        if( radioRSSI_remote_per < 0.0 )   radioRSSI_remote_per = 0.0;

        break;
    }

    return 0;
}

int UAS_Telem::gen_listmap_important(String2StringMap &lm)
{
    lm["RSSI"] = trim(fmt::sprintf("%3d(%5.1f%%), %3d(%5.1f%%)",
                                    radioRSSI, radioRSSI_per,
                                    radioRSSI_remote, radioRSSI_remote_per));

    return 0;
}

int UAS_Telem::gen_listmap_all(String2StringMap &lm)
{
    return 0;
}

int UAS_Telem::timerFunction(void *arg)
{
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void UAS_Manager_timerFunc(void *arg)
{
    UAS_Manager *u = (UAS_Manager*) arg;

    u->timerFunction(arg);
}


UAS_Manager::UAS_Manager()
{
    gcsID     = 254;
    gcsCompID = 1;

    mavlinkChan = 0;

    init();

    // create timer
    m_timerCount = 0;

    m_timer = 0;
    startTimer();
}

UAS_Manager::~UAS_Manager()
{
    release();

    stopTimer();

    m_timer = 0;
    m_timerCount = 0;
}

void UAS_Manager::init(void)
{
    m_arrUAS.clear();
    m_mapUAS.clear();

    // init active objs
    m_activeMAV   = NULL;
    m_activeGCS   = NULL;
    m_activeTelem = NULL;

    // initialize timer & mutex for msg wirting
    m_mutexMsgWrite = NULL;

    // output buffer
    m_mutexMsgWrite = new pi::RMutex();
    m_msgBuffer.reserve(1024);

    m_nReadLast = 0;
    m_nWriteLast = 0;
    m_tLast = 0;

    bpsIn  = 0;
    bpsOut = 0;
}

void UAS_Manager::release(void)
{
    // clear UAS array & map
    if( m_arrUAS.size() > 0 ) {
        UAS_Array::iterator it;
        UAS_Base *u;

        for(it=m_arrUAS.begin(); it!=m_arrUAS.end(); it++) {
            u = *it;
            delete u;
        }

        m_arrUAS.clear();
    }

    m_mapUAS.clear();

    // set active objs
    m_activeMAV   = NULL;
    m_activeGCS   = NULL;
    m_activeTelem = NULL;

    // clear msg buffer & mutex
    delete m_mutexMsgWrite;
    m_mutexMsgWrite = NULL;

    m_msgBuffer.clear();
}

void UAS_Manager::reset(void)
{
    // clear UAS array & map
    if( m_arrUAS.size() > 0 ) {
        UAS_Array::iterator it;
        UAS_Base *u;

        for(it=m_arrUAS.begin(); it!=m_arrUAS.end(); it++) {
            u = *it;
            delete u;
        }

        m_arrUAS.clear();
    }

    m_mapUAS.clear();

    // set active objs
    m_activeMAV   = NULL;
    m_activeGCS   = NULL;
    m_activeTelem = NULL;

    // clear message buffer
    m_msgBuffer.clear();
}

int UAS_Manager::startTimer(void)
{
    if( m_timer == 0 ) {
        osa_tm_create(&m_timer, 100, UAS_Manager_timerFunc, this);
    }

    return 0;
}

int UAS_Manager::stopTimer(void)
{
    if( m_timer != 0 ) {
        osa_tm_delete(m_timer);

        m_timer = 0;
    }

    return 0;
}

int UAS_Manager::timerFunction(void *arg)
{
    uint64_t tNow = tm_get_millis();

    m_timerCount ++;

    // for every second
    if( m_timerCount % 10 == 0 ) {
        // stastic communication IO usage
#if 0
        if( m_commManager != NULL ) {
            uint64_t dt = tNow - m_tLast;
            uint64_t ni, no, ci, co;

            ci = m_commManager->getReadBytes();
            co = m_commManager->getWriteBytes();

            ni = ci - m_nReadLast;
            no = co - m_nWriteLast;

            bpsIn  = 8.0*ni / dt;
            bpsOut = 8.0*no / dt;

            m_nReadLast  = ci;
            m_nWriteLast = co;
            m_tLast      = tNow;
        }
#endif

        // send heartbeat
        mavlink_message_t msg_hb;
        mavlink_msg_heartbeat_pack(gcsID, gcsCompID,
                                   &msg_hb,
                                   MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID,
                                   MAV_MODE_MANUAL_ARMED, 0, MAV_STATE_ACTIVE);
        sendMavlinkMsg(&msg_hb);
    }

    // call each uas's timerFunction
    for(UAS_Array::iterator it=m_arrUAS.begin(); it!=m_arrUAS.end(); it++) {
        UAS_Base *u = *it;
        u->timerFunction(u);
    }

    return 0;
}



// FIXME: parse MAVLINK message based on sysid
//      0 ~  49: MAV
//     50 ~ 249: Telemetry
//    250 ~ 255: GCS
int UAS_Manager::parseMavlinkMsg(mavlink_message_t *msg)
{
    // find exist item
    UAS_IDMap::iterator it;

    it = m_mapUAS.find(msg->sysid);
    if( it != m_mapUAS.end() ) {
        return it->second->parseMavlinkMsg(msg);
    }

    // create new item
    UAS_Base *u;

    if( msg->sysid < 50 ) {
        u = new UAS_MAV;
        u->uasType = UAS_TYPE_MAV;
        if( m_activeMAV == NULL ) m_activeMAV = u;
    } else if ( msg->sysid >= 50 && msg->sysid < 250 ) {
        u = new UAS_Telem;
        u->uasType = UAS_TYPE_TELEM;
        if( m_activeTelem == NULL ) m_activeTelem = u;
    } else {
        u = new UAS_GCS;
        u->uasType = UAS_TYPE_GCS;
        if( m_activeGCS == NULL ) m_activeGCS = u;
    }

    u->ID           = msg->sysid;
    u->compID       = msg->compid;
    u->gcsID        = gcsID;
    u->gcsCompID    = gcsCompID;
    u->m_uasManager = this;

    m_arrUAS.push_back(u);
    m_mapUAS.insert(std::make_pair(msg->sysid, u));

    return u->parseMavlinkMsg(msg);
}

int UAS_Manager::gen_listmap_important(String2StringMap &lm)
{
    UAS_Array::iterator it;

    for(it=m_arrUAS.begin(); it!=m_arrUAS.end(); it++) {
        (*it)->gen_listmap_important(lm);
    }

    // set communication stastics
#if 0
    char buf[256];
    sprintf(buf, "R:%6.2f kbps, W: %6.2f kbps", bpsIn, bpsOut);
    lm["COM_Stat"] = buf;
#endif

    return 0;
}

int UAS_Manager::gen_listmap_all(String2StringMap &lm)
{
    UAS_Array::iterator it;

    for(it=m_arrUAS.begin(); it!=m_arrUAS.end(); it++) {
        (*it)->gen_listmap_all(lm);
    }

    return 0;
}


UAS_Array* UAS_Manager::get_uas(void)
{
    return &m_arrUAS;
}

UAS_Base* UAS_Manager::get_uas(int id)
{
    UAS_IDMap::iterator it;

    it = m_mapUAS.find(id);
    if( it != m_mapUAS.end() ) {
        return it->second;
    } else {
        return NULL;
    }
}

int UAS_Manager::get_uas(UAS_Type t, UAS_Array &uasArr)
{
    UAS_Array::iterator it;

    for(it=m_arrUAS.begin(); it!=m_arrUAS.end(); it++) {
        if( (*it)->uasType == t ) {
            uasArr.push_back(*it);
        }
    }

    return 0;
}


int UAS_Manager::get_mav(UAS_Array &uasArr)
{
    // clear input array
    uasArr.clear();

    // get all MAVs
    return get_uas(UAS_TYPE_MAV, uasArr);
}

int UAS_Manager::get_gcs(UAS_Array &uasArr)
{
    // clear input array
    uasArr.clear();

    // get all GCSs
    return get_uas(UAS_TYPE_GCS, uasArr);
}

int UAS_Manager::get_telem(UAS_Array &uasArr)
{
    // clear input array
    uasArr.clear();

    // get all TELEMs
    return get_uas(UAS_TYPE_TELEM, uasArr);
}

UAS_MAV* UAS_Manager::get_active_mav(void)
{
    return (UAS_MAV*) m_activeMAV;
}

UAS_GCS* UAS_Manager::get_active_gcs(void)
{
    return (UAS_GCS*) m_activeGCS;
}

UAS_Telem* UAS_Manager::get_active_telem(void)
{
    return (UAS_Telem*) m_activeTelem;
}

UAS_MAV* UAS_Manager::set_active_mav(int id)
{
    UAS_MAV *m = (UAS_MAV*) get_uas(id);
    if( m != NULL ) {
        m_activeMAV = m;
        return m;
    }

    return NULL;
}

int UAS_Manager::sendMavlinkMsg(mavlink_message_t *msg)
{
    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];

    m_mutexMsgWrite->lock();

    // Write message into buffer, prepending start sign
    static uint8_t messageKeys[256] = MAVLINK_MESSAGE_CRCS;
    mavlink_finalize_message_chan(msg, gcsID, gcsCompID,
                                  mavlinkChan,
                                  msg->len, messageKeys[msg->msgid]);

    int len = mavlink_msg_to_send_buffer(buffer, msg);

    for(int i=0; i<len; i++) m_msgBuffer.push_back(buffer[i]);

    m_mutexMsgWrite->unlock();

    return 0;
}

int UAS_Manager::put_msg_buff(uint8_t *buf, int len)
{
    RMutex m(m_mutexMsgWrite);

    for(int i=0; i<len; i++) m_msgBuffer.push_back(buf[i]);

    return 0;
}

int UAS_Manager::get_msg_buff(uint8_t *buf, int *len)
{
    RMutex m(m_mutexMsgWrite);

    int l = m_msgBuffer.size();
    if( *len > l ) *len = l;

    for(int i=0; i<*len; i++) buf[i] = m_msgBuffer[i];
    m_msgBuffer.clear();

    return 0;
}
