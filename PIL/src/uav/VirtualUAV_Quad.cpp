
#include "base/utils/utils.h"
#include "base/types/SO3.h"
#include "network/Socket++.h"
#include "hardware/Gps/utils_GPS.h"

#include "VirtualUAV_Quad.h"


using namespace std;
using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define PI          (3.1415926)
#define D2R         (3.1415926 / 180.0)
#define R2D         (180.0/3.1415926)
#define M2FT        (0.3048)                //transfer (m/sec^2) to (ft/sec^2)
#define R_EARTH     (6378100.0)


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

VirtualUAV_Quad::VirtualUAV_Quad()
{
    init();
}

VirtualUAV_Quad::~VirtualUAV_Quad()
{
    release();
}

int VirtualUAV_Quad::init()
{
    string uavName = fmt::sprintf("vuav_%d", ID);

    sim_m     = svar.GetDouble(uavName + ".m", 10.0);
    sim_dragK = svar.GetDouble(uavName + ".dragK", 0.25);

    sim_Ax = 0;
    sim_Ay = 0;
    sim_Az = 0;
    sim_Vx = 0;
    sim_Vy = 0;
    sim_Vz = 0;

    sim_tLast       = 0;
    sim_tNow        = 0;
    sim_totalThrust = 0;

    return 0;
}

int VirtualUAV_Quad::release()
{
    return 0;
}

int VirtualUAV_Quad::timerFunction(void *arg)
{
    return VirtualUAV::timerFunction(arg);
}

int VirtualUAV_Quad::simulation(pi::JS_Val *jsv)
{
    double      js0, js1, js2, js3;
    double      js_deadzone;
    double      lat1, lng1;
    double      dt;

    // time
    sim_tNow = tm_getTimeStamp();
    dt = sim_tNow - sim_tLast;
    if( dt > 100 ) {
        sim_tLast = sim_tNow;
        return -1;
    }
    sim_tLast = sim_tNow;

    // process joystick values
    js_deadzone = 0.015;
    js0 = jsv->AXIS[0];         // roll
    js1 = jsv->AXIS[1];         // pitch
    js2 = jsv->AXIS[2];         // thrust
    js3 = jsv->AXIS[3];         // yaw

    if( js0 > -js_deadzone && js0 < js_deadzone ) js0 = 0;
    if( js1 > -js_deadzone && js1 < js_deadzone ) js1 = 0;
    if( js2 > -js_deadzone && js2 < js_deadzone ) js2 = 0;
    if( js3 > -js_deadzone && js3 < js_deadzone ) js3 = 0;


    // calculate roll, pitch, yaw
    double _yaw, _roll, _pitch;

    _roll  = 30.0 * js0 * D2R;
    _pitch = 30.0 * js1 * D2R;
    _yaw   = yaw * D2R;
    _yaw  += 30.0 * js3 * D2R*dt;


    // rotation matrix of Euler
    // FIXME: angle need to be processed
    SO3d convert;
    convert.FromEuler(-_pitch, -_yaw + M_PI/2, _roll);

    // calc  D_earth
    Point3d V_earth(sim_Vx, sim_Vy, sim_Vz);
    if( fabs(js0) < 0.05 && fabs(js1) < 0.05 && fabs(js2) < 0.05  ) {
        V_earth = V_earth*0.3;
    }

    Point3d D_earth = -V_earth * V_earth.norm()*sim_dragK;

    // calc F_earth
    //Point3d F_body(0.0,0.0,double(G*(1-js2)));
    Point3d F_body(0,0,1);
    Point3d F_earth =  convert * F_body;

    double G = sim_m * 9.8;
    double fVertical = G*(1-js2);
    if( fVertical < G/10.0 ) fVertical = G/10.0;

    double fratio = 1.0 / F_earth.z;
    F_earth = F_earth*(fratio*fVertical);

    // calc TotalThrust
    sim_totalThrust = fVertical;

    // calc  A_earth
    Point3d G_earth = Point3d(0, 0, -G);
    Point3d A_earth = (F_earth + G_earth + D_earth)/sim_m;

    // calc  V_earth
    V_earth  = V_earth + A_earth*dt;

    // process speed, FIXME: need set configure for maximum speed
    if( V_earth.x > 20.0 ) V_earth.x = 20; if( V_earth.x < -20.0 ) V_earth.x = -20.0;
    if( V_earth.y > 20.0 ) V_earth.y = 20; if( V_earth.y < -20.0 ) V_earth.y = -20.0;
    if( V_earth.z > 20.0 ) V_earth.z = 20; if( V_earth.z < -20.0 ) V_earth.z = -20.0;

    sim_Vx = V_earth.x;
    sim_Vy = V_earth.y;
    sim_Vz = V_earth.z;

    // calculate x,y,z offset in earth frame
    Point3d P_earth(0, 0, gpH);
    P_earth = P_earth + V_earth*dt;
    // FIXME: need DEM data to prevent below ground
    if( P_earth.z < 0 ) P_earth.z = 0;

    // convert dx,dy to lng, lat
    calcLngLatFromDistance(
                gpLon, gpLat,
                P_earth.x, P_earth.y,
                lng1, lat1);

    // update VirtualUAV information
    yawSpd          = (yaw * D2R - _yaw) / dt;
    rollSpd         = (roll * D2R - _roll) / dt;
    pitchSpd        = (pitch * D2R - _pitch ) / dt;
    yaw             = angleNormalize(_yaw * R2D);
    roll            = _roll  * R2D;
    pitch           = _pitch * R2D;

    gpLon           = lng1;
    gpLat           = lat1;
    gpH             = P_earth.z;
    gpAlt           = homeAlt + gpH;
    gpHeading       = yaw;

    gpsTime         = sim_tNow * 1e6;
    lat             = gpLat;
    lon             = gpLon;
    alt             = gpAlt;
    HDOP_h          = 1;
    HDOP_v          = 1.5;
    gpsGroundSpeed  = sqrt(sim_Vx*sim_Vx + sim_Vy*sim_Vy);
    gpsFixType      = 3;
    nSat            = 20;

    systimeUnix     = sim_tNow * 1e6;
    bootTime        = (sim_tNow - m_tmStart) * 1000;
    cpuLoad         = 50.0;
    battVolt        = 12.0;
    battCurrent     = 10.0;
    battRemaining   = 80.0;
    commDropRate    = 0.0;

    // update Flightgear data
    m_flightGearData.altitude  = gpAlt;
    m_flightGearData.longitude = gpLon;
    m_flightGearData.latitude  = gpLat;

    m_flightGearData.A_X_pilot = A_earth.x;
    m_flightGearData.A_Y_pilot = A_earth.y;
    m_flightGearData.A_Z_pilot = A_earth.z;

    m_flightGearData.v_body_u  = V_earth.x;
    m_flightGearData.v_body_v  = V_earth.y;
    m_flightGearData.v_body_w  = V_earth.z;

    // print information
    if( svar.GetInt("VirtualUAV.ShowSimulationInfo", 0) ) {
        printf("js               = %f %f %f %f\n", js0, js1, js2, js3);
        printf("dt               = %f\n", dt);
        printf("roll, pitch, yaw = %f %f %f\n", roll, pitch, yaw);

        cout<<"SO3              = " << convert << endl;
        cout<<"D_earth          = " << D_earth << endl;
        cout<<"F_earth          = " << F_earth << "(" << fratio << ")\n";
        cout<<"A_earth          = " << A_earth << endl;
        cout<<"V_earth          = " << V_earth << endl;
        cout<<"d(x,y,z)         = " << P_earth << endl;

        printf("totalThrust      = %f\n", sim_totalThrust);
        printf("Position         = %f %f %f\n",
               m_flightGearData.latitude, m_flightGearData.longitude, m_flightGearData.altitude);
        printf("\n");
    }

    return 0;
}

int VirtualUAV_Quad::toFlightGear(FGNetFDM *fgData)
{
    FGNetFDM &fdm = *fgData;

    fdm.phi                 = convByteOrder_h2n<float>(roll  * D2R );
    fdm.theta               = convByteOrder_h2n<float>(pitch * D2R );
    fdm.psi                 = convByteOrder_h2n<float>(yaw   * D2R );

    fdm.A_X_pilot           = convByteOrder_h2n<float>(m_flightGearData.A_X_pilot * M2FT);
    fdm.A_Y_pilot           = convByteOrder_h2n<float>(m_flightGearData.A_Y_pilot * M2FT);
    fdm.A_Z_pilot           = convByteOrder_h2n<float>(m_flightGearData.A_Z_pilot * M2FT);    //Z accel in body frame ft/sec^2

    fdm.v_body_u            = convByteOrder_h2n<float>(m_flightGearData.v_body_u);
    fdm.v_body_v            = convByteOrder_h2n<float>(m_flightGearData.v_body_v);
    fdm.v_body_w            = convByteOrder_h2n<float>(m_flightGearData.v_body_w);

    fdm.longitude           = convByteOrder_h2n<double>(m_flightGearData.longitude * D2R);
    fdm.latitude            = convByteOrder_h2n<double>(m_flightGearData.latitude  * D2R);
    fdm.altitude            = convByteOrder_h2n<double>(m_flightGearData.altitude);

    float rpm               = (sim_totalThrust/sim_m)*1000;
    fdm.num_engines         = convByteOrder_h2n<uint32_t>(4);
    fdm.rpm[0]              = convByteOrder_h2n<float>(rpm);
    fdm.rpm[1]              = convByteOrder_h2n<float>(rpm);
    fdm.rpm[2]              = convByteOrder_h2n<float>(rpm);
    fdm.rpm[3]              = convByteOrder_h2n<float>(rpm);

    fdm.num_tanks           = convByteOrder_h2n<uint32_t>(1);
    fdm.fuel_quantity[0]    = convByteOrder_h2n<float>(100.0);

    fdm.num_wheels          = convByteOrder_h2n<uint32_t>(3);

    fdm.cur_time            = convByteOrder_h2n<uint32_t>(tm_getTimeStampUnix());
    fdm.warp                = convByteOrder_h2n<uint32_t>(1);

    fdm.visibility          = convByteOrder_h2n<float>(m_flightGearData.visibility);

    fdm.version             = convByteOrder_h2n<uint32_t>(FG_NET_FDM_VERSION);

    return 0;
}
