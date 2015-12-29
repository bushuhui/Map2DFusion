#ifndef __FLIGHTDATA_TRANSFER__
#define __FLIGHTDATA_TRANSFER__

#include <stdint.h>

#include <string>
#include <vector>
#include <deque>

#include <base/osa/osa++.h>
#include <network/Socket++.h>


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct FlightData_Simp
{
    double      lat;                ///< latitude
    double      lng;                ///< longitude
    double      alt;                ///< alititude

    double      H;                  ///< hight above ground

    double      yaw, pitch, roll;   ///< attitude

    uint8_t     crc[8];

public:
    FlightData_Simp() {
        lat         = 0;
        lng         = 0;
        alt         = 0;
        H           = 0;

        yaw         = 0;
        pitch       = 0;
        roll        = 0;

        for(int i=0; i<4; i++) crc[i] = 0;
    }

    int calcCRC(void) {
        uint8_t *p = (uint8_t*) this;

        int n = sizeof(this) - 8;
        int sum = 0;

        for(int i=0; i<n; i++) {
            sum += p[i];
        }

        sum = sum % 256;
        crc[0] = sum;

        return 0;
    }

    int checkCRC(void) {
        uint8_t *p = (uint8_t*) this;

        int n = sizeof(this) - 8;
        int sum = 0;

        for(int i=0; i<n; i++) {
            sum += p[i];
        }

        sum = sum % 256;
        if( sum != crc[0] ) return -1;
        else return 0;
    }

    int print(void) {
        printf("    lat, lng, alt, H  = %12f, %12f - %12f (%12f)\n", lat, lng, alt, H);
        printf("    yaw, pitch, roll  = %12f %12f %12f\n", yaw, pitch, roll);

        return 0;
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef std::vector<FlightData_Simp> FlightDataArray;

int saveFlightData(const std::string &fn, FlightDataArray &fda);
int loadFlightData(const std::string &fn, FlightDataArray &fda);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class FlightData_Transfer : public pi::RThread
{
public:
    FlightData_Transfer();
    virtual ~FlightData_Transfer();

    virtual int thread_func(void *arg=NULL);

    int begin(int port);
    int stop(void);
    int isRunning(void) { return m_bOpened; }

    int sendPOS(FlightData_Simp *d);

private:
    pi::RSocket     m_socket;
    int             m_bOpened;

    std::deque<FlightData_Simp> m_dataQueue;
    pi::RMutex                  m_mutex;
};

#endif // end of __FLIGHTDATA_TRANSFER__
