#include <stdio.h>
#include <stdlib.h>

#include <base/time/Time.h>

#include "FlightData_Transfer.h"

using namespace std;
using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int saveFlightData(const string &fn, FlightDataArray &fda)
{
    FILE        *fp;

    fp = fopen(fn.c_str(), "wb");
    if( fp == NULL ) {
        dbg_pe("Can not open file: %s", fn.c_str());
        return -1;
    }

    if( fda.size() <= 0 ) {
        dbg_pe("No flight data to save!");
        return -2;
    }

    // save flight data
    int32_t nData = fda.size();

    fwrite(&nData, sizeof(int32_t), 1, fp);
    fwrite(fda.data(), sizeof(FlightData_Simp)*nData, 1, fp);

    fclose(fp);

    return 0;
}

int loadFlightData(const string &fn, FlightDataArray &fda)
{
    FILE        *fp;
    int32_t     nData;
    int         ret;

    fp = fopen(fn.c_str(), "rb");
    if( fp == NULL ) {
        dbg_pe("Can not open file: %s", fn.c_str());
        return -1;
    }

    // read data number
    ret = fread(&nData, sizeof(int32_t), 1, fp);
    if( 1 != ret ) {
        dbg_pe("Read data number failed! %d", ret);
        return -2;
    }

    // read flight data
    fda.resize(nData);
    ret = fread(fda.data(), sizeof(FlightData_Simp), nData, fp);
    if( ret != nData ) {
        dbg_pw("Read data incorrect! %d", ret);
        return -3;
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

FlightData_Transfer::FlightData_Transfer()
{
    m_bOpened = 0;
}

FlightData_Transfer::~FlightData_Transfer()
{
    if( m_bOpened ) {
        m_socket.close();
        kill();
    }

    m_bOpened = 0;
}

int FlightData_Transfer::thread_func(void *arg)
{
    int             data_len;
    int             ret;

    data_len = sizeof(FlightData_Simp);

    while( getAlive() ) {
        RSocket new_socket;

        if( 0 != m_socket.accept(new_socket) ) {
            dbg_pe("server.accept failed!");
            continue;
        }

        while( getAlive() ) {
            // send data to remote client
            m_mutex.lock();

            if( m_dataQueue.size() <= 0 ) {
                m_mutex.unlock();

                tm_sleep(5);
                continue;
            }

            FlightData_Simp d = m_dataQueue.front();
            m_dataQueue.pop_front();
            m_mutex.unlock();

            ret = new_socket.send((uint8_t*) &d, data_len);

            if( ret < 0 ) {
                dbg_pe("Connection lost!");
                break;
            } else if( ret < data_len ) {
                dbg_pw("Send data not correct!");
                continue;
            }
        }

        if( !getAlive() ) return 0;

FLIGHT_DATA_NEW_CONNECTION:
        {
            // clear data queue
            pi::RMutex m(&m_mutex);
            m_dataQueue.clear();
        }
    }

    return 0;
}

int FlightData_Transfer::begin(int port)
{
    if( m_bOpened ) {
        dbg_pe("socket is open. Please close first!");
        return -1;
    }

    if( m_socket.startServer(port) != 0 ) return -2;

    start(NULL);

    return 0;
}


int FlightData_Transfer::stop(void)
{
    if( !m_bOpened ) {
        dbg_pe("Socket is not open!");
        return -1;
    }

    // stop thread
    setAlive(0);
    wait(10);
    kill();

    // close socket
    m_socket.close();

    m_bOpened = 0;
}

int FlightData_Transfer::sendPOS(FlightData_Simp *d)
{
    pi::RMutex  m(&m_mutex);

    m_dataQueue.push_back(*d);

    return 0;
}
