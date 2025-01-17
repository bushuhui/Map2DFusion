/******************************************************************************

  Robot Toolkit ++ (RTK++)

  Copyright (c) 2007-2013 Shuhui Bu <bushuhui@nwpu.edu.cn>
  http://www.adv-ci.com

  ----------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/


#ifndef __SOCKET_PP_H__
#define __SOCKET_PP_H__

#include <string.h>
#include <stdint.h>
#include <string>

#include "base/osa/osa++.h"
#include "base/Svar/DataStream.h"

namespace pi {


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
enum RSocketType
{
    SOCKET_TCP,
    SOCKET_UDP
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class RSocketAddress
{
public:
    RSocketAddress() {
        port = -1;
        addr_inet = 0;
        type = SOCKET_TCP;
        memset(addr_str, 0, sizeof(addr_str));
    }

    ~RSocketAddress() {
        port = -1;
        addr_inet = 0;
        type = SOCKET_TCP;
        memset(addr_str, 0, sizeof(addr_str));
    }

public:
    int         port;                   ///> port number
    uint32_t    addr_inet;              ///> uint32_t address
    char        addr_str[32];           ///> address string
    RSocketType type;                   ///> socket type
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// \brief The RSocket class support TCP/UDP
///
class RSocket
{
public:
    RSocket();
    virtual ~RSocket();

    // socket creation & close
    int create(void);
    int close(void);

    // start server or client
    int startServer(int port, RSocketType t=SOCKET_TCP);
    int startClient(std::string host, int port, RSocketType t=SOCKET_TCP);
    int startClient(uint32_t addr, int port, RSocketType t=SOCKET_TCP);

    // Data Transmission
    int send(uint8_t *dat, int len);
    int send(std::string& msg);
    int send(RDataStream &ds);

    int recv(uint8_t *dat, int len);
    int recv(std::string& msg, int maxLen = 4096);
    int recv(RDataStream &ds);
    int recv_until(uint8_t *dat, int len);

    // server functions
    int bind(int port);
    int listen(void);
    int accept(RSocket& s);

    // client functions
    int connect(std::string host, int port);
    int connect(uint32_t addr, int port);

    // get address
    int getMyAddress(RSocketAddress &a);
    int getClientAddress(RSocketAddress &a);

    int setNonBlocking(int nb = 1);

    int isOpened(void) {
        return m_sock != -1;
    }

    bool isSever(){return m_server;}


protected:
    int             m_sock;                         ///< socket file descriptor
    int             m_server;                       ///< server or client
    RSocketType     m_socketType;                   ///< socket type
    void*           m_priData;                      ///< socket private data

    int             m_maxConnections;               ///< maximum connections (default: 100)
    int             m_maxHostname;                  ///< maximum length of hostname (default: 1024)
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// \brief The simple data transfer class which uses UDP.
///     server - receive data
///     client - send data
///
/// \ref
///     UDP buffer size:
///         Linux           131071
///         Windows         No known limit
///         Solaris         262144
///         FreeBSD, Darwin	262144
///         AIX             1048576
///
///     https://access.redhat.com/documentation/en-US/JBoss_Enterprise_Web_Platform/5/html/Administration_And_Configuration_Guide/jgroups-perf-udpbuffer.html
///
class NetTransfer_UDP : public RThread
{
public:
    NetTransfer_UDP() {
        m_isServer      = 0;
        m_isConnected   = 0;
        m_port          = 30000;

        m_magicNum      = 0x49A5825A;
        m_bufLen        = 60*1024;
        //m_bufLen      = 128*1024;
    }

    virtual ~NetTransfer_UDP() {
        close();
    }

    ///
    /// \brief thread_func - receiving thread
    ///
    /// \param arg          - thread argument
    ///
    /// \return
    ///
    virtual int thread_func(void *arg);


    ///
    /// \brief open the UDP transfer - for simple transmitting message
    ///
    /// \param isServer     - 0: client, send message
    ///                       1: server, receive message
    /// \param port         - port number (default 30000)
    /// \param addr         - remote address (for client)
    ///
    /// \return
    ///         0           - success
    ///         other       - failed
    ///
    virtual int open(int isServer=0, int port=30000, std::string addr="127.0.0.1");

    ///
    /// \brief close the UDP transfer
    ///
    /// \return
    ///
    virtual int close(void);

    ///
    /// \brief send data to remote terminal
    ///
    /// \param dat          - message buffer
    /// \param len          - message length
    /// \param msgid        - message ID
    ///
    /// \return
    ///     if success return sended message length
    ///     -1              - failed (maybe UDP not opened)
    ///
    virtual int send(ru8 *dat, ru32 len, ru32 msgid=0);


    ///
    /// \brief recved message processing function (user must override this function)
    ///
    /// \param dat          - message buffer
    /// \param len          - message length
    /// \param msgid        - message ID
    ///
    /// \return
    ///     0               - success
    ///
    virtual int recv(ru8 *dat, ru32 len, ru32 msgid=0) = 0;


    ///
    /// \brief return connection is established or not
    ///
    /// \return
    ///     1               - connected
    ///     0               - not connected
    ///
    int isOpened(void) { return m_isConnected; }

protected:
    RSocket         m_socket;                       ///< socket obj
    int             m_isServer;                     ///< 1-server(RX), 0-client(TX)
    int             m_isConnected;                  ///< connected or not

    std::string     m_addr;                         ///< remote terminal address
    int             m_port;                         ///< UDP port

    ru32            m_magicNum;                     ///< magic number for TX/RX
    ru32            m_bufLen;                       ///< TX/RX buffer length (128k bytes)
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int inet4_addr_str2i(std::string na, ru32 &nai);
int inet4_addr_i2str(ru32 nai, std::string &na);

int inet4_addr_ni2hi(ru32 ni, ru32 &hi);
int inet4_addr_hi2ni(ru32 hi, ru32 &ni);

int inet4_port_n2h(ru16 np, ru16 &hp);
int inet4_port_h2n(ru16 hp, ru16 &np);



void convByteOrder_h2n_16(void *s, void *d);
void convByteOrder_n2h_16(void *s, void *d);
void convByteOrder_h2n_32(void *s, void *d);
void convByteOrder_n2h_32(void *s, void *d);
void convByteOrder_h2n_64(void *s, void *d);
void convByteOrder_n2h_64(void *s, void *d);


template<class T>
T convByteOrder_h2n(T s)
{
    if( sizeof(T) == 2 ) {
        uint16_t *_s, *_d;
        T        v;

        _s = (uint16_t*) &s;
        _d = (uint16_t*) &v;

        convByteOrder_h2n_16(_s, _d);
        return v;
    } else if( sizeof(T) == 4 ) {
        uint32_t *_s, *_d;
        T        v;

        _s = (uint32_t*) &s;
        _d = (uint32_t*) &v;

        convByteOrder_h2n_32(_s, _d);
        return v;
    } else if( sizeof(T) == 8 ) {
        uint64_t *_s, *_d;
        T        v;

        _s = (uint64_t*) &s;
        _d = (uint64_t*) &v;

        convByteOrder_h2n_64(_s, _d);
        return v;
    }
}

} // end of namespace pi

#endif // end of __SOCKET_PP_H__

