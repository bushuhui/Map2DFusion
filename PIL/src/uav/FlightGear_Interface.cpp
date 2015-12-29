
#include "base/Svar/Svar.h"
#include "network/Socket++.h"

#include "FlightGear_Interface.h"


using namespace std;
using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

FlightGear_Transfer::FlightGear_Transfer()
{
    SvarWithType<FlightGear_Transfer*>::instance()["FlightGear_Transfer.ptr"] = this;

    m_pData = new RSocket();
}

FlightGear_Transfer::~FlightGear_Transfer()
{
    RSocket *s = (RSocket *) m_pData;

    if( s != NULL ) {
        delete s;
        m_pData = NULL;
    }

    SvarWithType<FlightGear_Transfer*>::instance()["FlightGear_Transfer.ptr"] = NULL;
}

int FlightGear_Transfer::connect(const std::string &hn, int port)
{
    RSocket *s = (RSocket *) m_pData;

    if( s == NULL ) return -1;

    return s->startClient(hn, port, SOCKET_UDP);
}

int FlightGear_Transfer::close(void)
{
    RSocket *s = (RSocket *) m_pData;

    if( s == NULL ) return -1;

    return s->close();
}

int FlightGear_Transfer::isRunning(void)
{
    RSocket *s = (RSocket *) m_pData;

    if( s == NULL ) return 0;
    else return 1;
}

int FlightGear_Transfer::trans(FGNetFDM *fdm)
{
    RSocket *s = (RSocket *) m_pData;

    if( s == NULL ) return -1;

    return s->send((uint8_t*) fdm, sizeof(*fdm));
}
