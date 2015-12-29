
#include "base/utils/utils.h"
#include "base/Svar/Svar_Inc.h"

#include "GPS.h"

using namespace std;

namespace pi {

/**
    .BaseDate  - Time shift
    .port_type:
            0:real uart
            1:file uart
            2:data_manager saved file
            3:NULL
    .port: UART port or file name, default: /dev/ttyUSB0
    .port_speed: baud rate, default 115200
    .date_type: binary data or text file (only for .port_type == 1)

*/

GPS::GPS(string name)
    :pathTable(0.1)
{
    int64_t dt_time = svar.get_var<int64_t>(name+".BaseData",0);
    if( dt_time ) {
        DateTime dt;
        dt.fromTimeStamp(dt_time);
        setBaseDate(dt);

        //m_baseDate.fromTimeStamp(dt_time);
    }

    portType = svar.GetInt(name+".port_type", 3);
    string port = svar.GetString(name+".port", "/dev/ttyUSB0");
    if(portType == 2) //treat as data_manager,no thread will be run
    {
        load(port.c_str());
        for(int i=0;i<POS_DataManager::length();i++)
        {
            POS_Data pt=POS_DataManager::at(i);
            if(pt.correct)
            {
                pathTable.Add(pt.time.toTimeStampF(),pi::Point3d(pt.x,pt.y,pt.z));
            }
        }
        return;
    }

    //treat as uart
    if( portType == 0 )
        uart = new UART;
    else if( portType == 1 )
        uart =new VirtualUART;

    if( uart ) {
        uart->port_name = port;
        uart->baud_rate = svar.GetInt(name+".port_speed", 115200);
    }

    string  fn_base = svar.GetString("fn_autosave", "");
    m_fnAutoSave = auto_filename(fn_base);

    dataType = svar.GetInt(name+".data_type", 0);
    if(dataType == 0 )   m_fnAutoSave = m_fnAutoSave + ".bin";
    else                 m_fnAutoSave = m_fnAutoSave + ".txt";

    // begin POS recving thread
    if( uart )
        begin();
}

GPS::~GPS()
{
//    uart->close();
}

void GPS::addFrame(POS_Data& frame)
{
    processData(frame);
    frameQueue.push_back(frame);
    pathTable.Add(frame.time.toTimeStampF(), pi::Point3d(frame.x,frame.y,frame.z));
}

} // end of namespace pi
