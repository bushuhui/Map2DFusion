#ifndef GPS_H
#define GPS_H

#include "POS_reader.h"
#include "PathTable.h"

namespace pi {

class GPS : public POS_DataManager, public POS_Reader
{
public:
    /** When  means a file,then we treat it as a data manager,otherwise we open a uart
    port_type: 0:real uart 1:file uart 2:data_manager
    */
    GPS(std::string name="GPS");

    bool hasTime(int64_t tm){return (tm<=tsMax&&tm>=tsMin);}
    virtual ~GPS();

    virtual void addFrame(POS_Data& frame);

    FastPathTable  pathTable;
};

} // end of namespace pi

#endif // GPS_H
