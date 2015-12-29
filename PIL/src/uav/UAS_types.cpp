
#include <algorithm>

#include "base/osa/osa++.h"
#include "base/utils/utils.h"

#include "UAS_types.h"
#include "UAS.h"

using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

AP_MessageQueue::AP_MessageQueue()
{
    m_mutex = new RMutex();
}

AP_MessageQueue::~AP_MessageQueue()
{
    if( m_mutex != NULL ) {
        delete m_mutex;
        m_mutex = NULL;
    }

    m_msgQueue.clear();
}

int AP_MessageQueue::push(mavlink_message_t &m)
{
    m_mutex->lock();
    m_msgQueue.push_back(m);
    m_mutex->unlock();

    return 0;
}

int AP_MessageQueue::pop(mavlink_message_t &m)
{
    int n, ret = 0;

    m_mutex->lock();

    n = m_msgQueue.size();
    if( n > 0 ) {
        m = m_msgQueue.front();
        m_msgQueue.pop_front();
        ret = 0;
    } else {
        ret = -1;
    }

    m_mutex->unlock();

    return ret;
}

int AP_MessageQueue::size(void)
{
    int n;

    m_mutex->lock();
    n = m_msgQueue.size();
    m_mutex->unlock();

    return n;
}



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

AP_ParamItem::AP_ParamItem()
{
    init();
}

AP_ParamItem::~AP_ParamItem()
{
    release();
}

void AP_ParamItem::init()
{
    index = 0;
    memset(id, 0, 17);
    type = MAV_PARAM_TYPE_REAL32;
    value = 0;

    modified = 0;
}

void AP_ParamItem::release()
{
    index = 0;
    memset(id, 0, 17);
    type = MAV_PARAM_TYPE_REAL32;
    value = 0;

    modified = 0;
}

int8_t      AP_ParamItem::toInt8(void)
{
    return (int8_t) value;
}

uint8_t     AP_ParamItem::toUint8(void)
{
    return (uint8_t) value;
}

int16_t     AP_ParamItem::toInt16(void)
{
    return (int16_t) value;
}

uint16_t    AP_ParamItem::toUint16(void)
{
    return (uint16_t) value;
}

int32_t     AP_ParamItem::toInt32(void)
{
    return (int32_t) value;
}

uint32_t    AP_ParamItem::toUint32(void)
{
    return (uint32_t) value;
}

float       AP_ParamItem::toFloat(void)
{
    return value;
}

void        AP_ParamItem::fromInt8(int8_t v)
{
    value = v;
}

void        AP_ParamItem::fromUint8(uint8_t v)
{
    value = v;
}

void        AP_ParamItem::fromInt16(int16_t v)
{
    value = v;
}

void        AP_ParamItem::fromUint16(uint16_t v)
{
    value = v;
}

void        AP_ParamItem::fromInt32(int32_t v)
{
    value = v;
}

void        AP_ParamItem::fromUint32(uint32_t v)
{
    value = v;
}

void        AP_ParamItem::fromFloat(float v)
{
    value = v;
}



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void AP_ParamArray_timerFunc(void *arg)
{
    AP_ParamArray *pa = (AP_ParamArray*) arg;

    pa->timerFunction(arg);
}

AP_ParamArray::AP_ParamArray()
{
    m_mutex = new pi::RMutex();
    m_timer = 0;
    m_uas = NULL;

    init();
}

AP_ParamArray::~AP_ParamArray()
{
    release();

    // delete mutex
    if( m_mutex != NULL ) {
        delete m_mutex;
        m_mutex = NULL;
    }

    // stop timer
    if( m_timer != 0 ) {
        osa_tm_delete(m_timer);
        m_timer = 0;
    }

    m_uas = NULL;
}

void AP_ParamArray::init(void)
{
    m_nParam = 0;
    m_nParamRW = 0;
    m_idxCurrent = 0;
    m_idxReading = 0;
    m_idxWritting = 0;

    m_lstAll.clear();
    m_lstWrite.clear();
    m_mapIndex.clear();
    m_mapID.clear();

    m_stParamReading = IDLE;
    m_tLastReading = 0;

    m_bLoaded = 0;
}

void AP_ParamArray::release(void)
{
    std::vector<AP_ParamItem*>::iterator it;

    for(it=m_lstAll.begin(); it!=m_lstAll.end(); it++) {
        delete *it;
    }
    m_lstAll.clear();
    m_lstWrite.clear();
    m_lstRead.clear();

    m_mapIndex.clear();
    m_mapID.clear();

    m_bLoaded = 0;
}

int AP_ParamArray::clear(void)
{
    m_mutex->lock();

    release();

    m_nParam = 0;
    m_nParamRW = 0;
    m_idxCurrent = 0;
    m_idxReading = 0;
    m_idxWritting = 0;

    m_stParamReading = IDLE;
    m_tLastReading = 0;

    m_mutex->unlock();

    return 0;
}

int AP_ParamArray::lock(void)
{
    m_mutex->lock();

    return 0;
}

int AP_ParamArray::unlock(void)
{
    m_mutex->unlock();

    return 0;
}


int AP_ParamArray::requireParameters(int clearOld)
{
    mavlink_param_request_list_t packet;
    mavlink_message_t msg;

    if( m_stParamReading != IDLE ) return -1;

    // clear old contents
    if( clearOld ) clear();

    // send request msg
    if( 1 ) {
        packet.target_system    = m_uas->ID;
        packet.target_component = m_uas->compID;
        mavlink_msg_param_request_list_encode(m_uas->gcsID, m_uas->gcsCompID,
                                              &msg, &packet);

        m_uas->sendMavlinkMsg(&msg);
        setStatus(READING_BATCH);
    } else {
        _requireParameter(0);
        setStatus(READING_ALL);
    }

    m_nParam = 1;                   // FIXME: set 1 by default
    m_nParamRW = 1;
    m_idxCurrent = 0;
    m_idxReading = 0;
    m_tLastReading = tm_get_millis();
    m_bLoaded = 0;

    return 0;
}

int AP_ParamArray::_requireParameter(int index)
{
    mavlink_param_request_read_t packet;
    mavlink_message_t msg;

    packet.target_system    = m_uas->ID;
    packet.target_component = m_uas->compID;
    packet.param_index      = index;
    memset(packet.param_id, 0, 16);

    mavlink_msg_param_request_read_encode(m_uas->gcsID, m_uas->gcsCompID,
                                          &msg, &packet);

    m_uas->sendMavlinkMsg(&msg);

    m_tLastReading = tm_get_millis();

    return 0;
}

int AP_ParamArray::_requireParameter(char *id)
{
    mavlink_param_request_read_t packet;
    mavlink_message_t msg;


    packet.target_system    = m_uas->ID;
    packet.target_component = m_uas->compID;
    packet.param_index      = -1;
    strncpy(packet.param_id, id, 16);

    mavlink_msg_param_request_read_encode(m_uas->gcsID, m_uas->gcsCompID,
                                          &msg, &packet);

    m_uas->sendMavlinkMsg(&msg);

    m_tLastReading = tm_get_millis();

    return 0;
}

int AP_ParamArray::requireParameter(int index)
{
    if( m_stParamReading != IDLE ) return -1;

    _requireParameter(index);

    setStatus(READING);
    m_nParamRW = 1;
    m_idxReading = index;
    m_idxCurrent = 0;

    return 0;
}

int AP_ParamArray::requireParameter(char *id)
{
    if( m_stParamReading != IDLE ) return -1;

    _requireParameter(id);
    AP_ParamItem *p = get(id);

    setStatus(READING);
    m_nParamRW = 1;
    m_idxReading = p->index;
    m_idxCurrent = 0;

    return 0;
}

int AP_ParamArray::updateParameters(void)
{
    AP_ParamVector::iterator    it;
    AP_ParamItem                *p;

    // check current state
    if( m_stParamReading != IDLE ) return -1;

    // clear write list
    m_lstWrite.clear();

    // get all modified items
    for(it=m_lstAll.begin(); it!=m_lstAll.end(); it++) {
        p = *it;

        if( p->modified ) m_lstWrite.push_back(p);
    }

    // check modified item number
    if( m_lstWrite.size() < 1 ) return 0;

    setStatus(WRITING_ALL);
    m_nParamRW = m_lstWrite.size();
    m_idxCurrent = 0;

    // update first item
    p = m_lstWrite[0];
    m_idxWritting = p->index;
    _updateParameter(p);

    return 0;
}

int AP_ParamArray::updateParameters(StringArray &lstParams)
{
    StringArray::iterator    it;
    AP_ParamItem             *p;

    // check current state
    if( m_stParamReading != IDLE ) return -1;

    // clear write list
    m_lstWrite.clear();

    // get all modified items
    for(it=lstParams.begin(); it!=lstParams.end(); it++) {
        p = get(*it);
        if( p != NULL ) m_lstWrite.push_back(p);
    }

    // check modified item number
    if( m_lstWrite.size() < 1 ) return 0;

    setStatus(WRITING_ALL);
    m_nParamRW = m_lstWrite.size();
    m_idxCurrent = 0;

    // update first item
    p = m_lstWrite[0];
    m_idxWritting = p->index;
    _updateParameter(p);

    return 0;
}

int AP_ParamArray::_updateParameter(AP_ParamItem *pi)
{
    mavlink_param_set_t packet;
    mavlink_message_t msg;

    packet.target_system    = m_uas->ID;
    packet.target_component = m_uas->compID;
    packet.param_type       = (uint8_t) pi->type;
    packet.param_value      = pi->value;
    strncpy(packet.param_id, pi->id, 16);

    mavlink_msg_param_set_encode(m_uas->gcsID, m_uas->gcsCompID,
                                 &msg, &packet);
    m_uas->sendMavlinkMsg(&msg);

    m_idxWritting = pi->index;
    m_tLastReading = tm_get_millis();

    return 0;
}

int AP_ParamArray::updateParameter(int index)
{
    AP_ParamItem *pi;

    if( m_stParamReading != IDLE ) return -1;

    pi = get(index);
    _updateParameter(pi);

    m_nParamRW = 1;
    m_idxCurrent = 0;
    setStatus(WRITING);

    return 0;
}

int AP_ParamArray::updateParameter(char *id)
{
    AP_ParamItem *pi;

    if( m_stParamReading != IDLE ) return -1;

    pi = get(id);
    _updateParameter(pi);

    m_nParamRW = 1;
    m_idxCurrent = 0;
    setStatus(WRITING);

    return 0;
}



int AP_ParamArray::set(AP_ParamArray &pa)
{
    std::vector<AP_ParamItem*>::iterator it;
    AP_ParamItem *pi, *pi2;

    // copy all items
    for(it=pa.m_lstAll.begin(); it!=pa.m_lstAll.end(); it++) {
        pi = *it;

        pi2 = get(pi->id);
        if( pi2 == NULL ) {
            set(*pi);
        } else {
            pi2->type = pi->type;
            if( pi2->value != pi->value ) {
                pi2->modified = 1;
            }

            pi2->value = pi->value;
        }
    }

    // set flags
    m_mutex->lock();

    m_stParamReading = IDLE;
    m_tLastReading = tm_get_millis();
    m_bLoaded = 0;

    m_mutex->unlock();

    return 0;
}

int AP_ParamArray::set(AP_ParamItem &item)
{
    AP_ParamItem    *pi;

    pi = get(item.id);

    m_mutex->lock();

    m_tLastReading = tm_get_millis();
    m_idxCurrent = item.index;

    if( pi != NULL ) {
        pi->value = item.value;
        pi->modified = item.modified;
    } else {
        AP_ParamItem *ni;

        ni = new AP_ParamItem;
        *ni = item;

        m_lstAll.push_back(ni);
        m_mapIndex[item.index] = ni;
        m_mapID[item.id] = ni;
    }

    m_mutex->unlock();

    return 0;
}

AP_ParamItem* AP_ParamArray::get(int idx)
{
    AP_ParamItem *pi = NULL;
    std::map<int, AP_ParamItem*>::iterator it;

    m_mutex->lock();

    it = m_mapIndex.find(idx);
    if( it != m_mapIndex.end() ) {
        pi = it->second;
    }

    m_mutex->unlock();

    return pi;
}

AP_ParamItem* AP_ParamArray::get(std::string id)
{
    AP_ParamItem *pi = NULL;
    std::map<std::string, AP_ParamItem*>::iterator it;

    m_mutex->lock();

    it = m_mapID.find(id);
    if( it != m_mapID.end() ) {
        pi = it->second;
    }

    m_mutex->unlock();

    return pi;
}

AP_ParamVector* AP_ParamArray::get_allParam(void)
{
    return &m_lstAll;
}

AP_ParamIndexMap* AP_ParamArray::get_paramIndexMap(void)
{
    return &m_mapIndex;
}


int AP_ParamArray::set_paramN(int nParam)
{
    m_mutex->lock();
    m_nParam = nParam;
    m_mutex->unlock();

    return 0;
}

int AP_ParamArray::get_paramN(void)
{
    int n;

    m_mutex->lock();
    n = m_nParam;
    m_mutex->unlock();

    return n;
}

int AP_ParamArray::get_currIdx(void)
{
    int idx;

    m_mutex->lock();
    idx = m_idxCurrent;
    m_mutex->unlock();

    return idx;
}

int AP_ParamArray::collectUnreadedItems(void)
{
    // get unreaded items
    m_lstRead.clear();

    for(int i=0; i<m_nParamRW; i++) {
        if( m_mapIndex.find(i) == m_mapIndex.end() ) {
            m_lstRead.push_back(i);
        }
    }

    return 0;
}

int AP_ParamArray::getUnreadedItems(int n, std::vector<int> &lst)
{
    int i = 0, idx;

    // reserve n item
    lst.clear();
    lst.reserve(n);

    // insert to list
    while(m_lstRead.size() > 0 ) {
        if( i >= n ) break;

        idx = m_lstRead.front();
        lst.push_back(idx);
        m_lstRead.pop_front();

        i++;
    }

    return 0;
}

int AP_ParamArray::isLoaded(void)
{
    int l;

    m_mutex->lock();
    l = m_bLoaded;
    m_mutex->unlock();

    return l;
}

int AP_ParamArray::setLoaded(int bl)
{
    m_mutex->lock();
    m_bLoaded = bl;
    m_mutex->unlock();

    return 0;
}

int AP_ParamArray::setStatus(PARAM_RW_STATUS st)
{
    m_mutex->lock();
    m_stParamReading = st;
    m_mutex->unlock();

    return 0;
}

AP_ParamArray::PARAM_RW_STATUS AP_ParamArray::getStatus(int &nParamRW, int &idxCurr)
{
    AP_ParamArray::PARAM_RW_STATUS st;

    m_mutex->lock();

    nParamRW = m_nParamRW;
    idxCurr  = m_idxCurrent;
    st       = m_stParamReading;

    m_mutex->unlock();

    return st;
}

AP_ParamArray::PARAM_RW_STATUS AP_ParamArray::getStatus()
{
    AP_ParamArray::PARAM_RW_STATUS st;

    m_mutex->lock();
    st = m_stParamReading;
    m_mutex->unlock();

    return st;
}

int AP_ParamArray::get_tm_lastRead(uint64_t &t)
{
    m_mutex->lock();
    t = m_tLastReading;
    m_mutex->unlock();

    return 0;
}

int AP_ParamArray::save(std::string fname)
{
    FILE                        *fp = NULL;
    AP_ParamVector::iterator    it;
    AP_ParamItem                *p;

    // open file
    fp = fopen(fname.c_str(), "wt");
    if( fp == NULL ) {
        dbg_pe("Cannot open file: %s", fname.c_str());
        return -1;
    }

    m_mutex->lock();

    // output waypoints
    fprintf(fp, "#AP parameter number\n");
    fprintf(fp, "%d\n", m_lstAll.size());

    fprintf(fp, "#parameter list\n");
    fprintf(fp, "# index    ID/name    type   value\n");

    for(it=m_lstAll.begin(); it!=m_lstAll.end(); it++) {
        p = *it;

        fprintf(fp, "%4d %17s %2d %f\n",
                p->index,
                p->id,
                (int)(p->type),
                p->value);
    }

    m_mutex->unlock();

    // close file
    fclose(fp);

    return 0;
}

int AP_ParamArray::load(std::string fname)
{
    FILE                        *fp = NULL;

    char                        *buf;
    std::string                 _b;
    int                         max_line_size;
    int                         s, idx, n;
    int                         i1, i2;
    float                       val;

    // open file
    fp = fopen(fname.c_str(), "rt");
    if( fp == NULL ) {
        dbg_pe("Cannot open file: %s", fname.c_str());
        return -1;
    }

    // free old contents
    clear();

    // alloc memory buffer
    max_line_size = 1024;
    buf = new char[max_line_size];

    s = 0;
    idx = 0;

    while(!feof(fp)) {
        // read a line
        if( NULL == fgets(buf, max_line_size, fp) )
            break;

        // remove blank & CR
        _b = trim(buf);

        if( _b.size() < 1 )
            continue;

        // skip comment
        if( _b[0] == '#' || _b[0] == ':' )
            continue;

        if( s == 0 ) {
            // read wp number
            sscanf(_b.c_str(), "%d", &n);
            m_nParam = n;
            s = 1;
        } else {
            AP_ParamItem pi;

            // read wp item
            sscanf(_b.c_str(), "%d %s %d %f",
                   &i1, pi.id,
                   &i2, &val);

            pi.index = i1;
            pi.type = (MAV_PARAM_TYPE) i2;
            pi.value = val;
            pi.modified = 0;

            // insert to map
            set(pi);

            idx ++;
        }
    }

    // set flags
    m_mutex->lock();

    m_stParamReading = IDLE;
    m_tLastReading = tm_get_millis();
    m_bLoaded = 0;

    m_mutex->unlock();

    // free buffer
    delete [] buf;

    // close file
    fclose(fp);

    return 0;
}



int AP_ParamArray::parseMavlinkMsg(mavlink_message_t *msg)
{
    if ( msg->msgid == MAVLINK_MSG_ID_PARAM_VALUE ) {
        mavlink_param_value_t pv;
        mavlink_msg_param_value_decode(msg, &pv);

        AP_ParamItem pi, *p;

        for(int i=0; i<16; i++) pi.id[i] = pv.param_id[i];
        pi.id[16] = 0;
        pi.index = pv.param_index;
        pi.type  = (MAV_PARAM_TYPE) pv.param_type;
        pi.value = pv.param_value;

        dbg_pt("[MAV:%3d %4d/%4d] ID: %16s, TYPE: %2d, VALUE: %f\n",
               m_uas->ID,
               pi.index, pv.param_count,
               pi.id, pv.param_type, pv.param_value);

        m_tLastReading = tm_get_millis();

        // processing received value
        if( m_stParamReading == WRITING ) {
            p = get(m_idxWritting);

            // confirm wrote value
            if( fabs(p->value-pi.value) < 1e6 && p->type == pi.type ) {
                dbg_pt("[MAV:%3d %4d/%4d] ID: %16s, TYPE: %2d, VALUE: %f (confirmed)\n",
                       m_uas->ID,
                       pi.index, pv.param_count,
                       pi.id, pv.param_type, pv.param_value);

                p->modified = 0;
                m_stParamReading = IDLE;
            } else {
                _updateParameter(p);
            }
        } else if ( m_stParamReading == WRITING_ALL ) {
            p = get(m_idxWritting);

            // confirm wrote value
            if( fabs(p->value-pi.value) < 1e6 && p->type == pi.type ) {
                dbg_pt("[MAV:%3d %4d/%4d] ID: %16s, TYPE: %2d, VALUE: %f (confirmed)\n",
                       m_uas->ID,
                       pi.index, pv.param_count,
                       pi.id, pv.param_type, pv.param_value);

                p->modified = 0;

                // update next parameter or stop
                if( m_idxCurrent < m_nParamRW-1 ) {
                    m_idxCurrent++;
                    p = m_lstWrite[m_idxCurrent];
                    _updateParameter(p);
                } else {
                    m_stParamReading = IDLE;
                }
            } else {
                _updateParameter(p);
            }
        } else if ( m_stParamReading == READING ) {
            set_paramN(pv.param_count);
            set(pi);
            m_stParamReading = IDLE;
            m_bLoaded = 1;
        } else if ( m_stParamReading == READING_BATCH ) {
            set_paramN(pv.param_count);
            m_nParamRW = m_nParam;
            set(pi);

            if( m_lstAll.size() == m_nParam ) {
                m_stParamReading = IDLE;
                m_bLoaded = 1;
            } else {
                m_idxCurrent = m_lstAll.size() - 1;
            }
        } else if ( m_stParamReading == READING_ALL ) {
            set_paramN(pv.param_count);
            m_nParamRW = m_nParam;
            set(pi);

            m_idxCurrent = m_lstAll.size() - 1;

            // check if all readed
            if( m_lstRead.size() < 1 ) {
                collectUnreadedItems();

                if( m_lstRead.size() < 1 ) {
                    m_stParamReading = IDLE;
                    m_bLoaded = 1;

                    return 1;
                }
            }

            // send request
            std::vector<int> l;
            getUnreadedItems(5, l);
            for(int i=0; i<l.size(); i++) {
                m_idxReading = l[i];
                _requireParameter(m_idxReading);
            }
        }

        return 1;
    }

    return 0;
}

int AP_ParamArray::timerFunction(void *arg)
{
    uint64_t tnow = tm_get_millis(), dt;

    if( tnow <= m_tLastReading ) return 0;
    dt = tnow - m_tLastReading;

    switch( m_stParamReading ) {
    case IDLE:
        //stopTimer();

        break;

    case READING:
        if( dt > 500 ) {
            _requireParameter(m_idxReading);
        }

        break;

    case READING_ALL:
        if( dt > 500 ) {
            // send request
            std::vector<int> l;
            getUnreadedItems(5, l);

            for(int i=0; i<l.size(); i++) {
                m_idxReading = l[i];
                _requireParameter(m_idxReading);
            }
        }

        break;

    case READING_BATCH:
        if( dt > 1000 ) {
            // if no one item readed
            if( m_idxCurrent == 0 && m_nParamRW == 1 ) {
                m_stParamReading = IDLE;
                requireParameters();

                break;
            }

            // get unreaded items
            collectUnreadedItems();
            m_stParamReading = READING_ALL;

            // send request
            std::vector<int> l;
            getUnreadedItems(5, l);

            for(int i=0; i<l.size(); i++) {
                m_idxReading = l[i];
                _requireParameter(m_idxReading);
            }
        }

        break;

    case WRITING:
        if( dt > 1000 ) {
            AP_ParamItem *p = get(m_idxWritting);
            _updateParameter(p);
        }

        break;

    case WRITING_ALL:
        if( dt > 1000 ) {
            AP_ParamItem *p = get(m_idxWritting);
            _updateParameter(p);
        }

        break;
    }

    return 0;
}

int AP_ParamArray::startTimer(void)
{
    if( m_timer == 0 ) {
        // create timer
        if( 0 != osa_tm_create(&m_timer, 100, AP_ParamArray_timerFunc, this) ) {
            dbg_pe("Can not creat timer");
        }
    }

    return 0;
}

int AP_ParamArray::stopTimer(void)
{
    if( m_timer != 0 ) osa_tm_delete(m_timer);
    m_timer = 0;

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define AP_Mission_CMD_TYPE_STRUCT(n)   { n, #n }

static AP_Mission_CMD_Types g_arrMapType[] =
{
    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_WAYPOINT),
    //AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_SPLINE_WAYPOINT),

    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_LAND),
    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_TAKEOFF),
    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_MISSION_START),
    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_RETURN_TO_LAUNCH),

    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_LOITER_TURNS),
    AP_Mission_CMD_TYPE_STRUCT(MAV_CMD_NAV_LOITER_TIME),

    {MAV_CMD_ENUM_END, "MAV_CMD_ENUM_END"}
};


int AP_Mission_CMD_getAll(AP_Mission_CMD_Types **allTypes)
{
    *allTypes = g_arrMapType;
    return 0;
}

char* AP_Mission_CMD_cmd(MAV_CMD cmd)
{
    int i = 0;

    while(1) {
        if( g_arrMapType[i].cmd == cmd ) {
            return g_arrMapType[i].cmd_name;
        }

        if( g_arrMapType[i].cmd == MAV_CMD_ENUM_END )
            return NULL;

        i++;
    }

    return NULL;
}

MAV_CMD AP_Mission_CMD_cmd(char *cmd)
{
    int i = 0;

    while(1) {
        if( strcmp(cmd, g_arrMapType[i].cmd_name) == 0 ) {
            return g_arrMapType[i].cmd;
        }

        if( g_arrMapType[i].cmd == MAV_CMD_ENUM_END )
            return MAV_CMD_ENUM_END;

        i++;
    }

    return MAV_CMD_ENUM_END;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void AP_WayPoint::init(void)
{
    idx = -1;

    cmd = MAV_CMD_NAV_WAYPOINT;

    lat = -9999;
    lng = -9999;
    alt = 50;

    heading = 0;

    param1 = 0;
    param2 = 0;
    param3 = 0;
    param4 = 0;

    current = 0;

    frame = MAV_FRAME_GLOBAL;
    autocontinue = 1;

    writeConfirmed = 0;
}

void AP_WayPoint::release(void)
{
    writeConfirmed = 0;

    return;
}

void AP_WayPoint::set(double lat_, double lng_, double alt_)
{
    lat = lat_;
    lng = lng_;
    alt = alt_;
}

void AP_WayPoint::setPos(double lat_, double lng_)
{
    lat = lat_;
    lng = lng_;
}

void AP_WayPoint::setAlt(double alt_)
{
    alt = alt_;
}

void AP_WayPoint::setHeading(double heading_)
{
    heading = heading_;
}

void AP_WayPoint::get(double &lat_, double &lng_, double &alt_)
{
    lat_ = lat;
    lng_ = lng;
    alt_ = alt;
}

void AP_WayPoint::getPos(double &lat_, double &lng_)
{
    lat_ = lat;
    lng_ = lng;
}

void AP_WayPoint::getAlt(double &alt_)
{
    alt_ = alt;
}

void AP_WayPoint::getHeading(double &heading_)
{
    heading_ = heading;
}


int AP_WayPoint::to_mission_item(mavlink_mission_item_t *mi)
{
    mi->command         = cmd;           // mission type
    mi->seq             = idx;           // mission index

    mi->param1          = param1;
    mi->param2          = param2;
    mi->param3          = param3;
    mi->param4          = param4;

    mi->x               = lat;
    mi->y               = lng;
    mi->z               = alt;

    mi->frame           = frame;
    mi->autocontinue    = autocontinue;

    if( idx == 0 )
        mi->current     = 1;
    else
        mi->current     = 0;

    if( cmd == MAV_CMD_NAV_WAYPOINT ) {
        mi->param1 = 1.2;                       // stay time (in second)
    }

    return 0;
}

int AP_WayPoint::from_mission_item(mavlink_mission_item_t *mi)
{
    cmd             = (MAV_CMD) mi->command;    // mission type
    idx             = mi->seq;                  // mission index

    param1          = mi->param1;
    param2          = mi->param2;
    param3          = mi->param3;
    param4          = mi->param4;

    lat             = mi->x;
    lng             = mi->y;
    alt             = mi->z;

    frame           = (MAV_FRAME) mi->frame;
    autocontinue    = mi->autocontinue;

    current         = mi->current;

    return 0;
}


int AP_WayPoint::compare(AP_WayPoint *wp)
{
    double          diff, diff_thr = 0.00005;
    AP_WayPoint     *p1 = this, *p2 = wp;
    int             fail = 0;

    // FIXME: better way?
    diff = fabs(p1->cmd - p2->cmd); //printf("diff_cmd = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    diff = fabs(p1->lat - p2->lat); //printf("diff_lat = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    diff = fabs(p1->lng - p2->lng); //printf("diff_lng = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    diff = fabs(p1->alt - p2->alt); //printf("diff_alt = %f\n", diff);
    if( diff > diff_thr ) fail = 1;

    diff = fabs(p1->frame - p2->frame); //printf("diff_frame = %f\n", diff);
    if( diff > diff_thr ) fail = 1;

    /*
    diff = fabs(p1->param1 - p2->param1); printf("diff_param1 = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    diff = fabs(p1->param2 - p2->param2); printf("diff_param2 = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    diff = fabs(p1->param3 - p2->param3); printf("diff_param3 = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    diff = fabs(p1->param4 - p2->param4); printf("diff_param4 = %f\n", diff);
    if( diff > diff_thr ) fail = 1;
    */

    return fail;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

AP_WPArray::AP_WPArray()
{
    init();

    m_uas = NULL;
    m_timer = 0;
}

AP_WPArray::~AP_WPArray()
{
    release();

    stopTimer();

    m_uas = NULL;
    m_timer = 0;
}

void AP_WPArray::init(void)
{
    m_arrWP.clear();

    m_iRW = 0;
    m_nRW = 0;
    m_status = IDLE;

    m_currWaypoint = 0;
}

void AP_WPArray::release(void)
{
    if( m_arrWP.size() > 0 ) {
        AP_WayPointMap::iterator it;
        AP_WayPoint *p;

        for(it=m_arrWP.begin(); it!=m_arrWP.end(); it++) {
            p = it->second;
            delete p;
            p = NULL;
        }

        m_arrWP.clear();
    }

    m_iRW = 0;
    m_nRW = 0;

    m_currWaypoint = 0;
}

int AP_WPArray::set(AP_WPArray *wpa)
{
    // clear old wps
    clear();

    // insert all wps
    AP_WayPointMap::iterator it;
    AP_WayPoint *p;

    for(it=wpa->m_arrWP.begin(); it!=wpa->m_arrWP.end(); it++) {
        p = it->second;

        set(*p);
    }

    return 0;
}

int AP_WPArray::set(AP_WayPoint &wp)
{
    AP_WayPointMap::iterator it;

    it = m_arrWP.find(wp.idx);
    if( it != m_arrWP.end() ) {
        *(it->second) = wp;
    } else {
        AP_WayPoint *p = new AP_WayPoint;
        *p = wp;

        m_arrWP.insert(std::pair<int, AP_WayPoint*>(wp.idx, p));
    }

    return 0;
}

int AP_WPArray::get(AP_WayPoint &wp)
{
    AP_WayPointMap::iterator it;

    it = m_arrWP.find(wp.idx);
    if( it != m_arrWP.end() ) {
        wp = *(it->second);
    } else {
        return -1;
    }

    return 0;
}

AP_WayPoint* AP_WPArray::get(int idx)
{
    AP_WayPointMap::iterator it;
    AP_WayPoint *wp;

    it = m_arrWP.find(idx);
    if( it != m_arrWP.end() ) {
        wp = it->second;
    } else {
        wp = NULL;
    }

    return wp;
}

int AP_WPArray::getAll(AP_WayPointVector &wps)
{
    AP_WayPointMap::iterator it;

    wps.clear();

    for(it=m_arrWP.begin(); it!=m_arrWP.end(); it++) {
        wps.push_back(it->second);
    }

    return 0;
}

AP_WayPointMap* AP_WPArray::getAll(void)
{
    return &m_arrWP;
}


// FIXME: need reorder other items?
int AP_WPArray::remove(int idx)
{
    AP_WayPointMap::iterator it;

    it = m_arrWP.find(idx);
    if( it != m_arrWP.end() ) {
        m_arrWP.erase(it);
    } else {
        return -1;
    }

    return 0;
}

int AP_WPArray::size(void)
{
    return m_arrWP.size();
}

int AP_WPArray::clear(void)
{
    release();

    return 0;
}


int AP_WPArray::save(std::string fname)
{
    FILE                        *fp = NULL;
    AP_WayPointMap::iterator    it;
    AP_WayPoint                 *p;

    // open file
    fp = fopen(fname.c_str(), "wt");
    if( fp == NULL ) {
        dbg_pe("Cannot open file: %s", fname.c_str());
        return -1;
    }

    // output waypoints
    fprintf(fp, "#waypoint number\n");
    fprintf(fp, "%d\n", m_arrWP.size());

    fprintf(fp, "#waypoints list\n");
    fprintf(fp, "# idx              lat                   lng                  alt         heading\n");

    for(it=m_arrWP.begin(); it!=m_arrWP.end(); it++) {
        p = it->second;

        fprintf(fp, "%4d %24.16f %24.16f %12.3f %12.3f\n",
                p->idx,
                p->lat, p->lng, p->alt,
                p->heading);
    }

    // close file
    fclose(fp);

    return 0;
}

int AP_WPArray::load(std::string fname)
{
    FILE                        *fp = NULL;

    char                        *buf;
    std::string                 _b;
    int                         max_line_size;
    int                         s, idx, n;
    int                         i1;
    double                      lat, lng, alt, heading;

    // open file
    fp = fopen(fname.c_str(), "rt");
    if( fp == NULL ) {
        dbg_pe("Cannot open file: %s", fname.c_str());
        return -1;
    }

    // free old contents
    clear();

    // alloc memory buffer
    max_line_size = 1024;
    buf = new char[max_line_size];

    s = 0;
    idx = 0;

    while(!feof(fp)) {
        // read a line
        if( NULL == fgets(buf, max_line_size, fp) )
            break;

        // remove blank & CR
        _b = trim(buf);

        if( _b.size() < 1 )
            continue;

        // skip comment
        if( _b[0] == '#' || _b[0] == ':' )
            continue;

        if( s == 0 ) {
            // read wp number
            sscanf(_b.c_str(), "%d", &n);
            s = 1;
        } else {
            // read wp item
            sscanf(_b.c_str(), "%d %lf %lf %lf %lf",
                   &i1,
                   &lat, &lng, &alt, &heading);

            AP_WayPoint w;

            w.idx = i1;
            w.lat = lat;
            w.lng = lng;
            w.alt = alt;
            w.heading = heading;

            // insert to map
            set(w);

            idx ++;
        }
    }

    // free buffer
    delete [] buf;

    // close file
    fclose(fp);

    return 0;
}


int AP_WPArray::writeWaypoints(void)
{
    // check waypoint number
    if( size() <= 1 ) {
        dbg_pe("Waypoints number (%d) is not enough!", size());
        return -1;
    }

    // clear confirm flag
    AP_WayPointMap::iterator it;
    AP_WayPoint *p;

    for(it=m_arrWP.begin(); it!=m_arrWP.end(); it++) {
        p = it->second;
        p->writeConfirmed = 0;
    }

    // send waypoints number
    writeWaypointsNum();

    return 0;
}

int AP_WPArray::readWaypoints(void)
{
    m_status = READING_NUM;
    m_nReadTry = 4;

    readWaypointsNum();

    return 0;
}

int AP_WPArray::writeWaypointsNum(void)
{
    // set numbers & status
    m_status = WRITTING_NUM;
    m_iRW = 0;
    m_nRW = size();

    // send waypoint count
    mavlink_mission_count_t mc;
    mavlink_message_t msg;

    //mc.count = m_nRW - 1;       // exclude home wp
    mc.count = m_nRW;           // include home wp
    mc.target_system = m_uas->ID;
    mc.target_component = m_uas->compID;

    mavlink_msg_mission_count_encode(m_uas->gcsID, m_uas->gcsCompID, &msg, &mc);
    m_uas->sendMavlinkMsg(&msg);

    dbg_pi("writeWaypointsNum: %d", m_nRW);

    m_tLast = tm_get_millis();

    return 0;
}


int AP_WPArray::readWaypointsNum(void)
{
    // clear old contents
    // FIXME: better way?
    clear();

    return _readWaypointsNum();
}


int AP_WPArray::clearWaypoints(void)
{
    mavlink_message_t msg;
    mavlink_mission_clear_all_t mca;

    m_status = CLEAR_WP;

    mca.target_system = m_uas->ID;
    mca.target_component = m_uas->compID;

    mavlink_msg_mission_clear_all_encode(m_uas->gcsID, m_uas->gcsCompID, &msg, &mca);
    m_uas->sendMavlinkMsg(&msg);

    dbg_pi("clearWaypoints");

    m_tLast = tm_get_millis();

    return 0;
}

int AP_WPArray::setCurrentWaypoint(int idx)
{
    mavlink_message_t msg;
    mavlink_mission_set_current_t msc;

    // check index range
    if( idx >= size() || idx < 0 ) {
        dbg_pe("Given waypoint index (%d) out of rang [%d~%d].\n", idx, 1, size());
        return -1;
    }

    m_currWaypoint = idx;
    m_uas->currMission = idx;

    msc.seq = idx;
    msc.target_system = m_uas->ID;
    msc.target_component = m_uas->compID;

    mavlink_msg_mission_set_current_encode(m_uas->gcsID, m_uas->gcsCompID, &msg, &msc);
    m_uas->sendMavlinkMsg(&msg);

    dbg_pi("setCurrentWaypoint: [%d]", idx);

    m_tLast = tm_get_millis();

    return 0;
}

int AP_WPArray::_readWaypointsNum(void)
{
    m_iRW = 0;

    // require mission list
    mavlink_mission_request_list_t mrl;
    mavlink_message_t msg;

    mrl.target_system = m_uas->ID;
    mrl.target_component = m_uas->compID;

    mavlink_msg_mission_request_list_encode(m_uas->gcsID, m_uas->gcsCompID, &msg, &mrl);
    m_uas->sendMavlinkMsg(&msg);

    dbg_pi("readWaypointsNum");

    m_tLast = tm_get_millis();

    return 0;
}

int AP_WPArray::_readWaypoint(int idx)
{
    // require given wp
    mavlink_mission_request_t mr;
    mavlink_message_t msgSend;

    mr.seq = idx;
    mr.target_system = m_uas->ID;
    mr.target_component = m_uas->compID;

    mavlink_msg_mission_request_encode(m_uas->gcsID, m_uas->gcsCompID,
                                       &msgSend, &mr);
    m_uas->sendMavlinkMsg(&msgSend);

    dbg_pi("readWaypoint");

    m_tLast = tm_get_millis();

    return 0;
}

int AP_WPArray::_writeWaypoint(int idx)
{
    // get waypoint
    AP_WayPoint *p = get(idx);
    if( p == NULL ) {
        dbg_pe("Given waypoint[%d] not exist!\n", idx);
        return -1;
    }

    // send waypoint item to MAV
    mavlink_mission_item_t mmi;
    mavlink_message_t msgSend;

    p->to_mission_item(&mmi);
    mmi.seq = idx;
    mmi.target_system = m_uas->ID;
    mmi.target_component = m_uas->compID;

    dbg_pi("send mission[%3d/%3d] cmd = %d, x,y,z = %f %f %f\n",
           idx, size(), mmi.command,
           mmi.x, mmi.y, mmi.z);
    dbg_pi("                  p1, p2, p3, p4 = %f %f %f %f\n",
           mmi.param1, mmi.param2, mmi.param3, mmi.param4);
    dbg_pi("                  frame = %d, current = %d, autocontinue = %d\n",
           mmi.frame, mmi.current, mmi.autocontinue);

    mavlink_msg_mission_item_encode(m_uas->gcsID, m_uas->gcsCompID, &msgSend, &mmi);
    m_uas->sendMavlinkMsg(&msgSend);

    m_tLast = tm_get_millis();

    return 0;
}

int AP_WPArray::timerFunction(void *arg)
{
    if( m_status == IDLE ) return 0;

    // get current time & dt
    uint64_t dt, dt_thr = 500,
             tnow = tm_get_millis();
    if( tnow < m_tLast ) return 0;
    dt = tnow - m_tLast;

    // if do not receive any mission count message then assume the mission
    //  number is zero
    if( m_status == READING_NUM && dt > dt_thr ) {
        if( m_nReadTry > 0 ) {
            m_nReadTry --;          
            _readWaypointsNum();
        } else {
            dbg_pw("MAV really dont have any mission!\n");
            m_tLast = tm_get_millis();
            m_status = IDLE;
        }

        return 0;
    }

    // if stop receiving waypoint then resend request msg again
    if( m_status == READING_ITEM && dt > dt_thr ) {
        if( m_iRW < m_nRW - 1 ) {
            // require current wp
            _readWaypoint(m_iRW);
        } else {
            // finished reading
            mavlink_mission_ack_t ma;
            mavlink_message_t msgSend;

            ma.type = MAV_MISSION_ACCEPTED;
            mavlink_msg_mission_ack_encode(m_uas->gcsID, m_uas->gcsCompID, &msgSend, &ma);

            m_uas->sendMavlinkMsg(&msgSend);

            m_status = IDLE;
            m_tLast = tm_get_millis();
        }

        return 0;
    }


    // if write waypoint, but receiving any request then send mission count again
    if( m_status == WRITTING_NUM && dt > dt_thr ) {
        writeWaypointsNum();

        return 0;
    }

    if( m_status == WRITTING_ITEM && dt > dt_thr ) {
        if( m_iRW < m_nRW - 1 ) {
            // send waypoint item to MAV
            _writeWaypoint(++m_iRW);
        } else {
            // confirm uploaded waypoint number
            m_status = WRITTING_CONFIRM_NUM;
            _readWaypointsNum();
        }

        return 0;
    }

    // if confirm wrote missions
    if( m_status == WRITTING_CONFIRM_NUM && dt > dt_thr ) {
        _readWaypointsNum();

        return 0;
    }

    // if confirm uploaded mission, then resend the request
    if( m_status == WRITTING_CONFIRM_ITEM && dt > dt_thr ) {
        if( m_iRW < m_nRW - 1 ) {
            // require next wp
            _readWaypoint(m_iRW);
        } else {
            // check received items are correct
            AP_WayPointMap::iterator it;
            AP_WayPoint *p;
            int fail = 0;

            for(it=m_arrWP.begin(); it!=m_arrWP.end(); it++) {
                p = it->second;

                if( p->idx > 0 && !p->writeConfirmed ) fail = 1;
            }

            if( fail ) {
                // upload missions again
                writeWaypoints();
            } else {
                // finished mession confirm
                m_status = IDLE;
                m_tLast = tm_get_millis();

                setCurrentWaypoint(1);
            }
        }

        return 0;
    }

    if( m_status == CLEAR_WP && dt > dt_thr ) {
        clearWaypoints();

        return 0;
    }

    return 0;
}

int AP_WPArray::parseMavlinkMsg(mavlink_message_t *msg)
{
    int ret = 0;

    switch(msg->msgid) {
    case MAVLINK_MSG_ID_MISSION_ITEM:               // 39
    {
        ret = 1;

        mavlink_mission_item_t mi;
        mavlink_msg_mission_item_decode(msg, &mi);

        dbg_pi("recv mission[%3d] cmd = %d, x,y,z = %f %f %f\n",
               mi.seq, mi.command, mi.x, mi.y, mi.z);
        dbg_pi("                  p1, p2, p3, p4 = %f %f %f %f\n",
               mi.param1, mi.param2, mi.param3, mi.param4);
        dbg_pi("                  frame = %d, current = %d, autocontinue = %d\n",
               mi.frame, mi.current, mi.autocontinue);

        AP_WayPoint wp;
        wp.from_mission_item(&mi);

        m_tLast = tm_get_millis();

        if( m_iRW == mi.seq ) {
            if( m_status == READING_ITEM )
                set(wp);
            else if( m_status == WRITTING_CONFIRM_ITEM ) {
                AP_WayPoint *p = get(wp.idx);

                // check uploaded waypoint is correct or not
                if( wp.idx>0 && wp.compare(p) ) {
                    // re-uploading waypoints
                    writeWaypoints();
                    break;
                } else
                    p->writeConfirmed = 1;
            }

            if( mi.seq >= m_nRW - 1 ) {
                // send ACK to MAV
                mavlink_mission_ack_t ma;
                mavlink_message_t msgSend;

                ma.type = MAV_MISSION_ACCEPTED;
                mavlink_msg_mission_ack_encode(m_uas->gcsID, m_uas->gcsCompID, &msgSend, &ma);

                m_uas->sendMavlinkMsg(&msgSend);

                if( m_status == READING_ITEM ) {
                    dbg_pi("recv mission finished!\n");

                    m_status = IDLE;
                } else if ( m_status == WRITTING_CONFIRM_ITEM ) {
                    dbg_pi("confirm mission finished!\n");

                    m_status = IDLE;

                    setCurrentWaypoint(1);
                }


            } else {
                // require next wp
                m_iRW ++;
                _readWaypoint(m_iRW);
            }
        } else {
            // require current wp again
            _readWaypoint(m_iRW);
        }

        break;
    }

    case MAVLINK_MSG_ID_MISSION_REQUEST:            // 40
    {
        ret = 1;

        mavlink_mission_request_t mr;
        mavlink_msg_mission_request_decode(msg, &mr);

        m_tLast = tm_get_millis();
        m_status = WRITTING_ITEM;

        // check & get waypoint item
        dbg_pi("req mission [%3d/%3d]\n", mr.seq, size());

        int mi = mr.seq;
        //if( mi != m_iRW ) mi = m_iRW;

        AP_WayPoint *wp = get(mi);
        if( wp == NULL ) {
            dbg_pe("given wp[%3d] not exist!\n", mi);
            m_status = IDLE;
            break;
        }

        // send waypoint item to MAV
        _writeWaypoint(mi);

        m_iRW = mi;

        break;
    }

    case MAVLINK_MSG_ID_MISSION_CURRENT:            // 42
    {
        ret = 1;

        mavlink_mission_current_t mc;
        mavlink_msg_mission_current_decode(msg, &mc);

        m_currWaypoint = mc.seq;
        m_uas->currMission = mc.seq;

        break;
    }

    case MAVLINK_MSG_ID_MISSION_COUNT:              // 44
    {
        ret = 1;

        mavlink_mission_count_t mc;
        mavlink_msg_mission_count_decode(msg, &mc);

        m_tLast = tm_get_millis();

        dbg_pi("recv missions number: %d, m_status = %d\n", mc.count, m_status);

        // set wp count & check count
        if( m_status == READING_NUM ) {
            m_nRW = mc.count;

            if( mc.count <= 0 ) {
                // set reading status to 3 (finished)
                m_status = IDLE;
                break;
            }

            m_status = READING_ITEM;
        } else if( m_status == WRITTING_CONFIRM_NUM ) {
            // check uploaded item number is correct
            if( mc.count != size() ) {
                dbg_pe("upload waypoints number incorrect %d\n", mc.count);
                writeWaypoints();
                break;
            }

            m_status = WRITTING_CONFIRM_ITEM;
        }

        // set rw index to first wp
        m_iRW = 0;
        m_nReadTry = 0;
        _readWaypoint(m_iRW);

        break;
    }

    case MAVLINK_MSG_ID_MISSION_ACK:                // 47
    {
        ret = 1;

        mavlink_mission_ack_t ma;
        mavlink_msg_mission_ack_decode(msg, &ma);

        dbg_pi("mission_ack: %d\n", ma.type);

        m_tLast = tm_get_millis();

        // if current state is waypoint writting
        if( m_status == WRITTING_ITEM ) {
            if( ma.type == 0 ) {
                m_status = WRITTING_CONFIRM_NUM;

                _readWaypointsNum();
            } else {
                // if failed send wp number again
                writeWaypointsNum();
            }
        }

        // current state is waypoint clear
        if( m_status == CLEAR_WP ) {
            if( ma.type == 0 ) {
                m_status = IDLE;
            } else {
                clearWaypoints();
            }
        }

        break;
    }

    } // end of switch(msg->msgid)

    return ret;
}

int AP_WPArray::startTimer(void)
{
    if( m_timer == 0 ) {
        // create timer
        if( 0 != osa_tm_create(&m_timer, 200, AP_ParamArray_timerFunc, this) ) {
            dbg_pe("Can not creat timer");
        }
    }

    return 0;
}

int AP_WPArray::stopTimer(void)
{
    if( m_timer != 0 ) osa_tm_delete(m_timer);
    m_timer = 0;

    return 0;
}
