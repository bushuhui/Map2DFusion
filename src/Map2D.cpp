#include "Map2D.h"
#include "Map2DCPU.h"
#include "Map2DRender.h"
#include "Map2DGPU.h"


SPtr<Map2D> Map2D::create(int type,bool thread)
{
    if(type==NoType) return SPtr<Map2D>();
    else if(type==TypeCPU)    return SPtr<Map2D>(new Map2DCPU(thread));
    else if(type==TypeRender)    return SPtr<Map2D>(new Map2DRender(thread));
    else if(type==TypeGPU)    return SPtr<Map2D>(new Map2DGPU(thread));
}
