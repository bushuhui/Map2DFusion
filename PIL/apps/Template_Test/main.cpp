#include <iostream>
#include <base/types/Static_Templates.h>

int main()
{
#if pi::IS_INT<int>::Result
    int a;
#else
    int b;
#endif
}
