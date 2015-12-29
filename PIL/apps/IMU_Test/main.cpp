// Copyright 2008 Isis Innovation Limited
// This is the main extry point for PTAM
#include <stdlib.h>
#include <iostream>
#include <qapplication.h>

#include <base/Svar/Svar_Inc.h>

#include "System.h"
#include <base/utils.h>
using namespace std;

int main(int argc,char **argv)
{

    dbg_stacktrace_setup();
    svar.ParseMain(argc,argv);

    QApplication application(argc, argv);
    System system;
    system.start();
    return application.exec();
}
