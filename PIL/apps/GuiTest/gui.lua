--New Window
MainWidget=CreateWidgetByType("QWidget","MainWidget","MainWindow");
MainWindow=GetWidgetByName("MainWindow");
MainWindow:setMaximumHeight(50);
MainWidget:setMinimumHeight(700);
MainWidget:setMinimumWidth(1000);
--setMinimumSize("MainWidget",1000,700);

size=Point2d:new();
GetWidgetSize("MainWidget",size);
print("size of MainWidget is [", MainWindow:width(), ",", MainWindow:height() ,"]");

--right pannel
CreateWidgetByType("QADI","mainADI","MainWidget");
CreateWidgetByType("QCompass","mainCompass","MainWidget");
CreateWidgetByType("QWidget","wLeftPanel","MainWidget");
CreateWidgetByType("QVBoxLayout","vl","wLeftPanel");
setLayout("wLeftPanel","vl");
addWidget("vl","mainADI");
addWidget("vl","mainCompass");

--left pannel
CreateWidgetByType("QTabWidget","mTab","MainWidget");
CreateWidgetByType("Win3D","win3d","MainWidget");
CreateWidgetByType("SvarWidget","SvarWidget","MainWidget");
addTab("mTab","win3d");
addTab("mTab","SvarWidget");

CreateWidgetByType("QHBoxLayout","hl","MainWidget");
setLayout("MainWidget","hl");

addWidget("hl","mTab");
addWidget("hl","wLeftPanel");
