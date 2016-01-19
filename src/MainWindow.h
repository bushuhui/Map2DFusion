#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <gui/gl/Win3D.h>
#include <queue>
#include <opmapcontrol/opmapcontrol.h>

class MainWindow:public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = 0);
    virtual ~MainWindow(){}

    virtual int setupLayout(void);

    pi::gl::Win3D* getWin3D(){return win3d;}

    bool    setMapType(const std::string& MapType);

    void    call(const std::string& cmd);

signals:
    void call_signal();

protected slots:
    void call_slot();

protected:
    void keyPressEvent(QKeyEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent *event);
    void timerEvent(QTimerEvent *event);

    pi::gl::Win3D* win3d;
    mapcontrol::OPMapWidget* mapwidget;

    std::queue<std::string>       cmds;
};

#endif // MAINWINDOW_H
