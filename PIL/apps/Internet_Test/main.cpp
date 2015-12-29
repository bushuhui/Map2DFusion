#include <iostream>
#include <unistd.h>


#include <base/Svar/Svar_Inc.h>
#include <base/debug/debug_config.h>
#include <network/MessagePassing.h>
#include <base/Svar/Scommand.h>
using namespace std;
using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void test_internetCommand(void *ptr,string command,string parament)
{
    cout<<"Entered the test internet_Command.\n";
    cout<<"parament="<<parament<<endl;
    RMessagePassing* mp=(RMessagePassing*)ptr;
    mp->sendString(parament,"I have received your command "+command);
}

void test_Scommand()
{
    RMessagePassing mp;
    string my_name=svar.GetString("Node.name","Master");
    Scommand::instance().RegisterCommand("test_internetCommand",test_internetCommand,&mp);
    if(0!=mp.begin(my_name))
    {
        cout<<"Initial failed.\n";
        return;
    }
    if("Master"==my_name)
    {
        while(mp.getAlive())
        {
            string str;
            RMP_Node* node=mp.recvString(str);
            SvarWithType<RMP_Node*>::instance()["OtherNode"]=node;
            if(str.size())
            {
                cout<<"Received:"<<str<<endl;
//                mp.sendString(node->nodeName,"I have received the message");
            }
            sleep(1);
        }
    }
    else
    {
        while(mp.getAlive())
        {

            string str;
            RMP_Node* node=mp.recvString(str);
            if(str.size())
            {
                cout<<"Received:"<<str<<endl;
                mp.sendString(node->nodeName,"I have received the message.");
            }
            static string command="test_internetCommand "+my_name;
            mp.sendString("Master",command,true);
            sleep(1);
        }
    }

}




int test_server()
{
    int         imsg=0;
    string      msg;
    RSocket     server;
    int         ret;

    int         port = 30000;

    // parse input argument
    port=svar.GetInt("ServerPort", port);


    // start socket server
    std::cout << "running....\n";
    server.startServer(port);

    while(1)
    {
        RSocket new_socket;

        if( 0 != server.accept(new_socket) )
        {
            dbg_pe("server.accept failed!");
            continue;
        }

        RSocketAddress ca;
        new_socket.getClientAddress(ca);

        printf("\n");
        dbg_pt("accept a new connection! client: %s (%d)\n",
               ca.addr_str, ca.port);

        while(1)
        {
            ret = new_socket.recv(msg);
            usleep(100);

            if( ret < 0 ) break;
            else if( ret == 0 ) continue;

            new_socket.send(msg);

            printf("[%3d] %s\n", imsg++, msg.c_str());

            if( msg == "quit" ) {
                goto SERVER_QUIT;
            }
        }
        usleep(10000);
    }

SERVER_QUIT:
    server.close();

    return 0;
}


int test_client()
{
    string      msg, msg_s;
    ri64        pid;
    RSocket     client;
    int         ret1, ret2;

    string      addr;
    int         port;

    // parse input arguments
    addr = "10.138.31.178";
    port = 30000;

    addr=svar.GetString("client.addr", addr);
    port=svar.GetInt("client.port", port);


    // generate default message
    osa_get_pid(&pid);
    msg = fmt::sprintf("Test message! from pid=%d", pid);
    msg=svar.GetString("msg", msg);


    // begin socket
    if( 0 != client.startClient(addr, port) ) {
        dbg_pe("client.start_client failed!");
        return -1;
    }

    dbg_pt("client started!\n");

    for(int i=0; i<1000;i++) {
        msg_s = msg;

        ret1 = client.send(msg_s);
        ret2 = client.recv(msg_s);
        printf("receive message from sever: %s (%d), ret = %d, %d\n",
               msg.c_str(), msg.size(),
               ret1, ret2);
        usleep(500000);
    }

    client.close();

    return 0;
}

int test_messagepassing()
{
    RMessagePassing     mp;
    RMessage            *msg, msg_send;
    int                 msgType = 1, msgID = 0;
    StringArray         nl;


    string              my_name;

    int                 i;
    ru64                t0, t1, dt;

    int                 ret;

    my_name=svar.GetString("Node.name","Master");
    if( mp.begin(my_name) != 0 ) {
        return -1;
    }

    t0 = tm_get_millis();

    while(1) {
        // receive a message & show it
        msg = mp.recvMsg();
        if( msg != NULL ) {
            string      n1, n2;
            DateTime   t0, t1;
            double      dt;

            t1.setCurrentDateTime();
            msg->data.rewind();
            msg->data.read(n1);
            msg->data.read(n1);
            t0.fromStream(msg->data);
            dt = t1.diffTime(t0);

            printf("\nrecved message, msg.size = %d, dt = %f\n   ",
                   msg->data.size(), dt);
            msg->print();

            delete msg;
        }

        // send message to other nodes
        t1 = tm_get_millis();
        dt = t1 - t0;
        if( dt > 60 ) {
            mp.getNodeMap()->getNodeList(nl);

            printf("\n");
            mp.getNodeMap()->print();
            printf("\n");


            for(i=0; i<nl.size(); i++) {
                if( nl[i] != my_name ) {
                    msg_send.msgType = msgType;
                    msg_send.msgID   = msgID++;

                    DateTime tm;
                    tm.setCurrentDateTime();

                    msg_send.data.clear();
                    msg_send.data.write(my_name);
                    msg_send.data.write(nl[i]);
                    tm.toStream(msg_send.data);

                    printf("send message: %s -> %s, msg_size = %d\n",
                           my_name.c_str(), nl[i].c_str(), msg_send.data.size());

                    mp.sendMsg(nl[i], &msg_send);
                }
            }

            t0 = t1;
        }

        usleep(1000);
    }
}

int main(int argc, char *argv[])
{

    pi::dbg_stacktrace_setup();
    svar.ParseMain(argc,argv);
    string act=svar.GetString("Act","test_server");
    if("test_server"==act) test_server();
    if("test_client"==act) test_client();
    if("test_messagepassing"==act) test_messagepassing();
    if("test_Scommand"==act)test_Scommand();
    return 0;
}
