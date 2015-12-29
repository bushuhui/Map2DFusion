#include <iostream>

#include "base/Svar/Svar_Inc.h"
#include "base/time/Global_Timer.h"

#include <TooN/TooN.h>
//using namespace TooN;
using namespace std;
using namespace pi;

int times=100000;




template<class Precision,int Row=3,int Col=Row>
bool TestMatrix()
{
    bool passed=true;
    Precision p[Row][Col];
    cout<<"P=";
    for(int i=0;i<Row;i++)
    {
        for(int j=0;j<Col;j++)
        {
            p[i][j]=((Precision)rand())/RAND_MAX;
            cout<<p[i][j]<<",";
        }
        cout<<endl;
    }

    TooN::Matrix<Row,Col,Precision>* mat=(TooN::Matrix<Row,Col,Precision>*)p ;

    cout<<"\nMatrix="<<*mat;
    return passed;
}

template<class Precision,int Demession=3>
bool TestVector()
{
    bool passed=true;
    Precision p[Demession];
    cout<<"P=";
    for(int i=0;i<Demession;i++)
    {
        p[i]=((Precision)rand())/RAND_MAX;
        cout<<p[i]<<",";
    }

    TooN::Vector<Demession,Precision>* vec=(TooN::Vector<Demession,Precision>*)p;
    cout<<"\nVector="<<*vec<<endl;
    return passed;
}

void TestTooNVector()
{
    //Square
    double result;

    double a=0.44456464;
    double b=5.45256467;
    double c=1.15646462;

    TooN::Vector<2> v2=TooN::makeVector(a,b);
    timer.enter("SquareV2::TooN");
    for(int i=0;i<times;i++)
        result=v2*v2;
    timer.leave("SquareV2::TooN");

    timer.enter("SquareV2::Direct");
    for(int i=0;i<times;i++)
        result=a*a+b*b;
    timer.leave("SquareV2::Direct");

    TooN::Vector<3> v3=TooN::makeVector(a,b,c);
    timer.enter("SquareV3::TooN");
    for(int i=0;i<times;i++)
        result=v3*v3;
    timer.leave("SquareV3::TooN");

    timer.enter("SquareV3::Direct");
    for(int i=0;i<times;i++)
        result=a*a+b*b+c*c;
    timer.leave("SquareV3::Direct");
}


void TestMultiDiv()
{
//Test Multiple and division time usage
    double result=133;
    double a=2.45641215;
    double b=0.442565465646458;
    timer.enter("Multiple");
    for(int i=0;i<times;i++)
        result=a*b;
    timer.leave("Multiple");

    timer.enter("Division");
    for(int i=0;i<times;i++)
        result=a/b;
    timer.leave("Division");
}


int main(int argc,char** argv)
{
    svar.ParseMain(argc,argv);
    TestVector<double>();
    TestMatrix<double>();
    TestVector<float>();
    TestMatrix<float>();
    TestTooNVector();
    TestMultiDiv();

    return 0;
}

