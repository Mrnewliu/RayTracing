#pragma once
#include <assert.h>
#include <ostream>
using namespace std;

namespace SimpleOBJ
{
    class Vec3f
    {
    public:

        //Constructors
        Vec3f();
        Vec3f(double x,double y, double z);
        Vec3f(const Vec3f& v);
        //Deconstructor
        virtual ~Vec3f();
    public:
        //Operators

        //Operator []
        inline double& operator [](int index)
        {
            assert(index>=0&&index<3);
            return _p[index];
        }
        inline const double& operator [](int index) const
        {
            assert(index>=0&&index<3);
            return _p[index];
        }
        
        //Operator =
        Vec3f& operator = (const Vec3f& v);

        //Operators +=,-=, *=, /=
        void operator +=(const Vec3f& v);
        void operator +=(double f);
        void operator -=(const Vec3f& v);
        void operator -=(double f);
        void operator *=(const Vec3f& v);
        void operator *=(double f);
        void operator /=(const Vec3f& v);
        void operator /=(double f);

        //Operators +,-.*,/
        Vec3f operator +(const Vec3f&v) const;
        Vec3f operator +(double f) const;
        Vec3f operator -(const Vec3f&v) const;
        Vec3f operator -(double f) const;
        Vec3f operator *(const Vec3f&v) const;
        Vec3f operator *(double f) const;
        Vec3f operator /(const Vec3f&v) const;
        Vec3f operator /(double f) const;

        Vec3f operator -() const;
        
        friend Vec3f operator *(double f, const Vec3f& v);
        friend ostream& operator <<(ostream& os, Vec3f& v);
        friend ostream& operator <<(ostream& os, const Vec3f& v);

    public:
        void Normalize();
        double L2Norm_Sqr();
     
    public:
        union
        {
            struct
            { double _p[3]; };
            struct
            { double x,y,z; };
            struct
            { double r,g,b; };
        };
        enum {_len = 3};   
        
    };
    
    
}

 
