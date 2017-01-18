#ifndef _SHORTVECS_H_
#define _SHORTVECS_H_

#include <cmath>
// floating point constants
//#define RT_POS_INFINITY 0x7F800000
//#define RT_NEG_INFINITY 0xFF800000
//#define PI   float(3.1415926535897932384626433832795)
//#define EPSILON float(0.0001)		// used in triangle intersect function for avoiding self-intersection

// float2 and float3 definition and constructor functions
////////////////////////////////////////////////////////////////////////////////
namespace vltools
{
	struct int2
	{
	  int x, y;
	  inline int & operator[](int n) {return *(&x + n);}
	  inline const int & operator[](int n) const {return *(&x + n);}
	  inline bool operator==(const int2 & rhs) const {
	    return x == rhs.x && y == rhs.y; }
	  inline bool operator!=(const int2 & rhs) const {
	    return !(*this == rhs); }
	};

	struct int3
	{
	  int x, y, z;
	  inline int & operator[](int n) {return *(&x + n);}
	  inline const int & operator[](int n) const {return *(&x + n);}
	  inline bool operator==(const int3 & rhs) const {
	    return x == rhs.x && y == rhs.y && z == rhs.z; }
	  inline bool operator!=(const int3 & rhs) const {
	    return !(*this == rhs); }
	};

	struct float2
	{
	  float x, y;
	  inline float & operator[](int n) {return *(&x + n);}
	  inline const float & operator[](int n) const {return *(&x + n);}
	  inline bool operator==(const float2 & rhs) const {
	    return x == rhs.x && y == rhs.y; }
	  inline bool operator!=(const float2 & rhs) const {
	    return !(*this == rhs); }
	};

	struct float3
	{
	  float x, y, z;
	  inline float & operator[](int n) {return *(&x + n);}
	  inline const float & operator[](int n) const {return *(&x + n);}
	  inline float lengthSquared() const
	  {
		 return (x * x) + (y * y) + (z * z);
	  }
	  inline bool operator==(const float3 & rhs) const {
	    return x == rhs.x && y == rhs.y && z == rhs.z; }
	  inline bool operator!=(const float3 & rhs) const {
	    return !(*this == rhs); }
	};

	struct float4
	{
	  float x, y, z, w;
	  inline float & operator[](int n) {return *(&x + n);}
	  inline const float & operator[](int n) const {return *(&x + n);}
	  inline bool operator==(const float4 & rhs) const {
	    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w; }
	  inline bool operator!=(const float4 & rhs) const {
	    return !(*this == rhs); }
	};

	inline float2 make_float2(float x, float y)
	{
	  float2 t; t.x = x; t.y = y; return t;
	}

	inline float3 make_float3(float x, float y, float z)
	{
	  float3 t; t.x = x; t.y = y; t.z = z; return t;
	}

	inline float4 make_float4(float x, float y, float z, float w)
	{
	  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
	}

	inline int2 make_int2(int x, int y)
	{
	  int2 t; t.x = x; t.y = y; return t;
	}

	inline int3 make_int3(int x, int y, int z)
	{
	  int3 t; t.x = x; t.y = y; t.z = z; return t;
	}

	// min, max, rsqrtf functions
	////////////////////////////////////////////////////////////////////////////////

	inline float fminf(float a, float b)
	{
	  return a < b ? a : b;
	}

	inline float fmaxf(float a, float b)
	{
	  return a > b ? a : b;
	}

	//inline int max(int a, int b)
	//{
	//  return a > b ? a : b;
	//}

	//inline int min(int a, int b)
	//{
	//  return a < b ? a : b;
	//}

	inline float rsqrtf(float x)
	{
		return 1.0f / sqrtf(x);
	}

	// fast floor function
	////////////////////////////////////////////////////////////////////////////////

/*
	inline int floor(float x) 
	{
		// Sree's Real2Int (http://stereopsis.com/FPU.html)
		double _double2fixmagic = 68719476736.0*1.5;
		double val = (double)x + _double2fixmagic;
		return ((int*)&val)[0] >> 16;
	}
*/

	// float functions
	////////////////////////////////////////////////////////////////////////////////

	// lerp
	inline float lerp(float a, float b, float t)
	{
		return a + t*(b-a);
	}

	// clamp
	inline float clamp(float f, float a, float b)
	{
		return fmaxf(a, fminf(f, b));
	}

	// float2 functions
	////////////////////////////////////////////////////////////////////////////////

	// additional constructors
	inline float2 make_float2(float s)
	{
		return make_float2(s, s);
	}
	inline float2 make_float2(int2 a)
	{
		return make_float2(float(a.x), float(a.y));
	}

	// negate
	inline float2 operator-(float2 &a)
	{
		return make_float2(-a.x, -a.y);
	}

	// addition
	inline float2 operator+(float2 a, float2 b)
	{
		return make_float2(a.x + b.x, a.y + b.y);
	}
	inline void operator+=(float2 &a, float2 b)
	{
		a.x += b.x; a.y += b.y;
	}

	// subtract
	inline float2 operator-(float2 a, float2 b)
	{
		return make_float2(a.x - b.x, a.y - b.y);
	}
	inline void operator-=(float2 &a, float2 b)
	{
		a.x -= b.x; a.y -= b.y;
	}

	// multiply
	inline float2 operator*(float2 a, float2 b)
	{
		return make_float2(a.x * b.x, a.y * b.y);
	}
	inline float2 operator*(float2 a, float s)
	{
		return make_float2(a.x * s, a.y * s);
	}
	inline float2 operator*(float s, float2 a)
	{
		return make_float2(a.x * s, a.y * s);
	}
	inline void operator*=(float2 &a, float s)
	{
		a.x *= s; a.y *= s;
	}

	// divide
	inline float2 operator/(float2 a, float2 b)
	{
		return make_float2(a.x / b.x, a.y / b.y);
	}
	inline float2 operator/(float2 a, float s)
	{
		float inv = 1.0f / s;
		return a * inv;
	}
	inline float2 operator/(float s, float2 a)
	{
		float inv = 1.0f / s;
		return a * inv;
	}
	inline void operator/=(float2 &a, float s)
	{
		float inv = 1.0f / s;
		a *= inv;
	}
	//inline float & operator[](float2 a, int n)
	//{
	//    float * ptr = (float*)&a;
	//    ptr += n;
	//    return *ptr;
	//}


	// lerp
	inline float2 lerp(float2 a, float2 b, float t)
	{
		return a + t*(b-a);
	}

	// clamp
	inline float2 clamp(float2 v, float a, float b)
	{
		return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
	}

	inline float2 clamp(float2 v, float2 a, float2 b)
	{
		return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
	}

	// dot product
	inline float dot(float2 a, float2 b)
	{ 
		return a.x * b.x + a.y * b.y;
	}

	// length
	inline float length(float2 v)
	{
		return sqrtf(dot(v, v));
	}

	// normalize
	inline float2 normalize(float2 v)
	{
		float invLen = rsqrtf(dot(v, v));
		return v * invLen;
	}

	// floor
	inline float2 floor(const float2 v)
	{
		return make_float2(std::floor(v.x), std::floor(v.y));
	}

	// reflect
	inline float2 reflect(float2 i, float2 n)
	{
		return i - 2.0f * n * dot(n,i);
	}

	// absolute value
	inline float2 fabs(float2 v)
	{
		return make_float2(std::fabs(v.x), std::fabs(v.y));
	}

	// float3 functions
	////////////////////////////////////////////////////////////////////////////////

	// additional constructors
	inline float3 make_float3(float s)
	{
		return make_float3(s, s, s);
	}
	inline float3 make_float3(float2 a)
	{
		return make_float3(a.x, a.y, 0.0f);
	}
	inline float3 make_float3(float2 a, float s)
	{
		return make_float3(a.x, a.y, s);
	}
	inline float3 make_float3(float4 a)
	{
		return make_float3(a.x, a.y, a.z);  // discards w
	}
	inline float3 make_float3(int3 a)
	{
		return make_float3(float(a.x), float(a.y), float(a.z));
	}

	// negate
	inline float3 operator-(float3 &a)
	{
		return make_float3(-a.x, -a.y, -a.z);
	}

	// min
	static inline float3 fminf(float3 a, float3 b)
	{
		return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
	}

	// max
	static inline float3 fmaxf(float3 a, float3 b)
	{
		return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
	}

	// addition
	inline float3 operator+(float3 a, float3 b)
	{
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	inline float3 operator+(float3 a, float b)
	{
		return make_float3(a.x + b, a.y + b, a.z + b);
	}
	inline void operator+=(float3 &a, float3 b)
	{
		a.x += b.x; a.y += b.y; a.z += b.z;
	}

	// subtract
	inline float3 operator-(float3 a, float3 b)
	{
		return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	inline float3 operator-(float3 a, float b)
	{
		return make_float3(a.x - b, a.y - b, a.z - b);
	}
	inline void operator-=(float3 &a, float3 b)
	{
		a.x -= b.x; a.y -= b.y; a.z -= b.z;
	}

	// multiply
	inline float3 operator*(float3 a, float3 b)
	{
		return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
	}
	inline float3 operator*(float3 a, float s)
	{
		return make_float3(a.x * s, a.y * s, a.z * s);
	}
	inline float3 operator*(float s, float3 a)
	{
		return make_float3(a.x * s, a.y * s, a.z * s);
	}
	inline void operator*=(float3 &a, float s)
	{
		a.x *= s; a.y *= s; a.z *= s;
	}

	// divide
	inline float3 operator/(float3 a, float3 b)
	{
		return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
	}
	inline float3 operator/(float3 a, float s)
	{
		float inv = 1.0f / s;
		return a * inv;
	}
	inline float3 operator/(float s, float3 a)
	{
		float inv = 1.0f / s;
		return a * inv;
	}
	inline void operator/=(float3 &a, float s)
	{
		float inv = 1.0f / s;
		a *= inv;
	}
	//inline float & operator[](float3 a, int n)
	//{
	//    float * ptr = (float*)&a;
	//    ptr += n;
	//    return *ptr;
	//}

	// lerp
	inline float3 lerp(float3 a, float3 b, float t)
	{
		return a + t*(b-a);
	}

	// clamp
	inline float3 clamp(float3 v, float a, float b)
	{
		return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
	}

	inline float3 clamp(float3 v, float3 a, float3 b)
	{
		return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
	}

	// dot product
	inline float dot(float3 a, float3 b)
	{ 
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	// cross product
	inline float3 cross(float3 a, float3 b)
	{ 
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
	}

	// length
	inline float length(float3 v)
	{
		return sqrtf(dot(v, v));
	}

	// length2
	inline float length2(float3 v)
	{
		return dot(v, v);
	}

	// normalize
	inline float3 normalize(float3 v)
	{
		float invLen = rsqrtf(dot(v, v));
		return v * invLen;
	}

	// floor
	inline float3 floor(const float3 v)
	{
		return make_float3(std::floor(v.x), std::floor(v.y), std::floor(v.z));
	}

	// ceil
	inline float3 ceil(const float3 v)
	{
		return make_float3(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z));
	}

	// reflect
	inline float3 reflect(float3 i, float3 n)
	{
		return i - 2.0f * n * dot(n,i);
	}

	// absolute value
	inline float3 fabs(float3 v)
	{
		return make_float3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
	}

	inline unsigned int floorLog2 (unsigned int x) {
		unsigned int y = 0;
		while (x >>= 1) y ++;
		return y;
	}
#if defined(__APPLE__) || defined(WIN32) || defined(__CYGWIN__)
	typedef unsigned int uint;
#endif
	typedef float3 Point;
	typedef float3 Vector;
}	// namespace vltools

#endif
