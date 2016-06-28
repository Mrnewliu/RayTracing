#define _USE_MATH_DEFINES

#include <algorithm>
#include <iostream>
#include <fstream>
#include <climits>
#include <sstream>
#include <limits>
#include <vector>
#include <cmath>
#include <queue>
#include <list>
#include <set>
#include <map>
#include <omp.h>
#include "lib_SimpleObject.h"
#include "lib_INIReader.h"
#include "lib_EasyBMP.h"

using namespace std;

static const int OBJECT_SPHERE = 1;
static const int OBJECT_PLANE = 2;
static const int OBJECT_POLYHEDRON = 3;
static const int OBJECT_PLANE_IN_POLYHEDRON = 4;

static const double DOUBLE_MIN = 1e-8;
static const int UNIFORM_GRIDS_X = 8;
static const int UNIFORM_GRIDS_Y = 8;
static const int UNIFORM_GRIDS_Z = 8;

class Object {
public:  
  SimpleOBJ::Vec3f Ka; 
  SimpleOBJ::Vec3f Kds;
  SimpleOBJ::Vec3f Ks;
  SimpleOBJ::Vec3f Kdt;
  SimpleOBJ::Vec3f Kt;
  double ns;
  double nt;
  double eta;
  vector<vector<unsigned int> > texture;
  bool hasTexture;
};

class Sphere : public Object {
public:
  Sphere() : position(), radius(0) {}
  Sphere(SimpleOBJ::Vec3f newPosition, double newRadius) : position(newPosition), radius(newRadius) {}
  
  SimpleOBJ::Vec3f position;
  double radius;
};

typedef pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f> AABB;

class Polyhedron : public Object {
public:
  Polyhedron() : uniformGrids(UNIFORM_GRIDS_X, vector<vector<set<int> > >(UNIFORM_GRIDS_Y, vector<set<int> >(UNIFORM_GRIDS_Z, set<int>()))) {}
  Polyhedron(int vertexCount, int faceCount) : vertices(vertexCount, SimpleOBJ::Vec3f()), triangles(faceCount, SimpleOBJ::Vec3f()), vertexNormal(vertexCount, SimpleOBJ::Vec3f()),
    uniformGrids(UNIFORM_GRIDS_X, vector<vector<set<int> > >(UNIFORM_GRIDS_Y, vector<set<int> >(UNIFORM_GRIDS_Z, set<int>()))) {}
  
  SimpleOBJ::Vec3f position;
  vector<SimpleOBJ::Vec3f> vertices;
  vector<SimpleOBJ::Vec3f> triangles;
  Sphere boundingSphere;
  vector<SimpleOBJ::Vec3f> vertexNormal;
  vector<vector<vector<set<int> > > > uniformGrids;
  AABB aabb;
};


class LightSource {
public:
  SimpleOBJ::Vec3f position;
  SimpleOBJ::Vec3f color;
  double intensity;
};

class ViewPort {
public:
  SimpleOBJ::Vec3f viewPoint;
  SimpleOBJ::Vec3f referencePoint;
  pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f> baseVectors;
  pair<double, double> boundary;
};

class Ray {
public:
  Ray() {}
  Ray(SimpleOBJ::Vec3f newSource, SimpleOBJ::Vec3f newDirection, bool newInsideMaterial) : source(newSource), direction(newDirection), insideMaterial(newInsideMaterial) {}
  SimpleOBJ::Vec3f source;
  SimpleOBJ::Vec3f direction; // A normalized vector
  bool insideMaterial;
};

class Plane : public Object {
public:
  Plane() {}
  Plane(double newA, double newB, double newC, double newD) : a(newA), b(newB), c(newC), d(newD) {}
  // ax+by+cz+d=0
  double a, b, c, d;
};

class AmbientLight {
public:
  SimpleOBJ::Vec3f color;
  double intensity;    
};

AABB getAABB(const Sphere& sphere) {
  return AABB(sphere.position - SimpleOBJ::Vec3f(sphere.radius, sphere.radius, sphere.radius), sphere.position + SimpleOBJ::Vec3f(sphere.radius, sphere.radius, sphere.radius)); 
}

AABB getAABB(SimpleOBJ::Vec3f triangle, const vector<SimpleOBJ::Vec3f>& vertices, SimpleOBJ::Vec3f position) {
  return AABB(SimpleOBJ::Vec3f(min(min(vertices[triangle.x].x, vertices[triangle.y].x), vertices[triangle.z].x),
    min(min(vertices[triangle.x].y, vertices[triangle.y].y), vertices[triangle.z].y),
    min(min(vertices[triangle.x].z, vertices[triangle.y].z), vertices[triangle.z].z)) + position,
    SimpleOBJ::Vec3f(max(max(vertices[triangle.x].x, vertices[triangle.y].x), vertices[triangle.z].x),
    max(max(vertices[triangle.x].y, vertices[triangle.y].y), vertices[triangle.z].y),
    max(max(vertices[triangle.x].z, vertices[triangle.y].z), vertices[triangle.z].z)) + position);
}

AABB getAABB(const Polyhedron& polyhedron) {
  double minX = numeric_limits<double>::max();
  double minY = numeric_limits<double>::max();
  double minZ = numeric_limits<double>::max();
  double maxX = - numeric_limits<double>::max();
  double maxY = - numeric_limits<double>::max();
  double maxZ = - numeric_limits<double>::max();
  for (vector<SimpleOBJ::Vec3f>::const_iterator it = polyhedron.vertices.begin(); it != polyhedron.vertices.end(); ++it) {
    minX = min(it->x, minX);
    minY = min(it->y, minY);
    minZ = min(it->z, minZ);
    maxX = max(it->x, maxX);
    maxY = max(it->y, maxY);
    maxZ = max(it->z, maxZ);
  }
  return AABB(SimpleOBJ::Vec3f(minX, minY, minZ) + polyhedron.position, SimpleOBJ::Vec3f(maxX, maxY, maxZ) + polyhedron.position);
}

inline double dot(SimpleOBJ::Vec3f vec1, SimpleOBJ::Vec3f vec2) {
  return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

inline SimpleOBJ::Vec3f symmetricVector(SimpleOBJ::Vec3f targetVec, SimpleOBJ::Vec3f baseVec) {
  baseVec.Normalize();
  return 2 * dot(targetVec, baseVec) * baseVec - targetVec;
}

inline SimpleOBJ::Vec3f normalizedVector(SimpleOBJ::Vec3f targetVec) {
  targetVec.Normalize();
  return targetVec;
}

inline int clamp(int lo, int hi, int number) { 
  return max(lo, min(hi, number)); 
}

// <<intersection count, intersection position>, <<first intersection point, first intersection normal vector>, <second intersection point>, <second intersection normal> > >
typedef pair<pair<int, double>, pair<pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>, pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f> > > Intersection;

class IntersectionInfo {
public:
  IntersectionInfo() {}
  IntersectionInfo(const Intersection& newIntersection, bool newHasVolume, const pair<int, int>& newInfo = pair<int, int>(), const pair<int, int>& newExtraInfo = pair<int, int>()) : 
    intersection(newIntersection), hasVolume(newHasVolume), info(newInfo), extraInfo(newExtraInfo) {}
  
  Intersection intersection;
  bool hasVolume;
  pair<int, int> info;
  pair<int, int> extraInfo;
};

Intersection intersectRaySphere(const Ray& ray, const Sphere& sphere) {
  Intersection result;
  SimpleOBJ::Vec3f Sc_minus_R0 = sphere.position - ray.source; 
  double Loc_squared = dot(Sc_minus_R0, Sc_minus_R0);
  double tca = dot(Sc_minus_R0, ray.direction);
  double thc_squared = sphere.radius * sphere.radius - Loc_squared + tca * tca;
  if (thc_squared < -DOUBLE_MIN) {
    result.first.first = 0;
  }
  else if (thc_squared >= -DOUBLE_MIN && thc_squared <= DOUBLE_MIN) {
    result.first.first = 1;
    double t = tca; 
    result.first.second = t;
    result.second.first.first = ray.source + t * ray.direction;
    result.second.first.second = (result.second.first.first - sphere.position) / sphere.radius;
  }
  else {
    double t1 = tca - sqrt(thc_squared);
    double t2 = tca + sqrt(thc_squared);
    if (t1 < DOUBLE_MIN && t2 < DOUBLE_MIN) {
      result.first.first = 0;
    }
    else if (t1 < DOUBLE_MIN) {
      result.first.first = 1;
      result.first.second = t2;
      result.second.first.first = ray.source + t2 * ray.direction;
      result.second.first.second = - (result.second.first.first - sphere.position) / sphere.radius;
    }
    else {
      result.first.first = 2;
      result.first.second = t1;
      result.second.first.first = ray.source + t1 * ray.direction;
      result.second.first.second = (result.second.first.first - sphere.position) / sphere.radius;
      result.second.second.first = ray.source + t2 * ray.direction;
      result.second.second.second = - (result.second.second.first - sphere.position) / sphere.radius;
    }
  }
  return result;
}

Intersection intersectRayPlane(const Ray& ray, const Plane& plane) {
  Intersection result;
  SimpleOBJ::Vec3f N(plane.a, plane.b, plane.c);
  double dot_N_direction = dot(N, ray.direction);
  if (dot_N_direction < - DOUBLE_MIN || dot_N_direction > DOUBLE_MIN) {
    double t = - (dot(N, ray.source) + plane.d) / dot_N_direction;
    if (t > DOUBLE_MIN) {
      result.first.first = 1;
      result.first.second = t;
      result.second.first.first = ray.direction * t + ray.source;
      if (dot(ray.direction, N) < - DOUBLE_MIN) {
        result.second.first.second = N;
      }
      else {
        result.second.first.second = - N;
      }
    }
    else {
      result.first.first = 0;
    }
  }
  else {
    result.first.first = 0;
  }
  return result;
}

Plane triangleToPlane(SimpleOBJ::Vec3f triangle, const vector<SimpleOBJ::Vec3f>& vertices, SimpleOBJ::Vec3f position) {
  Plane result;
  double a = (vertices[triangle.y].y - vertices[triangle.x].y) * 
    (vertices[triangle.z].z - vertices[triangle.x].z) -
    (vertices[triangle.z].y - vertices[triangle.x].y) * 
    (vertices[triangle.y].z - vertices[triangle.x].z);
  double b = (vertices[triangle.y].z - vertices[triangle.x].z) * 
    (vertices[triangle.z].x - vertices[triangle.x].x) -
    (vertices[triangle.z].z - vertices[triangle.x].z) * 
    (vertices[triangle.y].x - vertices[triangle.x].x);
  double c = (vertices[triangle.y].x - vertices[triangle.x].x) * 
    (vertices[triangle.z].y - vertices[triangle.x].y) -
    (vertices[triangle.z].x - vertices[triangle.x].x) * 
    (vertices[triangle.y].y - vertices[triangle.x].y);
  double length = sqrt(a * a + b * b + c * c);
  a /= length;
  b /= length;
  c /= length;
  double d = - a * (position.x + vertices[triangle.x].x)
    - b * (position.y + vertices[triangle.x].y) 
    - c * (position.z + vertices[triangle.x].z);

  result.a = a;
  result.b = b;
  result.c = c;
  result.d = d;
  return result;
}


Sphere getBoundingSphere(const Polyhedron& polyhedron) {
  double minX = numeric_limits<double>::max();
  double minY = numeric_limits<double>::max();
  double minZ = numeric_limits<double>::max();
  double maxX = - numeric_limits<double>::max();
  double maxY = - numeric_limits<double>::max();
  double maxZ = - numeric_limits<double>::max();
  for (vector<SimpleOBJ::Vec3f>::const_iterator it = polyhedron.vertices.begin(); it != polyhedron.vertices.end(); ++it) {
    minX = min(it->x, minX);
    minY = min(it->y, minY);
    minZ = min(it->z, minZ);
    maxX = max(it->x, maxX);
    maxY = max(it->y, maxY);
    maxZ = max(it->z, maxZ);
  }
  
  double xLength = maxX - minX;
  double yLength = maxY - minY;
  double zLength = maxZ - minZ;
  
  SimpleOBJ::Vec3f center(0, 0, 0);
  double radius = 0;
  
  if (xLength > yLength && xLength > zLength) {
    radius = xLength / 2;
    center.x = minX + radius;
  }
  else if (yLength > xLength && yLength > zLength) {
    radius = yLength / 2;
    center.y = minY + radius;
  }
  else {
    radius = zLength / 2;
    center.z = minZ + radius;
  }
  
  for (vector<SimpleOBJ::Vec3f>::const_iterator it = polyhedron.vertices.begin(); it != polyhedron.vertices.end(); ++it) {
    SimpleOBJ::Vec3f OP = *it - center;
    double OP_length = sqrt(OP.L2Norm_Sqr());
    if (OP_length > radius) {
      center += OP * ((1 - radius / OP_length) / 2);
      radius = (radius + OP_length) / 2;
    }
  }
  return Sphere(center + polyhedron.position, radius);
}

Intersection intersectRayTriangle(const Ray& ray, SimpleOBJ::Vec3f triangle, const vector<SimpleOBJ::Vec3f>& vertices, SimpleOBJ::Vec3f position) {
  Plane plane = triangleToPlane(triangle, vertices, position);
  Intersection intersection = intersectRayPlane(ray, plane);
  if (intersection.first.first > 0) {
    pair<double, double> projectedIntersection;
    pair<double, double> projectedVertex1;
    pair<double, double> projectedVertex2;
    pair<double, double> projectedVertex3;
    if (abs(plane.c) > DOUBLE_MIN) {
      projectedIntersection.first = intersection.second.first.first.x;
      projectedIntersection.second = intersection.second.first.first.y;
      projectedVertex1.first = position.x + vertices[triangle.x].x;
      projectedVertex1.second = position.y + vertices[triangle.x].y;
      projectedVertex2.first = position.x + vertices[triangle.y].x;
      projectedVertex2.second = position.y + vertices[triangle.y].y;
      projectedVertex3.first = position.x + vertices[triangle.z].x;
      projectedVertex3.second = position.y + vertices[triangle.z].y;
    }
    else if (abs(plane.b) > DOUBLE_MIN) {
      projectedIntersection.first = intersection.second.first.first.x;
      projectedIntersection.second = intersection.second.first.first.z;
      projectedVertex1.first = position.x + vertices[triangle.x].x;
      projectedVertex1.second = position.z + vertices[triangle.x].z;
      projectedVertex2.first = position.x + vertices[triangle.y].x;
      projectedVertex2.second = position.z + vertices[triangle.y].z;
      projectedVertex3.first = position.x + vertices[triangle.z].x;
      projectedVertex3.second = position.z + vertices[triangle.z].z;
    }
    else {
      projectedIntersection.first = intersection.second.first.first.y;
      projectedIntersection.second = intersection.second.first.first.z;
      projectedVertex1.first = position.y + vertices[triangle.x].y;
      projectedVertex1.second = position.z + vertices[triangle.x].z;
      projectedVertex2.first = position.y + vertices[triangle.y].y;
      projectedVertex2.second = position.z + vertices[triangle.y].z;
      projectedVertex3.first = position.y + vertices[triangle.z].y;
      projectedVertex3.second = position.z + vertices[triangle.z].z;
    }
    pair<double, double> PA = pair<double, double>(projectedVertex1.first - projectedIntersection.first, projectedVertex1.second - projectedIntersection.second);
    pair<double, double> PB = pair<double, double>(projectedVertex2.first - projectedIntersection.first, projectedVertex2.second - projectedIntersection.second);
    pair<double, double> PC = pair<double, double>(projectedVertex3.first - projectedIntersection.first, projectedVertex3.second - projectedIntersection.second);
    double PA_PB = PA.first * PB.second - PA.second * PB.first;
    double PB_PC = PB.first * PC.second - PB.second * PC.first;
    double PC_PA = PC.first * PA.second - PC.second * PA.first;
    if (!((PA_PB > -DOUBLE_MIN && PB_PC > -DOUBLE_MIN && PC_PA > -DOUBLE_MIN) || (PA_PB < DOUBLE_MIN && PB_PC < DOUBLE_MIN && PC_PA < DOUBLE_MIN))) {
      intersection.first.first = 0;
    }
  }
  return intersection;
}

IntersectionInfo intersectRayPolyhedron(const Ray& ray, const Polyhedron& polyhedron) {
  if (intersectRaySphere(ray, polyhedron.boundingSphere).first.first > 0) {
    set<int> possibleTriangles;
    AABB aabb = getAABB(polyhedron);
    SimpleOBJ::Vec3f selectedPlanes(ray.direction.x > 0 ? aabb.first.x : aabb.second.x,
      ray.direction.y > 0 ? aabb.first.y : aabb.second.y,
      ray.direction.z > 0 ? aabb.first.z : aabb.second.z);
    Intersection intersectionX = intersectRayPlane(ray, Plane(1, 0, 0, - selectedPlanes.x));
    Intersection intersectionY = intersectRayPlane(ray, Plane(0, 1, 0, - selectedPlanes.y));
    Intersection intersectionZ = intersectRayPlane(ray, Plane(0, 0, 1, - selectedPlanes.z));
    
    SimpleOBJ::Vec3f t(-1, -1, -1);
    
    if (intersectionX.first.first > 0 &&
      intersectionX.second.first.first.y > aabb.first.y && intersectionX.second.first.first.y < aabb.second.y &&
      intersectionX.second.first.first.z > aabb.first.z && intersectionX.second.first.first.z < aabb.second.z) {
      t.x = intersectionX.first.second;
    }
    
    if (intersectionY.first.first > 0 &&
      intersectionY.second.first.first.x > aabb.first.x && intersectionY.second.first.first.x < aabb.second.x &&
      intersectionY.second.first.first.z > aabb.first.z && intersectionY.second.first.first.z < aabb.second.z) {
      t.y = intersectionY.first.second;
    }
    
    if (intersectionZ.first.first > 0 &&
      intersectionZ.second.first.first.x > aabb.first.x && intersectionZ.second.first.first.x < aabb.second.x &&
      intersectionZ.second.first.first.y > aabb.first.y && intersectionZ.second.first.first.y < aabb.second.y) {
      t.z = intersectionZ.first.second;
    }
    
    double minT = -1;
    if (t.x > 0 && t.y > 0 && t.z > 0) {
      minT = min(min(t.x, t.y), t.z);
    }
    else if (t.x > 0 && t.y > 0) {
      minT = min(t.x, t.y);
    }
    else if (t.x > 0 && t.z > 0) {
      minT = min(t.x, t.z);
    }
    else if (t.y > 0 && t.z > 0) {
      minT = min(t.y, t.z);
    }
    else if (t.x > 0) {
      minT = t.x;
    }
    else if (t.y > 0) {
      minT = t.y;
    }
    else if (t.z > 0) {
      minT = t.z;
    }
    
    int mainAxis = -1;
    
    if (abs(minT - t.x) < DOUBLE_MIN) {
      mainAxis = 0;
    }
    else if (abs(minT - t.y) < DOUBLE_MIN) {
      mainAxis = 1;
    }
    else if (abs(minT - t.z) < DOUBLE_MIN) {
      mainAxis = 2;
    }
    else {
      return IntersectionInfo(Intersection(pair<int, double>(0, 0), pair<pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>, pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>>()),
        true, pair<int, int>(OBJECT_PLANE_IN_POLYHEDRON, 0));
    }
    
    SimpleOBJ::Vec3f inPoint = ray.insideMaterial ? ray.source : (ray.source + minT * ray.direction);
    int xStart = (ray.direction.x < 0) ? UNIFORM_GRIDS_X - 1 : 0;
    int yStart = (ray.direction.y < 0) ? UNIFORM_GRIDS_Y - 1 : 0;
    int zStart = (ray.direction.z < 0) ? UNIFORM_GRIDS_Z - 1 : 0;
    SimpleOBJ::Vec3f aabbSize = aabb.second - aabb.first;
    int currentX = (mainAxis == 0) ? xStart : (inPoint - aabb.first).x / aabbSize.x * UNIFORM_GRIDS_X;
    int currentY = (mainAxis == 1) ? yStart : (inPoint - aabb.first).y / aabbSize.y * UNIFORM_GRIDS_Y;
    int currentZ = (mainAxis == 2) ? zStart : (inPoint - aabb.first).z / aabbSize.z * UNIFORM_GRIDS_Z;
    currentX = clamp(0, UNIFORM_GRIDS_X - 1, currentX);
    currentY = clamp(0, UNIFORM_GRIDS_Y - 1, currentY);
    currentZ = clamp(0, UNIFORM_GRIDS_Z - 1, currentZ);
    SimpleOBJ::Vec3f dt = (aabb.second - aabb.first) / SimpleOBJ::Vec3f(UNIFORM_GRIDS_X, UNIFORM_GRIDS_Y, UNIFORM_GRIDS_Z) / ray.direction;
    
    SimpleOBJ::Vec3f tNext = inPoint - SimpleOBJ::Vec3f(aabb.first.x + currentX * (aabbSize.x / UNIFORM_GRIDS_X), aabb.first.y + currentY * (aabbSize.y / UNIFORM_GRIDS_Y),
      aabb.first.z + currentZ * (aabbSize.z / UNIFORM_GRIDS_Z));
    /*
    cout << "currentX = " << currentX << ", currentY = " << currentY << ", currentZ = " << currentZ << endl;
    cout << "tNext = " << tNext << endl;
    cout << "ray.direction = " << ray.direction << endl;
    cout << "inPoint = " << inPoint << endl;
    cout << "aabb = " << aabb.first << ", " << aabb.second << endl << endl;
    */
    while (currentX >= 0 && currentX < UNIFORM_GRIDS_X &&
      currentY >= 0 && currentY < UNIFORM_GRIDS_Y &&
      currentZ >= 0 && currentZ < UNIFORM_GRIDS_Z) {
      //cout << polyhedron.uniformGrids[currentX][currentY][currentZ].size() << endl;
      possibleTriangles.insert(polyhedron.uniformGrids[currentX][currentY][currentZ].begin(), polyhedron.uniformGrids[currentX][currentY][currentZ].end());
      if (tNext.x < tNext.y && tNext.x < tNext.z) {
        currentX += (ray.direction.x > 0) ? 1 : -1;
        tNext.x += dt.x;
      }
      else if (tNext.y < tNext.x && tNext.y < tNext.z) {
        currentY += (ray.direction.y > 0) ? 1 : -1;
        tNext.y += dt.y;
      }
      else {
        currentZ += (ray.direction.z > 0) ? 1 : -1;
        tNext.z += dt.z;
      }
    }
    
    vector<IntersectionInfo> localIntersections;
    //cout << possibleTriangles.size() << endl;
    
    for (set<int>::iterator it2 = possibleTriangles.begin(); it2 != possibleTriangles.end(); ++it2) {
      Intersection localIntersection = intersectRayTriangle(ray, polyhedron.triangles[*it2], polyhedron.vertices, polyhedron.position);
      if (localIntersection.first.first > 0) {
        localIntersections.push_back(IntersectionInfo(localIntersection, true, pair<int, int>(OBJECT_PLANE_IN_POLYHEDRON, *it2)));
      }
    }
    /*
    for (vector<SimpleOBJ::Vec3f>::const_iterator it2 = polyhedron.triangles.begin(); it2 != polyhedron.triangles.end(); ++it2) {
      Intersection localIntersection = intersectRayTriangle(ray, *it2, polyhedron.vertices, polyhedron.position);
      if (localIntersection.first.first > 0) {
        localIntersections.push_back(IntersectionInfo(localIntersection, true, pair<int, int>(OBJECT_PLANE_IN_POLYHEDRON, it2 - polyhedron.triangles.begin())));
      }
    }
    */
  
    if (localIntersections.size() > 0) {
      double minT = numeric_limits<double>::max();
      vector<IntersectionInfo>::iterator minId;
      for (vector<IntersectionInfo>::iterator it2 = localIntersections.begin(); it2 != localIntersections.end(); ++it2) {
        if (it2->intersection.first.second < minT) {
          minT = it2->intersection.first.second;
          minId = it2;
        }
      }
      //cout << "minId->info.second = " << minId->info.second << endl;
      return IntersectionInfo(minId->intersection, true, minId->info);
    }
    else {
      return IntersectionInfo(Intersection(pair<int, double>(0, 0), pair<pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>, pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>>()),
        true, pair<int, int>(OBJECT_PLANE_IN_POLYHEDRON, 0));
    }
  }
  else {
    return IntersectionInfo(Intersection(pair<int, double>(0, 0), pair<pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>, pair<SimpleOBJ::Vec3f, SimpleOBJ::Vec3f>>()),
      true, pair<int, int>(OBJECT_PLANE_IN_POLYHEDRON, 0));
  }
}

SimpleOBJ::Vec3f RayTracing(const Ray& sight, const vector<Plane>& planes, const vector<Sphere>& spheres,
  const vector<Polyhedron>& polyhedrons, const vector<LightSource>& lightSources, const AmbientLight& ambientLight, SimpleOBJ::Vec3f weight, double minWeight) {
  SimpleOBJ::Vec3f currentIntensity(0, 0, 0);
  
  if (weight.r + weight.g + weight.b > minWeight) {    
    vector<IntersectionInfo> intersections;
    for (vector<Sphere>::const_iterator it = spheres.begin(); it != spheres.end(); ++it) {
      Intersection intersection = intersectRaySphere(sight, *it);
      if (intersection.first.first > 0) {
        intersections.push_back(IntersectionInfo(intersection, true, pair<int, int>(OBJECT_SPHERE, it - spheres.begin())));          
      }
    }
    
    for (vector<Plane>::const_iterator it = planes.begin(); it != planes.end(); ++it) {
      Intersection intersection = intersectRayPlane(sight, *it);
      if (intersection.first.first > 0) {
        intersections.push_back(IntersectionInfo(intersection, false, pair<int, int>(OBJECT_PLANE, it - planes.begin())));
      }
    }
    
    for (vector<Polyhedron>::const_iterator it = polyhedrons.begin(); it != polyhedrons.end(); ++it) {
      IntersectionInfo intersectionInfo = intersectRayPolyhedron(sight, *it);
      if (intersectionInfo.intersection.first.first > 0) {
        intersections.push_back(IntersectionInfo(intersectionInfo.intersection, true, pair<int, int>(OBJECT_POLYHEDRON, it - polyhedrons.begin()), intersectionInfo.info));
      }
    }
    
    if (intersections.size() > 0) {
      double minT = numeric_limits<double>::max();
      vector<IntersectionInfo>::iterator minId;
      for (vector<IntersectionInfo>::iterator it = intersections.begin(); it != intersections.end(); ++it) {
        if (it->intersection.first.second < minT) {
          minT = it->intersection.first.second;
          minId = it;
        }
      }

      const Object* minObject = 0;
      if (minId->info.first == OBJECT_PLANE) {
        minObject = &planes[minId->info.second];
      }
      else if (minId->info.first == OBJECT_SPHERE) {
        minObject = &spheres[minId->info.second];
      }
      else if (minId->info.first == OBJECT_POLYHEDRON) {
        minObject = &polyhedrons[minId->info.second];
      }
      
      SimpleOBJ::Vec3f N;
      if (minId->info.first != OBJECT_POLYHEDRON) {
        N = normalizedVector(minId->intersection.second.first.second);
      }
      else {
        SimpleOBJ::Vec3f A = polyhedrons[minId->info.second].position + polyhedrons[minId->info.second].vertices[polyhedrons[minId->info.second].triangles[minId->extraInfo.second].x];
        SimpleOBJ::Vec3f B = polyhedrons[minId->info.second].position + polyhedrons[minId->info.second].vertices[polyhedrons[minId->info.second].triangles[minId->extraInfo.second].y];
        SimpleOBJ::Vec3f C = polyhedrons[minId->info.second].position + polyhedrons[minId->info.second].vertices[polyhedrons[minId->info.second].triangles[minId->extraInfo.second].z];
        SimpleOBJ::Vec3f I = minId->intersection.second.first.first;
        
        double p = abs(B.x * I.y - I.x * B.y - C.x * I.y + I.x * C.y - B.x * A.y + A.x * B.y + C.x * A.y - A.x * C.y) /
          abs(A.x * B.y - B.x * A.y - A.x * C.y + C.x * A.y + B.x * C.y - C.x * B.y);

        SimpleOBJ::Vec3f D = (1 - p) * A + p * B;
        SimpleOBJ::Vec3f E = (1 - p) * A + p * C;
        
        double q = sqrt((I - D).L2Norm_Sqr()) / sqrt((E - D).L2Norm_Sqr());
        
        SimpleOBJ::Vec3f NA = polyhedrons[minId->info.second].vertexNormal[polyhedrons[minId->info.second].triangles[minId->extraInfo.second].x];
        SimpleOBJ::Vec3f NB = polyhedrons[minId->info.second].vertexNormal[polyhedrons[minId->info.second].triangles[minId->extraInfo.second].y];
        SimpleOBJ::Vec3f NC = polyhedrons[minId->info.second].vertexNormal[polyhedrons[minId->info.second].triangles[minId->extraInfo.second].z];
        
        SimpleOBJ::Vec3f ND = normalizedVector(p * NB + (1 - p) * NA);
        SimpleOBJ::Vec3f NE = normalizedVector(p * NC + (1 - p) * NA);
        
        N = normalizedVector(q * NE + (1 - q) * ND);
        
        if (sight.insideMaterial) {
          N = -N;
        }
      }
      SimpleOBJ::Vec3f V = normalizedVector(- sight.direction);
      SimpleOBJ::Vec3f VR = symmetricVector(V, N);
      SimpleOBJ::Vec3f VT(0, 0, 0);
      double dot_N_V = dot(N, V);
      
      double eta = sight.insideMaterial ? minObject->eta : 1 / minObject->eta;
      double eta_dot_N_V = 1 - eta * eta * (1 - dot_N_V * dot_N_V);
      if (eta_dot_N_V > DOUBLE_MIN) {
        VT = normalizedVector(-eta * V - (sqrt(eta_dot_N_V) - eta * dot_N_V) * N);
      }

      if (!sight.insideMaterial) {
        currentIntensity.r += ambientLight.intensity * ambientLight.color.r * minObject->Ka.r;
        currentIntensity.g += ambientLight.intensity * ambientLight.color.g * minObject->Ka.g;
        currentIntensity.b += ambientLight.intensity * ambientLight.color.b * minObject->Ka.b;
        
        for (vector<LightSource>::const_iterator it = lightSources.begin(); it != lightSources.end(); ++it) {
          bool isHidden = false;
          SimpleOBJ::Vec3f L = it->position - minId->intersection.second.first.first;
          Ray shadowTest(minId->intersection.second.first.first, normalizedVector(L), false);
          
          for (vector<Sphere>::const_iterator it2 = spheres.begin(); it2 != spheres.end(); ++it2) {
            if (minId->info.first != OBJECT_SPHERE || minId->info.second != it2 - spheres.begin()) {
              Intersection intersection = intersectRaySphere(shadowTest, *it2);
              if (intersection.first.first > 0) {
                if (intersection.first.second * intersection.first.second < L.L2Norm_Sqr()) {
                  isHidden = true;
                  break;    
                }   
              }
            }
            else {
              if (dot(N, L) < 0) {
                isHidden = true;
                break;
              }
            }
          }
          if (isHidden) {
            continue;
          }
          
          for (vector<Plane>::const_iterator it2 = planes.begin(); it2 != planes.end(); ++it2) {
            if (minId->info.first != OBJECT_PLANE || minId->info.second != it2 - planes.begin()) {
              Intersection intersection = intersectRayPlane(shadowTest, *it2);
              if (intersection.first.first > 0) {
                if (intersection.first.second * intersection.first.second < L.L2Norm_Sqr()) {
                  isHidden = true;
                  break;    
                }   
              }
            }
            else {
              if (dot(N, L) < 0) {
                isHidden = true;
                break;
              }
            }
          }
          if (isHidden) {
            continue;
          }
          
          for (vector<Polyhedron>::const_iterator it2 = polyhedrons.begin(); it2 != polyhedrons.end(); ++it2) {
            if (minId->info.second != it2 - polyhedrons.begin()) {
              IntersectionInfo intersectionInfo = intersectRayPolyhedron(shadowTest, *it2);
              if (intersectionInfo.intersection.first.first > 0) {
                if (intersectionInfo.intersection.first.second * intersectionInfo.intersection.first.second < L.L2Norm_Sqr()) {
                  isHidden = true;
                  break;
                }
              }
            }
            else {
              if (dot(N, L) < 0) {
                isHidden = true;
                break;
              }
              else {
                for (vector<SimpleOBJ::Vec3f>::const_iterator it3 = it2->triangles.begin(); it3 != it2->triangles.end(); ++it3) {
                  if (minId->extraInfo.second != it3 - it2->triangles.begin()) {
                    Intersection intersection = intersectRayTriangle(shadowTest, *it3, it2->vertices, it2->position);
                    if (intersection.first.first > 0) {
                      if (intersection.first.second * intersection.first.second < L.L2Norm_Sqr()) {
                        isHidden = true;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
          if (isHidden) {
            continue;
          }
          
          SimpleOBJ::Vec3f L_normalized = normalizedVector(L);
          SimpleOBJ::Vec3f R = symmetricVector(L_normalized, N);
          SimpleOBJ::Vec3f intensity_color = it->intensity * it->color;
          double dot_N_L = dot(N, L_normalized);
          double eta_dot_N_L = 1 - eta * eta * (1 - dot_N_L * dot_N_L);
          SimpleOBJ::Vec3f T;
          double dot_T_V = 0; 
          if (eta_dot_N_L > DOUBLE_MIN) {
            T = normalizedVector(-eta * L_normalized - (sqrt(eta_dot_N_L) - eta * dot_N_L) * N);
            dot_T_V = dot(T, V);
          }
          double dot_R_V = dot(R, V);
          double dot_L_N = dot(normalizedVector(L), N);
          
          SimpleOBJ::Vec3f Kds;

          if (minObject->hasTexture) {
            int idX = 0;
            int idY = 0;
            if (minId->info.first == OBJECT_PLANE) {
              SimpleOBJ::Vec3f vecX;
              SimpleOBJ::Vec3f vecY; 
              if (abs(minId->intersection.second.first.second.x) <= DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) <= DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) > DOUBLE_MIN) {
                  vecX = SimpleOBJ::Vec3f(1, 0, 0);
                  vecY = SimpleOBJ::Vec3f(0, 1, 0);
              }
              else if (abs(minId->intersection.second.first.second.x) <= DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) > DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) <= DOUBLE_MIN) {
                  vecX = SimpleOBJ::Vec3f(1, 0, 0);
                  vecY = SimpleOBJ::Vec3f(0, 0, 1);
              }
              else if (abs(minId->intersection.second.first.second.x) > DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) <= DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) <= DOUBLE_MIN) {
                  vecX = SimpleOBJ::Vec3f(0, 1, 0);
                  vecY = SimpleOBJ::Vec3f(0, 0, 1);
              }
              else if (abs(minId->intersection.second.first.second.x) <= DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) > DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) > DOUBLE_MIN) {
                  vecX = normalizedVector(SimpleOBJ::Vec3f(0, minId->intersection.second.first.second.z, - minId->intersection.second.first.second.y));
                  vecY = normalizedVector(SimpleOBJ::Vec3f(minId->intersection.second.first.second.z, 0, - minId->intersection.second.first.second.x));
              }
              else if (abs(minId->intersection.second.first.second.x) > DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) <= DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) > DOUBLE_MIN) {
                  vecX = normalizedVector(SimpleOBJ::Vec3f(minId->intersection.second.first.second.z, 0, - minId->intersection.second.first.second.x));
                  vecY = normalizedVector(SimpleOBJ::Vec3f(minId->intersection.second.first.second.y, - minId->intersection.second.first.second.x, 0));
              }
              else if (abs(minId->intersection.second.first.second.x) > DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) > DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) <= DOUBLE_MIN) {
                  vecX = normalizedVector(SimpleOBJ::Vec3f(minId->intersection.second.first.second.y, - minId->intersection.second.first.second.x, 0));
                  vecY = normalizedVector(SimpleOBJ::Vec3f(0, minId->intersection.second.first.second.z, - minId->intersection.second.first.second.y));
              }
              else if (abs(minId->intersection.second.first.second.x) > DOUBLE_MIN && 
                abs(minId->intersection.second.first.second.y) > DOUBLE_MIN &&
                abs(minId->intersection.second.first.second.z) > DOUBLE_MIN) {
                  vecX = normalizedVector(SimpleOBJ::Vec3f(0, minId->intersection.second.first.second.z, - minId->intersection.second.first.second.y));
                  vecY = normalizedVector(SimpleOBJ::Vec3f(minId->intersection.second.first.second.z, 0, - minId->intersection.second.first.second.x));
              }
              double posX = dot(minId->intersection.second.first.first, vecX);
              double posY = dot(minId->intersection.second.first.first, vecY);
              idX = (int)posX % minObject->texture.size();
              idY = (int)posY % minObject->texture.front().size();
              if (idX < 0) {
                idX = minObject->texture.size() + idX;
              }
              if (idY < 0) {
                idY = minObject->texture.front().size() + idY;
              }
            }
            else if (minId->info.first == OBJECT_SPHERE) {
              SimpleOBJ::Vec3f relPos = normalizedVector(minId->intersection.second.first.first - spheres[minId->info.second].position);
              idX = (atan2(relPos.y, relPos.x) / M_PI / 2 + 0.5) * minObject->texture.size();
              idY = (asin(relPos.z) / M_PI + 0.5) * minObject->texture.front().size();
            }
            else if (minId->info.first == OBJECT_POLYHEDRON) {
              // TODO
            }
            idX = clamp(0, minObject->texture.size() - 1, idX);
            idY = clamp(0, minObject->texture.size() - 1, idY);
            Kds.r = (double)((minObject->texture[idX][idY] & (unsigned int)16711680) >> 16) / 256;
            Kds.g = (double)((minObject->texture[idX][idY] & (unsigned int)65280) >> 8) / 256;
            Kds.b = (double)(minObject->texture[idX][idY] & (unsigned int)255) / 256;
          }
          else {
            Kds = minObject->Kds;
          }
          
          currentIntensity += intensity_color * Kds * (dot_L_N > 0 ? dot_L_N : 0);
          currentIntensity += intensity_color * minObject->Kdt * (dot_L_N < 0 ? dot_L_N : 0);
          currentIntensity += intensity_color * minObject->Ks * (dot_R_V > 0 ? pow(dot_R_V, minObject->ns) : 0);
          currentIntensity += intensity_color * minObject->Kt * (dot_T_V > 0 ? pow(dot_T_V, minObject->nt) : 0);
        }
      }
      
      currentIntensity += RayTracing(Ray(minId->intersection.second.first.first, VR, sight.insideMaterial),
        planes, spheres, polyhedrons, lightSources, ambientLight, weight * minObject->Ks, minWeight);
      if (minId->hasVolume && eta_dot_N_V > DOUBLE_MIN) {
        currentIntensity += RayTracing(Ray(minId->intersection.second.first.first, VT, !sight.insideMaterial),
          planes, spheres, polyhedrons, lightSources, ambientLight, weight * minObject->Kt, minWeight);
      }
    }
  }
  return weight * currentIntensity;
}

void generateLineLightSource(const LightSource& lightSource1, const LightSource& lightSource2, double segments, vector<LightSource>& lightSources) {  
  SimpleOBJ::Vec3f positionStep = (lightSource2.position - lightSource1.position) / segments;
  SimpleOBJ::Vec3f colorStep = (lightSource2.color - lightSource1.color) / segments;
  double intensityStep = (lightSource2.intensity - lightSource1.intensity) / segments / segments * 2;
  double baseIntensity = lightSource1.intensity / segments * 2;
  
  for (int i = 0; i < segments; ++i) {
    LightSource currentLightSource;
    currentLightSource.position = lightSource1.position + positionStep * i;
    currentLightSource.color = lightSource1.color + colorStep * i;
    currentLightSource.intensity = baseIntensity + intensityStep * i;
    lightSources.push_back(currentLightSource);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " <input file> <output file>" << endl;
    return 1;
  }
  INIReader inputFile(argv[1]);
  if (inputFile.ParseError() < 0) {
    cout << "Input file error" << endl;
    return 1;
  }
  
  omp_set_num_threads(4);
  
  cout << "Reading file..." << endl;
  
  // general settings
  int pictureWidth = inputFile.GetInteger("general", "pictureWidth", 640);
  int pictureHeight = inputFile.GetInteger("general", "pictureHeight", 480);
  int polyhedronCount = inputFile.GetInteger("general", "polyhedronCount", 0);
  int lightSourceCount = inputFile.GetInteger("general", "lightSourceCount", 0);
  int sphereCount = inputFile.GetInteger("general", "sphereCount", 0);
  int planeCount = inputFile.GetInteger("general", "planeCount", 0);
  int lineLightSourceCount = inputFile.GetInteger("general", "lineLightSourceCount", 0);
  int rectangleLightSourceCount = inputFile.GetInteger("general", "rectangleLightSourceCount", 0);
  double minWeight = inputFile.GetReal("general", "minWeight", 0.05);
  bool antiAliasing = inputFile.GetBoolean("general", "antiAliasing", false);
  
  // objects
  vector<Polyhedron> polyhedrons;
  vector<Sphere> spheres;
  vector<Plane> planes; 
  
  // light sources
  vector<LightSource> lightSources;

  // view point and view port
  ViewPort viewPort;
  
  // ambient light
  AmbientLight ambientLight;
  
  ambientLight.color.r = inputFile.GetReal("general", "ambientLightR", 0);
  ambientLight.color.g = inputFile.GetReal("general", "ambientLightG", 0);
  ambientLight.color.b = inputFile.GetReal("general", "ambientLightB", 0);
  ambientLight.intensity = inputFile.GetReal("general", "ambientLightIntensity", 0);
  
  viewPort.viewPoint.x = inputFile.GetReal("general", "viewPointX", 0);
  viewPort.viewPoint.y = inputFile.GetReal("general", "viewPointY", 0);
  viewPort.viewPoint.z = inputFile.GetReal("general", "viewPointZ", 0);
  viewPort.referencePoint.x = inputFile.GetReal("general", "viewPortX0", 0);
  viewPort.referencePoint.y = inputFile.GetReal("general", "viewPortY0", 0);
  viewPort.referencePoint.z = inputFile.GetReal("general", "viewPortZ0", 0);
  viewPort.baseVectors.first.x = inputFile.GetReal("general", "viewPortA1", 0);
  viewPort.baseVectors.first.y = inputFile.GetReal("general", "viewPortA2", 0);
  viewPort.baseVectors.first.z = inputFile.GetReal("general", "viewPortA3", 0);
  viewPort.baseVectors.second.x = inputFile.GetReal("general", "viewPortB1", 0);
  viewPort.baseVectors.second.y = inputFile.GetReal("general", "viewPortB2", 0);
  viewPort.baseVectors.second.z = inputFile.GetReal("general", "viewPortB3", 0);
  viewPort.boundary.first = inputFile.GetReal("general", "viewPortMaxS", 0);
  viewPort.boundary.second = inputFile.GetReal("general", "viewPortMaxT", 0);
  
  for (int i = 0; i < polyhedronCount; ++i) {
    ostringstream tempStringStream;
    tempStringStream << "polyhedron" << i;
    SimpleOBJ::CSimpleObject objFile;
    
    objFile.LoadFromObj(inputFile.Get(tempStringStream.str(), "file", "").c_str());
    polyhedrons.push_back(Polyhedron(objFile.m_nVertices, objFile.m_nTriangles));
    double magnification = inputFile.GetReal(tempStringStream.str(), "magnification", 1);
    polyhedrons.back().position.x = inputFile.GetReal(tempStringStream.str(), "baseX", 0);
    polyhedrons.back().position.y = inputFile.GetReal(tempStringStream.str(), "baseY", 0);
    polyhedrons.back().position.z = inputFile.GetReal(tempStringStream.str(), "baseZ", 0);
    
    for (int j = 0; j < objFile.m_nVertices; ++j) {
      polyhedrons.back().vertices[j].x = objFile.m_pVertexList[j].x * magnification;
      polyhedrons.back().vertices[j].y = objFile.m_pVertexList[j].y * magnification;
      polyhedrons.back().vertices[j].z = (- objFile.m_pVertexList[j].z) * magnification;
    }
    
    vector<SimpleOBJ::Vec3f> faceNormal(objFile.m_nTriangles, SimpleOBJ::Vec3f());
    vector<set<int> > triangleOfVertices(objFile.m_nVertices, set<int>());
    
    polyhedrons.back().aabb = getAABB(polyhedrons.back());
    //cout << polyhedrons.back().aabb.first << ", " << polyhedrons.back().aabb.second << endl;
    
    for (int j = 0; j < objFile.m_nTriangles; ++j) {
      polyhedrons.back().triangles[j].x =  objFile.m_pTriangleList[j][0] - 1;
      polyhedrons.back().triangles[j].y =  objFile.m_pTriangleList[j][1] - 1;
      polyhedrons.back().triangles[j].z =  objFile.m_pTriangleList[j][2] - 1;
      Plane plane = triangleToPlane(polyhedrons.back().triangles[j], polyhedrons.back().vertices, polyhedrons.back().position);
      faceNormal[j] = SimpleOBJ::Vec3f(plane.a, plane.b, plane.c);
      triangleOfVertices[polyhedrons.back().triangles[j].x].insert(j);
      triangleOfVertices[polyhedrons.back().triangles[j].y].insert(j);
      triangleOfVertices[polyhedrons.back().triangles[j].z].insert(j);
      AABB aabb = getAABB(polyhedrons.back().triangles[j], polyhedrons.back().vertices, polyhedrons.back().position);
      SimpleOBJ::Vec3f polyhedronAABBSize = polyhedrons.back().aabb.second - polyhedrons.back().aabb.first;
      SimpleOBJ::Vec3f minPos = (aabb.first - polyhedrons.back().aabb.first) / polyhedronAABBSize;
      SimpleOBJ::Vec3f maxPos = (aabb.second - polyhedrons.back().aabb.first) / polyhedronAABBSize;
      int minX = floor(minPos.x * UNIFORM_GRIDS_X - 100 * DOUBLE_MIN);
      int minY = floor(minPos.y * UNIFORM_GRIDS_Y - 100 * DOUBLE_MIN);
      int minZ = floor(minPos.z * UNIFORM_GRIDS_Z - 100 * DOUBLE_MIN);
      int maxX = ceil(maxPos.x * UNIFORM_GRIDS_X + 100 * DOUBLE_MIN);
      int maxY = ceil(maxPos.y * UNIFORM_GRIDS_Y + 100 * DOUBLE_MIN);
      int maxZ = ceil(maxPos.z * UNIFORM_GRIDS_Z + 100 * DOUBLE_MIN);
      minX = clamp(0, UNIFORM_GRIDS_X, minX);
      minY = clamp(0, UNIFORM_GRIDS_Y, minY);
      minZ = clamp(0, UNIFORM_GRIDS_Z, minZ);
      maxX = clamp(0, UNIFORM_GRIDS_X, maxX);
      maxY = clamp(0, UNIFORM_GRIDS_Y, maxY);
      maxZ = clamp(0, UNIFORM_GRIDS_Z, maxZ);
      for (int k = minX; k < maxX; ++k) {
        for (int l = minY; l < maxY; ++l) {
          for (int m = minZ; m < maxZ; ++m) {
            polyhedrons.back().uniformGrids[k][l][m].insert(j);
            //cout << "Inserted " << j << " into " << "[" << k << "][" << l << "][" << m << "]" << endl; 
          } 
        }
      }
    }
    
    polyhedrons.back().boundingSphere = getBoundingSphere(polyhedrons.back());
    
    for (int j = 0; j < objFile.m_nVertices; ++j) {

      for (set<int>::iterator it = triangleOfVertices[j].begin(); it != triangleOfVertices[j].end(); ++it) {
        polyhedrons.back().vertexNormal[j] += faceNormal[*it];
      }
      polyhedrons.back().vertexNormal[j].Normalize();
    }
    
    polyhedrons.back().Ka.r = inputFile.GetReal(tempStringStream.str(), "Kar", 0);
    polyhedrons.back().Ka.g = inputFile.GetReal(tempStringStream.str(), "Kag", 0);
    polyhedrons.back().Ka.b = inputFile.GetReal(tempStringStream.str(), "Kab", 0);
    polyhedrons.back().Kds.r = inputFile.GetReal(tempStringStream.str(), "Kdsr", 0);
    polyhedrons.back().Kds.g = inputFile.GetReal(tempStringStream.str(), "Kdsg", 0);
    polyhedrons.back().Kds.b = inputFile.GetReal(tempStringStream.str(), "Kdsb", 0);
    polyhedrons.back().Ks.r = inputFile.GetReal(tempStringStream.str(), "Ksr", 0);
    polyhedrons.back().Ks.g = inputFile.GetReal(tempStringStream.str(), "Ksg", 0);
    polyhedrons.back().Ks.b = inputFile.GetReal(tempStringStream.str(), "Ksb", 0);
    polyhedrons.back().Kdt.r = inputFile.GetReal(tempStringStream.str(), "Kdtr", 0);
    polyhedrons.back().Kdt.g = inputFile.GetReal(tempStringStream.str(), "Kdtg", 0);
    polyhedrons.back().Kdt.b = inputFile.GetReal(tempStringStream.str(), "Kdtb", 0);
    polyhedrons.back().Kt.r = inputFile.GetReal(tempStringStream.str(), "Ktr", 0);
    polyhedrons.back().Kt.g = inputFile.GetReal(tempStringStream.str(), "Ktg", 0);
    polyhedrons.back().Kt.b = inputFile.GetReal(tempStringStream.str(), "Ktb", 0);
    polyhedrons.back().ns = inputFile.GetReal(tempStringStream.str(), "ns", 0);
    polyhedrons.back().nt = inputFile.GetReal(tempStringStream.str(), "nt", 0);
    polyhedrons.back().eta = inputFile.GetReal(tempStringStream.str(), "eta", 0);
    string textureFile = inputFile.Get(tempStringStream.str(), "texture", "");
    
    if (textureFile.empty()) {
      polyhedrons.back().hasTexture = false;
    }
    else {
      BMP texture;
      texture.ReadFromFile(textureFile.c_str());
      for (int i = 0; i < texture.TellWidth(); ++i) {
        polyhedrons.back().texture.push_back(vector<unsigned int>(texture.TellHeight(), 0));
        for (int j = 0; j < texture.TellHeight(); ++j) {
          polyhedrons.back().texture[i][j] = (texture(i, j)->Red << 16) + (texture(i, j)->Green << 8) + texture(i, j)->Blue;
        }
      }
      polyhedrons.back().hasTexture = true;
    }
  }
  
  for (int i = 0; i < sphereCount; ++i) {
    ostringstream tempStringStream;
    tempStringStream << "sphere" << i;
    spheres.push_back(Sphere());
    spheres.back().radius = inputFile.GetReal(tempStringStream.str(), "radius", 0);
    spheres.back().position.x = inputFile.GetReal(tempStringStream.str(), "baseX", 0);
    spheres.back().position.y = inputFile.GetReal(tempStringStream.str(), "baseY", 0);
    spheres.back().position.z = inputFile.GetReal(tempStringStream.str(), "baseZ", 0);
    spheres.back().Ka.r = inputFile.GetReal(tempStringStream.str(), "Kar", 0);
    spheres.back().Ka.g = inputFile.GetReal(tempStringStream.str(), "Kag", 0);
    spheres.back().Ka.b = inputFile.GetReal(tempStringStream.str(), "Kab", 0);
    spheres.back().Kds.r = inputFile.GetReal(tempStringStream.str(), "Kdsr", 0);
    spheres.back().Kds.g = inputFile.GetReal(tempStringStream.str(), "Kdsg", 0);
    spheres.back().Kds.b = inputFile.GetReal(tempStringStream.str(), "Kdsb", 0);
    spheres.back().Ks.r = inputFile.GetReal(tempStringStream.str(), "Ksr", 0);
    spheres.back().Ks.g = inputFile.GetReal(tempStringStream.str(), "Ksg", 0);
    spheres.back().Ks.b = inputFile.GetReal(tempStringStream.str(), "Ksb", 0);
    spheres.back().Kdt.r = inputFile.GetReal(tempStringStream.str(), "Kdtr", 0);
    spheres.back().Kdt.g = inputFile.GetReal(tempStringStream.str(), "Kdtg", 0);
    spheres.back().Kdt.b = inputFile.GetReal(tempStringStream.str(), "Kdtb", 0);
    spheres.back().Kt.r = inputFile.GetReal(tempStringStream.str(), "Ktr", 0);
    spheres.back().Kt.g = inputFile.GetReal(tempStringStream.str(), "Ktg", 0);
    spheres.back().Kt.b = inputFile.GetReal(tempStringStream.str(), "Ktb", 0);
    spheres.back().ns = inputFile.GetReal(tempStringStream.str(), "ns", 0);
    spheres.back().nt = inputFile.GetReal(tempStringStream.str(), "nt", 0);
    spheres.back().eta = inputFile.GetReal(tempStringStream.str(), "eta", 0);
    string textureFile = inputFile.Get(tempStringStream.str(), "texture", "");
    if (textureFile.empty()) {
      spheres.back().hasTexture = false;
    }
    else {
      BMP texture;
      texture.ReadFromFile(textureFile.c_str());
      for (int i = 0; i < texture.TellWidth(); ++i) {
        spheres.back().texture.push_back(vector<unsigned int>(texture.TellHeight(), 0));
        for (int j = 0; j < texture.TellHeight(); ++j) {
          spheres.back().texture[i][j] = (texture(i, j)->Red << 16) + (texture(i, j)->Green << 8) + texture(i, j)->Blue;
        }
      }
      spheres.back().hasTexture = true;
    }
  }
  
  for (int i = 0; i < planeCount; ++i) {
    ostringstream tempStringStream;
    tempStringStream << "plane" << i;
    planes.push_back(Plane());
    planes.back().a = inputFile.GetReal(tempStringStream.str(), "a", 0);
    planes.back().b = inputFile.GetReal(tempStringStream.str(), "b", 0);
    planes.back().c = inputFile.GetReal(tempStringStream.str(), "c", 0);
    planes.back().d = inputFile.GetReal(tempStringStream.str(), "d", 0);
    planes.back().Ka.r = inputFile.GetReal(tempStringStream.str(), "Kar", 0);
    planes.back().Ka.g = inputFile.GetReal(tempStringStream.str(), "Kag", 0);
    planes.back().Ka.b = inputFile.GetReal(tempStringStream.str(), "Kab", 0);
    planes.back().Kds.r = inputFile.GetReal(tempStringStream.str(), "Kdsr", 0);
    planes.back().Kds.g = inputFile.GetReal(tempStringStream.str(), "Kdsg", 0);
    planes.back().Kds.b = inputFile.GetReal(tempStringStream.str(), "Kdsb", 0);
    planes.back().Ks.r = inputFile.GetReal(tempStringStream.str(), "Ksr", 0);
    planes.back().Ks.g = inputFile.GetReal(tempStringStream.str(), "Ksg", 0);
    planes.back().Ks.b = inputFile.GetReal(tempStringStream.str(), "Ksb", 0);
    planes.back().Kdt.r = inputFile.GetReal(tempStringStream.str(), "Kdtr", 0);
    planes.back().Kdt.g = inputFile.GetReal(tempStringStream.str(), "Kdtg", 0);
    planes.back().Kdt.b = inputFile.GetReal(tempStringStream.str(), "Kdtb", 0);
    planes.back().Kt.r = inputFile.GetReal(tempStringStream.str(), "Ktr", 0);
    planes.back().Kt.g = inputFile.GetReal(tempStringStream.str(), "Ktg", 0);
    planes.back().Kt.b = inputFile.GetReal(tempStringStream.str(), "Ktb", 0);
    planes.back().ns = inputFile.GetReal(tempStringStream.str(), "ns", 0);
    planes.back().nt = inputFile.GetReal(tempStringStream.str(), "nt", 0);
    planes.back().eta = inputFile.GetReal(tempStringStream.str(), "eta", 0);
    string textureFile = inputFile.Get(tempStringStream.str(), "texture", "");
    if (textureFile.empty()) {
      planes.back().hasTexture = false;
    }
    else {
      BMP texture;
      texture.ReadFromFile(textureFile.c_str());
      for (int i = 0; i < texture.TellWidth(); ++i) {
        planes.back().texture.push_back(vector<unsigned int>(texture.TellHeight(), 0));
        for (int j = 0; j < texture.TellHeight(); ++j) {
          planes.back().texture[i][j] = (texture(i, j)->Red << 16) + (texture(i, j)->Green << 8) + texture(i, j)->Blue;
        }
      }
      planes.back().hasTexture = true;
    }
  }
  
  for (int i = 0; i < lightSourceCount; ++i) {
    ostringstream tempStringStream;
    tempStringStream << "lightSource" << i;
    lightSources.push_back(LightSource());
    lightSources.back().position.x = inputFile.GetReal(tempStringStream.str(), "x", 0);
    lightSources.back().position.y = inputFile.GetReal(tempStringStream.str(), "y", 0);
    lightSources.back().position.z = inputFile.GetReal(tempStringStream.str(), "z", 0);
    lightSources.back().color.r = inputFile.GetReal(tempStringStream.str(), "r", 0);
    lightSources.back().color.g = inputFile.GetReal(tempStringStream.str(), "g", 0);
    lightSources.back().color.b = inputFile.GetReal(tempStringStream.str(), "b", 0);
    lightSources.back().intensity = inputFile.GetReal(tempStringStream.str(), "I", 0);
  }
  
  for (int i = 0; i < lineLightSourceCount; ++i) {
    ostringstream tempStringStream;
    tempStringStream << "lineLightSource" << i;
    LightSource lightSource1, lightSource2;
    
    lightSource1.position.x = inputFile.GetReal(tempStringStream.str(), "x1", 0);
    lightSource1.position.y = inputFile.GetReal(tempStringStream.str(), "y1", 0);
    lightSource1.position.z = inputFile.GetReal(tempStringStream.str(), "z1", 0);
    lightSource1.color.r = inputFile.GetReal(tempStringStream.str(), "r1", 0);
    lightSource1.color.g = inputFile.GetReal(tempStringStream.str(), "g1", 0);
    lightSource1.color.b = inputFile.GetReal(tempStringStream.str(), "b1", 0);
    lightSource1.intensity = inputFile.GetReal(tempStringStream.str(), "I1", 0);
    
    lightSource2.position.x = inputFile.GetReal(tempStringStream.str(), "x2", 0);
    lightSource2.position.y = inputFile.GetReal(tempStringStream.str(), "y2", 0);
    lightSource2.position.z = inputFile.GetReal(tempStringStream.str(), "z2", 0);
    lightSource2.color.r = inputFile.GetReal(tempStringStream.str(), "r2", 0);
    lightSource2.color.g = inputFile.GetReal(tempStringStream.str(), "g2", 0);
    lightSource2.color.b = inputFile.GetReal(tempStringStream.str(), "b2", 0);
    lightSource2.intensity = inputFile.GetReal(tempStringStream.str(), "I2", 0);
  
    double segments = inputFile.GetInteger(tempStringStream.str(), "segments", 0);
    generateLineLightSource(lightSource1, lightSource2, segments, lightSources);
  }
  
  for (int i = 0; i < rectangleLightSourceCount; ++i) {
    ostringstream tempStringStream;
    tempStringStream << "rectangleLightSource" << i;
    LightSource lightSource11, lightSource12, lightSource21, lightSource22;
    lightSource11.position.x = inputFile.GetReal(tempStringStream.str(), "x11", 0);
    lightSource11.position.y = inputFile.GetReal(tempStringStream.str(), "y11", 0);
    lightSource11.position.z = inputFile.GetReal(tempStringStream.str(), "z11", 0);
    lightSource11.color.r = inputFile.GetReal(tempStringStream.str(), "r11", 0);
    lightSource11.color.g = inputFile.GetReal(tempStringStream.str(), "g11", 0);
    lightSource11.color.b = inputFile.GetReal(tempStringStream.str(), "b11", 0);
    lightSource11.intensity = inputFile.GetReal(tempStringStream.str(), "I11", 0);
    
    lightSource12.position.x = inputFile.GetReal(tempStringStream.str(), "x12", 0);
    lightSource12.position.y = inputFile.GetReal(tempStringStream.str(), "y12", 0);
    lightSource12.position.z = inputFile.GetReal(tempStringStream.str(), "z12", 0);
    lightSource12.color.r = inputFile.GetReal(tempStringStream.str(), "r12", 0);
    lightSource12.color.g = inputFile.GetReal(tempStringStream.str(), "g12", 0);
    lightSource12.color.b = inputFile.GetReal(tempStringStream.str(), "b12", 0);
    lightSource12.intensity = inputFile.GetReal(tempStringStream.str(), "I12", 0);
    
    lightSource21.position.x = inputFile.GetReal(tempStringStream.str(), "x21", 0);
    lightSource21.position.y = inputFile.GetReal(tempStringStream.str(), "y21", 0);
    lightSource21.position.z = inputFile.GetReal(tempStringStream.str(), "z21", 0);
    lightSource21.color.r = inputFile.GetReal(tempStringStream.str(), "r21", 0);
    lightSource21.color.g = inputFile.GetReal(tempStringStream.str(), "g21", 0);
    lightSource21.color.b = inputFile.GetReal(tempStringStream.str(), "b21", 0);
    lightSource21.intensity = inputFile.GetReal(tempStringStream.str(), "I21", 0);
    
    lightSource22.position.x = inputFile.GetReal(tempStringStream.str(), "x22", 0);
    lightSource22.position.y = inputFile.GetReal(tempStringStream.str(), "y22", 0);
    lightSource22.position.z = inputFile.GetReal(tempStringStream.str(), "z22", 0);
    lightSource22.color.r = inputFile.GetReal(tempStringStream.str(), "r22", 0);
    lightSource22.color.g = inputFile.GetReal(tempStringStream.str(), "g22", 0);
    lightSource22.color.b = inputFile.GetReal(tempStringStream.str(), "b22", 0);
    lightSource22.intensity = inputFile.GetReal(tempStringStream.str(), "I22", 0);
    
    double segmentsX = inputFile.GetInteger(tempStringStream.str(), "segmentsX", 0);
    double segmentsY = inputFile.GetInteger(tempStringStream.str(), "segmentsY", 0);
    
    SimpleOBJ::Vec3f positionStepX1 = (lightSource12.position - lightSource11.position) / segmentsX;
    SimpleOBJ::Vec3f positionStepX2 = (lightSource22.position - lightSource21.position) / segmentsX;
    SimpleOBJ::Vec3f colorStepX1 = (lightSource12.color - lightSource11.color) / segmentsX;
    SimpleOBJ::Vec3f colorStepX2 = (lightSource22.color - lightSource21.color) / segmentsX;
    double intensityStepX1 = (lightSource12.intensity - lightSource11.intensity) / segmentsX / segmentsX * 2;
    double intensityStepX2 = (lightSource22.intensity - lightSource21.intensity) / segmentsX / segmentsX * 2;
    double baseIntensityX1 = lightSource11.intensity / segmentsX * 2;
    double baseIntensityX2 = lightSource21.intensity / segmentsX * 2;
    
    for (int i = 0; i < segmentsX; ++i) {
      LightSource lightSource1, lightSource2;
      lightSource1.position = lightSource11.position + positionStepX1 * i;
      lightSource2.position = lightSource21.position + positionStepX2 * i;
      lightSource1.color = lightSource11.color + colorStepX1 * i;
      lightSource2.color = lightSource21.color + colorStepX2 * i;
      lightSource1.intensity = baseIntensityX1 + intensityStepX1 * i;
      lightSource2.intensity = baseIntensityX2 + intensityStepX2 * i;
      
      generateLineLightSource(lightSource1, lightSource2, segmentsY, lightSources);
    }
  }
  
  BMP outputFile;
  outputFile.SetBitDepth(24);
  outputFile.SetSize(pictureWidth, pictureHeight);
  
  if (antiAliasing) {
    pictureWidth *= 3;
    pictureHeight *= 3;
  }
  
  double viewPortXStep = viewPort.boundary.first / pictureWidth;
  double viewPortYStep = viewPort.boundary.second / pictureHeight;
  
  vector<vector<unsigned int> > imageBuffer(pictureWidth, vector<unsigned int>(pictureHeight, 0));
  
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < pictureWidth; ++i) {
    putc('.', stderr);
    for (int j = 0; j < pictureHeight; ++j) {
      SimpleOBJ::Vec3f currentPosition;
      currentPosition.x = viewPort.referencePoint.x + viewPort.baseVectors.first.x * i * viewPortXStep + viewPort.baseVectors.second.x * j * viewPortYStep;
      currentPosition.y = viewPort.referencePoint.y + viewPort.baseVectors.first.y * i * viewPortXStep + viewPort.baseVectors.second.y * j * viewPortYStep;
      currentPosition.z = viewPort.referencePoint.z + viewPort.baseVectors.first.z * i * viewPortXStep + viewPort.baseVectors.second.z * j * viewPortYStep;
      SimpleOBJ::Vec3f direction = currentPosition - viewPort.viewPoint;
      direction.Normalize();
      Ray sight(currentPosition, direction, false);
      
      SimpleOBJ::Vec3f currentIntensity = RayTracing(sight, planes, spheres, polyhedrons, lightSources, ambientLight, SimpleOBJ::Vec3f(1, 1, 1), minWeight);
      
      imageBuffer[i][j] = (clamp(0, 255, currentIntensity.r * 256) << 16) + (clamp(0, 255, currentIntensity.g * 256) << 8) + clamp(0, 255, currentIntensity.b * 256);
    }
  }
  
  if (antiAliasing) {
    outputFile(0, 0)->Red = (imageBuffer[0][0] & (unsigned int)16711680) >> 16;
    outputFile(0, 0)->Green = (imageBuffer[0][0] & (unsigned int)65280) >> 8;
    outputFile(0, 0)->Blue = imageBuffer[0][0] & (unsigned int)255;
    for (int j = 3; j < pictureHeight - 3; j += 3) {
      outputFile(0, j / 3)->Red = ((imageBuffer[0][j] & (unsigned int)16711680) +
        (imageBuffer[0][j - 1] & (unsigned int)16711680) +
        (imageBuffer[0][j + 1] & (unsigned int)16711680)) / 3 >> 16;
      outputFile(0, j / 3)->Green = ((imageBuffer[0][j] & (unsigned int)65280) +
        (imageBuffer[0][j - 1] & (unsigned int)65280) +
        (imageBuffer[0][j + 1] & (unsigned int)65280)) / 3 >> 8;
      outputFile(0, j / 3)->Blue = ((imageBuffer[0][j] & (unsigned int)255) +
        (imageBuffer[0][j - 1] & (unsigned int)255) +
        (imageBuffer[0][j + 1] & (unsigned int)255)) / 3;
    }
    outputFile(0, (pictureHeight - 1) / 3)->Red = (imageBuffer[0][pictureHeight - 1] & (unsigned int)16711680) >> 16;
    outputFile(0, (pictureHeight - 1) / 3)->Green = (imageBuffer[0][pictureHeight - 1] & (unsigned int)65280) >> 8;
    outputFile(0, (pictureHeight - 1) / 3)->Blue = imageBuffer[0][pictureHeight - 1] & (unsigned int)255;
    
    for (int i = 3; i < pictureWidth - 3; ++i) {
      outputFile(i / 3, 0)->Red = ((imageBuffer[i][0] & (unsigned int)16711680) +
        (imageBuffer[i - 1][0] & (unsigned int)16711680) +
        (imageBuffer[i + 1][0] & (unsigned int)16711680)) / 3 >> 16;
      outputFile(i / 3, 0)->Green = ((imageBuffer[i][0] & (unsigned int)65280) +
        (imageBuffer[i - 1][0] & (unsigned int)65280) +
        (imageBuffer[i + 1][0] & (unsigned int)65280)) / 3 >> 8;
      outputFile(i / 3, 0)->Blue = ((imageBuffer[i][0] & (unsigned int)255) +
        (imageBuffer[i - 1][0] & (unsigned int)255) +
        (imageBuffer[i + 1][0] & (unsigned int)255)) / 3;
        
      for (int j = 3; j < pictureHeight - 3; ++j) {
        outputFile(i / 3, j / 3)->Red = ((imageBuffer[i][j] & (unsigned int)16711680) +
          (imageBuffer[i - 1][j] & (unsigned int)16711680) +
          (imageBuffer[i + 1][j] & (unsigned int)16711680) +
          (imageBuffer[i][j - 1] & (unsigned int)16711680) +
          (imageBuffer[i][j + 1] & (unsigned int)16711680) +
          (imageBuffer[i - 1][j - 1] & (unsigned int)16711680) +
          (imageBuffer[i - 1][j + 1] & (unsigned int)16711680) +
          (imageBuffer[i + 1][j - 1] & (unsigned int)16711680) +
          (imageBuffer[i + 1][j + 1] & (unsigned int)16711680)) / 9 >> 16;
        outputFile(i / 3, j / 3)->Green = ((imageBuffer[i][j] & (unsigned int)65280) +
          (imageBuffer[i - 1][j] & (unsigned int)65280) +
          (imageBuffer[i + 1][j] & (unsigned int)65280) +
          (imageBuffer[i][j - 1] & (unsigned int)65280) +
          (imageBuffer[i][j + 1] & (unsigned int)65280) +
          (imageBuffer[i - 1][j - 1] & (unsigned int)65280) +
          (imageBuffer[i - 1][j + 1] & (unsigned int)65280) +
          (imageBuffer[i + 1][j - 1] & (unsigned int)65280) +
          (imageBuffer[i + 1][j + 1] & (unsigned int)65280)) / 9 >> 8;
        outputFile(i / 3, j / 3)->Blue = ((imageBuffer[i][j] & (unsigned int)255) +
          (imageBuffer[i - 1][j] & (unsigned int)255) +
          (imageBuffer[i + 1][j] & (unsigned int)255) +
          (imageBuffer[i][j - 1] & (unsigned int)255) +
          (imageBuffer[i][j + 1] & (unsigned int)255) +
          (imageBuffer[i - 1][j - 1] & (unsigned int)255) +
          (imageBuffer[i - 1][j + 1] & (unsigned int)255) +
          (imageBuffer[i + 1][j - 1] & (unsigned int)255) +
          (imageBuffer[i + 1][j + 1] & (unsigned int)255)) / 9;
      }
      
      outputFile(i / 3, (pictureHeight - 1) / 3)->Red = ((imageBuffer[i][pictureHeight - 1] & (unsigned int)16711680) +
        (imageBuffer[i - 1][pictureHeight - 1] & (unsigned int)16711680) +
        (imageBuffer[i + 1][pictureHeight - 1] & (unsigned int)16711680)) / 3 >> 16;
      outputFile(i / 3, (pictureHeight - 1) / 3)->Green = ((imageBuffer[i][pictureHeight - 1] & (unsigned int)65280) +
        (imageBuffer[i - 1][pictureHeight - 1] & (unsigned int)65280) +
        (imageBuffer[i + 1][pictureHeight - 1] & (unsigned int)65280)) / 3 >> 8;
      outputFile(i / 3, (pictureHeight - 1) / 3)->Blue = ((imageBuffer[i][pictureHeight - 1] & (unsigned int)255) +
        (imageBuffer[i - 1][pictureHeight - 1] & (unsigned int)255) +
        (imageBuffer[i + 1][pictureHeight - 1] & (unsigned int)255)) / 3;
    }
    
    outputFile((pictureWidth - 1) / 3, 0)->Red = (imageBuffer[pictureWidth - 1][0] & (unsigned int)16711680) >> 16;
    outputFile((pictureWidth - 1) / 3, 0)->Green = (imageBuffer[pictureWidth - 1][0] & (unsigned int)65280) >> 8;
    outputFile((pictureWidth - 1) / 3, 0)->Blue = imageBuffer[pictureWidth - 1][0] & (unsigned int)255;
    
    for (int j = 3; j < pictureHeight - 3; ++j) {
      outputFile((pictureWidth - 1) / 3, j / 3)->Red = ((imageBuffer[pictureWidth - 1][j] & (unsigned int)16711680) +
        (imageBuffer[pictureWidth - 1][j - 1] & (unsigned int)16711680) +
        (imageBuffer[pictureWidth - 1][j + 1] & (unsigned int)16711680)) / 3 >> 16;
      outputFile((pictureWidth - 1) / 3, j / 3)->Green = ((imageBuffer[pictureWidth - 1][j] & (unsigned int)65280) +
        (imageBuffer[pictureWidth - 1][j - 1] & (unsigned int)65280) +
        (imageBuffer[pictureWidth - 1][j + 1] & (unsigned int)65280)) / 3 >> 8;
      outputFile((pictureWidth - 1) / 3, j / 3)->Blue = ((imageBuffer[pictureWidth - 1][j] & (unsigned int)255) +
        (imageBuffer[pictureWidth - 1][j - 1] & (unsigned int)255) +
        (imageBuffer[pictureWidth - 1][j + 1] & (unsigned int)255)) / 3;
    }
    outputFile((pictureWidth - 1) / 3, (pictureHeight - 1) / 3)->Red = (imageBuffer[pictureWidth - 1][pictureHeight - 1] & (unsigned int)16711680) >> 16;
    outputFile((pictureWidth - 1) / 3, (pictureHeight - 1) / 3)->Green = (imageBuffer[pictureWidth - 1][pictureHeight - 1] & (unsigned int)65280) >> 8;
    outputFile((pictureWidth - 1) / 3, (pictureHeight - 1) / 3)->Blue = imageBuffer[pictureWidth - 1][pictureHeight - 1] & (unsigned int)255;
  }
  else {
    for (int i = 0; i < pictureWidth; ++i) {
      for (int j = 0; j < pictureHeight; ++j) {
        outputFile(i, j)->Red = (imageBuffer[i][j] & (unsigned int)16711680) >> 16;
        outputFile(i, j)->Green = (imageBuffer[i][j] & (unsigned int)65280) >> 8;
        outputFile(i, j)->Blue = imageBuffer[i][j] & (unsigned int)255;
      }
    }
  }

  outputFile.WriteToFile(argv[2]);
  return 0;
} 
