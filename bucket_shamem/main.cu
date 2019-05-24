#define TINYOBJLOADER_IMPLEMENTATION
#include "../tiny_obj_loader.h"

// Define _POSIX_C_SOURCE before including time.h so we can access the timers
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 201902L
#endif

#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>  // pow()
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm> // min(), max()
#include <set>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

using namespace std;

// ======================= Part1  common structures ===========================
struct point {
    float x; // x coordinate
    float y; // y coordinate
    float z; // z coordinate
    __host__ __device__ point() : x(0), y(0), z(0) {}
    __host__ __device__ point(float x, float y, float z)
	: x(x)
	, y(y)
	, z(z)
    {}
};

struct myvector {
    float x; // x component
    float y; // y component
    float z; // z component
    __host__ __device__ myvector() : x(0), y(0), z(0) {}
    __host__ __device__ myvector(float x, float y, float z)
	: x(x)
	, y(y)
	, z(z)
    {}
};

struct sphere {
    point c; // center
    float r; // radius
    float rr; // radius ^ 2 for speed-up reasons
    int id;
    __host__ __device__ sphere() {}
    __host__ __device__ sphere(point c, float r, int id)
	: c(c)
	, r(r)
	, id(id)
    { rr = pow(r, 2.0f); }
};


static float getPlaneD(const point& p, const myvector& v) {
    return -1 * (p.x * v.x + p.y * v.y + p.z * v.z);
}

struct triangle {
    point v1; // vertex 1
    point v2; // vertex 2
    point v3; // vertex 3
    float n_x; // x component of the normal vector
    float n_y; // y component of the normal vector
    float n_z; // z component of the normal vector
    float d; // a constant from the triangle’s plane equation: n_x * x + n_y * y + n_z * z + d = 0
    int id; // triangle's id
    triangle() {}
    triangle(point v1, point v2, point v3, float n_x, float n_y, float n_z, int id)
	: v1(v1)
	, v2(v2)
	, v3(v3)
	, n_x(n_x)
	, n_y(n_y)
	, n_z(n_z)
	, id(id)
    { d = getPlaneD(v1, myvector(n_x, n_y, n_z)); }
};

// ======================= Part2  parsing methods =============================
static void normalize(myvector & v) {
    float total = pow(v.x, 2.0f) + pow(v.y, 2.0f) + pow(v.z, 2.0f);
    v.x /=total;
    v.y /=total;
    v.z /=total;
}

static void parseTriangles(const tinyobj::attrib_t& attrib,
	                   const std::vector<tinyobj::shape_t> & shapes,
			   std::vector<triangle>* triangles) {
  size_t dimension = 3; // for triangle it is always 3

  // need to make sure have normal
  assert(attrib.normals.size() != 0);

  // For each shape
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t index_offset = 0;

    assert(shapes[i].mesh.num_face_vertices.size() ==
           shapes[i].mesh.material_ids.size());

    assert(shapes[i].mesh.num_face_vertices.size() ==
           shapes[i].mesh.smoothing_group_ids.size());

    int id = 0;
    // For each face
    for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
      size_t fnum = shapes[i].mesh.num_face_vertices[f];

      assert(fnum == 3); // make sure triangle
      vector<point> points;
      vector<myvector> normals;
      // For each vertex in the face
      for (size_t v = 0; v < fnum; v++) {
        tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
	size_t vid = idx.vertex_index * dimension;
	size_t nid = idx.normal_index * dimension;
	points.push_back(point(attrib.vertices[vid], attrib.vertices[vid+1], attrib.vertices[vid+2]));
        // get normals, each of the three points should have same normal, cuz triangle
        normals.push_back(myvector(attrib.normals[nid], attrib.normals[nid+1], attrib.normals[nid+2]));
      }

      assert(normals[0].x == normals[1].x && normals[0].y == normals[1].y && normals[0].z == normals[1].z);
      normalize(normals[0]);

      index_offset += fnum;

      triangle t = triangle(points[0], points[1], points[2], normals[0].x, normals[0].y, normals[0].z, id++);
      triangles->push_back(t);
    }
  }
}

static bool parseSpheres(const char* filename,
	                 const float radius,
	                 vector<sphere>* spheres,
			 float& min_x, float& max_x,
			 float& min_y, float& max_y,
			 float& min_z, float& max_z) {
    ifstream infile(filename);
    if (!infile.is_open() || !infile.good()) {
        return false;
    }
    std::string line;
    if (!getline(infile, line)) {
	return false;
    } else {
	std::cout << "First line in file " << line << std::endl;
    }
    int id = 0;
    bool first = true;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
	vector<float> result;

	while( iss.good() ) {
	    string substr;
            getline( iss, substr, ',' );
            result.push_back( atof(substr.c_str()) );
	}
	if (result.size() != 3) {
	    return false;
	}
	point c = point(result[0], result[1], result[2]);
	if (first) {
	    min_x = max_x = result[0];
	    min_y = max_y = result[1];
	    min_z = max_z = result[2];
	    first = false;
	}
	if (result[0] < min_x)
	    min_x = result[0];
	else if (result[0] > max_x)
	    max_x = result[0];

	if (result[1] < min_y)
	    min_y = result[1];
	else if (result[1] > max_y)
	    max_y = result[1];

	if (result[2] < min_z)
	    min_z = result[2];
	else if (result[2] > max_z)
	    max_z = result[2];
	sphere s = sphere(c, radius, id++);
	spheres->push_back(s);
    }
    // min value - radius, max + radius
    min_x -= radius;
    min_y -= radius;
    min_z -= radius;
    max_x += radius;
    max_y += radius;
    max_z += radius;
    return true;
}

// =======================  Part3 collison detection ==========================
__host__ __device__ float calPointsDistSquare(point p1, point p2) {
    return pow((p1.x - p2.x), 2.0f) + pow((p1.y - p2.y), 2.0f) + pow((p1.z - p2.z), 2.0f);
}

__host__ __device__ float calInnerProduct(myvector v1, myvector v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ float calInnerProduct(myvector v1, point p1) {
    return v1.x * p1.x + v1.y * p1.y + v1.z * p1.z;
}

__host__ __device__ float detectCollisionSphereLine(point p1, point p2, sphere s) {
    myvector N;
    N.x = p2.x - p1.x;
    N.y = p2.y - p1.y;
    N.z = p2.z - p1.z;

    float t = (calInnerProduct(N, s.c) - calInnerProduct(N, p1)) / calInnerProduct(N, N);

    point Q;
    Q.x = p1.x + t * N.x;
    Q.y = p1.y + t * N.y;
    Q.z = p1.z + t * N.z;

    float dd = calPointsDistSquare(Q, s.c);

    if (dd > s.rr)
        return false;
    else if (dd < s.rr) {
        if (0 <= t && t <= 1) 
            return true;
        else
            return false;
    }
    else {
        if (0 <= t && t <= 1) 
            return true;
        else
            return false;
    }        
}

// Check whether the triangle’s plane intersects with the sphere.
__host__ __device__ bool isCollisionStep1(sphere s, triangle t, float &dist) {
    //printf("isCollisionStep1...\n");

    dist = t.n_x * s.c.x + t.n_y * s.c.y + t.n_z * s.c.z + t.d;
    if (fabs(dist) <= s.r)
        return true;
    return false;
}

// Check whether any of the triangle vertices is inside the sphere.
__host__ __device__ bool isCollisionStep2(sphere s, triangle t) {
    //printf("isCollisionStep2...\n");

    if (calPointsDistSquare(s.c, t.v1) <= s.rr)
        return true;
    if (calPointsDistSquare(s.c, t.v2) <= s.rr)
        return true;
    if (calPointsDistSquare(s.c, t.v3) <= s.rr)
        return true;        
    return false;
}

// Check if the projected center Cproj lies inside the triangle.
__host__ __device__ bool isCollisionStep3(sphere s, triangle t, float dist) {
    //printf("isCollisionStep3...\n");

    point c_proj;
    c_proj.x = s.c.x - dist * t.n_x;
    c_proj.y = s.c.y - dist * t.n_y;
    c_proj.z = s.c.z - dist * t.n_z;
    
    // Reference: http://mathworld.wolfram.com/TriangleInterior.html
    // Ref: https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle?rq=1
    // Ref: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    myvector v0, v1, v2;
    v0.x = t.v1.x - t.v3.x;
    v0.y = t.v1.y - t.v3.y;
    v0.z = t.v1.z - t.v3.z;

    v1.x = t.v2.x - t.v3.x;
    v1.y = t.v2.y - t.v3.y;
    v1.z = t.v2.z - t.v3.z;

    v2.x = c_proj.x - t.v3.x;
    v2.y = c_proj.y - t.v3.y;
    v2.z = c_proj.z - t.v3.z;

    float d00 = calInnerProduct(v0, v0);
    float d01 = calInnerProduct(v0, v1);
    float d11 = calInnerProduct(v1, v1);
    float d20 = calInnerProduct(v2, v0);
    float d21 = calInnerProduct(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    float alpha = (d11 * d20 - d01 * d21) / denom;
    float beta = (d00 * d21 - d01 * d20) / denom;
    //float gamma = 1.0f - alpha - beta;
    /*
    printf("(alpha, beta) = (%f, %f)\n", alpha, beta);
    printf("v0 = (%f, %f, %f)\n", v0.x, v0.y, v0.z);
    printf("v1 = (%f, %f, %f)\n", v1.x, v1.y, v1.z);
    printf("v2 = (%f, %f, %f)\n", v2.x, v2.y, v2.z);
    printf("t.v3 = (%f, %f, %f)\n", t.v3.x,  t.v3.y,  t.v3.z);
    */

    if ((alpha + beta <= 1.0) && alpha >= 0 && beta >= 0)
        return true;
    return false;
}

// Check whether the sphere intersects with a triangle edge.
__host__ __device__ bool isCollisionStep4(sphere s, triangle t) {
    //printf("isCollisionStep4...\n");

    if (detectCollisionSphereLine(t.v1, t.v2, s))
        return true;
    if (detectCollisionSphereLine(t.v2, t.v3, s))
        return true;
    if (detectCollisionSphereLine(t.v3, t.v1, s))
        return true;
    return false;
}

__host__ __device__ bool detectCollision(sphere s, triangle t) {
    float dist; // the triangle plane’s distance from the sphere’s center

    //printf("collision with id %d step 1 res %d\n", id, isCollisionStep1(s, t, dist));
    // Condition 1
    bool step1 = false;
    dist = t.n_x * s.c.x + t.n_y * s.c.y + t.n_z * s.c.z + t.d;
    if (fabs(dist) <= s.r)
        step1 = true;
    if (!step1)
	return false;

    //printf("collision with id %d step 2 res %d\n", id, isCollisionStep2(s, t));
    // Condition 2
    bool cond1 = calPointsDistSquare(s.c, t.v1) <= s.rr;
    bool cond2 = calPointsDistSquare(s.c, t.v2) <= s.rr;
    bool cond3 = calPointsDistSquare(s.c, t.v3) <= s.rr;
    if (cond1 || cond2 || cond3)
	return true;

    //printf("collision with id %d step 3 res %d\n", id, isCollisionStep3(s, t, dist));
    // Condition 3
    point c_proj;
    c_proj.x = s.c.x - dist * t.n_x;
    c_proj.y = s.c.y - dist * t.n_y;
    c_proj.z = s.c.z - dist * t.n_z;
    
    // Reference: http://mathworld.wolfram.com/TriangleInterior.html
    // Ref: https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle?rq=1
    // Ref: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    myvector v0, v1, v2;
    v0.x = t.v1.x - t.v3.x;
    v0.y = t.v1.y - t.v3.y;
    v0.z = t.v1.z - t.v3.z;

    v1.x = t.v2.x - t.v3.x;
    v1.y = t.v2.y - t.v3.y;
    v1.z = t.v2.z - t.v3.z;

    v2.x = c_proj.x - t.v3.x;
    v2.y = c_proj.y - t.v3.y;
    v2.z = c_proj.z - t.v3.z;

    float d00 = calInnerProduct(v0, v0);
    float d01 = calInnerProduct(v0, v1);
    float d11 = calInnerProduct(v1, v1);
    float d20 = calInnerProduct(v2, v0);
    float d21 = calInnerProduct(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    float alpha = (d11 * d20 - d01 * d21) / denom;
    float beta = (d00 * d21 - d01 * d20) / denom;
    //float gamma = 1.0f - alpha - beta;
    if ((alpha + beta <= 1.0) && alpha >= 0 && beta >= 0)
        return true;

    //printf("collision with id %d step 4 res %d\n", id, isCollisionStep4(s, t));
    // Condition 4
    cond1 = detectCollisionSphereLine(t.v1, t.v2, s);
    cond2 = detectCollisionSphereLine(t.v2, t.v3, s);
    cond3 = detectCollisionSphereLine(t.v3, t.v1, s);
    if (cond1 || cond2 || cond3)
	return true;

    return false;
}

// ========================= Part4 helper methods =============================
static void printTriangles(std::vector<triangle>* ts) {
    for (size_t i = 0; i < ts->size(); ++i) {
	triangle t = ts->at(i);
	printf("Traingle[%d]\n", i);
	printf("  id=%d\n", t.id);
	printf("  point1=(%f, %f, %f)\n", t.v1.x, t.v1.y, t.v1.z);
	printf("  point2=(%f, %f, %f)\n", t.v2.x, t.v2.y, t.v2.z);
	printf("  point3=(%f, %f, %f)\n", t.v3.x, t.v3.y, t.v3.z);
	printf("  normal=(%f, %f, %f)\n", t.n_x, t.n_y, t.n_z);
	printf("  dist=%f\n", t.d);
    }
}

static void printSpheres(std::vector<sphere>* ss) {
    for (size_t i = 0; i < ss->size(); ++i) {
	sphere s = ss->at(i);
	printf("Sphere[%d]\n", i);
	printf("  id=%d\n", s.id);
	printf("  center=(%f, %f, %f)\n", s.c.x, s.c.y, s.c.z);
	printf("  radius=%f\n", s.r);
	printf("  radius^2=%f\n", s.rr);
    }
}

static void myPrintInfo(const tinyobj::attrib_t& attrib,
	                const std::vector<tinyobj::shape_t> & shapes,
                        const std::vector<tinyobj::material_t>& materials) {
  size_t dimension = 3; // for triangle it is always 3
  std::cout << "# of vertices  : " << (attrib.vertices.size() / dimension) << std::endl;
  std::cout << "# of normals  : " << (attrib.normals.size() / dimension) << std::endl;
  for (size_t i = 0; i < attrib.vertices.size(); i += dimension) {
      printf("vertices[%d] = (%f, %f, %f)\n", i / dimension, attrib.vertices[i], attrib.vertices[i+1], attrib.vertices[i+2]);
  }
  std::cout << "# of shapes    : " << shapes.size() << std::endl;
  // For each shape
  for (size_t i = 0; i < shapes.size(); i++) {
    printf("shape[%ld].name = %s\n", static_cast<long>(i),
           shapes[i].name.c_str());
    printf("Size of shape[%ld].mesh.indices: %lu\n", static_cast<long>(i),
           static_cast<unsigned long>(shapes[i].mesh.indices.size()));
    printf("Size of shape[%ld].lines.indices: %lu\n", static_cast<long>(i),
           static_cast<unsigned long>(shapes[i].lines.indices.size()));
    printf("Size of shape[%ld].points.indices: %lu\n", static_cast<long>(i),
           static_cast<unsigned long>(shapes[i].points.indices.size()));

    size_t index_offset = 0;

    assert(shapes[i].mesh.num_face_vertices.size() ==
           shapes[i].mesh.material_ids.size());

    assert(shapes[i].mesh.num_face_vertices.size() ==
           shapes[i].mesh.smoothing_group_ids.size());

    printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
           static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

    // For each face
    for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
      size_t fnum = shapes[i].mesh.num_face_vertices[f];

      printf("  face[%ld].fnum = %ld\n", static_cast<long>(f),
             static_cast<unsigned long>(fnum));

      // For each vertex in the face
      for (size_t v = 0; v < fnum; v++) {
        tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
	size_t vid = idx.vertex_index * dimension;
	size_t nid = idx.normal_index * dimension;
        printf("    face[%ld].v[%ld].idx = %d/%d/%d, coord=(%f, %f, %f), normal=(%f, %f, %f)\n", 
	       static_cast<long>(f), static_cast<long>(v), idx.vertex_index, idx.normal_index,
               idx.texcoord_index, attrib.vertices[vid], attrib.vertices[vid+1], attrib.vertices[vid+2],
               attrib.normals[nid], attrib.normals[nid+1], attrib.normals[nid+2]);
      }

      index_offset += fnum;
    }
  }
}

// ========================= Part5 get bins for sphere and triangle ================================
// determine the number of bins that v_min ~ v_max overlaps
void get_bins_per_axis(float v_min, float v_max, float min, float len, float bin_width,
	               int& bin_num_s, int& bin_num_e) {
    bin_num_s = static_cast<int>(ceil((v_min - min) / bin_width)) - 1;
    bin_num_e = static_cast<int>(floor((v_max - min) / bin_width));
    if (bin_num_s < 0)
	bin_num_s = 0;
    if (bin_num_e >= len)
	bin_num_e = len - 1;
}
void partition_triangle(vector<triangle>* triangles,
	                vector<vector<triangle> >* binToTriangles,
			int& totalNum,
                        float min_x, int x_len,
                        float min_y, int y_len,
                        float min_z, int z_len,
                        float bin_width) {
    int sum = 0;
    for (int i = 0; i < triangles->size(); ++i) {
	point p1 = triangles->at(i).v1;
	point p2 = triangles->at(i).v2;
	point p3 = triangles->at(i).v3;
        float t_x_min, t_x_max;
        float t_y_min, t_y_max;
        float t_z_min, t_z_max;
	t_x_min = min( min(p1.x, p2.x), p3.x);
	t_y_min = min( min(p1.y, p2.y), p3.y);
	t_z_min = min( min(p1.z, p2.z), p3.z);
	
	t_x_max = max( max(p1.x, p2.x), p3.x);
	t_y_max = max( max(p1.y, p2.y), p3.y);
	t_z_max = max( max(p1.z, p2.z), p3.z);
	int bin_x_s, bin_x_e;
	int bin_y_s, bin_y_e;
	int bin_z_s, bin_z_e;
	get_bins_per_axis(t_x_min, t_x_max, min_x, x_len, bin_width, bin_x_s, bin_x_e);
	get_bins_per_axis(t_y_min, t_y_max, min_y, y_len, bin_width, bin_y_s, bin_y_e);
	get_bins_per_axis(t_z_min, t_z_max, min_z, z_len, bin_width, bin_z_s, bin_z_e);
	for (int z = bin_z_s; z <= bin_z_e; ++z) {
	    for (int y = bin_y_s; y <= bin_y_e; ++y) {
	        for (int x = bin_x_s; x <= bin_x_e; ++x) {
		    int bin_index = x + y * x_len + z * x_len * y_len;
		    //printf("bin_index is %d\n", bin_index);
		    sum += 1;
		    binToTriangles->at(bin_index).push_back(triangles->at(i));
		}
	    }
	}
    }
    totalNum = sum;
}

void partition_sphere(vector<sphere>* spheres,
	              vector<vector<sphere> >* binToSpheres,
		      int& totalNum,
                      float min_x, int x_len,
                      float min_y, int y_len,
                      float min_z, int z_len,
                      float bin_width) {
    int sum = 0;
    for (int i = 0; i < spheres->size(); ++i) {
	float s_x = spheres->at(i).c.x;
	float s_y = spheres->at(i).c.y;
	float s_z = spheres->at(i).c.z;
	float radius = spheres->at(i).r;
	int bin_x_s, bin_x_e;
	int bin_y_s, bin_y_e;
	int bin_z_s, bin_z_e;
	get_bins_per_axis(s_x - radius, s_x + radius, min_x, x_len, bin_width, bin_x_s, bin_x_e);
	get_bins_per_axis(s_y - radius, s_y + radius, min_y, y_len, bin_width, bin_y_s, bin_y_e);
	get_bins_per_axis(s_z - radius, s_z + radius, min_z, z_len, bin_width, bin_z_s, bin_z_e);
	/*printf("Sphere[%d]\n", i);
        printf("  x_s, x_e is %d, %d\n", bin_x_s, bin_x_e);
        printf("  y_s, y_e is %d, %d\n", bin_y_s, bin_y_e);
        printf("  z_s, z_e is %d, %d\n", bin_z_s, bin_z_e);*/
	
	for (int z = bin_z_s; z <= bin_z_e; ++z) {
	    for (int y = bin_y_s; y <= bin_y_e; ++y) {
	        for (int x = bin_x_s; x <= bin_x_e; ++x) {
		    int bin_index = x + y * x_len + z * x_len * y_len;
		    //printf("bin_index is %d\n", bin_index);
		    binToSpheres->at(bin_index).push_back(spheres->at(i));
		    sum += 1;
		}
	    }
	}
    }
    totalNum = sum;
}

void serialize_bin_sphere(vector<vector<sphere> >* binToSpheres,
	                  int* sphereOffset,
			  sphere* sphereH,
			  int& max_in_a_bin) {
    int sum = 0;
    int i = 0;
    for (; i < binToSpheres->size(); ++i) {
	sphereOffset[i] = sum;
	for (int j = 0; j < binToSpheres->at(i).size(); ++j) {
	    sphereH[sum + j] = binToSpheres->at(i)[j];
	}
	sum += binToSpheres->at(i).size();
	max_in_a_bin = max(max_in_a_bin, (int)binToSpheres->at(i).size());
    }
    sphereOffset[i] = sum;
}

void serialize_bin_triangle(vector<vector<triangle> >* binToTriangles,
	                    int* triangleOffset,
			    triangle* triangleH,
			    int& max_in_a_bin) {
    int sum = 0;
    int i = 0;
    for (; i < binToTriangles->size(); ++i) {
	triangleOffset[i] = sum;
	for (int j = 0; j < binToTriangles->at(i).size(); ++j) {
	    triangleH[sum + j] = binToTriangles->at(i)[j];
	}
	sum += binToTriangles->at(i).size();
	max_in_a_bin = max(max_in_a_bin, (int)binToTriangles->at(i).size());
    }
    triangleOffset[i] = sum;
}

__global__ void getCollisionNumCUDA(
    sphere* sphereD, 
    triangle* triangleD,
    int* sphereOffset,
    int* triangleOffset,
    int totalBins,
    int* binToCollisionNum) {

    // for every bin there is a thread
    int bin_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_index < totalBins) {
	int sum = 0;
	int totalSphere = sphereOffset[bin_index + 1] - sphereOffset[bin_index];
	int totalTriangle = triangleOffset[bin_index + 1] - triangleOffset[bin_index];
	int t_offset = triangleOffset[bin_index];
	int s_offset = sphereOffset[bin_index];
	for (int j = 0; j < totalTriangle; ++j) {
	    int triangleID = t_offset + j;
	    triangle t = triangleD[triangleID];
	    for (int i = 0; i < totalSphere; ++i) {
	        int sphereID = s_offset + i;
	        sphere s = sphereD[sphereID];
		if (detectCollision(s, t))
		    sum += 1;
	    }
	}
	binToCollisionNum[bin_index] = sum;
    }
}

// call detect Collision for every bucket
__global__ void detectCollisionCUDA(
    sphere* sphereD, 
    triangle* triangleD,
    int* sphereOffset,
    int* triangleOffset,
    int totalBins,
    int* outputOffsets,
    int* sphereCllsnID,
    int* triangleCllsnID) {

    // first b+1 contains num of spheres
    // then b+1 contains num of triangles
    extern __shared__ int buffer[];
    int bin_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_index < totalBins) {
	// load last entry
	if (threadIdx.x == 0 && blockIdx.x * blockDim.x + blockDim.x < totalBins) {
	    buffer[blockDim.x] = sphereOffset[blockIdx.x * blockDim.x + blockDim.x];
	    buffer[blockDim.x + blockDim.x + 1] = triangleOffset[blockIdx.x * blockDim.x + blockDim.x];
	}
	buffer[threadIdx.x] = sphereOffset[bin_index];
	buffer[threadIdx.x + blockDim.x + 1] = triangleOffset[bin_index];
    }
    __syncthreads();

    // for every bin there is a thread
    if (bin_index < totalBins) {
	int sum = 0;
	//int totalSphere = sphereOffset[bin_index + 1] - sphereOffset[bin_index];
	//int totalTriangle = triangleOffset[bin_index + 1] - triangleOffset[bin_index];
	int totalSphere = buffer[threadIdx.x + 1] - buffer[threadIdx.x];
	int totalTriangle = buffer[threadIdx.x + blockDim.x + 1 + 1] - 
	                    buffer[threadIdx.x + blockDim.x + 1];
	//int s_offset = sphereOffset[bin_index];
	//int t_offset = triangleOffset[bin_index];
	int s_offset = buffer[threadIdx.x];
	int t_offset = buffer[threadIdx.x + blockDim.x + 1];
	int outOffset = 0;
	if (bin_index > 0)
	    outOffset = outputOffsets[bin_index - 1];
	for (int j = 0; j < totalTriangle; ++j) {
	    int triangleID = t_offset + j;
	    triangle t = triangleD[triangleID];
	    for (int i = 0; i < totalSphere; ++i) {
	        int sphereID = s_offset + i;
	        sphere s = sphereD[sphereID];
		if (detectCollision(s, t)) {
		    int output_index = outOffset + sum;
		    sphereCllsnID[output_index] = s.id;
		    triangleCllsnID[output_index] = t.id;
		    sum += 1;
		}
	    }
	}
    }
}

// ========================= Part6 main method ================================
int main(int argc, char *argv[])
{
    if (argc != 5) {
        printf("[Error] The number of arguments is wrong and should be 5.\n");
        printf("Usage: ./collide meshfile spherefile radius outfile\n");
        return 0;
    }

    cout << "Loading objects..." << endl;
    const char* filename = argv[1];
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string warn;
    string err;
    const char* basepath = NULL;
    // whether to transfom all thing to traingles
    bool triangulate = true;
    vector<triangle> triangles;
    float radius = atof(argv[3]);
    const char* out_filename = argv[4];

    const char* sphere_filename = argv[2];
    vector<sphere> spheres;
    float min_x, max_x, min_y, max_y, min_z, max_z; 
    
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, basepath, triangulate);

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "ERR: " << err << std::endl;
    }

    if (!ret) {
        printf("Failed to load/parse .obj.\n");
        return 1;
    }

    //myPrintInfo(attrib, shapes, materials);

    cout << "Parsing to vector of triangles..." << endl;
    parseTriangles(attrib, shapes, &triangles);
    vector<tinyobj::shape_t>().swap(shapes);
    vector<tinyobj::material_t>().swap(materials);

    cout << "Num of triangles is " << triangles.size() << endl;
    //printTriangles(triangles);
    
    cout << "Parsing to vector of shperes..." << endl;
    if (!parseSpheres(sphere_filename, radius, &spheres, min_x, max_x, min_y, max_y, min_z, max_z)) {
	cout << "Failed to load/parse sphere file: " << sphere_filename << endl;
	return 1;
    }

    cout << "Num of spheres is " << spheres.size() << endl;
    //printSpheres(spheres);
    int sphereNum = spheres.size();
    int triangleNum = triangles.size();

    float elapsedTimeIn;
    cudaEvent_t startEventIn, stopEventIn;
    cudaEventCreate(&startEventIn); 
    cudaEventCreate(&stopEventIn);

    // deal with the bin part
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        //printf("vertices[%d] = (%f, %f, %f)\n", i / 3, attrib.vertices[i], attrib.vertices[i+1], attrib.vertices[i+2]);
	min_x = attrib.vertices[i] < min_x ? attrib.vertices[i] : min_x;
	max_x = attrib.vertices[i] > max_x ? attrib.vertices[i] : max_x;
	min_y = attrib.vertices[i+1] < min_y ? attrib.vertices[i+1] : min_y;
	max_y = attrib.vertices[i+1] > max_y ? attrib.vertices[i+1] : max_y;
	min_z = attrib.vertices[i+2] < min_z ? attrib.vertices[i+2] : min_z;
	max_z = attrib.vertices[i+2] > max_z ? attrib.vertices[i+2] : max_z;
    }

    // Ref: A scalable parallel method for large collision detect problems
    // note that for sequential case BIN_WIDTH should be small 
    // for parallel BIN_WIDTH should be larger so that each bin 
    // contains ~ 1000 spheres * 1000 triangles
    //const float BIN_WIDTH = .1 * ceil(max_x - min_x);
    const float BIN_WIDTH = 4 * radius;
    int x_len = static_cast<int>(ceil((max_x - min_x) / BIN_WIDTH));
    int y_len = static_cast<int>(ceil((max_y - min_y) / BIN_WIDTH));
    int z_len = static_cast<int>(ceil((max_z - min_z) / BIN_WIDTH));
    int totalBins = x_len * y_len * z_len;
    printf("x_len, y_len, z_len, totalBins = (%d, %d, %d, %d)\n", x_len, y_len, z_len, totalBins);

    vector<vector<sphere> > binToSpheres;
    vector<vector<triangle> > binToTriangles;
    for (int i = 0; i < totalBins; ++i) {
	vector<sphere> ss;
	vector<triangle> ts;
	binToSpheres.push_back(ss);
	binToTriangles.push_back(ts);
    }

    // this number should be >= sphereNum because some spheres across the boundary
    // would be count twice or serveral times
    int sphereNum_bin = 0; // the number of total spheres in buckets
    partition_sphere(&spheres, &binToSpheres, sphereNum_bin, min_x, x_len, min_y, y_len, min_z, z_len, BIN_WIDTH);
    vector<sphere>().swap(spheres); // clear spheres vector

    int triangleNum_bin = 0;
    partition_triangle(&triangles, &binToTriangles, triangleNum_bin, min_x, x_len, min_y, y_len, min_z, z_len, BIN_WIDTH);
    vector<triangle>().swap(triangles); // clear triangles vector

    set<pair<int, int> > flags; // have flag to make sure each pair only write once

    ofstream outFile(out_filename);
    if (!outFile) {
        cout << "Unable to open " << out_filename << endl;
        exit(1); // terminate with error
    }



    int sphereSize_bin = sphereNum_bin * sizeof(sphere);
    int triangleSize_bin = triangleNum_bin * sizeof(triangle);

    sphere* sphereH = (sphere*) malloc(sphereSize_bin);
    triangle* triangleH = (triangle*) malloc(triangleSize_bin);
    sphere* sphereD;          cudaMalloc((void **)&sphereD, sphereSize_bin);
    triangle* triangleD;      cudaMalloc((void **)&triangleD, triangleSize_bin);

    int offset_size = sizeof(int) * (totalBins + 1); // cumulative sum
    int* sphereOffsetH = (int *) malloc(offset_size);
    int* triangleOffsetH = (int *) malloc(offset_size);
    int* sphereOffsetD;  cudaMalloc((void **)&sphereOffsetD, offset_size);
    int* triangleOffsetD;  cudaMalloc((void **)&triangleOffsetD, offset_size);

    int max_sphere_in_a_bin = 0;
    int max_triangle_in_a_bin = 0;

    cout << "Begin serializing" << endl;

    serialize_bin_sphere(&binToSpheres, sphereOffsetH, sphereH, max_sphere_in_a_bin);
    serialize_bin_triangle(&binToTriangles, triangleOffsetH, triangleH, max_triangle_in_a_bin);

    cout << "End serializing" << endl;
    vector<vector<sphere> >().swap(binToSpheres); // clear binToSpheres vector
    vector<vector<triangle> >().swap(binToTriangles); // clear binToSpheres vector

    int* binToCollisionNum;   cudaMalloc((void **)&binToCollisionNum, offset_size);

    // For inclusive timing
    cudaEventRecord(startEventIn, 0);

    cudaMemcpy(sphereD, sphereH, sphereSize_bin, cudaMemcpyHostToDevice);
    cudaMemcpy(triangleD, triangleH, triangleSize_bin, cudaMemcpyHostToDevice);
    cudaMemcpy(sphereOffsetD, sphereOffsetH, offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(triangleOffsetD, triangleOffsetH, offset_size, cudaMemcpyHostToDevice);
    int nthreads = 512;
    // Stage 7, write num of collision to binToCollisionNum
    getCollisionNumCUDA<<<(totalBins + nthreads - 1) / nthreads, nthreads>>>(
	    sphereD,
	    triangleD,
	    sphereOffsetD,
	    triangleOffsetD,
	    totalBins,
	    binToCollisionNum
	);

    // use thrust to do inclusive scan
    // stage 8
    thrust::device_ptr<int> binToCollisionNum_thrust(binToCollisionNum);
    thrust::inclusive_scan(binToCollisionNum_thrust, binToCollisionNum_thrust + totalBins, binToCollisionNum_thrust);

    // stage 9
    int totalCollision = binToCollisionNum_thrust[totalBins - 1];
    printf("totalCollision is %d\n", totalCollision);
    //assert(totalCollision == 39709);

    int* sphereCllsnID_H = (int *) malloc (totalCollision * sizeof(int));
    int* triangleCllsnID_H = (int *) malloc (totalCollision * sizeof(int));
    int* sphereCllsnID_D;  cudaMalloc((void **) &sphereCllsnID_D, totalCollision * sizeof(int));
    int* triangleCllsnID_D;  cudaMalloc((void **) &triangleCllsnID_D, totalCollision * sizeof(int));
    detectCollisionCUDA<<<(totalBins + nthreads - 1) / nthreads, nthreads, sizeof(int) * 2 * (nthreads+1) >>>(
	    sphereD,
	    triangleD,
	    sphereOffsetD,
	    triangleOffsetD,
	    totalBins,
	    thrust::raw_pointer_cast(binToCollisionNum_thrust),
	    sphereCllsnID_D,
	    triangleCllsnID_D);

    cudaMemcpy(sphereCllsnID_H, sphereCllsnID_D, totalCollision*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(triangleCllsnID_H, triangleCllsnID_D, totalCollision*sizeof(int), cudaMemcpyDeviceToHost);

    free(sphereH);
    free(triangleH);
    free(sphereOffsetH);
    free(triangleOffsetH);

    vector<int> sCllsnID;
    vector<int> tCllsnID;
    // Get rid of duplicate entries
    for (long long i = 0; i < totalCollision; i++) {
	pair<int, int> index_pair = make_pair(sphereCllsnID_H[i], triangleCllsnID_H[i]);
	if (flags.find(index_pair) == flags.end()) {
            flags.insert(index_pair);
	    sCllsnID.push_back(sphereCllsnID_H[i]);
	    tCllsnID.push_back(triangleCllsnID_H[i]);
	}
    }

    // For inclusive timing
    cudaEventRecord(stopEventIn, 0); 
    cudaEventSynchronize(stopEventIn);     
    cudaEventElapsedTime(&elapsedTimeIn, startEventIn, stopEventIn);

    cudaEventDestroy(startEventIn); 
    cudaEventDestroy(stopEventIn);

    cudaFree(sphereD);
    cudaFree(triangleD);
    cudaFree(sphereOffsetD);
    cudaFree(triangleOffsetD);
    cudaFree(binToCollisionNum);
    cudaFree(sphereCllsnID_D);
    cudaFree(triangleCllsnID_D);

    free(sphereCllsnID_H);
    free(triangleCllsnID_H);

    cout << "start writing outfile " << endl;
    outFile << elapsedTimeIn << endl;
    

    for (long long i = 0; i < sCllsnID.size(); i++) {
        outFile << sCllsnID[i] << "," << tCllsnID[i] << endl;
    }


    return 0;
}
