#define TINYOBJLOADER_IMPLEMENTATION
#include "../tiny_obj_loader.h"

// Define _POSIX_C_SOURCE before including time.h so we can access the timers
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 201902L
#endif

#include <time.h>
#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h> 
#include <string>
#include <sstream>
#include <fstream>

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
  printf("# of shapes    : %d\n", shapes.size());

  // need to make sure have normal
  assert(attrib.normals.size() != 0);

  // For each shape
  for (size_t i = 0; i < shapes.size(); i++) {
    printf("shape[%ld].name = %s\n", static_cast<long>(i),
           shapes[i].name.c_str());
    printf("Size of shape[%ld].mesh.indices: %lu\n", static_cast<long>(i),
           static_cast<unsigned long>(shapes[i].mesh.indices.size()));

    size_t index_offset = 0;

    assert(shapes[i].mesh.num_face_vertices.size() ==
           shapes[i].mesh.material_ids.size());

    assert(shapes[i].mesh.num_face_vertices.size() ==
           shapes[i].mesh.smoothing_group_ids.size());

    printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
           static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

    int id = 0;
    // For each face
    for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
      size_t fnum = shapes[i].mesh.num_face_vertices[f];

      assert(fnum == 3); // make sure triangle
      vector<point> points;
      vector<myvector> normals;
      float d = 0.0;
      // For each vertex in the face
      for (size_t v = 0; v < fnum; v++) {
        tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
	size_t vid = idx.vertex_index * dimension;
	size_t nid = idx.normal_index * dimension;
        printf("    face[%ld].v[%ld].idx = %d/%d/%d, coord=(%f, %f, %f)\n", static_cast<long>(f),
               static_cast<long>(v), idx.vertex_index, idx.normal_index,
               idx.texcoord_index, attrib.vertices[vid], attrib.vertices[vid+1], attrib.vertices[vid+2]);
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
	                 std::vector<sphere>* spheres) {
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
	sphere s = sphere(c, radius, id++);
	spheres->push_back(s);
    }
    return true;
}

// =======================  Part3 collison detection ==========================
__host__ __device__ float calPointsDistSquare(point p1, point p2) {
    return pow((p1.x - p2.x), 2.0f) + pow((p1.y - p2.y), 2.0f) + pow((p1.z - p2.z), 2.0f);
}

__host__ __device__ float calDeterminant(point p1, myvector v1) {
    return p1.x * v1.y - p1.y * v1.x;
}

__host__ __device__ float calDeterminant(myvector v1, myvector v2) {
    return v1.x * v2.y - v1.y * v2.x;
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
    if (!isCollisionStep1(s, t, dist))
        return false;

    //printf("collision with id %d step 2 res %d\n", id, isCollisionStep2(s, t));
    // Condition 2
    if (isCollisionStep2(s, t))
        return true;

    //printf("collision with id %d step 3 res %d\n", id, isCollisionStep3(s, t, dist));
    // Condition 3
    if (isCollisionStep3(s, t, dist))
        return true;

    //printf("collision with id %d step 4 res %d\n", id, isCollisionStep4(s, t));
    // Condition 4
    if (isCollisionStep4(s, t))
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

// ========================= Part5 main method ================================
//  Kernel – thread specification
__global__ void detectCollisionCUDA(
    sphere* sphereD, 
    triangle* triangleD, 
    int sphereSegId,
    int triangleSegId,
    int sphereSegSize, 
    int triangleSegSize,
    int segSize, 
    int sphereNum, 
    int* sphereCllsnIdD, 
    int* triangleCllsnIdD) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < segSize) {    
        int sphereIdOft = i % sphereNum;
        int triangleIdOft = i / sphereNum;
        int sphereId = sphereSegId * sphereSegSize + sphereIdOft;
        int triangleId = triangleSegId * triangleSegSize + triangleIdOft;
        if (detectCollision(sphereD[sphereId], triangleD[triangleId])) {
            sphereCllsnIdD[i] = sphereId;
            triangleCllsnIdD[i] = triangleId;
        }
        //printf("Collision i%i (sphere: %d, triangle: %d) = %d\n", i, sphereID, triangleID, collisionD[i]);
    }
}

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
    vector<triangle>* triangles = new vector<triangle>();
    float radius = atof(argv[3]);
    const char* out_filename = argv[4];

    const char* sphere_filename = argv[2];
    vector<sphere>* spheres = new vector<sphere>();
    
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
    parseTriangles(attrib, shapes, triangles);

    cout << "Num of triangles is " << triangles->size() << endl;
    printTriangles(triangles);
    
    cout << "Parsing to vector of shperes..." << endl;
    if (!parseSpheres(sphere_filename, radius, spheres)) {
	cout << "Failed to load/parse sphere file: " << sphere_filename << endl;
	return 1;
    }

    cout << "Num of spheres is " << spheres->size() << endl;
    printSpheres(spheres);

    ofstream outFile(out_filename);
    if (!outFile) {
        cout << "Unable to open " << out_filename << endl;
        exit(1); // terminate with error
    }
    int sphereNum = spheres->size();
    int sphereSize = sphereNum * sizeof(sphere);
    int triangleNum = triangles->size();
    int triangleSize = triangleNum * sizeof(triangle);
    long long totalCombNum = (long long) sphereNum * triangleNum; // total combination
    printf("totalCombNum = %lld\n", totalCombNum);

    float elapsedTimeIn;
    cudaEvent_t startEventIn, stopEventIn;
    cudaEventCreate(&startEventIn); 
    cudaEventCreate(&stopEventIn);

    //allocate resources
    sphere* sphereH = (sphere *) malloc(sphereSize);
    triangle* triangleH = (triangle *) malloc(triangleSize);
    vector<int> sphereCllsnID; // sphere collision ID
    vector<int> triangleCllsnID; // triangle collision ID 

    for(int i = 0; i < sphereNum; i++)
        sphereH[i] = (*spheres)[i];
    for(int i = 0; i < triangleNum; i++)
        triangleH[i] = (*triangles)[i];

#if 1 // Serial processing
    struct timespec start;
    struct timespec end;

    // Get the starting timestamp
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < triangleNum; i++) {
        for (int j = 0; j < sphereNum; j++) {
            if (detectCollision(sphereH[j], triangleH[i])) {
                sphereCllsnID.push_back(j);
                triangleCllsnID.push_back(i);
            }
        }
    }

    // Get the ending timestamp
    clock_gettime(CLOCK_MONOTONIC, &end);
    size_t duration_nsec = (end.tv_sec - start.tv_sec) * 1000 * 1000 * 1000;
    duration_nsec += (end.tv_nsec - start.tv_nsec);
    elapsedTimeIn = (float) duration_nsec / 1000000;
#else // Parallel processing
    int sphereSegSize = 10000; // sphere segmentation size
    int triangleSegSize = 10000; // triangle segmentation size
    bool isSphereSegDividible = sphereNum % sphereSegSize == 0;
    int sphereSegNum = isSphereSegDividible? (sphereNum / sphereSegSize) : (sphereNum / sphereSegSize + 1);
    bool isTriangleSegDividible = triangleNum % triangleSegSize == 0;
    int triangleSegNum = isTriangleSegDividible? (triangleNum / triangleSegSize) : (triangleNum / triangleSegSize + 1);
    int segSize = sphereSegSize * triangleSegSize;
    int segSizeByte = segSize * sizeof(int);
    int* sphereCllsnIdH = (int *) malloc(segSizeByte);
    int* triangleCllsnIdH = (int *) malloc(segSizeByte);

    sphere* sphereD;          cudaMalloc(&sphereD, sphereSize);
    triangle* triangleD;      cudaMalloc(&triangleD, triangleSize);
    int* sphereCllsnIdD;      cudaMalloc(&sphereCllsnIdD, segSizeByte);
    int* triangleCllsnIdD;    cudaMalloc(&triangleCllsnIdD, segSizeByte);

    // For inclusive timing
    cudaEventRecord(startEventIn, 0);

    cudaMemcpy(sphereD, sphereH, sphereSize, cudaMemcpyHostToDevice);
    cudaMemcpy(triangleD, triangleH, triangleSize, cudaMemcpyHostToDevice);

    int nthreads = 512;

    for (int j = 0; j < triangleSegNum; j++) {
        for (int i = 0; i < sphereSegNum; i++) { 
            for(int k = 0; k < segSize; k++) {
                sphereCllsnIdH[k] = -1;
                triangleCllsnIdH[k] = -1;
            }

            cudaMemcpy(sphereCllsnIdD, sphereCllsnIdH, segSizeByte, cudaMemcpyHostToDevice);
            cudaMemcpy(triangleCllsnIdD, triangleCllsnIdH, segSizeByte, cudaMemcpyHostToDevice);

            int trueSphereNum = sphereSegSize;
            if (i == (sphereSegNum - 1) && (isSphereSegDividible == false))
                trueSphereNum = sphereNum % sphereSegSize;

            int trueTriangleNum = triangleSegSize;
            if (j == (triangleSegNum - 1) && (isTriangleSegDividible == false))
                trueTriangleNum = triangleNum % triangleSegSize;

            int trueSegSize = trueSphereNum * trueTriangleNum;
            
            detectCollisionCUDA<<<(trueSegSize + nthreads - 1) / nthreads, nthreads>>>(
                sphereD, triangleD, i, j, sphereSegSize, triangleSegSize, 
                trueSegSize, trueSphereNum, sphereCllsnIdD, triangleCllsnIdD);

            cudaMemcpy(sphereCllsnIdH, sphereCllsnIdD, segSizeByte, cudaMemcpyDeviceToHost);
            cudaMemcpy(triangleCllsnIdH, triangleCllsnIdD, segSizeByte, cudaMemcpyDeviceToHost);

            for(int k = 0; k < segSize; k++) {
                if (sphereCllsnIdH[k] != -1) {
                    sphereCllsnID.push_back(sphereCllsnIdH[k]);
                    triangleCllsnID.push_back(triangleCllsnIdH[k]);
                }
            }
        }
    }

    // For inclusive timing
    cudaEventRecord(stopEventIn, 0); 
    cudaEventSynchronize(stopEventIn);     
    cudaEventElapsedTime(&elapsedTimeIn, startEventIn, stopEventIn);

    cudaEventDestroy(startEventIn); 
    cudaEventDestroy(stopEventIn);

    cudaFree(sphereD);  cudaFree(triangleD);  cudaFree(sphereCllsnIdD);  cudaFree(triangleCllsnIdD);
#endif

    outFile << elapsedTimeIn << endl;
    for (long long i = 0; i < sphereCllsnID.size(); i++)
        outFile << sphereCllsnID[i] << "," << triangleCllsnID[i] << endl;

    free(sphereH);  free(triangleH);

    return 0;
}
