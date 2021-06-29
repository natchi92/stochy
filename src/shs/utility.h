/*n
 * utility.h
 *
 *  Created on: 11 Jan 2018
 *      Author: nathalie
 */


#ifndef UTILITY_H_
#define UTILITY_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include <valarray>
#include <vector>
#define ARMA_DONT_PRINT_WARNINGS
#define ARMA_DONT_PRINT_ERRORS

#define ARMA_USE_HDF5
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <bitset>
#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>

enum Library {simulator =1 , mdp=2, imdp=3};
enum grid {uniform =1 , adaptive=2};
enum property {verify_safety=1, verify_reach_avoid=2,synthesis_safety=3, synthesis_reach_avoid=4};

/* 
 * The signed area of a 2d triangle defined by its vertices 
 * 
 * Algo:
 *  Computes the determinant of the matrix
 *    | x1  y1  1 |
 *    | x2  y2  1 |
 *    | x3  y3  1 |
 *  where 
 *    p1 = [x1, y1]
 *    p2 = [x2, y2]
 *    p3 = [x3, y3]
 * 
 * Input:
 *  @param p1 first vertex of the triangle
 *  @param p2 second vertex of the triangle
 *  @param p3 third vertex of the triangle
 * 
 */ 
static double signed_area(arma::vec p1, arma::vec p2, arma::vec p3) {
  if( arma::approx_equal(p1, p2, "reldiff", 1e-3) ||
      arma::approx_equal(p2, p3, "reldiff", 1e-3) ||
      arma::approx_equal(p3, p1, "reldiff", 1e-3))
    return 0;

  double area2 =  (p1(0) - p2(1)) * (p1(0) - p2(1)) +
                  (p2(0) - p3(1)) * (p2(0) - p3(1)) +
                  (p3(0) - p1(1)) * (p3(0) - p1(1)) ;

  int sign = area2 > 0 ? 1 : -1;
  return sign * std::sqrt(std::abs(area2));
}

/* 
 * The area of a 2d triangle defined by its vertices 
 * 
 * Input:
 *  @param p1 first vertex of the triangle
 *  @param p2 second vertex of the triangle
 *  @param p3 third vertex of the triangle
 */ 
static double area(arma::vec p1, arma::vec p2, arma::vec p3) {
  return std::abs(signed_area(p1, p2, p3));
}

/* 
 * Whether the given 2D polygon is convex 
 * 
 * Pre: - the give matrix p represents a 2D polygon
 *        with the vertices in the order on the boundary
 *      - the polygon p has at least 3 vertices 
 * 
 * Input:
 *  @param p a matrix that represents a 2D polygon
 * 
 */ 
static bool is_convex(arma::mat p) {
  if(p.n_rows != 2)
    throw "This function only supports 2d polygons";
  if(p.n_cols < 3) 
    throw "This function only supports polygons - at least 3 vertices";

  double s_area = signed_area(p.col(0), p.col(1), p.col(2));
  if(s_area == 0) return false;
  
  // whether the vertices are in trigonometric order
  bool trig_order = s_area > 0;
  
  for(int i = 1; i < p.n_cols; i++) {
    s_area = signed_area(p.col(i), p.col((i+1)%p.n_cols), p.col((i+2)%p.n_cols));
    if(trig_order != s_area > 0 || s_area == 0)
      return false;
  }

  return true;
}

// function to check whether a folder exists
// according to given path
static int checkFolderExists(const char *path) {
  struct stat info;

  if (stat(path, &info) != 0) {
    return -1;
  } else if (info.st_mode & S_IFDIR) {
    return 0;
  } else {
    return -1;
  }

}

// find Location of characters in string
static std::vector<int> findLocation(std::string str, char findIt) {
  std::vector<int> characterLocations;
  for (unsigned int i = 0; i < str.size(); i++) {
    if (str[i] == findIt)
      characterLocations.push_back(i);
  }
  return characterLocations;
}

// function to split string according to chosen
// delimeter
static std::vector<std::string> splitStr(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(std::back_inserter(elems)++) = item;
  }

  return elems;
}

static arma::mat strtodMatrix(std::vector<std::string> M) {
  arma::mat m = arma::zeros<arma::mat>(M.size(), M.size());
  std::vector<std::string> elem;
  for (unsigned int j = 0; j < M.size(); j++) {
    elem = splitStr(M[j], ' ');
    for (unsigned int i = 0; i < elem.size(); i++) {
      char str[20];
      std::strcpy(str, elem[i].c_str());
      double value = strtod(str, NULL);
      m(j, i) = value;
    }
  }

  return m;
}

static arma::mat strtodMatrix(std::vector<std::string> M, unsigned cols) {
  arma::mat m = arma::zeros<arma::mat>(M.size(), cols);
  std::vector<std::string> elem;
  for (unsigned int j = 0; j < M.size(); j++) {
    elem = splitStr(M[j], ' ');
    for (unsigned int i = 0; i < elem.size(); i++) {
      char str[20];
      std::strcpy(str, elem[i].c_str());
      double value = strtod(str, NULL);
      m(j, i) = value;
    }
  }
  return m;
}

// convert arma matrix to array of std::vectors
static std::vector<std::vector<double>> mat_to_std_vec(arma::mat &A) {
  std::vector<std::vector<double>> V(A.n_rows);
  for (size_t i = 0; i < A.n_rows; ++i) {
    V[i] = arma::conv_to<std::vector<double>>::from(A.row(i));
  };
  return V;
}

// Check if input matrix m is stochastic i.e.
// sum of row / sum of col == 1
// If not stochastic normalising using the largest
// value in row to be so
static arma::mat checkStochasticity(arma::mat m) {
  arma::mat r = sum(m, 1);

  if (accu(r) > m.n_rows) {
    std::cout << "Matrix not stochastic >1, Normalising" << std::endl;
    // Normalise each row
    for (unsigned i = 0; i < (unsigned)m.n_rows; i++) {
      m.row(i) = m.row(i) / r(i);
    }
    std::cout << "Normalised matrix : " << m << std::endl;
  } else if (accu(r) < m.n_rows) {
    std::cout << "Matrix not stochastic <1" << std::endl;
    // Make stochastic
    for (unsigned i = 0; i < (unsigned)m.n_rows; i++) {
      m(i, i) = 1 - (r(i) - m(i, i));
    }
  }
  return m;
}

// delete column from arma matrix
// index of column to remove is colToRemove
static void removeColumn(arma::mat &matrix, unsigned int colToRemove) {
  unsigned int numRows = matrix.n_rows;
  unsigned int numCols = matrix.n_cols - 1;

  if (colToRemove < numCols)
    matrix = matrix(arma::span(0, numRows),
                    arma::span(colToRemove + 1, numCols - colToRemove));

  matrix.resize(numRows, numCols);
}

// compute the sigmoid function value given the
// input variable y and the sigmoid parameters d (dimension) and
// alpha (to control steepness)
static double sigmoidCompute(double y, double d, double alpha) {

  double beta = std::pow(y, d) / (std::pow(alpha, d) + std::pow(y, d));
  return beta;
}

// Obtain a sample value form the normal Distribution
// described via its mean and variance
static double getSampleNormal(double mean, double var) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  double updated = 0;
  std::normal_distribution<double> d{mean, var};
  updated = d(gen);
  return updated;
}

template <typename T>
inline bool rows_equal(const T &lhs, const T &rhs, double tol = 0.00000001) {
  return arma::approx_equal(lhs, rhs, "absdiff", tol);
}

// Read matrix file
static arma::mat readInputSignal(const char *t_path) {
  arma::mat U;
  if (U.load(t_path)) {
    return U;
  } else {
    std::cout << "File " << t_path << " not found. " << std::endl;
    exit(0);
  }
}

static int hashCode(const arma::mat &a) {
  if (a.is_empty()){
      return 0;
  }
  int result = 1;
  for (size_t nr =0; nr < a.n_rows; nr++) {
    for(size_t nc =0; nc < a.n_cols; nc++)
    {
      result = 31 * result + a(nr,nc);
    }
  }
     return result;
}

// Find unique rows in a matrix: equivalent to matlab unique(A,'rows') function
static arma::mat unique_rows(const arma::mat &x) {
  unsigned int count = 1, nr = x.n_rows, nc = x.n_cols;

  arma::mat result(nr, nc);
  result.row(0) = x.row(0);
  for (int i = 1; i < nr; i++) {
    bool matched = false;
    if (rows_equal(x.row(i), result.row(0)))
      continue;
    for (int j = i + 1; j < nr; j++) {
      if (rows_equal(x.row(i), x.row(j))) {
        matched = true;
        break;
      }
    }

    if (!matched)
      result.row(count++) = x.row(i);
  }
  return result.rows(0, count - 1);
}

// Convert a number represented in binary
// to its equivalent decimal (base 10) value
static arma::mat binary_to_decimal(arma::mat C) {
  int maxlength = 1024;
  int N = C.n_cols;
  arma::vec pvector = arma::linspace<arma::vec>(0, C.n_cols - 1, N);
  for (unsigned int i = 0; i < pvector.n_rows; ++i) {
    int powR = (int)pvector(i);
    arma::mat base(1, 1);
    base << 2;
    arma::mat a = arma::pow(base, powR);
    pvector(i) = a(0, 0);
  }
  int size_B = std::min(maxlength, N);

  arma::mat d_double = C.cols(0, size_B - 1) * pvector.rows(0, size_B - 1);
  return d_double;
}

// Recursive binary search It returns
// location of x in given array arr[l..r] is present,
// otherwise -1
static int binarySearch(int arr[], int l, int r, int x) {
  if (r >= l) {
    int mid = l + (r - l) / 2;

    // If the element is present at the middle itself
    if (arr[mid] == x)
      return mid;

    // If element is smaller than mid, then it can only
    // be presen in left subarray
    if (arr[mid] > x)
      return binarySearch(arr, l, mid - 1, x);

    // Else the element can only be present in right subarray
    return binarySearch(arr, mid + 1, r, x);
  }

  // We reach here when element is not present in array
  return -1;
}

// union of arr1[0..m-1] and arr2[0..n-1]
static std::vector<int> Union(int arr1[], int arr2[], int m, int n) 
{
  // Before finding union, make sure arr1[0..m-1]
  // is smaller
  if (m > n) 
  {
    int *tempp = arr1;
    arr1 = arr2;
    arr2 = tempp;

    int temp = m;
    m = n;
    n = temp;
  }

  // Now arr1[] is smaller

  // Sort the first array and print its elements (these two
  // steps can be swapped as order in output is not important)
  std::sort(arr1, arr1 + m);
  for (int i = 0; i < m; i++)  { }

  // Search every element of bigger array in smaller array
  // and print the element if not found
  std::vector<int> res;
  int count = 0;
  for (int i = 0; i < n; i++) 
  {
    if (binarySearch(arr1, 0, m - 1, arr2[i]) == -1) 
    {
      res.push_back(arr2[i]);
      count++;
    }
  }
  return res;
}

// Function to determine whether input Matrix
// is a diagonal matrix i.e. of the form
// [ a 0 0;
//   0 b 0;
//   0 0 c];
static void checkDiagonal(arma::mat mat) 
{
  try 
  {
    unsigned sizeC_mat = mat.n_cols;
    unsigned sizeR_mat = mat.n_rows;
    if (sizeC_mat != sizeR_mat) 
    {
      throw " Matrix is not square, thus not diagonal!";
    } 
    else 
    {
      for (unsigned i = 0; i < sizeR_mat; ++i) 
      {
        for (unsigned j = 0; j < sizeC_mat; ++j) 
	{
          if (i == j)
	  {
            if (mat(i, j) == 0) 
	    {
              throw "Matrix is not diagonal! ";
            }
          } 
	  else
	  {
            if (mat(i, j) != 0) 
	    {
              throw "Matrix is not diagonal! ";
            }
          }
        }
      }
    }
  } 
  catch (const char *msg) 
  {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

// Check if matrix is diagonal
static bool isDiagonal(arma::mat mat) 
{
  unsigned sizeC_mat = mat.n_cols;
  unsigned sizeR_mat = mat.n_rows;
  if (sizeC_mat != sizeR_mat) 
  {
    return 0;
  } 
  else 
  {
    for (unsigned i = 0; i < sizeR_mat; ++i) 
    {
      for (unsigned j = 0; j < sizeC_mat; ++j) 
      {
        if (i == j) 
	{
          if (mat(i, j) == 0) 
	  {
            return 0;
          }
        }
       	else
       	{
          if (mat(i, j) != 0) 
	  {
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

static std::vector<std::string> getAllCombinations(int N) 
{
  // string to store N-digit binary number
  std::string str;
  std::vector<std::string> res;

  // construct N-digit binary number filled with all 0's
  int j = N;
  while (j--) 
  {
    str.push_back('0');
  }
  // get first set of all k-bit all zeros
  std::string temp = "0";
  j = N - 1;
  while (j--) 
  {
    temp += "0";
  }

  res.push_back(temp);
  
  // store all numbers with k-bit set together in ascending order
  for (int k = 1; k <= N; k++) 
  {
    // set last k bits to '1'
    str[N - k] = '1';
    std::string curr = str;

    do 
    {
      res.push_back(str);
      // std::cout << str << std::endl;
    } 
    while (next_permutation(str.begin(), str.end()));
  }
  return res;
}

static size_t skipControls(const char *pData, size_t pos, size_t length) 
{
  while (pos < length) 
  {
    if ((pData[pos] != 32) && (pData[pos] != 8) && (pData[pos] != '\n'))
      return pos;
    pos++;
  }
  return pos;
}

// Function to recursively compute all the possible
// grid cells given a set of vertices describing
// their upper and lower bound  values
static void rec(std::vector<arma::mat> &intervals, arma::mat cells,
                arma::mat temp, unsigned index, unsigned what, unsigned n_rows,
                unsigned cols) {

  for (unsigned r = 0; r < what; r = r + 2) 
  {
    arma::mat tempr = cells(index, arma::span(r, r + 1));

    if (n_rows < 3) 
    {
      tempr = join_vert(temp, tempr);
    } 
    else 
    {
      tempr = temp;

      tempr.row(cols) = cells(cols, arma::span(r, r + 1));
      cols++;
      if (cols > n_rows - 1) 
      {
        cols = 0;
      }
      intervals.push_back(tempr);
    }

    if ((n_rows - 1) == index) 
    {
      if (n_rows < 3) 
      {
        intervals.push_back(tempr);
      }
    } 
    else 
    {
      if (n_rows > 2) 
      {
        rec(intervals, cells,
            cells(arma::span(0, n_rows - 1), arma::span(r, r + 1)), index + 1,
            what, n_rows, cols);

      }
      else
      {
        rec(intervals, cells, cells(0, arma::span(r, r + 1)), index + 1, what,
            n_rows, 0);
      }
    }
  }
}

// Give a polygon (which may be open), close it such  that
// it is a closed polygon
static arma::mat close_loops(arma::mat xv, arma::mat yv) {
  bool xnan = xv.has_nan();
  bool ynan = yv.has_nan();

  if (!(xnan || ynan)) {
    // Simply connected polygon
    // Need a min of 3 points to define a polygon
    int nump = xv.n_elem;
    if (nump < 3) {
      return arma::zeros(2, 0);
    } else {
      // If the polygon is open, then close it.
      bool xvc = (xv(0, 0) != xv(0, nump - 1));
      bool yvc = (yv(0, 0) != yv(0, nump - 1));
      if (xvc || yvc) {
        int endx = xv.n_cols;
        int endy = yv.n_cols;
        arma::mat a = arma::resize(xv, xv.n_rows, endx + 1);
        a(0, endx) = xv(0, 0);
        arma::mat b = arma::resize(yv, yv.n_rows, endx + 1);
        b(0, endy) = yv(0, 0);
        arma::mat xyv = join_vert(a, b);
        return xyv;
      }
    }
    return arma::zeros(2, 0);
  }
}

// Determine whether given set of vertices can be found within a polygon
static arma::umat vec_inpolygon(int Nv, arma::mat x, arma::mat y, arma::mat xv,
                                arma::mat yv) {
  int Np = x.n_elem;
  x = repmat(x, Nv, 1);
  y = repmat(y, Nv, 1);

  // Compute scale factors for eps that are based on the original vertex
  // locations. This ensures that the test points that lie on the boundary
  // will be evaluated using an appropriately scaled tolerance.
  // (m and mp1 will be reused for setting up adjacent vertices later on.)

  arma::mat avx = arma::abs(0.5 * (xv.rows(0, Nv - 2) + xv.rows(1, Nv - 1)));
  arma::mat avy = arma::abs(0.5 * (yv.rows(0, Nv - 2) + yv.rows(1, Nv - 1)));

  arma::mat scaleFactor = arma::max(avx.rows(0, Nv - 2), avy.rows(0, Nv - 2));
  scaleFactor =
      arma::max(scaleFactor, avx.rows(0, Nv - 2) % avy.rows(0, Nv - 2));

  // Translate the vertices so that the test points are
  // at the origin.
  xv = repmat(xv, 1, Np);
  yv = repmat(yv, 1, Np);
  xv = xv.rows(0, Np) - x;
  yv = yv.rows(0, Np) - y;

  // Compute the quadrant number for the vertices relative
  // to the test points.
  arma::umat posX = xv > 0;
  arma::umat posY = yv > 0;
  arma::umat negX = xv <= 0;
  arma::umat negY = yv <= 0;

  arma::mat quad = arma::conv_to<arma::mat>::from(
      (negX && posY) + 2 * (negX && negY) + 3 * (posX && negY));
  // std::cout << "quad: " << quad << std::endl;
  // Compute the sign() of the cross product and dot product
  // of adjacent vertices.
  arma::mat theCrossProd = xv.rows(0, Nv - 2) % yv.rows(1, Nv - 1) -
                           xv.rows(1, Nv - 1) % yv.rows(0, Nv - 2);
  // std::cout << "theCrossProd: " << theCrossProd << std::endl;

  arma::mat signCrossProduct = arma::sign(theCrossProd);
  // std::cout << "signCrossProduct: " << signCrossProduct << std::endl;

  // Adjust values that are within epsilon of the polygon boundary.
  // Making epsilon larger will treat points close to the boundary as
  // being "on" the boundary. A factor of 3 was found from experiment to be
  // a good margin to hedge against roundoff.
  float eps = 2.220446049250313e-16;
  arma::mat scaledEps = scaleFactor * eps * 3;

  for (unsigned i = 0; i < theCrossProd.n_cols; i++) {
    for (unsigned j = 0; j < theCrossProd.n_rows; j++) {
      bool idx = abs(theCrossProd(j, i)) < scaledEps(j);
      if (idx) {
        signCrossProduct(j, i) = 0;
      }
    }
  }
  arma::mat dotProduct = xv.rows(0, Nv - 2) % xv.rows(1, Nv - 1) +
                         yv.rows(0, Nv - 2) % yv.rows(1, Nv - 1);

  // Compute the vertex quadrant changes for each test point.
  arma::mat diffQuadA = quad.rows(0, quad.n_rows - 2);
  arma::mat diffQuad = quad.rows(1, quad.n_rows - 1);

  diffQuad = diffQuad - diffQuadA;
  // Fix up the quadrant differences. Replace 3 by -1 and -3 by 1.
  // Any quadrant difference with an absolute value of 2 should have
  // the same sign as the cross product.
  for (unsigned i = 0; i < diffQuad.n_cols; i++) {
    for (unsigned j = 0; j < diffQuad.n_rows; j++) {
      bool idx = (abs(diffQuad(j, i)) == 3);
      if (idx) {
        diffQuad(j, i) = -diffQuad(j, i) / 3;
      }
      idx = (abs(diffQuad(j, i)) == 2);
      if (idx) {
        diffQuad(j, i) = 2 * signCrossProduct(j, i);
      }
    }
  }
  // Find the inside points.
  // Ignore crossings between distinct loops that are separated by NaNs
  arma::mat sdiffQuad = arma::sum(diffQuad);

  arma::umat ssdiffQuad = arma::conv_to<arma::umat>::from(sdiffQuad);

  arma::umat in = (sdiffQuad != 0);

  // Find the points on the polygon. If the cross product is 0 and
  // the dot product is nonpositive anywhere, then the corresponding
  // point must be on the contour.
  arma::umat scP = (signCrossProduct == 0);

  arma::umat dP = (dotProduct <= 0);
  //  std::cout << "dP: " << dP << std::endl;
  arma::umat on = any(scP && dP);
  //  std::cout << "on: " << on << std::endl;

  in = in || on;
  return in;
}

// Determine whether given point can be found within a 2D polygon
static arma::umat vec_inpolygon(int Nv, double x1, double y1, arma::mat xv,
                                arma::mat yv) {

  int Np = 1;
  arma::mat x = arma::ones<arma::mat>(Nv, 1) * x1;
  //  std::cout << "x: " << x << std::endl;
  arma::mat y = arma::ones<arma::mat>(Nv, 1) * y1;
  //  std::cout << "y: " << y << std::endl;

  // Compute scale factors for eps that are based on the original vertex
  // locations. This ensures that the test points that lie on the boundary
  // will be evaluated using an appropriately scaled tolerance.
  // (m and mp1 will be reused for setting up adjacent vertices later on.)

  arma::mat avx = arma::abs(0.5 * (xv.rows(0, Nv - 2) + xv.rows(1, Nv - 1)));
  arma::mat avy = arma::abs(0.5 * (yv.rows(0, Nv - 2) + yv.rows(1, Nv - 1)));

  arma::mat scaleFactor = arma::max(avx.rows(0, Nv - 2), avy.rows(0, Nv - 2));
  scaleFactor =
      arma::max(scaleFactor, avx.rows(0, Nv - 2) % avy.rows(0, Nv - 2));

  // Translate the vertices so that the test points are
  // at the origin.
  xv = repmat(xv, 1, Np);
  yv = repmat(yv, 1, Np);
  xv = xv - x;
  yv = yv - y;

  // Compute the quadrant number for the vertices relative
  // to the test points.
  arma::umat posX = xv > 0;
  arma::umat posY = yv > 0;
  arma::umat negX = xv <= 0;
  arma::umat negY = yv <= 0;

  arma::mat quad = arma::conv_to<arma::mat>::from(
      (negX && posY) + 2 * (negX && negY) + 3 * (posX && negY));

  // Compute the sign() of the cross product and dot product
  // of adjacent vertices.
  arma::mat theCrossProd = xv.rows(0, Nv - 2) % yv.rows(1, Nv - 1) -
                           xv.rows(1, Nv - 1) % yv.rows(0, Nv - 2);

  arma::mat signCrossProduct = arma::sign(theCrossProd);

  // Adjust values that are within epsilon of the polygon boundary.
  // Making epsilon larger will treat points close to the boundary as
  // being "on" the boundary. A factor of 3 was found from experiment to be
  // a good margin to hedge against roundoff.
  float eps = 2.220446049250313e-16;
  arma::mat scaledEps = scaleFactor * eps * 3;

  for (unsigned i = 0; i < theCrossProd.n_cols; i++) {
    for (unsigned j = 0; j < theCrossProd.n_rows; j++) {
      bool idx = abs(theCrossProd(j, i)) < scaledEps(j);
      if (idx) {
        signCrossProduct(j, i) = 0;
      }
    }
  }

  arma::mat dotProduct = xv.rows(0, Nv - 2) % xv.rows(1, Nv - 1) +
                         yv.rows(0, Nv - 2) % yv.rows(1, Nv - 1);

  // Compute the vertex quadrant changes for each test point.
  arma::mat diffQuadA = quad.rows(0, quad.n_rows - 2);
  arma::mat diffQuad = quad.rows(1, quad.n_rows - 1);

  diffQuad = diffQuad - diffQuadA;

  // Fix up the quadrant differences. Replace 3 by -1 and -3 by 1.
  // Any quadrant difference with an absolute value of 2 should have
  // the same sign as the cross product.
  for (unsigned i = 0; i < diffQuad.n_cols; i++) {
    for (unsigned j = 0; j < diffQuad.n_rows; j++) {
      bool idx = (abs(diffQuad(j, i)) == 3);
      if (idx) {
        diffQuad(j, i) = -diffQuad(j, i) / 3;
      }
      idx = (abs(diffQuad(j, i)) == 2);
      if (idx) {
        diffQuad(j, i) = 2 * signCrossProduct(j, i);
      }
    }
  }

  // Find the inside points.
  // Ignore crossings between distinct loops that are separated by NaNs
  arma::mat sdiffQuad = arma::sum(diffQuad);

  arma::umat ssdiffQuad = arma::conv_to<arma::umat>::from(sdiffQuad);

  arma::umat in = (sdiffQuad != 0);

  // Find the points on the polygon. If the cross product is 0 and
  // the dot product is nonpositive anywhere, then the corresponding
  // point must be on the contour.
  arma::umat scP = (signCrossProduct == 0);

  arma::umat dP = (dotProduct <= 0);
  // std::cout << "dP: " << dP << std::endl;
  arma::umat on = any(scP && dP);
  // std::cout << "on: " << on << std::endl;

  in = in || on;
  return in;
}

// Check if point is in polygon using ray-casting
// I run a semi-infinite ray horizontally (increasing x, fixed y) out from the
// test point, and count how many edges it crosses. At each crossing, the ray
// switches between inside and outside. This is called the Jordan curve theorem.
// Inputs:
// nvert: Number of vertices in the polygon.
// vertx, verty: Arrays containing the x- and y-coordinates of the polygon's
// vertices testx, testy: X- and y-coordinate of the test point.
static int pnpoly(int nvert, arma::mat vertx, arma::mat verty, arma::mat testx,
                  arma::mat testy) {

  arma::mat maxX = arma::max(vertx, 1);
  arma::mat minX = arma::min(vertx, 1);
  arma::mat maxY = arma::max(verty, 1);
  arma::mat minY = arma::min(verty, 1);
  int lhs1 = 0, lhs2 = 0, rhs1 = 0, rhs2 = 0;
  for (auto k = 0; k < testx.n_cols; k++) {
    arma::uvec l1 = (testx(0, k) >= minX);
    lhs1 += l1(0);
    arma::uvec l2 = (testx(0, k) <= maxX);
    lhs2 += l2(0);
    arma::uvec l3 = (testy(0, k) >= minY);
    rhs1 += l3(0);
    arma::uvec l4 = (testy(0, k) <= maxY);
    rhs2 += l4(0);
  }

  if (!(lhs1 != 0 && lhs2 != 0 && rhs1 != 0 && rhs2 != 0)) {
    return 0;
  }
  // close loop
  arma::mat xyv = close_loops(vertx, verty);
  arma::umat in =
      vec_inpolygon(xyv.n_cols, testx, testy, xyv.row(0).t(), xyv.row(1).t());
  int in_acc = arma::accu(in);
  if (in_acc == nvert) {
    return 1;
  } else {
    return 0;
  }
}

// Check if point is in polygon using ray-casting
// I run a semi-infinite ray horizontally (increasing x, fixed y) out from the
// test point, and count how many edges it crosses. At each crossing, the ray
// switches between inside and outside. This is called the Jordan curve theorem.
// Inputs:
// nvert: Number of vertices in the polygon.
// vertx, verty: Arrays containing the x- and y-coordinates of the polygon's
// vertices testx, testy: X- and y-coordinate of the test point.
static int pnspoly(int nvert, arma::mat vertx, arma::mat verty, arma::mat testx,
                   arma::mat testy) {

  arma::mat maxX = arma::max(vertx, 1);
  arma::mat minX = arma::min(vertx, 1);
  arma::mat maxY = arma::max(verty, 1);
  arma::mat minY = arma::min(verty, 1);

  int lhs1 = 0, lhs2 = 0, rhs1 = 0, rhs2 = 0;
  for (auto k = 0; k < testx.n_cols; k++) {
    arma::uvec l1 = (testx(0, k) >= minX);
    lhs1 += l1(0);
    arma::uvec l2 = (testx(0, k) <= maxX);
    lhs2 += l2(0);
    arma::uvec l3 = (testy(0, k) >= minY);
    rhs1 += l3(0);
    arma::uvec l4 = (testy(0, k) <= maxY);
    rhs2 += l4(0);
  }

  if (!(lhs1 != 0 && lhs2 != 0 && rhs1 != 0 && rhs2 != 0)) {
    return 0;
  }
  // close loop
  arma::mat xyv = close_loops(vertx, verty);
  //  std::cout << "xyv: " << xyv << std::endl;

  arma::umat in =
      vec_inpolygon(xyv.n_cols, testx, testy, xyv.row(0).t(), xyv.row(1).t());
  int in_acc = arma::accu(in);
  return in_acc;
}

// Check whether a 2D test point is inside the polygon described using
// vertx and verty
static int pnpoly(int nvert, arma::mat vertx, arma::mat verty, double testx,
                  double testy) {

  arma::mat maxX = arma::max(vertx, 1);
  arma::mat minX = arma::min(vertx, 1);
  arma::mat maxY = arma::max(verty, 1);
  arma::mat minY = arma::min(verty, 1);

  int lhs1 = (testx >= minX(0, 0));
  int lhs2 = (testx <= maxX(0, 0));
  int rhs1 = (testy >= minY(0, 0));
  int rhs2 = (testy <= maxY(0, 0));

  if (!(lhs1 != 0 && lhs2 != 0 && rhs1 != 0 && rhs2 != 0)) {
    return 0;
  }
  // close loop
  arma::mat xyv = close_loops(vertx, verty);
  // std::cout << "xyv: " << xyv << std::endl;

  arma::umat in =
      vec_inpolygon(xyv.n_cols, testx, testy, xyv.row(0).t(), xyv.row(1).t());
  int in_acc = arma::accu(in);
  if (in_acc == nvert) {
    return 1;
  } else {
    return 0;
  }
}

#endif /* UTILITY_H_ */
;
