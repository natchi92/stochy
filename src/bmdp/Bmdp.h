/*
 * Bmdp.h
 *
 *  Created on: 14 Jun 2018
 *      Author: natchi
 */

#ifndef LEARNING_BMDP_H_
#define LEARNING_BMDP_H_
#include "InputSpec.h"
#include <armadillo>
#include <bmdp/BMDP.hpp>
#include <bmdp/IntervalValueIteration.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlopt.hpp>

// Definition of grids for each mode
struct Strmode {
  std::vector<arma::mat> vertices;
  arma::mat mode_center;
  arma::mat states;
};

// Description of the individual modes
struct Intmode {
  arma::mat transfermatrix;
  // unsigned dynid;
  arma::mat boundary;
  arma::mat reftol;
  arma::mat gridsize;
  arma::mat postCov;
};

struct Dyn {
  std::vector<ssmodels_t> dynamics;
  std::vector<int> mode;
};
// Definition of grid properties
// and dynamics
struct Sys {
  Dyn dyn;
  std::vector<Intmode> mode;
  std::vector<arma::mat> mean; // noise mean
  std::vector<arma::mat> cov;  // noise cov
  arma::mat boundary;
  arma::mat reftol;
  arma::mat gridsize;
};

class bmdp_t {
public:
  std::vector<Strmode> mode; // input action
  arma::cube vertices;       // original vertices
  arma::mat states;          // all states
  arma::sp_mat Stepsmin;     // lower probability bound
  arma::sp_mat Stepsmax;     // upper probability bound
  arma::mat L;               // Labels of states
  arma::mat Solution;        // Verification solution (Pmin, Pmax)
  arma::mat Policy;          // Synthesis policy genereated
  double E_max;              // Max resulting abstraction error
  double actNum;
  double eps; // min allowable abstraction error for adaptive-refinement
  Sys desc;   // Structure of internal dynamics
  std::mutex steps_mutex; // mutex used for writing in Stepsmin and Stepsmax
  bmdp_t();   // initialising the bmdp
  bmdp_t(taskSpec_t &myTask);
  bmdp_t(taskSpec_t &myTask, shs_t<arma::mat, int> &inModel);
  void bmdpAbstraction(int T, int RA); // Generate the abstraction
  void getHybSysModes();               // Grid domain
  void getSteps(int q); // computes the probabilitis starting from state q
  void getSteps(); // compute lower and upper probabtilities (uniform)
  void
  getSteps(double epsilon); // compute lower and upper probabilitoes (adaptive)
  void createPyGrid(std::vector<Strmode> ver_valid,
                    arma::mat boundary); // print grid
  void createPyGrid(std::vector<Strmode> ver_valid, arma::mat boundary,
                    std::vector<double> minP); // print grid
  arma::vec getGrid(arma::mat boundary, arma::mat gridsize, arma::mat tol,
                    int m);
  arma::vec getGrid1D(arma::mat boundary, arma::mat gridsize, arma::mat tol,
                      int i);
  double checkPmin(double pmin, arma::mat qpost_TF, arma::mat qprime_TF,
                   size_t dim);
  double checkPmax(double pmax, arma::mat qpost_TF, arma::mat qprime_TF,
                   size_t dim);
  double getMinTranProb(arma::mat qpost_TF, arma::mat qprime_TF, size_t dim);
  double getMinTranProb2Rect(arma::mat qpost_TF, arma::mat qprime_TF,
                             arma::mat qprime_ctr, size_t dim);
  double getMaxTranProb(arma::mat qpost_TF, arma::mat qprime_TF, size_t dim);
  double getMaxTranProb2Rect(arma::mat qpost_TF, arma::mat qprime_TF,
                             arma::mat qprime_ctr, size_t dim);
  void getModes(Sys sys);
  void createSynthFile(arma::uvec phi1, arma::uvec Labelsd);
  // Run synthesis and store values next
  void runSynthesis(double eps, double iterationNum);
  void runSafety(double eps,
                 double iterationNum); // Perform safety verification
  arma::vec getESafety(double eps, double iterationNum);
  void readSpec(const char *t_path); // read specification from text files
  void obtainBMDPdetailsfromMat(const char *fn);
  void populateBMDPSpec(matvar_t &content);
  void getStepsBasedonMedian(double epsilon, int T);
  arma::vec getGridNondiag(arma::mat boundary, arma::mat gridsize,
                           arma::mat tol, int i);
  arma::vec getGridNondiagRA(arma::mat boundary, arma::mat gridsize,
                             arma::mat tol, int i,
                             std::vector<std::vector<arma::mat>> v_phi1,
                             std::vector<std::vector<arma::mat>> v_phi2);
  double getMinTranProbNonDiag(arma::mat qpost_TF, arma::mat qprime_TF,
                               size_t dim);
  double getMaxTranProbNonDiag(arma::mat qpost_TF, arma::mat qprime_TF,
                               size_t dim);
  void getStepsNonDiag(int q); // Computes the steps from state q
  void getStepsNonDiag();
  arma::uvec getLabels(std::string phiFile, int x_dim, bool under_approx);
  std::vector<std::vector<arma::mat>> getLabelVertices(std::string phiFile,
                                                       int x_dim);
  arma::vec getESynthesis(double eps, double iterationNum);
  void formatOutput(double time, std::string cT);
  virtual ~bmdp_t();
};

// this function splits a rectangle into 4 rectangles
// input:    - ver: vertices of a rectangle of size 1 x 8. Elements 1-4
//                   are the x-coordinates and elements 5-8 are the
//                   y-cooreidnates.
// output:   - newver: is a matrix of 4 x 8. Each row corresponds to the
//                       vertices of a rectangle. vernew(i,1:4) are the
//                       x-coordinates of the vertices vernew(i,5:8) are the
//                       y-coordinates of the vertices
static arma::mat refinRectangle(arma::vec ver) {
  // find the center of the rectangel
  double xctr = arma::mean(ver(arma::span(0, 3)));
  double yctr = arma::mean(ver(arma::span(4, 7)));
  double xmin = arma::min(ver(arma::span(0, 3)));
  double xmax = arma::max(ver(arma::span(0, 3)));
  double ymin = arma::min(ver(arma::span(4, 7)));
  double ymax = arma::max(ver(arma::span(4, 7)));

  // bottom left, bottom right, top left, top right
  arma::mat newvec = {{xmin, xctr, xctr, xmin, ymin, ymin, yctr, yctr},
                      {xctr, xmax, xmax, xctr, ymin, ymin, yctr, yctr},
                      {xmin, xctr, xctr, xmin, yctr, yctr, ymax, ymax},
                      {xctr, xmax, xmax, xctr, yctr, yctr, ymax, ymax}};

  return newvec;
}

// Additional necessary functions
static arma::mat discretiseCellBoundaries(arma::mat vert,
                                          std::vector<float> resol) {
  // discretises the boundaries of the rectangle given the vertices
  arma::vec vx =
      arma::regspace(arma::min(vert.row(0)), resol[0], arma::max(vert.row(0)));
  arma::vec vy =
      arma::regspace(arma::min(vert.row(1)), resol[1], arma::max(vert.row(1)));

  // set boundary
  arma::vec x =
      vx; //, arma::min(vert.row(0))*arma::ones<arma::vec>(vy.size());//, vx,
          //arma::max(vert.row(0))*arma::ones<arma::vec>(vy.size())};
  arma::vec y =
      vy; // arma::min(vert.row(1))*arma::ones<arma::vec>(vx.size()), vy,
          // arma::max(vert.row(1))*arma::ones<arma::vec>(vx.size()), vy};
  arma::mat v(x.n_elem, 2);
  v = arma::join_vert(x, y);

  return v;
}

// this function splits a rectangle into 4 rectangles
// input:    - ver: vertices of a rectangle of size 1 x 8. Elements 1-4
//                   are the x-coordinates and elements 5-8 are the
//                   y-cooreidnates.
// output:   - newver: is a matrix of 4 x 8. Each row corresponds to the
//                       vertices of a rectangle. vernew(i,1:4) are the
//                       x-coordinates of the vertices vernew(i,5:8) are the
//                       y-coordinates of the vertices
static arma::mat refineRectangle(arma::mat ref) {
  // find center of the rectangle
  double xctr = arma::mean(ref(0, arma::span(0, 3)));
  double yctr = arma::mean(ref(0, arma::span(4, 7)));
  double xmin = arma::min(ref(0, arma::span(0, 3)));
  double xmax = arma::max(ref(0, arma::span(0, 3)));
  double ymin = arma::min(ref(0, arma::span(5, 7)));
  double ymax = arma::max(ref(0, arma::span(5, 7)));

  arma::mat newver(4, 8);
  // bottom left, bottom right, top left, top right
  newver = {
      {xmin, xctr, xctr, xmin, ymin, ymin, yctr, yctr},
      {xctr, xmax, xmax, xctr, ymin, ymin, yctr, yctr},
      {xmin, xctr, xctr, xmin, yctr, yctr, ymax, ymax},
      {xctr, xmax, xmax, xctr, yctr, yctr, ymax, ymax},
  };

  return newver;
}

// this function splits a hyperrectangle into 2 rectangles
static arma::mat refineHyperRectangle(arma::mat ref) {

  arma::mat rmean = arma::mean(ref, 1);
  arma::mat rmin = arma::min(ref, 1);

  arma::mat rmax = arma::max(ref, 1);

  arma::mat newver = join_horiz(rmin, rmean);
  arma::mat newvar2 = join_horiz(rmean, rmax);

  return join_horiz(newver, newvar2);
}

// this function splits a hyperrectangle along
// largest dimension in specific diameter
static arma::mat refineHyperRectangleLargestDiam(arma::mat ref) {
  // Find largest diameter
  double lrg_idx = 0;
  if (ref.n_rows > 1) {
    arma::vec delta = ref.col(1) - ref.col(0);
    //	std::cout << "delta" << delta << std::endl;
    lrg_idx = delta.index_max();
    //	std::cout << "lrg_idx" << lrg_idx << std::endl;
  }
  arma::mat rmin = arma::min(ref, 1);

  arma::mat rmax = arma::max(ref, 1);
  arma::mat rmean = rmax; // arma::mean(ref,1);//
  rmean(lrg_idx, 0) = arma::mean(ref.row(lrg_idx));
  std::cout << "rmean: " << rmean << std::endl;

  arma::mat newver = join_horiz(rmin, rmean);
  arma::mat newvar2 = join_horiz(rmean, rmax);

  return join_horiz(newver, newvar2);
}

// this function checks if matrices are within tolerenace
static bool tolInvalid(arma::mat vt, arma::mat tol) {

  arma::mat v1d = arma::max(vt, 1) - arma::min(vt, 1);
  if (arma::accu(arma::all(v1d < tol.t())) < tol.n_elem) {
    return 1;
  } else {
    return 0;
  }
}

static double polygonArea(arma::vec x, arma::vec y) {
  double area = 0;
  int j = y.n_elem - 1;

  for (size_t i = 0; i < y.n_elem; ++i) {
    area += (x(j) + x(i)) * (y(j) - y(i));
    j = i;
  }

  return std::abs(area * 0.5);
}

// Find whethere point lies within HyperRect
// described using intervals for each dimension
// dim
/// [v_l v_u] x ...x [ ]
static bool pnHyperRect(arma::mat point, arma::mat rect) {
  bool inHyperRect = 0;
  if ((arma::all(point.col(0) >= rect.col(0)) ||
       arma::approx_equal(point.col(0), rect.col(0), "absdiff", 0.01)) &&
      (arma::all(point.col(1) <= rect.col(1)) ||
       arma::approx_equal(point.col(1), rect.col(1), "absdiff", 0.01))) {
    inHyperRect = 1;
  }
  return inHyperRect;
}

static double myMinOptfuncBMDP(const std::vector<double> &x,std::vector<double> &grad, void *my_func_data) {
  auto *v = (arma::mat *)my_func_data;
  arma::mat vm = v[0];
  size_t dim = (size_t)vm(vm.n_rows - 1, 0);
  double sig = (double)vm(vm.n_rows -1,1);
  double denom = std::sqrt(2)*sig;
  double outer = 1 / (std::pow(2, dim));
  double inner = 1;
  int j = 0;
  for (unsigned i = 0; i < dim; ++i) {
    double lower = ((x[i] - vm(j, 0)) / denom);
    double upper = ((x[i] - vm(j, 1)) / denom);
    inner *= std::erf(lower) - std::erf(upper);
    ++j;
  }
  double f = std::log(outer * inner);

  return f;
}

static double myMinOptfuncRectBMDP(const std::vector<double> &x,
                                   std::vector<double> &grad,
                                   void *my_func_data) {
  auto *v = (arma::mat *)my_func_data;
  arma::mat vm = v[0];

  size_t dim = (size_t)vm(vm.n_rows - 2, 0);
  size_t sig = (size_t)vm(vm.n_rows - 2, 1);
  size_t qcols = (size_t)vm(vm.n_rows - 1, 0);
  size_t size_qp = (size_t)std::sqrt(qcols - 1);
  dim = 2;
  double denom = std::sqrt(2)*sig;
  double outer = 1 / (std::pow(2, dim));
  arma::vec inner = arma::ones<arma::vec>(size_qp);
  int j = 0;
  for (unsigned i = 0; i < dim; ++i) {
    arma::vec xnew = arma::ones<arma::vec>(qcols) * x[i];
    arma::vec lower =
        ((vm(arma::span(i * size_qp, (i + 1) * size_qp - 1), 1) -
          xnew(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0)) /
         denom);
    arma::vec upper =
        ((vm(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0) -
          xnew(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0)) /
         denom);
    inner %= arma::erf(lower) - arma::erf(upper);
    ++j;
  }
  return std::abs(outer * arma::sum(inner));
}

static void readFile(const char *filename, MDP::BMDP &bmdp) {
  std::vector<MDP::State> vecState;
  std::vector<MDP::Action> vecAction;
  unsigned stateNum, actNum, termState, termNum, cur, act, dest;
  std::vector<unsigned int> stateTerm;
  double pmin, pmax;
  std::ifstream fin(filename);

  if (!fin) {
    std::cout << "Could NOT open the generated imdp file. I.e.Abstraction into "
                 "imdp not correctly saved "
              << std::endl;
    exit(0);
  }

  // read the first 3 numbers
  fin >> stateNum >> actNum >> termNum;

  // read terminal states
  for (unsigned int i = 0; i < termNum; i++) {
    fin >> termState;
    // std::cout << termState << std::endl;
    stateTerm.push_back(termState);
  }

  // add bmdp states with the correct reward
  for (unsigned int i = 0; i < stateNum; i++) {
    vecState.push_back(MDP::State(i));

    bool termFlag = false;
    for (unsigned int j = 0; j < stateTerm.size(); j++) {
      if (stateTerm[j] == i) {
        termFlag = true;
        break;
      }
    }

    if (termFlag)
      bmdp.addState(&vecState[i], 1.0);
    else
      bmdp.addState(&vecState[i], 0.0);
  }

  // add actions
  for (unsigned int i = 0; i < actNum; i++) {
    vecAction.push_back(MDP::Action(i));
    bmdp.addAction(&vecAction[i]);
  }

  // add transitions
  while (!fin.eof()) {
    fin >> cur;
    if (fin.eof())
      break;

    fin >> act >> dest >> pmin >> pmax;

    bool termFlag = false;

    // fin >> cur >> act >> dest >> pmin >> pmax;

    for (unsigned int i = 0; i < stateTerm.size(); i++) {
      if (stateTerm[i] == cur) {
        termFlag = true;
        break;
      }
    }

    if (!termFlag)
      bmdp.addTransition(cur, act, dest, pmin, pmax);
  }
  fin.close();
}

#endif /* LEARNING_BMDP_H_ */
