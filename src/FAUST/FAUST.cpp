/* Author: Nathalie Cauchi
 * File: FAUST.cpp
 * Created on: 26 Apr 2018
 */
#include "FAUST.h"

#include <cmath>
#include <ostream>

#include <armadillo>
#include <ginac/basic.h>
#include <cubature.h>
#include <nlopt.hpp>

// Generic instatiation of FAUST object class
// All fields initialised to empty constructs
faust_t::faust_t() {
  X = arma::zeros<arma::mat>(1, 1);
  X.reset(); // this resets the size of the matrix to 0
  U = arma::zeros<arma::mat>(1, 1);
  U.reset();
  Tp.push_back(arma::zeros<arma::mat>(1, 1));
  E = 0;
  Kernel = 0;
  KernelInternal = 0;
  V = arma::zeros<arma::mat>(1, 1);
  V.reset();
  constant = 1;
  shs_t<arma::mat, int> mod;
  model = mod;
  OptimalPol = arma::zeros<arma::umat>(1, 1);
  OptimalPol.reset();
}

// Instatiation of FAUST object class when the
// structure of the model is known
// and has the form of a DTSHS
faust_t::faust_t(shs_t<arma::mat, int> &inModel) {
  X = arma::zeros<arma::mat>(1,1);
  X.reset();
  U = arma::zeros<arma::mat>(1,1);
  U.reset();
  Tp.push_back(arma::zeros<arma::mat>(1, 1));
  E = 10;
  myKernel(inModel);
  V = X;
  model = inModel;
  OptimalPol = arma::zeros<arma::umat>(1,1);
  OptimalPol.reset();
}

// Destructor of FAUST object class
faust_t::~faust_t() {}

// Definition of stochastic Kernel using Symbolic
// methods and based on input model
// Input:
//       shs_t<arma::mat,int> myModel: Reference to input model in form of
//       DTSHS
void faust_t::myKernel(shs_t<arma::mat, int> &inModel) {
  // Dimension of x
  int dim = inModel.x_mod[0].A.n_cols;
  // Dimension of u
  int dim_u = inModel.x_mod[0].B.n_cols;

  // Definition of symbolic variables for "x" and "u"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};
  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};

  // Create list of symbols to be used
  // These are a function of the number of continuous and input variables
  // in the model
  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }

  // Define symbolic matrices for Kernel
  GiNaC::matrix A(dim, dim);
  GiNaC::matrix Sm(dim, dim);
  GiNaC::matrix Qm(dim, 1);
  GiNaC::matrix B(dim, dim_u);
  GiNaC::matrix Xvec(dim, 1);
  GiNaC::matrix Xbvec(dim, 1);
  GiNaC::matrix Uvec(dim_u, 1);

  int count = 0;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      A(i, j) = inModel.x_mod[0].A(i, j);
      Sm(i, j) = inModel.x_mod[0].sigma(i, j);
    }
    if(inModel.x_mod[0].Q.is_empty()== true) {
      Qm(i,0) = 0;
    }
    else {
      Qm(i,0) =  inModel.x_mod[0].Q(i,0);
    }
    Xvec(i, 0) = symbols[i + count];
    Xbvec(i, 0) = symbols[i + 1 + count];
    count++;
  }
  for (int i = 0; i < dim_u; ++i) {
    for (int j = 0; j < dim; ++j) {
      B(j, i) = inModel.x_mod[0].B(j, i);
    }
    Uvec(i, 0) = symbols[2 * dim + i];
  }

  GiNaC::ex E_xbar;
  if (dim_u > 0) {
    E_xbar = A * Xvec + B * Uvec + Qm;
  } else {
    E_xbar = A * Xvec + Qm;
  }
  E_xbar = E_xbar.evalm();
  GiNaC::ex inner = (Xbvec - E_xbar).evalm();
  GiNaC::matrix inner2 = GiNaC::ex_to<GiNaC::matrix>(inner);
  GiNaC::ex lexp = inner2.transpose();
  GiNaC::ex Sminv = Sm.inverse();
  GiNaC::ex Smdet = GiNaC::determinant(Sm);
  GiNaC::ex exter =
      1 /
      (GiNaC::sqrt(GiNaC::pow((2 * GiNaC::Pi).evalf(), dim).evalf() * Smdet))
          .evalf();
  GiNaC::ex mat = (lexp.evalm() * Sminv.evalm() * inner.evalm()).evalm();
  GiNaC::ex Kernel = exter * (GiNaC::exp(-0.5 * mat).evalf());
  this->Kernel = Kernel;
  this->KernelInternal = mat;
  this->constant = GiNaC::ex_to<GiNaC::numeric>(exter).to_double();
  this->model = inModel;
}

// Evaluation of stochastic Kernel based on the input values
// of x
// Input:
//       std::vector<arma::mat> x: Vector of values for each continuous variables
// Assumes symbolic kernel was previously defined using
// faust_t::myKernel(shs_t<arma::mat, int> &inModel)  function
// and kernel has no control action
void faust_t::myKernel(std::vector<arma::mat> x) {
  // Dimension of continuous variables x
  int dim = x.size() / 2;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  //  Create list of symbols to be used
  // These are a function of the number of continuous variables
  // in the model
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; j++) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }

  // TODO: add checks to see if symbolic Kernel has been constructed
  GiNaC::ex SymbKernel = this->Kernel;

  // Evaluate kernel based on input values
  // this is done by tree reversal of the Kernel
  // expresssion
  GiNaC::exmap m;
  arma::mat tempTp = arma::zeros<arma::mat>(x[0].n_rows, x[0].n_cols);
  for (unsigned int j = 0; j < x[0].n_rows; j++) {
    for (unsigned int k = 0; k < x[0].n_cols; k++) {
      int count = 0;
      for (int i = 0; i < 2 * dim; i = i + dim) {
        m[symbols[i]] = x[count](j, k);
        count++;
      }
      for (int i = 1; i < 2 * dim; i = i + dim) {
        m[symbols[i]] = x[count](j, k);
        count++;
      }
      GiNaC::ex evalTp = SymbKernel.subs(m);
      GiNaC::ex getTp = evalTp;
      std::vector<double> subsEx;
      for (size_t n = 0; n != getTp.nops(); ++n) {
        GiNaC::ex temp = getTp.op(n).evalm();
        if (temp.nops() == 0) {
          if (GiNaC::is_a<GiNaC::matrix>(temp)) {
            GiNaC::ex t = GiNaC::ex_to<GiNaC::matrix>(temp);
            temp = t;
          } else {
            temp = temp.evalf();
          }
          subsEx.push_back(GiNaC::ex_to<GiNaC::numeric>(temp).to_double());
        } else {
          for (size_t j = 0; j != temp.nops(); ++j) {
            GiNaC::matrix t;
            GiNaC::ex t3;
            if (GiNaC::is_a<GiNaC::matrix>(temp.op(j))) {
              t = GiNaC::ex_to<GiNaC::matrix>(temp.op(j));
              t3 = GiNaC::exp(t(0, 0)).evalf();
            } else {
              GiNaC::matrix t;
              if (GiNaC::is_a<GiNaC::matrix>(temp.op(j))) {
                t = GiNaC::ex_to<GiNaC::matrix>(temp.op(j).evalm());
                t3 = (t(0, 0));
              } else {
                GiNaC::ex t2 = temp.op(j).evalm();
                for (size_t k = 0; k != t2.nops(); ++k) {
                  t = GiNaC::ex_to<GiNaC::matrix>(t2.op(k));
                }
                t3 = t(0, 0);
              }
            }
            subsEx.push_back(GiNaC::ex_to<GiNaC::numeric>(t3).to_double());
          }
        }
      }
      double multi = subsEx[0];
      for (unsigned int q = 1; q <= subsEx.size(); q++) {
        multi *= subsEx[q];
      }

      tempTp(j, k) = multi;
    }
  }
  this->Tp.clear();
  this->Tp.push_back(tempTp);
}

// Evaluation of stochastic Kernel based on the input values
// of x
// Input:
//       std::vector<arma::mat> x: Vector of values for each continuous variables
// Assumes symbolic kernel was previously defined using
// faust_t::myKernel(shs_t<arma::mat, int> &inModel)  function
// and kernel has no control action
void faust_t::myKernel(std::vector<arma::mat> x, double constant) {
  // dimension of continuous variables
  int dim = x.size() / 2;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  //  Create list of symbols to be used
  // These are a function of the number of continuous variables
  // in the model
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; j++) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }

  // TODO add checks to see if symbolic Kernel has been constructed
  GiNaC::ex internal = this->KernelInternal;

  // Evaluate kernel based on input values
  // this is done by tree reversal of the Kernel
  // expresssion
  GiNaC::exmap m;
  arma::mat tempTp = arma::zeros<arma::mat>(x[0].n_rows, x[0].n_cols);
  for (unsigned int j = 0; j < x[0].n_rows; j++) {
    for (unsigned int k = 0; k < x[0].n_cols; k++) {
      int count = 0;
      for (int i = 0; i < 2 * dim; i = i + dim) {
        m[symbols[i]] = x[count](j, k);
        count++;
      }
      for (int i = 1; i < 2 * dim; i = i + dim) {
        m[symbols[i]] = x[count](j, k);
        count++;
      }
      GiNaC::ex evalTp = internal.subs(m);
      GiNaC::matrix ev = GiNaC::ex_to<GiNaC::matrix>(evalTp.evalm());
      double res = GiNaC::ex_to<GiNaC::numeric>(ev(0, 0)).to_double();
      tempTp(j, k) = constant * std::exp(-0.5 * res);
    }
  }
  this->Tp.clear();
  this->Tp.push_back(tempTp);
}

// Create and evaluate stochastic Kernel based on the input values
// of x and the structure of the input model
// Input:
//       std::vector<arma::mat> x: Vector of values for each continuous variables
//      shs_t<arma::mat, int> &inModel:  Reference to input model in form of
//       DTSHS
void faust_t::myKernel(std::vector<arma::mat> x,
                       shs_t<arma::mat, int> &inModel) {
  // Dimensions
  int dim = x.size() / 2;

  // Create transition probabilities temporary store
  arma::mat tempTp = arma::zeros<arma::mat>(x[0].n_rows, x[0].n_cols);

  // Define matrices of underlying state space model
  // which takes the form of
  // x_k+1 = Ax_k + Smw_k
  arma::mat A = inModel.x_mod[0].A;
  arma::mat Sm = inModel.x_mod[0].sigma;
  arma::mat Qm = inModel.x_mod[0].Q;
  arma::mat Xvec(dim, 1);
  arma::mat Xbvec(dim, 1);

  int count = 0;

  // Compute the constant value of the gaussian distribution
  double exter =
      1 / (std::sqrt(std::pow((2 * arma::datum::pi), dim) * arma::det(Sm)));

  // Map the values of the continuous variables
  for (unsigned int j = 0; j < x[0].n_rows; j++) {
    for (unsigned int k = 0; k < x[0].n_cols; k++) {
      for (int i = 0; i < dim; i++) {
        Xvec(i, 0) = x[i + count](j, k);
        Xbvec(i, 0) = x[i + 1 + count](j, k);
        count++;
      }

      // Compute mean value of guassian distribution (g.d.)
      arma::mat E_xbar = (A * Xvec + Qm) - Xbvec;
      arma::mat inter = E_xbar.t() * arma::inv(Sm) * E_xbar;
      double res = inter(0, 0);
      // Compute the value of g.d. for the current x values
      tempTp(j, k) = exter * std::exp(-0.5 * res);
      count = 0;
    }
  }

  this->Tp.clear();
  // Store transition probabilities
  this->Tp.push_back(tempTp);
}

// Create and evaluate stochastic Kernel based on the input values
// of x and the structure of the input model
// Input:
//       std::vector<arma::mat> x: Vector of values for each continuous variables
//      shs_t<arma::mat, int> &inModel:  Reference to input model in form of
//       DTSHS
//      dim_x/dim_u : dimension of the control and state space variables
void faust_t::myKernel(std::vector<arma::mat> x, shs_t<arma::mat, int> &inModel,
                       int dim, int dim_u) {
  // Dimensions
  size_t m = x[0].n_rows;
  size_t q = x[0].n_cols;

  // Initialise matrices to be used
  arma::mat tempTp = arma::zeros<arma::mat>(m,q);

  arma::mat A= inModel.x_mod[0].A;
  arma::mat B = inModel.x_mod[0].B;
  arma::mat Qm = inModel.x_mod[0].Q;
  arma::mat Sm= inModel.x_mod[0].sigma;
  arma::vec Xvec(dim);
  arma::vec Xbvec(dim);
  arma::vec Uvec(dim_u);

  int count = 0;

  // Get constant value of normal distribution
  double exter =
      1 / (std::sqrt(std::pow((2 * arma::datum::pi), dim) * arma::det(Sm)));

  for (size_t j = 0; j < m; j++) {
    for (size_t k = 0; k < q;  k++) {
      for (size_t i = 0; i < dim; i++) {
        double x1 = x[i + count](k, j);
	      double x1b = x[i + 1 + count](k, j);
        Xvec(i) = x1;
 	      Xbvec(i) = x1b;
	      count++;
      }
      int count_u = 0;
      for (size_t i = dim * 2; i < dim * 2 + dim_u; i++) {
        double u1 = x[i](k,j);
        Uvec(count_u) = u1;
	      count_u++;
      }
      // Compute mean of the gaussian distribution (g.d)
      arma::mat E_xbar = (A * Xvec + B * Uvec + Qm) - Xbvec;
      arma::mat inter = E_xbar.t() * arma::inv(Sm) * E_xbar;
      double res = inter(0, 0);
      // Evaluate g.d.
      tempTp(k,j) = exter * std::exp(-0.5 * res);
      count = 0;
    }
  }
  this->Tp.clear();
  // Store transition probabilities
  this->Tp.push_back(tempTp);
}

// Computation of the Lipschitz constant based on
// the input Kernel, with initial values defined
// by a vector x
static double myKernelLipsch(shs_t<arma::mat, int> &model,
                             const std::vector<double> &x) {
  // Dimension of x
  int dim = model.x_mod[0].A.n_cols;

  // Dimension of u
  int dim_u = model.x_mod[0].B.n_cols;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Definition of symbolic variables for "u"
  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};

  //  Create list of symbols to be used
  // These are a function of the number of continuous variables
  // in the model
  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }
  // Mapping of state space models into symbolic expressions
  // to create symbolic function of the kernel
  GiNaC::matrix A(dim, dim);
  GiNaC::matrix Sm(dim, dim);
  GiNaC::matrix B(dim, dim_u);
  GiNaC::matrix Qm(dim, 1);
  GiNaC::matrix Xvec(dim, 1);
  GiNaC::matrix Xbvec(dim, 1);
  GiNaC::matrix Uvec(dim_u, 1);
  int count = 0;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      A(i, j) = model.x_mod[0].A(i, j);
      Sm(i, j) = model.x_mod[0].sigma(i, j);
    }
    if(model.x_mod[0].Q.is_empty() == true) {
      Qm(i,0) = 0;
    }
    else {
      Qm(i,0) = model.x_mod[0].Q(i,0);
    }
    Xvec(i, 0) = symbols[i + count];
    Xbvec(i, 0) = symbols[i + 1 + count];
    count++;
  }
  for (int i = 0; i < dim_u; ++i) {
    for (int j = 0; j < dim; ++j) {
      B(j, i) = model.x_mod[0].B(j, i);
    }
    Uvec(i, 0) = symbols[2 * dim + i];
  }
  // Mapping the values of the continuous varaibales
  // into the symbolic equaivalent
  GiNaC::exmap mb;
  GiNaC::exmap mx;
  for (int i = 0; i < dim * 2; i = i + 2) {
    mb[symbols[i]] = x[i];
    mx[symbols[i + 1]] = x[i + 1];
  }
  GiNaC::exmap mu;
  for (int i = dim * 2; i < (dim * 2 + dim_u); i++) {
    mu[symbols[i]] = x[i];
  }
  // evaluatte kernel
  GiNaC::ex E_xbar;
  if (dim_u > 0) {
    E_xbar = A * Xvec + B * Uvec + Qm;
  } else {
    E_xbar = A * Xvec + Qm;
  }
  E_xbar = E_xbar.evalm();
  GiNaC::ex inner = (-Xbvec + E_xbar).evalm();
  GiNaC::matrix inner2 = GiNaC::ex_to<GiNaC::matrix>(inner);
  GiNaC::ex lexp = inner2.transpose();
  GiNaC::ex Sminv = Sm.inverse();
  GiNaC::ex Smdet = GiNaC::determinant(Sm);
  GiNaC::ex exter =
      1 /
      (GiNaC::sqrt(GiNaC::pow((2 * GiNaC::Pi).evalf(), dim).evalf() * Smdet))
          .evalf();
  GiNaC::ex mat = -0.5 * (lexp.evalm() * Sminv.evalm() * inner.evalm()).evalm();
  GiNaC::ex Kernel = exter * (GiNaC::exp(mat).evalf());
  Kernel = Kernel.evalf();

  std::vector<GiNaC::ex> diffV;
  std::vector<GiNaC::ex> subsEx;
  for (int i = 0; i < dim; i++) {
    GiNaC::ex P2;
    // Compute Kernel derivative numerically
    if (i == 0) {
      P2 = Kernel.diff(get_symbol(x_alpha[i]), 1);
    } else {
      P2 = Kernel.diff(get_symbol(x_alpha[i + dim - 1]), 1);
    }

    P2 = P2.subs(mb);
    P2 = P2.subs(mx);
    if (dim_u > 0) {
      P2 = P2.subs(mu);
    }
    for (size_t i = 0; i != P2.nops(); ++i) {
      GiNaC::ex temp = P2.op(i).evalm();
      if (temp.nops() == 0) {
        subsEx.push_back(temp);
      } else {
        for (size_t j = 0; j != temp.nops(); ++j) {
          GiNaC::ex t2 = temp.op(j).evalm();
          GiNaC::matrix t;
          if (GiNaC::is_a<GiNaC::matrix>(t2)) {
            t = GiNaC::ex_to<GiNaC::matrix>(t2);
            t2 = t(0, 0);
          } else {
            GiNaC::matrix t;
            for (size_t k = 0; k != t2.nops(); ++k) {
              t = GiNaC::ex_to<GiNaC::matrix>(t2.op(k));
            }
            t2 = GiNaC::exp(t(0, 0)).evalf();
          }
          subsEx.push_back(t2);
        }
      }
    }
    GiNaC::ex subsa = subsEx[0] * subsEx[1] * subsEx[2];
    GiNaC::ex P3 = GiNaC::pow(subsa.evalf(), 2).evalf();
    diffV.push_back(P3);
    subsEx.clear();
  }
  GiNaC::ex objExt = diffV[0];
  for (unsigned int j = 1; j < diffV.size(); j++) {
    objExt += (diffV[j]);
  }
  double obj = GiNaC::ex_to<GiNaC::numeric>(objExt).to_double();
  obj = std::sqrt(obj);
  return -obj;
}

// Function to evaluation the differential of the stochastic
// kernel which has been encoded symbolically.
// This is used to compute the lipschitz constants
// for the generation of the abstractions of the state space
static double evalDiffKernel(std::vector<GiNaC::ex> &P2,
                             const std::vector<double> &x) {
  // Number of continuous variables
  int dim = x.size() / 2;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  //  Create list of symbols to be used
  // These are a function of the number of continuous variables
  // in the model
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; ++j) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }
  // Map kernel symbols to actual data points from the vector x
  GiNaC::exmap mx;
  for (int i = 0; i < dim * 2; i = i + 2) {
    mx[symbols[i]] = x[i];
    mx[symbols[i + 1]] = x[i + 1];
  }
  // Evaluate differential of Kernel
  std::vector<GiNaC::ex> diffV;
  std::vector<GiNaC::ex> subsEx;
  for (int k = 0; k < dim; k++) {
    GiNaC::ex P3 = P2[k].subs(mx);
    for (size_t i = 0; i != P3.nops(); ++i) {
      GiNaC::ex temp = P3.op(i).evalm();
      if (temp.nops() == 0) {
        subsEx.push_back(temp);
      } else {
        for (size_t j = 0; j != temp.nops(); ++j) {
          GiNaC::ex t2 = temp.op(j).evalm();
          GiNaC::matrix t;
          if (GiNaC::is_a<GiNaC::matrix>(t2)) {
            t = GiNaC::ex_to<GiNaC::matrix>(t2);
            t2 = t(0, 0);
          } else {
            GiNaC::matrix t;
            for (size_t k = 0; k != t2.nops(); ++k) {
              t = GiNaC::ex_to<GiNaC::matrix>(t2.op(k));
            }
            t2 = GiNaC::exp(t(0, 0)).evalf();
          }
          subsEx.push_back(t2);
        }
      }
    }
    GiNaC::ex subsa = subsEx[0];
    for (unsigned j = 1; j < subsEx.size(); ++j) {
      subsa *= subsEx[j];
    }
    subsEx.clear();
    GiNaC::ex P4 = GiNaC::pow(subsa.evalf(), 2).evalf();
    diffV.push_back(P4);
  }
  GiNaC::ex objExt = diffV[0];
  for (unsigned int j = 1; j < diffV.size(); j++) {
    objExt += diffV[j];
  }

  GiNaC::ex objEx = GiNaC::sqrt(objExt).evalf();

  double obj = GiNaC::ex_to<GiNaC::numeric>(objEx).to_double();

  return obj;
}

// Function to compute the lipschitz constants
// with respect to the \bar{x}
double myKernelXbarLipsch(shs_t<arma::mat, int> &model,
                          const std::vector<double> &x) {
  // Dimension of x
  int dim = model.x_mod[0].x_dim;

  // Dimension of u
  int dim_u = model.x_mod[0].u_dim;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Definition of symbolic variables for "u"
  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};


  //  Create list of symbols to be used
  // These are a function of the number of continuous variables
  // in the model
  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }

  // creating symbolic kernel for gaussian distribution
  // with input
  GiNaC::matrix A(dim, dim);
  GiNaC::matrix Sm(dim, dim);
  GiNaC::matrix Qm(dim,1);
  GiNaC::matrix B(dim, dim_u);
  GiNaC::matrix Xvec(dim, 1);
  GiNaC::matrix Xbvec(dim, 1);
  GiNaC::matrix Uvec(dim_u, 1);
  int count = 0;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      A(i, j) = model.x_mod[0].A(i, j);
      Sm(i, j) = model.x_mod[0].sigma(i, j);
    }
    Xvec(i, 0) = symbols[i + count];
    Xbvec(i, 0) = symbols[i + 1 + count];
    if(model.x_mod[0].Q.is_empty()== true) {
      Qm(i,0) = 0;
    }
    else {
      Qm(i,0) =  model.x_mod[0].Q(i,0);
    }
    count++;
  }
  for (int i = 0; i < dim_u; ++i) {
    for (int j = 0; j < dim; ++j) {
      B(j, i) = model.x_mod[0].B(j, i);
    }
    Uvec(i, 0) = symbols[2 * dim + i];
  }
  GiNaC::exmap mb;
  GiNaC::exmap mx;
  for (int i = 0; i < dim * 2; i = i + 2) {
    mb[symbols[i]] = x[i];
    mx[symbols[i + 1]] = x[i + 1];
  }
  GiNaC::exmap mu;
  for (int i = dim * 2; i < (dim * 2 + dim_u); i++) {
    mu[symbols[i]] = x[i];
  }
  GiNaC::ex E_xbar;
  if (dim_u > 0) {
    E_xbar = A * Xvec + B * Uvec + Qm;
  } else {
    E_xbar = A * Xvec + Qm;
  }
  E_xbar = E_xbar.evalm();
  GiNaC::ex inner = (-Xbvec + E_xbar).evalm();
  GiNaC::matrix inner2 = GiNaC::ex_to<GiNaC::matrix>(inner);
  GiNaC::ex lexp = inner2.transpose();
  GiNaC::ex Sminv = Sm.inverse();
  GiNaC::ex Smdet = GiNaC::determinant(Sm);
  GiNaC::ex exter =
      1 /
      (GiNaC::sqrt(GiNaC::pow((2 * GiNaC::Pi).evalf(), dim).evalf() * Smdet))
          .evalf();
  GiNaC::ex mat = -0.5 * (lexp.evalm() * Sminv.evalm() * inner.evalm()).evalm();
  GiNaC::ex Kernel = exter * (GiNaC::exp(mat).evalf());
  Kernel = Kernel.evalf();

  // differentiate kernel wrt xbar and evaluate
  std::vector<GiNaC::ex> diffV;
  std::vector<GiNaC::ex> subsEx;
  for (int i = 0; i < dim; i++) {
    GiNaC::ex P2;
    if (i == 0) {
      P2 = Kernel.diff(get_symbol(x_alpha[i + 1]), 1);
    } else {
      P2 = Kernel.diff(get_symbol(x_alpha[i + dim]), 1);
    }

    P2 = P2.subs(mx);
    P2 = P2.subs(mb);

    if (dim_u > 0) {
      P2 = P2.subs(mu);
    }
    for (size_t i = 0; i != P2.nops(); ++i) {
      GiNaC::ex temp = P2.op(i).evalm();
      if (temp.nops() == 0) {
        subsEx.push_back(temp);
      } else {
        for (size_t j = 0; j != temp.nops(); ++j) {
          GiNaC::ex t2 = temp.op(j).evalm();
          GiNaC::matrix t;
          if (GiNaC::is_a<GiNaC::matrix>(t2)) {
            t = GiNaC::ex_to<GiNaC::matrix>(t2);
            t2 = t(0, 0);
          } else {
            GiNaC::matrix t;
            for (size_t k = 0; k != t2.nops(); ++k) {
              t = GiNaC::ex_to<GiNaC::matrix>(t2.op(k));
            }
            t2 = GiNaC::exp(t(0, 0)).evalf();
          }
          subsEx.push_back(t2);
        }
      }
    }
    GiNaC::ex subsa = subsEx[0] * subsEx[1] * subsEx[2];
    GiNaC::ex P3 = GiNaC::pow(subsa.evalf(), 2).evalf();
    diffV.push_back(P3);
    subsEx.clear();
  }
  GiNaC::ex objExt = diffV[0];
  for (unsigned int j = 1; j < diffV.size(); j++) {
    objExt += diffV[j];
  }

  double obj = GiNaC::ex_to<GiNaC::numeric>(objExt).to_double();
  obj  = std::sqrt(obj);

  return -obj;
}

// Function to compute the lipschitz constant
// with respect to the input control actions u
double myKernelULipsch(shs_t<arma::mat, int> &model,
                       const std::vector<double> &x) {
  // Dimension of x
  int dim = model.x_mod[0].A.n_rows;

  // Dimension of u
  int dim_u = model.x_mod[0].B.n_cols;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};
  // Definition of symbolic variables for "u"
  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};

  // Mappinf of variables to symbols
  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }

  GiNaC::matrix A(dim,dim);
  GiNaC::matrix Sm(dim,dim);
  GiNaC::matrix Qm(dim, 1);
  GiNaC::matrix B(dim,dim_u);
  GiNaC::matrix Xvec(dim, 1);
  GiNaC::matrix Xbvec(dim, 1);
  GiNaC::matrix Uvec(dim_u, 1);
  int count = 0;

  // Create symbolic mapping of kernel
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      A(i, j) = model.x_mod[0].A(i, j);
      Sm(i, j) = model.x_mod[0].sigma(i, j);
    }
    Xvec(i, 0) = symbols[i + count];
    Xbvec(i, 0) = symbols[i + 1 + count];
    if(model.x_mod[0].Q.is_empty()== true) {
      Qm(i,0) = 0;
    }
    else {
      Qm(i,0) =  model.x_mod[0].Q(i,0);
    }
    count++;
  }
  for (int i = 0; i < dim_u; ++i) {
    for (int j = 0; j < dim; ++j) {
      B(j, i) = model.x_mod[0].B(j, i);
    }
    Uvec(i, 0) = symbols[2 * dim + i];
  }
  GiNaC::exmap mb;
  GiNaC::exmap mx;
  for (int i = 0; i < dim * 2; i = i + 2) {
    mb[symbols[i]] = x[i];
    mx[symbols[i + 1]] = x[i + 1];
  }
  GiNaC::exmap mu;
  for (int i = dim * 2; i < (dim * 2 + dim_u); i++) {
    mu[symbols[i]] = x[i];
  }

  // create symbolic kernel
  GiNaC::ex E_xbar;
  E_xbar = A * Xvec + B * Uvec + Qm;
  E_xbar = E_xbar.evalm();
  GiNaC::ex inner = (-Xbvec + E_xbar).evalm();
  GiNaC::matrix inner2 = GiNaC::ex_to<GiNaC::matrix>(inner);

  GiNaC::ex lexp = inner2.transpose();
  GiNaC::ex Sminv = Sm.inverse();
  GiNaC::ex Smdet = GiNaC::determinant(Sm);

  GiNaC::ex exter =
      1 /
      (GiNaC::sqrt(GiNaC::pow((2 * GiNaC::Pi).evalf(), dim).evalf() * Smdet))
          .evalf();
  GiNaC::ex mat = -0.5 * (lexp.evalm() * Sminv.evalm() * inner.evalm()).evalm();
  GiNaC::ex Kernel = exter * (GiNaC::exp(mat).evalf());
  Kernel = Kernel.evalf();

  Kernel = Kernel.subs(mb);
  Kernel = Kernel.subs(mx);
  // compute derivative of kernel wrt u and evaluate
  std::vector<GiNaC::ex> diffV;
  std::vector<GiNaC::ex> subsEx;
  for (int i = 0; i < dim_u; i++) {
    GiNaC::ex P2;
    P2 = Kernel.diff(get_symbol(u_alpha[i]), 1);

    P2 = P2.subs(mu);

    for (size_t i = 0; i != P2.nops(); ++i) {
      GiNaC::ex temp = P2.op(i).evalm();
      if (temp.nops() == 0) {
        subsEx.push_back(temp);
      } else {
        for (size_t j = 0; j != temp.nops(); ++j) {
          GiNaC::ex t2 = temp.op(j).evalm();
          GiNaC::matrix t;
          if (GiNaC::is_a<GiNaC::matrix>(t2)) {
            t = GiNaC::ex_to<GiNaC::matrix>(t2);
            t2 = t(0, 0);
          } else {
            GiNaC::matrix t;
            for (size_t k = 0; k != t2.nops(); ++k) {
              t = GiNaC::ex_to<GiNaC::matrix>(t2.op(k));
            }
            t2 = GiNaC::exp(t(0, 0)).evalf();
          }
          subsEx.push_back(t2);
        }
      }
    }
    GiNaC::ex subsa = subsEx[0] * subsEx[1] * subsEx[2];
    GiNaC::ex P3 = GiNaC::pow(subsa.evalf(), 2).evalf();
    diffV.push_back(P3);
    subsEx.clear();
  }
  GiNaC::ex objExt = diffV[0];
  for (unsigned int j = 1; j < diffV.size(); j++) {
    objExt += diffV[j];
  }

  double obj = GiNaC::ex_to<GiNaC::numeric>(objExt).to_double();
  obj  = std::sqrt(obj);

  return -obj;
}

// Evaluate differential of kernel with respect
// to the control action u
static double evalDiffKernel_Contr(std::vector<GiNaC::ex> &P2,
                                   const std::vector<double> &x, int dim_u) {
  int dim = (x.size() - dim_u) / 2;

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Definition of symbolic variables for "u"
  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};

  // Mapping of variables to symbols
  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }

  // Mapping of values to symbols
  GiNaC::exmap mx;
  for (int i = 0; i < dim * 2; i = i + 2) {
    mx[symbols[i]] = x[i];
    mx[symbols[i + 1]] = x[i + 1];
  }
  for (int i = dim * 2; i < (dim * 2 + dim_u); i++) {
    mx[symbols[i]] = x[i];
  }

  // Evaluate differential of Kernel
  std::vector<GiNaC::ex> diffV;
  std::vector<GiNaC::ex> subsEx;
  for (unsigned k = 0; k < P2.size(); k++) {
    GiNaC::ex P3 = P2[k].subs(mx);
    for (size_t i = 0; i != P3.nops(); ++i) {
      GiNaC::ex temp = P3.op(i).evalm();
      if (temp.nops() == 0) {
        subsEx.push_back(temp);
      } else {
        for (size_t j = 0; j != temp.nops(); ++j) {
          GiNaC::ex t2 = temp.op(j).evalm();
          GiNaC::matrix t;
          if (GiNaC::is_a<GiNaC::matrix>(t2)) {
            t = GiNaC::ex_to<GiNaC::matrix>(t2);
            t2 = t(0, 0);
          } else {
            GiNaC::matrix t;
            for (size_t k = 0; k != t2.nops(); ++k) {
              t = GiNaC::ex_to<GiNaC::matrix>(t2.op(k));
            }
            t2 = GiNaC::exp(t(0, 0)).evalf();
          }
          subsEx.push_back(t2);
        }
      }
    }
    GiNaC::ex subsa = subsEx[0];
    for (unsigned j = 1; j < subsEx.size(); ++j) {
      subsa *= subsEx[j];
    }
    subsEx.clear();
    GiNaC::ex P4 = GiNaC::pow(subsa.evalf(), 2).evalf();
    diffV.push_back(P4);
  }
  GiNaC::ex objExt = diffV[0];
  for (unsigned int j = 1; j < diffV.size(); j++) {
    objExt += diffV[j];
  }

  GiNaC::ex objEx = GiNaC::sqrt(objExt).evalf();

  double obj = GiNaC::ex_to<GiNaC::numeric>(objEx).to_double();

  return obj;
}
// Objective function initialisation for minimising the
// Lipschitz constant wrt x
double myfunc(const std::vector<double> &x, std::vector<double> &grad,
              void *my_func_data) {
  auto *model = (shs_t<arma::mat, int> *)my_func_data;
  return myKernelLipsch(*model, x);
}
// Objective function initialisation for minimising the
// Lipschitz constant wrt \bar{x}
double myfuncXbar(const std::vector<double> &x, std::vector<double> &grad,
                  void *my_func_data) {
  auto *model = (shs_t<arma::mat, int> *)my_func_data;
  return myKernelXbarLipsch(*model, x);
}
// Objective function initialisation for minimising the
// Lipschitz constant wrt u
double myfuncU(const std::vector<double> &x, std::vector<double> &grad,
               void *my_func_data) {
  auto *model = (shs_t<arma::mat, int> *)my_func_data;

  return myKernelULipsch(*model, x);
}
// Objective function initialisation for minimising the
// differential of the kernel wrt x
double myfuncS2(const std::vector<double> &x, std::vector<double> &grad,
                void *my_func_data) {
  auto *model = (std::vector<GiNaC::ex> *)my_func_data;

  return -evalDiffKernel(*model, x);
}

// Objective function initialisation for minimising the
// differential of the kernel wrt u
double myfuncS2_Contr(const std::vector<double> &x, std::vector<double> &grad,
		void *my_func_data) {
  auto *model = (std::vector<GiNaC::ex> *)my_func_data;

  return -evalDiffKernel_Contr(*model,x,1);

}
// Function to compute the global Lipschitz value
// such that uniform gridding can be performed
// The lipschitz function is used to define
// how fine to split the state space given a specified
// maximal abstraction error
double faust_t::GlobalLipschitz(arma::mat SafeSet, bool Xbar) {
  // Dimension of the Set and thus the system
  int dim = SafeSet.n_rows;

  // Definition of the Kernel Function from the model
  double h = 0;

  // Global optimiser without need for defininig gradient
  nlopt::opt opt(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim * 2);
  nlopt::opt lopt(nlopt::LN_COBYLA, dim * 2);
  opt.set_local_optimizer(lopt);

  // Defining upper, lower bounds and starting point
  std::vector<double> lb(dim * 2);
  std::vector<double> ub(dim * 2);
  std::vector<double> x0(dim * 2);
  int count = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < dim; j++) {
      if (count == 0) {
        if (dim == 1) {
          lb[j] = SafeSet(0, 0);
          ub[j] = SafeSet(0, dim);
          x0[j] = arma::accu(SafeSet.row(0)) / 2;
        } else {
          lb[j] = SafeSet(count, 0);
          ub[j] = SafeSet(count, dim - 1);
          x0[j] = arma::accu(SafeSet.row(count)) / 2;
        }
      } else {
        if (dim == 1) {
          lb[j + dim] = SafeSet(0, 0);
          ub[j + dim] = SafeSet(0, dim);
          x0[j + dim] = arma::accu(SafeSet.row(0)) / 2;
        } else {
          lb[j + dim] = SafeSet(count, 0);
          ub[j + dim] = SafeSet(count, dim - 1);
          x0[j + dim] = arma::accu(SafeSet.row(count)) / 2;
        }
      }
    }
    count++;
  }

  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.remove_equality_constraints();
  opt.remove_inequality_constraints();
  opt.set_maxeval(10500);

  // Setting up the optimisation problem by and selecting
  // the objective function based on whether it needs to
  // minimised as a function of X or xbar
  if (Xbar) {
    opt.set_min_objective(myfuncXbar, (void *)&this->model);
  } else {
    opt.set_min_objective(myfunc, (void *)&this->model);
  }
  // Define tolerances
  opt.set_xtol_rel(1e-8);
  lopt.set_xtol_rel(1e-12);

  opt.set_ftol_abs(1e-20);
  lopt.set_ftol_abs(1e-22);
  double minf;
  // Perform minimisation using hcubature
  try {
    nlopt::result result = opt.optimize(x0, minf);
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    exit(0);
  }

  h = -minf;
  if (h < 1e-5) {
    std::cout << "The Lipschitz constant is very small. Please check if this "
                 "is according to expectations. If not, this can be solved by "
                 "adjusting the TolFun parameter"
              << std::endl;
  }

  return h;
}

// Numerically computes the local Lipschitz constants by taking the maximum
// local derivative over six different points. xl and xu denote the lower and
// upper bound from which cells the  Lipschitz constant is derived. This means
// that the output h_ij will
//  have dimensions [(xu-xl),m]. With m the cardinality of the partition defined
//  in X
arma::mat faust_t::LocalLipschitz(int xl, int xu) {

  arma::mat X = this->X;

  // Cardinality of the partition of the SafeSet
  int m = X.n_rows;
  // Dimension of the system
  int dim = X.n_cols / 2;

  // Initialisation
  arma::mat h = arma::zeros<arma::mat>((xu - xl) + 1, m);

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Mapping of variables to symbols
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; ++j) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }

  // Obtain symbolic Kernel and
  // compute its derivative
  Kernel = this->Kernel;
  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      te = Kernel.diff(get_symbol(x_alpha[i]), 1);
    } else {
      te = Kernel.diff(get_symbol(x_alpha[i + dim - 1]), 1);
    }
    P2.push_back(te);
  }

  // Selection matrix for k_ij
  arma::mat temp;
  temp << 0 << 1;
  arma::mat t2 = repmat(temp, 1, dim * 2);
  arma::mat t3 = nchoosek(t2, dim * 2);
  arma::mat Ct = unique_rows(t3);

  // Convert to matrix of decimal numbers and then sort
  arma::mat getIndex = binary_to_decimal(Ct);
  Ct = fliplr(Ct.rows(arma::stable_sort_index(getIndex)));
  arma::mat C4 = -0.5 * arma::ones<arma::mat>(Ct.n_rows, Ct.n_cols);
  arma::mat C = arma::zeros<arma::mat>(Ct.n_rows, Ct.n_cols);
  C = Ct + C4;
  int delta = std::ceil(m/C.n_rows)+1;
  // Get indices for adaptive grid
  arma::vec indices = arma::linspace(0,1,C.n_rows);
  for(int idx =1; idx < delta; ++idx) {
    indices = arma::join_vert(indices,arma::linspace(0,1,C.n_rows));
  }
  // Fast but memory expansive way of computing
  std::vector<std::vector<arma::mat>> P;
  std::vector<arma::mat> intP;

  // Creation of all the corner points to check (4-D array)
  for (size_t i = 0; i < dim; ++i) {
    for (size_t k = 0; k < m; ++k) {
      arma::mat tempP = X(arma::span(xl - 1, xu - 1), i) +
                        C(indices(k), i) * X(arma::span(xl - 1, xu - 1), dim + i);
      arma::mat tempP1 = arma::kron(tempP, arma::ones<arma::mat>(1, m));
      intP.push_back(tempP1);
    }
    P.push_back(intP);
    intP.clear();
  }
  for (int i = dim; i < 2 * dim; ++i) {
    for (int k = 0; k < m; ++k) {
      arma::mat tempP = X.col(i - dim).t() + X.col(i).t() * C(indices(k), i);
      arma::mat tempP1 =
          arma::kron(tempP, arma::ones<arma::mat>((int)(xu - xl + 1), 1));
      intP.push_back(tempP1);
    }
    P.push_back(intP);
    intP.clear();
  }

  // Creation of the local minima and maxima
  std::cout << "Creation of the local Lipschitz constants is in progress"
            << std::endl;

  // Evaluate differential of Kernel
  std::vector<double> x;
  arma::vec maxEl(m);
  arma::cube h_ij_aux(P[0][0].n_rows, P[0][0].n_cols, m);
  int index_r = 0, index_c = 0;
  for (unsigned int j = 0; j < m; ++j) {
    for (unsigned int n = 0; n < P[0][0].n_cols; ++n) {
      for (unsigned int nr = 0; nr < P[0][0].n_rows ;++nr) {
        for (int q = 0; q < dim * 2; q++) {
          x.push_back(P[q][j](nr, n));
        }
        h(nr,n) = evalDiffKernel(P2, x);
        x.clear();
      }

    }
    h_ij_aux.slice(j) = h;
  }

  return arma::max(h_ij_aux, 2);
}

// Numerically computes the local Lipschitz constants by taking the maximum
// local derivative over six different points. xl and xu denote the lower and
// upper bound from which cells the  Lipschitz constant is derived. This means
// that the output h_ij will
//  have dimensions [(xu-xl),m]. With m the cardinality of the partition defined
//  in X
arma::mat faust_t::LocalLipschitzXbar(int xl, int xu) {

  arma::mat X = this->X;
  // Cardinality of the partition of the SafeSet
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Initialisation
  arma::mat h = arma::zeros<arma::mat>((xu - xl) + 1, m);

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Mapping of symbols to variables
  GiNaC::lst symbols = {};
  for (int j = 1; j < 2 * dim; j = j + dim) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }
   // Get symbolic Kernel
  Kernel = this->Kernel;

  // Compute derivative of kernel
  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      te = Kernel.diff(get_symbol(x_alpha[i + dim]), 1);
    } else {
      te = Kernel.diff(get_symbol(x_alpha[i + dim]), 1);
    }

    P2.push_back(te);
  }

  // Selection matrix for k_ij
  arma::mat temp;
  temp << 0 << 1;
  arma::mat t2 = repmat(temp, 1, dim * 2);
  arma::mat t3 = nchoosek(t2, dim * 2);
  arma::mat Ct = unique_rows(t3);

  // Convert to matrix of decimal numbers and then sort
  arma::mat getIndex = binary_to_decimal(Ct);
  Ct = fliplr(Ct.rows(arma::stable_sort_index(getIndex)));
  arma::mat C4 = -0.5 * arma::ones<arma::mat>(Ct.n_rows, Ct.n_cols);
  arma::mat C = arma::zeros<arma::mat>(Ct.n_rows, Ct.n_cols);
  C = Ct + C4;
  int delta = std::ceil(m/C.n_rows)+1;

  // Get indices for adaptive grid
  arma::vec indices = arma::linspace(0,1,C.n_rows);
  for(int idx =1; idx < delta; ++idx) {
    indices = arma::join_vert(indices,arma::linspace(0,1,C.n_rows));
  }
  // Fast but memory expansive way of computing
  std::vector<std::vector<arma::mat>> P;
  std::vector<arma::mat> intP;

  // Creation of all the corner points to check (4-D array)
  for (size_t i = 0; i < dim; ++i) {
    for (size_t k = 0; k < m; ++k) {
      arma::mat tempP = X(arma::span(xl - 1, xu - 1), i) +
                        C(indices(k), i) * X(arma::span(xl - 1, xu - 1), dim + i);
      arma::mat tempP1 = arma::kron(tempP, arma::ones<arma::mat>(1, m));
      intP.push_back(tempP1);
    }
    P.push_back(intP);
    intP.clear();
  }
  for (int i = dim; i < 2 * dim; ++i) {
    for (int k = 0; k < m; ++k) {
      arma::mat tempP = X.col(i - dim).t() + X.col(i).t() * C(indices(k), i);
      arma::mat tempP1 =
          arma::kron(tempP, arma::ones<arma::mat>((int)(xu - xl + 1), 1));
      intP.push_back(tempP1);
    }
    P.push_back(intP);
    intP.clear();
  }

  // Creation of the local minima and maxima
  std::cout << "Creation of the local Lipschitz constants is in progress"
            << std::endl;

  // Evaluate differential of Kernel
  std::vector<double> x;
  arma::vec maxEl(pow(2, ((2 * dim))));
  arma::cube h_ij_aux(P[0][0].n_rows, P[0][0].n_cols, pow(2, ((2 * dim))));

  for (unsigned int j = 0; j < pow(2, ((2 * dim))); ++j) {
    for (unsigned int n = 0; n < P[0][0].n_cols; ++n) {
      for (unsigned int m = 0; m < P[0][0].n_rows; ++m) {
        for (int q = 0; q < dim * 2; q++) {
          x.push_back(P[q][j](m, n));
        }
        h(m, n) = evalDiffKernel(P2, x);
        x.clear();
      }
    }
    h_ij_aux.slice(j) = h;
  }

  return arma::max(h_ij_aux, 2);
}

// Numerically computes the semi-local Lipschitz constants
arma::mat faust_t::SemiLocalLipschitz(arma::mat SafeSet, int xl, int xu) {

  // Get X
  arma::mat X = this->X;

  // Cardinality of the partition of the SafeSet
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Adapt if xl and xu =0
  if (xl == 0)
    xl = 0;
  if (xu == 0)
    xu = m-1;

  // Initialisation
  arma::mat h_i = arma::zeros<arma::mat>((xu - xl)+1, 1);

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Map variables to symbols
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; j++) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }

  // Get symbolic Kernel
  Kernel = this->Kernel;

  // Compute derivative of Kernel
  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      te = Kernel.diff(get_symbol(x_alpha[i]), 1);
    } else {
      te = Kernel.diff(get_symbol(x_alpha[i + dim - 1]), 1);
    }

    P2.push_back(te);
  }

  // Global optimiser without need for defininig gradient
  nlopt::opt opt(nlopt::GN_ORIG_DIRECT, dim * 2);
  nlopt::opt lopt(nlopt::LN_COBYLA, dim * 2);
  opt.set_local_optimizer(lopt);
  opt.remove_equality_constraints();
  opt.remove_inequality_constraints();
  opt.set_maxeval(500);
  opt.set_xtol_rel(1e-6);
  lopt.set_xtol_rel(1e-10);

  opt.set_ftol_abs(1e-20);
  lopt.set_ftol_abs(1e-22);

  std::vector<double> lb(dim * 2);
  std::vector<double> ub(dim * 2);
  std::vector<double> x0(dim * 2);

  for (int i = xl; i < xu; ++i) {
    // Setting the initial condition, lower and upper bounds
    for (int j = 0; j < 2 * dim; ++j) {
      if (j < dim) {
        x0[j] = X(i, j);
        lb[j] = X(i, j) - 0.5 * X(i, j + dim);
        ub[j] = X(i, j) + 0.5 * X(i, j + dim);
      } else {
        x0[j] = (SafeSet(j - dim, 1) + SafeSet(j - dim, 0)) / 2;
        lb[j] = SafeSet(j - dim, 0);
        ub[j] = SafeSet(j - dim, 1);
      }
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    // Set optimisation function
    opt.set_min_objective(myfuncS2, (void *)&P2);
    double minf;
    try {
      nlopt::result result = opt.optimize(x0, minf);
      // std::cout << "found minimum: "  << std::setprecision(10) << minf <<
      // std::endl;
      h_i(i-xl+1, 0) = -minf;
    } catch (std::exception &e) {
      std::cout << "nlopt failed: " << e.what() << std::endl;
      exit(0);
    }
  }
  // catch singular points where h_i=0
  arma::uvec zz = find(h_i < 0.00001);
  int i2 = 0;
  if (arma::accu(zz) > 0) {
    for (unsigned k = 0; k < zz.n_elem; k++) {
      i2 = k+xl-1;
      // Setting the initial condition, lower and upper bounds
      for (int j = 0; j < 2 * dim; j++) {
        if (j < dim) {
          x0[j] = X(i2, j) - 0.3 * X(i2, j + dim);
          lb[j] = X(i2, j) - 0.5 * X(i2, j + dim);
          ub[j] = X(i2, j) + 0.5 * X(i2, j + dim);
        } else {
          x0[j] = (SafeSet(j - dim, 1) + SafeSet(j - dim, 0)) / 2;
          lb[j] = SafeSet(j - dim, 0);
          ub[j] = SafeSet(j - dim, 1);
        }
      }
      opt.set_lower_bounds(lb);
      opt.set_upper_bounds(ub);

      // Set optimisation function
      opt.set_min_objective(myfuncS2, (void *)&P2);
      double minf;
      try {
        nlopt::result result = opt.optimize(x0, minf);
        h_i(zz(k), 0) = -minf;
      } catch (std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
        exit(0);
      }
    }
  }
  return h_i;
}

// Numerically computes the semi-local Lipschitz constants in presence of
// control actions in the kernel
arma::mat faust_t::SemiLocalLipschitz_Contr(arma::mat SafeSet, int xl, int xu, int ul, int uu) {

  // Get X
  arma::mat X = this->X;

  // Cardinality of the partition of the SafeSet
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Get U
  arma::mat U = this->U;

  // Cardinality of the Input
  int q = U.n_rows;

  // Dimension of the input
  int dim_u = U.n_cols/2;

  // Adapt if xl and xu =0
  if (xl == 0)
    xl = 0;
  if (xu == 0)
    xu = m-1;

  // Initialisation
  arma::mat h_i = arma::zeros<arma::mat>((xu - xl)+1, (uu-ul)+1);

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Map symbols
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; j++) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }

  // Obtain symbolic Kernel
  Kernel = this->Kernel;

  // Compute derivative of Kernel
  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      te = Kernel.diff(get_symbol(x_alpha[i]), 1);
    } else {
      te = Kernel.diff(get_symbol(x_alpha[i + dim - 1]), 1);
    }

    P2.push_back(te);
  }

  // Global optimiser without need for defininig gradient
  nlopt::opt opt(nlopt::GN_ORIG_DIRECT, dim * 2);
  nlopt::opt lopt(nlopt::LN_COBYLA, dim * 2);
  opt.set_local_optimizer(lopt);
  opt.remove_equality_constraints();
  opt.remove_inequality_constraints();
  opt.set_maxeval(500);
  opt.set_xtol_rel(1e-6);
  lopt.set_xtol_rel(1e-10);

  opt.set_ftol_abs(1e-20);
  lopt.set_ftol_abs(1e-22);

  std::vector<double> lb(dim * 2);
  std::vector<double> ub(dim * 2);
  std::vector<double> x0(dim * 2);

  for (int i = xl; i < xu; ++i) {
    for(int r = ul; r < uu; ++r) {
      // Setting the initial condition, lower and upper bounds
      for (int j = 0; j < 2 * dim; ++j) {
        if (j < dim) {
          x0[j] = X(i, j);
          lb[j] = X(i, j) - 0.5 * X(i, j + dim);
          ub[j] = X(i, j) + 0.5 * X(i, j + dim);
        } else {
          x0[j] = (SafeSet(j - dim, 1) + SafeSet(j - dim, 0)) / 2;
          lb[j] = SafeSet(j - dim, 0);
          ub[j] = SafeSet(j - dim, 1);
        }
      }
      opt.set_lower_bounds(lb);
      opt.set_upper_bounds(ub);

      // Set optimisation function
      opt.set_min_objective(myfuncS2, (void *)&P2);
      double minf;
      try {
        nlopt::result result = opt.optimize(x0, minf);
        h_i(i-xl+1,r-ul+1) = -minf;
      } catch (std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
        exit(0);
      }
    }
  }
  return h_i;
}

// Numerically computes the semi-local Lipschitz constants  wrt to
// the control action u in presence of
// control actions in the kernel
arma::mat faust_t::SemiLocalLipschitzToU_Contr(arma::mat SafeSet, int xl, int xu, int ul, int uu) {

  // Get X
  arma::mat X = this->X;

  // Cardinality of the partition of the SafeSet
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Get U
  arma::mat U = this->U;

  // Cardinality of the Input
  int q = U.n_rows;

  // Dimension of the input
  int dim_u = U.n_cols/2;

  // Adapt if xl and xu =0
  if (xl == 0)
    xl = 0;
  if (xu == 0)
    xu = m-1;

  // Initialisation
  arma::mat h_i = arma::zeros<arma::mat>((xu - xl)+1, (uu-ul)+1);

  // Definition of symbolic variables for "x"
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  // Map symbols
  GiNaC::lst symbols = {};
  for (int j = 0; j < 2 * dim; j++) {
    GiNaC::symbol t = get_symbol(x_alpha[j]);
    symbols.append(t);
  }

  // Get symbolic Kernel
  Kernel = this->Kernel;

  // Compute derivative of Kernel
  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      te = Kernel.diff(get_symbol(x_alpha[i]), 1);
    } else {
      te = Kernel.diff(get_symbol(x_alpha[i + dim - 1]), 1);
    }

    P2.push_back(te);
  }

  // Global optimiser without need for defininig gradient
  nlopt::opt opt(nlopt::GN_ORIG_DIRECT, dim * 2);
  nlopt::opt lopt(nlopt::LN_COBYLA, dim * 2);
  opt.set_local_optimizer(lopt);
  opt.remove_equality_constraints();
  opt.remove_inequality_constraints();
  opt.set_maxeval(500);
  opt.set_xtol_rel(1e-6);
  lopt.set_xtol_rel(1e-10);

  opt.set_ftol_abs(1e-20);
  lopt.set_ftol_abs(1e-22);

  std::vector<double> lb(dim * 2);
  std::vector<double> ub(dim * 2);
  std::vector<double> x0(dim * 2);

  for (int i = xl; i < xu; ++i) {
    for(int r = ul; r < uu; ++r) {
      // Setting the initial condition, lower and upper bounds
      for (int j = 0; j < 2 * dim; ++j) {
        if (j < dim) {
          x0[j] = X(i, j);
          lb[j] = X(i, j) - 0.5 * X(i, j + dim);
          ub[j] = X(i, j) + 0.5 * X(i, j + dim);
        } else {
          x0[j] = (SafeSet(j - dim, 1) + SafeSet(j - dim, 0)) / 2;
          lb[j] = SafeSet(j - dim, 0);
          ub[j] = SafeSet(j - dim, 1);
        }
      }
      opt.set_lower_bounds(lb);
      opt.set_upper_bounds(ub);

      // Set optimisation function
      opt.set_min_objective(myfuncS2_Contr, (void *)&P2);
      double minf;
      try {
        nlopt::result result = opt.optimize(x0, minf);
        std::cout << "found minimum: "  << std::setprecision(10) << minf <<
        std::endl;
        h_i(i-xl+1,r-ul+1) = -minf;
      } catch (std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
        exit(0);
      }
    }
  }
  return h_i;
}
// Function to create uniform grid
// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*h*N*L(A). Here L(A) denotes the Lebesgue
// measure of the set A, N the number of time steps, h  is the global Lipschitz
// constant and delta the dimension of the cells. Input of this function is the
// desired maxmial error epsilon, the model from which the KernelFunction is
// constructed the number of time steps T and the upper and  lower bounds of the
// (Safe)Set. The output X is a matrix, which consists of the centres of the
// cell as well  as the length of the edges. This is stored in the object faust
void faust_t::Uniform_grid(double epsilon, double T, arma::mat SafeSet) {

  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // Derivation of the global Lipschitz constant
  double h = this->GlobalLipschitz(SafeSet, 0);

  // The length of the edges of the partitions
  arma::mat temp = arma::prod(r);
  double t0 = temp(0, 0);
  double delta = (epsilon / (t0 * h * T)) / sqrt(dim);

  // Adjust delta if it is very large compared to r
  arma::mat a = arma::prod(ceil(r / delta));
  double a1 = a(0, 0);
  double b = (std::pow(2, dim) + 10);
  while (a1 < b) // appr number of bins
  {
    delta =
        (delta /
         (std::pow(
             2, (1 / dim)))); // This will cause the number of cells to double
    a = arma::prod(ceil(r / delta));
    a1 = a(0, 0);
    b = (std::pow(2, dim) + 10);
  }

  // Warn if a dimension of the set has not been partitioned.
  // This implies that in this direction no gradient will be found
  unsigned int temp2 = arma::accu((r - delta) < 0);

  if (temp2 > 0) {
    std::cout << "A dimension of the set has not been partitioned, reshaping "
                 "the set might solve this issue"
              << std::endl;
  }

  // Create the location of the representative points
  // delta is adjusted so that the edges of the set are part of the partitions
  // Create grid
  std::vector<arma::mat> C;
  arma::mat delta_adj = arma::zeros<arma::mat>(1, dim);
  for (int i = 0; i < dim; i++) {
    delta_adj.col(i) = (SafeSet(i, 1) - SafeSet(i, 0)) /
                       std::ceil((SafeSet(i, 1) - SafeSet(i, 0)) / delta);
    double start = SafeSet(i, 0) + 0.5 * delta_adj(i);
    double Delta = delta_adj(i);
    double end = SafeSet(i, 1) - 0.5 * delta_adj(i);

    arma::vec A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }
  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  int m = C[0].n_elem;
  // Make the output
  arma::mat X = arma::zeros<arma::mat>(m, dim);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < dim; ++j) {
      C[j].reshape(m, 1);
      X(i, j) = C[j](i, 0);
    }
  }
  arma::mat Y = arma::kron(delta_adj, arma::ones<arma::mat>(m, 1));

  // The error made
  arma::mat pR = arma::prod(r);

  double right = pR(0, 0) * h * T;
  double left = arma::accu(pow(delta_adj, 2));
  double E = sqrt(left) * right;

  this->X = join_horiz(X, Y);
  this->E = E;
}

// Uniform Markov Chain approximation
// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*(h+hbar)*N*L(A). Here L(A) denotes the
// Lebesgue measure of the set A, T the number of time steps, h is the global
// Lipschitz constant with respect to x, while hbar is the global Lipschitz
// constant with respect to xbar. delta is measure of the dimension of the
// cells. Input of this function is the desired maximal error epsilon, the model
// used to describe the Kernel, the number of time steps N and the upper and
// lower bounds of the (Safe)Set. The output X is a matrix, which consists of
// the centres of the cell as well  as the length of the edges.
void faust_t::Uniform_grid_MCapprox(double epsilon, double T,
                                    arma::mat SafeSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // Derivation of the global Lipschitz constants
  double h = this->GlobalLipschitz(SafeSet, 0);
  double hbar = this->GlobalLipschitz(SafeSet, 1);

  // The length of the edges of the partition
  double hnew = h + hbar;
  arma::mat temp = arma::prod(r);
  double t0 = temp(0, 0);
  double delta = (epsilon / (t0 * hnew * T)) / sqrt(dim);

  // Adjust delta if it is very large compared to r
  arma::mat a = arma::prod(ceil(r / delta));
  double a1 = a(0, 0);
  double b = (std::pow(2, dim) + 10);
  while (a1 < b) // appr number of bins
  {
    delta = delta /
            (std::pow(
                2, (1 / dim))); // This will cause the number of cells to double
    a = arma::prod(ceil(r / delta));
    a1 = a(0, 0);
    b = (std::pow(2, dim) + 10);
  }

  // Warn if a dimension of the set has not been partitioned.
  // This implies that in this direction no gradient will be found
  double temp2 = arma::accu((r - delta) < 0);

  if (temp2 > 0) {
    std::cout << "A dimension of the set has not been partitioned, reshaping "
                 "the set might solve this issue"
              << std::endl;
  }

  // Create the location of the representative points
  // delta is adjusted so that the edges of the set are part of the partitions
  std::vector<arma::mat> C;
  arma::mat delta_adj = arma::zeros<arma::mat>(1, dim);
  for (int i = 0; i < dim; i++) {
    delta_adj.col(i) = (SafeSet(i, 1) - SafeSet(i, 0)) /
                       std::ceil((SafeSet(i, 1) - SafeSet(i, 0)) / delta);
    double start = SafeSet(i, 0) + 0.5 * delta_adj(i);
    double Delta = delta_adj(i);
    double end = SafeSet(i, 1) - 0.5 * delta_adj(i);
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }
  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  int m = C[0].n_elem;
  // Make the output
  arma::mat X = arma::zeros<arma::mat>(m, dim);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < dim; ++j) {
      C[j].reshape(m, 1);
      X(i, j) = C[j](i, 0);
    }
  }
  arma::mat Y = arma::kron(delta_adj, arma::ones<arma::mat>(m, 1));

  // The error made
  arma::mat pR = arma::prod(r);
  double right = pR(0, 0) * hnew * T;
  double left = arma::accu(pow(delta_adj, 2));
  double E = sqrt(left) * right;

  this->X = join_horiz(X, Y);
  this->E = E;
}

// Uniform grid for ReachAvoid
// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*h*N*L(A). Here L(A) denotes the Lebesgue
// measure of the set A, N the number of time steps, h is the global Lipschitz
// constant and delta the dimension of the cells. Input of this function is the
// desired maxmial error epsilon, the  model to describe the Kernel, the number
// of time steps T and the upper and lower bounds of the SafeSet and the
// TargetSet.  The output X is a matrix, which consists of the centres of the
// cell as well as the length of the edges.
void faust_t::Uniform_grid_ReachAvoid(double epsilon, double T,
                                      arma::mat SafeSet, arma::mat TargetSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Target Set
  arma::mat rT = TargetSet.col(1) - TargetSet.col(0);

  // The length of the edges of the Set
  arma::mat rS = SafeSet.col(1) - SafeSet.col(0);

  // Derivation of the global Lipschitz constants
  double h = this->GlobalLipschitz(SafeSet, 0); // over approximation

  // The length of the edges of the partitions
  arma::mat temp = arma::prod(rS);
  double t0 = temp(0, 0);
  double delta = (epsilon / (t0 * h * T)) / sqrt(dim);

  // Adjust delta if it is very large compared to r
  arma::mat a = arma::prod(ceil(rS / delta));
  double a1 = a(0, 0);
  double b = (std::pow(2, dim) + 10);
  while (a1 < b) // appr number of bins
  {
    delta = delta /
            (std::pow(
                2, (1 / dim))); // This will cause the number of cells to double
    arma::mat a = arma::prod(ceil(rS / delta));
    a1 = a(0, 0);
    b = (std::pow(2, dim) + 10);
  }

  // delta is adjusted so that the edges of the TargetSet are part of the
  // partitions
  arma::mat delta_adj = arma::zeros<arma::mat>(SafeSet.n_rows, 1);
  for (int i = 0; i < dim; i++) {
    double diffT = (TargetSet(i, 1) - TargetSet(i, 0));
    double inner = diffT / delta;
    delta_adj.row(i) = diffT / std::ceil(inner);
  }
  // The adjusted SafeSet to fit with delta_adj
  arma::mat Residuals = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  arma::mat tempR = arma::repmat(delta_adj, 1, 2);
  arma::mat tempR2 = join_horiz(-delta_adj, 0 * delta_adj);


  arma::mat diff = SafeSet - TargetSet;
  Residuals = (diff - arma::floor(diff/tempR)%tempR) + tempR2;

  arma::mat SafeSet_adj = SafeSet - Residuals;

  // Partition (Safe-Target) into 3^dim-1 separate parts
  arma::mat D = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  arma::mat Dt = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  std::vector<arma::mat> D1;
  std::vector<arma::mat> Dt1;
  for (int i = 0; i < dim; i++) {
    arma::mat tempD;
    tempD << SafeSet(i, 0);
    tempD = join_horiz(tempD, SafeSet_adj.row(i)); // concatenate
    arma::mat tempDD;
    tempDD << SafeSet(i, 1);
    arma::mat tD = join_horiz(tempD, tempDD);
    D = unique(tD);
    Dt = D.cols(0, D.n_cols - 2);
    arma::vec tempV = D.t(); // convert to vector
    D1.push_back(tempV);
    arma::vec tempVt = Dt.t();
    Dt1.push_back(tempVt);
  }
  std::vector<arma::mat> F;

  F = nDgrid(Dt1);

    // Cardinality
  double n = F[0].n_rows;
  // Make the output
  arma::mat X = arma::zeros<arma::mat>(0, dim * 2); // Initialisation of X with an empty matrix
  for (int i = 0; i < n; i++) {
    arma::mat Set = arma::zeros<arma::mat>(dim, 2);
    std::vector<arma::mat> C;
    arma::mat delta_adj_local = arma::zeros<arma::mat>(1, dim);
    int index = 0;
    for (int j = 0; j < dim; j++) {
      double left = F[j](i, 0);
      for (unsigned k = 0; k < D1[j].n_rows; k++) {
        if (D1[j](k, 0) == F[j](i, 0)) {
          index = k;
        }
      }
      double right = D1[j](index + 1, 0);
      arma::mat templ;
      templ << left;
      arma::mat tempr;
      tempr << right;
      Set.row(j) = join_horiz(templ, tempr);

      // The ndiv variable eliminates rounding errors
      double ndiv = round((Set(j, 1) - Set(j, 0)) / delta_adj(j));
      if (ndiv == 0) {
        ndiv = 1;
      }
      delta_adj_local(j) = (Set(j, 1) - Set(j, 0)) / ndiv;

      double start = Set(j, 0) + 0.5 * delta_adj_local(j);
      double Delta = delta_adj_local(j);
      double end = Set(j, 1) - 0.4999 * delta_adj_local(j);
      arma::mat A = arma::regspace(start, Delta, end);
      C.push_back(A);
      // 0.499 adjusts for rounding errors; it has no influence on the
      // position of the representative points
    }
    // Construct grid
    C = nDgrid(C);

    // Cardinality
    double m = C[0].n_rows;
    // Make the output
    arma::mat Y = arma::zeros<arma::mat>(m, dim);
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < dim; k++) {
        Y(j, k) = C[k](j, 0);
      }
    }

    arma::mat J = arma::kron(delta_adj, arma::ones<arma::mat>(Y.n_rows, 1));
    int mid = J.n_rows/2;
    arma::mat st = J.rows(0,mid-1);
    arma::mat Res1 = join_horiz(Y, J.rows(0,mid-1));
    Res1 = join_horiz(Res1,J.rows(mid,J.n_rows-1));

    if (X.n_cols != Res1.n_cols) {
      X = Res1;
    } else {
      X = arma::join_vert(X, Res1);
    }
  }

  // The error made
  arma::mat pR = arma::prod(rS);
  double right = pR(0, 0) * h * T;
  double left = arma::accu(pow(delta_adj, 2));
  double E = sqrt(left) * right;
  this->X = X;
  this->E = E;
}

// UNIFORM_GRID_ReachAvoid_MCapprox This function makes a uniform grid that will
// produce a certain maximal abstraction error defined by E=delta*h*N*L(A). Here
// L(A) denotes the Lebesgue measure of the set A, T the number of time steps, h
// is the global Lipschitz constant and delta the dimension of the cells. Input
// of this function is the desired maxmial error epsilon, the  model from which
// the Kernel Function is obtained  the number of time steps T and the upper and
//  lower bounds of the SafeSet and the TargetSet.  The output X is a matrix,
//  which consists of the centres of the cell as well  as the length of the
//  edges.
void faust_t::Uniform_grid_ReachAvoid_MCapprox(double epsilon, double T,
                                               arma::mat SafeSet,
                                               arma::mat TargetSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // Check if Target set is outside Safeset
  arma::umat greater = TargetSet.col(dim - 1) > SafeSet.col(dim - 1);
  double isgreater = arma::accu(greater);
  if (isgreater > 0) {
    std::cout << "Target set outside Safe Set" << std::endl;
    exit(0);
  }

  // The length of the edges of the Target Set
  arma::mat rT = TargetSet.col(1) - TargetSet.col(0);

  // The length of the edges of the Set
  arma::mat rS = SafeSet.col(1) - SafeSet.col(0);

  // Derivation of the global Lipschitz constants
  double h = this->GlobalLipschitz(SafeSet, 0); // over approximation
  double hbar = this->GlobalLipschitz(SafeSet, 1);

  // The length of the edges of the partitions
  arma::mat temp = arma::prod(rS);
  double t0 = temp(0, 0);
  double hnew = h + hbar;
  double delta = (epsilon / (t0 * hnew * T)) / sqrt(dim);

  // Adjust delta if it is very large compared to r
  arma::mat a = arma::prod(ceil(rS / delta));
  double a1 = a(0, 0);
  double b = (std::pow(2, dim) + 10);
  while (a1 < b) // appr number of bins
  {
    delta = delta / (std::pow(2, (1 / dim))); // This will cause the number of cells to double
    a = arma::prod(ceil(rS / delta));
    a1 = a(0, 0);
    b = (std::pow(2, dim) + 10);
  }

  // delta is adjusted so that the edges of the TargetSet are part of the
  // partitions
  arma::mat delta_adj = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  for (int i = 0; i < dim; i++) {
    double diffT = (TargetSet(i, 1) - TargetSet(i, 0));
    double inner = diffT / delta;
    delta_adj.col(i) = diffT / std::ceil(inner);
  }
  // The adjusted SafeSet to fit with delta_adj
  arma::mat Residuals = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  arma::mat tempR = arma::repmat(delta_adj, 1, 2);
  arma::mat tempR2 = join_horiz(-delta_adj, 0 * delta_adj);

  arma::mat diff = SafeSet - TargetSet;

  Residuals = (diff - arma::floor(diff/tempR)%tempR) + tempR2;
  arma::mat SafeSet_adj = SafeSet - Residuals;

  // Partition (Safe-Target) into 3^dim-1 separate parts
  arma::mat D = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  arma::mat Dt = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  std::vector<arma::mat> D1;
  std::vector<arma::mat> Dt1;
  for (int i = 0; i < dim; i++) {
    arma::mat tempD;
    tempD << SafeSet(i, 0);
    tempD = join_horiz(tempD, SafeSet_adj.row(i)); // concatenate
    arma::mat tempDD;
    tempDD << SafeSet(i, 1);
    tempD = join_horiz(tempD, tempDD);
    D = unique(tempD);
    Dt = D.cols(0, D.n_cols - 2);
    D1.push_back(D.t());
    Dt1.push_back(Dt.t());
  }
  std::vector<arma::mat> F;
  F = nDgrid(Dt1);

  // Cardinality
  double n = F[0].n_elem;

  // Make the output
  arma::mat X = arma::zeros<arma::mat>(0, dim * 2); // Initialisation of X with an empty matrix
  for (int i = 0; i < n; i++) {
    arma::mat Set = arma::zeros<arma::mat>(dim, 2);
    std::vector<arma::mat> C;
    arma::mat delta_adj_local = arma::zeros<arma::mat>(1, dim);
    int index = 0;
    for (int j = 0; j < dim; j++) {
      double left = F[j](i);
      for (unsigned k = 0; k < D1[j].n_rows; k++) {
         if (D1[j](k) == left) {
           index = k;
         }
      }
      double right = D1[j](index + 1);
      arma::mat templ;
      templ << left;
      arma::mat tempr;
      tempr << right;
      Set.row(j) = join_horiz(templ, tempr);

      // The ndiv variable eliminates rounding errors
      double ndiv = round((Set(j, 1) - Set(j, 0)) / delta_adj(j));
      if (ndiv == 0) {
        ndiv = 1;
      }
      delta_adj_local(j) = (Set(j, 1) - Set(j, 0)) / ndiv;
      double start = Set(j, 0) + 0.5 * delta_adj_local(j);
      double Delta = delta_adj_local(j);
      double end = Set(j, 1) - 0.4999 * delta_adj_local(j);
      arma::mat A = arma::regspace(start, Delta, end);
      C.push_back(A);
      // 0.499 adjusts for rounding errors; it has no influence on the
      // position of the representative points
    }

    // Construct grid
    C = nDgrid(C);

    // Cardinality
    double m = C[0].n_elem;
    int sm = std::sqrt(m);

    // Make the output
    int index2 = 0;
    int k = 0;
    arma::mat Y = arma::zeros<arma::mat>(m, dim);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < dim; j++) {
        if (C[j].n_cols == 1) {
          Y(i, j) = C[j](i, 0);
        } else if (C[j].n_rows == 1) {
          Y(i, j) = C[j](0, i);
        } else {
          Y(i, j) = C[j](k, index2);
        }
      }
      if (k >= (sm - 1)) {
        k = 0;
        index2++;
      } else {
        k++;
      }
    }
    arma::mat J = arma::kron(delta_adj_local, arma::ones<arma::mat>(m, 1));
    arma::mat Res1 = join_horiz(Y, J);

    if (X.n_cols != Res1.n_cols) {
      X = Res1;
    } else {
      X = arma::join_vert(X, Res1);
    }
  }
  // The error made
  arma::mat pR = arma::prod(rS);
  double right = pR(0, 0) * hnew * T;
  double left = arma::accu(pow(delta_adj, 2));
  double E = sqrt(left) * right;

  this->X = X;
  this->E = E;
}

// Creates the Markov Chain that corresponds to the representative points
// contained in X.   X is a matrix input which consists of the centres of the
// cell as well
//   as the length of the edges.  The output Tp is a m by m matrix, where m
//   equals the cardinality of the partition defined in X.
void faust_t::MCapprox(double epsilon) {

  arma::mat X = this->X;
  // Cardinality
  int m = X.n_rows;

  // Dimension of the system
  int dim = (X.n_cols) / 2;

  // Calculation of the point-wise transition probabilities
  std::vector<arma::mat> p;
  for (int i = 0; i < dim; i++) {
    arma::mat tempP = arma::kron(X.col(i), arma::ones<arma::mat>(1, m));
    //  std::cout << "tempP: " << tempP << std::endl;
    p.push_back(tempP);
  }
  for (int j = dim; j < 2 * dim; j++) {
    //  std::cout << X.col(j-dim).t() << std::endl;
    arma::mat tempP2 =
        arma::kron(X.col(j - dim).t(), arma::ones<arma::mat>(m, 1));
    //  std::cout << "tempP2: " << tempP2 << std::endl;
    p.push_back(tempP2);
  }
  //	std::cout << "Creation of the Markov Chain in Progress" << std::endl;

  //	//  Make Tp transition probabilities by multiplying with the Lebesque
  //measure 	std::cout << "Kernel Function: " << this->Kernel << std::endl;
  //	// std::cout << "Kernel internal: " << this->KernelInternal <<std::endl;
  //	std::cout << "Kernel outside: " << this->constant << std::endl;
  this->myKernel(p, this->constant);
  arma::mat Tp = this->Tp[0];
  // std::cout << "Tp : " << Tp << std::endl;
  arma::mat r = arma::prod(X.cols(dim, 2 * dim - 1).t());
  arma::mat inright = arma::kron(r, arma::ones<arma::mat>(m, 1));
  Tp = Tp % inright;

  this->Tp[0] = Tp;
}

// Creates the Markov Chain that corresponds to the representative points
// contained in X.   X is a matrix input which consists of the centres of the
// cell as well
//   as the length of the edges.  The output Tp is a m by m matrix, where m
//   equals the cardinality of the partition defined in X.
void faust_t::MCapprox(double epsilon, shs_t<arma::mat, int> &model) {

  arma::mat X = this->X;
  // Cardinality
  int m = X.n_rows;

  // Dimension of the system
  int dim = (X.n_cols) / 2;

  // Calculation of the point-wise transition probabilities
  std::vector<arma::mat> p;
  for (int i = 0; i < dim; i++) {
    arma::mat tempP = arma::kron(X.col(i), arma::ones<arma::mat>(1, m));
    p.push_back(tempP);
  }
  for (int j = dim; j < 2 * dim; j++) {
    //  std::cout << X.col(j-dim).t() << std::endl;
    arma::mat tempP2 =
        arma::kron(X.col(j - dim).t(), arma::ones<arma::mat>(m, 1));
    p.push_back(tempP2);
  }
  std::cout << "Creation of the Markov Chain in Progress" << std::endl;

  //  Make Tp transition probabilities by multiplying with the Lebesque measure
  // std::cout << "Kernel Function: " << this->Kernel <<std::endl;
  // std::cout << "Kernel internal: " << this->KernelInternal <<std::endl;
  // std::cout << "Kernel outside: " << this->constant <<std::endl;
  this->myKernel(p, model);
  arma::mat Tp = this->Tp[0];
  // std::cout << "Tp : " << Tp << std::endl;
  arma::mat r = arma::prod(X.cols(dim, 2 * dim - 1).t());
  arma::mat inright = arma::kron(r, arma::ones<arma::mat>(m, 1));
  Tp = Tp % inright;

  this->Tp[0] = Tp;
}

// Creates the Markov Chain that corresponds to the representative points
// contained in X.   X is a matrix input which consists of the centres of the
// cell as well
//   as the length of the edges.  The output Tp is a m by m matrix, where m
//   equals the cardinality of the partition defined in X.
void faust_t::MCapprox_Contr(double epsilon, shs_t<arma::mat, int> &model) {

  arma::mat X = this->X;

  arma::mat U = this->U;

  // Cardinality
  int m = X.n_rows;
  int q = U.n_rows;

  // Dimension of the system
  int dim = (X.n_cols) / 2;

  // Dimension of the input
  int dim_u = (U.n_cols) / 2;

  // Calculation of the point-wise transition probabilities
  std::vector<arma::cube> p;
  arma::cube tempP(m, m, q);

  for (int i = 0; i < dim; i++) {
    std::vector<arma::mat> t = {X.col(i), arma::ones<arma::mat>(1, m),
                                arma::ones<arma::mat>(1, q)};
    tempP = nDgrid_cb(t);
    p.push_back(tempP);
  }
  tempP.clear();
  tempP = arma::zeros<arma::cube>(m, X.n_cols, q);
  for (int i = dim; i < 2 * dim; i++) {
    arma::mat A = X.col(i - dim).t();
    std::vector<arma::mat> t1 = {arma::ones<arma::mat>(1, m), A,
                                 arma::ones<arma::mat>(1, q)};
    tempP = nDgrid_cb(t1);
    p.push_back(tempP);
  }
  tempP.clear();
  tempP = arma::zeros<arma::cube>(m, m, U.n_cols);
  for (int i = 2 * dim; i < 2 * dim + dim_u; i++) {
    std::vector<arma::mat> t2 = {arma::ones<arma::mat>(1, m),
                                 arma::ones<arma::mat>(1, m),
                                 U.col(i - 2 * dim)};
    tempP = nDgrid_cb(t2);
    p.push_back(tempP);
  }
  std::cout << "Creation of the Markov Chain in Progress" << std::endl;
  //  Make Tp transition probabilities by multiplying with the Lebesque measure
  // 1. Get Tp
  std::vector<arma::mat> Tp;
  for (size_t i = 0; i < q; ++i) {
    std::vector<arma::mat> pt;
    for (size_t j = 0; j < p.size(); ++j) {
      pt.push_back(p[j].slice(i));
    }
    this->myKernel(pt, model, dim, dim_u);

    Tp.push_back(this->Tp[0]);
  }
  std::vector<arma::mat> tem = {arma::ones<arma::mat>(m, 1),
                                arma::prod(X.cols(dim, 2 * dim - 1).t(), 0),
                                arma::ones<arma::mat>(q, 1)};
  std::vector<arma::mat> L = nDgrid(tem);
  // 2. Multiplying with the Lebesque measure
  for(size_t k = 0; k < Tp.size(); ++k) {
   Tp[k] = Tp[k]%L[k];
  }
  this->Tp = Tp;
}

// Creates the Markov Chain that corresponds to the representative points
// contained in X.
void faust_t::MCcreator(double epsilon) {
  // Cardinality
  double m = this->X.n_rows;
  // Dimensions of system
  int dim = this->X.n_cols / 2;

  // Timing of integral

  // Tolerances of the integration
  double absTol = 1e-14;/// m / 1000;
  double relTol = 1e-10 ;/// 1000;

  // Max number of iter
  double maxIter = 500;

  arma::mat tempTp = arma::zeros<arma::mat>(m, m);

  // Calculate the transition probabilities
  double val, err;
  arma::mat r, inright;

  switch (dim) {
  case 6:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double xmin[6] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                            (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) - this->X(j, dim + 2) * 0.5),
                            (this->X(j, 3) - this->X(j, dim + 3) * 0.5),
                            (this->X(j, 4) - this->X(j, dim + 4) * 0.5),
                            (this->X(j, 5) - this->X(j, dim + 5) * 0.5)};
        double xmax[6] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                            (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) + this->X(j, dim + 2) * 0.5),
                            (this->X(j, 3) + this->X(j, dim + 3) * 0.5),
                            (this->X(j, 4) + this->X(j, dim + 4) * 0.5),
                            (this->X(j, 5) + this->X(j, dim + 5) * 0.5)};
        std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
        X.push_back(this->model.x_mod[0].A);
        X.push_back(this->model.x_mod[0].sigma);
        X.push_back(this->model.x_mod[0].Q);
        hcubature(dim, ff_kernel, &X, dim, xmin, xmax, maxIter, absTol, relTol,
                  ERROR_INDIVIDUAL, &val, &err);
        tempTp(j, i) = val;
      }
    }
    this->Tp.clear();
    this->Tp.push_back(tempTp);
    break;
  case 5:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double xmin[5] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                            (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) - this->X(j, dim + 2) * 0.5),
                            (this->X(j, 3) - this->X(j, dim + 3) * 0.5),
                            (this->X(j, 4) - this->X(j, dim + 4) * 0.5)};
        double xmax[5] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                            (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) + this->X(j, dim + 2) * 0.5),
                            (this->X(j, 3) + this->X(j, dim + 3) * 0.5),
                            (this->X(j, 4) + this->X(j, dim + 4) * 0.5)};
        std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
        X.push_back(this->model.x_mod[0].A);
        X.push_back(this->model.x_mod[0].sigma);
        X.push_back(this->model.x_mod[0].Q);
        hcubature(dim, ff_kernel, &X, dim, xmin, xmax, maxIter, absTol, relTol,
                  ERROR_INDIVIDUAL, &val, &err);
        tempTp(j, i) = val;
      }
    }
    this->Tp.clear();
    this->Tp.push_back(tempTp);
    break;
  case 4:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double xmin[4] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                            (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) - this->X(j, dim + 2) * 0.5),
                            (this->X(j, 3) - this->X(j, dim + 3) * 0.5)};
        double xmax[4] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                            (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) + this->X(j, dim + 2) * 0.5),
                            (this->X(j, 3) + this->X(j, dim + 3) * 0.5)};
        std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
        X.push_back(this->model.x_mod[0].A);
        X.push_back(this->model.x_mod[0].sigma);
        X.push_back(this->model.x_mod[0].Q);
        hcubature(dim, ff_kernel, &X, dim, xmin, xmax, maxIter, absTol, relTol,
                  ERROR_INDIVIDUAL, &val, &err);
        tempTp(j, i) = val;
      }
    }
    this->Tp.clear();
    this->Tp.push_back(tempTp);

    break;
  case 3:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double xmin[3] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                            (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) - this->X(j, dim + 2) * 0.5)};
        double xmax[3] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                            (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                            (this->X(j, 2) + this->X(j, dim + 2) * 0.5)};
        std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
        X.push_back(this->model.x_mod[0].A);
        X.push_back(this->model.x_mod[0].sigma);
        X.push_back(this->model.x_mod[0].Q);
        hcubature(dim, ff_kernel, &X, dim, xmin, xmax, maxIter, absTol, relTol,
                  ERROR_INDIVIDUAL, &val, &err);
        tempTp(j, i) = val;
      }
    }
    this->Tp.clear();
    this->Tp.push_back(tempTp);
    break;
  case 2:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double xmin[2] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                            (this->X(j, 1) - this->X(j, dim + 1) * 0.5)},
               xmax[2] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                          (this->X(j, 1) + this->X(j, dim + 1) * 0.5)};
        std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
        X.push_back(this->model.x_mod[0].A);
        X.push_back(this->model.x_mod[0].sigma);
        X.push_back(this->model.x_mod[0].Q);
        hcubature(dim, ff_kernel, &X, dim, xmin, xmax, maxIter, absTol, relTol,
                  ERROR_INDIVIDUAL, &val, &err);
        tempTp(j, i) = val;
      }
    }

    this->Tp.clear();
    this->Tp.push_back(tempTp);
    break;
  case 1:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double xmin[1] = {(this->X(j, 0) - this->X(j, dim) * 0.5)},
               xmax[2] = {(this->X(j, 0) + this->X(j, dim) * 0.5)};
        std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
        X.push_back(this->model.x_mod[0].A);
        X.push_back(this->model.x_mod[0].sigma);
        X.push_back(this->model.x_mod[0].Q);
        hcubature(dim, ff_kernel, &X, dim, xmin, xmax, maxIter, absTol, relTol,
                  ERROR_INDIVIDUAL, &val, &err);
        tempTp(j, i) = val;
      }
    }
    this->Tp.clear();
    this->Tp.push_back(tempTp);

    break;
  default:
    std::cout << "There will be an aditional error, because of the high "
                 "dimensionality of the system. The approximation error will "
                 "be approximately twice as large. For a formal approach, "
                 "please consider the Lipschitz method that includes the MC "
                 "approximation error"
              << std::endl;

    // Calculation of the pointwise transition probabilities
    this->MCapprox(epsilon);
  }
}

// Creates the Markov Chain that corresponds to the representative points
// contained in X.
void faust_t::MCcreator_Contr(double epsilon) {
  // Cardinality
  int m = this->X.n_rows;

  // Dimensions of system
  int dim = this->X.n_cols / 2;

  // Cardinality of input
  int q = this->U.n_rows;

  // Dimensions of system
  int dim_u = this->U.n_cols / 2;

  // Timing of integral
  // clock_t begin, end;
  // double time;
  // begin = clock();

  // Tolerances of the integration
  double absTol = 1 / m / 1000;
  double relTol = 1 / 1000;

  // Max number of iter
  int maxIter = 10000;
  arma::mat tempTp1 = arma::zeros<arma::mat>(m, m);
  this->Tp.clear();
  for (unsigned i = 0; i < q; i++) {
    this->Tp.push_back(tempTp1);
  }

  // Calculate the transition probabilities
  double val, err;
  arma::mat r, inright;

  switch (dim) {
  case 6:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < q; k++) {
          double xmin[6] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                              (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) - this->X(j, dim + 2) * 0.5),
                              (this->X(j, 3) - this->X(j, dim + 3) * 0.5),
                              (this->X(j, 4) - this->X(j, dim + 4) * 0.5),
                              (this->X(j, 5) - this->X(j, dim + 5) * 0.5)};
          double xmax[6] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                              (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) + this->X(j, dim + 2) * 0.5),
                              (this->X(j, 3) + this->X(j, dim + 3) * 0.5),
                              (this->X(j, 4) + this->X(j, dim + 4) * 0.5),
                              (this->X(j, 5) + this->X(j, dim + 5) * 0.5)};
          std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
          X.push_back(this->U(k, arma::span(0, dim_u - 1)));
          X.push_back(this->model.x_mod[0].A);
          X.push_back(this->model.x_mod[0].B);
          X.push_back(this->model.x_mod[0].sigma);
          X.push_back(this->model.x_mod[0].Q);
          X.push_back(this->model.x_mod[0].N);
          hcubature(dim, ff_kernel_U, &X, dim, xmin, xmax, maxIter, absTol,
                    relTol, ERROR_INDIVIDUAL, &val, &err);
          this->Tp[k](j, i) = val;
        }
      }
    }
    break;
  case 5:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < q; k++) {
          double xmin[5] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                              (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) - this->X(j, dim + 2) * 0.5),
                              (this->X(j, 3) - this->X(j, dim + 3) * 0.5),
                              (this->X(j, 4) - this->X(j, dim + 4) * 0.5)};
          double xmax[5] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                              (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) + this->X(j, dim + 2) * 0.5),
                              (this->X(j, 3) + this->X(j, dim + 3) * 0.5),
                              (this->X(j, 4) + this->X(j, dim + 4) * 0.5)};
          std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
          X.push_back(this->U(k, arma::span(0, dim_u - 1)));
          X.push_back(this->model.x_mod[0].A);
          X.push_back(this->model.x_mod[0].B);
          X.push_back(this->model.x_mod[0].sigma);
          X.push_back(this->model.x_mod[0].Q);
          X.push_back(this->model.x_mod[0].N);
          hcubature(dim, ff_kernel_U, &X, dim, xmin, xmax, maxIter, absTol,
                    relTol, ERROR_INDIVIDUAL, &val, &err);
          this->Tp[k](j, i) = val;
        }
      }
    }
    break;
  case 4:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < q; k++) {
          double xmin[4] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                              (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) - this->X(j, dim + 2) * 0.5),
                              (this->X(j, 3) - this->X(j, dim + 3) * 0.5)};
          double xmax[4] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                              (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) + this->X(j, dim + 2) * 0.5),
                              (this->X(j, 3) + this->X(j, dim + 3) * 0.5)};
          std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
          X.push_back(this->U(k, arma::span(0, dim_u - 1)));
          X.push_back(this->model.x_mod[0].A);
          X.push_back(this->model.x_mod[0].B);
          X.push_back(this->model.x_mod[0].sigma);
          X.push_back(this->model.x_mod[0].Q);
          X.push_back(this->model.x_mod[0].N);
          hcubature(dim, ff_kernel_U, &X, dim, xmin, xmax, maxIter, absTol,
                    relTol, ERROR_INDIVIDUAL, &val, &err);
          this->Tp[k](j, i) = val;
        }
      }
    }
    break;
  case 3:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < q; k++) {
          double xmin[3] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                              (this->X(j, 1) - this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) - this->X(j, dim + 2) * 0.5)};
          double xmax[3] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                              (this->X(j, 1) + this->X(j, dim + 1) * 0.5),
                              (this->X(j, 2) + this->X(j, dim + 2) * 0.5)};
          std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
          X.push_back(this->U(k, arma::span(0, dim_u - 1)));
          X.push_back(this->model.x_mod[0].A);
          X.push_back(this->model.x_mod[0].B);
          X.push_back(this->model.x_mod[0].sigma);
          X.push_back(this->model.x_mod[0].Q);
          X.push_back(this->model.x_mod[0].N);
          hcubature(dim, ff_kernel_U, &X, dim, xmin, xmax, maxIter, absTol,
                    relTol, ERROR_INDIVIDUAL, &val, &err);
          this->Tp[k](j, i) = val;
        }
      }
    }
    break;
  case 2:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < q; k++) {
          double xmin[2] = {(this->X(j, 0) - this->X(j, dim) * 0.5),
                              (this->X(j, 1) - this->X(j, dim + 1) * 0.5)},
                 xmax[2] = {(this->X(j, 0) + this->X(j, dim) * 0.5),
                            (this->X(j, 1) + this->X(j, dim + 1) * 0.5)};
          std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
          X.push_back(this->U(k, arma::span(0, dim_u - 1)));
          X.push_back(this->model.x_mod[0].A);
          X.push_back(this->model.x_mod[0].B);
          X.push_back(this->model.x_mod[0].sigma);
          X.push_back(this->model.x_mod[0].Q);
          X.push_back(this->model.x_mod[0].N);
          hcubature(dim, ff_kernel_U, &X, dim, xmin, xmax, maxIter, absTol,
                    relTol, ERROR_INDIVIDUAL, &val, &err);

          this->Tp[k](j, i) = val;
        }
      }
    }
    break;
  case 1:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < q; k++) {
          double xmin[1] = {(this->X(j, 0) - this->X(j, dim) * 0.5)},
                 xmax[2] = {(this->X(j, 0) + this->X(j, dim) * 0.5)};
          std::vector<arma::mat> X = {this->X(i, arma::span(0, dim - 1))};
          X.push_back(this->U(k, arma::span(0, dim_u - 1)));
          X.push_back(this->model.x_mod[0].A);
          X.push_back(this->model.x_mod[0].B);
          X.push_back(this->model.x_mod[0].sigma);
          X.push_back(this->model.x_mod[0].Q);
          X.push_back(this->model.x_mod[0].N);
          hcubature(dim, ff_kernel_U, &X, dim, xmin, xmax, maxIter, absTol,
                    relTol, ERROR_INDIVIDUAL, &val, &err);
          this->Tp[k](j, i) = val;
        }
      }
    }
    break;
  default:
    std::cout << "There will be an aditional error, because of the high "
                 "dimensionality of the system. The approximation error will "
                 "be approximately twice as large. For a formal approach, "
                 "please consider the Lipschitz method that includes the MC "
                 "approximation error"
              << std::endl;
    // Calculation of the pointwise transition probabilities
    this->MCapprox_Contr(epsilon, this->model);
  }
}

// Computes the reach avoid probability given N time steps
// and the transition matrix Tp for a hybrid sytem
void faust_t::StandardProbSafety(int modes, arma::mat Tq,
                                 std::vector<arma::mat> Tpvec, int T) {

  this->Tp.clear();

  std::vector<arma::mat> Tpshift;

  for (int i = 0; i < modes; i++) // Store Tp for each mode
  {
    this->Tp.push_back(Tpvec[i]);
  }

  // Take care of the discrete probabilities which shift the values
  // of Tp for each mode
  int value = 1;
  for(int j =0; j < Tq.n_rows; j++) {
    for(int k =0; k < Tq.n_cols; k++) {
      Tpshift.push_back(arma::pow(Tq(j,k)*Tpvec[k],T));
    }
    value *= Tpvec[j].n_rows;
    if(value > 100000){
      std::cout << "Generated abstraction (parallel composition for each mode is too large) "<<std::endl;
      std::cout << "Storing generated abstractions into PRISM files" <<std::endl;
      this->V = arma::zeros<arma::mat>(1,1);
      auto t = std::time(nullptr);
      auto tm = *std::localtime(&t);
      std::ostringstream oss;
      oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
      auto str = oss.str();
      this->formatOutput(0, str, 2, T);
      exit(0);
    }

  }

  // Combine to get final result
  arma::mat Tpsum = arma::zeros<arma::mat>(value, value);
  for(int i = 0; i < modes*modes; i=i+modes) {
    for(int l =0; l < modes; l=l+2) {
      if(l ==0 ) {
        Tpsum = arma::kron(Tpshift[0], Tpshift[1]);
     }
     else {
       arma::mat temp = arma::kron(Tpsum, Tpshift[l+1]);
       Tpsum.reshape(temp.n_rows, temp.n_cols);
       Tpsum = temp;
     }
    }
  }
  this->V = Tpsum*arma::ones<arma::mat>(value,1);
  std::cout << "Computed transition probabilities" <<std::endl;

}

// Computes the safety probability given T time steps
// and the transition matrix Tp
void faust_t::StandardProbSafety(int T) {
  // The Cardinality
  int elSmallest = this->Tp[0].n_rows;

  arma::mat V = arma::ones<arma::mat>(elSmallest, 1);

  arma::mat Tp = this->Tp[0];
  V = arma::pow(Tp, T) * V;
  this->V = V;
}

// TODO: Extend to SHS with multiple modes
// Computes the reach avoid probability given N time steps
// and the transition matrix Tp for a hybrid system
void faust_t::StandardReachAvoid(int modes, arma::mat Tq,
                                 std::vector<arma::mat> Tpvec, arma::mat X,
                                 arma::mat Target, int elSmallest, int T) {
  arma::mat V = arma::ones<arma::mat>(elSmallest, modes);
  arma::mat Tpsum = arma::zeros<arma::mat>(elSmallest, modes);

  // The Cardinality
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Initialization of value function W
  arma::mat A = repmat(Target.col(1), 1, m);
  //  A =A.replicate(1,m);
  arma::mat Xi = X.cols(0, dim - 1);
  // arma::mat lhs=  Xi < A;
  arma::mat C = repmat(Target.col(0), 1, m);

  arma::umat inner = Xi < A.t();
  arma::umat outer = Xi > C.t();
  arma::mat a = arma::conv_to<arma::mat>::from(inner);
  arma::mat b = arma::conv_to<arma::mat>::from(outer);

  V = arma::prod(a % b, 1);

  // Matrices used for the calculation of V
  arma::mat W_help1 = arma::ones<arma::mat>(elSmallest, 1) - V.col(0);
  arma::mat W_help2 = V.col(0);

  // The solution
  for (int i = 1; i <= T; i++) {
    for (int j = 0; j < modes; j++) {
      arma::mat Tp = Tpvec[j];
      arma::mat temp = Tp * V.col(j);
      V.col(j) = temp % (W_help1);
      V.col(j) += W_help2;
    }
    if (modes > 1)
      V = V * Tq;
  }

  this->V = V;
}

// Computes the reach avoid probability given N time steps
// and the transition matrix Tp
void faust_t::StandardReachAvoid(arma::mat Target, int T) {
  // The dimensions
  int elSmallest = this->Tp[0].n_rows;
  arma::mat V = arma::ones<arma::mat>(elSmallest, 1);

  // The Cardinality
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Initialization of value function W
  arma::mat A = repmat(Target.col(1), 1, m);

  arma::mat Xi = X.cols(0, dim - 1);

  // arma::mat lhs=  Xi < A;
  arma::mat C = repmat(Target.col(0), 1, m);

  arma::umat inner = Xi < A.t();
  arma::umat outer = Xi > C.t();
  arma::mat a = arma::conv_to<arma::mat>::from(inner);
  arma::mat b = arma::conv_to<arma::mat>::from(outer);
  std::cout << "a " << a<<std::endl;
  std::cout << "b " << b<<std::endl;
  V = arma::prod(a % b, 1);

  // Matrices used for the calculation of V
  arma::mat W_help1 = arma::ones<arma::mat>(elSmallest, 1) - V;
  arma::mat W_help2 = V.col(0);

  // The solution
  arma::mat Tp = this->Tp[0];
  for (int i = 1; i <= T; i++) {
    arma::mat temp = Tp * V;
    V = temp % (W_help1);
    V += W_help2;
  }

  this->V = V;
}

// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon.
//   Input of this function is the desired maximal error epsilon and the number
//   of time steps T
//  The output is a new partition X and a maximum error
void faust_t::Adaptive_grid_multicell(double epsilon, double T) {
  // Dimension of the system
  int dim = this->X.n_cols / 2;

  // The initial Lipschitz constants
  arma::mat h_ij = this->LocalLipschitz(1, this->X.n_rows);

  //  definition of gamma_i
  arma::mat internal = arma::prod(X.cols(dim, 2 * dim - 1).t());
  arma::mat gamma_i = T * h_ij * internal.t();

  // The resulting local errors
  arma::mat internal2 = arma::sum(arma::pow(X.cols(dim, 2 * dim - 1).t(), 2));
  arma::mat E_i = gamma_i % arma::pow(internal2, 0.5).t();
  int q = 0;

  arma::mat Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  while (arma::accu(Ei_comp) > 0) {
    double t = arma::max(arma::size(E_i));
    if (E_i.is_empty()) {
      t = 0;
    }
    int end = (t - q);
    for (int i = 0; i < end; ++i) {
      if (E_i(q) > epsilon) {
        // finding the index related to maximum dimension
        arma::uword m0 = X(q, arma::span(dim, 2 * dim - 1)).index_max();

        // splitting the cell into two cells along its largest edge
        arma::mat Y = arma::repmat(X.row(q), 2, 1);

        // the first smaller cell
        Y(0, dim + m0) = X(q, dim + m0) / 2;

        Y(0, m0) = X(q, m0) - Y(0, dim + m0) / 2;

        // the second smaller cell
        Y(1, dim + m0) = X(q, dim + m0) / 2;
        Y(1, m0) = X(q, m0) + Y(1, dim + m0) / 2;

        // Update X
        if ((q - 1) < 0) {
          X = X.rows(q + 1, X.n_rows - 1);
        } else {
          X = arma::join_vert(X.rows(0, q - 1), X.rows(q + 1, X.n_rows - 1));
        }

        X = arma::join_vert(X, Y);
        // Update E_i
        if (i == (end - 1)) {
          E_i.clear();
        } else {
          if ((q - 1) < 0) {
            E_i = E_i.rows(q + 1, E_i.n_rows - 1);
          } else {
            E_i = arma::join_vert(E_i.rows(0, q - 1),
                                  E_i.rows(q + 1, E_i.n_rows - 1));
          }
        }
        if(E_i.is_empty()) {
          break;
        }
      } else {
        q++;
      }
    }
    // Update X values
    this->X = X;
    // Update local error
    // The updated Lipschitz constants
    h_ij = this->LocalLipschitz(q+1, this->X.n_rows);
    // definition of gamma_i
    internal = arma::prod(X.cols(dim, 2 * dim - 1).t());
    gamma_i = T * h_ij * internal.t();

    // The resulting local errors
    int E_iNr = E_i.n_rows;
    int E_iNc = E_i.n_cols;
    if (!E_i.is_empty()) {
      internal2 = arma::sum(arma::pow(X(arma::span(q, X.n_rows -1), arma::span(dim, 2 * dim - 1)).t(), 2));
      arma::mat E_i_add = gamma_i % arma::pow(internal2, 0.5).t();
      E_i = arma::join_vert(E_i, E_i_add);
    }
    else {
      internal2 = arma::sum(arma::pow(X(arma::span(q, X.n_rows -1), arma::span(dim, 2 * dim - 1)).t(), 2));
      E_i = gamma_i % arma::pow(internal2, 0.5).t();
    }
    Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
    q = 0;
  }
  arma::mat E_temp = arma::max(E_i);
  this->E = E_temp(0, 0);
  this->X = X;
}

// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon.
void faust_t::Adaptive_grid_multicell_semilocal(double epsilon, double T,
                                                arma::mat SafeSet) {
  // Get X
  arma::mat X = this->X;

  // Dimension of the system
  int dim = X.n_cols / 2;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The initial Lipschitz constants (from X(n) to all X)
  arma::mat h_i = this->SemiLocalLipschitz(SafeSet, 0, 0);

  // definition of gamma_i
  arma::mat gamma_i = T * h_i * arma::prod(r, 0);

  //  The resulting local errors
  arma::mat tempLocal = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
  tempLocal = arma::sum(tempLocal, 0);
  tempLocal = arma::pow(tempLocal, 0.5);
  arma::mat E_i = gamma_i % tempLocal.t();

  int q = 0;
  arma::mat Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  while (arma::accu(Ei_comp) > 0) {
    double t = arma::max(arma::size(E_i));
    if (E_i.is_empty()) {
      t = 0;
    }
    int end = (t - q);
    for (int i = 0; i < end; ++i) {
      if (E_i(q) > epsilon) {
        // finding the index related to maximum dimension
        double m0 = X(q, arma::span(dim, 2 * dim - 1)).index_max();

        // splitting the cell into two cells along its largest edge
        arma::mat Y = arma::repmat(X.row(q), 2, 1);

        // the first smaller cell
        Y(0, dim + m0) = X(q, dim + m0) / 2;

        Y(0, m0) = X(q, m0) - Y(0, dim + m0) / 2;

        // the second smaller cell
        Y(1, dim + m0) = X(q, dim + m0) / 2;
        Y(1, m0) = X(q, m0) + Y(1, dim + m0) / 2;

        // Update X
        if ((q - 1) < 0) {
          X = X.rows(q + 1, X.n_rows - 1);
        } else {
          X = arma::join_vert(X.rows(0, q - 1), X.rows(q + 1, X.n_rows - 1));
        }

        X = arma::join_vert(X, Y);

        // Update E_i
        if (i == (end - 1)) {
          E_i.clear();
        } else {
          if ((q - 1) < 0) {
            E_i = E_i.rows(q + 1, E_i.n_rows - 1);
          } else {
            E_i = arma::join_vert(E_i.rows(0, q - 1),
                                  E_i.rows(q + 1, E_i.n_rows - 1));
          }
        }
      } else {
        q++;
      }
    }
    // Store new X
    this->X = X;

    // Update local error
    // The updated Lipschitz constants
    h_i = this->SemiLocalLipschitz(SafeSet, q, X.n_rows-1);

    // definition of gamma_i
    gamma_i = T * h_i * arma::prod(r, 0);

    // The resulting local errors
    int E_iNr = E_i.n_rows;
    int E_iNc = E_i.n_cols;
    arma::mat internal2 = arma::sum(arma::pow(X(arma::span(q, X.n_rows -1), arma::span(dim, 2 * dim - 1)).t(), 2));
    if (!E_i.is_empty()) {
      arma::mat E_i_add = gamma_i % arma::pow(internal2, 0.5).t();
      E_i = arma::join_vert(E_i, E_i_add);
    }
    else {
      E_i = gamma_i % arma::pow(internal2, 0.5).t();
    }
    Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
    q = 0;
  }
  arma::mat E_temp = arma::max(E_i);
  this->E = E_temp(0, 0);
}

// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon.
void faust_t::Adaptive_grid_ReachAvoid(double epsilon, double T,
                                       arma::mat SafeSet, arma::mat TargetSet) {
  // Get X
  arma::mat X = this->X;

  // Cardinality
  int m = X.n_rows;

  // Dimension of the system
  int dim = (X.n_cols) / 2;

  // Extract the Target Set (creation of SafeSet minus TargetSet)
  // Indexing the target set
  arma::mat inner1 = arma::repmat(TargetSet.col(1), 1, m).t();
  arma::mat outer1 = X.cols(0, dim - 1);
  arma::mat lhs = arma::conv_to<arma::mat>::from(outer1 < inner1);

  arma::mat inner2 = arma::repmat(TargetSet.col(0), 1, m).t();
  arma::mat rhs = arma::conv_to<arma::mat>::from(outer1 > inner2);

  arma::mat all = lhs % rhs;

  arma::mat TargetIndex = arma::prod(all, 1);

  // Reshape X so that the first entries are the target set
  arma::uvec in1 = arma::find(TargetIndex == 1);
  arma::uvec in2 = arma::find(TargetIndex != 1);
  X = arma::join_vert(X.rows(in1), X.rows(in2));

  // The initial Lipschitz constants
  int k = arma::accu(TargetIndex) + 1;
  arma::mat h_ij = this->LocalLipschitz(k, m);

  // definition of gamma_i
  arma::mat gamma_i =
      T * h_ij * arma::prod(X.cols(dim, 2 * dim - 1).t(), 0).t();

  // The resulting local errors
  arma::mat E_i = arma::zeros<arma::mat>(1, 1);
  arma::mat internal2 = arma::sum(
      arma::pow(X(arma::span(k - 1, m - 1), arma::span(dim, 2 * dim - 1)), 2)
          .t());
  arma::mat E_in = gamma_i % arma::pow(internal2, 0.5).t();
  if (k > 1) {
    E_i = arma::join_vert(arma::zeros<arma::mat>(k - 1, 1), E_in);
  } else {
    E_i = E_in;
  }

  int q = 0;
  arma::mat Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  while (arma::accu(Ei_comp) > 0) {
    double t = arma::max(arma::size(E_i));
    if (E_i.is_empty()) {
      t = 0;
    }
    int end = (t - q);
    for (int i = 0; i < end; ++i) {
      if (E_i(q) > epsilon) {
        // finding the index related to maximum dimension
        double m0 = X(q, arma::span(dim, 2 * dim - 1)).index_max();

        // splitting the cell into two cells along its largest edge
        arma::mat Y = arma::repmat(X.row(q), 2, 1);

        // the first smaller cell
        Y(0, dim + m0) = X(q, dim + m0) / 2;

        Y(0, m0) = X(q, m0) - Y(0, dim + m0) / 2;

        // the second smaller cell
        Y(1, dim + m0) = X(q, dim + m0) / 2;
        Y(1, m0) = X(q, m0) + Y(1, dim + m0) / 2;

        // Update X
        if ((q - 1) < 0) {
          X = X.rows(q + 1, X.n_rows - 1);
        } else {
          X = arma::join_vert(X.rows(0, q - 1), X.rows(q + 1, X.n_rows - 1));
        }
        X = arma::join_vert(X, Y);

        // Update E_i
        if (i == (end - 1)) {
          E_i.clear();
        } else {
          if ((q - 1) < 0) {
            E_i = E_i.rows(q + 1, E_i.n_rows - 1);
          } else {
            E_i = arma::join_vert(E_i.rows(0, q - 1),
                                  E_i.rows(q + 1, E_i.n_rows - 1));
          }
        }
      } else {
        q++;
      }
    }
    // Store X into object
    this->X = X;

    // Update local error
    // The updated Lipschitz constants
    h_ij = this->LocalLipschitz(q + 1, X.n_rows);

    // definition of gamma_i
    gamma_i = T * h_ij * arma::prod(X.cols(dim, 2 * dim - 1).t(), 0).t();

    // The resulting local errors
    internal2 = arma::sum(
        arma::pow(X(arma::span(q, X.n_rows - 1), arma::span(dim, 2 * dim - 1)),
                  2)
            .t());
    int E_iNr = E_i.n_rows;
    int E_iNc = E_i.n_cols;
    if (!E_i.is_empty()) {
      internal2 = arma::sum(arma::pow(X(arma::span(q, X.n_rows -1), arma::span(dim, 2 * dim - 1)).t(), 2));
      arma::mat E_i_add = gamma_i % arma::pow(internal2, 0.5).t();
      E_i = arma::join_vert(E_i, E_i_add);
    }
    else {
      internal2 = arma::sum(arma::pow(X(arma::span(q, X.n_rows -1), arma::span(dim, 2 * dim - 1)).t(), 2));
      E_i = gamma_i % arma::pow(internal2, 0.5).t();
    }
    Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
    q = 0;
  }
  arma::mat E_temp = arma::max(E_i);
  this->E = E_temp(0, 0);
}

// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon.
void faust_t::Adaptive_grid_ReachAvoid_semilocal(double epsilon, double T,
                                                 arma::mat SafeSet,
                                                 arma::mat TargetSet) {
  // Get X
  arma::mat X = this->X;

  // Cardinality
  int m = this->X.n_rows;

  // Dimension of the system
  int dim = this->X.n_cols / 2;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // Extract the Target Set (creation of SafeSet minus TargetSet)
  // Indexing the target set
  arma::mat inner1 = arma::repmat(TargetSet.col(1), 1, m).t();
  arma::mat outer1 = X.cols(0, dim - 1);
  arma::mat lhs = arma::conv_to<arma::mat>::from(outer1 < inner1);

  arma::mat inner2 = arma::repmat(TargetSet.col(0), 1, m).t();
  arma::mat rhs = arma::conv_to<arma::mat>::from(outer1 > inner2);

  arma::mat all = lhs % rhs;

  arma::mat TargetIndex = arma::prod(all, 1);

  // Reshape X so that the first entries are the target set
  arma::uvec in1 = arma::find(TargetIndex == 1);
  arma::uvec in2 = arma::find(TargetIndex != 1);
  X = arma::join_vert(X.rows(in1), X.rows(in2));

  // The initial Lipschitz constants (from X(n) to all X)
  int k = arma::accu(TargetIndex) + 1;
  arma::mat h_i = this->SemiLocalLipschitz(SafeSet, k-1, m-1);

  // definition of gamma_i
  arma::mat gamma_i = T * h_i * arma::prod(r, 0);

  //  The resulting local errors
  arma::mat E_i = arma::zeros<arma::mat>(k-1, 1);
  int E_iNr = E_i.n_rows;
  int E_iNc = E_i.n_cols;
  arma::mat internal2 = arma::sum(arma::pow(X(arma::span(k-1, m-1 ), arma::span(dim, 2 * dim - 1)).t(), 2));
  if (!E_i.is_empty()) {
    arma::mat E_i_add = gamma_i % arma::pow(internal2, 0.5).t();
    E_i = arma::join_vert(E_i, E_i_add);
  }
  else {
    E_i = gamma_i % arma::pow(internal2, 0.5).t();
  }
  arma::mat Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  int q = 0;
  while (arma::accu(Ei_comp) > 0) {
    double t = arma::max(arma::size(E_i));
    if (E_i.is_empty()) {
      t = 0;
    }
    int end = (t - q);
    for (int i = 0; i < end; ++i) {
      if (E_i(q) > epsilon) {
        // finding the index related to maximum dimension
        double m0 = X(q, arma::span(dim, 2 * dim - 1)).index_max();

        // splitting the cell into two cells along its largest edge
        arma::mat Y = arma::repmat(X.row(q), 2, 1);

        // the first smaller cell
        Y(0, dim + m0) = X(q, dim + m0) / 2;

        Y(0, m0) = X(q, m0) - Y(0, dim + m0) / 2;

	// the second smaller cell
        Y(1, dim + m0) = X(q, dim + m0) / 2;
        Y(1, m0) = X(q, m0) + Y(1, dim + m0) / 2;

        // Update X
        if ((q - 1) < 0) {
          X = X.rows(q + 1, X.n_rows - 1);
        } else {
          X = join_vert(X.rows(0, q - 1), X.rows(q + 1, X.n_rows - 1));
        }
        X = arma::join_vert(X, Y);

        // Update E_i
        if (i == (end - 1)) {
          E_i.clear();
        } else {
          if ((q - 1) < 0) {
            E_i = E_i.rows(q + 1, E_i.n_rows - 1);
          } else {
            E_i = arma::join_vert(E_i.rows(0, q - 1),
                                  E_i.rows(q + 1, E_i.n_rows - 1));
          }
        }
      } else {
        q++;
      }
    }
    // Store X
    this->X = X;

    // Update local error
    // The updated Lipschitz constants
    h_i = this->SemiLocalLipschitz(SafeSet, q, X.n_rows);
    // definition of gamma_i
    gamma_i = T * h_i * arma::prod(r, 0);

    // The resulting local errors
    internal2 = arma::sum(
          arma::pow(
              X(arma::span(q-1, X.n_rows - 1), arma::span(dim, 2 * dim - 1)), 2)
              .t());
   E_iNr = E_i.n_rows;
   E_iNc = E_i.n_cols;
   if (!E_i.is_empty()) {
    arma::mat E_i_add = gamma_i % arma::pow(internal2, 0.5).t();
    E_i = arma::join_vert(E_i, E_i_add);
   }
   else {
    E_i = gamma_i % arma::pow(internal2, 0.5).t();
   }
   q = 0;
   Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  }
  arma::mat E_temp = arma::max(E_i);
  this->E = E_temp(0, 0);
}

// Adaptive_grid_MCapprox This function makes an adaptive grid that will
// produce a certain maximal abstraction error defined by epsilon.
void faust_t::Adaptive_grid_MCapprox(double epsilon, double T,
                                     arma::mat SafeSet) {
  // The dimension of the system
  int dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // Generate Uniform grid
  // Derivation of the global Lipschitz constants
  double h = this->GlobalLipschitz(SafeSet, 1);

  // The length of the edges of the partition
  arma::mat temp = arma::prod(r);
  double t0 = temp(0, 0);
  double delta = (epsilon / (t0 * h * T)) / sqrt(dim);

  // Adjust delta if it is very large compared to r
  arma::mat a = arma::prod(ceil(r / delta));
  double a1 = a(0, 0);
  double b = (std::pow(2, dim) + 10);
  while (a1 < b) // appr number of bins
  {
    delta = delta / (std::pow(2, (1 / dim))); // This will cause the number of cells to double
    a = arma::prod(ceil(r / delta));
    a1 = a(0, 0);
    b = (std::pow(2, dim) + 10);
  }

  // Warn if a dimension of the set has not been partitioned.
  // This implies that in this direction no gradient will be found
  double temp2 = arma::accu((r - delta) < 0);

  if (temp2 > 0) {
    std::cout << "A dimension of the set has not been partitioned, reshaping "
                 "the set might solve this issue"
              << std::endl;
  }

  // Create the location of the representative points delta is adjusted so that
  // the edges of the set are part of the partitions
  std::vector<arma::mat> C;
  arma::mat delta_adj = arma::zeros<arma::mat>(1, dim);
  for (int i = 0; i < dim; i++) {
    delta_adj.col(i) = (SafeSet(i, 1) - SafeSet(i, 0)) /
                       std::ceil((SafeSet(i, 1) - SafeSet(i, 0)) / delta);
    double start = SafeSet(i, 0) + 0.5 * delta_adj(i);
    double Delta = delta_adj(i);
    double end = SafeSet(i, 1) - 0.5 * delta_adj(i);
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  int m = C[0].n_elem;
  int sm = std::sqrt(m);

  // Make the output
  int index = 0;
  int k = 0;
  arma::mat X = arma::zeros<arma::mat>(m, dim);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < dim; j++) {
      X(i, j) = C[j](k, index);
    }
    if (k >= (sm - 1)) {
      k = 0;
      index++;
    } else {
      k++;
    }
  }
  arma::mat Y = arma::kron(delta_adj, arma::ones<arma::mat>(m, 1));

  X = join_horiz(X, Y);

  C.clear();

  this->X = X;

  // Adaptive grid
  // The initial Lipschitz constants
  arma::mat h_ij = this->LocalLipschitz(1, this->X.n_rows);
  // std::cout << "h_ij : " << h_ij.n_rows << std::endl;

  arma::mat h_ij_bar = arma::ones(this->X.n_rows,2);//this->LocalLipschitzXbar(1, this->X.n_rows);
  std::cout << "h_ij_bar : " << h_ij_bar.n_rows << std::endl;

  // Definition of h_tot
  arma::mat internal = arma::pow(arma::sum(X.cols(dim, 2 * dim - 1)), 2);
  internal = arma::pow(internal.t(), 0.5).t();
  arma::mat internal2 = arma::repmat(internal, 1, m);
  arma::mat internal3 = arma::repmat(internal,m,1);
  std::cout << "internal2 : " << internal2.n_rows << ", " << internal2.n_cols << std::endl;
  arma::mat h_tot = h_ij % internal2 + h_ij_bar % internal3;
  std::cout << "h_tot : " << h_tot << std::endl;

  // Definition of K_i
  arma::mat K_i = h_tot * arma::prod(X.cols(dim, 2 * dim - 1).t(), 0).t();
  // The resulting local errors
  arma::mat E_i = T * K_i;

  int q = 0;
  arma::mat Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  while (arma::accu(Ei_comp) > 0) {
    double t = arma::max(arma::size(E_i));
    if (E_i.is_empty()) {
      t = 0;
    }
    int end = (t - q);
    for (int i = 0; i < end; ++i) {
      if (E_i(q) > epsilon) {
        // finding the index related to maximum dimension
        double m0 = X(q, arma::span(dim, 2 * dim - 1)).index_max();

        // splitting the cell into two cells along its largest edge
        arma::mat Y = arma::repmat(X.row(q), 2, 1);

        // the first smaller cell
        Y(0, dim + m0) = X(q, dim + m0) / 2;

        Y(0, m0) = X(q, m0) - Y(0, dim + m0) / 2;

        // the second smaller cell
        Y(1, dim + m0) = X(q, dim + m0) / 2;
        Y(1, m0) = X(q, m0) + Y(1, dim + m0) / 2;

        // Update X

        if ((q - 1) < 0) {
          X = X.rows(q + 1, X.n_rows - 1);
        } else {
          X = arma::join_vert(X.rows(0, q - 1), X.rows(q + 1, X.n_rows - 1));
        }
        X = arma::join_vert(X, Y);

        // Update E_i
        if (i == (end - 1)) {
          E_i.clear();
        } else {
          if ((q - 1) < 0) {
            E_i = E_i.rows(q + 1, E_i.n_rows - 1);
          } else {
            E_i = arma::join_vert(E_i.rows(0, q - 1),
                                  E_i.rows(q + 1, E_i.n_rows - 1));
          }
        }
      } else {
        q++;
      }
    }
    // Store X
    this->X = X;

    // Update local error
    h_ij = this->LocalLipschitz(q, X.n_rows);
    h_ij_bar = this->LocalLipschitz(q, X.n_rows);

    // Definition of h_tot
    internal = arma::pow(
        arma::sum(X(arma::span(q, X.n_rows - 1), arma::span(dim, 2 * dim - 1))),
        2);
    internal = arma::pow(internal.t(), 0.5).t();
    internal2 = arma::repmat(internal, 1, m);
    arma::mat internal3 = arma::pow(arma::sum(X.cols(dim, 2 * dim - 1)), 2);
    internal3 = arma::pow(internal.t(), 0.5).t();
    arma::mat internal4 = arma::repmat(internal3, X.n_rows - q + 1, 1);
    std::cout << "internal2 : " << internal2.n_elem << std::endl;
    std::cout << "internal4 : " << internal4.n_elem << std::endl;
    arma::mat h_tot = h_ij % internal2 + h_ij_bar % internal4;
    std::cout << "h_tot : " << h_tot.n_elem << std::endl;

    // Definition of K_i
    arma::mat k_i = arma::prod(X.cols(dim, 2 * dim - 1).t(), 0);
    K_i = h_tot * k_i.t();
    std::cout << "K_i: " << K_i.n_elem <<std::endl;

    // The resulting local errors
    int E_iNr = E_i.n_rows;
    int E_iNc = E_i.n_cols;
    if (!E_i.is_empty()) {
      arma::mat E_i_add = T*K_i;
      E_i = arma::join_vert(E_i, E_i_add);
    }
    else {
      E_i = T*K_i;
    }
    Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
    q = 0;
  }

  arma::mat E_temp = arma::max(E_i);
  this->E = E_temp(0, 0);
}

// Adaptive_grid_MCapprox This function makes an adaptive grid that will
// produce a certain maximal abstraction error defined by epsilon.
void faust_t::Adaptive_grid_ReachAvoidMCapprox(double epsilon, double T,
                                               arma::mat SafeSet,
                                               arma::mat TargetSet) {
  // The dimension of the system
  int dim = SafeSet.n_rows;

  // The length of the edges of the Target Set
  arma::mat rT = TargetSet.col(1) - TargetSet.col(0);

  // The length of the edges of the Set
  arma::mat rS = SafeSet.col(1) - SafeSet.col(0);

  // Derivation of the global Lipschitz constants
  double h = this->GlobalLipschitz(SafeSet, 0); // over approximation

  // The length of the edges of the partitions
  arma::mat temp = arma::prod(rS);
  double t0 = temp(0, 0);
  double delta = (epsilon / (t0 * h * T)) / sqrt(dim);

  // Adjust delta if it is very large compared to r
  arma::mat a = arma::prod(ceil(rS / delta));
  double a1 = a(0, 0);
  double b = (std::pow(2, dim) + 10);
  while (a1 < b) // appr number of bins
  {
    delta = delta /(std::pow(2, (1 / dim))); // This will cause the number of cells to double
    arma::mat a = arma::prod(ceil(rS / delta));
    a1 = a(0, 0);
    b = (std::pow(2, dim) + 10);
  }

  // delta is adjusted so that the edges of the TargetSet are part of the
  // partitions
  arma::mat delta_adj = arma::zeros<arma::mat>(SafeSet.n_rows, 1);
  for (int i = 0; i < dim; i++) {
    double diffT = (TargetSet(i, 1) - TargetSet(i, 0));
    double inner = diffT / delta;
    delta_adj.row(i) = diffT / std::ceil(inner);
  }
  // The adjusted SafeSet to fit with delta_adj
  arma::mat Residuals = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  arma::mat tempR = arma::repmat(delta_adj, 1, 2);
  arma::mat tempR2 = join_horiz(-delta_adj, 0 * delta_adj);
  std::cout << tempR2 << std::endl;
  arma::mat diff = SafeSet - TargetSet;
  Residuals = (diff - arma::floor(diff/tempR)%tempR) + tempR2;

  arma::mat SafeSet_adj = SafeSet - Residuals;
  // Partition (Safe-Target) into 3^dim-1 separate parts
  arma::mat D = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  arma::mat Dt = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  std::vector<arma::mat> D1;
  std::vector<arma::mat> Dt1;
  for (int i = 0; i < dim; i++) {
    arma::vec tempD = {{SafeSet(i,0)}, {SafeSet_adj(i,0)}, {SafeSet_adj(i,1)}, {SafeSet(i,1)}};
    arma::mat tempD1 = tempD.t();
    D = unique(tempD1);
    Dt = D.cols(0, D.n_cols - 2);
    arma::vec tempV = D.t(); // convert to vector
    D1.push_back(tempV);
    arma::vec tempVt = Dt.t();
    Dt1.push_back(tempVt);
  }
  std::vector<arma::mat> F;
  F = nDgrid(Dt1);
  // Cardinality
  double n = F[0].n_elem;
  std::cout << "n: " << n <<std::endl;
  // Make the output
  arma::mat X = arma::zeros<arma::mat>(0, dim * 2); // Initialisation of X with an empty matrix
  for (int i = 0; i < n; i++) {
    arma::mat Set = arma::zeros<arma::mat>(dim, 2);
    std::vector<arma::mat> C;
    arma::mat delta_adj_local = arma::zeros<arma::mat>(1, dim);
    int index = 0, index_c = 0;
    for (int j = 0; j < dim; j++) {
	arma::uvec s = ind2sub( arma::size(F[j]), i );
	double left = F[j](s(0),s(1));
        for (unsigned k = 0; k < D1[j].n_rows; k++) {
	  for (unsigned kc = 0; kc < D1[j].n_cols; kc++) {
            if (D1[j](k, kc) == F[j](s(0),s(1))) {
              index = k;
	      index_c = kc;
	    }
	}
 	double right = D1[j](index + 1, index_c);
        arma::mat templ;
        templ << left;
        arma::mat tempr;
        tempr << right;
        Set.row(j) = join_horiz(templ, tempr);

      }
           // The ndiv variable eliminates rounding errors
      double ndiv = round((Set(j, 1) - Set(j, 0)) / delta_adj(j));
      if (ndiv == 0) {
        ndiv = 1;
      }
      delta_adj_local(j) = (Set(j, 1) - Set(j, 0)) / ndiv;
      double start = Set(j, 0) + 0.5 * delta_adj_local(j);
      double Delta = delta_adj_local(j);
      double end = Set(j, 1) - 0.4999 * delta_adj_local(j);
      arma::mat A = arma::regspace(start, Delta, end);
      C.push_back(A.t());
      // 0.499 adjusts for rounding errors; it has no influence on the
      // position of the representative points
    }
    // Construct grid
    C = nDgrid(C);
    // Cardinality
    double m = C[0].n_rows;
    std::cout << "m" << m << "col: " << C[0].n_cols<< std::endl;
    // Make the output
    arma::mat Y = arma::zeros<arma::mat>(m, dim);
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < dim; k++) {
        Y(j, k) = C[k](j, 0);
      }
    }
    std::cout << "delta_adj: " << delta_adj <<std::endl;
    arma::mat J = arma::kron(delta_adj, arma::ones<arma::mat>(m, 1));
    int mid = J.n_rows/2;
    arma::mat st = J.rows(0,mid-1);
    arma::mat Res1 = join_horiz(Y, J.rows(0,mid-1));
    Res1 = join_horiz(Res1,J.rows(mid,J.n_rows-1));
    std::cout << "Res1: " << Res1 <<std::endl;

    if (X.n_cols != Res1.n_cols) {
      X = Res1;
    } else {
      X = arma::join_vert(X, Res1);
    }
    std::cout << "i: "<< i <<std::endl;
  }

  this->X = X;
  // Adaptive grid
  // Cardinality
  int m = X.n_rows;
  // Extract the Target Set (creation of SafeSet minus TargetSet)
  // Indexing the target set
  arma::mat inner1 = arma::repmat(TargetSet.col(1), 1, m).t();
  arma::mat outer1 = X.cols(0, dim - 1);
  arma::mat lhs = arma::conv_to<arma::mat>::from(outer1 < inner1);

  arma::mat inner2 = arma::repmat(TargetSet.col(0), 1, m).t();
  arma::mat rhs = arma::conv_to<arma::mat>::from(outer1 > inner2);

  arma::mat all = lhs % rhs;
  arma::mat TargetIndex = arma::prod(all, 1);

  // Reshape X so that the first entries are the target set
  arma::uvec in1 = arma::find(TargetIndex == 1);
  arma::uvec in2 = arma::find(TargetIndex != 1);
  X = arma::join_vert(X.rows(in1), X.rows(in2));

  // The initial Lipschitz constants
  int k = arma::accu(TargetIndex) + 1;
  arma::mat h_ij = this->LocalLipschitz(k, m);

  arma::mat h_ij_bar = this->LocalLipschitzXbar(k, m);

  std::cout << "h_ij : " << h_ij.n_rows << ", " << h_ij.n_cols << std::endl;
  std::cout << "h_ij_bar : " << h_ij_bar.n_rows << ", " << h_ij_bar.n_cols << std::endl;

  // Definition of h_tot

  arma::mat internal = arma::pow(arma::sum(X(arma::span(k - 1, X.n_rows - 1),
                                             arma::span(dim, 2 * dim - 1))), 2);
  internal = arma::pow(internal.t(), 0.5).t();
  arma::mat internal2 = arma::repmat(internal, 1, m);
  std::cout << "internal2 : " << internal2.n_rows << ", " << internal2.n_cols << std::endl;
  arma::mat internal3 = arma::pow(arma::sum(X.cols(dim, 2 * dim - 1)), 2);
  internal3 = arma::pow(internal3.t(), 0.5);
  arma::mat internal4 = arma::repmat(internal3,(m - (k - 1)), 1);
  std::cout << "internal4 : " << internal4.n_rows << ", " << internal4.n_cols << std::endl;
  arma::mat h_tot = h_ij % internal2;// + h_ij_bar % internal4;
  std::cout << "h_tot : " << h_tot << std::endl;

  // Definition of K_i
  arma::mat K_i = h_tot * arma::prod(X.cols(dim, 2 * dim - 1).t(), 0).t();

  // The resulting local errors
  arma::mat E_i = arma::zeros<arma::mat>(1, 1);
  if (k > 1) {
    E_i = arma::join_vert(arma::zeros<arma::mat>(k - 1, 1), T * K_i);
  } else {
    E_i = T * K_i;
  }

  int q = k - 1;
  arma::mat Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
  while (arma::accu(Ei_comp) > 0) {
    double t = arma::max(arma::size(E_i));
    if (E_i.is_empty()) {
      t = 0;
    }
    int end = (t - q);
    for (int i = 0; i < end; ++i) {
      if (E_i(q) > epsilon) {
        // finding the index related to maximum dimension
        double m0 = X(q, arma::span(dim, 2 * dim - 1)).index_max();
        // std::cout << "m0:" << m0 << std::endl;

        // splitting the cell into two cells along its largest edge
        arma::mat Y = arma::repmat(X.row(q), 2, 1);
        //	std::cout << "Y:" << Y << std::endl;

        // the first smaller cell
        Y(0, dim + m0) = X(q, dim + m0) / 2;
        //		std::cout << "Y(0,dim+m0) :" << Y(0, dim + m0) <<
        //std::endl;

        Y(0, m0) = X(q, m0) - Y(0, dim + m0) / 2; // TODO check if m0 or m0-1
        //	std::cout << "Y(0,m0) :" << Y(0, m0) << std::endl;

        // the second smaller cell
        Y(1, dim + m0) = X(q, dim + m0) / 2;
        Y(1, m0) = X(q, m0) + Y(1, dim + m0) / 2;
        // std::cout << "Y(1,dim+m0) :" << Y(1, dim + m0) << std::endl;
        // std::cout << "Y(1,m0) :" << Y(1, m0) << std::endl;

        // Update X
        std::cout << "q: " << q << std::endl;

        if ((q - 1) < 0) {
          X = X.rows(q + 1, X.n_rows - 1);
        } else {
          X = arma::join_vert(X.rows(0, q - 1), X.rows(q + 1, X.n_rows - 1));
        }
        X = arma::join_vert(X, Y);

        // Update E_i
        if (i == (end - 1)) {
          E_i.clear();
        } else {
          if ((q - 1) < 0) {
            E_i = E_i.rows(q + 1, E_i.n_rows - 1);
          } else {
            E_i = arma::join_vert(E_i.rows(0, q - 1),
                                  E_i.rows(q + 1, E_i.n_rows - 1));
          }
        }
      } else {
        q++;
      }
    }
    // Store X
    this->X = X;

    // Update local error
    h_ij = this->LocalLipschitz(q, X.n_rows);
    h_ij_bar = this->LocalLipschitz(q, X.n_rows);

    // Definition of h_tot
    internal = arma::pow(
        arma::sum(X(arma::span(q, X.n_rows - 1), arma::span(dim, 2 * dim - 1))),
        2);
    internal = arma::pow(internal.t(), 0.5).t();
    internal2 = arma::repmat(internal, 1, X.n_rows);
    arma::mat internal3 = arma::pow(arma::sum(X.cols(dim, 2 * dim - 1)), 2);
    internal3 = arma::pow(internal.t(), 0.5).t();
    arma::mat internal4 = arma::repmat(internal3, X.n_rows - q + 1, 1);
    //	std::cout << "internal2 : " << internal2 << std::endl;
    // std::cout << "internal4 : " << internal4 << std::endl;
    arma::mat h_tot = h_ij % internal2 + h_ij_bar % internal4;
    // std::cout << "h_tot : " << h_tot << std::endl;

    // Definition of K_i
    arma::mat k_i = arma::prod(X.cols(dim, 2 * dim - 1).t(), 0);
    K_i = h_tot * k_i.t();

    // The resulting local errors
    if (!E_i.is_empty()) {
      arma::mat E_i_add = T*K_i;
      E_i = arma::join_vert(E_i, E_i_add);
    }
    else {
      E_i = T*K_i;
    }
    Ei_comp = arma::conv_to<arma::mat>::from(E_i >= epsilon);
    q = 0;
  }
  arma::mat E_temp = arma::max(E_i);

  this->E = E_temp(0, 0);
}

/////////////////// Controlled
// der : 0 - X, 1 - Xbar, 2 -U
double faust_t::GlobalLipschitz_Contr(arma::mat SafeSet, arma::mat InputSet,
                                      int der) {
  // Dimension of the Set and thus the state space
  int dim = SafeSet.n_rows;

  // Dimension of the input
  int dim_u = InputSet.n_rows;

  // Definition of the Kernel Function from the model
  double h = 0;

  // Global optimiser without need for defininig gradient
  nlopt::opt opt(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim * 2 + dim_u);
  nlopt::opt lopt(nlopt::LN_COBYLA, dim * 2 + dim_u);
  opt.set_local_optimizer(lopt);

  // Defining upper, lower bounds and starting point
  std::vector<double> lb(dim * 2 + dim_u);
  std::vector<double> ub(dim * 2 + dim_u);
  std::vector<double> x0(dim * 2 + dim_u);
  int count = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < dim; ++j) {
      lb[j + count] = SafeSet(j, 0);
      ub[j + count] = SafeSet(j, 1);
      x0[j + count] = arma::accu(SafeSet.row(j)) / 2;
    }
    count = dim;
  }
  for (int k = 2 * dim; k < 2 * dim + dim_u; ++k) {
    lb[k] = InputSet(k - 2 * dim, 0);
    ub[k] = InputSet(k - 2 * dim, 1);
    x0[k] = arma::accu(InputSet.row(dim_u-1)) / 2;
  }
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.remove_equality_constraints();
  opt.remove_inequality_constraints();

  opt.set_maxeval(5*500);
  switch (der) {
  case 0: {
    opt.set_min_objective(myfunc, (void *)&this->model);
    break;
  }
  case 1: {
    opt.set_min_objective(myfuncXbar, (void *)&this->model);
    break;
  }
  case 2: {
    opt.set_min_objective(myfuncU, (void *)&this->model);
    break;
  }
  }

  opt.set_xtol_rel(1e-10);
  lopt.set_xtol_rel(1e-12);

  opt.set_ftol_abs(1e-30);
  lopt.set_ftol_abs(1e-32);
  double minf;
  try {
    nlopt::result result = opt.optimize(x0, minf);
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    exit(0);
  }

  h = -minf;
  if (h < 1e-5) {
    std::cout << "The Lipschitz constant is very small:" << h << ". Please check if this "
                 "is according to expectations. If not, this can be solved by "
                 "adjusting the TolFun parameter" <<std::endl;
    std::cout << "Setting Lipschitz constant to 1e-5 " << std::endl;
    h = 1e-5;
  }

  return h;
}

// Numerically computes the local Lipschitz constants by taking the maximum
// local derivative over six different points for a system having control inputs
arma::cube faust_t::LocalLipschitz_Contr(int xl, int xu, int ul, int uu) {

  // Get X
  arma::mat X = this->X;
  // Get U
  arma::mat U = this->U;

  // Cardinality of the system
  int m = X.n_rows;

  // Dimension of the system
  int dim = X.n_cols/2;

  // Dimension of the input
  int dim_u = U.n_cols/2;

  // Initialisation
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};

  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }
  Kernel = this->Kernel;

  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim; i++) {
    if (i == 0) {
      te = Kernel.diff(get_symbol(x_alpha[i]), 1);
    } else {
      te = Kernel.diff(get_symbol(x_alpha[i + dim - 1]), 1);
    }

    P2.push_back(te);
  }

  // Selection matrix for h_ijr
  arma::mat temp;
  temp << 0 << 1;
  arma::mat t2 = repmat(temp, 1, dim * 2 + dim_u);
  arma::mat t3 = nchoosek(t2, dim * 2 + dim_u);
  arma::mat Ct = unique_rows(t3);
  // Convert to matrix of decimal numbers and then sort
  arma::mat getIndex = binary_to_decimal(Ct);
  Ct = fliplr(Ct.rows(arma::stable_sort_index(getIndex)));
  arma::mat C4 = -0.5 * arma::ones<arma::mat>(Ct.n_rows, Ct.n_cols);
  arma::mat C = arma::zeros<arma::mat>(Ct.n_rows, Ct.n_cols);
  C = Ct + C4;
  // Fast but memory expansive way of computing
  std::vector<std::vector<arma::cube>> P;
  std::vector<arma::cube> intP;
  arma::cube tempP((xu - xl + 1), m, (uu - ul + 1));
  // Creation of all the corner points to check (4-D array)
  for (int i = 0; i < dim; ++i) {
    for (int k = 0; k < pow(2, ((2 * dim + dim_u))); ++k) {
      for (int j = 0; j < (uu - ul + 1); ++j) {
        arma::mat tempInner = arma::repmat(
            X(arma::span(xl - 1, xu - 1), i) +
                X(arma::span(xl - 1, xu - 1), dim + i ) * C(k, i),
            1, m);
        arma::mat tempInnerNew = reshape(tempInner, (xu - xl + 1),m);
        tempP.slice(j) =tempInnerNew;
      }
      intP.push_back(tempP);
    }
    P.push_back(intP);
    intP.clear();
  }
  tempP.clear();
  tempP = arma::zeros<arma::cube>((xu - xl + 1), m, (uu - ul + 1));
  for (int i = dim; i < 2 * dim; ++i) {
    for (int k = 0; k < pow(2, ((2 * dim + dim_u))); ++k) {
      for (int j = 0; j < (uu - ul + 1); ++j) {
        arma::mat tempInner = arma::repmat(
            X.col(i - dim).t() + X.col(i).t() * C(k, i), (xu - xl + 1), 1);
        arma::mat tempInnerNew = reshape(tempInner, (xu - xl + 1),m);
        tempP.slice(j) = tempInnerNew;
      }
      intP.push_back(tempP);
    }
    P.push_back(intP);
    intP.clear();
  }
  tempP.clear();
  tempP = arma::zeros<arma::cube>((xu - xl + 1), m, (uu-ul +1));
  for (int i = 2 * dim; i < 2 * dim + dim_u; ++i) {
    for (int k = 0; k < pow(2, ((2 * dim + dim_u))); ++k) {
      for (int j = 0; j < (uu-ul +1); ++j) {
        arma::mat inner = U(arma::span(ul - 1, uu - 1), i - 2 * dim).t() + U(arma::span(ul-1, uu-1), dim_u+i - 2*dim).t()*C(k,i);
        arma::mat tempInner = arma::repmat( inner.col(j),(xu - xl + 1), m);
        arma::mat tempInnerNew = reshape(tempInner, (xu - xl + 1),m);
        tempP.slice(j) = tempInnerNew;

      }
      intP.push_back(tempP);
    }

    P.push_back(intP);
    intP.clear();
  }
  tempP.clear();
  // Creation of the local minima and maxima
  std::cout << "Creation of the local Lipschitz constants is in progress" << std::endl;
  // Evaluate differential of Kernel
  std::vector<double> x;
  arma::vec maxEl(m);
  std::vector<arma::cube> h_ijr_aux;
  std::vector<double> accu2FindMax;

  arma::cube h_ijr(P[0][0].n_rows, P[0][0].n_cols, P[0][0].n_slices);
  for (unsigned int n = 0; n < P[0].size(); ++n) {
    for (unsigned int l = 0; l < P[0][0].n_rows; ++l) {
      for (unsigned int nn = 0; nn < P[0][0].n_cols; ++nn) {
        for (unsigned int nr = 0; nr < P[0][0].n_slices; ++nr) {
          for (int k = 0; k < dim * 2 + dim_u; ++k) {
            x.push_back(P[k][n](l,nn, nr));
          }
          h_ijr(l,nn,nr) = evalDiffKernel_Contr(P2, x, dim_u);
          x.clear();
        }
      }
    }
    h_ijr_aux.push_back(h_ijr);
    accu2FindMax.push_back(arma::accu(h_ijr));
  }
  std::vector<double>::iterator res =
      std::max_element(accu2FindMax.begin(), accu2FindMax.end());
  int inRes = std::distance(accu2FindMax.begin(), res);
  return h_ijr_aux[inRes];
}

// Numerically computes the local Lipschitz constants by taking the maximum
// local derivative over six different points for a system having control inputs
arma::cube faust_t::LocalLipschitzToU_Contr(int xl, int xu, int ul, int uu) {
  // Get X
  arma::mat X = this->X;
  // Get U
  arma::mat U = this->U;
  // Cardinality of the system
  int m = X.n_rows;

  // Dimension of the system
  int dim = this->model.x_mod[0].A.n_cols;

  // Dimension of the input
  int dim_u = this->model.x_mod[0].B.n_cols;

  // TODO generalise
  std::string x_alpha[26] = {"x1",   "x1b", "x2",   "x2b", "x3",  "x3b",  "x4",
                             "x4b",  "x5",  "x5b",  "x6",  "x6b", "x7",   "x7b",
                             "x8",   "x8b", "x9",   "x9b", "x10", "x10b", "x11",
                             "x11b", "x12", "x12b", "x13", "x13b"};

  std::string u_alpha[13] = {"u1", "u2", "u3",  "u4",  "u5",  "u6", "u7",
                             "u8", "u9", "u10", "u11", "u12", "u13"};

  GiNaC::lst symbols = {};
  GiNaC::symbol t;
  for (int j = 0; j < 2 * dim + dim_u; j++) {
    if (j < 2 * dim) {
      t = get_symbol(x_alpha[j]);
    } else {
      t = get_symbol(u_alpha[j - 2 * dim]);
    }
    symbols.append(t);
  }
  Kernel = this->Kernel;
  std::vector<GiNaC::ex> P2;
  GiNaC::ex te;
  for (int i = 0; i < dim_u; i++) {
    te = Kernel.diff(get_symbol(u_alpha[i]), 1);
    P2.push_back(te);
  }

  // Selection matrix for h_ijr
  arma::mat temp;
  temp << 0 << 1;
  arma::mat t2 = repmat(temp, 1, dim * 2 + dim_u);
  arma::mat t3 = nchoosek(t2, dim * 2 + dim_u);
  arma::mat Ct = unique_rows(t3);

  // Convert to matrix of decimal numbers and then sort
  arma::mat getIndex = binary_to_decimal(Ct);
  Ct = fliplr(Ct.rows(arma::stable_sort_index(getIndex)));
  arma::mat C4 = -0.5 * arma::ones<arma::mat>(Ct.n_rows, Ct.n_cols);
  arma::mat C = arma::zeros<arma::mat>(Ct.n_rows, Ct.n_cols);
  C = Ct + C4;

  // Fast but memory expansive way of computing
  std::vector<std::vector<arma::cube>> P;
  std::vector<arma::cube> intP;
  arma::cube tempP((xu - xl + 1), m, (uu - ul + 1));
  // Creation of all the corner points to check (4-D array)o
  for (int i = 0; i < dim; ++i) {
    for (int k = 0; k < pow(2, ((2 * dim + dim_u))); ++k) {
      for (int j = 0; j < (uu - ul + 1); ++j) {
        arma::mat tempInner = arma::repmat(
            X(arma::span(xl - 1, xu - 1), i) +
                X(arma::span(xl - 1, xu - 1), dim + i ) * C(k, i),
            1, m);
        arma::mat tempInnerNew = reshape(tempInner, (xu - xl + 1),m);
        tempP.slice(j) =tempInnerNew;
      }
      intP.push_back(tempP);
    }
    P.push_back(intP);
    intP.clear();
  }
  tempP.clear();
  tempP = arma::zeros<arma::cube>((xu - xl + 1), m, (uu - ul + 1));
  for (int i = dim; i < 2 * dim; ++i) {
    for (int k = 0; k < pow(2, ((2 * dim + dim_u))); ++k) {
      for (int j = 0; j < (uu - ul + 1); ++j) {
        arma::mat tempInner = arma::repmat(
            X.col(i - dim).t() + X.col(i).t() * C(k, i), (xu - xl + 1), 1);
        arma::mat tempInnerNew = reshape(tempInner,(xu - xl + 1),m);
        tempP.slice(j) =tempInnerNew;
      }
      intP.push_back(tempP);
    }
    P.push_back(intP);
    intP.clear();
  }

  tempP.clear();
  tempP = arma::zeros<arma::cube>((xu - xl + 1), m, (uu-ul +1));
  for (int i = 2 * dim; i < 2 * dim + dim_u; ++i) {
    for (int k = 0; k < pow(2, ((2 * dim + dim_u))); ++k) {
      for (int j = 0; j < (uu-ul +1); ++j) {
        arma::mat inner = U(arma::span(ul - 1, uu - 1), i - 2 * dim).t() + U(arma::span(ul-1, uu-1), dim_u+i - 2*dim).t()*C(k,i);
        arma::mat tempInner  = arma::repmat( inner.col(j),(xu - xl + 1), m);
        arma::mat tempInnerNew = reshape(tempInner,(xu - xl + 1),m);
        tempP.slice(j) =tempInnerNew;
      }
      intP.push_back(tempP);
    }

    P.push_back(intP);
    intP.clear();
  }

  tempP.clear();
  // Creation of the local minima and maxima
  std::cout << "Creation of the local Lipschitz constants is in progress" 	<< std::endl;

  // Evaluate differential of Kernel
  std::vector<double> x;
  arma::vec maxEl(pow(2, ((2 * dim))));
  std::vector<arma::cube> h_ijr_aux;
  std::vector<double> accu2FindMax;
  arma::cube h_ijr(P[0][0].n_rows, P[0][0].n_cols, P[0][0].n_slices);

  for (unsigned int n = 0; n < P[0].size(); ++n) {
    for (unsigned int l = 0; l < P[0][0].n_rows; ++l) {
      for (unsigned int nn = 0; nn < P[0][0].n_cols; ++nn) {
        for (unsigned int nr = 0; nr < P[0][0].n_slices; ++nr) {
          for (int k = 0; k < dim * 2 + dim_u; ++k) {
            x.push_back(P[k][n](l,nn, nr));
          }
          h_ijr(l,nn,nr) = evalDiffKernel_Contr(P2, x, dim_u);
          x.clear();
        }
      }
    }
    h_ijr_aux.push_back(h_ijr);
    accu2FindMax.push_back(arma::accu(h_ijr));
  }

  std::vector<double>::iterator res =
      std::max_element(accu2FindMax.begin(), accu2FindMax.end());
  int inRes = std::distance(accu2FindMax.begin(), res);
  return h_ijr_aux[inRes];
}
double myx1constraint(const std::vector<double> &x, std::vector<double> &grad,
                      void *data) {
  return -x[0];
}
double myx2constraint(const std::vector<double> &x, std::vector<double> &grad,
                      void *data) {
  return -x[1];
}
double myx3constraint(const std::vector<double> &x, std::vector<double> &grad,
                      void *data) {
  auto *d = (std::vector<double> *)(data);
  double epsilon = d[0][0], h_a = d[0][1], h_u = d[0][2], L_A = d[0][3],
         T = d[0][4];
  return ((2 * T * L_A * h_a * x[0]) + (2 * T * L_A * h_u * x[1]) - epsilon);
}

double ff_deltas(const std::vector<double> &x, std::vector<double> &grad,
                 void *my_func_data) {

  auto *dim = (std::vector<double> *)my_func_data; // dim, dim_u
  // std::cout << "dim: " << dim[0][0] << ", " << dim[0][1] <<std::endl;
  double x_d = 2 * dim[0][0];
  double u_d = dim[0][1];
  // std::cout << "x: " << x[0] << ", " << x[1] << std::endl;
  if (!grad.empty()) {
    std::cout << -x_d * (std::pow(x[0], (x_d - 1)) * (std::pow(x[1], (u_d))))
              << std::endl;
    grad[0] = -x_d * (std::pow(x[0], (x_d - 1)) * (std::pow(x[1], (u_d))));
    std::cout << -u_d * (std::pow(x[0], x_d - 1) * (std::pow(x[1], u_d - 1)))
              << std::endl;
    grad[1] = -u_d * (std::pow(x[0], x_d - 1) * (std::pow(x[1], u_d - 1)));
  }

  //  std::cout << "Obj: " << -(std::pow(x[0],x_d)*(std::pow(x[1],u_d)))
  //  <<std::endl;
  return -(std::pow(x[0], x_d) * std::pow(x[1], u_d));
}

// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*h*N*L(A). Here L(A)
// denotes the Lebesgue measure of the set A, N the number of time steps, h  is
// the global Lipschitz constant and delta the dimension of the cells.
void faust_t::Uniform_grid_Contr(double epsilon, double T, arma::mat SafeSet,
                                 arma::mat InputSet) {

  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The dimension of the input of system
  double dim_u = InputSet.n_rows;

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  //  Derivation of the global Lipschitz constant
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  // Set optimisation function
  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate the solution to the length of the edges
  double delta_a = solution[0];
  delta_a = delta_a / sqrt(dim);
  double delta_u = solution[1];
  delta_u = delta_u / sqrt(dim_u);

  // Create the location of the representative points
  // delta_a is adjusted so that the edges of the set are part of the partitions
  // Create grid
  std::vector<arma::mat> C;
  arma::mat delta_adj_a = arma::zeros<arma::mat>(1, dim);
  for (int i = 0; i < dim; ++i) {
    delta_adj_a.col(i) = (SafeSet(i, 1) - SafeSet(i, 0)) /
                         std::ceil((SafeSet(i, 1) - SafeSet(i, 0)) / delta_a);
    double start = SafeSet(i, 0) + 0.5 * delta_adj_a(i);
    double Delta = delta_adj_a(i);
    double end = SafeSet(i, 1) - 0.499 * delta_adj_a(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::vec A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }
  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  int m = C[0].n_elem;

  // Make the output
  arma::mat X = arma::zeros<arma::mat>(m, 2 * dim);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < dim; ++j) {
      C[j].reshape(m, 1);
      X(i, j) = C[j](i, 0);
    }
  }

  X(arma::span(0, m - 1), arma::span(dim, 2 * dim - 1)) =
      arma::kron(delta_adj_a, arma::ones<arma::mat>(m, 1));

  C.clear();
  // Create the location of the representative points
  // delta_u is adjusted so that the edges of the set are part of the partitions
  arma::mat delta_adj_u = arma::zeros<arma::mat>(1, dim_u);
  for (int i = 0; i < dim_u; ++i) {
    delta_adj_u.col(i) = (InputSet(i, 1) - InputSet(i, 0)) /
                         std::ceil((InputSet(i, 1) - InputSet(i, 0)) / delta_u);
    double start = InputSet(i, 0) + 0.5 * delta_adj_u(i);
    double Delta = delta_adj_u(i);
    double end = InputSet(i, 1) - 0.499 * delta_adj_u(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  m = C[0].n_elem;

  // Make the output
  arma::mat U = arma::zeros<arma::mat>(m, 2 * dim_u);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < dim_u; j++) {
      C[j].reshape(m, 1);
      U(i, j) = C[j](i, 0);
    }
  }
  U(arma::span(0, m - 1), arma::span(dim_u, 2 * dim_u - 1)) =
      arma::kron(delta_adj_u, arma::ones<arma::mat>(m, 1));

  // The error made
  // The factor 2 comes from the fact that the system is controlled and the
  // control based on the abstracted system is applied to the original system
  double in1 = arma::accu((delta_adj_u%delta_adj_u));
  double inner = h_u * std::sqrt(in1);
  double in2 = arma::accu(delta_adj_a%delta_adj_a);

  double outer = h_a * std::sqrt(in2);
  double E = (inner + outer) * L_A(0, 0) * T * 2;

  this->X = X;
  this->U = U;
  this->E = E;
}

// Uniform Markov Chain approximation
// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*(h+hbar)*N*L(A). Here L(A) denotes the
// Lebesgue measure of the set A, T the number of time steps, h is the global
// Lipschitz constant with respect to x, while hbar is the global Lipschitz
// constant with respect to xbar. delta is measure of the dimension of the
// cells. Input of this function is the desired maximal error epsilon, the model
// used to describe the Kernel, the number of time steps N and the upper and
// lower bounds of the (Safe)Set. The output X is a matrix, which consists of
// the centres of the cell as well  as the length of the edges.
void faust_t::Uniform_grid_MCapprox_Contr(double epsilon, double T,
                                          arma::mat SafeSet,
                                          arma::mat InputSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The dimension of the input of system
  double dim_u = InputSet.n_rows;

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  //  Derivation of the global Lipschitz constant
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  double h_s = this->GlobalLipschitz_Contr(SafeSet, InputSet, 1);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();

  opt.set_maxeval(5*500);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a + h_s;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;


  // Set optimisation function
  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate the solution to the length of the edges
  double delta_a = solution[0];
  delta_a = delta_a / sqrt(dim);
  double delta_u = solution[1];
  delta_u = delta_u / sqrt(dim_u);

  // Warn if a dimension of the set has not been partitioned.
  // This implies that in this direction no gradient will be found
  arma::mat rd = r - delta_a;

  arma::umat rd0 = (rd < 0);
  double ineq = arma::accu(rd0);

  if (ineq > 0) {
    std::cout << "A dimension of the SafeSet has not been partitioned, "
                 "reshaping the SafeSet might solve this issue"
              << std::endl;
  }
  arma::mat ud = u - delta_u;
  arma::umat ud0 = (ud < 0);
  double inequ = arma::accu(ud0);

  if (inequ > 0) {
    std::cout << "A dimension of the InputSet has not been partitioned, "
                 "reshaping the SafeSet might solve this issue"
              << std::endl;
  }

  // Create the location of the representative points
  // delta is adjusted so that the edges of the set are part of the partitions
  std::vector<arma::mat> C;
  arma::mat delta_adj_a = arma::zeros<arma::mat>(1, dim);
  for (int i = 0; i < dim; i++) {
    delta_adj_a.col(i) = (SafeSet(i, 1) - SafeSet(i, 0)) /
                         std::ceil((SafeSet(i, 1) - SafeSet(i, 0)) / delta_a);
    double start = SafeSet(i, 0) + 0.5 * delta_adj_a(i);
    double Delta = delta_adj_a(i);
    double end = SafeSet(i, 1) - 0.499 * delta_adj_a(i);
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }
  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  int m = C[0].n_elem;
  // Make the output
  arma::mat X = arma::zeros<arma::mat>(m, 2 * dim);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < dim; ++j) {
      C[j].reshape(m, 1);
      X(i, j) = C[j](i, 0);
    }
  }
  X(arma::span(0, m - 1), arma::span(dim, 2 * dim - 1)) =
      arma::kron(delta_adj_a, arma::ones<arma::mat>(m, 1));

  C.clear();
  // Create the location of the representative points
  // delta_u is adjusted so that the edges of the set are part of the partitions
  arma::mat delta_adj_u = arma::zeros<arma::mat>(1, dim_u);
  for (int i = 0; i < dim_u; ++i) {
    delta_adj_u.col(i) = (InputSet(i, 1) - InputSet(i, 0)) /
                         std::ceil((InputSet(i, 1) - InputSet(i, 0)) / delta_u);
    double start = InputSet(i, 0) + 0.5 * delta_adj_u(i);
    double Delta = delta_adj_u(i);
    double end = InputSet(i, 1) - 0.499 * delta_adj_u(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  m = C[0].n_elem;

  // Make the output
  arma::mat U = arma::zeros<arma::mat>(m, 2 * dim_u);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < dim_u; j++) {
      C[j].reshape(m, 1);
      U(i, j) = C[j](i, 0);
    }
  }
  U(arma::span(0, m - 1), arma::span(dim_u, 2 * dim_u - 1)) =
      arma::kron(delta_adj_u, arma::ones<arma::mat>(m, 1));

  // The error made
  // The factor 2 comes from the fact that the system is controlled and the
  // control based on the abstracted system is applied to the original system
  double inner = h_u * std::sqrt(arma::accu(arma::pow(delta_adj_u, 2)));
  double outer = (h_a + h_s) * std::sqrt(arma::accu(arma::pow(delta_adj_a, 2)));
  double E = (inner + outer) * L_A(0, 0) * T * 2;

  this->X = X;
  this->U = U;
  this->E = E;
}

// Uniform Markov Chain approximation for reach and avoid problem
// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*(h+hbar)*N*L(A). Here L(A) denotes the
// Lebesgue measure of the set A, T the number of time steps, h is the global
// Lipschitz constant with respect to x, while hbar is the global Lipschitz
// constant with respect to xbar. delta is measure of the dimension of the
// cells. Input of this function is the desired maximal error epsilon, the model
// used to describe the Kernel, the number of time steps N and the upper and
// lower bounds of the (Safe)Set. The output X is a matrix, which consists of
// the centres of the cell as well  as the length of the edges.
void faust_t::Uniform_grid_ReachAvoid_Contr(double epsilon, double T,
                                            arma::mat SafeSet,
                                            arma::mat InputSet,
                                            arma::mat TargetSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Safe Set
  arma::mat rS = SafeSet.col(1) - SafeSet.col(0);

  // The length of the edges of the Target Set
  arma::mat rT = TargetSet.col(1) - TargetSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(rS);

  // The dimension of the input of system
  double dim_u = InputSet.n_rows;

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  // Derivation of the global Lipschitz constants wrt U
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2);
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  // Set optimisation function
  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate the solution to the length of the edges
  double delta_a = solution[0];
  delta_a = delta_a / sqrt(dim);
  double delta_u = solution[1];
  delta_u = delta_u / sqrt(dim_u);


  // Delta is adjusted so that the edges of the TargetSet are part of the
  // partitions
  arma::mat delta_adj_a = arma::zeros<arma::mat>(1, dim);
  for (unsigned i = 0; i < dim; ++i) {
    delta_adj_a.col(i) =
        (TargetSet(i, 1) - TargetSet(i, 0)) /
        std::ceil((TargetSet(i, 1) - TargetSet(i, 0)) / delta_a);
  }

  // The adjusted SafeSet to fit with delta_adj
  arma::mat Residuals = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  arma::mat Rin = SafeSet - TargetSet;
  arma::mat Rou = arma::repmat(delta_adj_a.t(), 1, 2);
  arma::mat Rot = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  Rot = arma::join_horiz(-delta_adj_a.t(), 0 * delta_adj_a.t());
  Rou = modM(Rin, Rou);
  Residuals = Rou + Rot;
  arma::mat SafeSet_adj = SafeSet - Residuals;

  // Partition (Safe-Target) into 3^dim-1 separate parts
  arma::mat D = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  arma::mat Dt = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  std::vector<arma::mat> D1;
  std::vector<arma::mat> Dt1;
  for (int i = 0; i < dim; i++) {
    arma::mat tempD;
    tempD << SafeSet(i, 0);
    tempD = join_horiz(tempD, SafeSet_adj.row(i)); // concatenate
    arma::mat tempDD;
    tempDD << SafeSet(i, 1);
    tempD = join_horiz(tempD, tempDD);
    D = unique(tempD);
    Dt = D.cols(0, D.n_cols - 2);
    arma::vec tempV = D.t(); // convert to vector
    D1.push_back(tempV);
    arma::vec tempVt = Dt.t();
    Dt1.push_back(tempVt);
  }

  std::vector<arma::mat> F;
  F = nDgrid(Dt1);

  // Cardinality
  double n = F[0].n_elem;

  // Make the output
  arma::mat X; // Initialisation of X with an empty matrix
  for (int i = 0; i < n; i++) {
    arma::mat Set = arma::zeros<arma::mat>(dim, 2);
    std::vector<arma::mat> C;
    arma::mat delta_adj_local = arma::zeros<arma::mat>(1, dim);
    int index = 0;
    for (int j = 0; j < dim; j++) {
      arma::umat inu = ind2sub(arma::size(F[j]),i);
      double left = F[j](inu(0),inu(1));
      for (unsigned k = 0; k < D1[j].n_rows; k++) {
        if (D1[j](k, 0) == left) {
          index = k;
        }
      }
      double right = D1[j](index + 1, 0);
      arma::mat templ;
      templ << left;
      arma::mat tempr;
      tempr << right;
      Set.row(j) = join_horiz(templ, tempr);

      // The ndiv variable eliminates rounding errors
      double ndiv = round((Set(j, 1) - Set(j, 0)) / delta_adj_a(j));
      if (ndiv == 0) {
        ndiv = 1;
      }

      delta_adj_local(j) = (Set(j, 1) - Set(j, 0)) / ndiv;
      double start = Set(j, 0) + 0.5 * delta_adj_local(j);
      double Delta = delta_adj_local(j);
      double end = Set(j, 1) - 0.499 * delta_adj_local(j);
      arma::mat A = arma::regspace(start, Delta, end);
      C.push_back(A);
      // 0.499 adjusts for rounding errors; it has no influence on the
      // position of the representative points
    }

    // Construct grid
    C = nDgrid(C);

    // Cardinality
    int m = C[0].n_elem;

    // Make the output
    arma::mat Y = arma::zeros<arma::mat>(m, dim);
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < dim; k++) {
	arma::umat inu = ind2sub(arma::size(C[k]),j);
        Y(j, k) = C[k](inu(0),inu(1));
      }
    }

    arma::mat J = arma::kron(delta_adj_local, arma::ones<arma::mat>(m, 1));

    arma::mat Res1 = join_horiz(Y, J);
    if (X.n_cols != Res1.n_cols) {
      X = Res1;
    } else {
      X = arma::join_vert(X, Res1);
    }
    C.clear();
  }
  this->X = X;

  // Create the location of the representative points
  // delta_u is adjusted so that the edges of the set are part of the partitions
  arma::mat delta_adj_u = arma::zeros<arma::mat>(1, dim_u);
  std::vector<arma::mat> C;
  for (int i = 0; i < dim_u; ++i) {
    delta_adj_u.col(i) = (InputSet(i, 1) - InputSet(i, 0)) /
                         std::ceil((InputSet(i, 1) - InputSet(i, 0)) / delta_u);
    double start = InputSet(i, 0) + 0.5 * delta_adj_u(i);
    double Delta = delta_adj_u(i);
    double end = InputSet(i, 1) - 0.499 * delta_adj_u(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  double m = C[0].n_elem;
  double sm = std::sqrt(m);

  // Make the output
  arma::mat U = arma::zeros<arma::mat>(m, 2 * dim_u);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < dim_u; j++) {
      C[j].reshape(m, 1);
      U(i, j) = C[j](i, 0);
    }
  }
  U(arma::span(0, m - 1), arma::span(dim_u, 2 * dim_u - 1)) =
      arma::kron(delta_adj_u, arma::ones<arma::mat>(m, 1));

  // The error made
  // The factor 2 comes from the fact that the system is controlled and the
  // control based on the abstracted system is applied to the original system
  double inner = h_u * std::sqrt(arma::accu(arma::pow(delta_adj_u, 2)));
  double outer = h_a * std::sqrt(arma::accu(arma::pow(delta_adj_a, 2)));
  double E = (inner + outer) * L_A(0, 0) * T * 2;

  this->X = X;
  this->U = U;
  this->E = E;
}

// Uniform Markov Chain approximation for reach and avoid problem
// This function makes a uniform grid that will produce a certain maximal
// abstraction error defined by E=delta*(h+hbar)*N*L(A). Here L(A) denotes the
// Lebesgue measure of the set A, T the number of time steps, h is the global
// Lipschitz constant with respect to x, while hbar is the global Lipschitz
// constant with respect to xbar. delta is measure of the dimension of the
// cells. Input of this function is the desired maximal error epsilon, the model
// used to describe the Kernel, the number of time steps N and the upper and
// lower bounds of the (Safe)Set. The output X is a matrix, which consists of
// the centres of the cell as well  as the length of the edges.
void faust_t::Uniform_grid_ReachAvoidMCapprox_Contr(double epsilon, double T,
                                                    arma::mat SafeSet,
                                                    arma::mat InputSet,
                                                    arma::mat TargetSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Safe Set
  arma::mat rS = SafeSet.col(1) - SafeSet.col(0);

  // The length of the edges of the Target Set
  arma::mat rT = TargetSet.col(1) - TargetSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(rS);

  // The dimension of the input of system
  double dim_u = InputSet.n_rows;

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  //  Derivation of the global Lipschitz constant
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  double h_s = this->GlobalLipschitz_Contr(SafeSet, InputSet, 1);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a + h_s;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate the solution to the length of the edges
  double delta_a = solution[0];
  delta_a = delta_a / sqrt(dim);
  double delta_u = solution[1];
  delta_u = delta_u / sqrt(dim_u);

  // Warn if a dimension of the set has not been partitioned.
  // This implies that in this direction no gradient will be found
  arma::mat rd = rS - delta_a;
  arma::umat rd0 = (rd < 0);
  double ineq = arma::accu(rd0);
  if (ineq > 0) {
    std::cout << "A dimension of the SafeSet has not been partitioned, "
                 "reshaping the SafeSet might solve this issue"
              << std::endl;
  }
  arma::mat ud = u - delta_u;
  arma::umat ud0 = (ud < 0);
  double inequ = arma::accu(ud0);
  if (inequ > 0) {
    std::cout << "A dimension of the InputSet has not been partitioned, "
                 "reshaping the SafeSet might solve this issue"
              << std::endl;
  }

  // Delta is adjusted so that the edges of the TargetSet are part of the
  // partitions
  arma::mat delta_adj_a = arma::zeros<arma::mat>(1, dim);
  for (unsigned i = 0; i < dim; ++i) {
    delta_adj_a(i) = (TargetSet(i, 1) - TargetSet(i, 0)) /
                     std::ceil((TargetSet(i, 1) - TargetSet(i, 0)) / delta_a);
  }

  // The adjusted SafeSet to fit with delta_adj
  arma::mat Residuals = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  arma::mat Rin = SafeSet - TargetSet;
  arma::mat Rou = arma::repmat(delta_adj_a.t(), 1, 2);
  arma::mat Rot = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  Rot = arma::join_horiz(-delta_adj_a.t(), 0 * delta_adj_a.t());
  Residuals = modM(Rin, Rou) + Rot;
  arma::mat SafeSet_adj = SafeSet - Residuals;

  // Partition (Safe-Target) into 3^dim-1 separate parts
  arma::mat D = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  arma::mat Dt = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  std::vector<arma::mat> D1;
  std::vector<arma::mat> Dt1;
  for (int i = 0; i < dim; i++) {
    arma::mat tempD;
    tempD << SafeSet(i, 0);
    tempD = join_horiz(tempD, SafeSet_adj.row(i)); // concatenate
    arma::mat tempDD;
    tempDD << SafeSet(i, 1);
    tempD = join_horiz(tempD, tempDD);
    D = unique(tempD);
    Dt = D.cols(0, D.n_cols - 2);
    arma::vec tempV = D.t(); // convert to vector
    D1.push_back(tempV);
    arma::vec tempVt = Dt.t();
    Dt1.push_back(tempVt);
  }

  std::vector<arma::mat> F;
  F = nDgrid(Dt1);

  // Cardinality
  double n = F[0].n_elem;

  // Make the output
  arma::mat X; // Initialisation of X with an empty matrix
  for (int i = 0; i < n; i++) {
    arma::mat Set = arma::zeros<arma::mat>(dim, 2);
    std::vector<arma::mat> C;
    arma::mat delta_adj_local = arma::zeros<arma::mat>(1, dim);
    int index = 0;
    for (int j = 0; j < dim; j++) {
      arma::uvec inu = ind2sub(arma::size(F[j]), i);
      double left = F[j](inu(0), inu(1) );
      for (unsigned k = 0; k < D1[j].n_rows; k++) {
        if (D1[j](k, 0) == left) {
          index = k;
        }
      }
      double right = D1[j](index + 1, 0);
      arma::mat templ;
      templ << left;
      arma::mat tempr;
      tempr << right;
      Set.row(j) = join_horiz(templ, tempr);
      // The ndiv variable eliminates rounding errors
      double ndiv = round((Set(j, 1) - Set(j, 0)) / delta_adj_a(j));
      if (ndiv == 0) {
        ndiv = 1;
      }
      delta_adj_local(j) = (Set(j, 1) - Set(j, 0)) / ndiv;
      double start = Set(j, 0) + 0.5 * delta_adj_local(j);
      double Delta = delta_adj_local(j);
      double end = Set(j, 1) - 0.499 * delta_adj_local(j);
      arma::mat A = arma::regspace(start, Delta, end);
      C.push_back(A);
      // 0.499 adjusts for rounding errors; it has no influence on the
      // position of the representative points
    }

    // Construct grid
    C = nDgrid(C);

    // Cardinality
    int m = C[0].n_elem;

    // Make the output
    arma::mat Y = arma::zeros<arma::mat>(m, dim);
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < dim; k++) {
       arma::uvec inu = ind2sub(arma::size(C[k]), j);
       Y(j, k) = C[k](inu(0), inu(1));
      }
    }
    arma::mat J = arma::kron(delta_adj_local, arma::ones<arma::mat>(m, 1));
    arma::mat Res1 = join_horiz(Y, J);
    if (X.n_cols != Res1.n_cols) {
      X = Res1;
    } else {
      X = arma::join_vert(X, Res1);
    }
    C.clear();
  }
  this->X = X;

  // Create the location of the representative points
  // delta_u is adjusted so that the edges of the set are part of the partitions
  arma::mat delta_adj_u = arma::zeros<arma::mat>(1, dim_u);
  std::vector<arma::mat> C;
  for (int i = 0; i < dim_u; ++i) {
    delta_adj_u.col(i) = (InputSet(i, 1) - InputSet(i, 0)) /
                         std::ceil((InputSet(i, 1) - InputSet(i, 0)) / delta_u);
    double start = InputSet(i, 0) + 0.5 * delta_adj_u(i);
    double Delta = delta_adj_u(i);
    double end = InputSet(i, 1) - 0.499 * delta_adj_u(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::mat A = arma::regspace(start, Delta, end);
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  double m = C[0].n_elem;

  // Make the output
  arma::mat U = arma::zeros<arma::mat>(m, 2 * dim_u);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < dim_u; j++) {
      C[j].reshape(m, 1);
      U(i, j) = C[j](i, 0);
    }
  }
  U(arma::span(0, m - 1), arma::span(dim_u, 2 * dim_u - 1)) =
      arma::kron(delta_adj_u, arma::ones<arma::mat>(m, 1));

  // The error made
  // The factor 2 comes from the fact that the system is controlled and the
  // control based on the abstracted system is applied to the original system
  double inner = h_u * std::sqrt(arma::accu(arma::pow(delta_adj_u, 2)));
  double outer = (h_a + h_s) * std::sqrt(arma::accu(arma::pow(delta_adj_a, 2)));
  double E = (inner + outer) * L_A(0, 0) * T * 2;

  this->X = X;
  this->U = U;
  this->E = E;
}

// Computes the safety probability given N time steps
// and the transition matrix Tp, System has control inputs
void faust_t::StandardProbSafety_Contr(int T) {

  // The Cardinality of X
  int m = this->Tp[0].n_rows;

  // The Cardinality of U
  int q = this->Tp.size();

  arma::mat V = arma::ones<arma::mat>(m, 1);
  arma::umat OptimalPol = arma::zeros<arma::umat>(m, T);

  // need to convert it to cube frm std::vector
  arma::cube Tp(this->Tp[0].n_rows, this->Tp[0].n_cols, this->Tp.size());
  for (unsigned i = 0; i < this->Tp.size(); i++) {
    Tp.slice(i) = this->Tp[i];
  }
  arma::cube in = arma::zeros<arma::cube>(m, m, q);
  arma::cube Vaux = arma::zeros<arma::cube>(m, 1, q);

  for (unsigned i = T; i > 0; --i) {
    // For each slice compute the repmat
    for (int j = 0; j < q; ++j) {
      in.slice(j) = arma::repmat(V.t(), m, 1);
    }
    Vaux = arma::sum(Tp % in, 1);
    V = arma::max(Vaux, 2);
    arma::ucolvec index = arma::index_max(Vaux, 2);
    OptimalPol.col(i - 1) = arma::conv_to<arma::umat>::from(index);
  }
  this->V = V;
  this->OptimalPol = OptimalPol;
}


// Computes the reach avoid probability given N time steps
// and the transition matrix Tp. System has control inputs
void faust_t::StandardReachAvoid_Contr(arma::mat Target, int T) {

  // Get X
  arma::mat X = this->X;

  // Get U
  arma::mat U = this->U;

  // The Cardinality of X
  int m = X.n_rows;

  // The Cardinality of U
  int q = U.n_rows;


  // Get Tp
  // need to convert it to cube frm std::vector
  arma::cube Tp(this->Tp[0].n_rows, this->Tp[0].n_cols, q);
  for (unsigned i = 0; i < q; i++) {
    Tp.slice(i) = this->Tp[i];
  }

  // Dimension of the system
  int dim = X.n_cols / 2;

  // Initialization of value function W
  arma::mat A = repmat(Target.col(1), 1, m);
  arma::mat Xi = X.cols(0, dim - 1);
  arma::mat C = repmat(Target.col(0), 1, m);

  arma::umat inner = Xi < A.t();
  arma::umat outer = Xi > C.t();
  arma::mat a = arma::conv_to<arma::mat>::from(inner);
  arma::mat b = arma::conv_to<arma::mat>::from(outer);
  V = arma::prod(a % b, 1);

  // Matrices used for the calculation of V
  arma::mat W_help1 = arma::ones<arma::mat>(V.n_rows, 1) - V;
  arma::mat W_help2 = V.col(0);
  arma::cube in = arma::zeros<arma::cube>(m, m, q);
  arma::cube Vaux = arma::zeros<arma::cube>(m,m, q);
  this->OptimalPol = arma::zeros<arma::umat>(m,T);

  // The solution
  for (size_t i = 1; i <= T; i++) {
    // For each slice compute the repmat
    for (size_t j = 0; j < q; ++j) {
      arma::mat tempin = arma::repmat(V.t(),m,1);
      in.slice(j) = arma::repmat(V.t(), m, 1);
    }
    Vaux = arma::sum(Tp % in, 1);
    arma::mat tempV = arma::max(Vaux,2);
    V = arma::max(Vaux, 2);
    arma::uvec pol = arma::index_max(Vaux,2);
    OptimalPol.col(i-1) = pol;
    V = V % (W_help1);
    V += W_help2;
  }

  this->V = V;
  this->OptimalPol = OptimalPol;
}

// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon.
//   Input of this function is the desired maximal error epsilon and the number
//   of time steps T
//  The output is a new partition X and a maximum error
void faust_t::Adaptive_grid_multicell_Contr(double epsilon, double T,
                                            arma::mat SafeSet,
                                            arma::mat InputSet) {
  // Get X
  arma::mat X = this->X;

  // Get U
  arma::mat U = this->U;

  // Dimension of the system
  int dim = SafeSet.n_rows;

  // Dimension of the input
  int dim_u = InputSet.n_rows;

  // Cardinality
  int m = this->X.n_rows;

  // Cardinality of the input
  int q = this->U.n_rows;

  // finding the optimal ratio between delta_a and delta_u
  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebesque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-10);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate solution to the optimal ratio
  double delta_a = solution[0];
  double delta_u = solution[1];
  double optimratio = delta_a / delta_u;

  // The initial Lipschitz constants
  arma::cube h_ijr = this->LocalLipschitz_Contr(1, m, 1, q);

  arma::cube h_ijr_u = this->LocalLipschitzToU_Contr(1, m, 1, q);

  // definition of K_ij
  arma::cube innerTemp(m, m, q);
  arma::mat inner(m,1);
  for (int j = 0; j < q; j++) {
    innerTemp.slice(j) = arma::repmat(arma::prod(X.cols(dim,2*dim-1).t(),0),m,1);
  }
  arma::cube K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), m, q, 1);

  // definition of K_ij_u
  arma::cube K_ij_u = arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), m, q, 1);
  innerTemp.clear();

  // The resulting local errors
  arma::mat K_in = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
  arma::mat K_in1 = arma::pow(arma::sum(K_in, 0), 0.5);
  arma::mat tt = arma::repmat(K_in1.t(),1,q);
  arma::mat K_in2 = K_ij.slice(0) % (arma::repmat(K_in1.t(), 1, q));

  arma::mat K_in_u = arma::pow(U.cols(dim_u, 2 * dim_u - 1), 2).t();
  arma::mat K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
  arma::mat K_in_u2 = K_ij_u.slice(0) % (arma::repmat(K_in_u1, m, 1));
  K_ij.clear();
  K_ij_u.clear();
  arma::mat E_ij = 2 * T * (K_in2 + K_in_u2);

  // Initialisation counters
  int count_x = 1;
  int count_u = 1;
  size_t t = 0;
  arma::mat Ei_comp =
      arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));

  while (arma::accu(Ei_comp) > 0) {
    arma::colvec a = arma::min(K_in1,1) / (arma::min(K_in_u1,1));
    double ratio = arma::min(a);
    if (ratio > optimratio) // Choose to split either U or X
    {
      t = E_ij.n_rows;
      int end = t - count_x + 1;
      for (int i = 0; i < end; ++i) {
        if (arma::accu(E_ij.row(count_x-1) > epsilon)) {

          // finding the index related to maximum dimension
          double m0 = X(count_x-1, arma::span(dim, 2 * dim - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(X.row(count_x-1), 2, 1);

          // the first smaller cell
          Y(0, dim + m0) = X(count_x-1, dim + m0) / 2;

          Y(0, m0) =
              X(count_x-1, m0) - Y(0, dim + m0) / 2;

          // the second smaller cell
          Y(1, dim + m0) = X(count_x-1, dim + m0) / 2;
          Y(1, m0) = X(count_x-1, m0) + Y(1, dim + m0) / 2;

          // Update X
          if(count_x -2 < 0) {
            if(X.n_rows > 1) {
              X = arma::join_vert(X.rows(1,X.n_rows-1),Y);
            }
            else {
              X = Y;
            }
          }
          else
          {
            arma::mat Xred = X.rows(0,count_x-1);
            if((count_x < X.n_rows-1)) {
              Xred = arma::join_vert(Xred, X.rows(count_x, X.n_rows - 1));
            }
            X = arma::join_vert(Xred, Y);
          }

          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if( E_ij.n_cols - 1 == 0) {
              E_ij = E_ij.col(0);
             }
            else {
              if(count_x-2 <0 || E_ij.n_cols -2 < 0) {
                E_ij = E_ij.cols(count_x, E_ij.n_cols - 1);
              }
              else if (  E_ij.n_cols <=count_x-1){
                E_ij = E_ij.cols(0, E_ij.n_cols-2);
              }
              else {
              if(count_x == E_ij.n_cols ) {
                E_ij = E_ij.cols(0, count_x -2);
              }
              else {
                E_ij = arma::join_horiz(E_ij.cols(0, count_x -2),
                                     E_ij.cols(count_x, E_ij.n_cols - 1));
              }
              }
            }
          }
        } else {
          count_x++;
        }
      }
      // Update X values
      this->X = X;

      // Cardinality
      m = X.n_rows;

      // Cardinality of the input
      q = U.n_rows;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u , q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(count_x  , m, count_u , q);

      // The updated of K_ij
      size_t x_new = h_ijr.n_rows;
      size_t u_new = h_ijr_u.n_slices;

      innerTemp.resize(x_new, m, u_new);

      K_ij.resize(x_new, u_new, 1);
      K_ij_u.resize(x_new, u_new, 1);
      for (size_t j = 0; j < u_new; j++) {
        arma::mat temp = arma::repmat(arma::prod(X.cols(dim, 2 * dim - 1).t()), x_new, 1);
        innerTemp.slice(j) =
            arma::repmat(arma::prod(X.cols(dim, 2 * dim - 1).t()), x_new, 1);
      }
      K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), x_new, u_new, 1);
      K_ij_u =
            arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), x_new, u_new, 1);

      // The resulting local errors
      if (x_new == 0) {
        arma::mat Kt = arma::sum(arma::pow(X.cols(arma::span(dim, 2 * dim - 1)), 2).t());
        if (u_new == 0) {
          Kt = arma::pow(Kt, 0.5).t();
          K_in = K_ij.slice(0) % (Kt);
        }
        K_in_u = arma::sum(arma::pow(U(arma::span(count_u-1, U.n_rows - 1),arma::span(dim_u, 2 * dim_u - 1)),2).t());
        K_in_u1 = arma::pow(K_in_u, 0.5);
        K_in_u = K_ij_u.slice(0) % (K_in_u1.t());

        arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);
        E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
        E_ij = E_ij_aux;
      } else {

        arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),arma::span(dim, 2 * dim - 1)),2).t());

        Kt = arma::pow(Kt,0.5).t();

        K_in = K_ij.slice(0) % (arma::repmat(Kt, 1, u_new ));

        K_in_u = arma::sum(arma::pow(U(arma::span(count_u-1, U.n_rows - 1),arma::span(dim_u, 2 * dim_u - 1)),2).t());
        K_in_u1 = arma::pow(K_in_u,0.5);

        arma::mat t = arma::repmat(K_in_u1, x_new, 1);
        arma::mat K_in_u2 =  K_ij_u.slice(0) % t ;

        arma::mat E_ij_aux = 2 * T * (K_in + K_in_u2);
        E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
        E_ij = E_ij_aux;
      }
      K_ij.clear();
      K_ij_u.clear();
      innerTemp.clear();
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
    } else {
      t = E_ij.n_cols;
      int end = t - count_u + 1;
      for (int i = 0; i < end; ++i) {
        if (arma::accu(E_ij.col(count_u-1) >  epsilon)) {
          // finding the index related to maximum dimension
          double m0 = U(count_u-1, arma::span(dim_u, 2 * dim_u - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(U.row(count_u-1), 2, 1);
          // the first smaller cell
          Y(0, dim_u + m0) = U(count_u-1, dim_u + m0) / 2;

          Y(0, m0) =
              U(count_u-1, m0) - Y(0, dim_u + m0) / 2;

          // the second smaller cell
          Y(1, dim_u + m0) = U(count_u-1, dim_u + m0) / 2;
          Y(1, m0) = U(count_u-1, m0) + Y(1, dim_u + m0) / 2;

          // Update U
          if(count_u -2 < 0) {
            if(U.n_rows > 1) {
              U = arma::join_vert(U.rows(1,U.n_rows-1),Y);
            }
            else {
              U = Y;
            }
          }
          else
          {
            arma::mat Ured = U.rows(0,count_u-1);
            if((count_u < U.n_rows-1)) {
              Ured = arma::join_vert(Ured, U.rows(count_u, U.n_rows - 1));
            }
            U = arma::join_vert(Ured, Y);
          }

          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if( E_ij.n_cols - 1 == 0) {
              E_ij = E_ij.col(0);
             }
            else {
              if(count_u-2 <0 || E_ij.n_cols -2 < 0) {
                E_ij = E_ij.cols(count_u, E_ij.n_cols - 1);
              }
              else if (  E_ij.n_cols <=count_u-1){
                E_ij = E_ij.cols(0, E_ij.n_cols-2);
              }
              else {
              if(count_u == E_ij.n_cols ) {
                E_ij = E_ij.cols(0, count_u -2);
              }
              else {
                E_ij = arma::join_horiz(E_ij.cols(0, count_u-2),
                                     E_ij.cols(count_u, E_ij.n_cols - 1));
              }
              }
            }
          }
        } else {
          count_u++;
        }
      }
      // Update X values
      this->U = U;

      // Cardinality of the input
      q = U.n_rows;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(count_x, m, count_u, q);

      size_t x_new = h_ijr.n_rows;
      size_t u_new = h_ijr_u.n_slices;

      // The updated of K_ij
      innerTemp.resize(x_new,m, u_new);
      K_ij.resize(x_new, u_new, 1);
      K_ij_u.resize(x_new, u_new, 1);


      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) =
            arma::repmat(arma::prod(X.cols(dim, 2 * dim - 1).t(), 0), x_new, 1);
      }
      if (x_new == 0 && u_new == 0) {
        K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), 1, 1, 1);
        K_ij_u = arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), 1, 1, 1);
      } else if (x_new > 0 && u_new == 0) {
        K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), x_new, 1, 1);
        K_ij_u = arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), x_new, 1, 1);
      } else if (x_new == 0 && u_new > 0) {
        K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), 1, u_new, 1);
        K_ij_u = arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), 1, u_new, 1);
      } else {
        K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), x_new, u_new, 1);
        K_ij_u =
            arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), x_new, u_new, 1);
      }
      // The resulting local errors
      if (x_new == 0) {
        arma::mat Kt = arma::sum(arma::pow(X.cols(arma::span(dim, 2 * dim - 1)), 2).t());
        if (u_new == 0) {
          Kt = arma::pow(Kt, 0.5).t();
          K_in = K_ij.slice(0) % (Kt);
        }
        K_in_u = arma::sum(arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                                       arma::span(dim_u, 2 * dim_u - 1)),2).t());
        K_in_u1 = arma::pow(K_in_u, 0.5);
        K_in_u2 = K_ij_u.slice(0) % (K_in_u1.t());

        arma::mat E_ij_aux = 2 * T * (K_in + K_in_u2);
        E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
        E_ij = E_ij_aux;
      } else {
        arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),arma::span(dim, 2 * dim - 1)),2).t());
        arma::mat Kt2 = arma::sqrt(Kt).t();

        K_in = K_ij.slice(0) % (arma::repmat(Kt2, 1, u_new));
        K_in_u = arma::sum(arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                                       arma::span(dim_u, 2 * dim_u - 1)),2).t());

        K_in_u1 = arma::pow(K_in_u,0.5);
        arma::mat t = arma::repmat(K_in_u1, x_new, 1);
        arma::mat K_in_u2 =  K_ij_u.slice(0) % t ;
        arma::mat E_ij_aux = 2 * T * (K_in + K_in_u2);

        E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
        E_ij = E_ij_aux;
      }
      K_ij.clear();
      K_ij_u.clear();
      innerTemp.clear();
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
    }
  }
  this->U = U;
  this->X = X;
  arma::mat E_temp = arma::max(E_ij);
  this->E = E_temp(0, 0);
}

// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon.
//   Input of this function is the desired maximal error epsilon and the number
//   of time steps T
//  The output is a new partition X and a maximum error
void faust_t::Adaptive_grid_semilocal_Contr(double epsilon, double T,
                                            arma::mat SafeSet,
                                            arma::mat InputSet) {
  // Get X
  arma::mat X = this->X;

  // Get U
  arma::mat U = this->U;

  // Dimension of the system
  int dim = SafeSet.n_rows;

  // Dimension of the input
  int dim_u = InputSet.n_rows;

  // Cardinality
  int m = this->X.n_rows;

  // Cardinality of the input
  int q = this->U.n_rows;

  // finding the optimal ratio between delta_a and delta_u
  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate solution to the optimal ratio
  double delta_a = solution[0];
  double delta_u = solution[1];
  double optimratio = delta_a / delta_u;

  // The initial Lipschitz constants
  arma::mat h_ir = this->SemiLocalLipschitz_Contr(SafeSet, 1, m, 1, q);
  arma::mat h_ir_u = this->SemiLocalLipschitzToU_Contr(SafeSet, 1, m,1, q);

  // definition of K_ij
  arma::mat K_ij = h_ir * L_A;

  // definition of K_ij_u
  arma::mat K_ij_u = h_ir_u * L_A;

  // The resulting local errors
  arma::mat K_in = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
  arma::mat K_in1 = arma::pow(arma::sum(K_in, 0), 0.5);
  K_in = K_ij % (arma::repmat(K_in1.t(), 1, q));

  arma::mat K_in_u = arma::pow(U.cols(dim_u, 2 * dim_u - 1), 2).t();
  arma::mat K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
  K_in_u = K_ij_u % (arma::repmat(K_in_u1.t(), m, 1));

  arma::mat E_ij = 2 * T * (K_in + K_in_u);

  // Initialisation counters
  int count_x = 1;
  int count_u = 1;
  size_t t = 0;
  arma::mat Ei_comp =
      arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));

  while (arma::accu(Ei_comp) > 0) {
    double ratio = 1;       // arma::min(K_in1)/(arma::min(K_in_u1));
    if (ratio > optimratio) // Choose to split either U or X
    {
      t = E_ij.n_rows;
      int end = t - count_x + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.row(count_x-1) > epsilon)) {
          // finding the index related to maximum dimension
          double m0 = X(count_x-1, arma::span(dim, 2 * dim - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(X.row(count_x-1), 2, 1);
          // the first smaller cell
          Y(0, dim + m0) = X(count_x-1, dim + m0) / 2;

          Y(0, m0) =
              X(count_x-1, m0) - Y(0, dim + m0) / 2;

          // the second smaller cell
          Y(1, dim + m0) = X(count_x-1, dim + m0) / 2;
          Y(1, m0) = X(count_x -1, m0) + Y(1, dim + m0) / 2;

          // Update X
          if(count_x -2 < 0) {
            X = Y;
          }
          else
          {
            arma::mat Xred = X.rows(0,count_x-2);
            if(count_x < X.n_rows) {
              Xred = arma::join_vert(Xred, X.rows(count_x, X.n_rows - 1));
            }
            X = arma::join_vert(Xred, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_x+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                     E_ij.cols(count_x+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_x++;
        }
      }
      // Update X values
      this->X = X;

      // The updated Lipschitz constants to x
      h_ir = this->SemiLocalLipschitz_Contr(
          SafeSet, count_x,
          X.n_rows, count_u, U.n_rows);

      // The updated Lipschitz constants  to u
      h_ir_u = this->SemiLocalLipschitzToU_Contr(
          SafeSet, count_x ,
          X.n_rows, count_u, U.n_rows);

      // The updated of K_ij
      K_ij = h_ir * L_A;

      // definition of K_ij_u
      K_ij_u = h_ir_u * L_A;

      // The resulting local errors
      size_t x_new = h_ir.n_cols;
      int u_new = h_ir_u.n_cols;

      arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x -1, X.n_rows - 1),
                                           arma::span(dim, 2 * dim - 1)),
                                         2)
                                   .t(),
                               0);
      Kt = arma::pow(Kt, 0.5).t();
      K_in = K_ij % (arma::repmat(Kt, 1, u_new));

      K_in_u = arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                           arma::span(dim_u, 2 * dim_u - 1)),
                         2)
                   .t();
      K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
      K_in_u = K_ij_u % (arma::repmat(K_in_u1.t(), x_new, 1));

      arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);

      // The resulting local errors
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));

    } else {
      t = E_ij.n_cols;
      int end = t - count_u + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.col(count_u -1) > epsilon)) {
          // finding the index related to maximum dimension
          double m0 = U(count_u-1, arma::span(dim_u, 2 * dim_u - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(U.row(count_u -1), 2, 1);

          // the first smaller cell
          Y(0, dim_u + m0) = U(count_u -1, dim_u + m0) / 2;

          Y(0, m0) =
              U(count_u-1, m0) - Y(0, dim_u + m0) / 2; // TODO check if m0 or m0-1

          // the second smaller cell
          Y(1, dim_u + m0) = X(count_u-1, dim_u + m0) / 2;
          Y(1, m0) = X(count_u-1, m0) + Y(1, dim_u + m0) / 2;

          // Update U
          if(count_u -2 < 0) {
            U = Y;
          }
          else
          {
            arma::mat Ured = U.rows(0,count_u-2);
            if(count_u < U.n_rows) {
              Ured = arma::join_vert(Ured, U.rows(count_u, U.n_rows - 1));
            }
            U = arma::join_vert(Ured, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_u+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                     E_ij.cols(count_u+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_u++;
        }
        Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
      }
      // Update X values
      this->U = U;

      // The updated Lipschitz constants to x
      h_ir = this->SemiLocalLipschitz_Contr(
          SafeSet, count_x,
          X.n_rows, count_u, U.n_rows);

      // The updated Lipschitz constants  to u
      h_ir_u = this->SemiLocalLipschitzToU_Contr(
          SafeSet, count_x,
          X.n_rows, count_u, U.n_rows);

      size_t x_new = h_ir.n_cols;
      size_t u_new = h_ir_u.n_cols;

      // The updated of K_ij
      K_ij = h_ir * L_A;

      // definition of K_ij_u
      K_ij_u = h_ir_u * L_A;

      // The resulting local errors
      arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),
                                           arma::span(dim, 2 * dim - 1)),
                                         2)
                                   .t(),
                               0);
      Kt = arma::pow(Kt, 0.5).t();
      K_in = K_ij % (arma::repmat(Kt, 1, u_new));
      K_in_u = arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                           arma::span(dim_u, 2 * dim_u - 1)),
                         2)
                   .t();
      K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
      K_in_u = K_ij_u % (arma::repmat(K_in_u1.t(), x_new, 1));

      arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);
      // The resulting local errors
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
    }
  }
  this->U = U;
  this->X = X;
  arma::mat E_temp = arma::max(E_ij);
  this->E = E_temp(0, 0);
}

// This function makes an adaptive grid,  for a controlled system, that will
// produce a certain maximal abstraction error defined by epsilon.
void faust_t::Adaptive_grid_ReachAvoid_Contr(double epsilon, double T,
                                             arma::mat SafeSet,
                                             arma::mat InputSet,
                                             arma::mat TargetSet) {
  // Get X
  arma::mat X = this->X;

  // Get U
  arma::mat U = this->U;

  // Dimension of the system
  int dim = SafeSet.n_rows;

  // Dimension of the input
  int dim_u = InputSet.n_rows;

  // Cardinality
  int m = this->X.n_rows;

  // Cardinality of the input
  int q = this->U.n_rows;

  // finding the optimal ratio between delta_a and delta_u
  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-10);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate solution to the optimal ratio
  double delta_a = solution[0];
  double delta_u = solution[1];
  double optimratio = delta_a / delta_u;

  // Extract the Target Set (creation of SafeSet minus TargetSet)
  // Indexing the target set
  arma::mat inner1 = arma::repmat(TargetSet.col(1), 1, m).t();
  arma::mat outer1 = X.cols(0, dim - 1);
  arma::mat lhs = arma::conv_to<arma::mat>::from(outer1 < inner1);
  // std::cout << "lhs: " << lhs << std::endl;

  arma::mat inner2 = arma::repmat(TargetSet.col(0), 1, m).t();
  arma::mat rhs = arma::conv_to<arma::mat>::from(outer1 > inner2);
  // std::cout << "rhs: " << rhs << std::endl;

  arma::mat all = lhs % rhs;
  // std::cout << "all: " << all << std::endl;

  arma::mat TargetIndex = arma::prod(all, 1);

  // Reshape X so that the first entries are the target set
  arma::uvec in1 = arma::find(TargetIndex == 1);
  // std::cout << "in1: " << in1 << std::endl;
  arma::uvec in2 = arma::find(TargetIndex != 1);
  // std::cout << "in2: " << in2 << std::endl;
  X = arma::join_vert(X.rows(in1), X.rows(in2));

  // The initial Lipschitz constants to X
  int k = arma::accu(TargetIndex) + 1;
  arma::cube h_ijr = this->LocalLipschitz_Contr(k, m, 1, q);

  // The initial Lipschitz constants to U
  arma::cube h_ijr_u = this->LocalLipschitzToU_Contr(k, m, 1, q);

  // definition of K_ij
  arma::cube innerTemp(m, 1, q);
  for (size_t j = 0; j < q; j++) {
    innerTemp.slice(j) = arma::repmat(
        arma::prod(X.cols(dim, 2 * dim - 1).t(), 0), (m - k - 1), 1);
  }
  arma::cube K_ij =
      arma::reshape(arma::sum(h_ijr % innerTemp, 1), (m - k - 1), q, 1);

  // definition of K_ij_u
  arma::cube K_ij_u =
      arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), (m - k - 1), q, 1);

  // The resulting local errors
  arma::mat K_in =
      arma::pow(X(arma::span(k, m), arma::span(dim, 2 * dim - 1)), 2).t();
  arma::mat K_in1 = arma::pow(arma::sum(K_in, 0), 0.5);
  K_in = K_ij.slice(0) % (arma::repmat(K_in1.t(), 1, q));

  arma::mat K_in_u = arma::pow(U.cols(dim_u, 2 * dim_u - 1), 2).t();
  arma::mat K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
  K_in_u = K_ij_u.slice(0) % (arma::repmat(K_in_u1.t(), (m - k - 1), 1));
  arma::mat E_ij = 2 * T * (K_in + K_in_u);
  if ((k - 2) > 0) {
    E_ij = join_vert(arma::zeros<arma::mat>(k - 1, q), E_ij);
  }

  // Initialisation counters
  int count_x = 1 + (k - 1);
  int count_u = 1;
  size_t t = 0;
  arma::mat Ei_comp =
      arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
  while (arma::accu(Ei_comp) > 0) {
    arma::vec K_tin = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
    arma::vec K_tin1 = arma::pow(arma::sum(K_in, 0), 0.5).t();
    double ratio = arma::min(K_tin1)/(arma::min(K_tin1));
    if (ratio > optimratio) // Choose to split either U or X
    {
      t = E_ij.n_rows;
      int end = t - count_x;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.row(count_x - 1) > epsilon)) {
          // finding the index related to maximum dimension
          double m0 = X(count_x -1 , arma::span(dim, 2 * dim - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(X.row(count_x-1), 2, 1);

          // the first smaller cell
          Y(0, dim + m0) = X(count_x -1, dim + m0) / 2;
          Y(0, m0) =
              X(count_x -1, m0) - Y(0, dim + m0) / 2;

          // the second smaller cell
          Y(1, dim + m0) = X(count_x-1, dim + m0) / 2;
          Y(1, m0) = X(count_x-1, m0) + Y(1, dim + m0) / 2;

          // Update X
          if(count_x -2 < 0) {
            X = Y;
          }
          else
          {
            arma::mat Xred = X.rows(0,count_x-2);
            if(count_x < X.n_rows) {
              Xred = arma::join_vert(Xred, X.rows(count_x, X.n_rows - 1));
            }
            X = arma::join_vert(Xred, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_x+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                     E_ij.cols(count_x+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_x++;
        }
      }
      // Update X values
      this->X = X;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(count_x, m, count_u, q);

      // The updated of K_ij
      size_t x_new = h_ijr.n_rows;
      size_t u_new = h_ijr_u.n_slices;
      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) =
            arma::repmat(arma::prod(X.cols(dim, 2 * dim - 1).t(), 0), x_new, 1);
      }
      K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), x_new, u_new, 1);

      // definition of K_ij_u
      K_ij_u =
          arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), x_new, u_new, 1);

      // The resulting local errors
      arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),
                                           arma::span(dim, 2 * dim - 1)),
                                         2)
                                   .t(),
                               0);
      Kt = arma::pow(Kt, 0.5).t();
      K_in = K_ij.slice(0) % (arma::repmat(Kt, 1, u_new));

      K_in_u = arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                           arma::span(dim_u, 2 * dim_u - 1)),
                         2)
                   .t();
      K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
      K_in_u = K_ij_u.slice(0) % (arma::repmat(K_in_u1.t(), x_new, 1));

      arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
    } else {
      t = E_ij.n_cols;
      int end = t - count_u + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.col(count_u-1) > epsilon)) {

          // finding the index related to maximum dimension
          double m0 = U(count_u -1, arma::span(dim_u, 2 * dim_u - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(U.row(count_u -1), 2, 1);

          // the first smaller cell
          Y(0, dim_u + m0) = U(count_u -1 , dim_u + m0) / 2;

          Y(0, m0) =
              U(count_u-1, m0) - Y(0, dim_u + m0) / 2; // TODO check if m0 or m0-1

          // the second smaller cell
          Y(1, dim_u + m0) = X(count_u -1, dim_u + m0) / 2;
          Y(1, m0) = X(count_u -1 , m0) + Y(1, dim_u + m0) / 2;

          // Update U
          if(count_u -2 < 0) {
            U = Y;
          }
          else
          {
            arma::mat Ured = U.rows(0,count_u-2);
            if(count_u < U.n_rows) {
              Ured = arma::join_vert(Ured, U.rows(count_u, U.n_rows - 1));
            }
            U = arma::join_vert(Ured, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_u+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                     E_ij.cols(count_u+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_u++;
        }
        Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
      }
      // Update X values
      this->U = U;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u,q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(count_x, m, count_u, q);

      // The updated of K_ij
      size_t x_new = h_ijr.n_rows;
      size_t u_new = h_ijr.n_cols;
      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) =
            arma::repmat(arma::prod(X.cols(dim, 2 * dim - 1).t(), 0), x_new, 1);
      }
      K_ij = arma::reshape(arma::sum(h_ijr % innerTemp, 1), x_new, u_new, 1);

      // definition of K_ij_u
      K_ij_u =
          arma::reshape(arma::sum(h_ijr_u % innerTemp, 1), x_new, u_new, 1);

      // The resulting local errors
      arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),
                                           arma::span(dim, 2 * dim - 1)),
                                         2)
                                   .t(),
                               0);
      Kt = arma::pow(Kt, 0.5).t();
      K_in = K_ij.slice(0) % (arma::repmat(Kt, 1, u_new));

      K_in_u = arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                           arma::span(dim_u, 2 * dim_u - 1)),
                         2)
                   .t();
      K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
      K_in_u = K_ij_u.slice(0) % (arma::repmat(K_in_u1.t(), x_new, 1));

      arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
    }
  }
  this->U = U;
  this->X = X;
  arma::mat E_temp = arma::max(E_ij);
  this->E = E_temp(0, 0);
}

// This function makes an adaptive grid,  for a controlled system, that will
// produce a certain maximal abstraction error defined by epsilon.
void faust_t::Adaptive_grid_ReachAvoid_semilocal_Contr(double epsilon, double T,
                                                       arma::mat SafeSet,
                                                       arma::mat InputSet,
                                                       arma::mat TargetSet) {
  // Get X
  arma::mat X = this->X;

  // Get U
  arma::mat U = this->U;

  // Dimension of the system
  int dim = SafeSet.n_rows;

  // Dimension of the input
  int dim_u = InputSet.n_rows;

  // Cardinality
  int m = this->X.n_rows;

  // Cardinality of the input
  int q = this->U.n_rows;

  // finding the optimal ratio between delta_a and delta_u
  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  // Translate solution to the optimal ratio
  double delta_a = solution[0];
  double delta_u = solution[1];
  double optimratio = delta_a / delta_u;

  // Extract the Target Set (creation of SafeSet minus TargetSet)
  // Indexing the target set
  arma::mat inner1 = arma::repmat(TargetSet.col(1), 1, m).t();
  arma::mat outer1 = X.cols(0, dim - 1);
  arma::mat lhs = arma::conv_to<arma::mat>::from(outer1 < inner1);
  // std::cout << "lhs: " << lhs << std::endl;

  arma::mat inner2 = arma::repmat(TargetSet.col(0), 1, m).t();
  arma::mat rhs = arma::conv_to<arma::mat>::from(outer1 > inner2);
  // std::cout << "rhs: " << rhs << std::endl;

  arma::mat all = lhs % rhs;
  // std::cout << "all: " << all << std::endl;

  arma::mat TargetIndex = arma::prod(all, 1);

  // Reshape X so that the first entries are the target set
  arma::uvec in1 = arma::find(TargetIndex == 1);
  // std::cout << "in1: " << in1 << std::endl;
  arma::uvec in2 = arma::find(TargetIndex != 1);
  // std::cout << "in2: " << in2 << std::endl;
  X = arma::join_vert(X.rows(in1), X.rows(in2));

  // The initial Lipschitz constants to X
  int k = arma::accu(TargetIndex) + 1;
  arma::mat h_ir = this->SemiLocalLipschitz_Contr(
      SafeSet, k, m, 1, q);

  // The initial Lipschitz constants to U
  arma::mat h_ir_u = this->SemiLocalLipschitzToU_Contr(
      SafeSet, k, m, 1, q);

  // definition of K_ij
  arma::mat K_ij = h_ir * L_A;

  // definition of K_ij_u
  arma::mat K_ij_u = h_ir_u * L_A;

  // The resulting local errors
  arma::mat K_in =
      arma::pow(X(arma::span(k, m), arma::span(dim, 2 * dim - 1)), 2).t();
  arma::mat K_in1 = arma::pow(arma::sum(K_in, 0), 0.5);
  K_in = K_ij % (arma::repmat(K_in1.t(), 1, q));

  arma::mat K_in_u = arma::pow(U.cols(dim_u, 2 * dim_u - 1), 2).t();
  arma::mat K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
  K_in_u = K_ij_u % (arma::repmat(K_in_u1.t(), (m - k - 1), 1));
  arma::mat E_ij = 2 * T * (K_in + K_in_u);
  if ((k - 2) > 0) {
    E_ij = join_vert(arma::zeros<arma::mat>(k - 1, q), E_ij);
  }

  // Initialisation counters
  int count_x = 1 + (k - 1);
  int count_u = 1;
  size_t t = 0;
  arma::mat Ei_comp =
      arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
  while (arma::accu(Ei_comp) > 0) {
    arma::vec K_tin = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
    arma::vec K_tin1 = arma::pow(arma::sum(K_tin, 0), 0.5).t();
    double ratio = arma::min(K_tin1)/(arma::min(K_tin1));
    if (ratio > optimratio) // Choose to split either U or X
    {
      t = E_ij.n_rows;
      int end = t - count_x + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.row(count_x -1) > epsilon)) {

          // finding the index related to maximum dimension
          double m0 = X(count_x -1, arma::span(dim, 2 * dim - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(X.row(count_x-1), 2, 1);

          // the first smaller cell
          Y(0, dim + m0) = X(count_x-1, dim + m0) / 2;

          Y(0, m0) =
              X(count_x-1, m0) - Y(0, dim + m0) / 2;

          // the second smaller cell
          Y(1, dim + m0) = X(count_x -1, dim + m0) / 2;
          Y(1, m0) = X(count_x -1, m0) + Y(1, dim + m0) / 2;

          // Update X
          if(count_x -2 < 0) {
            X = Y;
          }
          else
          {
            arma::mat Xred = X.rows(0,count_x-2);
            if(count_x < X.n_rows) {
              Xred = arma::join_vert(Xred, X.rows(count_x, X.n_rows - 1));
            }
            X = arma::join_vert(Xred, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_x+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                     E_ij.cols(count_x+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_x++;
        }
      }
      // Update X values
      this->X = X;

      // The updated Lipschitz constants to x
      h_ir = this->SemiLocalLipschitz_Contr(
          SafeSet, count_x,
          X.n_rows, count_u, U.n_rows);

      // The updated Lipschitz constants  to u
      h_ir_u = this->SemiLocalLipschitzToU_Contr(
          SafeSet, count_x,
          X.n_rows, count_u, U.n_rows);

      size_t x_new = X.n_rows - count_x;
      size_t u_new = U.n_rows - count_u;

      // The updated of K_ij
      K_ij = h_ir * L_A;

      // definition of K_ij_u
      K_ij_u = h_ir_u * L_A;

      // The resulting local errors
      arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),
                                           arma::span(dim, 2 * dim - 1)),
                                         2)
                                   .t(),
                               0);
      Kt = arma::pow(Kt, 0.5).t();
      K_in = K_ij % (arma::repmat(Kt, 1, u_new));

      K_in_u = arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                           arma::span(dim_u, 2 * dim_u - 1)),
                         2)
                   .t();
      K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
      K_in_u = K_ij_u % (arma::repmat(K_in_u1.t(), x_new, 1));

      arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
      std::cout << "Ei_comp: " << Ei_comp << "epsilon: " << epsilon
                << std::endl;

    } else {
      t = E_ij.n_cols;
      int end = t - count_u + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.col(count_u-1) > epsilon)) {
          // finding the index related to maximum dimension
          double m0 = U(count_u-1, arma::span(dim_u, 2 * dim_u - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(U.row(count_u -1), 2, 1);

          // the first smaller cell
          Y(0, dim_u + m0) = U(count_u -1, dim_u + m0) / 2;

          Y(0, m0) =
              U(count_u-1, m0) - Y(0, dim_u + m0) / 2; // TODO check if m0 or m0-1

          // the second smaller cell
          Y(1, dim_u + m0) = X(count_u-1, dim_u + m0) / 2;
          Y(1, m0) = X(count_u-1, m0) + Y(1, dim_u + m0) / 2;

          // Update U
          if(count_u -2 < 0) {
            U = Y;
          }
          else
          {
            arma::mat Ured = U.rows(0,count_u-2);
            if(count_u < U.n_rows) {
              Ured = arma::join_vert(Ured, U.rows(count_u, U.n_rows - 1));
            }
            U = arma::join_vert(Ured, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_u+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                     E_ij.cols(count_u+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_u++;
        }
        Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
        std::cout << "Ei_comp: " << Ei_comp << "epsilon: " << epsilon
                  << std::endl;
      }
      // Update X values
      this->U = U;

      // The updated Lipschitz constants to x
      h_ir = this->SemiLocalLipschitz_Contr(
          SafeSet, count_x, X.n_rows, count_u,
          U.n_rows);

      // The updated Lipschitz constants  to u
      h_ir_u = this->SemiLocalLipschitzToU_Contr(
          SafeSet, count_x,
          X.n_rows, count_u, U.n_rows);
      int x_new = X.n_rows - count_x;
      int u_new = U.n_rows - count_u;

      // The updated of K_ij
      K_ij = h_ir * L_A;

      // definition of K_ij_u
      K_ij_u = h_ir_u * L_A;

      // The resulting local errors
      arma::mat Kt = arma::sum(arma::pow(X(arma::span(count_x-1, X.n_rows - 1),
                                           arma::span(dim, 2 * dim - 1)),
                                         2)
                                   .t(),
                               0);
      Kt = arma::pow(Kt, 0.5).t();
      K_in = K_ij % (arma::repmat(Kt, 1, u_new));

      K_in_u = arma::pow(U(arma::span(count_u-1, U.n_rows - 1),
                           arma::span(dim_u, 2 * dim_u - 1)),
                         2)
                   .t();
      K_in_u1 = arma::pow(arma::sum(K_in_u, 0), 0.5);
      K_in_u = K_ij_u % (arma::repmat(K_in_u1.t(), x_new, 1));

      arma::mat E_ij_aux = 2 * T * (K_in + K_in_u);
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
    }
  }
  this->U = U;
  this->X = X;
  arma::mat E_temp = arma::max(E_ij);
  this->E = E_temp(0, 0);
}

// Adaptive Markov Chain approximation
// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon
void faust_t::Adaptive_grid_MCapprox_Contr(double epsilon, double T,
                                           arma::mat SafeSet,
                                           arma::mat InputSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat r = SafeSet.col(1) - SafeSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(r);

  // The dimension of the input of system
  double dim_u = InputSet.n_rows;

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  // This factor is used for the calculation of the variable optimratio
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  double h_s = this->GlobalLipschitz_Contr(SafeSet, InputSet, 1);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  //
  double delta_a = solution[0];
  double delta_u = solution[1];

  double factor =
      h_a /
      (h_a + h_s); // This factor acounts for the MC error in the optimal ratio
  double optimratio = factor * delta_a / delta_u;
  // Translate the solution to the length of the edges
  delta_a = delta_a / sqrt(dim);
  delta_u = delta_u / sqrt(dim_u);

  // Create the location of the representative points
  // delta is adjusted so that the edges of the set are part of the partitions
  std::vector<arma::mat> C;
  arma::mat delta_adj_a = arma::zeros<arma::mat>(1, dim);
  for (size_t i = 0; i < dim; i++) {
    delta_adj_a.col(i) = (SafeSet(i, 1) - SafeSet(i, 0)) /
                         std::ceil((SafeSet(i, 1) - SafeSet(i, 0)) / delta_a);
    // std::cout << "delta_adj(i): " << delta_adj(i) << std::endl;
    double start = SafeSet(i, 0) + 0.5 * delta_adj_a(i);
    double Delta = delta_adj_a(i);
    double end = SafeSet(i, 1) - 0.499 * delta_adj_a(i);
    arma::mat A = arma::regspace(start, Delta, end);
    // std::cout << "A: " << A << std::endl;
    C.push_back(A);
  }
  // Construct the grid
  C = nDgrid(C);
  // std::cout << "2nd Dim: " << C[1] << std::endl;

  // Cardinality
  int m = C[0].n_elem;
  int sm = std::sqrt(m);
  // Make the output
  int index = 0;
  int k = 0;
  arma::mat X = arma::zeros<arma::mat>(m, 2 * dim);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < dim; j++) {
      X(i, j) = C[j](k, index);
    }
    if (k >= (sm - 1)) {
      k = 0;
      index++;
    } else {
      k++;
    }
  }
  X(arma::span(0, m), arma::span(dim, 2 * dim - 1)) =
      arma::kron(delta_adj_a, arma::ones<arma::mat>(m, 1));
  C.clear();
  // Create the location of the representative points
  // delta_u is adjusted so that the edges of the set are part of the partitions
  arma::mat delta_adj_u = arma::zeros<arma::mat>(1, dim_u);
  for (size_t i = 0; i < dim_u; ++i) {
    delta_adj_u.col(i) = (InputSet(i, 1) - InputSet(i, 0)) /
                         std::ceil((InputSet(i, 1) - InputSet(i, 0)) / delta_u);
    double start = InputSet(i, 0) + 0.5 * delta_adj_u(i);
    double Delta = delta_adj_u(i);
    double end = InputSet(i, 1) - 0.499 * delta_adj_u(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::mat A = arma::regspace(start, Delta, end);
    // std::cout << "A: " << A << std::endl;
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  m = C[0].n_elem;
  sm = std::sqrt(m);

  // Make the output
  index = 0;
  k = 0;
  arma::mat U = arma::zeros<arma::mat>(m, 2 * dim_u);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < dim_u; j++) {
      U(i, j) = C[j](k, index);
    }
    if (k >= (sm - 1)) {
      k = 0;
      index++;
    } else {
      k++;
    }
  }
  U(arma::span(0, m - 1), arma::span(dim_u, 2 * dim_u - 1)) =
      arma::kron(delta_adj_u, arma::ones<arma::mat>(m, 1));

  this->X = X;
  this->U = U;

  // End of uniform part
  // Cardinality
  m = this->X.n_rows;

  // Cardinality of the input
  int q = this->U.n_rows;

  // The initial Lipschitz constants
  arma::cube h_ijr = this->LocalLipschitz_Contr(1, m, 1, q);
  arma::cube h_ijr_u = this->LocalLipschitzToU_Contr(1, m, 1, q);
  arma::cube h_ijr_s = this->LocalLipschitz_Contr(1, m, 1, q);

  // The local Lipschitz constants multiplied by their respective delta's
  arma::cube innerTemp(1, m, q);
  for (size_t j = 0; j < q; j++) {
    innerTemp.slice(j) = arma::repmat(
        arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0), 0.5)
            .t(),
        1, m);
  }
  arma::cube h1 = h_ijr % innerTemp;
  std::cout << "h1: " << h1 << std::endl;

  // definition of h2
  arma::cube innerTemp2(m, 1, q);
  for (size_t j = 0; j < q; j++) {
    innerTemp2.slice(j) = arma::repmat(
        arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0), 0.5)
            .t(),
        m, 1);
  }
  arma::cube h2 = h_ijr_s % innerTemp2;
  std::cout << "h2: " << h2 << std::endl;

  // definition of h3
  arma::cube innerTemp3 = arma::ones<arma::cube>(1, 1,q);
  for(size_t j = 0 ; j < q; j++) {
    innerTemp3.slice(j) = arma::reshape(arma::pow(arma::sum(arma::pow(U.cols(dim_u,2*dim_u-1),2).t(),0),0.5).t(),1,1);
  }
  arma::cube it;
  for (size_t j = 0; j < 1; j++) {
    arma::mat temp = innerTemp3.tube(0, 0);
    it.slice(j) = arma::repmat(temp, m, m);
  }
  arma::cube h3 = h_ijr_u % it;
  std::cout << "h3: " << h3 << std::endl;

  arma::cube t0 = h1 + h2 + h3;
  for (size_t j = 0; j < q; j++) {
    innerTemp2.slice(j) = arma::repmat(
        arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0), 0.5)
            .t(),
        m, 1);
  }
  arma::cube t1 = arma::sum(t0 % innerTemp2, 1);
  arma::mat E_ij = 2 * T * arma::reshape(t1.slice(0), m, q);


  // Initialisation counters
  int count_x = 1;
  int count_u = 1;
  size_t t = 0;
  arma::mat Ei_comp =
      arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
  while (arma::accu(Ei_comp) > 0) {
    arma::vec K_tin = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
    arma::vec K_tin1 = arma::pow(arma::sum(K_tin, 0), 0.5).t();
    arma::vec K_uin = arma::pow(U.cols(dim_u, 2 * dim_u - 1), 2).t();
    arma::vec K_uin1 = arma::pow(arma::sum(K_uin, 0), 0.5).t();
    double ratio = arma::min(K_tin1)/(arma::min(K_uin1));
    if (ratio > optimratio) // Choose to split either U or X
    {
      t = E_ij.n_rows;
      int end = t - count_x + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.row(count_x-1) > epsilon)) {
          // finding the index related to maximum dimension
          double m0 = X(count_x-1, arma::span(dim, 2 * dim - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(X.row(count_x-1), 2, 1);

          // the first smaller cell
          Y(0, dim + m0) = X(count_x-1, dim + m0) / 2;

          Y(0, m0) =
              X(count_x-1, m0) - Y(0, dim + m0) / 2;

          // the second smaller cell
          Y(1, dim + m0) = X(count_x-1, dim + m0) / 2;
          Y(1, m0) = X(count_x-1, m0) + Y(1, dim + m0) / 2;

          // Update X
          if(count_x -2 < 0) {
            X = Y;
          }
          else
          {
            arma::mat Xred = X.rows(0,count_x-2);
            if(count_x < X.n_rows) {
              Xred = arma::join_vert(Xred, X.rows(count_x, X.n_rows - 1));
            }
            X = arma::join_vert(Xred, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_x+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_x -1),
                                     E_ij.cols(count_x+1, E_ij.n_cols - 1));
              }
          }
          } else {
          count_x++;
          }
      }
      // Update X values
      this->X = X;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants to x_bar
      h_ijr_s =
          this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The local Lipschitz constants multiplied by their respective delta's
      size_t x_new = X.n_rows - count_x;
      size_t u_new = U.n_rows - count_u;

      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X(arma::span(count_x-1, m - 1),
                                            arma::span(dim, 2 * dim - 1)),
                                          2)
                                    .t(),
                                0),
                      0.5)
                .t(),
            1, m);
      }
      h1 = h_ijr % innerTemp;

      // definition of h2
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      h2 = h_ijr_s % innerTemp2;

      // definition of h3
      innerTemp3 = arma::ones<arma::cube>(1, 1,q);
      for(size_t j = 0 ; j < q; j++) {
        innerTemp3.slice(j) = arma::reshape(arma::pow(arma::sum(arma::pow(U.cols(dim_u,2*dim_u-1),2).t(),0),0.5).t(),1,1);
      }
      for (size_t j = 0; j < 1; j++) {
        arma::mat temp = innerTemp3.tube(0, 0);
        it.slice(j) = arma::repmat(temp, x_new, u_new);
      }
      h3 = h_ijr_u % it;

      t0 = h1 + h2 + h3;
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      t1 = arma::sum(t0 % innerTemp2, 1);
      arma::mat E_ij_aux = 2 * T * arma::reshape(t1.slice(0), x_new, u_new);

      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
    }
    else {
      t = E_ij.n_cols;
      int end = t - count_u + 1;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.col(count_u-1) > epsilon)) {

          // finding the index related to maximum dimension
          double m0 = U(count_u-1, arma::span(dim_u, 2 * dim_u - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(U.row(count_u-1), 2, 1);

          // the first smaller cell
          Y(0, dim_u + m0) = U(count_u-1, dim_u + m0) / 2;

          Y(0, m0) =
              U(count_u-1, m0) - Y(0, dim_u + m0) / 2; // TODO check if m0 or m0-1

          // the second smaller cell
          Y(1, dim_u + m0) = X(count_u-1, dim_u + m0) / 2;
          Y(1, m0) = X(count_u-1, m0) + Y(1, dim_u + m0) / 2;

          // Update U
          if(count_u -2 < 0) {
            U = Y;
          }
          else
          {
            arma::mat Ured = U.rows(0,count_u-2);
            if(count_u < U.n_rows) {
              Ured = arma::join_vert(Ured, U.rows(count_u, U.n_rows - 1));
            }
            U = arma::join_vert(Ured, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_u+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                     E_ij.cols(count_u+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_u++;
        }
        Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
      }
      // Update X values
      this->U = U;
      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants to x_bar
      h_ijr_s = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The local Lipschitz constants multiplied by their respective delta's
      int x_new = X.n_rows - count_x;
      int u_new = U.n_rows - count_u;

      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X(arma::span(count_x-1, m - 1),
                                            arma::span(dim, 2 * dim - 1)),
                                          2)
                                    .t(),
                                0),
                      0.5)
                .t(),
            1, m);
      }
      h1 = h_ijr % innerTemp;

      // definition of h2
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      h2 = h_ijr_s % innerTemp2;

      // definition of h3
      innerTemp3 = arma::ones<arma::cube>(1, 1,
          q);
      for(size_t j = 0 ; j < q; j++) {
        innerTemp3.slice(j) = arma::reshape(arma::pow(arma::sum(arma::pow(U.cols(dim_u,2*dim_u-1),2).t(),0),0.5).t(),1,1);

      }
      for (size_t j = 0; j < 1; j++) {
        arma::mat temp = innerTemp3.tube(0, 0);
        it.slice(j) = arma::repmat(temp, x_new, u_new);
      }
      h3 = h_ijr_u % it;

      t0 = h1 + h2 + h3;
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      t1 = arma::sum(t0 % innerTemp2, 1);
      arma::mat E_ij_aux = 2 * T * arma::reshape(t1.slice(0), x_new, u_new);
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
      std::cout << "E_ij: " << E_ij << std::endl;
    }
  }
  this->U = U;
  this->X = X;
  arma::mat E_temp = arma::max(E_ij);
  this->E = E_temp(0, 0);
}

// Adaptive Markov Chain approximation
// This function makes an adaptive grid that will produce a certain maximal
// abstraction error defined by epsilon
void faust_t::Adaptive_grid_ReachAvoid_MCapprox_Contr(double epsilon, double T,
                                                      arma::mat SafeSet,
                                                      arma::mat InputSet,
                                                      arma::mat TargetSet) {
  // The dimension of the system
  double dim = SafeSet.n_rows;

  // The length of the edges of the Set
  arma::mat rS = SafeSet.col(1) - SafeSet.col(0);

  // The length of the edges of the Target Set
  arma::mat rT = TargetSet.col(1) - TargetSet.col(0);

  // The Lebesque measure
  arma::mat L_A = arma::prod(rS);

  // The dimension of the input of system
  double dim_u = InputSet.n_rows;

  // The length of the edges of the Set
  arma::mat u = InputSet.col(1) - InputSet.col(0);

  // The Lebewque measure
  arma::mat L_U = arma::prod(u);

  // Derivation of the global Lipschitz constants
  // This factor is used for the calculation of the variable optimratio
  double h_a = this->GlobalLipschitz_Contr(SafeSet, InputSet, 0);

  double h_s = this->GlobalLipschitz_Contr(SafeSet, InputSet, 1);

  // Derivation of the global Lipschitz constant towards the input
  double h_u = this->GlobalLipschitz_Contr(SafeSet, InputSet, 2);

  // Global optimiser without need for defining gradient
  std::vector<double> solution(2);
  nlopt::opt opt(nlopt::LN_COBYLA, 2); // LD_SLSQP
  opt.remove_equality_constraints();
  opt.set_maxeval(2e7);
  opt.set_xtol_rel(1e-6);
  opt.set_ftol_abs(1e-25);

  std::vector<double> dataC(5);
  dataC[0] = epsilon;
  dataC[1] = h_a;
  dataC[2] = h_u;
  dataC[3] = L_A(0, 0);
  dataC[4] = T;
  std::vector<double> data(2);
  data[0] = dim;
  data[1] = dim_u;
  double x10 = (0.5 * epsilon) / (h_a * 2 * L_A(0, 0) * T);
  double x20 = (0.5 * epsilon) / (h_u * 2 * L_A(0, 0) * T);
  std::vector<double> x0(2);
  x0[0] = x10;
  x0[1] = x20;

  opt.add_inequality_constraint(myx1constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx2constraint, NULL, 1e-8);
  opt.add_inequality_constraint(myx3constraint, &dataC, 1e-8);
  opt.set_min_objective(ff_deltas, &data);
  double minf;
  nlopt::result result;
  try {
    result = opt.optimize(x0, minf);
    solution[0] = x0[0];
    solution[1] = x0[1];
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    solution[0] = std::pow((L_A(0, 0) / ((0.5 * epsilon) / h_a) / dim), dim);
    solution[1] =
        std::pow((L_U(0, 0) / ((0.5 * epsilon) / h_u)) / dim_u, dim_u);
  }

  double delta_a = solution[0];
  double delta_u = solution[1];

  double factor =
      h_a /
      (h_a + h_s); // This factor acounts for the MC error in the optimal ratio
  double optimratio = factor * delta_a / delta_u;
  // Translate the solution to the length of the edges
  delta_a = delta_a / sqrt(dim);
  delta_u = delta_u / sqrt(dim_u);

  // Create the location of the representative points
  // delta is adjusted so that the edges of the set are part of the partitions
  arma::mat delta_adj_a = arma::zeros<arma::mat>(1, dim);
  for (size_t i = 0; i < dim; i++) {
    delta_adj_a.col(i) =
        (TargetSet(i, 1) - TargetSet(i, 0)) /
        std::ceil((TargetSet(i, 1) - TargetSet(i, 0)) / delta_a);
  }

  // The adjusted SafeSet to fit with delta_adj
  arma::mat Residuals = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  arma::mat Rin = SafeSet - TargetSet;
  arma::mat Rou = arma::repmat(delta_adj_a.t(), 1, 2);
  arma::mat Rot = arma::zeros<arma::mat>(SafeSet.n_rows, SafeSet.n_cols);
  Rot = arma::join_horiz(-delta_adj_a.t(), 0 * delta_adj_a.t());
  Rou = Rou + Rot;
  Residuals = mod(Rin, Rou);
  arma::mat SafeSet_adj = SafeSet - Residuals;

  // Partition (Safe-Target) into 3^dim-1 separate parts
  arma::mat D = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  arma::mat Dt = arma::zeros<arma::mat>(1, SafeSet.n_cols);
  std::vector<arma::mat> D1;
  std::vector<arma::mat> Dt1;
  for (size_t i = 0; i < dim; i++) {
    arma::mat tempD;
    tempD << SafeSet(i, 0);
    tempD = join_horiz(tempD, SafeSet_adj.row(i)); // concatenate
    arma::mat tempDD;
    tempDD << SafeSet(i, 1);
    tempD = join_horiz(tempD, tempDD);
    D = unique(tempD);
    // std::cout << "D: " <<D.cols(0,D.n_cols-2) << std::endl;
    Dt = D.cols(0, D.n_cols - 2);
    arma::vec tempV = D.t(); // convert to vector
    D1.push_back(tempV);
    arma::vec tempVt = Dt.t();
    Dt1.push_back(tempVt);
  }

  std::vector<arma::mat> F;
  F = nDgrid(Dt1);

  // Cardinality
  double n = F[0].n_elem;
  // Make the output
  arma::mat X = arma::zeros<arma::mat>(
      0, dim * 2); // Initialisation of X with an empty matrix
  for (size_t i = 0; i < n; i++) {
    arma::mat Set = arma::zeros<arma::mat>(dim, 2);
    std::vector<arma::mat> C;
    arma::mat delta_adj_local = arma::zeros<arma::mat>(1, dim);
    int index = 0; // find(D{j}==F{j}(i))+1)
    for (size_t j = 0; j < dim; j++) {
      double left = F[j](i, 0);
      for (size_t k = 0; k < D1[j].n_rows; k++) {
        if (D1[j](k, 0) == F[j](i, 0)) {
          index = k;
        }
      }
      double right = D1[j](index + 1, 0);
      arma::mat templ;
      templ << left;
      arma::mat tempr;
      tempr << right;
      Set.row(j) = join_horiz(templ, tempr);
      // The ndiv variable eliminates rounding errors
      double ndiv = round((Set(j, 1) - Set(j, 0)) / delta_adj_a(j));
      if (ndiv == 0) {
        ndiv = 1;
      }
      delta_adj_local(j) = (Set(j, 1) - Set(j, 0)) / ndiv;
      double start = Set(j, 0) + 0.5 * delta_adj_local(j);
      double Delta = delta_adj_local(j);
      double end = Set(j, 1) - 0.499 * delta_adj_local(j);
      arma::mat A = arma::regspace(start, Delta, end);
      C.push_back(A);
      // 0.499 adjusts for rounding errors; it has no influence on the
      // position of the representative points
    }
    // Construct grid
    C = nDgrid(C);

    // Cardinality
    double m = C[0].n_elem;
    // Make the output
    arma::mat Y = arma::zeros<arma::mat>(m, dim);
    for (size_t j = 0; j < m; j++) {
      for (size_t k = 0; k < dim; k++) {
        Y(j, k) = C[k](j, 0);
      }
    }
    arma::mat J = arma::kron(delta_adj_local, arma::ones<arma::mat>(m, 1));
    arma::mat Res1 = join_horiz(Y, J);
    if (X.n_cols != Res1.n_cols) {
      X = Res1;
    } else {
      X = arma::join_vert(X, Res1);
    }
  }
  this->X = X;

  // Create the location of the representative points
  // delta_u is adjusted so that the edges of the set are part of the partitions
  std::vector<arma::mat> C;
  arma::mat delta_adj_u = arma::zeros<arma::mat>(1, 2 * dim_u);
  for (size_t i = 0; i < dim_u; ++i) {
    delta_adj_u.col(i) = (InputSet(i, 1) - InputSet(i, 0)) /
                         std::ceil((InputSet(i, 1) - InputSet(i, 0)) / delta_u);
    double start = InputSet(i, 0) + 0.5 * delta_adj_u(i);
    double Delta = delta_adj_u(i);
    double end = InputSet(i, 1) - 0.499 * delta_adj_u(i);
    // 0.499 adjusts for rounding errors; it has no influence on the
    // position of the representative points
    arma::mat A = arma::regspace(start, Delta, end);
    // std::cout << "A: " << A << std::endl;
    C.push_back(A);
  }

  // Construct the grid
  C = nDgrid(C);

  // Cardinality
  int m = C[0].n_elem;
  double sm = std::sqrt(m);

  // Make the output
  int index = 0;
  int k = 0;
  arma::mat U = arma::zeros<arma::mat>(m, dim_u);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < dim_u; j++) {
      U(i, j) = C[j](k, index);
    }
    if (k >= (sm - 1)) {
      k = 0;
      index++;
    } else {
      k++;
    }
  }
  U(arma::span(0, m - 1), arma::span(dim_u, 2 * dim_u - 1)) =
      arma::kron(delta_adj_u, arma::ones<arma::mat>(m, 1));

  this->X = X;
  this->U = U;

  // End of uniform part

  // Cardinality
  m = this->X.n_rows;

  // Cardinality of the input
  int q = this->U.n_rows;

  // Extract the Target Set (creation of SafeSet minus TargetSet)
  // Indexing the target set
  arma::mat inner1 = arma::repmat(TargetSet.col(1), 1, m).t();
  arma::mat outer1 = X.cols(0, dim - 1);
  arma::mat lhs = arma::conv_to<arma::mat>::from(outer1 < inner1);
  // std::cout << "lhs: " << lhs << std::endl;

  arma::mat inner2 = arma::repmat(TargetSet.col(0), 1, m).t();
  arma::mat rhs = arma::conv_to<arma::mat>::from(outer1 > inner2);
  // std::cout << "rhs: " << rhs << std::endl;

  arma::mat all = lhs % rhs;

  arma::mat TargetIndex = arma::prod(all, 1);

  // Reshape X so that the first entries are the target set
  arma::uvec in1 = arma::find(TargetIndex == 1);
  arma::uvec in2 = arma::find(TargetIndex != 1);
  X = arma::join_vert(X.rows(in1), X.rows(in2));
  int p = arma::accu(TargetIndex) + 1;

  // The initial Lipschitz constants
  arma::cube h_ijr =
      this->LocalLipschitz_Contr(1, this->X.n_rows, 1, this->U.n_rows);
  arma::cube h_ijr_u =
      this->LocalLipschitzToU_Contr(1, this->X.n_rows, 1, this->U.n_rows);
  arma::cube h_ijr_s = this->LocalLipschitz_Contr(1, this->X.n_rows, 1,
                                                  this->U.n_rows); // TODO check

  // The local Lipschitz constants multiplied by their respective delta's
  arma::cube innerTemp(1, m, q);
  for (size_t j = 0; j < q; j++) {
    innerTemp.slice(j) = arma::repmat(
        arma::pow(arma::sum(arma::pow(X(arma::span(p, m - 1),
                                        arma::span(dim, 2 * dim - 1)),
                                      2)
                                .t(),
                            0),
                  0.5)
            .t(),
        1, m);
  }
  arma::cube h1 = h_ijr % innerTemp;

  // definition of h2
  arma::cube innerTemp2((m - p - 1), 1, q);
  for (size_t j = 0; j < q; j++) {
    innerTemp2.slice(j) = arma::repmat(
        arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0), 0.5)
            .t(),
        (m - p - 1), 1);
  }
  arma::cube h2 = h_ijr_s % innerTemp2;

  // definition of h3
  arma::cube innerTemp3 = arma::ones<arma::cube>(1, 1,
      q);
  for(size_t j = 0 ; j < q; j++) {
    innerTemp3.slice(j) = arma::reshape(arma::pow(arma::sum(arma::pow(U.cols(dim_u,2*dim_u-1),2).t(),0),0.5).t(),1,1);

  }
  arma::cube it;
  for (size_t j = 0; j < 1; j++) {
    arma::mat temp = innerTemp3.tube(0, 0);
    it.slice(j) = arma::repmat(temp, (m - p - 1), m);
  }
  arma::cube h3 = h_ijr_u % it;
  std::cout << "h3: " << h3 << std::endl;

  arma::cube t0 = h1 + h2 + h3;
  for (size_t j = 0; j < q; j++) {
    innerTemp2.slice(j) = arma::repmat(
        arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0), 0.5)
            .t(),
        (m - p - 1), 1);
  }
  arma::cube t1 = arma::sum(t0 % innerTemp2, 1);
  arma::mat E_ij = 2 * T * arma::reshape(t1.slice(0), (m - p - 1), q);
  if (q - 2 > 0) {
    E_ij = join_vert(arma::zeros<arma::mat>(p - 1, q), E_ij);
  }

  // Initialisation counters
  int count_x = p - 1;
  int count_u = 0;
  size_t t = 0;
  arma::mat Ei_comp =
      arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
  while (arma::accu(Ei_comp) > 0) {
    arma::mat K_tin = arma::pow(X.cols(dim, 2 * dim - 1), 2).t();
    arma::mat K_tin1 = arma::pow(arma::sum(K_tin, 0), 0.5).t();
    arma::mat K_uin = arma::pow(U.cols(dim_u, 2 * dim_u - 1), 2).t();
    arma::mat K_uin1 = arma::pow(arma::sum(K_uin, 0), 0.5).t();
    double ratio = 1;       // arma::min(K_tin1)/(arma::min(K_uin1));
    if (ratio > optimratio) // Choose to split either U or X
    {
      t = E_ij.n_rows;
      int end = t - count_x;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.row(count_x)) > epsilon) {
          // finding the index related to maximum dimension
          double m0 = X(count_x, arma::span(dim, 2 * dim - 1)).index_max();
          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(X.row(count_x), 2, 1);
          // the first smaller cell
          Y(0, dim + m0) = X(count_x, dim + m0) / 2;

          Y(0, m0) =
              X(count_x, m0) - Y(0, dim + m0) / 2; // TODO check if m0 or m0-1

          // the second smaller cell
          Y(1, dim + m0) = X(count_x, dim + m0) / 2;
          Y(1, m0) = X(count_x, m0) + Y(1, dim + m0) / 2;

          // Update X
          if ((count_x - 1) < 0) {
            X = X.rows(count_x + 1, X.n_rows - 1);
          } else {
            X = arma::join_vert(X.rows(0, count_x - 1),
                                X.rows(count_x + 1, X.n_rows - 1));
          }

          X = arma::join_vert(X, Y);

          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
            if ((count_x - 1) < 0) {
              E_ij = E_ij.rows(count_x + 1, E_ij.n_cols - 1);
            } else {
              E_ij = arma::join_vert(E_ij.rows(0, count_x - 1),
                                     E_ij.rows(count_x + 1, E_ij.n_cols - 1));
            }
          }
        } else {
          count_x++;
        }
      }
      // Update X values
      this->X = X;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(
          count_x, m, count_u,
          q);
      // The updated Lipschitz constants to x_bar
      h_ijr_s = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The local Lipschitz constants multiplied by their respective delta's
      int x_new = X.n_rows - count_x;
      int u_new = U.n_rows - count_u;

      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X(arma::span(count_x, m - 1),
                                            arma::span(dim, 2 * dim - 1)),
                                          2)
                                    .t(),
                                0),
                      0.5)
                .t(),
            1, m);
      }
      h1 = h_ijr % innerTemp;
      std::cout << "h1: " << h1 << std::endl;

      // definition of h2
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      h2 = h_ijr_s % innerTemp2;

      // definition of h3
      innerTemp3 = arma::ones<arma::cube>(1, 1,q);
      for(size_t j = 0 ; j < q; j++) {
        innerTemp3.slice(j) = arma::reshape(arma::pow(arma::sum(arma::pow(U.cols(dim_u,2*dim_u-1),2).t(),0),0.5).t(),1,1);
      }
      for (size_t j = 0; j < 1; j++) {
        arma::mat temp = innerTemp3.tube(0, 0);
        it.slice(j) = arma::repmat(temp, x_new, u_new);
      }
      h3 = h_ijr_u % it;
      std::cout << "h3: " << h3 << std::endl;
      t0 = h1 + h2 + h3;
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      t1 = arma::sum(t0 % innerTemp2, 1);
      arma::mat E_ij_aux = 2 * T * arma::reshape(t1.slice(0), x_new, u_new);

      E_ij(arma::span(count_x, m - 1), arma::span(count_u, q - 1)) = E_ij;
      Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));

    } else {
      t = E_ij.n_cols;
      int end = t - count_u;
      for (size_t i = 0; i < end; ++i) {
        if (arma::accu(E_ij.col(count_u-1) > epsilon)) {

          // finding the index related to maximum dimension
          double m0 = U(count_u-1, arma::span(dim_u, 2 * dim_u - 1)).index_max();

          // splitting the cell into two cells along its largest edge
          arma::mat Y = arma::repmat(U.row(count_u-1), 2, 1);
          // the first smaller cell
          Y(0, dim_u + m0) = U(count_u-1, dim_u + m0) / 2;

          Y(0, m0) =
              U(count_u-1, m0) - Y(0, dim_u + m0) / 2; // TODO check if m0 or m0-1

          // the second smaller cell
          Y(1, dim_u + m0) = X(count_u-1, dim_u + m0) / 2;
          Y(1, m0) = X(count_u-1, m0) + Y(1, dim_u + m0) / 2;

          // Update U
          if(count_u -2 < 0) {
            U = Y;
          }
          else
          {
            arma::mat Ured = U.rows(0,count_u-2);
            if(count_u < U.n_rows) {
              Ured = arma::join_vert(Ured, U.rows(count_u, U.n_rows - 1));
            }
            U = arma::join_vert(Ured, Y);
          }
          // Update E_i
          if (i == (end - 1)) {
            E_ij.clear();
          } else {
             if(count_u+1 >= E_ij.n_cols - 1) {
               E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                      E_ij.col(E_ij.n_cols - 1));
             }
            else {
              E_ij = arma::join_horiz(E_ij.cols(0, count_u -1),
                                     E_ij.cols(count_u+1, E_ij.n_cols - 1));
              }
          }
        } else {
          count_u++;
        }
        Ei_comp = arma::conv_to<arma::mat>::from(arma::sum(E_ij >= epsilon));
      }
      // Update X values
      this->U = U;

      // The updated Lipschitz constants to x
      h_ijr = this->LocalLipschitz_Contr(count_x, m, count_u, q);

      // The updated Lipschitz constants  to u
      h_ijr_u = this->LocalLipschitzToU_Contr(
          count_x, m, count_u,
          q);
      // The updated Lipschitz constants to x_bar
      h_ijr_s = this->LocalLipschitz_Contr(count_x, m, count_u, q);



      // The local Lipschitz constants multiplied by their respective delta's
      int x_new = X.n_rows - count_x;
      int u_new = U.n_rows - count_u;

      for (size_t j = 0; j < u_new; j++) {
        innerTemp.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X(arma::span(count_x, m - 1),
                                            arma::span(dim, 2 * dim - 1)),
                                          2)
                                    .t(),
                                0),
                      0.5)
                .t(),
            1, m);
      }
      h1 = h_ijr % innerTemp;

      // definition of h2
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      h2 = h_ijr_s % innerTemp2;

      // definition of h3
      innerTemp3 = arma::ones<arma::cube>(1, 1,q);
      for(size_t j = 0 ; j < q; j++) {
        innerTemp3.slice(j) = arma::reshape(arma::pow(arma::sum(arma::pow(U.cols(dim_u,2*dim_u-1),2).t(),0),0.5).t(),1,1);
      }
      for (size_t j = 0; j < 1; j++) {
        arma::mat temp = innerTemp3.tube(0, 0);
        it.slice(j) = arma::repmat(temp, x_new, u_new);
      }
      h3 = h_ijr_u % it;
      t0 = h1 + h2 + h3;
      for (size_t j = 0; j < u_new; j++) {
        innerTemp2.slice(j) = arma::repmat(
            arma::pow(arma::sum(arma::pow(X.cols(dim, 2 * dim - 1), 2).t(), 0),
                      0.5)
                .t(),
            x_new, 1);
      }
      t1 = arma::sum(t0 % innerTemp2, 1);
      arma::mat E_ij_aux = 2 * T * arma::reshape(t1.slice(0), x_new, u_new);
      E_ij.resize(E_ij_aux.n_rows,E_ij_aux.n_cols);
      E_ij = E_ij_aux;
    }
  }
  this->U = U;
  this->X = X;
  arma::mat E_temp = arma::max(E_ij);
  this->E = E_temp(0, 0);
}

void faust_t::export2PRISM(int m, int q, int problem, int N) {
  try {
    std::ofstream file, file2;

    // Get current time to time stamp outputs
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    auto str = oss.str();
    std::string mS = std::to_string(m);
    std::string f_name = "../results/PRISM_" + str + ".prism";

    file.open(f_name);
    file.exceptions(std::ofstream::eofbit | std::ofstream::failbit |
                    std::ofstream::badbit);

    // Opening text %MDP
    file << "//Transition probabilities" << std::endl;

    if (q > 1) {
      file << "mdp" << std::endl;
      file << "module M1 " << std::endl;
    } else {
      file << "dtmc" << std::endl;
      file << "module M1 " << std::endl;
    }

    // Main Body
    arma::mat Tp_col1 = this->Tp[0];
    int num_modes = this->Tp.size();

    m = this->Tp[0].n_cols;
    // Definition of  the states
    file << " //Definition of the states\r\ns : [0.. " << m + 1 << "];\r\n\r\n"
         << std::endl;

    // Creation of the one line for timing
    clock_t begin, end;
    double time;

    begin = clock();

    for (size_t k = 0; k < m; k++) {
      file << "[] s=" << k << " -> ";
      for (size_t i = 0; i < m; i++) {
        if(Tp_col1(k,i) > 0) {
          file << Tp_col1(k, i) << " : (s' = " << i << ") + ";
        }
      }
      double tempS = arma::sum(Tp_col1.row(k));
      file << 1 - tempS << ": (s' =  " << m << ");" << std::endl;
    }
    end = clock();
    time = (double)(end - begin) / CLOCKS_PER_SEC;
    double Ctime = time * (q - 1);
    if (Ctime == 30 * 60) {
      throw "The creation of the PRISM file exceeds time limit of 30 minutes";
    }
    std::cout << "Creation of PRISM file will take: " << Ctime << " minutes"
              << std::endl;

    if (q <= 1) {
      Tp_col1 = this->Tp[0];
      for (size_t k = 0; k < m; k++) {
        file << "[] s=" << k << " -> ";
        for (size_t i = 0; i < m; i++) {
          if(Tp_col1(k,i) > 0) {
            file << Tp_col1(k, i) << " : (s' = " << i << ") + ";
          }
        }
        double tempS = arma::sum(Tp_col1.row(k));
        file << 1 - tempS << ": (s' = " << m << ");" << std::endl;
      }
    } else {
      for (size_t j = 1; j < q; j++) {
        Tp_col1 = this->Tp[j];
        for (size_t k = 0; k < m; k++) {
          file << "[] s=" << k << " -> ";
          for (size_t i = 0; i < m; i++) {
            if(Tp_col1(k,i) > 0) {
              file << Tp_col1(k, i) << " : (s' = " << i << ") + ";
            }
          }
          double tempS = arma::sum(Tp_col1.row(k));
          file << 1 - tempS << ": (s' = " << m << ");" << std::endl;
        }
      }
    }
    file << std::endl;
    // Add the final non-safe state
    file << "[] s = " << m << "-> 1: (s' = " << m
         << "); // This represents the unsafe state \r \n"
         << std::endl;
    file << std::endl;

    file << "endmodule" << std::endl;

    switch (problem) {
    case 1: {
      file << "label \"SafeSet\" = s <= " << m - 1 << ";" << std::endl;
      file << "label \"nonSafeSet\" = s = " << m << ";" << std::endl;
      break;
    }
    case 2: {
      file << "label \"SafeSet\" = s <= " << m - 1 << ";" << std::endl;
      file << "label \"nonSafeSet\" = s = " << m << ";" << std::endl;
      char j;
      file << "label \"TargetSet\" = s = ";
      while (j != 'n') {
        std::cout << "Input target set indices: (type 'n' to stop):"
                  << std::endl;
        std::cin >> j;
        file << j << "|";
      }
      file << ";" << std::endl;
      break;
    }
    case 3: {
      file << "label \"SafeSet\" = s <= " << m - 1 << ";" << std::endl;
      file << "label \"nonSafeSet\" = s = " << m
           << "; // Note that probability to go to this state should be zero"
           << std::endl;
      break;
    }
    default: {
      std::cout << "Option not available" << std::endl;
      exit(0);
      break;
    }
    }
    file.close();

    // For safety and reach avoid create a property file
    if (problem == 1) {
      if (q > 1) {
        std::string f2_name = "../results/PRISM_" + mS + "_" + str + ".props";
        file2.open(f2_name);
        file2.exceptions(std::ofstream::eofbit | std::ofstream::failbit |
                         std::ofstream::badbit);
        file2 << "Pmax=?[ (G <= " << N << "\"SafeSet\")];" << std::endl;
        file2 << "filter(first,Pmax=? [ (G <= " << N << "\"SafeSet\")],s=1);"
              << std::endl;
        file2.close();
      } else {
        std::string f2_name = "../results/PRISM_" + mS + "_" + str + ".props";
        file2.open(f2_name);
        file2.exceptions(std::ofstream::eofbit | std::ofstream::failbit |
                         std::ofstream::badbit);
        file2 << "P=?[ (G <= " << N << "\"SafeSet\")];" << std::endl;
        file2 << "filter(first,P=? [ (G <= " << N << "\"SafeSet\")],s=1);"
              << std::endl;
        file2.close();
      }
    } else if (problem == 2) {
      if (q > 1) {
        std::string f2_name = "../results/PRISM_" + mS + "_" + str + ".props";
        file2.open(f2_name);
        file2.exceptions(std::ofstream::eofbit | std::ofstream::failbit |
                         std::ofstream::badbit);
        file2 << "Pmax=?[ (\"SafeSet\") U<= " << N << "(\"TargetSet\")];"
              << std::endl;
        file2 << "filter(first,Pmax=? [ (\"SafeSet\") U<= " << N
              << "(\"TargetSet\")],s=1);" << std::endl;
        file2.close();
      } else {
        std::string f2_name = "../results/PRISM_" + mS + "_" + str + ".props";
        file2.open(f2_name);
        file2.exceptions(std::ofstream::eofbit | std::ofstream::failbit |
                         std::ofstream::badbit);
        file2 << "P=?[ (\"SafeSet\") U<= " << N << "(\"TargetSet\")];"
              << std::endl;
        file2 << "filter(first,P=? [ (\"SafeSet\") U<= " << N
              << "(\"TargetSet\")],s=1);" << std::endl;
        file2.close();
      }
    }
    std::cout << "Any additional labels need to be added manually by the user"
              << std::endl;

  } catch (std::exception const &e) {
    std::cout << " Unable to create the PRISM text file: " << e.what()
              << std::endl;
    exit(0);
  }
}
void faust_t::formatOutput(double time, std::string cT, int Problem, int timeHorizon) {
  // get initial variables from FAUST results
  // 1. size of abstraction
  int nstates = 0;
  for(size_t ns = 0; ns < this->Tp.size(); ns++) {
    nstates += this->Tp[ns].n_rows;
  }

  std::string nS = std::to_string(nstates); // getting string version
  // 2. abstraction error
  double error = this->E;

  // Saturate error to 1
  if(error > 1) {
    error = 1;
  }

  // 3. whether have control actions or not
  double control = 0;
  if(this->U.n_rows > 0) {
    control = 1;
  }

  // 4. Adjust problem solution to include the abstracation error  i.e.Vnew = V-error
  std::cout << "V " << V <<std::endl;
  this->V = this->V - error*arma::ones<arma::vec>(this->V.n_rows);

  // Check if Vnew has elements < 0, then saturate to 0
  this->V = arma::clamp(this->V, 0,1);  // replace each value < 0 with 0

  std::cout << std::endl;
  std::cout << "---------------------------------------" <<std::endl;
  std::cout << "Method |  |Q| states | Time(s) | Error " << std::endl;
  std::cout << "---------------------------------------" <<std::endl;
  std::cout << "MDP    | "<< nstates << "     | " << time << "     | " << error << std::endl;
  std::cout << "---------------------------------------" <<std::endl;
  std::cout << std::endl;

  // Option to export to file results
  std::ofstream myfile;
  std::string exportOpt;
  std::string str("y");
  std::string str1("yes");
  std::string str2("Y");
  std::string str3("YES");
  exportOpt = "yes";
  std::cout << "Would you like to store the results generated by FAUST^2 [y- yes, n - no] " << std::endl;
  std::cin >> exportOpt;
  if ((exportOpt.compare(str) == 0) || (exportOpt.compare(str1) == 0)|| (exportOpt.compare(str2) == 0)|| (exportOpt.compare(str3) == 0)) {
    if(checkFolderExists("../results") == -1) {
      if(mkdir("../results", 0777) == -1) {
        std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
        exit(0);
       }
    }
    // Store results in file
    std::string f_name = "../results/FAUST_Run_Time_" + cT + ".txt";
    myfile.open(f_name);
    myfile << time <<std::endl;
    myfile.close();

    // Storing value functions
    std::string Problem_solutionname = "../results/FAUST_Problem_solution_"+ nS+ "_"  + cT + ".txt";
    this->V.save(Problem_solutionname, arma::raw_ascii);

    // Storing Transition matrix for each mode
    arma::mat Tp_all = this->Tp[0];
    for(size_t j = 1; j < this->Tp.size(); ++j) {
      Tp_all = join_vert(Tp_all, this->Tp[j]);
    }
    std::string Tp_name =  "../results/FAUST_Transition_matrix_" + nS + "_" + cT + ".txt";
    Tp_all.save(Tp_name,arma::raw_ascii);
    Tp_all.save("../Tp.txt", arma::raw_ascii);

    // Storing representative points
    std::string X_name = "../results/FAUST_Representative_points_" + nS + "_" + cT + ".txt";
    this->X.save(X_name, arma::raw_ascii);

    if(control) {
      // Store control representative points
      std::string U_name = "../results/FAUST_Representative_U_points_" + nS + "_" + cT + ".txt";
      this->U.save(U_name, arma::raw_ascii);

      // Store optimal policy
      std::string Pol_name = "../results/FAUST_OptimalPolicy_" + nS  + "_" + cT + ".txt";
      this->OptimalPol.save(Pol_name, arma::raw_ascii);

    }

    // Storing abstraction error
    f_name = "../results/FAUST_E_" + nS + "_" + cT + ".txt";
    myfile.open(f_name);
    myfile << error<< std::endl;
    myfile.close();
    f_name = "../E.txt";
    myfile.open(f_name);
    myfile << error<< std::endl;
    myfile.close();
 }

 // Option to export to PRISM
 std::cout << "Would you like to export abstraction into PRISM format [y "
              "- yes, n - no] "
           << std::endl;
 std::cin >> exportOpt;
 if ((exportOpt.compare(str) == 0) || (exportOpt.compare(str1) == 0)|| (exportOpt.compare(str2) == 0)|| (exportOpt.compare(str3) == 0)) {
   std::cout << "Export to PRISM" << std::endl;
   this->export2PRISM(nstates, this->Tp.size(), Problem, timeHorizon);
 }


 // Plotting of grid
 // 1. Dimension Definition
 unsigned dim = this->X.n_cols/2;
 std::cout << "Would you like to generate abstraction figure [y "
              "- yes, n - no] "
           << std::endl;
 std::cin >> exportOpt;
 if ((exportOpt.compare(str) == 0) || (exportOpt.compare(str1) == 0) || (exportOpt.compare(str2) == 0)|| (exportOpt.compare(str3) == 0)) {
   if(checkFolderExists("../results") == -1) {
     if(mkdir("../results", 0777) == -1) {
       std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
       exit(0);
      }
   }
   if (dim == 2) {
     // 2. Size of Abstraction
     unsigned m = this->X.n_rows;

     myfile.open("../results/FAUST_gridPlot.py");
     myfile << "import numpy as np \n";
     myfile << "from mpl_toolkits.axes_grid1 import make_axes_locatable \n";
     myfile << "import matplotlib.pyplot as plt \n";
     myfile << "import matplotlib.patches as patches\n";
     myfile << "import matplotlib.path as path \n";
     myfile << "import matplotlib.colorbar as cbar\n\n";

     myfile << "fig = plt.figure()\n";
     myfile << "ax = fig.add_subplot(111)\n"; //rectangles

     // store in grid for python to plot
     //1. Preparing X data to form grid
     arma::mat X = this->X;
     arma::mat innerX = {{1,1,1,1}, {0,0,0,0}, {-0.5,0.5,0.5,-0.5},{0,0,0,0}};
     arma::mat X_data = X*innerX;
     X_data = X_data.t();
     arma::mat left = X_data.row(0);
     arma::mat right = X_data.row(1);

     arma::mat innerY = {{0,0,0,0}, {1,1,1,1},{0,0,0,0}, {-0.5,0.5,0.5,-0.5}};
     arma::mat Y_data = X*innerY;
     Y_data = Y_data.t();
     arma::mat bottom = Y_data.row(0);
     arma::mat top = Y_data.row(1);
     arma::mat C_data = this->V;

     myfile << "left" << "= np.array(["; // Creating variable for x-coordinates of grid
     for (unsigned j = 0; j < m; ++j) {
       if (j == (m-1)) {
          myfile << left.col(j) << "])\n" << std::endl;
       }
       else {
          myfile << left.col(j) << ",";
       }
     }

     myfile << "right" << "= np.array(["; // Creating variable for x-coordinates of grid
     for (unsigned j = 0; j < m; ++j) {
       if (j == (m-1)) {
          myfile << right.col(j) << "])\n" << std::endl;
       }
       else {
          myfile << right.col(j) << ",";
       }
     }

     myfile << "top" << "= np.array(["; // Creating variable for x-coordinates of grid
     for (unsigned j = 0; j < m; ++j) {
       if (j == (m-1)) {
          myfile << top.col(j) << "])\n" << std::endl;
       }
       else {
          myfile << top.col(j) << ",";
       }
     }

     myfile << "bottom" << "= np.array(["; // Creating variable for x-coordinates of grid
     for (unsigned j = 0; j < m; ++j) {
       if (j == (m-1)) {
          myfile << bottom.col(j) << "])\n" << std::endl;
       }
       else {
          myfile << bottom.col(j) << ",";
       }
     }

     myfile << "c" << "= np.array(["; // Creating variable for colormap values
     for (unsigned j = 0; j <  V.n_rows; ++j) {
       if (j == (V.n_rows-1)) {
          myfile   << C_data.row(j) << "])\n" << std::endl;
       }
       else {
          myfile << C_data.row(j)<< ", ";
       }
      }
     myfile << "# creation of colorbar and mapping\n";
     myfile << "normal = plt.Normalize(c.min(), c.max())\n";
     myfile << "cmap = plt.cm.binary(normal(c))\n\n";

     myfile << "nrects = len(left)\n";
     myfile << "nverts = nrects*(1+3+1)\n";
     myfile << "verts = np.zeros((nverts, 2))\n";
     myfile << "codes = np.ones(nverts, int) * path.Path.LINETO\n";
     myfile << "codes[0::5] = path.Path.MOVETO\n";
     myfile << "codes[4::5] = path.Path.CLOSEPOLY\n";
     myfile << "verts[0::5,0] = left\n";
     myfile << "verts[0::5,1] = bottom\n";
     myfile << "verts[1::5,0] = left\n";
     myfile << "verts[1::5,1] = top\n";
     myfile << "verts[2::5,0] = right\n";
     myfile << "verts[2::5,1] = top\n";
     myfile << "verts[3::5,0] = right\n";
     myfile << "verts[3::5,1] = bottom\n\n";

     myfile << "barpath = path.Path(verts, codes)\n";
     myfile << "patch = patches.PathPatch(barpath, facecolor='none', alpha=0.5)\n";
     myfile << "ax.add_patch(patch)\n\n";
     myfile << "ax.set_xlim(left.min(), right.max())\n";
     myfile << "ax.set_ylim(bottom.min(), top.max())\n\n";

     myfile << "# Super imposing the resulting solution \n \n";
     myfile << "delta_x = " << std::ceil(std::sqrt(X.n_rows)) << " \n";
     myfile << "x = np.linspace(left.min(), right.max(),num=delta_x )\n";
     myfile << "y = np.linspace(bottom.min(), top.max(), num=delta_x)\n";
     myfile << "X, Y = np.meshgrid(x,y)\n";
     myfile << "Z = np.zeros((len(x),len(y)))\n";
     myfile << "count = 0\n";
     myfile << "for i in range(len(x)-1):\n";
     myfile << "   for j in range(len(y)-1):\n";
     myfile << "       Z[j,i] = c[count]\n";
     myfile << "       count  = count + 1\n";


     myfile <<"plt.pcolor(X, Y, Z, cmap=plt.cm.binary, norm=normal)\n\n";
     myfile << "# create an axes on the right side of ax. The width of cax will be 5%\n";
     myfile << "# of ax and the padding between cax and ax will be fixed at 0.05 inch.\n";
     myfile << "divider = make_axes_locatable(ax)\n";
     myfile << "cax = divider.append_axes(\"right\", size=\"10%\", pad=0.5)\n";
     myfile << "plt.colorbar(cax=cax)\n";
     myfile << "plt.savefig(\"FAUST_SatProbability.svg\",dpi=150)\n";
     myfile << "plt.show()\n";



     myfile.close();
    }
    else if(dim == 1){
      // 2. Size of Abstraction
      unsigned m = this->X.n_rows;

      myfile.open("../results/FAUST_gridPlot.py");
      myfile << "import numpy as np \n";
      myfile << "from mpl_toolkits.axes_grid1 import make_axes_locatable \n";
      myfile << "import matplotlib.pyplot as plt \n";
      myfile << "import matplotlib.patches as patches\n";
      myfile << "import matplotlib.path as path \n\n";

      myfile << "fig = plt.figure()\n";
      myfile << "ax = fig.add_subplot(111)\n"; //rectangles

      // store in grid for python to plot
      //1. Preparing X data to form grid
      arma::mat X = this->X;
      arma::mat left = X.col(0);

      arma::mat C_data = this->V;

      myfile << "statespace" << "= np.array(["; // Creating variable for x-coordinates of grid
      for (unsigned j = 0; j < m; ++j) {
        if (j == (m-1)) {
           myfile << left.row(j) << "])\n" << std::endl;
        }
        else {
           myfile << left.row(j) << ",";
        }
      }

      myfile << "c" << "= np.array(["; // Creating variable for probability values
      for (unsigned j = 0; j <  V.n_rows; ++j) {
        if (j == (V.n_rows-1)) {
           myfile   << C_data.row(j) << "])\n" << std::endl;
        }
        else {
           myfile << C_data.row(j)<< ", ";
        }
       }
      myfile << "ax.set_xlim(statespace.min(), statespace.max())\n";
      myfile << "ax.set_ylim(0, 1)\n\n";

      myfile <<"plt.plot(statespace,c)\n\n";
      myfile << "# labelling the axes on the right side of ax. \n";
      myfile << "plt.xlabel('State space')\n";
      myfile << "plt.ylabel('Probability')\n";
      myfile << "plt.savefig(\"FAUST_SatProbability.svg\",dpi=150)\n";
      myfile << "plt.show()\n";

      myfile.close();
    }
    else {
      std::cout << "Dimension > 2; plots for dimension == 3 is a work in progress  " <<std::endl;

    }
  }
}
