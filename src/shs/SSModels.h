/*
 * SSModels.h
 *
 *  Created on: 10 Jan 2018
 *      Author: nathalie
 */


#ifndef SSMODELS_H_
#define SSMODELS_H_

#include "ExtDatat.h"
#include "matio.h" /*Reading .arma::mat files*/
#include "utility.h"
#include <armadillo>

typedef arma::mat (*ssmodels_func)(unsigned n, const double *x, void *fdata);

typedef ssmodels_func func; // nlopt::func synoynm

typedef emptyMatrix = arma::zeros<arma::mat>(1,1);

class ssmodels_t
{
  public:
  ssmodels_t() : xDim{0}, uDim{0}, dDim{0}, deltaT{-1}, A{}, B{}, F{}, Q{}, N{}, Sigma{} {}
  ssmodels_t(const int _xDim, const int uDim, const int dDim) noexcept : xDim{_xDim}, uDim{_uDim}, dDim{_dDim}, deltaT{-1}, A{}, B{},F{}, Q{}, N{}, Sigma{} {}

  // Model type : x_k+1 = Ax_k + Bu_k + N(x_k (.) u_k)+ F*Norm(0, Sigma) + Q  : Bilinear
  ssmodels_t(const int _deltaT, const arma::mat _A, const arma::mat _B, arma::mat _N, const arma::mat _F, const arma::mat _Q, const arma::mat _Sigma) noexpect:
            xDim{_A.n_rows}, uDim{_B.n_rows}, dDim{_D.n_rows}, deltaT{_deltaT}, A{_A}, B{_B}, F{_F}, Q{_Q}, N{_N}, Sigma{_Sigma} {}

  // Model type : x_k+1 = Ax_k + Bu_k + F*Norm(0, Sigma) + Q  : Bilinear
  ssmodels_t(int _deltaT, arma::mat _A, arma::mat _B, arma::mat _F, arma::mat _Q, arma::mat _Sigma)
  {
    return ssmodels_t(_deltaT, _A, _B, _F, arma::zeros<arma::matrix>(1,1).reset(), _Q, _Sigma);
  };

  // Model type : x_k+1 = Ax_k + Bu_k + F*Norm(0, Sigma) : Bilinear
  ssmodels_t(int _deltaT, arma::mat _A, arma::mat _B, arma::mat _F, arma::mat _Sigma)
  {
    return ssmodels_t(_deltaT, _A, _B, _F, arma::zeros<arma::matrix>(1,1).reset(), arma::zeros<arma::matrix>(1,1).reset(), _Sigma);
  };

  // Model type : x_k+1 = Ax_k + F*Norm(0, Sigma) : Bilinear
  ssmodels_t(int _deltaT, arma::mat _A, arma::mat _F, arma::mat _Sigma)
  {
    return ssmodels_t(_deltaT, _A, arma::zeros<arma::matrix>(1,1).reset(), _F, arma::zeros<arma::matrix>(1,1).reset(), arma::zeros<arma::matrix>(1,1).reset(), _Sigma);
  };

  // Model type : x_k+1 = Ax_k + F*Norm(0, Sigma) + Q : Bilinear
  ssmodels_t(int _deltaT, arma::mat _A, arma::mat _F, arma::mat _Q, arma::mat _Sigma)
  {
    return ssmodels_t(_deltaT, _A, arma::zeros<arma::matrix>(1,1).reset(), _F, arma::zeros<arma::matrix>(1,1).reset(), _Q, _Sigma);
  };

  // modelialise state space model based on data from input Matlab file for MDP abstraction based abstractions
  void obtainSSfromMat(const char *fn, ssmodels_t &model , int currentMode);
  void obtainSSfromMat(const char *fn, ssmodels_t &model)
  {
    return obtainSSfromMat(fn, model, 0);
  };

  // modelialise state space model based on data from input Matlab file for IMDB based abstractions
  void obtainIMDPfromMat(const char *fn, ssmodels_t &model, int currentMode); //TODO(ncauchi) replace all BMDP to IMDP

  /// Update ssmodel based on input data read from MATLAB file for MDP abstraction based abstractions
  // 0. Variable name = "Variable + mode"
  // 1. populate state space based on variable name
  // 2. Check if dimensions match
  void populateStateSpaceModelForMDP(matvar_t &content, int currentMode);
  void populateStateSpaceModelForMDP(matvar_t &content)
  {
    return populateStateSpaceModelForMDP(model, content 0);
  };

  /// Update ssmodel based on input data read from MATLAB file for IMDP based abstractions
  // 0. Variable name = "Variable + mode"
  // 1. populate state space based on variable name
  // 2. Check if dimensions match
  void populateStateSpaceModelForIMDP(matvar_t &content, int currentMode);
  // Fill state space matrices from corresponding variable in MATLAB file
  const arma::mat fillMatrix(matvar_t &content);
  // Read contents from MATLAB file of type cell,  this is used for Tq when in hybrid mode
  const std::string readCells(matvar_t &content);
  void checkModel(ssmodels_t &model);

  const arma::mat getDeterministicPartOfModelUpdate(const arma::mat x_k, const arma::mat u_k, const arma::mat d_k);
  const arma::mat getDeterministicPartOfModelUpdate(const arma::mat x_k, const arma::mat z_k);
  const arma::mat getDeterministicPartOfModelUpdate(const arma::mat x_k);
  const arma::mat ssmodels_t::getStochasticModelUpdate(const arma::mat x_k, const arma::mat u_k, const arma::mat d_k);
private:
  int xDim; // Dimension of state spaces i.e. number of continuous var
  int uDim; // Dimension of control input
  int dDim; // Dimension of disturbances vector - Note all dimensions can be
             // either input or computed

  double deltaT; // If <= 0 then in CT model
  arma::mat A;
  arma::mat B;
  arma::mat F; // F for disturbances of in case of switching systems to correspond to the G function in Gw[k]
  arma::mat N; // N for bilinear models
  arma::mat Q;
  arma::mat Sigma; // If Zeros then deterministic models

  virtual ~ssmodels_t();
};

static arma::mat readMatrixMat(const char *fn, const char *var) {
  mat_t *matf;
  matvar_t *matvar, *contents;
  arma::mat tq;
  try {
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model baseatfd on variable name
      contents = Mat_VarRead(matf, var);
      if (contents == NULL) {
        std::cout << "Variable not found in file" << std::endl;
      } else {
        if (contents->data_type == MAT_T_DOUBLE) {
          std::string str;
          size_t stride = Mat_SizeOf(contents->data_type);
          char *data = (char *)contents->data;
          unsigned i, j = 0, r = contents->dims[0], c = 0;
          if (contents->rank > 1) {
            c = contents->dims[1];
          } else {
            c = 1;
          }
          for (i = 0; i < r; i++) {
            for (j = 0; j < c; j++) {
              size_t idx = r * j + i;
              void *t = data + idx * stride;
              char substr[100];
              sprintf(substr, "%g",
                      *(double *)t); // Assumes values are of type double
              str.append(substr);
              str.append(" ");
            }
            str.append(";");
          }
          std::vector<std::string> x = splitStr(str, ';');
          // int numEl = x.size();
          // std::cout<< "x size: " << numEl<< std::endl;
          if (r == c) {
            tq = strtodMatrix(x);
          } else {
            tq = strtodMatrix(x, c);
          }

        } else {
          std::cout << "Incorrect format" << std::endl;
        }
      }
    } else // unsuccessfull in opening file
    {
      throw "Error opening arma::mat file";
    }
    Mat_Close(matf);
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
  }

  return tq;
}

static std::vector<arma::mat> read3DMatrixMat(const char *fn, const char *var) {
  mat_t *matf;
  matvar_t *matvar, *contents;
  std::vector<arma::mat> tq;
  try {
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      contents = Mat_VarRead(matf, var);
      if (contents == NULL) {
        std::cout << "Variable not found in file" << std::endl;
      } else {
        if (contents->data_type == MAT_T_DOUBLE) {
          std::string str;
          size_t stride = Mat_SizeOf(contents->data_type);
          char *data = (char *)contents->data;
          unsigned i, n, j = 0, r = contents->dims[0], c = 0, m = 0;
          if (contents->rank > 1) {
            c = contents->dims[1];
            if (contents->rank > 2) {
              m = contents->dims[2];
            } else {
              m = 1;
            }

          } else {
            c = 1;
          }

          std::cout << "m: " << m << "c: " << c << "r: " << r << std::endl;

          for (n = 0; n < m; n++) {
            for (i = 0; i < r; i++) {
              for (j = 0; j < c; j++) {
                size_t idx = r * j + i;
                void *t = data + idx * stride;
                char substr[100];
                sprintf(substr, "%g",
                        *(double *)t); // Assumes values are of type double
                str.append(substr);
                str.append(" ");
              }
              str.append(";");
            }
            std::vector<std::string> x = splitStr(str, ';');
            if (r == c) {
              tq.push_back(strtodMatrix(x));
            } else {
              tq.push_back(strtodMatrix(x, c));
            }
            str.clear();
            //   std::cout << " Dim: " << tq[n].n_rows() << " ,"
            //   <<tq[n].n_cols() << std::endl;
          }

        } else {
          std::cout << "Incorrect format" << std::endl;
        }
      }
    } else // unsuccessfull in opening file
    {
      throw "Error opening arma::mat file";
    }
    Mat_Close(matf);
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
  }

  return tq;
}

#endif /* SSMODELS_H_ */
