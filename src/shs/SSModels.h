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

class ssmodels_t {
public:
  int x_dim; // Dimension of state spaces i.e. number of continuous var
  int u_dim; // Dimension of control input
  int d_dim; // Dimension of disturbances vector - Note all dimensions can be
             // either input or computed
  double delta_t; // If <= 0 then in CT model
  arma::mat A;
  arma::mat B;
  arma::mat C;
  arma::mat F; // F for disturbances of in case of switching systems to
               // correspond to the G function in Gw[k]
  arma::mat N; // N for bilinear models
  arma::mat Q;
  arma::mat sigma; // If Zeros then deterministic models
public:
  ssmodels_t();
  ssmodels_t(int x_d, int u_d, int d_d);
  ssmodels_t(int x_d, arma::mat Am, arma::mat Sm);
  ssmodels_t(arma::mat Am, arma::mat Sm);
  ssmodels_t(arma::mat Am, arma::mat Qm, arma::mat Sm);
  ssmodels_t(arma::mat Am, arma::mat Bm, arma::mat Qm, arma::mat Sm);
  ssmodels_t(int x_d, arma::mat Am, arma::mat Fm, arma::mat Sm, int sig);
  ssmodels_t(int x_d, arma::mat Am, arma::mat Bm, arma::mat Sm);
  ssmodels_t(int dt, arma::mat Am, arma::mat Bm, arma::mat Nm, arma::mat Sm);
  ssmodels_t(int x_d, int u_d, int d_d, double d_t, arma::mat Am, arma::mat Bm,
             arma::mat Cm, arma::mat Fm, arma::mat Qm, arma::mat Sm);
  ssmodels_t(double d_t, arma::mat Am, arma::mat Bm, arma::mat Cm, arma::mat Fm,
             arma::mat Qm, arma::mat Sm);
  ssmodels_t(double d_t,  arma::mat Am, arma::mat Bm, arma::mat Fm,arma::mat Qm, arma::mat Sm);
  ssmodels_t(arma::mat Am, arma::mat Bm, arma::mat Nm, arma::mat Qm, arma::mat Sm);
  void obtainSSfromMat(const char *fn, ssmodels_t &init);
  void obtainSSfromMat(const char *fn, ssmodels_t &init, int curMode);
  void obtainBMDPfromMat(const char *fn, ssmodels_t &init, int curMode);
  void populate(ssmodels_t &init, matvar_t &content);
  void populate(ssmodels_t &init, matvar_t &content, int curMode);
  void populateBMDP(ssmodels_t &init, matvar_t &content, int curMode);
  int checkDimFields(int dim, arma::mat mt);
  void checkModel(ssmodels_t &init);
  int checkMatrices(int size, int rows, int cols);
  arma::mat fillMatrix(ssmodels_t &init, matvar_t &content);
  std::string readCells(matvar_t &content);
  void printmodel(ssmodels_t &init);
  virtual ~ssmodels_t();
  arma::mat updateLTI(ssmodels_t &old, arma::mat x_k, arma::mat u_k);
  arma::mat updateLTIad(ssmodels_t &old, arma::mat x_k, arma::mat u_k,
                        arma::mat d_k);
  arma::mat updateLTIst(ssmodels_t &old, arma::mat x_k, arma::mat u_k,
                        arma::mat d_k);
  arma::mat updateLTI(arma::mat A, arma::mat B, arma::mat Q, arma::mat x_k,
                      arma::mat u_k);
  arma::mat updateLTIad(arma::mat A, arma::mat B, arma::mat F, arma::mat Q,
                        arma::mat x_k, arma::mat u_k, arma::mat d_k);
  arma::mat updateLTI(arma::mat A, arma::mat Q, arma::mat x_k);
  arma::mat updatefunc(func f, void *f_data, arma::mat x_k);
  arma::mat updateBi(arma::mat A, arma::mat B, arma::mat N, arma::mat Q,
                     arma::mat x_k, arma::mat u_k);
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
