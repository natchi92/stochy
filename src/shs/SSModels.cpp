//

#include "SSModels.h"
#include "matio.h" /*Reading MATLAB mat files*/
#include <armadillo>

void ssmodels_t::obtainSSfromMat(const char *fn, ssmodels_t &model, int currentMode)
{
  // Reading model file input in MATLAB mat format and storing into ssmodel class
  mat_t *matf;
  matvar_t *matvar, *contents;
  try
  {
    // Read mat file
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      matvar = Mat_VarReadNextInfo(matf);
      while (matvar != NULL)
      {
        contents = Mat_VarRead(matf, matvar->name);
        if (contents != NULL)
        {
          populateStateSpaceModelForMDP(*contents, currentMode);
        }
        Mat_VarFree(matvar);
        matvar = NULL;
        contents = NULL;
        matvar = Mat_VarReadNextInfo(matf);
      }
    } else // unsuccessful in opening file
    {
      throw "ERROR: Could not open MAT file";
    }
    Mat_Close(matf);
  }
  catch (const char *msg)
  {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

void ssmodels_t::obtainIMDPfromMat(const char *fn, ssmodels_t &model,int currentMode)
{
  // Reading model file input in MATLAB mat format and storing into ssmodel class
  mat_t *matf;
  matvar_t *matvar, *contents;
  try
  {
    // Read mat file
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      matvar = Mat_VarReadNextInfo(matf);
      while (matvar != NULL)
      {
        contents = Mat_VarRead(matf, matvar->name);
        if (contents != NULL)
        {
          populateStateSpaceModelForIMDP(*contents, currentMode);
        }
        Mat_VarFree(matvar);
        matvar = NULL;
        contents = NULL;
        matvar = Mat_VarReadNextInfo(matf);
      }
    }
    else // unsuccessful in opening file
    {
      throw "ERROR: Could not open MAT file";
    }
    Mat_Close(matf);
  }
  catch (const char *msg)
  {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

void ssmodels_t::populateStateSpaceModelForMDP(matvar_t &content, int currentMode)
{
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity
  std::string As("A"), Bs("B"), Cs("C"), Fs("F"), Qs("Q"), Sigmas("Sigma"), Ns("N");
  std::string mode = std::to_string(currentMode);
  As += mode;
  Bs += mode;
  Cs += mode;
  Fs += mode;
  Qs += mode;
  Sigmas += mode;
  Ns += mode;
  const char *cA = As.c_str(), *cB = Bs.c_str(), *cC = Cs.c_str();
  const char *cF = Fs.c_str(), *cQ = Qs.c_str(), *cSigmas = Sigma.c_str();
  const char *cN = Ns.c_str();

  if (strcmp(content.name, cA) == 0)
  {
    A = fillMatrix(model, content);
    xDim = A.n_rows;
  }
  else if (strcmp(content.name, cB) == 0)
  {
    B = fillMatrix(model, content);
    uDim = B.n_rows;
  }
  else if (strcmp(content.name, cC) == 0)
  {
    C = fillMatrix(model, content);
  }
  else if (strcmp(content.name, cF) == 0)
  {
    F = fillMatrix(model, content);
    dDim = F.n_rows;
  }
  else if (strcmp(content.name, cQ) == 0)
  {
    Q = fillMatrix(model, content);
  }
  else if (strcmp(content.name, cSigma) == 0)
  {
    Sigma = fillMatrix(model, content);
  }
  else if (strcmp(content.name, cN) == 0)
  {
    N = fillMatrix(model, content);
  }
  else if (strcmp(content.name, "dt") == 0)
  {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double dt = strtod(str, NULL);
    deltaT = dt;
  }

  std::cout << "Populated state space model for MDP abstraction using mat file" << std::endl;
}

void ssmodels_t::populateStateSpaceModelForIMDP(matvar_t &content, int currentMode)
{
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity
  std::string As("A"), Bs("B"), Fs("G"), Sigmas("Sigma"), Ns("N");
  std::string mode = std::to_string(currentMode);
  As += mode;
  Bs += mode;
  Fs += mode;
  Sigmas += mode;
  Ns += mode;
  const char *cA = As.c_str(), *cB = Bs.c_str(), *cN = Ns.c_str();
  const char *cF = Fs.c_str(), *cSigmas = Sigmas.c_str();

  if (strcmp(content.name, cA) == 0)
  {
    A = fillMatrix(model, content);
    xDim = A.n_rows;
  }
  else if (strcmp(content.name, cB) == 0)
  {
    B = fillMatrix(model, content);
    uDim = B.n_rows;
  }
  else if (strcmp(content.name, cF) == 0)
  {
    F = fillMatrix(model, content);
    dDim = F.n_rows;
  }
  else if (strcmp(content.name, cSigma) == 0)
  {
    Sigma = fillMatrix(model, content);
  }
  else if (strcmp(content.name, cN) == 0)
  {
    N = fillMatrix(model, content);
  }

  std::cout << "Populated state space model for IMDP abstraction using mat file" << std::endl;
}

const arma::mat ssmodels_t::fillMatrix(matvar_t &content)
{
  size_t stride = Mat_SizeOf(content.data_type);
  char *data = (char *)content.data;
  arma::mat mt(content.dims[0], content.dims[1]);
  unsigned i, j = 0;
  for (i = 0; i < content.dims[0] && i < 15; i++)
  {
    for (j = 0; j < content.dims[1] && j < 15; j++)
    {
      size_t idx = content.dims[0] * j + i;
      char str[64];
      void *t = data + idx * stride;
      sprintf(str, "%g", *(double *)t); // Assumes values are of type double
      double val = strtod(str, NULL);
      mt(i, j) = val;
    }
  }
  return mt;
}

const std::string ssmodels_t::readCells(matvar_t &content)
{
  std::string str;
  try
  {
    matvar_t *cell;
    unsigned int i, j, k = 0;
    // Get all the cells
    for (i = 0; i < content.dims[0] * content.dims[1]; i++) {
      cell = Mat_VarGetCell(&content, i);
      char *data = (char *)cell->data;
      if (NULL == cell) {
        throw "Error reading 'Tq' variable information";
      }
      for (k = 0; k < cell->dims[0]; k++) {
        for (j = 0; j < cell->dims[1]; j++) {
          char substr[100];
          sprintf(substr, "%c", data[j * cell->dims[0] + k]);
          str.append(substr);
        }
        str.append(" ");
      }
    }
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
  return str;
}

void ssmodels_t::checkModel()
{
  try
  {
    xDim = (xDim == A.n_rows) ? xDim : A.n_rows;
    uDim = (uDim == B.n_rows) ? uDim : B.n_rows;
    dDim = (dDim == F.n_rows) ? dDim : F.n_rows;

    // Assuming square matrix which is as should be
    if (A.n_rows  !=  xDim || A.n_cols != xDim)
    {
      throw "Incorrect dimensions of A-matrix";
    }

    if (uDim != 0)
    {
      if (B.n_rows != xDim || B.n_cols != uDim)
      {
        throw "Incorrect dimensions of B-matrix";
      }
    }

    if (F.n_rows != xDim || F.n_cols != dDim)
    {
      throw "Incorrect dimensions of F-matrix";
    }

    if (Q.n_rows != xDim || Q.n_cols != 1)
    {
      throw "Incorrect dimensions of Q-matrix";
    }

    if (Sigma.n_rows != xDim || Sigma.n_cols != xDim)
    {
      throw "Incorrect dimensions of Sigma-matrix";
    }

    std::cout << "Correctly compiled state space model" << std::endl;
  }
  catch (const char *msg)
  {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

const arma::mat ssmodels_t::getDeterministicPartOfModelUpdate(const arma::mat x_k, const arma::mat u_k, const arma::mat d_k)
{
  if (uDim == 0)
  {
    if (dDim != 0)
    {
      return A*x_k + F*d_k + Q;
    }
    else
    {
      return A*x_k + Q;
    }
  }
  else
  {
    if (!N.is_empty())
    {
      if (dDim != 0)
      {
        return A*x_k + B*u_k + N * kron(u_k.col(0), x_k) + F*d_k + Q;
      }
      else
      {
        return A*x_k + B*u_k + N * kron(u_k.col(0), x_k) + Q;
      }
    }
    else
    {
      if (dDim != 0)
      {
        return A*x_k + B*u_k  + F*d_k + Q;
      }
      else
      {
        return A*x_k + B*u_k + Q;
      }
    }
  }
}

const arma::mat ssmodels_t::getDeterministicPartOfModelUpdate(const arma::mat x_k, const arma::mat z_k)
{
  if (uDim > 0 && dDim == 0)
  {
    getDeterministicPartOfModelUpdate(x_k, z_k, emptyMatrix);
  }
  else if (uDim == 0 && dDim > 0)
  {
    getDeterministicPartOfModelUpdate(x_k, emptyMatrix,  d_k);
  }
  else
  {
    // We should never enter here
    std::cout << "ERROR: Incorrect function input used to update model" << std:: endl;
  }
};

const arma::mat ssmodels_t::getDeterministicPartOfModelUpdate(const arma::mat x_k)
{
  if (uDim == 0 && dDim == 0)
  {
    getDeterministicPartOfModelUpdate(x_k, emptyMatrix, emptyMatrix);
  }
  else
  {
    // We should never enter here
    std::cout << "ERROR: Incorrect function input used to update model" << std:: endl;
  }
};

// TODO (ncauchi) replaces updateLTIst
const arma::mat ssmodels_t::getStochasticModelUpdate(const arma::mat x_k, const arma::mat u_k, const arma::mat d_k)
{
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<double> norm(0, 1);
  double nr = Sigma.n_cols;
  arma::mat dW = arma::zeros<arma::mat>(nr, monte);
  for(size_t j = 0; j < monte; j++)
  {
    for (size_t i= 0; i < nr; i++)
    {
      dW(i,j) = sqrt(deltaT) * norm(generator); // Weiner increment
    }
  }

  if (uDim  == 0)
  {
    if (dDim != 0)
    {
      return getDeterministicPartOfModelUpdate(x_k, d_k) + Sigma*dW;
    }
    else
    {
      return getDeterministicPartOfModelUpdate(x_k) + Sigma*dW;
    }
  }
  else
  {
    if (dDim != 0)
    {
      return getDeterministicPartOfModelUpdate(x_k, u_k, d_k) + Sigma*dW;
    }
    else
    {
      return getDeterministicPartOfModelUpdate(x_k, d_k) + Sigma*dW;
    }
  }
}


ssmodels_t::~ssmodels_t() {}
