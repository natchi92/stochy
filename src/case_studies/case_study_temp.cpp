///
///

#include "case_studies.h"

#include <taskExec.h>
#include <SHS.h>


enum BenchMark
{
  BAS_4D = 0,
  BAS_7D = 1,
  ANESTHESIA = 2,
  INTEGRATOR=3,
};

int case_study_arch(int argc, char **argv) 
{
    std::cout << " _______  _______  _______  _______  __   __  __   __ "
              << std::endl;
    std::cout << "|       ||       ||       ||       ||  | |  ||  | |  |"
              << std::endl;
    std::cout << "|  _____||_     _||   _   ||       ||  |_|  ||  |_|  |"
              << std::endl;
    std::cout << "| |_____   |   |  |  | |  ||       ||       ||       |"
              << std::endl;
    std::cout << "|_____  |  |   |  |  |_|  ||      _||       ||_     _|"
              << std::endl;
    std::cout << " _____| |  |   |  |       ||     |_ |   _   |  |   |  "
              << std::endl;
    std::cout << "|_______|  |___|  |_______||_______||__| |__|  |___| "
              << std::endl;
    std::cout << std::endl;
    std::cout << " Welcome!  Copyright (C) 2018  natchi92 " << std::endl;
    std::cout << std::endl;

    if (argc < 2) 
    {
      std::cout
          << "No case study selection was given, please make use of ./stochy i, "
             "where i=[0,1,2,3] is the required case study number"
          << std::endl;
      exit(0);
    }
    int selection = strtol(argv[1], NULL, 0);

    switch (selection) 
    {
      case BAS_4D: 
      {
        // ------------------------- Case study 1 - Verification
        // -----------------------------------
        std::cout << "------------ Performing BAS model  :  CS1BASssa"
                     "of ARCH -----------"
                  << std::endl;
        std::cout << std::endl;
        // --------------------------------------------

        //arma::mat Aq0 = { {0.6682, 0, 0.02632, 0},{0,0.6830,0,0.02096},{1.0005,0,-0.000499,0}, {0,  0.8004,0,0.1996}};
        //arma::vec Bq0 = {{0.1320},{0.1402},{0},{0}};
        //arma::mat Gq0 = { {0.0774, 0, 0,0}, {0 , 0.0774,0,0}, {0,0,0.3872,0}, {0,0,0,0.3098}};
        //arma::vec Qq0 = {{3.3378},{2.9272}, {13.0207}, {10.4166}};
        //ssmodels_t model(Aq0,Bq0,Qq0,Gq0);
        double theta = 0.683;
        arma::mat Aq0 = { {0.6682, 0},{0,theta}};
        arma::vec Bq0 = {{0.1320},{0.1402}};
        arma::mat Gq0 = { {0.0774, 0}, {0 , 0.0774*0.1}};
        arma::vec Qq0 =  {{3.653},{3.77}};
        ssmodels_t model(Aq0,Gq0);
        double u = 18.5;
        double x1_ss = (u*Bq0(0) + Qq0(0))/(1 - Aq0(0,0));
        double x2_ss = (u*Bq0(1) + Qq0(1))/(1 - Aq0(1,1));

        std::cout << x1_ss << ", " << x2_ss << std::endl;
        std::vector<ssmodels_t> models1 = {model};
        shs_t<arma::mat,int> cs1SHS(models1);

        // Define max error
        arma::mat grid = {{0.08,0.08}};
        arma::mat reft = {{1,1}};
        // Define safe set
        arma::mat Safe = {{18.5-x1_ss, 20.5-x1_ss}, {18.5-x2_ss, 20.5-x2_ss}};

        arma::mat Input = {{18, 19}};

        // Define grid type
        // (1 = uniform, 2 = adaptive)
        // For comparison with IMDP we use uniform grid
        int gridType =1;

        // Time horizon
        int K = 1;

        // Library (1 = simulator, 2 = faust^2, 3 = imdp)
        int lb = 3;

        // Property type
        // FIX (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
        int p = 1;

        // Task specification
        TaskSpec cs1SpecFAUST(3, K, 1, Safe, grid, reft);

        // Combine model and associated task
        InputSpec<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

        // Perform  Task
        performTask(cs1InputFAUST);


        // Perform  Task
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "------------Completed -----------"
                  << std::endl;


        break;
      }

      case BAS_7D: 
      {
        // ------------------------- Case study 1 - Verification
        // -----------------------------------
        std::cout << "------------ Performing BAS model  :  CS1BASssa"
                     "of ARCH -----------"
                  << std::endl;
        std::cout << std::endl;
        // --------------------------------------------

        double theta = 0.683;
        arma::mat Aq0 = {{ 0.9678 ,     0,    0.0036,        0,    0.0036,         0,    0.0036},
                        {0,    0.9682,         0,   0.0034,         0,    0.0034,    0.0034},
                    {0.0106,         0,    0.9494,         0,         0,         0,         0},
                  {       0,    0.0097,         0,    0.9523,         0,         0,         0},
                  {  0.0106,         0,         0,         0,    0.9494,         0,         0},
                  {       0,    0.0097,         0,         0,         0,    0.9523,         0},
                  {  0.0106,    0.0097,         0,         0,         0,         0,    0.9794}};
        arma::vec Bq0 = {{0.0195},
                         {0.0200},
                         {0.0459},
                         {0.0425},
                         {0.0397},
                         {0.0377},
                         {0.0109}};
        arma::mat Gq0 =arma::diagmat(Bq0);
        /* { {      0,         0   , 0.0000,         0   , 0.0019 ,        0,    0.0164},
    {      0,         0  ,       0,    0.0000   ,      0 ,   0.0015  , -0.0018},
    { 0.0459 ,        0 ,        0 ,        0   ,      0  ,       0  ,  0.0387},
    { 0.0425 ,        0 ,        0 ,        0   ,      0  ,       0 ,   0.0189},
    {     0  ,  0.0397  ,       0  ,       0   ,      0   ,      0 ,   0.0110},
    {     0  ,  0.0377  ,       0  ,       0  ,       0   ,      0 ,   0.0108},
    {     0  ,       0 ,        0   ,      0 ,        0    ,     0 ,   0.0109
}};*/
        ssmodels_t model(Aq0,Gq0);
        std::cout << "Gq0" << Gq0 <<std::endl;

        std::vector<ssmodels_t> models1 = {model};
        shs_t<arma::mat,int> cs1SHS(models1);

        // Define max error
        arma::mat grid = {{0.05,0.05,0.5, 0.5,0.5,0.5,0.5}};//, 0.5,0.5}};//,0.5,0.5,0.5}};
        arma::mat reft = {{1,1,1,1,1,1,1}};//,1,1,1}};
        // Define safe set
        arma::mat Safe = {{-0.5, 0.5}, {-0.5, 0.5}, {-0.5, 0.5},
         {-0.5,0.5}, {-0.5, 0.5},
         {-0.5, 0.5}, {-0.5,0.5}};
          //{-0.5,0.5}};

        //  arma::mat Input = {{18, 19}};

        // Define grid type
        // (1 = uniform, 2 = adaptive)
        // For comparison with IMDP we use uniform grid
        int gridType =1;

        // Time horizon
        int K = 6;

        // Library (1 = simulator, 2 = faust^2, 3 = imdp)
        int lb = 3;

        // Property type
        // FIX (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
        int p = 1;

        // Task specification
        TaskSpec cs1SpecFAUST(3, K, 1, Safe, grid, reft);

        // Combine model and associated task
        InputSpec<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

        // Perform  Task
        performTask(cs1InputFAUST);


        // Perform  Task
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "------------Completed -----------"
                  << std::endl;


      break;
      }
      case ANESTHESIA: {
      clock_t begin, end;
      begin = clock();
      double time = 0;

      // ------------------------- Case study 1 - Verification
      // -----------------------------------
      std::cout << "------------ Performing Anaesthesia model  :  Problem 2.1.1 "
                   "of ARCH - 3D case -----------"
                << std::endl;
      std::cout << std::endl;
      // --------------------------------------------
      arma::mat Aq0 = {{0.8192, 0.0341, 0.0126}, {0.0165, 0.9822, 0.0001},{0.0009, 1e-4, 0.9989}};
      arma::mat Gq0 = { {5, 0, 0}, {0 ,5,0 }, {0, 0, 5}};
      arma::vec Qq0 = {{0.0842},{0.0009},{1e-5}};
  //    Gq0 = Gq0 + arma::diagmat(Qq0.t()*7*10);
      ssmodels_t model(Aq0(arma::span(0,1), arma::span(0,1)),Gq0(arma::span(0,1), arma::span(0,1)));

      std::vector<ssmodels_t> models1 = {model};
      shs_t<arma::mat,int> cs1SHS(models1);

      // Define safe set
      arma::mat bound = {{1,6}, {0,10}};//,{0,10}};

      arma::mat grid = {{1,5}};//,1}};
      arma::mat reft =  {{.5,.5,}};//.5}};
      // Define grid type
      // (1 = uniform, 2 = adaptive)
      // For comparison with IMDP we use uniform grid
      int gridType = uniform;

      // Time horizon
      int K = 10;

      // Library (1 = simulator, 2 = faust^2, 3 = imdp)
      int lb = imdp;

      // Property type
      // (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
      int p = verify_safety;

      // Task specification
      TaskSpec cs1SpecFAUST(lb, K, p, bound, grid, reft);

      // Combine model and associated task
      InputSpec<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

          // Perform  Task
      performTask(cs1InputFAUST);
  break;
  }
      case INTEGRATOR:{
        std::cout << "------------ Performing INTEGRATOR benchmark"
                  "of ARCH -----------"
                  << std::endl;
                  std::cout << std::endl;
                  // --------------------------------------------
        clock_t begin, end;
        begin = clock();
        double time = 0;

        double N_s = 0.1;
        int T = 5;

      int dimension = argc >= 3 ? std::atoi(argv[2]): 2;

      arma::mat Aq0 = arma::zeros<arma::mat>(dimension, dimension);
      arma::vec Bq0 = arma::zeros<arma::mat>(dimension);
      arma::mat Gq0 = arma::zeros<arma::mat>(dimension, dimension);

      arma::mat bound = arma::ones<arma::mat>(dimension,dimension);
      arma::mat grid = arma::zeros<arma::mat>(1,dimension);
      arma::mat reft = arma::zeros<arma::mat>(1,dimension);
      switch (dimension)
      {
        case 2:
        {
          Aq0 = {{1, N_s},{0,1}};
          Bq0 = {{N_s*(N_s*N_s)/6 }, {(N_s*N_s)/2}};
          Gq0 = 0.04*arma::eye<arma::mat>(2,2) + arma::diagmat(Bq0);
          bound = {{-1,1},{-1,1}};
          grid = {{0.1,0.1}};//,1}};
          arma::mat reft = {{1,1}};//,1}};

          std::cout << "Aq0: " << Aq0 << std::endl;
          std::cout << "Bq0: " << Bq0 << std::endl;
          std::cout << "Gq0: " << Gq0 << std::endl;
          }
          break;
        case 3:
        {
          Aq0 = {{1, N_s, 0.5*(N_s*N_s)},
                 {0, 1, N_s},
                 {0, 0, 1}};
          Bq0 = {{N_s*(N_s*N_s)/6 }, {(N_s*N_s)/2}, {N_s}};
          Gq0 = 0.04*arma::eye<arma::mat>(3,3)+ arma::diagmat(Bq0);
          bound = {{-1,1},{-1,1},{-1,1}};
          grid = {{0.1,0.1,0.1}};//,1}};
          arma::mat reft = {{1,1,1}};//,1}};
          std::cout << "Aq0: " << Aq0 << std::endl;
          std::cout << "Bq0: " << Bq0 << std::endl;
          std::cout << "Gq0: " << Gq0 << std::endl;
        }
        break;

        case 4:
        {
          Aq0 = {{1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6},
                 {0,1, N_s, 0.5*(N_s*N_s) }, {0, 0, 1, N_s},
                  {0,0,0,1}};
          Bq0 = {{N_s*N_s*(N_s*N_s)/24 },{N_s*(N_s*N_s)/6 }, {(N_s*N_s)/2}, {N_s}};
          Gq0 = 0.04*arma::eye<arma::mat>(dimension, dimension)+ arma::diagmat(Bq0);
          bound = {{-1,1},{-1,1},{-1,1},{-1,1}};
          grid = {{0.1,0.1,0.1,0.1}};//,1}};
          arma::mat reft = {{1,1,1,1}};//,1}};
          std::cout << "Aq0: " << Aq0 << std::endl;
          std::cout << "Bq0: " << Bq0 << std::endl;
          std::cout << "Gq0: " << Gq0 << std::endl;
       }
       break;

        case 5:
        {
        Aq0 = {{1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6, N_s*N_s*(N_s*N_s)/24 },
               {0,1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6 }, {0, 0, 1, N_s, 0.5*(N_s*N_s)},
                {0,0,0,1,N_s}, {0,0,0,0,1}};
        Bq0 = {{N_s*N_s*(N_s*N_s*N_s)/120 },{N_s*N_s*(N_s*N_s)/24 },{N_s*(N_s*N_s)/6 }, {(N_s*N_s)/2}, {N_s}};
        Gq0 = 0.04*arma::eye<arma::mat>(dimension, dimension)+ arma::diagmat(Bq0);
        bound = {{-1,1},{-1,1},{-1,1},{-1,1},{-1,1}};
        grid = {{0.1,0.1,0.1,0.1,0.1}};//,1}};
        arma::mat reft = {{1,1,1,1,1}};//,1}};
        std::cout << "Aq0: " << Aq0 << std::endl;
        std::cout << "Bq0: " << Bq0 << std::endl;
        std::cout << "Gq0: " << Gq0 << std::endl;
        }
        break;

        case 7:
        {
        Aq0 = {{1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6, N_s*N_s*(N_s*N_s)/24, N_s*N_s*(N_s*N_s*N_s)/120 , N_s*N_s*N_s*(N_s*N_s*N_s)/720 },
               {0,1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6, N_s*N_s*(N_s*N_s)/24, N_s*N_s*(N_s*N_s*N_s)/120  },
                {0, 0, 1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6, N_s*N_s*(N_s*N_s)/24},
                {0, 0, 0,  1, N_s, 0.5*(N_s*N_s), N_s*(N_s*N_s)/6},
                {0, 0, 0,  0,   1, N_s, 0.5*(N_s*N_s)},
                {0, 0, 0,  0,   0,  1,  N_s},
                {0, 0, 0, 0, 0, 0, 1}};

        Bq0 = { {N_s*N_s*N_s*(N_s*N_s*N_s*N_s)/5040}, { N_s*N_s*N_s*(N_s*N_s*N_s)/720}, {N_s*N_s*(N_s*N_s*N_s)/120 },{N_s*N_s*(N_s*N_s)/24 },{N_s*(N_s*N_s)/6 }, {(N_s*N_s)/2}, {N_s}};
        Gq0 = 0.04*arma::eye<arma::mat>(dimension, dimension)+ arma::diagmat(Bq0);
        bound = {{-1,1},{-1,1},{-1,1},{-1,1},{-1,1},{-1,1},{-1,1}};
        grid = {{0.1,0.1,0.1,0.1,0.1,0.1,0.1}};//,1}};
        arma::mat reft = {{1,1,1,1,1,1,1}};//,1}};
        std::cout << "Aq0: " << Aq0 << std::endl;
        std::cout << "Bq0: " << Bq0 << std::endl;
        std::cout << "Gq0: " << Gq0 << std::endl;
        }
        break;

        default:
           std::cout << "TODO" << std::endl;
        break;
      };
    

  //FIXED POLICY TO BE ALWAYS ON FOR ALL TIME HORIZON
    ssmodels_t model(Aq0,Gq0);

    std::vector<ssmodels_t> models1 = {model};
    shs_t<arma::mat,int> cs1SHS(models1);



    // Define grid type
    // (1 = uniform, 2 = adaptive)
    // For comparison with IMDP we use uniform grid
    int gridType = 1;

    // Time horizon
    int K = 6;

    // Library (1 = simulator, 2 = faust^2, 3 = imdp)
    int lb = 2;

    // Property type
    // (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
    int p = 1;

    // Task specification
    TaskSpec cs1SpecFAUST(3, K, 1, bound, grid, reft);

    // Combine model and associated task
    InputSpec<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

        // Perform  Task
    performTask(cs1InputFAUST);



    break;
  }

      default: 
      {
        std::cout << " Invalid case study selection " << std::endl;
        break;
      }
    }

    // Perform  Task
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "------------Completed -----------"
                << std::endl;
}
