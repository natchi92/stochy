//
//

#include <iostream>

#include <case_studies.h>

static int parse_and_go(int argc, char **argv)
{
  // Input case study number
  // Case study 1 : Verification
  // Case study 2 : Synthesis
  // Case study 3 : Scaling in dimensions
  // Case study 4 : Simulation
  if (argc < 2) 
  {
    std::cout 
      << "No case study selection was given, please make use of ./stochy i, "
         "where i=[1,2,3,4] is the required case study number"
      << std::endl;
    return -1;
  }

  int selection = strtol(argv[1], NULL, 0);

  int res;

  switch (selection) 
  {
    case 1: 
      std::cout << "Performing Case Study 1 : Formal Verification of CO2 model"
                << std::endl << std::endl;
      res = case_study1("CS1.mat");
      break;
    case 2:
      std::cout << "Performing Case Study 2 : Strategy synthesis"
                << std::endl << std::endl;
      res = case_study2();
      std::cout << "Completed Case Study 2 : Results in results folder"
                << std::endl << std::endl;
      break;
    case 3:
      std::cout << "Performing Case Study 3 : Scaling in number of dimensions"
                << std::endl << std::endl;
      res = case_study3();
      std::cout << "Completed Case Study 3 : Results in results folder"
                << std::endl << std::endl;
      break;
    case 4:
      std::cout << "Performing Case Study 4 : Simulation of CO2 model"
                << std::endl << std::endl;
      res = case_study4("CS4.mat", "u.txt");
      std::cout << "Completed Case Study 4 : Results in results folder"
                << std::endl << std::endl;
      break;
    default: 
      std::cout << "Invalid case study selection" << std::endl;
      res = -1;
      break;
  }

  return res;
}

int main(int argc, char **argv) 
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

  return parse_and_go(argc, argv);
}
