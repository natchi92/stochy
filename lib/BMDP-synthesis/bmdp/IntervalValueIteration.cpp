#include "bmdp/IntervalValueIteration.hpp"
#include <algorithm>
#include <limits>
#include <iostream>
#include <boost/tuple/tuple.hpp>

using namespace MDP;

#define F_EQ(a,b) (fabs(a-b) < 1e-6)

MDP::IntervalValueIteration::IntervalValueIteration(const BMDP& bmdp) : bmdp_(bmdp), discount_(0.95)
{
}

MDP::IntervalValueIteration::~IntervalValueIteration()
{
}

void MDP::IntervalValueIteration::setDiscount(double discount)
{
    discount_ = discount;
}

// Sort values in ascending order.  Break ties based on the secondary (other) values
struct SortValsAsc
{
    SortValsAsc(const std::vector<double>& values, const std::vector<double>& others) : values_(values), others_(others) {}

    bool operator()(const State::StateID& s1, const State::StateID& s2) const
    {
        if (F_EQ(values_[s1], values_[s2]))
        {
            if (F_EQ(others_[s1], others_[s2])) // maintain strict-weak ordering
                return s1 < s2;
            return others_[s1] < others_[s2];
        }

        return values_[s1] < values_[s2];
    }

    const std::vector<double>& values_;
    const std::vector<double>& others_;
};

// Sort values in descending order.  Break ties based on the secondary (other) values
struct SortValsDesc
{
    SortValsDesc(const std::vector<double>& values, const std::vector<double>& others) : values_(values), others_(others) {}

    bool operator()(const State::StateID& s1, const State::StateID& s2) const
    {
        if (F_EQ(values_[s1], values_[s2]))
        {
            if (F_EQ(others_[s1], others_[s2])) // maintain strict-weak ordering
                return s1 < s2;
            return others_[s1] > others_[s2];
        }

        return values_[s1] > values_[s2];
    }

    const std::vector<double>& values_;
    const std::vector<double>& others_;
};

void MDP::IntervalValueIteration::computePessimisticPolicy(BMDP::Policy& policy, bool maxPolicy, std::vector<double>& maxVals, std::vector<double>& minVals, int iterationNum ,double eps) const
{
    computePolicy(policy, maxPolicy, iterationNum, eps, maxVals, minVals, true);
}

void MDP::IntervalValueIteration::computeOptimisticPolicy(BMDP::Policy& policy, bool maxPolicy, std::vector<double>& maxVals, std::vector<double>& minVals, int iterationNum, double eps) const
{
    computePolicy(policy, maxPolicy, iterationNum, eps, maxVals, minVals, false);
}


// STOP!  YOU - RIGHT NOW - STOP!  Before you change this code thinking you are all smart and that I screwed it up
// because the implementation is different from the algorithm in the paper, READ EVERYTHING THAT IS WRITTEN BELOW!
// This implementation minimizes cost rather than maximizing a reward.  You can condense that to mean that everything
// must be reversed.  To disambiguate things, I use max and min values rather than upper/lower bound.
/*void MDP::IntervalValueIteration::computePolicy(BMDP::Policy& policy, bool maxPolicy, int iterationNum, double eps, std::vector<double>& maxVals, std::vector<double>& minVals, bool pessimistic) const
{
    // Initializing the interval values for each state - zeroes for everybody!
    //std::vector<double> maxVals(bmdp_.getNumStates(), 0.0); // max expected costs
    //std::vector<double> minVals(bmdp_.getNumStates(), 0.0); // min expected costs
    maxVals.resize(bmdp_.getNumStates(), 0.0);
    minVals.resize(bmdp_.getNumStates(), 0.0);

    for (unsigned int i = 0; i < bmdp_.getNumStates(); i++)
    {
        if (bmdp_.getActionsAtState(i).size() == 0)
            minVals[i] = maxVals[i] = bmdp_.getCost(i);
    }

    // u1 and u2 are variables that indicate the largest difference in values for a bellman update
    double u1 = std::numeric_limits<double>::infinity();
    double u2 = std::numeric_limits<double>::infinity();
    unsigned int iterations = 0;

    MDPTransitions transitions;

    // Iterate until the max change is < eps
    while ((iterationNum == -1 && u1 > eps && u2 > eps) || (iterationNum != -1 && iterations < iterationNum))
    {
        //// max values update ////
        // Since we are minimizing cost, the order minimizing MDP is the one sorted in descending order
        getOrderMaxMDP(transitions, SortValsDesc(maxVals, minVals));
        u1 = maxPolicy ? bellmanUpdateMax(maxVals, transitions) : bellmanUpdate(maxVals, transitions);


        //// min values update ////
        // Since we are minimizing cost, the order maximizing MDP is the one sorted in ascending order
        getOrderMaxMDP(transitions, SortValsAsc(minVals, maxVals));
        u2 = maxPolicy ? bellmanUpdateMax(minVals, transitions) : bellmanUpdate(minVals, transitions);
        
        iterations++;
    }
    

    if (pessimistic)
    {
        // Extract the pessimistic policy by computing a policy over the order minimizing MDP using the max cost values
        // Think of it this way: you are finding the MDP that is maximizing the maximum expected cost.  pretty pessimistic if you ask me
        getOrderMaxMDP(transitions, SortValsDesc(maxVals, minVals));
        if (maxPolicy)
            extractPolicyMax(transitions, maxVals, policy);
        else
            extractPolicy(transitions, maxVals, policy);
    }
    else
    {
        // Extract the optimistic policy by computing a policy over the order maximizing MDP using the min cost values
        // Think of it this way: you are finding the MDP that is minimizing the minimum expected cost.  pretty optimistic if you ask me
        getOrderMaxMDP(transitions, SortValsAsc(minVals, maxVals));

        if (maxPolicy)
            extractPolicyMax(transitions, minVals, policy);
        else
            extractPolicy(transitions, minVals, policy);
    }

    // Debugging stuff
    std::cout << "Interval value function and policy:" << std::endl;
    for (unsigned int i = 0; i < bmdp_.getNumStates(); ++i)
    {
        if (bmdp_.getNumDestinations(i) == 0)
        {
            std::cout << "  State #" << i << " [" << minVals[i] << " " << maxVals[i]
                      << "]  action: TERMINAL" << std::endl;
        }
        else
        {
            std::cout << "  State #" << i << " [" << minVals[i] << " " << maxVals[i]
                      << "]   action: " << policy[i] << " to state " << bmdp_.getMostLikelyDestinationPessimistic(i, policy[i]) << std::endl;
        }
    }
    std::cout << iterations << " total value iterations" << std::endl;
}*/
void MDP::IntervalValueIteration::computePolicy(BMDP::Policy& policy, bool maxPolicy, int iterationNum, double eps, std::vector<double>& maxVals, std::vector<double>& minVals, bool pessimistic) const
{
    // Initializing the interval values for each state - zeroes for everybody!
    //std::vector<double> maxVals(bmdp_.getNumStates(), 0.0); // max expected costs
    //std::vector<double> minVals(bmdp_.getNumStates(), 0.0); // min expected costs
    maxVals.resize(bmdp_.getNumStates(), 0.0);
    minVals.resize(bmdp_.getNumStates(), 0.0);

    for (unsigned int i = 0; i < bmdp_.getNumStates(); i++)
    {
        if (bmdp_.getActionsAtState(i).size() == 0)
            minVals[i] = maxVals[i] = bmdp_.getCost(i);
    }

    // u1 and u2 are variables that indicate the largest difference in values for a bellman update
    double u = std::numeric_limits<double>::infinity();
    unsigned int iterations = 0;

    MDPTransitions transitions;

    // Iterate until the max change is < eps
    while ((iterationNum == -1 && u > eps ) || (iterationNum != -1 && iterations < iterationNum))
    {
        
        if (pessimistic)
        {
            //// min values update ////
            // Since we are maximizing reward, the order minimizing MDP is the one sorted in ascending order
            getOrderMaxMDP(transitions, SortValsAsc(minVals, maxVals));
            u = maxPolicy ? bellmanUpdateMax(minVals, transitions) : bellmanUpdate(minVals, transitions);
        }
        else
        {
            //// max values update ////
            // Since we are maximizing reward, the order maximizing MDP is the one sorted in descending order
            getOrderMaxMDP(transitions, SortValsDesc(maxVals, minVals));
            u = maxPolicy ? bellmanUpdateMax(maxVals, transitions) : bellmanUpdate(maxVals, transitions);    
        }       
        
        iterations++;
    }
    

    if (pessimistic)
    {
        // Extract the pessimistic policy by computing a policy over the order maximizing MDP using the min reward values
        // Think of it this way: you are finding the MDP that is maximizing the minimum expected reward.
        getOrderMaxMDP(transitions, SortValsAsc(minVals, maxVals));

        if (maxPolicy)
            extractPolicyMax(transitions, minVals, policy);
        else
            extractPolicy(transitions, minVals, policy);
    }
    else
    {
        // Extract the optimistic policy by computing a policy over the order minimizing MDP using the max reward values
        // Think of it this way: you are finding the MDP that is maximizing the maximum expected reward.  pretty optimistic if you ask me
        getOrderMaxMDP(transitions, SortValsDesc(maxVals, minVals));
        if (maxPolicy)
            extractPolicyMax(transitions, maxVals, policy);
        else
            extractPolicy(transitions, maxVals, policy);
    }

    /*// Debugging stuff
    std::cout << "Interval value function and policy:" << std::endl;
    for (unsigned int i = 0; i < bmdp_.getNumStates(); ++i)
    {
        if (bmdp_.getNumDestinations(i) == 0)
        {
            std::cout << "  State #" << i << " [" << minVals[i] << " " << maxVals[i]
                      << "]  action: TERMINAL" << std::endl;
        }
        else
        {
            std::cout << "  State #" << i << " [" << minVals[i] << " " << maxVals[i]
                      << "]   action: " << policy[i] << " to state " << bmdp_.getMostLikelyDestinationPessimistic(i, policy[i]) << std::endl;
        }
    }*/
    //std::cout << iterations << " total value iterations" << std::endl;
}

template<class Sorter>
void MDP::IntervalValueIteration::getOrderMaxMDP(MDPTransitions& transitions, const Sorter& sorter) const
{
    transitions.clear();

    // Iterating over all states
    for (unsigned int i = 0; i < bmdp_.getNumStates(); ++i)
    {
        const std::set<Action::ActionID>& actions = bmdp_.getActionsAtState(i);
        // For each action available to the state i:
        for (std::set<Action::ActionID>::iterator a = actions.begin(); a != actions.end(); ++a)
        {
            // Get the list of destinations available given state i and action *a
            std::vector<State::StateID> dest = bmdp_.getDestinations(i, *a);
            double used = 0.0;
            // Summing the lower bound probabilities.  By design, this cannot exceed 1
            for (size_t d = 0; d < dest.size(); ++d)
            {
                const std::pair<double, double>& prob = bmdp_.getProbabilityInterval(i, *a, dest[d]);
                used += prob.first;
            }

            double remaining = 1.0 - used;
            // Sorting dest based on sorter information
            std::sort(dest.begin(), dest.end(), sorter);

            for (size_t j = 0; j < dest.size(); ++j)
            {
                const std::pair<double, double>& prob = bmdp_.getProbabilityInterval(i, *a, dest[j]);
                double min = prob.first;
                assert (min >= 0.0);

                double desired = prob.second;
                if (desired <= (remaining +min) )
                    transitions[Transition(i, *a, dest[j])] = desired;
                else
                    transitions[Transition(i, *a, dest[j])] = min+remaining;

                remaining = std::max(0.0, remaining - (desired - min));
            }
        }
    }
}

// Updating the value function to minimize cost
double MDP::IntervalValueIteration::bellmanUpdate(std::vector<double>& values,
                                                  const MDPTransitions& transitions) const
{
    double maxDiff = 0.0;

    std::vector<double> newValues(values.size(), 0.0);

    for (size_t i = 0; i < bmdp_.getNumStates(); ++i)
    {
        double min = std::numeric_limits<double>::max();
        double cost = bmdp_.getCost(i);

        const std::set<Action::ActionID>& actions = bmdp_.getActionsAtState(i);
        for (std::set<Action::ActionID>::iterator a = actions.begin(); a != actions.end(); ++a)
        {
            cost = bmdp_.getCost(i);

            const std::vector<State::StateID>& dest = bmdp_.getDestinations(i, *a);
            for (size_t j = 0; j < dest.size(); ++j)
            {
                MDPTransitions::const_iterator it = transitions.find(Transition(i, *a, dest[j]));
                cost += (it->second * values[dest[j]]);
            }
            cost *= discount_;

            if (cost < min)
                min = cost;
        }

        if (actions.size() == 0) // Terminal states do not change value
        {
            if (cost < min)
                min = cost;
        }

        newValues[i] = min;
    }

    // Update value function with new vals
    for(size_t i = 0; i < values.size(); ++i)
    {
        double oldval = values[i];
        values[i] = newValues[i];

        // Check for the max diff.
        // TODO: Check if we need to do abs.  Probably, since we are working with upper and lower bounds
        if (fabs(values[i] - oldval) > maxDiff)
            maxDiff = fabs(values[i] - oldval);
    }

    return maxDiff;
}

// Updating the value function to maximize cost
double MDP::IntervalValueIteration::bellmanUpdateMax(std::vector<double>& values,
                                                  const MDPTransitions& transitions) const
{
    double maxDiff = 0.0;

    std::vector<double> newValues(values.size(), 0.0);

    for (size_t i = 0; i < bmdp_.getNumStates(); ++i)
    {
        double max = -std::numeric_limits<double>::max();
        double cost = bmdp_.getCost(i);

        const std::set<Action::ActionID>& actions = bmdp_.getActionsAtState(i);
        for (std::set<Action::ActionID>::iterator a = actions.begin(); a != actions.end(); ++a)
        {
            cost = bmdp_.getCost(i);

            const std::vector<State::StateID>& dest = bmdp_.getDestinations(i, *a);
            for (size_t j = 0; j < dest.size(); ++j)
            {
                MDPTransitions::const_iterator it = transitions.find(Transition(i, *a, dest[j]));
                cost += (it->second * values[dest[j]]);
            }
            cost *= discount_;

            if (cost > max)
                max = cost;
        }

        if (actions.size() == 0) // Terminal states do not change value
        {
            if (cost > max)
                max = cost;
        }

        newValues[i] = max;
    }

    // Update value function with new vals
    for(size_t i = 0; i < values.size(); ++i)
    {
        double oldval = values[i];
        values[i] = newValues[i];

        // Check for the max diff.
        // TODO: Check if we need to do abs.  Probably, since we are working with upper and lower bounds
        if (fabs(values[i] - oldval) > maxDiff)
            maxDiff = fabs(values[i] - oldval);
    }

    return maxDiff;
}

// Extracts a policy that minimizes cost
void MDP::IntervalValueIteration::extractPolicy(const MDPTransitions& transitions, const std::vector<double>& costFn,
                                                BMDP::Policy& policy) const
{
    policy.clear();

    // Need to map each state to an action
    for (unsigned int i = 0; i < bmdp_.getNumStates(); ++i)
    {
        double min = std::numeric_limits<double>::infinity();
        Action::ActionID minAction = std::numeric_limits<Action::ActionID>::max();

        const std::set<Action::ActionID>& actions = bmdp_.getActionsAtState(i);
        for (std::set<Action::ActionID>::iterator a = actions.begin(); a != actions.end(); ++a)
        {
            double cost = bmdp_.getCost(i);

            const std::vector<State::StateID>& dest = bmdp_.getDestinations(i, *a);
            for (size_t j = 0; j < dest.size(); ++j)
            {
                MDPTransitions::const_iterator it = transitions.find(Transition(i, *a, dest[j]));
                assert(it != transitions.end());
                cost += (it->second * costFn[dest[j]]);
            }
            cost *= discount_;

            if (cost < min)
            {
                min = cost;
                minAction = *a;
            }
        }

        policy[i] = minAction;
    }
}

// Extracts a policy that maximises cost
void MDP::IntervalValueIteration::extractPolicyMax(const MDPTransitions& transitions, const std::vector<double>& costFn,
                                                   BMDP::Policy& policy) const
{
    policy.clear();

    // Need to map each state to an action
    for (unsigned int i = 0; i < bmdp_.getNumStates(); ++i)
    {
        double max = -std::numeric_limits<double>::infinity();
        Action::ActionID maxAction = std::numeric_limits<Action::ActionID>::max();

        const std::set<Action::ActionID>& actions = bmdp_.getActionsAtState(i);
        for (std::set<Action::ActionID>::iterator a = actions.begin(); a != actions.end(); ++a)
        {
            double cost = bmdp_.getCost(i);

            const std::vector<State::StateID>& dest = bmdp_.getDestinations(i, *a);
            for (size_t j = 0; j < dest.size(); ++j)
            {
                MDPTransitions::const_iterator it = transitions.find(Transition(i, *a, dest[j]));
                assert(it != transitions.end());
                cost += (it->second * costFn[dest[j]]);
            }
            cost *= discount_;

            if (cost > max)
            {
                max = cost;
                maxAction = *a;
            }
        }

        policy[i] = maxAction;
    }
}
