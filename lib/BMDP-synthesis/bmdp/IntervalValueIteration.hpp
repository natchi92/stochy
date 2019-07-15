#ifndef MDP_IVI_HPP
#define MDP_IVI_HPP

#include "MarkovProcess.hpp"
#include "BMDP.hpp"
#include <boost/unordered_map.hpp>

#include <map>
#include <vector>

namespace MDP
{
    // Interval value iteration that minimizes an interval valued cost-to-go function
    class IntervalValueIteration
    {
    public:
        IntervalValueIteration(const BMDP& bmdp);
        ~IntervalValueIteration();

        void setDiscount(double discount);

        // Compute a policy that maximizes expected cost
        // In a game-theoretic context, this is an adversarial scenario where the MDP
        // chosen is the worst possible
        void computePessimisticPolicy(BMDP::Policy& policy, bool maxPolicy, std::vector<double>& maxVals, std::vector<double>& minVals, int iterationNum=-1, double eps=1e-4) const;

        // Compute a policy that minimizes expected cost
        // In a game-theoretic context, this is a cooperative scenario where the MDP
        // chosen is the friendliest possible
        void computeOptimisticPolicy(BMDP::Policy& policy, bool maxPolicy, std::vector<double>& maxVals, std::vector<double>& minVals, int iterationNum=-1, double eps=1e-4) const;

    protected:
        // Compute a policy (either optimistic or pessimistic)
        void computePolicy(BMDP::Policy& policy, bool maxPolicy, int iterationNum, double eps, std::vector<double>& maxVals, std::vector<double>& minVals, bool pessimistic) const;


        typedef boost::unordered_map<Transition, double, TransitionHash, TransitionHash> MDPTransitions;

        // Returns the transition probabilities for the order-maximizing MDP representative in the BMDP defined by the sorter class
        template<class Sorter>
        void getOrderMaxMDP(MDPTransitions& transitions, const Sorter& sorter) const;

        void extractPolicy(const MDPTransitions& transitions, const std::vector<double>& costFn, BMDP::Policy& policy) const;

        double bellmanUpdate(std::vector<double>& values, const MDPTransitions& transitions) const;

        double bellmanUpdateMax(std::vector<double>& values, const MDPTransitions& transitions) const;

        void extractPolicyMax(const MDPTransitions& transitions, const std::vector<double>& costFn,
                              BMDP::Policy& policy) const;

        const BMDP& bmdp_;
        double discount_;
    };
}

#endif
