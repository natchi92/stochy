#ifndef MARKOV_PROCESS_HPP
#define MARKOV_PROCESS_HPP

#include <boost/tuple/tuple.hpp>
#include <boost/functional/hash.hpp>
#include "bmdp/State.hpp"
#include "bmdp/Action.hpp"

namespace MDP
{
    /// A tuple that defines a transition.  A transition in a Markov process is a state-action-result triple
    typedef boost::tuple<State::StateID, Action::ActionID, State::StateID> Transition;
    /// A pair that defines a possible result.  A result is the lhs of a predicate that defines possible destination states
    typedef std::pair<State::StateID, Action::ActionID> Result;

    /// This structure hashes a transition, composed of a state-action pair and a resulting state
    struct TransitionHash
    {
        // Hash function
        size_t operator()(const Transition& t) const
        {
            size_t hash = 0;
            boost::hash_combine(hash, t.get<0>());
            boost::hash_combine(hash, t.get<1>());
            boost::hash_combine(hash, t.get<2>());
            return hash;
        }

        // Equality comparator for transitions
        bool operator()(const Transition& lhs, const Transition& rhs) const
        {
            return lhs.get<0>() == rhs.get<0>() &&
                   lhs.get<1>() == rhs.get<1>() &&
                   lhs.get<2>() == rhs.get<2>();
        }
    };

    /// This structure hashes a Result, composed of a state-action pair
    struct ResultHash
    {
        // Hash function
        size_t operator()(const Result& t) const
        {
            size_t hash = 0;
            boost::hash_combine(hash, t.first);
            boost::hash_combine(hash, t.second);
            return hash;
        }

        // Equality comparator for a Result
        bool operator()(const Result& lhs, const Result& rhs) const
        {
            return lhs.first == rhs.first && lhs.second == rhs.second;
        }
    };
}

#endif
