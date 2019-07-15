#ifndef MDP_STATE_HPP
#define MDP_STATE_HPP

/// Definition of a state in a Markov decision process
/// Requires a unique ID
namespace MDP
{
    class State
    {
        public:
            typedef unsigned int StateID;

            State(StateID id) : id_(id) {}
            virtual ~State() {}

            virtual StateID getID() const { return id_; }

        protected:
            StateID id_;
    };
}

#endif
