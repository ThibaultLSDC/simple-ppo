from collections import deque, namedtuple
import jax.numpy as jnp
import jax.random as rd
import numpy as np
import jax


class Buffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.next_states = []
        self.logprob = []

        self.transition = namedtuple(
            'Transition',
            [
                'states',
                'actions',
                'rewards',
                'done',
                'next_states',
                'logprob',
            ]
        )

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        res = self.transition(
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.done[idx],
            self.next_states[idx],
            self.logprob[idx],
        )
        return res

    def update(self,
               state,
               action,
               reward,
               done,
               next_state,
               logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
        self.next_states.append(next_state)
        self.logprob.append(logprob)
    
    def reset(self):
        self.__init__()
    
    def sample(self, key, length):
        idx = rd.randint(key, (), 0, len(self)-length)
        batch = (
            self.states[idx:idx+length],
            self.actions[idx:idx+length],
            self.rewards[idx:idx+length],
            self.done[idx:idx+length],
            self.next_states[idx:idx+length],
            self.logprob[idx:idx+length],
        )
        batch = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), batch)
        batch = map(lambda x: jnp.stack(x, axis=0), batch)
        return self.transition(*batch)