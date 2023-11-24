import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 5, "transition must have length = 5"

        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0 : int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)

        ### make it vectorised for envinronments
        states, actions, rewards, next_states, dones = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in indexes:
            states.append(np.array(self.to_tensor(self.buffer[i][0]), copy=False))
            actions.append(np.array(self.to_tensor(self.buffer[i][1]), copy=False))
            rewards.append(np.array(self.to_tensor(self.buffer[i][2]), copy=False))
            next_states.append(np.array(self.to_tensor(self.buffer[i][3]), copy=False))
            dones.append(np.array(self.to_tensor(self.buffer[i][4]), copy=False))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def to_tensor(self, input):
        return input.detach().cpu().numpy()