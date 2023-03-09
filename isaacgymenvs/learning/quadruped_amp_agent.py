import isaacgymenvs.learning.amp_continuous as amp_continuous

class QuadrupedAMPAgent(amp_continuous.AMPAgent):

    def play_eval_steps(self):
        """ Play steps to record episode statistics """
        self.set_eval()
        # TODO: fill this in 

    def eval_epoch(self):
        # TODO: fill this in
        pass 

    def train_epoch(self):
        train_info = super().train_epoch(self)
        eval_info = self.eval_epoch(self)
        return train_info