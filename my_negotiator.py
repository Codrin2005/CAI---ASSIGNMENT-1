from negmas import SAONegotiator, ResponseType

class MyNegotiator(SAONegotiator):
    def __init__(self, ufun, name=None, **kwargs):
        if name is None:
            name = "DummyAgent"
        super().__init__(ufun=ufun, name=name, **kwargs)

    def propose(self, state):
        # always propose the best outcome for its own utility
        return max(self.ufun.outcome_space.outcomes, key=self.ufun)

    def respond(self, state, offer):
        if offer is None:
            return ResponseType.REJECT
        return ResponseType.ACCEPT  # accept everything