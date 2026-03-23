from negmas import SAONegotiator, SAOResponse, SAOState, ResponseType

class Group34_Negotiator_Hybrid(SAONegotiator):
    def __init__(self, *args, beta=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self._outcomes = []
        self._reserved_value = 0.0
        self._best_seen = 0.0

    def on_preferences_changed(self, changes=None):
        if self.ufun is None or self.nmi is None:
            return

        self._reserved_value = float(getattr(self.ufun, "reserved_value", 0.0))
        self._best_seen = self._reserved_value

        try:
            outcomes = (
                list(self.nmi.outcome_space.sample(2000))
                if self.nmi.outcome_space.cardinality > 2000
                else list(self.nmi.outcome_space.enumerate())
            )

            self._outcomes = sorted(
                [o for o in outcomes if float(self.ufun(o)) >= self._reserved_value],
                key=self.ufun,
                reverse=True
            )
        except Exception:
            self._outcomes = []

    def _target(self, t):
        if not self._outcomes:
            return self._reserved_value

        max_u = float(self.ufun(self._outcomes[0]))
        span = max_u - self._reserved_value

        # 🔥 Boulware base
        target = max_u - span * (t ** (1 / self.beta))

        # 🔥 KEY IMPROVEMENT: controlled drop
        if t > 0.88:
            target -= 0.05 * span  # small concession jump

        return max(self._reserved_value, target)

    def __call__(self, state: SAOState) -> SAOResponse:
        t = state.relative_time
        target = self._target(t)

        if state.current_offer is not None:
            u = float(self.ufun(state.current_offer))

            if u > self._best_seen:
                self._best_seen = u

            # accept if good
            if u >= target:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

            # 🔥 take best late
            if t > 0.92 and u >= self._best_seen:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

            # safety
            if t > 0.97 and u >= self._reserved_value:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        if not self._outcomes:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        for o in self._outcomes:
            if float(self.ufun(o)) <= target:
                return SAOResponse(ResponseType.REJECT_OFFER, o)

        return SAOResponse(ResponseType.REJECT_OFFER, self._outcomes[-1])