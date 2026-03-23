from negmas import SAONegotiator, SAOResponse, SAOState, ResponseType
import random

class Group34_Negotiator(SAONegotiator):
    def __init__(self, *args, beta=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self._outcomes = []
        self._reserved_value = 0.0
        self._best_seen = 0.0
        self._rng = random.Random()

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

            # Keep only rational outcomes
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

        # 🔥 Boulware curve
        target = max_u - span * (t ** (1 / self.beta))

        # 🔥 Small controlled concession near the end
        if t > 0.88:
            target -= 0.05 * span

        return max(self._reserved_value, target)

    def __call__(self, state: SAOState) -> SAOResponse:
        t = state.relative_time
        target = self._target(t)

        if state.current_offer is not None:
            offer_u = float(self.ufun(state.current_offer))

            # Track best seen offer
            if offer_u > self._best_seen:
                self._best_seen = offer_u

            # Standard accept
            if offer_u >= target:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

            # Smart endgame accept (beats Boulware)
            if t > 0.92 and offer_u >= self._best_seen:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

            # Safety (avoid 0 score)
            if t > 0.97 and offer_u >= self._reserved_value:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        if not self._outcomes:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # 🔥 FLEXIBLE OFFER SELECTION (key improvement)
        candidates = [
            o for o in self._outcomes
            if float(self.ufun(o)) <= target
        ]

        if not candidates:
            chosen = self._outcomes[-1]
        else:
            # pick among top few instead of always the same
            top_k = candidates[:min(5, len(candidates))]
            chosen = self._rng.choice(top_k)

        return SAOResponse(ResponseType.REJECT_OFFER, chosen)