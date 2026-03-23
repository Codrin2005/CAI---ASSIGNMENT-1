from negmas import SAONegotiator, SAOResponse, SAOState, ResponseType
import bisect
import random

class Group34_Negotiator_Hybrid(SAONegotiator):
    def __init__(self, *args, exponent: float = 1.15, opponent_bias: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.exponent = exponent
        self._rng = random.Random()
        self.opponent_bias = opponent_bias  # small bias towards opponent frequent values
        self._opponent_freq = {}
        self._prepared_outcomes = []
        self._reserved_value = 0.0
        self._best_received_utility = 0.0
        self._offered_signatures = set()

    def on_preferences_changed(self, changes=None):
        if self.ufun is None or self.nmi is None:
            return
        self._reserved_value = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        # Prepare outcomes above reserved
        try:
            if self.nmi.outcome_space.cardinality > 20000:
                outcomes = list(self.nmi.outcome_space.sample(20000))
            else:
                outcomes = list(self.nmi.outcome_space.enumerate())
            self._prepared_outcomes = [o for o in outcomes if float(self.ufun(o)) >= self._reserved_value]
            self._prepared_outcomes.sort(key=self.ufun, reverse=True)
        except Exception:
            self._prepared_outcomes = []
        self._best_received_utility = self._reserved_value
        self._opponent_freq = {}
        self._offered_signatures = set()

    def _update_opponent_model(self, offer):
        if offer is None:
            return
        for i, v in (offer.items() if isinstance(offer, dict) else enumerate(offer)):
            i_str = str(i)
            self._opponent_freq.setdefault(i_str, {})
            self._opponent_freq[i_str][v] = self._opponent_freq[i_str].get(v, 0) + 1

    def _target_utility(self, t: float) -> float:
        if not self._prepared_outcomes:
            return self._reserved_value
        max_u = float(self.ufun(self._prepared_outcomes[0]))
        # Exponential concession curve
        target = self._reserved_value + (max_u - self._reserved_value) * (1 - t ** self.exponent)
        # Optional: phase floor to avoid early over-concession
        phase_floor = max(self._reserved_value, self._best_received_utility - 0.03)
        return max(target, phase_floor)

    def _candidate_outcomes(self, target):
        if not self._prepared_outcomes:
            return []
        utilities = [float(self.ufun(o)) for o in self._prepared_outcomes]
        idx = bisect.bisect_left(utilities, target)
        # Take a small window of outcomes near the target
        lower = max(0, idx - 20)
        upper = min(len(self._prepared_outcomes), idx + 20)
        return self._prepared_outcomes[lower:upper]

    def _score_outcome(self, outcome):
        score = float(self.ufun(outcome))
        # Slightly favor opponent frequent values
        for i, v in (outcome.items() if isinstance(outcome, dict) else enumerate(outcome)):
            i_str = str(i)
            freq = self._opponent_freq.get(i_str, {}).get(v, 0)
            score += self.opponent_bias * freq
        return score

    def __call__(self, state: SAOState) -> SAOResponse:
        t = state.relative_time
        # Update opponent model
        self._update_opponent_model(state.current_offer)
        if state.current_offer is not None:
            offer_util = float(self.ufun(state.current_offer))
            target = self._target_utility(t)
            if offer_util >= target:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
            self._best_received_utility = max(self._best_received_utility, offer_util)

        # Pick next offer
        candidates = self._candidate_outcomes(self._target_utility(t))
        if not candidates:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        # Rank candidates by own utility + opponent bias
        candidates.sort(key=self._score_outcome, reverse=True)
        for c in candidates:
            if self._signature(c) not in self._offered_signatures:
                chosen = c
                break
        else:
            chosen = candidates[0]
        self._offered_signatures.add(self._signature(chosen))
        return SAOResponse(ResponseType.REJECT_OFFER, chosen)

    @staticmethod
    def _signature(outcome):
        if outcome is None:
            return None
        if isinstance(outcome, dict):
            return tuple(sorted(outcome.items()))
        return tuple(outcome)