import random
from collections import defaultdict
from negmas.sao import SAONegotiator, SAOResponse, ResponseType


class Group34_Negotiator(SAONegotiator):
    def __init__(self, *args, beta=0.14, n_samples=2500, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta, self.n_samples = beta, n_samples
        self._rng = random.Random()
        self.reset_stats()

    def reset_stats(self):
        self._outcomes, self._my_u = [], {}
        self._received, self._opp_history = 0, []
        self._counts = []
        self._issue_volatility = []
        self._first_vals = None

    def on_preferences_changed(self, changes=None):
        self.reset_stats()
        if not self.ufun or not self.nmi:
            return

        space = self.nmi.outcome_space
        card = getattr(space, "cardinality", 0)
        raw_outcomes = list(
            space.sample(self.n_samples) if card > self.n_samples else space.enumerate()
        )

        for outcome in raw_outcomes:
            utility = float(self.ufun(outcome))
            if utility >= self.ufun.reserved_value:
                self._my_u[outcome] = utility
                self._outcomes.append(outcome)

        self._outcomes.sort(key=lambda o: self._my_u[o], reverse=True)
        self._max_u = self._my_u[self._outcomes[0]] if self._outcomes else 1.0

        issues = list(space.issues)
        self._counts = [defaultdict(int) for _ in issues]
        self._issue_volatility = [0] * len(issues)  # Initialize volatility per issue

    def _get_vals(self, outcome):
        if isinstance(outcome, dict):
            return list(outcome.values())
        return list(outcome) if outcome is not None else None

    def _update_opponent(self, offer):
        vals = self._get_vals(offer)
        if not vals:
            return
        if not self._first_vals:
            self._first_vals = vals

        # Weight Estimation Logic:
        # If the opponent changes a value in an issue, they are "conceding" on it.
        # Issues that never change are likely high-weight (important) for them.
        if self._opp_history:
            prev_vals = self._get_vals(self._opp_history[-1])
            for i, (v_curr, v_prev) in enumerate(zip(vals, prev_vals)):
                if v_curr != v_prev:
                    self._issue_volatility[i] += 1

        self._received += 1
        self._opp_history.append(offer)
        for i, value in enumerate(vals):
            if i < len(self._counts):
                self._counts[i][value] += 1

    def _opp_score(self, outcome):
        vals = self._get_vals(outcome)
        if not vals or not self._received:
            return 0.0

        # Calculate estimated weights based on inverse volatility
        # We use (v + 1) to avoid division by zero.
        inv_vol = [1.0 / (v + 1.0) for v in self._issue_volatility]
        total_inv_vol = sum(inv_vol)

        score = 0.0
        for i, value in enumerate(vals):
            if i < len(self._counts):
                # Normalize weight so sum(weights) = 1.0
                weight = inv_vol[i] / total_inv_vol
                # Frequency of value * Issue Weight
                score += (self._counts[i][value] / self._received) * weight
        return score

    def _target(self, t):
        res = self.ufun.reserved_value
        target = self._max_u - (self._max_u - res) * (t ** (1.0 / self.beta))
        if t > 0.95:
            target -= 0.05 * (self._max_u - res)
        return max(res, target)

    def __call__(self, state) -> SAOResponse:
        t = state.relative_time
        offer = state.current_offer

        if offer:
            self._update_opponent(offer)
            u_offer = self._my_u.get(offer, float(self.ufun(offer)))
            if u_offer >= self._target(t) or (t > 0.98 and u_offer >= self.ufun.reserved_value):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        target = self._target(t)
        candidates = [o for o in self._outcomes if self._my_u[o] >= target]
        if not candidates:
            candidates = self._outcomes[-1:]

        # Now picking from top candidates using the weighted opponent score
        best_bid = max(candidates[:100], key=lambda o: self._opp_score(o))
        return SAOResponse(ResponseType.REJECT_OFFER, best_bid)