from negmas import make_issue
from negmas.sao.negotiators import BoulwareTBNegotiator as Boulware
from negmas.sao.negotiators import LinearTBNegotiator as Linear
from negmas.sao.negotiators import NaiveTitForTatNegotiator
from negmas.tournaments.neg import cartesian_tournament
from negmas.preferences import LinearAdditiveUtilityFunction as UFun
from negmas.preferences.value_fun import IdentityFun, AffineFun
from negmas.outcomes.outcome_space import make_os
from negmas.inout import Scenario
from pathlib import Path
from my_negotiator import MyNegotiator

if __name__ == '__main__':
    issues = [make_issue(name="price", values=(0, 50))]
    os = make_os(issues=issues)
    path = Path.home()

    seller_utility = UFun(
        values=[IdentityFun()],   # utility = price
        outcome_space=os,
        reserved_value=0.0
    )

    buyer_utility = UFun(
        values=[AffineFun(slope=-1, bias=50)],  # utility = 50 - price
        outcome_space=os,
        reserved_value=0.0
    )

    seller_utility = seller_utility.normalize()
    buyer_utility = buyer_utility.normalize()

    scenario = Scenario(
        outcome_space=os,
        ufuns=[seller_utility, buyer_utility]
    )

    competitors = [Boulware, Linear, MyNegotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=[scenario],
        n_repetitions=2,
        n_steps=50,
        path=path,
        time_limit=2
    )

    print(results)