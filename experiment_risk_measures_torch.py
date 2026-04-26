from deephedging.run_train_torch import run_experiment

from deephedging.plot_training_torch import plot_training
from deephedging.plot_prototypes_torch import plot_prototypes


def compare_risk_measures():

    risks = [

        "entropic",

        "mean_variance",

        "cvar"

    ]

    results = {}

    for r in risks:

        print("\nRunning risk:", r)

        results[r] = run_experiment(

            override_objective=dict(

                risk_measure=r

            )

        )

        plot_training(

            results[r]

        )

        plot_prototypes(

            results[r]["model"]

        )

    return results