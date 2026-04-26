from deephedging.run_train_torch import run_experiment

from deephedging.plot_prototypes_torch import plot_prototypes


def volatility_sweep():

    vols = [

        0.1,

        0.2,

        0.3,

        0.5

    ]

    results = {}

    for v in vols:

        print("\nvol =", v)

        results[v] = run_experiment(

            override_world=dict(

                volatility=v

            )

        )

        plot_prototypes(

            results[v]["model"]

        )

    return results