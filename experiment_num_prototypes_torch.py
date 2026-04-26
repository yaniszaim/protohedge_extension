from deephedging.run_train_torch import run_experiment


def prototype_sweep():

    ks = [

        4,

        8,

        16

    ]

    results = {}

    for k in ks:

        print("\nnum prototypes =", k)

        results[k] = run_experiment(

            override_model=dict(

                num_prototypes=k

            )

        )

    return results