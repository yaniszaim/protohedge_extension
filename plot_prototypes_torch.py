import matplotlib.pyplot as plt


def plot_prototypes(model):

    """
    Visualize learned prototypes.

    model:
        ProtoHedgeAgent
    """

    if not hasattr(model, "prototypes"):
        print("Model does not expose prototypes; nothing to plot.")
        return

    # extract prototype tensor
    prototypes = model.prototypes

    prototypes = prototypes.detach().cpu()

    plt.figure()

    plt.imshow(
        prototypes,
        aspect="auto"
    )

    plt.colorbar()

    plt.title("Learned Prototypes")

    plt.xlabel("feature dimension")

    plt.ylabel("prototype index")

    plt.show()
