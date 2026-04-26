def get_prototypes(agent):

    return (
        agent.model
        .policy
        .prototype_layer
        .prototypes
        .detach()
    )