import torch


def evaluate_model(
    gym,
    data
):

    gym.agent.eval()

    with torch.no_grad():

        result = gym.forward(data)

    return {

        "mean_pnl": result["pnl"].mean().item(),
        "std_pnl": result["pnl"].std().item(),
        "loss": result["loss"].item()
    }