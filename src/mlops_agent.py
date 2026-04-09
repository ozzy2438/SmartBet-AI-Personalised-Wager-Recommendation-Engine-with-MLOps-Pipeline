"""Wrapper entrypoint for the MLOps agent."""

from smartbet_ai.agent.mlops_agent import MLOpsAgent


if __name__ == "__main__":
    agent = MLOpsAgent()
    agent.chat("What's the current status of our model?")
