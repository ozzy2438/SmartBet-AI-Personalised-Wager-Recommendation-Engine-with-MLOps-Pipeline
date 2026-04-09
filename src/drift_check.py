"""Wrapper entrypoint for drift detection."""

from smartbet_ai.monitoring.drift import check_drift


if __name__ == "__main__":
    check_drift()
