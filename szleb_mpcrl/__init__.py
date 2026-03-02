# szleb_mpcrl/__init__.py

"""
Keep imports lazy so QDLDL and MPC backends are loaded only when needed.
This prevents import-time crashes while debugging solvers.
"""

__all__ = ["MPCRLAgent", "make_szleb_env"]

def __getattr__(name: str):
    if name == "MPCRLAgent":
        from .agent import MPCRLAgent
        return MPCRLAgent
    if name == "make_szleb_env":
        from .envs import make_szleb_env
        return make_szleb_env
    raise AttributeError(name)