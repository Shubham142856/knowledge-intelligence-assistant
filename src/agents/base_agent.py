"""
src/agents/base_agent.py — BaseAgent Abstract Class
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all VYOR AI agents.
    Defines prompt structure and execution interface.
    """

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def build_prompt(self, content: str) -> str:
        """Helper to format prompt templates with input content."""
        return f"{self.system_prompt}\n\n---\n\nINPUT:\n{content}\n\nRESPONSE:"

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Execute agent task and return structured output dictionary."""
        pass
