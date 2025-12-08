from dataclasses import asdict, dataclass, field


@dataclass
class system_prompt:
    """Represents the different parts typically found in a system prompt."""

    role: str | None = field(default_factory=str)
    task: str | None = field(default_factory=str)
    rules: str | None = field(default_factory=str)
    decisions: str | None = field(default_factory=str)

    @property
    def full_prompt(self) -> str:
        """Concatenation of the parts of the prompt."""
        return "\n".join(
            value for _, value in asdict(self).items() if type(value) is str
        )
