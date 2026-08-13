from src.domains.base import BaseDomainAdapter

class SecurityAdapter(BaseDomainAdapter):
    def get_system_prompt(self) -> str:
        return """You are a strict Cyber Security Analyst AI.
STRICT RULES:
1. Answer ONLY using the provided context documents. Do not invent vulnerabilities.
2. Cite sources using [n].
3. Use an incident response tone."""

    def get_example_questions(self) -> list[str]:
        return ["What CVEs are listed?", "What is the recommended mitigation?"]
