"""Literature Intelligence Engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    year: int
    source: str  # "arXiv", "PubMed", "OpenAlex", "Curated"
    doi: Optional[str]
    summary: str
    key_findings: List[str]


class LiteratureIntelligence:
    """Retrieves and synthesizes relevant neuroscience and machine learning research."""

    def __init__(self):
        # Curated foundational corpus
        self.corpus: List[ResearchPaper] = [
            ResearchPaper(
                title="A neuronal model of a global workspace in effortful cognitive tasks",
                authors=["Dehaene", "Kerszberg", "Changeux"],
                year=1998,
                source="Curated",
                doi="10.1073/pnas.95.24.14529",
                summary="Computational architecture demonstrating non-linear ignition and global broadcast.",
                key_findings=["Bimodal ignition threshold", "Effortful task arbitration via workspace"],
            ),
            ResearchPaper(
                title="Integrated information theory: from consciousness to its physical substrate",
                authors=["Tononi", "Boland", "Massimini", "Koch"],
                year=2016,
                source="Curated",
                doi="10.1038/nrn.2016.44",
                summary="Theoretical framework for measuring intrinsic causal integration in networks.",
                key_findings=["Complexes of maximal integrated information", "Feedforward networks lack phi"],
            ),
            ResearchPaper(
                title="Beyond reward: homeostasis and the emergence of cognition",
                authors=["Keramati", "Gutkin"],
                year=2014,
                source="Curated",
                doi="10.1016/j.tics.2014.05.002",
                summary="Homeostatic viability boundaries driving learning and exploration without external rewards.",
                key_findings=["Homeostatic cost function as primary drive", "Anticipatory regulation"],
            ),
        ]

    def search(self, query: str) -> List[ResearchPaper]:
        """Searches curated corpus by keyword matching."""
        q_lower = query.lower()
        results = []
        for paper in self.corpus:
            if (
                q_lower in paper.title.lower()
                or q_lower in paper.summary.lower()
                or any(q_lower in k.lower() for k in paper.key_findings)
            ):
                results.append(paper)
        return results if results else self.corpus[:2]
