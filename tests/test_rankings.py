import json
from pathlib import Path

from cellresponse.data.source import sha256sum


def test_published_rankings_omit_outcomes():
    path = Path("reports/test_rankings.json")
    text = path.read_text()
    assert "observed" not in text
    document = json.loads(text)
    assert document["selected_model"] == "shared_shift"
    assert sha256sum(path) == Path("reports/test_rankings.sha256").read_text().strip()
    first = document["sets"]["test"][0]["ranking"][0]
    assert set(first) == {"compound_key", "predicted_utility"}
