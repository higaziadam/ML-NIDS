"""Regression tests for the generated EDA notebook template."""

from scripts.create_eda_notebook import create_eda_notebook


def test_generated_notebook_uses_the_standardized_label_name() -> None:
    notebook = create_eda_notebook()
    sources = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "combined_processed['label']" in sources
    assert "combined_processed['Label']" not in sources
