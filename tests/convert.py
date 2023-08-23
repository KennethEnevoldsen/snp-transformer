"""
"""

from utils import bed_data_path

from snp_transformer.convert import bed_to_arrow


def test_bed_to_arrow(bed_data_path):
    arrow_path = bed_data_path.stem  # datasets saves as a folder
    bed_to_arrow(bed_data_path, arrow_path)
    assert arrow_path.is_dir()
    assert (
        len(list(arrow_path.glob("*.arrow"))) >= 1
    ), "There should be at least one arrow file in the folder"
