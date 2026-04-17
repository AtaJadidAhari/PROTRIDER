import pytest
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent  # PROTRIDER/


@pytest.fixture(autouse=True)
def setup_logger():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


@pytest.fixture
def covariates_path():
    return ROOT / 'sample_data/sample_annotations.tsv'


@pytest.fixture
def protein_intensities_path():
    return str(ROOT / 'sample_data/protrider_sample_dataset.tsv')


@pytest.fixture
def protein_intensities_index_col():
    return 'protein_ID'


@pytest.fixture
def categorical_covariates():
    return ['BATCH_RUN', 'SEX']


@pytest.fixture
def continuous_covariates():
    return ['AGE']


@pytest.fixture
def gene_expression_path():
    return str(ROOT / 'sample_data/drop_demo_counts_all.tsv')


@pytest.fixture
def gene_annotation_path():
    return str(ROOT / 'sample_data/gencode_annotation_trunc.gtf')


@pytest.fixture
def config_path():
    return str(ROOT / 'config.yaml')

@pytest.fixture(scope="session")
def split_reads_path():
    return str(ROOT / 'sample_data/fraser/split_reads.tsv')

@pytest.fixture(scope="session")
def unsplit_reads_path():
    return str(ROOT / 'sample_data/fraser/unsplit_reads.tsv')

