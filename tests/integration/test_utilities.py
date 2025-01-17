"""Integration tests for utilities.

These tests check the utilities that PROCESS uses, mainly for file IO. They run
on each of the regression test scenarios.
"""

import logging

import pytest

import process.io.in_dat as indat

logger = logging.getLogger(__name__)

logger.info("Running utilities integration tests")

# TODO More utilities tests to be implemented
# test_convert_in_dat


@pytest.fixture
def mfile_path(temp_data, mfile_name):
    """Create a path to a scenario's MFile.

    :param temp_data: temporary path containing data files
    :type temp_data: Path
    :param mfile_name: name of the reference MFile
    :type mfile_name: str
    :return: Path to that scenario's MFile
    :rtype: Path
    """
    mfile_path = temp_data / mfile_name
    return mfile_path


@pytest.fixture
def input_file_path(temp_data):
    """Create a path to a scenario's input file.

    :param temp_data: temporary path containing data files
    :type temp_data: Path
    :return: Path to that scenario's IN.DAT
    :rtype: Path
    """
    return temp_data / "large_tokamak_IN.DAT"


def test_in_dat_lib(input_file_path):
    """Test the PROCESS in_dat library.

    :param input_file_path: Path to a scenario's input file
    :type input_file_path: Path
    """
    logger.info("Testing in_dat")

    # Test MFile for this scenario
    try:
        assert indat.test(str(input_file_path)) is True
    except AssertionError:
        logger.error(f"in_dat test for {input_file_path.name} has failed")
        raise
