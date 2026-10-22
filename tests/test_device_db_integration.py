# Integration test, which connects to the database and checks for some common devices

from torchruntime.device_db import get_device_infos, GPU, DEVICE_DB_FILE


def test_db_file_exists():
    import os
    from torchruntime import device_db

    db_path = os.path.join(os.path.dirname(device_db.__file__), DEVICE_DB_FILE)
    assert os.path.exists(db_path)


def test_get_single_device():
    """Test retrieving a single device."""
    result = get_device_infos([("8086", "56a7")])
    assert len(result) == 1
    assert result[0] == GPU("8086", "Intel Corporation", "56a7", "DG2 [Arc Xe Graphics]", True)


def test_get_multiple_devices():
    """Test retrieving multiple devices."""
    input_ids = [("10de", "2786"), ("10de", "2504")]
    result = get_device_infos(input_ids)
    assert len(result) == 2
    assert GPU("10de", "NVIDIA Corporation", "2786", "AD104 [GeForce RTX 4070]", True) in result
    assert GPU("10de", "NVIDIA Corporation", "2504", "GA106 [GeForce RTX 3060 Lite Hash Rate]", True) in result


def test_get_amd_discrete_devices():
    """Test retrieving AMD discrete devices."""
    input_ids = [("1002", "9495"), ("1002", "747e")]
    result = get_device_infos(input_ids)
    assert len(result) == 2
    assert (
        GPU("1002", "Advanced Micro Devices, Inc. [AMD/ATI]", "9495", "RV730 [Radeon HD 4600 AGP Series]", True)
        in result
    )
    assert (
        GPU("1002", "Advanced Micro Devices, Inc. [AMD/ATI]", "747e", "Navi 32 [Radeon RX 7700 XT / 7800 XT]", True)
        in result
    )


def test_get_amd_integrated_devices():
    """Test retrieving AMD integrated devices."""
    input_ids = [("1002", "164c"), ("1002", "15bf")]
    result = get_device_infos(input_ids)
    assert len(result) == 2
    assert GPU("1002", "Advanced Micro Devices, Inc. [AMD/ATI]", "164c", "Lucienne", False) in result
    assert GPU("1002", "Advanced Micro Devices, Inc. [AMD/ATI]", "15bf", "Phoenix1 [Radeon 740M / 760M / 780M Graphics]", False) in result


def test_get_intel_integrated_devices():
    """Test retrieving Intel integrated devices."""
    input_ids = [("8086", "46b6"), ("8086", "4e55")]
    result = get_device_infos(input_ids)
    assert len(result) == 2
    assert GPU("8086", "Intel Corporation", "46b6", "AlderLake-P [Iris Xe Graphics]", False) in result
    assert GPU("8086", "Intel Corporation", "4e55", "JasperLake [UHD Graphics]", False) in result


def test_get_nonexistent_device():
    """Test retrieving a device that doesn't exist in the database."""
    result = get_device_infos([("ffff", "ffff")])
    assert len(result) == 0


def test_get_mixed_existing_and_nonexistent():
    """Test retrieving a mix of existing and non-existing devices."""
    input_ids = [("8086", "56a7"), ("ffff", "ffff")]  # exists  # doesn't exist
    result = get_device_infos(input_ids)
    assert len(result) == 1
    assert result[0] == GPU("8086", "Intel Corporation", "56a7", "DG2 [Arc Xe Graphics]", True)
