"""Contains tests for the voraus-AD module."""

from pathlib import Path

import numpy
import pandas
import torch
from torch.utils.data import DataLoader, Dataset

import voraus_ad

DATASET_PATH = Path(__file__).parent / "test.parquet"


def test_load_pandas_dataframes() -> None:
    dfs_train, labels_train, dfs_test, labels_test = voraus_ad.load_pandas_dataframes(
        path=DATASET_PATH,
        columns=voraus_ad.Signals.machine(),
        normalize=True,
        frequency_divider=1,
        train_gain=1.0,
        pad=True,
    )

    assert len(dfs_train) == len(labels_train) == 2
    assert len(dfs_test) == len(labels_test) == 2

    assert dfs_train[0].shape == (100, 130)
    assert dfs_train[1].shape == (100, 130)
    assert dfs_test[0].shape == (100, 130)
    assert dfs_test[1].shape == (100, 130)

    for dataframe in dfs_train + dfs_test:
        assert isinstance(dataframe, pandas.DataFrame)

    assert dfs_train[0].columns.to_list() == list(voraus_ad.Signals.machine())

    assert labels_train[0] == {"anomaly": False, "category": "NORMAL_OPERATION", "sample": 755, "variant": "PRE_A"}
    assert labels_train[1] == {"sample": 1702, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_A"}
    assert labels_test[0] == {"sample": 12, "anomaly": True, "category": "AXIS_FRICTION", "variant": "A1_B"}
    assert labels_test[1] == {"sample": 1703, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_B"}


def test_load_numpy_arrays() -> None:
    arrays_train, labels_train, arrays_test, labels_test = voraus_ad.load_numpy_arrays(
        path=DATASET_PATH,
        columns=list(voraus_ad.Signals.machine()),
        normalize=False,
        frequency_divider=1,
        train_gain=1.0,
        pad=False,
    )

    assert len(arrays_train) == len(labels_train) == 2
    assert len(arrays_test) == len(labels_test) == 2

    assert arrays_train[0].shape == (100, 130)
    assert arrays_train[1].shape == (99, 130)
    assert arrays_test[0].shape == (100, 130)
    assert arrays_test[1].shape == (99, 130)

    for array in arrays_train + arrays_test:
        assert isinstance(array, numpy.ndarray)

    assert labels_train[0] == {"anomaly": False, "category": "NORMAL_OPERATION", "sample": 755, "variant": "PRE_A"}
    assert labels_train[1] == {"sample": 1702, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_A"}
    assert labels_test[0] == {"sample": 12, "anomaly": True, "category": "AXIS_FRICTION", "variant": "A1_B"}
    assert labels_test[1] == {"sample": 1703, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_B"}


def test_load_torch_tensors() -> None:
    tensors_train, labels_train, tensors_test, labels_test = voraus_ad.load_torch_tensors(
        path=DATASET_PATH,
        columns=list(voraus_ad.Signals.machine()),
        normalize=True,
        frequency_divider=1,
        train_gain=1.0,
        pad=True,
    )

    assert len(tensors_train) == len(labels_train) == 2
    assert len(tensors_test) == len(labels_test) == 2

    assert tensors_train[0].shape == (100, 130)
    assert tensors_train[1].shape == (100, 130)
    assert tensors_test[0].shape == (100, 130)
    assert tensors_test[1].shape == (100, 130)

    for array in tensors_train + tensors_test:
        assert isinstance(array, torch.Tensor)

    assert labels_train[0] == {"anomaly": False, "category": "NORMAL_OPERATION", "sample": 755, "variant": "PRE_A"}
    assert labels_train[1] == {"sample": 1702, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_A"}
    assert labels_test[0] == {"sample": 12, "anomaly": True, "category": "AXIS_FRICTION", "variant": "A1_B"}
    assert labels_test[1] == {"sample": 1703, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_B"}


def test_load_torch_tensors_train_gain_05() -> None:
    tensors_train, labels_train, tensors_test, labels_test = voraus_ad.load_torch_tensors(
        path=DATASET_PATH,
        columns=list(voraus_ad.Signals.machine()),
        normalize=True,
        frequency_divider=1,
        train_gain=0.5,
        pad=True,
    )

    assert len(tensors_train) == len(labels_train) == 1
    assert len(tensors_test) == len(labels_test) == 2


def test_load_torch_tensors_frequency_divider_2() -> None:
    tensors_train, _, tensors_test, _ = voraus_ad.load_torch_tensors(
        path=DATASET_PATH,
        columns=list(voraus_ad.Signals.machine()),
        normalize=True,
        frequency_divider=2,
        train_gain=1.0,
        pad=True,
    )

    assert tensors_train[0].shape == (50, 130)
    assert tensors_train[1].shape == (50, 130)
    assert tensors_test[0].shape == (50, 130)
    assert tensors_test[1].shape == (50, 130)


def test_voraus_ad_dataset() -> None:
    tensors_train, labels_train, _, _ = voraus_ad.load_torch_tensors(
        path=DATASET_PATH,
        columns=list(voraus_ad.Signals.machine()),
        normalize=True,
        frequency_divider=1,
        train_gain=1.0,
        pad=True,
    )

    voraus_ad_ds = voraus_ad.VorausADDataset(tensors_train, labels_train, columns=list(voraus_ad.Signals.machine()))

    assert len(voraus_ad_ds) == 2

    tensor_0, label_0 = voraus_ad_ds[0]
    assert tensor_0 is tensors_train[0]
    assert label_0 == {"anomaly": False, "category": "NORMAL_OPERATION", "sample": 755, "variant": "PRE_A"}

    tensor_1, label_1 = voraus_ad_ds[1]
    assert tensor_1 is tensors_train[1]
    assert label_1 == {"sample": 1702, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_A"}


def test_load_torch_dataloaders() -> None:
    train_dataset, test_dataset, train_dataloader, test_dataloader = voraus_ad.load_torch_dataloaders(
        dataset=DATASET_PATH,
        columns=list(voraus_ad.Signals.machine()),
        normalize=True,
        frequency_divider=1,
        train_gain=1.0,
        pad=True,
        batch_size=2,
        seed=10,
    )

    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)

    expected_train_labels = [
        {"anomaly": False, "category": "NORMAL_OPERATION", "sample": 755, "variant": "PRE_A"},
        {"sample": 1702, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_A"},
    ]

    expected_test_labels = [
        {"sample": 12, "anomaly": True, "category": "AXIS_FRICTION", "variant": "A1_B"},
        {"sample": 1703, "anomaly": False, "category": "NORMAL_OPERATION", "variant": "PRE_B"},
    ]

    assert train_dataset[0][1] == expected_train_labels[0]
    assert train_dataset[1][1] == expected_train_labels[1]
    assert test_dataset[0][1] == expected_test_labels[0]
    assert test_dataset[1][1] == expected_test_labels[1]

    for batch, labels in train_dataloader:
        for j in range(batch.shape[0]):
            label = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
            assert label == expected_train_labels[j]

    for batch, labels in test_dataloader:
        for j in range(batch.shape[0]):
            label = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
            assert label == expected_test_labels[j]


def test_signals_all() -> None:
    assert voraus_ad.Signals.all() == (
        "time",
        "sample",
        "anomaly",
        "category",
        "setting",
        "action",
        "active",
        "robot_voltage",
        "robot_current",
        "io_current",
        "system_current",
        "target_position_1",
        "target_velocity_1",
        "target_acceleration_1",
        "target_torque_1",
        "computed_inertia_1",
        "computed_torque_1",
        "motor_position_1",
        "motor_velocity_1",
        "joint_position_1",
        "joint_velocity_1",
        "motor_torque_1",
        "torque_sensor_a_1",
        "torque_sensor_b_1",
        "motor_iq_1",
        "motor_id_1",
        "power_motor_el_1",
        "power_motor_mech_1",
        "power_load_mech_1",
        "motor_voltage_1",
        "supply_voltage_1",
        "brake_voltage_1",
        "target_position_2",
        "target_velocity_2",
        "target_acceleration_2",
        "target_torque_2",
        "computed_inertia_2",
        "computed_torque_2",
        "motor_position_2",
        "motor_velocity_2",
        "joint_position_2",
        "joint_velocity_2",
        "motor_torque_2",
        "torque_sensor_a_2",
        "torque_sensor_b_2",
        "motor_iq_2",
        "motor_id_2",
        "power_motor_el_2",
        "power_motor_mech_2",
        "power_load_mech_2",
        "motor_voltage_2",
        "supply_voltage_2",
        "brake_voltage_2",
        "target_position_3",
        "target_velocity_3",
        "target_acceleration_3",
        "target_torque_3",
        "computed_inertia_3",
        "computed_torque_3",
        "motor_position_3",
        "motor_velocity_3",
        "joint_position_3",
        "joint_velocity_3",
        "motor_torque_3",
        "torque_sensor_a_3",
        "torque_sensor_b_3",
        "motor_iq_3",
        "motor_id_3",
        "power_motor_el_3",
        "power_motor_mech_3",
        "power_load_mech_3",
        "motor_voltage_3",
        "supply_voltage_3",
        "brake_voltage_3",
        "target_position_4",
        "target_velocity_4",
        "target_acceleration_4",
        "target_torque_4",
        "computed_inertia_4",
        "computed_torque_4",
        "motor_position_4",
        "motor_velocity_4",
        "joint_position_4",
        "joint_velocity_4",
        "motor_torque_4",
        "torque_sensor_a_4",
        "torque_sensor_b_4",
        "motor_iq_4",
        "motor_id_4",
        "power_motor_el_4",
        "power_motor_mech_4",
        "power_load_mech_4",
        "motor_voltage_4",
        "supply_voltage_4",
        "brake_voltage_4",
        "target_position_5",
        "target_velocity_5",
        "target_acceleration_5",
        "target_torque_5",
        "computed_inertia_5",
        "computed_torque_5",
        "motor_position_5",
        "motor_velocity_5",
        "joint_position_5",
        "joint_velocity_5",
        "motor_torque_5",
        "torque_sensor_a_5",
        "torque_sensor_b_5",
        "motor_iq_5",
        "motor_id_5",
        "power_motor_el_5",
        "power_motor_mech_5",
        "power_load_mech_5",
        "motor_voltage_5",
        "supply_voltage_5",
        "brake_voltage_5",
        "target_position_6",
        "target_velocity_6",
        "target_acceleration_6",
        "target_torque_6",
        "computed_inertia_6",
        "computed_torque_6",
        "motor_position_6",
        "motor_velocity_6",
        "joint_position_6",
        "joint_velocity_6",
        "motor_torque_6",
        "torque_sensor_a_6",
        "torque_sensor_b_6",
        "motor_iq_6",
        "motor_id_6",
        "power_motor_el_6",
        "power_motor_mech_6",
        "power_load_mech_6",
        "motor_voltage_6",
        "supply_voltage_6",
        "brake_voltage_6",
    )


def test_signals_meta() -> None:
    assert voraus_ad.Signals.meta() == (
        "time",
        "sample",
        "anomaly",
        "category",
        "setting",
        "action",
        "active",
    )


def test_signals_meta_constant() -> None:
    assert voraus_ad.Signals.meta_constant() == (
        "sample",
        "anomaly",
        "category",
        "setting",
    )


def test_signals_electrical() -> None:
    assert voraus_ad.Signals.electrical() == (
        "robot_voltage",
        "robot_current",
        "io_current",
        "system_current",
        "motor_iq_1",
        "motor_id_1",
        "power_motor_el_1",
        "motor_voltage_1",
        "supply_voltage_1",
        "brake_voltage_1",
        "motor_iq_2",
        "motor_id_2",
        "power_motor_el_2",
        "motor_voltage_2",
        "supply_voltage_2",
        "brake_voltage_2",
        "motor_iq_3",
        "motor_id_3",
        "power_motor_el_3",
        "motor_voltage_3",
        "supply_voltage_3",
        "brake_voltage_3",
        "motor_iq_4",
        "motor_id_4",
        "power_motor_el_4",
        "motor_voltage_4",
        "supply_voltage_4",
        "brake_voltage_4",
        "motor_iq_5",
        "motor_id_5",
        "power_motor_el_5",
        "motor_voltage_5",
        "supply_voltage_5",
        "brake_voltage_5",
        "motor_iq_6",
        "motor_id_6",
        "power_motor_el_6",
        "motor_voltage_6",
        "supply_voltage_6",
        "brake_voltage_6",
    )


def test_signals_measured() -> None:
    assert voraus_ad.Signals.measured() == (
        "robot_voltage",
        "robot_current",
        "io_current",
        "system_current",
        "motor_position_1",
        "motor_velocity_1",
        "joint_position_1",
        "joint_velocity_1",
        "torque_sensor_a_1",
        "torque_sensor_b_1",
        "motor_voltage_1",
        "supply_voltage_1",
        "brake_voltage_1",
        "motor_position_2",
        "motor_velocity_2",
        "joint_position_2",
        "joint_velocity_2",
        "torque_sensor_a_2",
        "torque_sensor_b_2",
        "motor_voltage_2",
        "supply_voltage_2",
        "brake_voltage_2",
        "motor_position_3",
        "motor_velocity_3",
        "joint_position_3",
        "joint_velocity_3",
        "torque_sensor_a_3",
        "torque_sensor_b_3",
        "motor_voltage_3",
        "supply_voltage_3",
        "brake_voltage_3",
        "motor_position_4",
        "motor_velocity_4",
        "joint_position_4",
        "joint_velocity_4",
        "torque_sensor_a_4",
        "torque_sensor_b_4",
        "motor_voltage_4",
        "supply_voltage_4",
        "brake_voltage_4",
        "motor_position_5",
        "motor_velocity_5",
        "joint_position_5",
        "joint_velocity_5",
        "torque_sensor_a_5",
        "torque_sensor_b_5",
        "motor_voltage_5",
        "supply_voltage_5",
        "brake_voltage_5",
        "motor_position_6",
        "motor_velocity_6",
        "joint_position_6",
        "joint_velocity_6",
        "torque_sensor_a_6",
        "torque_sensor_b_6",
        "motor_voltage_6",
        "supply_voltage_6",
        "brake_voltage_6",
    )


def test_signals_robot() -> None:
    assert voraus_ad.Signals.robot() == ("robot_voltage", "robot_current", "io_current", "system_current")


def test_signals_machine() -> None:
    assert voraus_ad.Signals.machine() == (
        "robot_voltage",
        "robot_current",
        "io_current",
        "system_current",
        "target_position_1",
        "target_velocity_1",
        "target_acceleration_1",
        "target_torque_1",
        "computed_inertia_1",
        "computed_torque_1",
        "motor_position_1",
        "motor_velocity_1",
        "joint_position_1",
        "joint_velocity_1",
        "motor_torque_1",
        "torque_sensor_a_1",
        "torque_sensor_b_1",
        "motor_iq_1",
        "motor_id_1",
        "power_motor_el_1",
        "power_motor_mech_1",
        "power_load_mech_1",
        "motor_voltage_1",
        "supply_voltage_1",
        "brake_voltage_1",
        "target_position_2",
        "target_velocity_2",
        "target_acceleration_2",
        "target_torque_2",
        "computed_inertia_2",
        "computed_torque_2",
        "motor_position_2",
        "motor_velocity_2",
        "joint_position_2",
        "joint_velocity_2",
        "motor_torque_2",
        "torque_sensor_a_2",
        "torque_sensor_b_2",
        "motor_iq_2",
        "motor_id_2",
        "power_motor_el_2",
        "power_motor_mech_2",
        "power_load_mech_2",
        "motor_voltage_2",
        "supply_voltage_2",
        "brake_voltage_2",
        "target_position_3",
        "target_velocity_3",
        "target_acceleration_3",
        "target_torque_3",
        "computed_inertia_3",
        "computed_torque_3",
        "motor_position_3",
        "motor_velocity_3",
        "joint_position_3",
        "joint_velocity_3",
        "motor_torque_3",
        "torque_sensor_a_3",
        "torque_sensor_b_3",
        "motor_iq_3",
        "motor_id_3",
        "power_motor_el_3",
        "power_motor_mech_3",
        "power_load_mech_3",
        "motor_voltage_3",
        "supply_voltage_3",
        "brake_voltage_3",
        "target_position_4",
        "target_velocity_4",
        "target_acceleration_4",
        "target_torque_4",
        "computed_inertia_4",
        "computed_torque_4",
        "motor_position_4",
        "motor_velocity_4",
        "joint_position_4",
        "joint_velocity_4",
        "motor_torque_4",
        "torque_sensor_a_4",
        "torque_sensor_b_4",
        "motor_iq_4",
        "motor_id_4",
        "power_motor_el_4",
        "power_motor_mech_4",
        "power_load_mech_4",
        "motor_voltage_4",
        "supply_voltage_4",
        "brake_voltage_4",
        "target_position_5",
        "target_velocity_5",
        "target_acceleration_5",
        "target_torque_5",
        "computed_inertia_5",
        "computed_torque_5",
        "motor_position_5",
        "motor_velocity_5",
        "joint_position_5",
        "joint_velocity_5",
        "motor_torque_5",
        "torque_sensor_a_5",
        "torque_sensor_b_5",
        "motor_iq_5",
        "motor_id_5",
        "power_motor_el_5",
        "power_motor_mech_5",
        "power_load_mech_5",
        "motor_voltage_5",
        "supply_voltage_5",
        "brake_voltage_5",
        "target_position_6",
        "target_velocity_6",
        "target_acceleration_6",
        "target_torque_6",
        "computed_inertia_6",
        "computed_torque_6",
        "motor_position_6",
        "motor_velocity_6",
        "joint_position_6",
        "joint_velocity_6",
        "motor_torque_6",
        "torque_sensor_a_6",
        "torque_sensor_b_6",
        "motor_iq_6",
        "motor_id_6",
        "power_motor_el_6",
        "power_motor_mech_6",
        "power_load_mech_6",
        "motor_voltage_6",
        "supply_voltage_6",
        "brake_voltage_6",
    )


def test_signals_mechanical() -> None:
    assert voraus_ad.Signals.mechanical() == (
        "target_position_1",
        "target_velocity_1",
        "target_acceleration_1",
        "target_torque_1",
        "computed_inertia_1",
        "computed_torque_1",
        "motor_position_1",
        "motor_velocity_1",
        "joint_position_1",
        "joint_velocity_1",
        "motor_torque_1",
        "torque_sensor_a_1",
        "torque_sensor_b_1",
        "power_motor_mech_1",
        "power_load_mech_1",
        "target_position_2",
        "target_velocity_2",
        "target_acceleration_2",
        "target_torque_2",
        "computed_inertia_2",
        "computed_torque_2",
        "motor_position_2",
        "motor_velocity_2",
        "joint_position_2",
        "joint_velocity_2",
        "motor_torque_2",
        "torque_sensor_a_2",
        "torque_sensor_b_2",
        "power_motor_mech_2",
        "power_load_mech_2",
        "target_position_3",
        "target_velocity_3",
        "target_acceleration_3",
        "target_torque_3",
        "computed_inertia_3",
        "computed_torque_3",
        "motor_position_3",
        "motor_velocity_3",
        "joint_position_3",
        "joint_velocity_3",
        "motor_torque_3",
        "torque_sensor_a_3",
        "torque_sensor_b_3",
        "power_motor_mech_3",
        "power_load_mech_3",
        "target_position_4",
        "target_velocity_4",
        "target_acceleration_4",
        "target_torque_4",
        "computed_inertia_4",
        "computed_torque_4",
        "motor_position_4",
        "motor_velocity_4",
        "joint_position_4",
        "joint_velocity_4",
        "motor_torque_4",
        "torque_sensor_a_4",
        "torque_sensor_b_4",
        "power_motor_mech_4",
        "power_load_mech_4",
        "target_position_5",
        "target_velocity_5",
        "target_acceleration_5",
        "target_torque_5",
        "computed_inertia_5",
        "computed_torque_5",
        "motor_position_5",
        "motor_velocity_5",
        "joint_position_5",
        "joint_velocity_5",
        "motor_torque_5",
        "torque_sensor_a_5",
        "torque_sensor_b_5",
        "power_motor_mech_5",
        "power_load_mech_5",
        "target_position_6",
        "target_velocity_6",
        "target_acceleration_6",
        "target_torque_6",
        "computed_inertia_6",
        "computed_torque_6",
        "motor_position_6",
        "motor_velocity_6",
        "joint_position_6",
        "joint_velocity_6",
        "motor_torque_6",
        "torque_sensor_a_6",
        "torque_sensor_b_6",
        "power_motor_mech_6",
        "power_load_mech_6",
    )


def test_signals_computed() -> None:
    assert voraus_ad.Signals.computed() == (
        "target_position_1",
        "target_velocity_1",
        "target_acceleration_1",
        "target_torque_1",
        "computed_inertia_1",
        "computed_torque_1",
        "motor_torque_1",
        "motor_iq_1",
        "motor_id_1",
        "power_motor_el_1",
        "power_motor_mech_1",
        "power_load_mech_1",
        "target_position_2",
        "target_velocity_2",
        "target_acceleration_2",
        "target_torque_2",
        "computed_inertia_2",
        "computed_torque_2",
        "motor_torque_2",
        "motor_iq_2",
        "motor_id_2",
        "power_motor_el_2",
        "power_motor_mech_2",
        "power_load_mech_2",
        "target_position_3",
        "target_velocity_3",
        "target_acceleration_3",
        "target_torque_3",
        "computed_inertia_3",
        "computed_torque_3",
        "motor_torque_3",
        "motor_iq_3",
        "motor_id_3",
        "power_motor_el_3",
        "power_motor_mech_3",
        "power_load_mech_3",
        "target_position_4",
        "target_velocity_4",
        "target_acceleration_4",
        "target_torque_4",
        "computed_inertia_4",
        "computed_torque_4",
        "motor_torque_4",
        "motor_iq_4",
        "motor_id_4",
        "power_motor_el_4",
        "power_motor_mech_4",
        "power_load_mech_4",
        "target_position_5",
        "target_velocity_5",
        "target_acceleration_5",
        "target_torque_5",
        "computed_inertia_5",
        "computed_torque_5",
        "motor_torque_5",
        "motor_iq_5",
        "motor_id_5",
        "power_motor_el_5",
        "power_motor_mech_5",
        "power_load_mech_5",
        "target_position_6",
        "target_velocity_6",
        "target_acceleration_6",
        "target_torque_6",
        "computed_inertia_6",
        "computed_torque_6",
        "motor_torque_6",
        "motor_iq_6",
        "motor_id_6",
        "power_motor_el_6",
        "power_motor_mech_6",
        "power_load_mech_6",
    )


def test_signals_axis() -> None:
    assert voraus_ad.Signals.axis() == (
        "target_position_1",
        "target_velocity_1",
        "target_acceleration_1",
        "target_torque_1",
        "computed_inertia_1",
        "computed_torque_1",
        "motor_position_1",
        "motor_velocity_1",
        "joint_position_1",
        "joint_velocity_1",
        "motor_torque_1",
        "torque_sensor_a_1",
        "torque_sensor_b_1",
        "motor_iq_1",
        "motor_id_1",
        "power_motor_el_1",
        "power_motor_mech_1",
        "power_load_mech_1",
        "motor_voltage_1",
        "supply_voltage_1",
        "brake_voltage_1",
        "target_position_2",
        "target_velocity_2",
        "target_acceleration_2",
        "target_torque_2",
        "computed_inertia_2",
        "computed_torque_2",
        "motor_position_2",
        "motor_velocity_2",
        "joint_position_2",
        "joint_velocity_2",
        "motor_torque_2",
        "torque_sensor_a_2",
        "torque_sensor_b_2",
        "motor_iq_2",
        "motor_id_2",
        "power_motor_el_2",
        "power_motor_mech_2",
        "power_load_mech_2",
        "motor_voltage_2",
        "supply_voltage_2",
        "brake_voltage_2",
        "target_position_3",
        "target_velocity_3",
        "target_acceleration_3",
        "target_torque_3",
        "computed_inertia_3",
        "computed_torque_3",
        "motor_position_3",
        "motor_velocity_3",
        "joint_position_3",
        "joint_velocity_3",
        "motor_torque_3",
        "torque_sensor_a_3",
        "torque_sensor_b_3",
        "motor_iq_3",
        "motor_id_3",
        "power_motor_el_3",
        "power_motor_mech_3",
        "power_load_mech_3",
        "motor_voltage_3",
        "supply_voltage_3",
        "brake_voltage_3",
        "target_position_4",
        "target_velocity_4",
        "target_acceleration_4",
        "target_torque_4",
        "computed_inertia_4",
        "computed_torque_4",
        "motor_position_4",
        "motor_velocity_4",
        "joint_position_4",
        "joint_velocity_4",
        "motor_torque_4",
        "torque_sensor_a_4",
        "torque_sensor_b_4",
        "motor_iq_4",
        "motor_id_4",
        "power_motor_el_4",
        "power_motor_mech_4",
        "power_load_mech_4",
        "motor_voltage_4",
        "supply_voltage_4",
        "brake_voltage_4",
        "target_position_5",
        "target_velocity_5",
        "target_acceleration_5",
        "target_torque_5",
        "computed_inertia_5",
        "computed_torque_5",
        "motor_position_5",
        "motor_velocity_5",
        "joint_position_5",
        "joint_velocity_5",
        "motor_torque_5",
        "torque_sensor_a_5",
        "torque_sensor_b_5",
        "motor_iq_5",
        "motor_id_5",
        "power_motor_el_5",
        "power_motor_mech_5",
        "power_load_mech_5",
        "motor_voltage_5",
        "supply_voltage_5",
        "brake_voltage_5",
        "target_position_6",
        "target_velocity_6",
        "target_acceleration_6",
        "target_torque_6",
        "computed_inertia_6",
        "computed_torque_6",
        "motor_position_6",
        "motor_velocity_6",
        "joint_position_6",
        "joint_velocity_6",
        "motor_torque_6",
        "torque_sensor_a_6",
        "torque_sensor_b_6",
        "motor_iq_6",
        "motor_id_6",
        "power_motor_el_6",
        "power_motor_mech_6",
        "power_load_mech_6",
        "motor_voltage_6",
        "supply_voltage_6",
        "brake_voltage_6",
    )


def test_signals_groups() -> None:
    groups = voraus_ad.Signals.groups()
    assert "mechanical" in groups
    assert "electrical" in groups
    assert "computed" in groups
    assert "measured" in groups
    assert "machine" in groups
