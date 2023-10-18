"""This module contains all utility functions for the voraus-AD dataset."""

import time
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from random import sample
from typing import Dict, Generator, List, Tuple, Union

import numpy
import pandas
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class Signals:
    """Contains the signals of the robot used in the dataset."""

    TIME = "time"
    SAMPLE = "sample"
    ANOMALY = "anomaly"
    CATEGORY = "category"
    SETTING = "setting"
    ACTION = "action"
    ACTIVE = "active"
    ROBOT_VOLTAGE = "robot_voltage"
    ROBOT_CURRENT = "robot_current"
    IO_CURRENT = "io_current"
    SYSTEM_CURRENT = "system_current"
    TARGET_POSITION_1 = "target_position_1"
    TARGET_VELOCITY_1 = "target_velocity_1"
    TARGET_ACCELERATION_1 = "target_acceleration_1"
    TARGET_TORQUE_1 = "target_torque_1"
    COMPUTED_INERTIA_1 = "computed_inertia_1"
    COMPUTED_TORQUE_1 = "computed_torque_1"
    MOTOR_POSITION_1 = "motor_position_1"
    MOTOR_VELOCITY_1 = "motor_velocity_1"
    JOINT_POSITION_1 = "joint_position_1"
    JOINT_VELOCITY_1 = "joint_velocity_1"
    MOTOR_TORQUE_1 = "motor_torque_1"
    TORQUE_SENSOR_A_1 = "torque_sensor_a_1"
    TORQUE_SENSOR_B_1 = "torque_sensor_b_1"
    MOTOR_IQ_1 = "motor_iq_1"
    MOTOR_ID_1 = "motor_id_1"
    POWER_MOTOR_EL_1 = "power_motor_el_1"
    POWER_MOTOR_MECH_1 = "power_motor_mech_1"
    POWER_LOAD_MECH_1 = "power_load_mech_1"
    MOTOR_VOLTAGE_1 = "motor_voltage_1"
    SUPPLY_VOLTAGE_1 = "supply_voltage_1"
    BRAKE_VOLTAGE_1 = "brake_voltage_1"
    TARGET_POSITION_2 = "target_position_2"
    TARGET_VELOCITY_2 = "target_velocity_2"
    TARGET_ACCELERATION_2 = "target_acceleration_2"
    TARGET_TORQUE_2 = "target_torque_2"
    COMPUTED_INERTIA_2 = "computed_inertia_2"
    COMPUTED_TORQUE_2 = "computed_torque_2"
    MOTOR_POSITION_2 = "motor_position_2"
    MOTOR_VELOCITY_2 = "motor_velocity_2"
    JOINT_POSITION_2 = "joint_position_2"
    JOINT_VELOCITY_2 = "joint_velocity_2"
    MOTOR_TORQUE_2 = "motor_torque_2"
    TORQUE_SENSOR_A_2 = "torque_sensor_a_2"
    TORQUE_SENSOR_B_2 = "torque_sensor_b_2"
    MOTOR_IQ_2 = "motor_iq_2"
    MOTOR_ID_2 = "motor_id_2"
    POWER_MOTOR_EL_2 = "power_motor_el_2"
    POWER_MOTOR_MECH_2 = "power_motor_mech_2"
    POWER_LOAD_MECH_2 = "power_load_mech_2"
    MOTOR_VOLTAGE_2 = "motor_voltage_2"
    SUPPLY_VOLTAGE_2 = "supply_voltage_2"
    BRAKE_VOLTAGE_2 = "brake_voltage_2"
    TARGET_POSITION_3 = "target_position_3"
    TARGET_VELOCITY_3 = "target_velocity_3"
    TARGET_ACCELERATION_3 = "target_acceleration_3"
    TARGET_TORQUE_3 = "target_torque_3"
    COMPUTED_INERTIA_3 = "computed_inertia_3"
    COMPUTED_TORQUE_3 = "computed_torque_3"
    MOTOR_POSITION_3 = "motor_position_3"
    MOTOR_VELOCITY_3 = "motor_velocity_3"
    JOINT_POSITION_3 = "joint_position_3"
    JOINT_VELOCITY_3 = "joint_velocity_3"
    MOTOR_TORQUE_3 = "motor_torque_3"
    TORQUE_SENSOR_A_3 = "torque_sensor_a_3"
    TORQUE_SENSOR_B_3 = "torque_sensor_b_3"
    MOTOR_IQ_3 = "motor_iq_3"
    MOTOR_ID_3 = "motor_id_3"
    POWER_MOTOR_EL_3 = "power_motor_el_3"
    POWER_MOTOR_MECH_3 = "power_motor_mech_3"
    POWER_LOAD_MECH_3 = "power_load_mech_3"
    MOTOR_VOLTAGE_3 = "motor_voltage_3"
    SUPPLY_VOLTAGE_3 = "supply_voltage_3"
    BRAKE_VOLTAGE_3 = "brake_voltage_3"
    TARGET_POSITION_4 = "target_position_4"
    TARGET_VELOCITY_4 = "target_velocity_4"
    TARGET_ACCELERATION_4 = "target_acceleration_4"
    TARGET_TORQUE_4 = "target_torque_4"
    COMPUTED_INERTIA_4 = "computed_inertia_4"
    COMPUTED_TORQUE_4 = "computed_torque_4"
    MOTOR_POSITION_4 = "motor_position_4"
    MOTOR_VELOCITY_4 = "motor_velocity_4"
    JOINT_POSITION_4 = "joint_position_4"
    JOINT_VELOCITY_4 = "joint_velocity_4"
    MOTOR_TORQUE_4 = "motor_torque_4"
    TORQUE_SENSOR_A_4 = "torque_sensor_a_4"
    TORQUE_SENSOR_B_4 = "torque_sensor_b_4"
    MOTOR_IQ_4 = "motor_iq_4"
    MOTOR_ID_4 = "motor_id_4"
    POWER_MOTOR_EL_4 = "power_motor_el_4"
    POWER_MOTOR_MECH_4 = "power_motor_mech_4"
    POWER_LOAD_MECH_4 = "power_load_mech_4"
    MOTOR_VOLTAGE_4 = "motor_voltage_4"
    SUPPLY_VOLTAGE_4 = "supply_voltage_4"
    BRAKE_VOLTAGE_4 = "brake_voltage_4"
    TARGET_POSITION_5 = "target_position_5"
    TARGET_VELOCITY_5 = "target_velocity_5"
    TARGET_ACCELERATION_5 = "target_acceleration_5"
    TARGET_TORQUE_5 = "target_torque_5"
    COMPUTED_INERTIA_5 = "computed_inertia_5"
    COMPUTED_TORQUE_5 = "computed_torque_5"
    MOTOR_POSITION_5 = "motor_position_5"
    MOTOR_VELOCITY_5 = "motor_velocity_5"
    JOINT_POSITION_5 = "joint_position_5"
    JOINT_VELOCITY_5 = "joint_velocity_5"
    MOTOR_TORQUE_5 = "motor_torque_5"
    TORQUE_SENSOR_A_5 = "torque_sensor_a_5"
    TORQUE_SENSOR_B_5 = "torque_sensor_b_5"
    MOTOR_IQ_5 = "motor_iq_5"
    MOTOR_ID_5 = "motor_id_5"
    POWER_MOTOR_EL_5 = "power_motor_el_5"
    POWER_MOTOR_MECH_5 = "power_motor_mech_5"
    POWER_LOAD_MECH_5 = "power_load_mech_5"
    MOTOR_VOLTAGE_5 = "motor_voltage_5"
    SUPPLY_VOLTAGE_5 = "supply_voltage_5"
    BRAKE_VOLTAGE_5 = "brake_voltage_5"
    TARGET_POSITION_6 = "target_position_6"
    TARGET_VELOCITY_6 = "target_velocity_6"
    TARGET_ACCELERATION_6 = "target_acceleration_6"
    TARGET_TORQUE_6 = "target_torque_6"
    COMPUTED_INERTIA_6 = "computed_inertia_6"
    COMPUTED_TORQUE_6 = "computed_torque_6"
    MOTOR_POSITION_6 = "motor_position_6"
    MOTOR_VELOCITY_6 = "motor_velocity_6"
    JOINT_POSITION_6 = "joint_position_6"
    JOINT_VELOCITY_6 = "joint_velocity_6"
    MOTOR_TORQUE_6 = "motor_torque_6"
    TORQUE_SENSOR_A_6 = "torque_sensor_a_6"
    TORQUE_SENSOR_B_6 = "torque_sensor_b_6"
    MOTOR_IQ_6 = "motor_iq_6"
    MOTOR_ID_6 = "motor_id_6"
    POWER_MOTOR_EL_6 = "power_motor_el_6"
    POWER_MOTOR_MECH_6 = "power_motor_mech_6"
    POWER_LOAD_MECH_6 = "power_load_mech_6"
    MOTOR_VOLTAGE_6 = "motor_voltage_6"
    SUPPLY_VOLTAGE_6 = "supply_voltage_6"
    BRAKE_VOLTAGE_6 = "brake_voltage_6"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """Returns all signals (machine data and meta) included in the voraus-AD dataset.

        Returns:
            All signals of the voraus-AD dataset.
        """
        return (
            cls.TIME,
            cls.SAMPLE,
            cls.ANOMALY,
            cls.CATEGORY,
            cls.SETTING,
            cls.ACTION,
            cls.ACTIVE,
            cls.ROBOT_VOLTAGE,
            cls.ROBOT_CURRENT,
            cls.IO_CURRENT,
            cls.SYSTEM_CURRENT,
            cls.TARGET_POSITION_1,
            cls.TARGET_VELOCITY_1,
            cls.TARGET_ACCELERATION_1,
            cls.TARGET_TORQUE_1,
            cls.COMPUTED_INERTIA_1,
            cls.COMPUTED_TORQUE_1,
            cls.MOTOR_POSITION_1,
            cls.MOTOR_VELOCITY_1,
            cls.JOINT_POSITION_1,
            cls.JOINT_VELOCITY_1,
            cls.MOTOR_TORQUE_1,
            cls.TORQUE_SENSOR_A_1,
            cls.TORQUE_SENSOR_B_1,
            cls.MOTOR_IQ_1,
            cls.MOTOR_ID_1,
            cls.POWER_MOTOR_EL_1,
            cls.POWER_MOTOR_MECH_1,
            cls.POWER_LOAD_MECH_1,
            cls.MOTOR_VOLTAGE_1,
            cls.SUPPLY_VOLTAGE_1,
            cls.BRAKE_VOLTAGE_1,
            cls.TARGET_POSITION_2,
            cls.TARGET_VELOCITY_2,
            cls.TARGET_ACCELERATION_2,
            cls.TARGET_TORQUE_2,
            cls.COMPUTED_INERTIA_2,
            cls.COMPUTED_TORQUE_2,
            cls.MOTOR_POSITION_2,
            cls.MOTOR_VELOCITY_2,
            cls.JOINT_POSITION_2,
            cls.JOINT_VELOCITY_2,
            cls.MOTOR_TORQUE_2,
            cls.TORQUE_SENSOR_A_2,
            cls.TORQUE_SENSOR_B_2,
            cls.MOTOR_IQ_2,
            cls.MOTOR_ID_2,
            cls.POWER_MOTOR_EL_2,
            cls.POWER_MOTOR_MECH_2,
            cls.POWER_LOAD_MECH_2,
            cls.MOTOR_VOLTAGE_2,
            cls.SUPPLY_VOLTAGE_2,
            cls.BRAKE_VOLTAGE_2,
            cls.TARGET_POSITION_3,
            cls.TARGET_VELOCITY_3,
            cls.TARGET_ACCELERATION_3,
            cls.TARGET_TORQUE_3,
            cls.COMPUTED_INERTIA_3,
            cls.COMPUTED_TORQUE_3,
            cls.MOTOR_POSITION_3,
            cls.MOTOR_VELOCITY_3,
            cls.JOINT_POSITION_3,
            cls.JOINT_VELOCITY_3,
            cls.MOTOR_TORQUE_3,
            cls.TORQUE_SENSOR_A_3,
            cls.TORQUE_SENSOR_B_3,
            cls.MOTOR_IQ_3,
            cls.MOTOR_ID_3,
            cls.POWER_MOTOR_EL_3,
            cls.POWER_MOTOR_MECH_3,
            cls.POWER_LOAD_MECH_3,
            cls.MOTOR_VOLTAGE_3,
            cls.SUPPLY_VOLTAGE_3,
            cls.BRAKE_VOLTAGE_3,
            cls.TARGET_POSITION_4,
            cls.TARGET_VELOCITY_4,
            cls.TARGET_ACCELERATION_4,
            cls.TARGET_TORQUE_4,
            cls.COMPUTED_INERTIA_4,
            cls.COMPUTED_TORQUE_4,
            cls.MOTOR_POSITION_4,
            cls.MOTOR_VELOCITY_4,
            cls.JOINT_POSITION_4,
            cls.JOINT_VELOCITY_4,
            cls.MOTOR_TORQUE_4,
            cls.TORQUE_SENSOR_A_4,
            cls.TORQUE_SENSOR_B_4,
            cls.MOTOR_IQ_4,
            cls.MOTOR_ID_4,
            cls.POWER_MOTOR_EL_4,
            cls.POWER_MOTOR_MECH_4,
            cls.POWER_LOAD_MECH_4,
            cls.MOTOR_VOLTAGE_4,
            cls.SUPPLY_VOLTAGE_4,
            cls.BRAKE_VOLTAGE_4,
            cls.TARGET_POSITION_5,
            cls.TARGET_VELOCITY_5,
            cls.TARGET_ACCELERATION_5,
            cls.TARGET_TORQUE_5,
            cls.COMPUTED_INERTIA_5,
            cls.COMPUTED_TORQUE_5,
            cls.MOTOR_POSITION_5,
            cls.MOTOR_VELOCITY_5,
            cls.JOINT_POSITION_5,
            cls.JOINT_VELOCITY_5,
            cls.MOTOR_TORQUE_5,
            cls.TORQUE_SENSOR_A_5,
            cls.TORQUE_SENSOR_B_5,
            cls.MOTOR_IQ_5,
            cls.MOTOR_ID_5,
            cls.POWER_MOTOR_EL_5,
            cls.POWER_MOTOR_MECH_5,
            cls.POWER_LOAD_MECH_5,
            cls.MOTOR_VOLTAGE_5,
            cls.SUPPLY_VOLTAGE_5,
            cls.BRAKE_VOLTAGE_5,
            cls.TARGET_POSITION_6,
            cls.TARGET_VELOCITY_6,
            cls.TARGET_ACCELERATION_6,
            cls.TARGET_TORQUE_6,
            cls.COMPUTED_INERTIA_6,
            cls.COMPUTED_TORQUE_6,
            cls.MOTOR_POSITION_6,
            cls.MOTOR_VELOCITY_6,
            cls.JOINT_POSITION_6,
            cls.JOINT_VELOCITY_6,
            cls.MOTOR_TORQUE_6,
            cls.TORQUE_SENSOR_A_6,
            cls.TORQUE_SENSOR_B_6,
            cls.MOTOR_IQ_6,
            cls.MOTOR_ID_6,
            cls.POWER_MOTOR_EL_6,
            cls.POWER_MOTOR_MECH_6,
            cls.POWER_LOAD_MECH_6,
            cls.MOTOR_VOLTAGE_6,
            cls.SUPPLY_VOLTAGE_6,
            cls.BRAKE_VOLTAGE_6,
        )

    @classmethod
    def meta(cls) -> tuple[str, ...]:
        """Returns the meta colums of the voraus-AD dataset.

        Returns:
            The meta columns of the dataset.
        """
        return (
            cls.TIME,
            cls.SAMPLE,
            cls.ANOMALY,
            cls.CATEGORY,
            cls.SETTING,
            cls.ACTION,
            cls.ACTIVE,
        )

    @classmethod
    def meta_constant(cls) -> tuple[str, ...]:
        """Returns time invariant meta colums of the voraus-AD dataset.

        Returns:
            The time invariant meta columns.
        """
        return (
            cls.SAMPLE,
            cls.ANOMALY,
            cls.CATEGORY,
            cls.SETTING,
        )

    @classmethod
    def electrical(cls) -> tuple[str, ...]:
        """Returns the part of the machine data columns, which describes electrical values.

        Returns:
            The electrical signals.
        """
        return (
            cls.ROBOT_VOLTAGE,
            cls.ROBOT_CURRENT,
            cls.IO_CURRENT,
            cls.SYSTEM_CURRENT,
            cls.MOTOR_IQ_1,
            cls.MOTOR_ID_1,
            cls.POWER_MOTOR_EL_1,
            cls.MOTOR_VOLTAGE_1,
            cls.SUPPLY_VOLTAGE_1,
            cls.BRAKE_VOLTAGE_1,
            cls.MOTOR_IQ_2,
            cls.MOTOR_ID_2,
            cls.POWER_MOTOR_EL_2,
            cls.MOTOR_VOLTAGE_2,
            cls.SUPPLY_VOLTAGE_2,
            cls.BRAKE_VOLTAGE_2,
            cls.MOTOR_IQ_3,
            cls.MOTOR_ID_3,
            cls.POWER_MOTOR_EL_3,
            cls.MOTOR_VOLTAGE_3,
            cls.SUPPLY_VOLTAGE_3,
            cls.BRAKE_VOLTAGE_3,
            cls.MOTOR_IQ_4,
            cls.MOTOR_ID_4,
            cls.POWER_MOTOR_EL_4,
            cls.MOTOR_VOLTAGE_4,
            cls.SUPPLY_VOLTAGE_4,
            cls.BRAKE_VOLTAGE_4,
            cls.MOTOR_IQ_5,
            cls.MOTOR_ID_5,
            cls.POWER_MOTOR_EL_5,
            cls.MOTOR_VOLTAGE_5,
            cls.SUPPLY_VOLTAGE_5,
            cls.BRAKE_VOLTAGE_5,
            cls.MOTOR_IQ_6,
            cls.MOTOR_ID_6,
            cls.POWER_MOTOR_EL_6,
            cls.MOTOR_VOLTAGE_6,
            cls.SUPPLY_VOLTAGE_6,
            cls.BRAKE_VOLTAGE_6,
        )

    @classmethod
    def measured(cls) -> tuple[str, ...]:
        """Returns the part of the machine data, which describes measured values.

        Returns:
            The measured signals.
        """
        return (
            cls.ROBOT_VOLTAGE,
            cls.ROBOT_CURRENT,
            cls.IO_CURRENT,
            cls.SYSTEM_CURRENT,
            cls.MOTOR_POSITION_1,
            cls.MOTOR_VELOCITY_1,
            cls.JOINT_POSITION_1,
            cls.JOINT_VELOCITY_1,
            cls.TORQUE_SENSOR_A_1,
            cls.TORQUE_SENSOR_B_1,
            cls.MOTOR_VOLTAGE_1,
            cls.SUPPLY_VOLTAGE_1,
            cls.BRAKE_VOLTAGE_1,
            cls.MOTOR_POSITION_2,
            cls.MOTOR_VELOCITY_2,
            cls.JOINT_POSITION_2,
            cls.JOINT_VELOCITY_2,
            cls.TORQUE_SENSOR_A_2,
            cls.TORQUE_SENSOR_B_2,
            cls.MOTOR_VOLTAGE_2,
            cls.SUPPLY_VOLTAGE_2,
            cls.BRAKE_VOLTAGE_2,
            cls.MOTOR_POSITION_3,
            cls.MOTOR_VELOCITY_3,
            cls.JOINT_POSITION_3,
            cls.JOINT_VELOCITY_3,
            cls.TORQUE_SENSOR_A_3,
            cls.TORQUE_SENSOR_B_3,
            cls.MOTOR_VOLTAGE_3,
            cls.SUPPLY_VOLTAGE_3,
            cls.BRAKE_VOLTAGE_3,
            cls.MOTOR_POSITION_4,
            cls.MOTOR_VELOCITY_4,
            cls.JOINT_POSITION_4,
            cls.JOINT_VELOCITY_4,
            cls.TORQUE_SENSOR_A_4,
            cls.TORQUE_SENSOR_B_4,
            cls.MOTOR_VOLTAGE_4,
            cls.SUPPLY_VOLTAGE_4,
            cls.BRAKE_VOLTAGE_4,
            cls.MOTOR_POSITION_5,
            cls.MOTOR_VELOCITY_5,
            cls.JOINT_POSITION_5,
            cls.JOINT_VELOCITY_5,
            cls.TORQUE_SENSOR_A_5,
            cls.TORQUE_SENSOR_B_5,
            cls.MOTOR_VOLTAGE_5,
            cls.SUPPLY_VOLTAGE_5,
            cls.BRAKE_VOLTAGE_5,
            cls.MOTOR_POSITION_6,
            cls.MOTOR_VELOCITY_6,
            cls.JOINT_POSITION_6,
            cls.JOINT_VELOCITY_6,
            cls.TORQUE_SENSOR_A_6,
            cls.TORQUE_SENSOR_B_6,
            cls.MOTOR_VOLTAGE_6,
            cls.SUPPLY_VOLTAGE_6,
            cls.BRAKE_VOLTAGE_6,
        )

    @classmethod
    def robot(cls) -> tuple[str, ...]:
        """Returns all columns, which are not related to the robot axes, but to the robot itself.

        Returns:
            The robot system signals.
        """
        return (
            cls.ROBOT_VOLTAGE,
            cls.ROBOT_CURRENT,
            cls.IO_CURRENT,
            cls.SYSTEM_CURRENT,
        )

    @classmethod
    def machine(cls) -> tuple[str, ...]:
        """Returns all columns, which are machine data.

        This excludes the meta columns of the dataset.
        The machine data should be used for training, it contains all available measurements and target values.

        Returns:
            The machine data signals.
        """
        return tuple(s for s in cls.all() if s not in cls.meta())

    @classmethod
    def mechanical(cls) -> tuple[str, ...]:
        """Returns the columns, which describe mechanical values.

        Returns:
            The machanical signals.
        """
        return tuple(s for s in cls.machine() if s not in cls.electrical())

    @classmethod
    def computed(cls) -> tuple[str, ...]:
        """Returns the columns, which describe computed values like targets.

        Returns:
            The computed signals.
        """
        return tuple(s for s in cls.machine() if s not in cls.measured())

    @classmethod
    def axis(cls) -> tuple[str, ...]:
        """Returns the columns, which describe robot axis specific values.

        Returns:
            The robot axis specific signals.
        """
        signals_axis = tuple(s for s in cls.machine() if s not in cls.robot())
        number_of_axis = 6
        assert len(signals_axis) % number_of_axis == 0
        signals_per_axis = round(len(signals_axis) / number_of_axis)
        print(signals_per_axis)
        return signals_axis

    @classmethod
    def groups(cls) -> dict[str, tuple[str, ...]]:
        """Access the signal groups by name.

        Returns:
            The signal group dictionary.
        """
        return {
            "mechanical": cls.mechanical(),
            "electrical": cls.electrical(),
            "computed": cls.computed(),
            "measured": cls.measured(),
            "machine": cls.machine(),  #  all machine data
        }


class Category(IntEnum):
    """Describes the anomaly category as published in the paper."""

    AXIS_FRICTION = 0
    AXIS_WEIGHT = 1
    COLLISION_FOAM = 2
    COLLISION_CABLE = 3
    COLLISION_CARTON = 4
    MISS_CAN = 5
    LOSE_CAN = 6
    CAN_WEIGHT = 7
    ENTANGLED = 8
    INVALID_POSITION = 9
    MOTOR_COMMUTATION = 10
    WOBBLING_STATION = 11
    NORMAL_OPERATION = 12


class Variant(IntEnum):
    """Describes the anomaly variant as published in the paper."""

    A1_A = 0
    A1_B = 1
    A2_A = 2
    A2_B = 3
    A3_A = 4
    A3_B = 5
    A3_C = 6
    A4_A = 7
    A4_B = 8
    A4_C = 9
    A5_A = 10
    A5_B = 11
    A6_A = 12
    A6_B = 13
    A1_115G = 14
    A1_231G = 15
    A1_500G = 16
    A2_500G = 17
    A2_231G = 18
    A2_115G = 19
    A3_115G = 20
    A3_231G = 21
    A3_500G = 22
    A4_500G = 23
    A4_231G = 24
    A4_115G = 25
    A5_115G = 26
    A5_231G = 27
    A5_500G = 28
    A6_500G = 29
    A6_231G = 30
    A6_115G = 31
    LARGE_FULL = 32
    SMALL_FULL = 33
    SMALL_STRIPE = 34
    SMALL_HANGING_FULL = 35
    LARGE_HANGING_FULL = 36
    SMALL_HANGING_STRIPE = 37
    CONNECTOR_TOP = 38
    CONNECTOR_SIDE = 39
    LAN = 40
    STRIPE = 41
    FULL_A = 42
    FULL_B = 43
    CONVEYOR = 44
    NO_VACUUM = 45
    DEFECT_GRIPPER_DROP_CAN = 46
    DROP_1400MS = 47
    DROP_1100MS = 48
    DROP_100MS = 49
    DROP_200MS = 50
    LOSE_200MS = 51
    LOSE_300MS = 52
    LOSE_PLACE_100MS = 53
    DECREASE_95G = 54
    DECREASE_165G = 55
    EMPTY = 56
    INCREASE_242G = 57
    INCREASE_373G = 58
    INCREASE_407G = 59
    CABLE = 60
    PICK = 61
    A1_HIGH = 62
    A1_MEDIUM = 63
    A2_MEDIUM = 64
    A3_MEDIUM = 65
    A4_MEDIUM = 66
    A5_MEDIUM = 67
    A6_MEDIUM = 68
    LOW_DAMPING = 69
    MEDIUM_DAMPING = 70
    HIGH_DAMPING = 71
    PRE_A = 72
    PRE_B = 73
    BETWEEN_A = 74
    BETWEEN_B = 75
    BETWEEN_C = 76


class Action(IntEnum):
    """Describes the actual action of the robot as published in the paper."""

    VOID_1 = 0
    TO_SCAN_PTP = 1
    VOID_2 = 2
    OVER_OBJECT_PTP = 3
    VOID_3 = 4
    DOWN_PICK_LIN = 5
    VOID_4 = 6
    UP_PICK_LIN = 7
    VOID_5 = 8
    TO_PLACE_PTP = 9
    VOID_6 = 10
    DOWN_PLACE_LIN = 11
    VOID_7 = 12
    UP_PLACE_LIN = 13
    VOID_8 = 14


ANOMALY_CATEGORIES = [
    Category.AXIS_FRICTION,
    Category.AXIS_WEIGHT,
    Category.COLLISION_FOAM,
    Category.COLLISION_CABLE,
    Category.COLLISION_CARTON,
    Category.MISS_CAN,
    Category.LOSE_CAN,
    Category.CAN_WEIGHT,
    Category.ENTANGLED,
    Category.INVALID_POSITION,
    Category.MOTOR_COMMUTATION,
    Category.WOBBLING_STATION,
]


@contextmanager
def measure_time(label: str) -> Generator[None, None, None]:
    """Measures the time and prints it to the console.

    Args:
        label: A label to identifiy the measured time.

    Yields:
        None.
    """
    start_time = time.time()
    yield
    print(f"{label} took {time.time()-start_time:.3f} seconds")


def extract_samples_and_labels(
    dataset: pandas.DataFrame, samples: List[int], meta_columns: List[str]
) -> Tuple[List[pandas.DataFrame], List[Dict]]:
    """Extracts one dataframe per sample from the dataset dataframe.

    Args:
        dataset: The dataset dataframe, containing all the samples.
        samples: The sample indices to extract.
        meta_columns: The meta columns to use during loading.

    Returns:
        The extracted dataframes and labels for each selected sample.
    """
    dfs = [dataset[dataset["sample"] == s].reset_index(drop=True) for s in samples]
    labels = [df.loc[0, meta_columns].to_dict() for df in dfs]
    dfs = [df.drop(columns=meta_columns) for df in dfs]
    return dfs, labels


# Disable pylint too many locals for better readability of the loading function.
def load_pandas_dataframes(  # pylint: disable=too-many-locals, too-complex
    path: Union[Path, str],
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool,
) -> Tuple[List[pandas.DataFrame], List[dict], List[pandas.DataFrame], List[dict]]:
    """Loads the dataset as pandas dataframes.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The dataframes and labels for each sample.
    """
    if isinstance(columns, tuple):
        columns = list(columns)
    with measure_time("loading data"):
        # Add required meta columns for preprocessing
        columns_with_meta = list(columns) + list(Signals.meta_constant())
        # Read the dataset parquet file columns as pandas dataframe
        dataset_dataframe = pandas.read_parquet(path, columns=columns_with_meta)
        # Rename the setting column to variant
        dataset_dataframe.rename(columns={"setting": "variant"}, inplace=True)
        # Create new meta columns list with variant inside
        meta_columns = [m for m in Signals.meta_constant() if m != "setting"] + ["variant"]

    # Check that down sampling factor is greater equal 1
    assert frequency_divider >= 1
    if frequency_divider > 1:
        with measure_time("downsampling"):
            print(f"downsample to every {frequency_divider}th frame")
            dataset_dataframe = dataset_dataframe[dataset_dataframe.index.values % frequency_divider == 0]
            dataset_dataframe = dataset_dataframe.reset_index(drop=True)

    # select train samples
    train_idx = sorted(dataset_dataframe.loc[dataset_dataframe["variant"] == Variant.PRE_A, "sample"].unique())
    if train_gain < 1.0:
        with measure_time("select train samples (train gain < 1.0)"):
            train_len = len(train_idx)
            train_k = round(train_len * train_gain)
            train_idx = sample(train_idx, k=train_k)
            print(f"select {train_k} of {train_len} train samples ({train_k/train_len:.0%})")

    # load training data
    with measure_time("extract train dfs and labels"):
        dfs_train, labels_train = extract_samples_and_labels(dataset_dataframe, train_idx, meta_columns)

    with measure_time("extract test dfs and labels"):
        dfs_test, labels_test = extract_samples_and_labels(
            dataset_dataframe,
            dataset_dataframe.loc[dataset_dataframe["variant"] != Variant.PRE_A, "sample"].unique(),
            meta_columns,
        )

    # update meta
    with measure_time("update meta data"):
        for label in labels_train + labels_test:
            label["category"] = Category(label["category"]).name
            label["variant"] = Variant(label["variant"]).name

    if normalize:
        with measure_time("normalize"):
            scale = StandardScaler()
            # using training data only
            scale.fit(pandas.concat(dfs_train))

            for df_i, dataframe in enumerate(dfs_train):
                dfs_train[df_i] = pandas.DataFrame(scale.transform(dataframe), columns=dataframe.columns)
            for df_i, dataframe in enumerate(dfs_test):
                dfs_test[df_i] = pandas.DataFrame(scale.transform(dataframe), columns=dataframe.columns)

    if pad:
        with measure_time("padding"):
            # Get maximum length from training samples
            target_length = max(len(df) for df in dfs_train)
            for df_i, dataframe in enumerate(dfs_train):
                pad = pandas.DataFrame(0, index=(range(len(dataframe), target_length)), columns=dataframe.columns)
                dfs_train[df_i] = pandas.concat([dataframe, pad])
            for df_i, dataframe in enumerate(dfs_test):
                dfs_test[df_i] = dataframe.loc[:target_length]
                pad = pandas.DataFrame(0, index=(range(len(dataframe), target_length)), columns=dataframe.columns)
                dfs_test[df_i] = pandas.concat([dataframe, pad])

    return dfs_train, labels_train, dfs_test, labels_test


def load_numpy_arrays(
    path: Union[Path, str],
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool,
) -> Tuple[List[numpy.ndarray], List[dict], List[numpy.ndarray], List[dict]]:
    """Loads the dataset as numpy arrays.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The numpy arrays and labels for each sample.
    """
    x_train, y_train, x_test, y_test = load_pandas_dataframes(
        path=path,
        columns=columns,
        normalize=normalize,
        frequency_divider=frequency_divider,
        train_gain=train_gain,
        pad=pad,
    )

    y_train_arrays = [s.values for s in x_train]
    x_test_arrays = [s.values for s in x_test]

    # return shape of each array: (t, signals)
    return y_train_arrays, y_train, x_test_arrays, y_test


def load_torch_tensors(
    path: Union[Path, str],
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool,
) -> Tuple[List[torch.Tensor], List[dict], List[torch.Tensor], List[dict]]:
    """Loads the dataset as torch tensors.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The tensors and labels for each sample.
    """
    x_train, y_train, x_test, y_test = load_numpy_arrays(
        path=path,
        columns=columns,
        normalize=normalize,
        frequency_divider=frequency_divider,
        train_gain=train_gain,
        pad=pad,
    )
    y_train_arrays = [torch.from_numpy(s).float() for s in x_train]
    x_test_arrays = [torch.from_numpy(s).float() for s in x_test]

    # return shape of each array: (t, signals)
    return y_train_arrays, y_train, x_test_arrays, y_test


class VorausADDataset(Dataset):
    """The voraus-AD dataset torch adapter."""

    def __init__(
        self,
        tensors: List[torch.Tensor],
        labels: List[dict],
        columns: list[str],
    ):
        """Initializes the voraus-AD dataset.

        Args:
            tensors: The tensors for each sample.
            labels: The labels for each sample.
            columns: The colums which are used.
        """
        self.tensors = tensors
        self.labels = labels
        self.columns = columns

        assert len(self.tensors) == len(self.labels), "Can not handle different label and array length."
        self.length = len(self.tensors)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return self.length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, Dict]:
        """Access single dataset samples.

        Args:
            item: The sample index.

        Returns:
            The sample and labels.
        """
        return self.tensors[item], self.labels[item]


# Disable pylint since we need all the arguments here.
def load_torch_dataloaders(  # pylint: disable=too-many-locals
    dataset: Union[Path, str],
    batch_size: int,
    seed: int,
    columns: Union[List[str], Tuple],
    normalize: bool,
    frequency_divider: int,
    train_gain: float,
    pad: bool = True,
) -> tuple[VorausADDataset, VorausADDataset, DataLoader, DataLoader]:
    """Loads the voraus-AD dataset (train and test) as torch data loaders and datasets.

    Args:
        dataset: The path to the dataset.
        batch_size: The batch size to use.
        seed: The seed o use for the dataloader random generator.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The data loaders and datasets.
    """
    x_train, y_train, x_test, y_test = load_torch_tensors(
        path=dataset,
        columns=columns,
        normalize=normalize,
        frequency_divider=frequency_divider,
        train_gain=train_gain,
        pad=pad,
    )

    train_dataset = VorausADDataset(x_train, y_train, list(columns))
    test_dataset = VorausADDataset(x_test, y_test, list(columns))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dataloader, test_dataloader
