"""
SG-DP3 Data Processing Script.

将 RoboTwin 采集的 HDF5 原始数据转换为 SG-DP3 训练用的 zarr 格式。

参考: policy/DP3/scripts/process_data.py

输入: data/{task_name}/{task_config}/data/episode{i}.hdf5
输出: policy/Pi0_Dp3/data/{task_name}/{task_config}/{task_name}-{task_config}-{num}.zarr
      包含: point_cloud, state, action, episode_ends (与 DP3 兼容)

使用方式:
    python scripts/process_data.py <task_name> <task_config> <expert_data_num>
    例:   python scripts/process_data.py beat_block_hammer demo_clean 50
"""

import os
import sys
import argparse
import shutil

import numpy as np
import zarr
import h5py


def load_hdf5(dataset_path):
    """
    加载单个 episode 的 HDF5 数据文件。

    与 DP3 的 load_hdf5 保持一致，额外加载图像数据 (SG-DP3 需要)。
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        vector = root["/joint_action/vector"][()]
        pointcloud = root["/pointcloud"][()]

        # 图像数据 (SG-DP3 额外需要)
        image_dict = {}
        if "/observation" in root:
            for cam_name in root["/observation"].keys():
                if "rgb" in root[f"/observation/{cam_name}"]:
                    image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud, image_dict


def main():
    parser = argparse.ArgumentParser(description="Process HDF5 episodes into zarr for SG-DP3 training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name (e.g., beat_block_hammer)",
    )
    parser.add_argument(
        "task_config",
        type=str,
        help="Task setting (e.g., demo_clean, demo_randomized)",
    )
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    # 原始数据路径: 相对于脚本所在目录的 ../../data/{task_name}/{task_config}
    # 即项目根目录下的 data/{task_name}/{task_config}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_dir = os.path.dirname(script_dir)  # policy/Pi0_Dp3/
    project_root = os.path.dirname(os.path.dirname(policy_dir))  # 项目根目录

    load_dir = os.path.join(project_root, "data", str(task_name), str(task_config))

    # 输出路径: policy/Pi0_Dp3/data/{task_name}/{task_config}/{task_name}-{task_config}-{num}.zarr
    task_data_dir = os.path.join(policy_dir, "data", str(task_name), str(task_config))
    os.makedirs(task_data_dir, exist_ok=True)
    save_dir = os.path.join(task_data_dir, f"{task_name}-{task_config}-{num}.zarr")

    print(f"[INFO] Loading from: {load_dir}")
    print(f"[INFO] Saving to:    {save_dir}")

    if not os.path.isdir(load_dir):
        print(f"[ERROR] Data directory not found: {load_dir}")
        exit(1)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    total_count = 0
    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays = []
    action_arrays = []
    state_arrays = []
    # 图像数据列表 (按相机分组)
    has_images = False
    image_arrays_dict = {}

    while current_ep < num:
        hdf5_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        if not os.path.isfile(hdf5_path):
            print(f"\n[WARN] Episode file not found: {hdf5_path}, skipping.")
            current_ep += 1
            continue

        print(f"Processing episode: {current_ep + 1} / {num}", end="\r")

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            pointcloud_all,
            image_dict,
        ) = load_hdf5(hdf5_path)

        # 兼容无点云数据集 (如 demo_clean): 当点云为空时，用零张量填充
        pointcloud_is_empty = (pointcloud_all.shape[1] == 0)
        if pointcloud_is_empty:
            num_steps = pointcloud_all.shape[0]
            # 根据配置决定点云维度 (3=XYZ, 6=XYZRGB)
            pc_dim = 3
            pointcloud_all = np.zeros((num_steps, 1024, pc_dim), dtype=np.float32)
            if current_ep == 0:
                print(f"\n[INFO] 检测到空点云数据集 (shape 第二维为 0)，将使用零张量 (1024, {pc_dim}) 填充")
                print(f"        这是正常行为，适用于不包含点云的采集配置 (如 demo_clean)")

        for j in range(left_gripper_all.shape[0]):
            pointcloud = pointcloud_all[j]
            joint_state = vector_all[j]

            if j != left_gripper_all.shape[0] - 1:
                point_cloud_arrays.append(pointcloud)
                state_arrays.append(joint_state)

                # 图像
                for cam_name, cam_data in image_dict.items():
                    if cam_name not in image_arrays_dict:
                        image_arrays_dict[cam_name] = []
                    image_arrays_dict[cam_name].append(cam_data[j])
                    has_images = True

            if j != 0:
                action_arrays.append(joint_state)

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()

    if total_count == 0:
        print("[ERROR] 没有处理任何有效数据。")
        print(f"        请检查 data/{task_name}/{task_config}/data/ 目录下是否存在有效的 episode HDF5 文件")
        exit(1)

    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.array(state_arrays)
        point_cloud_arrays = np.array(point_cloud_arrays)
        action_arrays = np.array(action_arrays)

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        action_chunk_size = (100, action_arrays.shape[1])
        point_cloud_chunk_size = (100,) + point_cloud_arrays.shape[1:]

        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=action_arrays,
            chunks=action_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )

        # 保存图像数据 (如果有)
        if has_images:
            image_group = zarr_data.create_group("images")
            for cam_name, cam_images in image_arrays_dict.items():
                cam_arrays = np.array(cam_images)
                cam_chunk = (100,) + cam_arrays.shape[1:]
                image_group.create_dataset(
                    cam_name,
                    data=cam_arrays,
                    chunks=cam_chunk,
                    overwrite=True,
                    compressor=compressor,
                )
            print(f"[INFO] Images saved for cameras: {list(image_arrays_dict.keys())}")

        print(f"[DONE] Processed {num} episodes, {total_count} transitions.")
        print(f"[DONE] Data saved to: {save_dir}")
        print(f"[INFO]   point_cloud: {point_cloud_arrays.shape}")
        print(f"[INFO]   state:       {state_arrays.shape}")
        print(f"[INFO]   action:      {action_arrays.shape}")

    except ZeroDivisionError:
        print("[ERROR] ZeroDivisionError: check that `data/pointcloud` in the task config is set to true.")
        raise
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
