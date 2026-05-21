#!/bin/bash
# SG-DP3 Data Processing
# Usage: bash process_data.sh <task_name> <task_config> <expert_data_num>
# Example: bash process_data.sh beat_block_hammer demo_clean 50

task_name=${1}
task_config=${2}
expert_data_num=${3}

python scripts/process_data.py $task_name $task_config $expert_data_num
    
    # 下采样点云到目标点数
    if point_clouds.shape[1] > num_points:
        indices = np.random.choice(point_clouds.shape[1], num_points, replace=False)
        point_clouds = point_clouds[:, indices]
    elif point_clouds.shape[1] < num_points:
        # 有放回重采样
        indices = np.random.choice(point_clouds.shape[1], num_points, replace=True)
        point_clouds = point_clouds[:, indices]
    
    all_actions.append(actions)
    all_states.append(states)
    all_point_clouds.append(point_clouds)
    
    # 处理图像 (如果存在)
    img_dir = os.path.join(ep_path, 'images')
    if os.path.exists(img_dir):
        from PIL import Image
        imgs = []
        for img_file in sorted(os.listdir(img_dir)):
            img = Image.open(os.path.join(img_dir, img_file)).resize((${IMAGE_SIZE}, ${IMAGE_SIZE}))
            imgs.append(np.array(img).transpose(2, 0, 1))  # HWC -> CHW
        all_images.append(np.stack(imgs))
    
    episode_ends.append(sum(a.shape[0] for a in all_actions))

# 合并并保存为 zarr
root = zarr.open(output_dir, mode='w')
root.create_dataset('action', data=np.concatenate(all_actions))
root.create_dataset('state', data=np.concatenate(all_states))
root.create_dataset('point_cloud', data=np.concatenate(all_point_clouds))
root.create_dataset('episode_ends', data=np.array(episode_ends))

if all_images:
    root.create_dataset('image', data=np.concatenate(all_images))

print(f'[INFO] Saved to {output_dir}')
print(f'[INFO] Total frames: {episode_ends[-1]}')
print(f'[INFO] Episodes: {len(episode_dirs)}')
"

echo "[DONE] Data processing complete!"
echo "[OUTPUT] ${OUTPUT_DIR}"
