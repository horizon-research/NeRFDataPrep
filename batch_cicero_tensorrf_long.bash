export scene=ignatius
export dsf=2
# gardne with cicero_tensorrf 6
python3 warping_evaluation.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_long/imgs_test_all \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 6 &
# gardne with cicero_tensorrf 16
python3 warping_evaluation.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_long/imgs_test_all \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 &

# gardne with cicero_tensorrf 30
python3 warping_evaluation.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_long/imgs_test_all \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 30 &

# gardne with cicero_tensorrf 60
python3 warping_evaluation.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_long/imgs_test_all \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 60 &
# gardne with cicero_tensorrf 180
python3 warping_evaluation.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_long/imgs_test_all \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 180 &
# gardne with cicero_tensorrf 480
python3 warping_evaluation.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_long/imgs_test_all \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 480 &