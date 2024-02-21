aabb_scale=$1
parsed_meta=$2
json_output_folder=$3
img_folder=$4
downscale_factor=$5
# generate training and validation data
python3 metashape2nerf_long.py --aabb_scale "$aabb_scale" --parsed_meta "$parsed_meta" --json_output_folder "$json_output_folder" --img_folder "$img_folder" --downscale_factor "$downscale_factor"
python3 metashape2nerf_long.py --aabb_scale "$aabb_scale" --parsed_meta "$parsed_meta" --json_output_folder "$json_output_folder" --img_folder "$img_folder" --downscale_factor "$downscale_factor" --val
# copy the validation data to test data in case some methods need it
cp $json_output_folder/transforms_val.json $json_output_folder/transforms_test.json
