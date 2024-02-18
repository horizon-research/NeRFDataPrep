aabb_scale=$1
parsed_meta=$2
json_output_folder=$3
img_folder=$4
downscale_factor=$5
# generate training and validation data
python3 metashape2nerf.py --aabb_scale "$aabb_scale" --parsed_meta "$parsed_meta" --json_output_folder "$json_output_folder" --img_folder "$img_folder" --downscale_factor "$downscale_factor" --all

cp $json_output_folder/transforms_all.json $json_output_folder/transforms_train.json
cp $json_output_folder/transforms_all.json $json_output_folder/transforms_val.json
cp $json_output_folder/transforms_all.json $json_output_folder/transforms_test.json

