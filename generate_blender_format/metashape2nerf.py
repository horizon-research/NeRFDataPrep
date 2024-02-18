#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

	parser.add_argument("--aabb_scale", default=1, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
	parser.add_argument("--parsed_meta", default="parsed_meta.pkl", help="The parsed meta file.")
	parser.add_argument("--json_output_folder", default="", help="json_output_folder")
	parser.add_argument("--img_folder", default="", help="the path to images")
	parser.add_argument("--all", action='store_true', help='Generate all json')
	parser.add_argument('--val', action='store_true', help='Generate val json')
	parser.add_argument('--downscale_factor', type=float, default=1.0, help='Downscale factor for the images')
	args = parser.parse_args()
	return args

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	# convert rgba to rgb
	if image.shape[2] == 4:
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm


if __name__ == "__main__":
	args = parse_args()
	if args.all:
		OUT_PATH = args.json_output_folder + "/transforms_all.json"
	elif args.val:
		OUT_PATH = args.json_output_folder + "/transforms_val.json"
	else:
		OUT_PATH = args.json_output_folder + "/transforms_train.json"

	AABB_SCALE = int(args.aabb_scale)

	# Check that we can save the output before we do a lot of work
	try:
		open(OUT_PATH, "a").close()
	except Exception as e:
		print(f"Could not save transforms JSON to {OUT_PATH}: {e}")
		sys.exit(1)

	print(f"outputting to {OUT_PATH}...")
	cameras = {}
	camera = {}

	with open(args.parsed_meta, "rb") as f:
		in_ex_param = pickle.load(f)

	fl_x = in_ex_param["intrinsics"]["f"] / args.downscale_factor
	fl_y = in_ex_param["intrinsics"]["f"] / args.downscale_factor
	cx = in_ex_param["intrinsics"]["cx"] / args.downscale_factor
	cy = in_ex_param["intrinsics"]["cy"] / args.downscale_factor
	img_width = round(in_ex_param["intrinsics"]["width"] / args.downscale_factor)
	img_height = round(in_ex_param["intrinsics"]["height"] / args.downscale_factor)
	name_poses = in_ex_param["name_poses"]

	camera["camera_angle_x"] = math.atan(img_width / ( fl_x * 2)) * 2
	camera["camera_angle_y"] = math.atan(img_height / (fl_y * 2)) * 2
	camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
	camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi
	camera["fl_x"] = fl_x
	camera["fl_y"] = fl_y
	camera["cx"] = cx
	camera["cy"] = cy
	camera["w"] = img_width
	camera["h"] = img_height
	camera_id = 0
	camera["k1"] = in_ex_param["intrinsics"]["k1"]
	camera["k2"] = in_ex_param["intrinsics"]["k2"]
	camera["k3"] = in_ex_param["intrinsics"]["k3"]
	camera["k4"] = 0
	camera["p1"] = in_ex_param["intrinsics"]["p1"]
	camera["p2"] = in_ex_param["intrinsics"]["p2"]
	camera["is_fisheye"] = False
	print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
	cameras[camera_id] = camera

	if len(cameras) == 0:
		print("No cameras found!")
		sys.exit(1)

	if len(cameras) == 1:
		camera = cameras[camera_id]
		out = {
			"camera_angle_x": camera["camera_angle_x"],
			"camera_angle_y": camera["camera_angle_y"],
			"fl_x": camera["fl_x"],
			"fl_y": camera["fl_y"],
			"k1": camera["k1"],
			"k2": camera["k2"],
			"k3": camera["k3"],
			"k4": camera["k4"],
			"p1": camera["p1"],
			"p2": camera["p2"],
			"is_fisheye": camera["is_fisheye"],
			"cx": camera["cx"],
			"cy": camera["cy"],
			"w": camera["w"],
			"h": camera["h"],
			"aabb_scale": AABB_SCALE,
			"frames": [],
		}
	else:
		out = {
			"frames": [],
			"aabb_scale": AABB_SCALE
		}

	up = np.zeros(3)
	for idx, name_pose in enumerate(name_poses):
		image_rel = os.path.relpath(args.img_folder, os.path.dirname(args.json_output_folder))
		img_name = name_pose["name"]
		pose = name_pose["transform"]
		if not args.all:
			if args.val:
				if idx % 8 != 0:
					continue
			else:
				if idx % 8 == 0:
					continue
		rel_path = str(f"./{image_rel}/"+ img_name + ".png")
		path = str(args.img_folder+ img_name + ".png")
		b = sharpness(path)
		print(path, "sharpness=",b)

		frame = {"file_path":rel_path,"sharpness":b,"transform_matrix": pose}
		out["frames"].append(frame)
	nframes = len(out["frames"])
	

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	print(nframes,"frames")
	print(f"writing {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)
