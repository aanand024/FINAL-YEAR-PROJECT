'''Helper script to label images with MiVOLO. To be used alongside the other code from the authors of MiVOLO.'''
import subprocess
import os
from glob import glob
import csv
import re

IMAGE_ROOT = "MiVOLO/sd3_label_images"   
OUT_CSV = "2mivolo_results.csv"

DETECTOR = "MiVOLO/models/yolov8x_person_face.pt"
CHECKPOINT = "MiVOLO/models/mivolo_imbd.pth.tar"
DEVICE = "cpu"  

rows = []
image_paths = glob(os.path.join(IMAGE_ROOT, "**/*.png"), recursive=True)

print(f"Found {len(image_paths)} images")

for img_path in image_paths:

    out_img = "tmp_out.jpg"

    cmd = [
        "python", "MiVOLO/demo.py",
        "--input", img_path,
        "--output", out_img,
        "--detector-weights", DETECTOR,
        "--checkpoint", CHECKPOINT,
        "--device", DEVICE
    ]
    
    result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )

    output = (result.stdout + result.stderr).lower()
    print(output)

    if "gender" not in output:
        rows.append([img_path, "unknown", 0.0])
        continue

    gender = None
    conf = 0.0

    for line in output.splitlines():
        print(line)
        
        
        if "gender:" in line:
            if re.search(r"\bfemale\b", line):
                gender = "female"
            elif re.search(r"\bmale\b", line):
                gender = "male"
            else:
                gender = line



            m = re.search(r"\[(\d+)%\]", line)
            if m:
                conf = int(m.group(1)) / 100.0

    rows.append([img_path, gender or "unknown", conf, line])
    print(img_path, gender, conf)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "gender", "confidence", "line"])
    w.writerows(rows)

print("Saved results to", OUT_CSV)
