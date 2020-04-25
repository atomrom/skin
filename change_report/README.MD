usage: generate_change_report.py [-h] [--old_image OLD_IMAGE]
                                 [--new_image NEW_IMAGE] [--smodel SMODEL]
                                 [--cmodel CMODEL] [--target_root TARGET_ROOT]

optional arguments:
  -h, --help            show this help message and exit
  --old_image OLD_IMAGE
                        Path to the old image.
  --new_image NEW_IMAGE
                        Path to the old image.
  --smodel SMODEL       Path to the rcnn model.
  --cmodel CMODEL       Path to the convolutional model.
  --target_root TARGET_ROOT
                        Report dir root.


Example run:
python generate_change_report.py --old_image old.jpg --new_image new.jpg --smodel maskrcnn.h5 --target_root c:\temp

Note that to run the script you need to have a maskrcnn network trained to segment skin lesions.