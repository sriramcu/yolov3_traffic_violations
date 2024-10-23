import os
import shutil

from deep_license_plate_recognition.plate_recognition import recognition_api
from helmet_violation_monitoring_gui import VIOLATIONS_IMAGES_DIR
from keras_yolo3.yolo import CROPPED_IMAGES_DIRECTORY


def main():
    af = open('api_key.txt','r')
    api_key = af.read()
    api_key = api_key.strip()
    af.close()
    run_lpr(api_key)


def run_lpr(api_key: str):
    for cropped_image in os.listdir(CROPPED_IMAGES_DIRECTORY):
        if "unknown" in cropped_image.lower():
            continue
        cropped_image_path = os.path.join(CROPPED_IMAGES_DIRECTORY, cropped_image)
        api_res = recognition_api(cropped_image_path, api_key=api_key)
        license_plate_str = api_res['results'][0]['plate']
        filename, file_extension = os.path.splitext(cropped_image)
        if 'unknown' in license_plate_str:
            shutil.move(cropped_image_path, os.path.join(CROPPED_IMAGES_DIRECTORY, filename + "_unknown" + file_extension))
        else:
            shutil.move(cropped_image_path, os.path.join(VIOLATIONS_IMAGES_DIR, license_plate_str.strip() + file_extension))


if __name__ == "__main__":
    main()
