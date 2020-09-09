import os
import shutil
import sys
af = open('api_key.txt','r')
api_key = af.read()
api_key = api_key.strip()
af.close()

#os.system('python3 ../deep-license-plate-recognition/plate_recognition.py --api-key d4f0e1367ffeada3c3480ed3efe549f95bcf240c cropped_images/8.jpg')


for f in os.listdir('cropped_images'):
    cmd = 'python3 ../deep-license-plate-recognition/plate_recognition.py --api-key {} {}'.format(api_key,os.path.join('cropped_images',f))
    rv = os.system(cmd)
    if rv != 0:
        raise SystemError
        break
    f1 = open('lplate.txt','r')
    lplate = f1.read()
    f1.close()
    if 'unknown' in lplate:
        os.remove(os.path.join('cropped_images',f))
    
    else:
        shutil.move(os.path.join('cropped_images',f),os.path.join('violations',lplate.strip()+'.jpg'))

