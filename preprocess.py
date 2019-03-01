import numpy as np
import cv2

def create_image_dataset(videofile, outpath):
    cap = cv2.VideoCapture(videofile)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(outpath+'/test_'+str(count)+'.png', frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
        if count % 1000 == 0:
            print(count)
    cap.release()
    cv2.destroyAllWindows()

# train_file = "data/train.mp4"
# train_path = 'data/train'
# create_image_dataset(train_file, train_path)

test_file = "data/test.mp4"
test_path = 'data/test'
create_image_dataset(test_file, test_path)

