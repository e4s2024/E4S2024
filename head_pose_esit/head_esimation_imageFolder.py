import sys, os, argparse
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import dlib
from tqdm import tqdm

from head_pose_esit import hopenet,utils, datasets

join = os.path.join

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation for images using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use. Default: 0',
                        default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot. Default: hopenet_robust_alpha1.pkl',
                        default='/apdcephfs/share_1290939/zhianliu/pretrained_models/Hopenet/hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model. Default: mmod_human_face_detector.dat',
                        default='./head_pose_esit/mmod_human_face_detector.dat', type=str)
    parser.add_argument('-i', '--input folder', dest='input_path', help='Path of image folder',
                        default='/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ-ourSwap500/MegaFS', type=str)
    parser.add_argument('-o', '--output_txt', dest='output', help='Output path of txt file. Default: output/celeba.txt. \nNote: you must write output in this format',
                        default='/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ-ourSwap500/MegaFS_headPose.txt', type=str)
    parser.add_argument('-f', '--flag', dest='flag', help='1: write the images; 0: do not write the images. Default: 1',
                        default='1', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.enabled = True
    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    input_path = args.input_path

    out_dir = os.path.split(args.output)[0]
    name = os.path.split(args.output)[1]
    
    if args.flag == 1:
        write_path = join(out_dir, "images_" + name[:-4])
        if not os.path.exists(write_path):
            os.makedirs(write_path)

    if not os.path.exists(args.input_path):
        sys.exit('Folder does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # -------------- for image operation ------------------
    images = os.listdir(input_path)
    images = [_ for _ in images if (_.endswith('jpg') or _.endswith('png'))]
    images.sort()

    txt_out = open(args.output, 'w')

    for image_name in tqdm(images,total=len(images)):
        image = cv2.imread(join(input_path, image_name))

        cv2_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # cv2_frame = cv2.resize(cv2_frame, (224, 224))

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width // 4
                x_max += 2 * bbox_width // 4
                y_min -= 3 * bbox_height // 4
                y_max += bbox_height // 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(image.shape[1], x_max); y_max = min(image.shape[0], y_max)
                # Crop image
                img = cv2_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)

                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                # Print new frame with cube and axis
                txt_out.write(str(image_name) + '\t%f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                print(str(image_name) + '\t%f %f %f' % (yaw_predicted, pitch_predicted, roll_predicted))
                
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                if args.flag == 1:
                    drawed_img = utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx =(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size =bbox_height / 2)

                    # write the images

                    cv2.imwrite(join(write_path, image_name), drawed_img)




if __name__ == '__main__':
    main()