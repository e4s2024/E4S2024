from options.ui_options import UIOptions
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from ui_run.ui import Ui_Form
from ui_run.mouse_event import GraphicsScene
import cv2
import skimage.io
from ui_run.util import number_color, color_pred,celebAHQ_masks_to_faceParser_mask_detailed, my_number_object, COMPS
import qdarkstyle
import qdarkgraystyle
import os
import numpy as np
import skimage.io
from PIL import Image
import os
import torch
from PyQt5 import QtGui
from models.networks import Net3
from glob import glob
import copy
from utils import torch_utils
from datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE
import torchvision.transforms as transforms

class ExWindow(QMainWindow):
    def __init__(self, opt):
        super().__init__()
        self.EX = Ex(opt)
        self.setWindowIcon(QtGui.QIcon('ui_run/icons/edit_icon.svg'))


class Ex(QWidget, Ui_Form):
    
    @pyqtSlot()
    def change_brush_size(self):  # 改变画刷的 粗细
        self.scene.brush_size = self.brushSlider.value()
        self.brushsizeLabel.setText('Brush size: %d' % self.scene.brush_size)

    @pyqtSlot()
    def change_alpha_value(self):
        self.alpha = self.alphaSlider.value() / 20
        self.alphaLabel.setText('Alpha: %.2f' % self.alpha)

    @pyqtSlot()
    def switch_labels(self, label):  # 换了一种label颜色按钮
        self.scene.label = label
        self.scene.color = number_color[label]
        self.color_Button.setStyleSheet("background-color: %s;" % self.scene.color)


    @pyqtSlot()
    def undo(self):
        self.scene.undo()


    def __init__(self, opt):

        super().__init__()
        self.init_deep_model(opt)

        self.setupUi(self)
        self.show()
        
        # 下面都是一些默认值
        self.modes = 0
        self.alpha = 1  # 插值的alpha

        self.ref_style_img_path = None
        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes, self)  # 用来编辑的 scene
        self.scene.setSceneRect(0, 0, 512, 512)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignCenter)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.GT_scene = QGraphicsScene()
        self.graphicsView_GT.setScene(self.GT_scene)
        self.graphicsView_GT.setAlignment(Qt.AlignCenter)
        self.graphicsView_GT.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_GT.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        self.dlg = QColorDialog(self.graphicsView)

        self.init_screen()  # 初始化screen

    def init_screen(self):
        #self.image = QPixmap(self.graphicsView.size())
        self.image = QPixmap(QSize(512, 512))  # 这张是待编辑的mask可视化图片
        self.image.fill(QColor('#FFFFFF'))
        self.mat_img = np.zeros([512, 512, 3], np.uint8)  # mask图片, [0-12], 3通道


        self.mat_img_org = self.mat_img.copy()

        self.GT_img_path = None
        GT_img = np.ones([512, 512, 3], np.uint8)*255
        self.GT_img = Image.fromarray(GT_img)
        self.GT_img = self.GT_img.convert('RGB')

        #################### add GT image
        self.update_GT_image(GT_img)
        #####################

        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()
        self.scene.addPixmap(self.image)

        ############### load average features
        # TODO: 把这两行注释打开
        # self.load_average_feature()
        # self.run_deep_model()
        self.recorded_img_names = []

        self.clean_snapshots()
        self.clean_generated_result()

    def init_deep_model(self, opt):  # 初始化模型
        self.opt = opt

        assert self.opt.checkpoint_path is not None, "please specify the pre-trained weights!"
        print("Loading model and weights, please wait a few seconds...")

        self.net = Net3(self.opt).eval().to(self.opt.device)
        
        ckpt_dict=torch.load(self.opt.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opt.device) if self.opt.start_from_latent_avg else None
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
        print("Loading Done!")        

        # 固定noise
        channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * 2,
                128: 128 * 2,
                256: 64 * 2,
                512: 32 * 2,
                1024: 16 * 2,
            }
        self.noise = [torch.randn(1,512,4,4).to(self.opt.device)]
        for i in [8,16,32,64,128,256,512,1024]:
            self.noise.append(torch.randn(1,channels[i],i,i).to(self.opt.device))
            self.noise.append(torch.randn(1,channels[i],i,i).to(self.opt.device))

    # ===================================================

    def editing(self):  #  生成编辑的结果
        mat_img_seg12 = celebAHQ_masks_to_faceParser_mask_detailed(self.mat_img[:,:,0]) 
        mask_edit = (TO_TENSOR(mat_img_seg12)*255).long().to(self.opt.device).unsqueeze(0)
        
        # mask_edit = (TO_TENSOR(self.mat_img[:,:,0])*255).long().to(self.opt.device).unsqueeze(0)
        
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot_edit = torch_utils.labelMap2OneHot(mask_edit, num_cls=self.opt.num_seg_cls)
            
        with torch.no_grad():
            style_codes = self.net.cal_style_codes(self.mixed_style_vectors)
            generated, _, _ = self.net.gen_img(torch.zeros(1,512,32,32).to(onehot_edit.device), 
                                                style_codes, onehot_edit,
                                                randomize_noise=False,noise=self.noise)

        # 展示结果
        self.show_generated_result(generated[0])
   

    def mixing_ref_img_style(self):
        # 点击右上角正在编辑的图片
        if self.ref_style_img_path is None:  # TODO: 改变自己的style vectors
            return 

        else:
            # 首先更新一下编辑后mask对应的label count 
            self.label_count = []
            for i in range(self.opt.num_seg_cls):
                if np.sum(self.mat_img[:,:,0]==i) != 0:
                    self.label_count.append(i)  # 存在的component

            self.ref_mat_img_path = os.path.join(self.opt.label_dir, os.path.basename(self.ref_style_img_path)[:-4]+".png")
            # USE CV2 read images, because of using gray scale images, no matter the RGB orders
            ref_mat_img = cv2.imread(self.ref_mat_img_path)
            
            # 转成12个label的格式
            ref_mat_img = celebAHQ_masks_to_faceParser_mask_detailed(ref_mat_img[:,:,0]) 
            
            
            ref_img_path = os.path.join(self.opt.image_dir, os.path.basename(self.ref_style_img_path)[:-4] + '.jpg')
            ref_img = Image.open(ref_img_path).convert('RGB')
            
            # ************************跑模型*****************************
            # 先包装成batch，并放到cuda上去
            img = transforms.Compose([TO_TENSOR, NORMALIZE])(ref_img).to(self.opt.device).float().unsqueeze(0)
            mask = (TO_TENSOR(ref_mat_img)*255).long().to(self.opt.device).unsqueeze(0)
            # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
            onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opt.num_seg_cls)

            # reference image 的style codes
            with torch.no_grad():
                ref_style_vectors, ref_struc_code = self.net.get_style_vectors(img, onehot)
                # ref_style_codes = self.net.cal_style_codes(ref_style_vectors)
            # ************************************************************

            assert self.style_vectors is not None, "No source image was provided!"
            
            for i, cb_status in enumerate(self.checkbox_status):
                if cb_status and i in self.label_count:  # 复选框选中，并且自身也存在对应的label
                    self.mixed_style_vectors[0,i,:] = (1-self.alpha) * self.style_vectors[0,i,:] + self.alpha * ref_style_vectors[0,i,:]
                    self.style_img_mask_dic[my_number_object[i]] = self.ref_style_img_path  # 每个component的style由哪张图片提供
                # else:
                #     self.style_img_mask_dic[my_number_object[i]] = self.GT_img_path  
                    
            # forward 模型得到结果
            mixed_style_codes = self.net.cal_style_codes(self.mixed_style_vectors)
            # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
            onehot = torch_utils.labelMap2OneHot(
                (TO_TENSOR(self.mat_img[:,:,0])*255).long().to(self.opt.device).unsqueeze(0),
                num_cls=self.opt.num_seg_cls
            )
            with torch.no_grad():
                generated, _, _ = self.net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), mixed_style_codes, onehot,
                                                    randomize_noise=False,noise=self.noise)

            # 展示结果
            self.show_generated_result(generated[0])

            self.update_snapshots()

    def load_partial_average_feature(self):  # 这个函数还没实现

        # TODO: calculate average style vectors
        mean_style_vectors = 'XXX'

        mixed_style_vectors = copy.deepcopy(self.style_vectors)

        for i, cb_status in enumerate(self.checkbox_status):
            if cb_status:
                mixed_style_vectors[0,i,:] =  mean_style_vectors[0,i,:]

                if my_number_object[i] in self.style_img_mask_dic:
                    del self.style_img_mask_dic[my_number_object[i]]


        # forward 模型得到结果
        mixed_style_codes = self.net.cal_style_codes(mixed_style_vectors)
        onehot = torch_utils.labelMap2OneHot(self.mat_img, num_cls=self.opt.num_seg_cls)
        generated = self.net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), mixed_style_codes, onehot,
                                        randomize_noise=False,noise=self.noise)

        # 展示结果
        self.show_generated_result(generated)

        self.update_snapshots()

    def show_generated_result(self,generated):
        """假定输入的是网络生成的tensor格式数据"""
        generated_img = np.array(torch_utils.tensor2im(generated)) # np array
        qim = QImage(generated_img.data, generated_img.shape[1], generated_img.shape[0], QImage.Format_RGB888)
        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512,512),transformMode=Qt.SmoothTransformation))
        self.generated_img = generated_img

    # def show_reference_image(self, im_name):
    #     qim = QImage(im_name).scaled(QSize(256, 256),transformMode=Qt.SmoothTransformation)
    #     # self.referDialogImage.setPixmap(QPixmap.fromImage(qim).scaled(QSize(512, 512), transformMode=Qt.SmoothTransformation))
    #     # # self.referDialog.setWindowTitle('Input:' + os.path.basename(self.GT_img_path) + '\t \t Reference:' + os.path.basename(im_name))
    #     # self.referDialog.show()
    #     self.GT_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512, 512), transformMode=Qt.SmoothTransformation))

    def compNames2Indices(self,comp_names=[]):
        return [COMPS.index(name) for name in comp_names]

    

    def update_snapshots(self):
        self.clean_snapshots()
        self.recorded_img_names = list(set(self.style_img_mask_dic.values())) # 由哪些图片的style 共同组成
        self.recorded_mask_dic = {}

        tmp_count = 0

        for i, name in enumerate(self.recorded_img_names):
            self.recorded_mask_dic[name] = [comp_name for comp_name in self.style_img_mask_dic if self.style_img_mask_dic[comp_name]==name]

            gray_mask = skimage.io.imread(os.path.join(self.opt.label_dir, os.path.basename(name)[:-4] + '.png'))
            # 转成12个label种类的格式
            gray_mask_12seg = celebAHQ_masks_to_faceParser_mask_detailed(gray_mask) 
        
            rgb_mask = color_pred(gray_mask_12seg)
        
            mask_snap = np.where(np.isin(np.repeat(np.expand_dims(gray_mask_12seg,2),3, axis=2),
                                        self.compNames2Indices(self.recorded_mask_dic[name])),
                                rgb_mask, 255)


            if not (mask_snap==255).all():  # 不全是255，即存在某个类别的mask
                self.mask_snap_style_button_list[tmp_count].setIcon(QIcon(
                        QPixmap.fromImage(QImage(mask_snap.data, mask_snap.shape[1], mask_snap.shape[0], mask_snap.strides[0], QImage.Format_RGB888)))
                    )

                self.snap_style_button_list[tmp_count].setIcon(QIcon(name))  # 展示这张图片来源的mask
                tmp_count += 1

    def update_GT_image(self, GT_img):  
        qim = QImage(GT_img.data, GT_img.shape[1], GT_img.shape[0], GT_img.strides[0],
                     QImage.Format_RGB888)
        qim = qim.scaled(QSize(256, 256), Qt.IgnoreAspectRatio, transformMode=Qt.SmoothTransformation)
        if len(self.GT_scene.items()) > 0:
            self.GT_scene.removeItem(self.GT_scene.items()[-1])
        self.GT_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512, 512),transformMode=Qt.SmoothTransformation))
    
    def clean_generated_result(self):
        """清空生成的结果"""
        dummy_img = np.ones([512, 512, 3], np.uint8)*255 # np array
        qim = QImage(dummy_img.data, dummy_img.shape[1], dummy_img.shape[0], QImage.Format_RGB888)
        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512,512),transformMode=Qt.SmoothTransformation))

    def clean_snapshots(self):
        for snap_style_button in self.snap_style_button_list:
            snap_style_button.setIcon(QIcon())
        for mask_snap_style_button in self.mask_snap_style_button_list:
            mask_snap_style_button.setIcon(QIcon())

    def open_snapshot_dialog(self, i):
        if i < len(self.recorded_img_names):
            im_name = self.recorded_img_names[i]
            qim = QImage(im_name).scaled(QSize(256, 256), transformMode=Qt.SmoothTransformation)
            self.snapshotDialogImage.setPixmap(
                QPixmap.fromImage(qim).scaled(QSize(512, 512), transformMode=Qt.SmoothTransformation))
            self.snapshotDialog.setWindowTitle('Reference:' + os.path.basename(im_name))
            self.snapshotDialog.show()
            self.snapshotDialog.count = i
        else:
            self.snapshotDialog.setWindowTitle('Reference:')
            self.snapshotDialogImage.setPixmap(QPixmap())
            self.snapshotDialog.show()
            self.snapshotDialog.count = i   

    def set_ref_img_path(self,style_img_path):
        self.ref_style_img_path = style_img_path

        self.ref_img_button.setIcon(QIcon(self.ref_style_img_path))  # 右下角图片
        self.opsLogTextBox.appendPlainText("Reference Image: %s"%self.ref_style_img_path)

    @pyqtSlot()
    def open(self):  # 打开文件逻辑

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath() +
                                                  '/ui_run/edit_comp/CelebA-HQ/test/vis')
                                                  # '/imgs/colormaps')
        if fileName:
            
            self.clean_snapshots()
            self.clean_generated_result() 

            self.mat_img_path = os.path.join(self.opt.label_dir, os.path.basename(fileName))
            # USE CV2 read images, because of using gray scale images, no matter the RGB orders
            mat_img = cv2.imread(self.mat_img_path)
            
            # 转成12个label的格式
            # mat_img = celebAHQ_masks_to_faceParser_mask_detailed(mat_img[:,:,0]) 
            
            # 直接使用19个类
            mat_img = mat_img[:,:,0]
            
            # mask 的可视化图片
            mat_img_vis = color_pred(mat_img)
            # image = QPixmap(fileName)
            image = QImage(mat_img_vis.data, mat_img_vis.shape[1], mat_img_vis.shape[0], mat_img_vis.strides[0],QImage.Format_RGB888)
            image = QPixmap(image)
            # image = Image.fromarray(mat_img_vis).toqpixmap()
            # self.image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.image = image.scaled(QSize(512, 512), Qt.IgnoreAspectRatio)

            self.mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)  # 这张是label map（0-11）
            # 再将12个类别的mask转成3通道
            self.mat_img = np.stack((self.mat_img,)*3, axis=-1)
            self.mat_img_org = self.mat_img.copy() # 原始的 label map

            self.GT_img_path = os.path.join(self.opt.image_dir, os.path.basename(fileName)[:-4] + '.jpg')
            GT_img = skimage.io.imread(self.GT_img_path)
            self.GT_img = Image.fromarray(GT_img)
            self.GT_img = self.GT_img.convert('RGB')

            self.input_img_button.setIcon(QIcon(self.GT_img_path))  # 右上角图片

            #################### add GT image
            self.update_GT_image(GT_img)
            #####################

            self.scene.reset()
            if len(self.scene.items()) > 0:  # 先清空3张图片
                self.scene.reset_items()
            self.scene.addPixmap(self.image)


    def recon(self):
        # ************************跑模型*****************************
        # 先包装成batch，并放到cuda上去
        img = transforms.Compose([TO_TENSOR, NORMALIZE])(self.GT_img).to(self.opt.device).float().unsqueeze(0)
        mat_img_seg12 = celebAHQ_masks_to_faceParser_mask_detailed(self.mat_img[:,:,0]) 
        mask = (TO_TENSOR(mat_img_seg12)*255).long().to(self.opt.device).unsqueeze(0)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opt.num_seg_cls)
        
        # source image 的style codes
        with torch.no_grad():
            self.style_vectors, struc_code = self.net.get_style_vectors(img, onehot)
            self.style_codes = self.net.cal_style_codes(self.style_vectors)

            generated, _, _ = self.net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), 
                                                self.style_codes, onehot,
                                                randomize_noise=False,noise=self.noise)

        # 展示结果
        self.show_generated_result(generated[0])

        # 为了后续的增量式编辑需要
        self.mixed_style_vectors = copy.deepcopy(self.style_vectors)

        # print("Load input image %s done !"% os.path.basename(fileName)[:-4])
        # print(self.style_vectors.size(), self.style_codes.size())
        
        # *****************************************************

        self.style_img_mask_dic = {}
        self.label_count = [] 
        for i in range(self.opt.num_seg_cls):
            if np.sum(self.mat_img[:,:,0]==i) != 0:
                self.style_img_mask_dic[my_number_object[i]] = self.GT_img_path  # 每个component的style由哪张图片提供
                self.label_count.append(i)  # 存在的component
        # 更新一下底部的 snapshot
        # self.update_snapshots()

    @pyqtSlot()
    def open2(self):  # 打开文件逻辑,只是为了编辑mask而设计的逻辑

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath() +
                                                #   '/ui_run/mini_testset/CelebA-HQ/test/vis')
                                                  '/our_swapping_dataset/faceVid2Vid_GPEN_driven2')
        if fileName:
            
            self.clean_snapshots()
            self.black_generated_result()

            self.mat_img_path = fileName
            # USE CV2 read images, because of using gray scale images, no matter the RGB orders
            mat_img = cv2.imread(self.mat_img_path)[:,:,0]
            
            # # 转成12个label的格式
            # mat_img = celebAHQ_masks_to_faceParser_mask_detailed(mat_img[:,:,0]) 
            
            # mask 的可视化图片
            mat_img_vis = color_pred(mat_img)
            # image = QPixmap(fileName)
            image = QImage(mat_img_vis.data, mat_img_vis.shape[1], mat_img_vis.shape[0], mat_img_vis.strides[0],QImage.Format_RGB888)
            image = QPixmap(image)
            # image = Image.fromarray(mat_img_vis).toqpixmap()
            # self.image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.image = image.scaled(QSize(512, 512), Qt.IgnoreAspectRatio)

            self.mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)  # 这张是label map（0-11）
            # 再将12个类别的mask转成3通道
            self.mat_img = np.stack((self.mat_img,)*3, axis=-1)
            self.mat_img_org = self.mat_img.copy() # 原始的 label map

            # self.GT_img_path = os.path.join(self.opt.image_dir, os.path.basename(fileName)[:-4] + '.jpg')
            # GT_img = skimage.io.imread(self.GT_img_path)
            # self.GT_img = Image.fromarray(GT_img)
            # self.GT_img = self.GT_img.convert('RGB')

            # self.input_img_button.setIcon(QIcon(self.GT_img_path))  # 右上角图片

            #################### add GT image
            # self.update_GT_image(GT_img)
            #####################

            self.scene.reset()
            if len(self.scene.items()) > 0:  # 先清空3张图片
                self.scene.reset_items()
            self.scene.addPixmap(self.image)

            # # ************************跑模型*****************************
            # # 先包装成batch，并放到cuda上去
            # img = transforms.Compose([TO_TENSOR, NORMALIZE])(self.GT_img).to(self.opt.device).float().unsqueeze(0)
            # mask = (TO_TENSOR(self.mat_img[:,:,0])*255).long().to(self.opt.device).unsqueeze(0)
            # # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
            # onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opt.num_seg_cls)
           
            # # source image 的style codes
            # with torch.no_grad():
            #     self.style_vectors, struc_code = self.net.get_style_vectors(img, onehot)
            #     self.style_codes = self.net.cal_style_codes(self.style_vectors)

            # # 为了后续的增量式编辑需要
            # self.mixed_style_vectors = copy.deepcopy(self.style_vectors)

            # # print("Load input image %s done !"% os.path.basename(fileName)[:-4])
            # print(self.style_vectors.size(), self.style_codes.size())
            
            # # *****************************************************

            self.style_img_mask_dic = {}
            self.label_count = [] 
            for i in range(self.opt.num_seg_cls):
                if np.sum(self.mat_img[:,:,0]==i) != 0:
                    self.style_img_mask_dic[my_number_object[i]] = self.GT_img_path  # 每个component的style由哪张图片提供
                    self.label_count.append(i)  # 存在的component
            # 更新一下底部的 snapshot
            # self.update_snapshots()

    @pyqtSlot()
    def mode_select(self, mode):  # 选择不同的模式
        self.modes = mode
        self.scene.modes = mode

        if mode == 0:  # 画笔模式
            self.brushButton.setStyleSheet("background-color: #85adad")
            self.recButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 1: # 矩形模式模式
            self.recButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 2: # 画刷模式
            self.fillButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.recButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.PointingHandCursor)

    @pyqtSlot()
    def save_img(self): # 保存编辑后的图片

        ui_result_folder = 'edit_methods_comp_mask'
        os.makedirs(ui_result_folder,exist_ok=True)
        
        # skimage.io.imsave(os.path.join(ui_result_folder, str(current_time) +'_G_img.png'), self.generated_img)
        # skimage.io.imsave(os.path.join(ui_result_folder, str(current_time) +'_mask.png'), self.mat_img[:, :, 0])
        # skimage.io.imsave(os.path.join(ui_result_folder, str(current_time) +'_ColorMask.png'), color_pred(self.mat_img[:, :, 0]))
        skimage.io.imsave(os.path.join(ui_result_folder, os.path.basename(self.mat_img_path)[:-4] +'_G_img.png'), self.generated_img)
        skimage.io.imsave(os.path.join(ui_result_folder, os.path.basename(self.mat_img_path)[:-4] + '_mask.png'), self.mat_img[:, :, 0])
        skimage.io.imsave(os.path.join(ui_result_folder, os.path.basename(self.mat_img_path)[:-4] + '_ColorMask.png'),color_pred(self.mat_img[:, :, 0]))

    def load_average_feature(self):

        # TODO
        pass

    # ===================================================


if __name__ == '__main__':
    opt = UIOptions().parse()
    opt.status = 'UI_mode'

    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    # app.setStyleSheet(qdarkstyle.load_stylesheet_PyQt5())
    ex = ExWindow(opt)
    # ex = Ex(opt)
    sys.exit(app.exec_())