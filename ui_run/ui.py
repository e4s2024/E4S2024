from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ui_run.util import number_color
from functools import partial
import glob
from ui_run.util import my_number_object
from ui_run.mouse_event import ReferenceDialog, SnapshotDialog
import copy

SCALE = int(1.0)

Lb_width = int(100 * SCALE)
Lb_height = int(40 * SCALE)
Lb_row_shift = int(25 * SCALE)
Lb_col_shift = int(5 * SCALE)
Lb_x = int(100 * SCALE)
Lb_y = int(690 * SCALE)

Tb_width = int(100 * SCALE)
Tb_height = int(40 * SCALE)
Tb_row_shift = int(50 * SCALE)
Tb_col_shift = int(5 * SCALE)
Tb_x = int(100 * SCALE)
Tb_y = int(60 * SCALE)

_translate = QtCore.QCoreApplication.translate


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        # Form.resize(1920, 1080)

        Form.resize(int(1920* SCALE), int(1080 * SCALE))

        # Form.resize(1980, 1100)


        # Label Buttons to change the semantic meanings of the Brush
        # First Row
        self.add_brush_widgets(Form)
        self.add_top_buttons(Form)
        self.add_label_buttons(Form)
        # self.add_label_buttons_seg19(Form)
        self.add_tool_buttons(Form)
        self.add_checkbox_widgets(Form)
        self.add_input_img_button(Form)
        self.add_ops_log_textBox(Form)
        self.add_ref_img_button(Form)

        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(652* SCALE, 140* SCALE, 518* SCALE, 518* SCALE))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(1204* SCALE, 140* SCALE, 518* SCALE, 518* SCALE))
        self.graphicsView_2.setObjectName("graphicsView_2")

        self.graphicsView_GT = QtWidgets.QGraphicsView(Form)
        self.graphicsView_GT.setGeometry(QtCore.QRect(100* SCALE, 140* SCALE, 518* SCALE, 518* SCALE))
        self.graphicsView_GT.setObjectName("graphicsView_GT")


        self.referDialog = ReferenceDialog(self)
        self.referDialog.setObjectName('Reference Dialog')
        # self.referDialog.setWindowTitle('Reference Image:')
        self.referDialog.setWindowTitle('Style Image')
        self.referDialogImage = QtWidgets.QLabel(self.referDialog)
        self.referDialogImage.setFixedSize(512, 512)
        # self.referDialog.show()

        self.snapshotDialog = SnapshotDialog(self)
        self.snapshotDialog.setObjectName('Snapshot Dialog')
        self.snapshotDialog.setWindowTitle('Reference Image:')
        self.snapshotDialogImage = QtWidgets.QLabel(self.snapshotDialog)
        self.snapshotDialogImage.setFixedSize(512, 512)

        self.add_intermediate_results_button(Form)
        self.add_alpha_bar(Form)

        QtCore.QMetaObject.connectSlotsByName(Form)  # 绑定 信号和槽

    def retranslateUi(self, Form):
        # Form.setWindowTitle(_translate("Form", "Let's Party Face Manipulation v0.2"))
        Form.setWindowTitle(_translate("Form", "Inteactive Editing"))
        self.pushButton.setText(_translate("Form", "Open Image"))
        self.pushButton_2.setText(_translate("Form", "Edit Style"))
        self.pushButton_3.setText(_translate("Form", "Edit Shape"))
        self.pushButton_4.setText(_translate("Form", "Recon"))

        self.saveImg.setText(_translate("Form", "Save Img"))

    def add_alpha_bar(self, Form):  # alpha value，控制插值的程度应该是


        self.alphaLabel = QtWidgets.QLabel(Form)
        self.alphaLabel.setObjectName("alphaLabel")
        self.alphaLabel.setGeometry(QtCore.QRect(Lb_x + 10*SCALE * Lb_row_shift + 10*SCALE * Lb_width + 40*SCALE, Lb_y, 150*SCALE, 20*SCALE))
        self.alphaLabel.setText('Alpha: 1.0')
        font = self.brushsizeLabel.font()
        font.setPointSize(10)
        font.setBold(True)
        self.alphaLabel.setFont(font)

        self.alphaSlider = QtWidgets.QSlider(Form)
        self.alphaSlider.setOrientation(QtCore.Qt.Horizontal)
        self.alphaSlider.setGeometry(QtCore.QRect(Lb_x + 10*SCALE * Lb_row_shift + 10 *SCALE* Lb_width + 150*SCALE, Lb_y, 225*SCALE, 10*SCALE))
        self.alphaSlider.setObjectName("alphaSlider")
        self.alphaSlider.setMinimum(0)
        self.alphaSlider.setMaximum(20)
        self.alphaSlider.setValue(20)
        self.alphaSlider.valueChanged.connect(Form.change_alpha_value)

    def add_intermediate_results_button(self, Form):  # 保存中间结果的 scroll Area

        self.snap_scrollArea = QtWidgets.QScrollArea(Form)
        self.snap_scrollArea.setGeometry(QtCore.QRect(100, Lb_y + Lb_height + Lb_col_shift + Lb_height, 1622* SCALE, 250* SCALE))
        self.snap_scrollArea.setWidgetResizable(True)
        self.snap_scrollArea.setObjectName("snap_scrollArea")
        self.snap_scrollArea.setAlignment(Qt.AlignCenter)
        #self.snap_scrollArea.setStyleSheet("border-color: transparent")
        self.snap_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        self.snap_scrollAreaWidgetContents = QtWidgets.QWidget()
        self.snap_scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1622* SCALE, 250* SCALE))
        self.snap_scrollAreaWidgetContents.setObjectName("snap_scrollAreaWidgetContents")

        self.snap_gridlLayout = QtWidgets.QGridLayout(self.snap_scrollAreaWidgetContents)
        # # snap_horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.snap_gridlLayout.setSpacing(20)
        self.snap_gridlLayout.setAlignment(Qt.AlignLeft)

        self.snap_style_button_list = []
        self.mask_snap_style_button_list = []

        for i in range(15):
            snap_style_button = QtWidgets.QPushButton()
            snap_style_button.setFixedSize(100, 100)
            snap_style_button.setStyleSheet("background-color: transparent")
            snap_style_button.setIcon(QIcon())
            snap_style_button.setIconSize(QSize(100, 100))
            snap_style_button.clicked.connect(partial(self.open_snapshot_dialog, i))
            # snap_style_button.snap_shot_name = None
            self.snap_style_button_list.append(snap_style_button)
            # style_button.hide()
            self.snap_gridlLayout.addWidget(snap_style_button, 1, i)


            mask_snap_style_button = QtWidgets.QPushButton()
            mask_snap_style_button.setFixedSize(100, 100)
            mask_snap_style_button.setStyleSheet("background-color: transparent")
            mask_snap_style_button.setIcon(QIcon())
            mask_snap_style_button.setIconSize(QSize(100, 100))
            self.mask_snap_style_button_list.append(mask_snap_style_button)
            # mask_snap_style_button.hide()
            self.snap_gridlLayout.addWidget(mask_snap_style_button, 0, i)


        self.snap_scrollArea.setWidget(self.snap_scrollAreaWidgetContents)

    def add_input_img_button(self, Form):  # 右上角当前编辑的图片
        self.input_img_button = QtWidgets.QPushButton(Form)
        self.input_img_button.setGeometry(QtCore.QRect(1770*SCALE , 15*SCALE, 100*SCALE, 100*SCALE))
        self.input_img_button.setStyleSheet("background-color: transparent")
        self.input_img_button.setFixedSize(100, 100)
        self.input_img_button.setIcon(QIcon(None))
        self.input_img_button.setIconSize(QSize(100, 100))
        self.input_img_button.clicked.connect(partial(Form.set_ref_img_path, 0))

    def add_checkbox_widgets(self, Form):  # 右上角的复选框
        self.checkBoxGroupBox = QtWidgets.QGroupBox("Replace Style of Components", Form)
        self.checkBoxGroupBox.setGeometry(QtCore.QRect(920* SCALE, 10* SCALE, 800, 100))

        layout = QtWidgets.QGridLayout()
        self.checkBoxGroup = QtWidgets.QButtonGroup(Form)
        self.checkBoxGroup.setExclusive(False)
        for i, j in enumerate(my_number_object):
            cb = QtWidgets.QCheckBox(my_number_object[j])
            self.checkBoxGroup.addButton(cb, i)
            layout.addWidget(cb, i//10, i%10)

        cb = QtWidgets.QCheckBox('ALL')
        self.checkBoxGroup.addButton(cb, )
        layout.addWidget(cb, (i+1)//10, (i+1)%10)

        self.checkBoxGroupBox.setLayout(layout)

        for i in range(len(my_number_object)):
            self.checkBoxGroup.button(i).setChecked(False)

        checkbox_status = [cb.isChecked() for cb in self.checkBoxGroup.buttons()]
        checkbox_status = checkbox_status[:len(my_number_object)]
        self.checkbox_status = checkbox_status
        self.checkBoxGroup.buttonToggled.connect(self.cb_event)

    def add_brush_widgets(self, Form):
        # KaustLogo = QtWidgets.QLabel(self)
        # # KaustLogo.setPixmap(QPixmap('icons/kaust_logo.svg').scaled(60, 60))
        # KaustLogo.setPixmap(QPixmap('ui_run/icons/1999780_200.png').scaled(60, 60))
        # KaustLogo.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 25, 80* SCALE, 80* SCALE))

        self.add_style_imgs_buttons(Form)  # 加载右边的备选图片
        self.brushsizeLabel = QtWidgets.QLabel(Form)
        self.brushsizeLabel.setObjectName("brushsizeLabel")
        self.brushsizeLabel.setGeometry(QtCore.QRect(int(Tb_x), 25, int(150 * SCALE), int(20 * SCALE)))
        self.brushsizeLabel.setText('Brush size: 6')
        font = self.brushsizeLabel.font()
        font.setPointSize(10)
        font.setBold(True)
        self.brushsizeLabel.setFont(font)

        self.brushSlider = QtWidgets.QSlider(Form)
        self.brushSlider.setOrientation(QtCore.Qt.Horizontal)
        self.brushSlider.setGeometry(QtCore.QRect(int(Tb_x + 150* SCALE), 25, int(600* SCALE), int(10* SCALE)))
        self.brushSlider.setObjectName("brushSlider")
        self.brushSlider.setMinimum(1)
        self.brushSlider.setMaximum(100)
        self.brushSlider.setValue(8)
        self.brushSlider.valueChanged.connect(Form.change_brush_size)  # 绑定slider bar的数值变化

    def add_top_buttons(self, Form):  # 添加顶部的按钮
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(int(Tb_x), int(Tb_y), int(Tb_width), int(Tb_height)))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(Form.open)

        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(int(Tb_x + 1 * Tb_row_shift + 1 * Tb_width), int(Tb_y), int(Tb_width), int(Tb_height)))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(Form.mixing_ref_img_style)

        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(int(Tb_x + 2 * Tb_row_shift + 2 * Tb_width), int(Tb_y), int(Tb_width), int(Tb_height)))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(Form.editing)

        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(int(Tb_x + 3 * Tb_row_shift + 3 * Tb_width), int(Tb_y), int(Tb_width), int(Tb_height)))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(Form.recon)

        self.saveImg = QtWidgets.QPushButton(Form)
        self.saveImg.setGeometry(QtCore.QRect(int(Tb_x + 4 * Tb_row_shift + 4 * Tb_width), int(Tb_y), int(Tb_width), int(Tb_height)))
        self.saveImg.setObjectName("saveImg")
        self.saveImg.clicked.connect(Form.save_img)

        self.retranslateUi(Form)

    def add_tool_buttons(self, Form):  # 左边的工具栏图片
        self.newButton = QtWidgets.QPushButton(Form)
        self.newButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60* SCALE), 140* SCALE + 60* SCALE*1 + 10* SCALE*1, 60* SCALE, 60* SCALE))
        self.newButton.setObjectName("openButton")
        self.newButton.setIcon(QIcon('ui_run/icons/reset200.png'))
        self.newButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        self.newButton.clicked.connect(Form.init_screen)  # 重置

        # self.openButton = QtWidgets.QPushButton(Form)
        # self.openButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60* SCALE), 140* SCALE, 60* SCALE, 60* SCALE))
        # self.openButton.setObjectName("openButton")
        # self.openButton.setIcon(QIcon('ui_run/icons/open.png'))
        # self.openButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        # self.openButton.clicked.connect(Form.open) 

        self.fillButton = QtWidgets.QPushButton(Form)
        self.fillButton.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60* SCALE), 140* SCALE + 60* SCALE*2 + 10* SCALE*2, 60* SCALE, 60* SCALE))
        self.fillButton.setObjectName("fillButton")
        self.fillButton.setIcon(QIcon('ui_run/icons/paint_can.png'))
        self.fillButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        self.fillButton.clicked.connect(partial(Form.mode_select, 2))

        self.brushButton = QtWidgets.QPushButton(Form)
        self.brushButton.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60* SCALE), 140* SCALE + 60* SCALE*3 + 10* SCALE*3, 60* SCALE, 60* SCALE))
        self.brushButton.setObjectName("brushButton")
        self.brushButton.setIcon(QIcon('ui_run/icons/paint_brush.png'))
        self.brushButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        self.brushButton.setStyleSheet("background-color: #85adad")
        #self.brushButton.setStyleSheet("background-color:")
        self.brushButton.clicked.connect(partial(Form.mode_select, 0))

        self.recButton = QtWidgets.QPushButton(Form)
        self.recButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60* SCALE), 140* SCALE + 60* SCALE * 4 + 10* SCALE * 4, 60* SCALE, 60* SCALE))
        self.recButton.setObjectName("undolButton")
        self.recButton.setIcon(QIcon('ui_run/icons/brush_square.png'))
        self.recButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        self.recButton.clicked.connect(partial(Form.mode_select, 1))

        self.undoButton = QtWidgets.QPushButton(Form)
        self.undoButton.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60* SCALE), 140* SCALE + 60* SCALE*5 + 10* SCALE*5, 60* SCALE, 60* SCALE))
        self.undoButton.setObjectName("undolButton")
        self.undoButton.setIcon(QIcon('ui_run/icons/undo.png'))
        self.undoButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        self.undoButton.clicked.connect(Form.undo)

        # self.saveButton = QtWidgets.QPushButton(Form)
        # self.saveButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60* SCALE), 140* SCALE + 60* SCALE * 6 + 10* SCALE * 6, 60* SCALE, 60* SCALE))
        # self.saveButton.setObjectName("saveButton")
        # self.saveButton.setIcon(QIcon('ui_run/icons/save.png'))
        # self.saveButton.setIconSize(QSize(60* SCALE, 60* SCALE))
        # self.saveButton.clicked.connect(Form.save_img)

    def add_style_imgs_buttons(self, Form):   # 添加一个style图片的部分（右边的滚动框）

        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(int(1756* SCALE), int(140* SCALE), 140, 512))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollArea.setAlignment(Qt.AlignCenter)
        # self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()  # 一个父widget，用来存放滚动区域的小图片
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, int(140 * SCALE), int(512 * SCALE)))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")


        verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        verticalLayout.setContentsMargins(11, 11, 11, 11)
        verticalLayout.setSpacing(6)


        # img_path_list = glob.glob('imgs/style_imgs_test/*.jpg')
        img_path_list = glob.glob('ui_run/testset/CelebA-HQ/test/images/*.jpg')
        img_path_list.sort()

        # style_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        # style_button.setFixedSize(100, 100)
        # style_button.setIcon(QIcon('ui_run/icons/random.png'))
        # style_button.setIconSize(QSize(100, 100))
        # # style_button.clicked.connect(Form.load_partial_average_feature)  # 随机加载一个特征，还没实现这个功能
        # verticalLayout.addWidget(style_button)

        for img_path in img_path_list:
            style_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
            style_button.setFixedSize(100, 100)
            style_button.setIcon(QIcon(img_path))
            style_button.setIconSize(QSize(100, 100))
            style_button.clicked.connect(partial(Form.set_ref_img_path, img_path))
            verticalLayout.addWidget(style_button)


        verticalLayout.addWidget(style_button)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

    def add_label_buttons(self, Form):  # 12个 mask的 颜色按钮

        self.color_Button = QtWidgets.QPushButton(Form)  # 当前选定的颜色
        self.color_Button.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), int(Lb_y), 60, 60))
        self.color_Button.setObjectName("labelButton_0")
        self.color_Button.setStyleSheet("background-color: %s;" % number_color[1])  # 默认为 idx = 1


        self.labelButton_0 = QtWidgets.QPushButton(Form)
        self.labelButton_0.setGeometry(QtCore.QRect(int(Lb_x), int(Lb_y), int(Lb_width), int(Lb_height)))
        self.labelButton_0.setObjectName("labelButton_0")
        self.labelButton_0.setText(_translate("Form", "background"))
        self.labelButton_0.setStyleSheet("background-color: %s;" % number_color[0]+ " color: black")
        self.labelButton_0.clicked.connect(partial(Form.switch_labels, 0))



        self.labelButton_1 = QtWidgets.QPushButton(Form)
        self.labelButton_1.setGeometry(QtCore.QRect(Lb_x + 1*Lb_row_shift + 1*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_1.setObjectName("labelButton_1")
        self.labelButton_1.setText(_translate("Form", "lip"))
        self.labelButton_1.setStyleSheet("background-color: %s;" % number_color[1] + " color: black")
        self.labelButton_1.clicked.connect(partial(Form.switch_labels, 1))


        self.labelButton_2 = QtWidgets.QPushButton(Form)
        self.labelButton_2.setGeometry(QtCore.QRect(Lb_x + 2*Lb_row_shift + 2*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_2.setObjectName("labelButton_2")
        self.labelButton_2.setText(_translate("Form", "eyebrows"))
        self.labelButton_2.setStyleSheet("background-color: %s;" % number_color[2] + " color: black")
        self.labelButton_2.clicked.connect(partial(Form.switch_labels, 2))
    

        self.labelButton_3 = QtWidgets.QPushButton(Form)
        self.labelButton_3.setGeometry(QtCore.QRect(Lb_x + 3*Lb_row_shift + 3*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_3.setObjectName("labelButton_3")
        self.labelButton_3.setText(_translate("Form", "eyes"))
        self.labelButton_3.setStyleSheet("background-color: %s;" % number_color[3] + " color: black")
        self.labelButton_3.clicked.connect(partial(Form.switch_labels, 3))


        self.labelButton_4 = QtWidgets.QPushButton(Form)
        self.labelButton_4.setGeometry(QtCore.QRect(Lb_x + 4*Lb_row_shift + 4*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_4.setObjectName("labelButton_4")
        self.labelButton_4.setText(_translate("Form", "hair"))
        self.labelButton_4.setStyleSheet("background-color: %s;" % number_color[4] + " color: black")
        self.labelButton_4.clicked.connect(partial(Form.switch_labels, 4))


        self.labelButton_5 = QtWidgets.QPushButton(Form)
        self.labelButton_5.setGeometry(QtCore.QRect(Lb_x + 5*Lb_row_shift + 5*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_5.setObjectName("labelButton_5")
        self.labelButton_5.setText(_translate("Form", "nose"))
        self.labelButton_5.setStyleSheet("background-color: %s;" % number_color[5] + " color: black")
        self.labelButton_5.clicked.connect(partial(Form.switch_labels, 5))


        self.labelButton_6 = QtWidgets.QPushButton(Form)
        self.labelButton_6.setGeometry(QtCore.QRect(Lb_x + 6*Lb_row_shift + 6*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_6.setObjectName("labelButton_6")
        self.labelButton_6.setText(_translate("Form", "skin"))
        self.labelButton_6.setStyleSheet("background-color: %s;" % number_color[6] + " color: black")
        self.labelButton_6.clicked.connect(partial(Form.switch_labels, 6))


        self.labelButton_7 = QtWidgets.QPushButton(Form)
        self.labelButton_7.setGeometry(QtCore.QRect(Lb_x + 7*Lb_row_shift + 7*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_7.setObjectName("labelButton_7")
        self.labelButton_7.setText(_translate("Form", "ears"))
        self.labelButton_7.setStyleSheet("background-color: %s;" % number_color[7] + " color: black")
        self.labelButton_7.clicked.connect(partial(Form.switch_labels, 7))


        self.labelButton_8 = QtWidgets.QPushButton(Form)
        self.labelButton_8.setGeometry(QtCore.QRect(Lb_x + 8*Lb_row_shift + 8*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_8.setObjectName("labelButton_8")
        self.labelButton_8.setText(_translate("Form", "belowface"))
        self.labelButton_8.setStyleSheet("background-color: %s;" % number_color[8] + " color: black")
        self.labelButton_8.clicked.connect(partial(Form.switch_labels, 8))

        self.labelButton_9 = QtWidgets.QPushButton(Form)
        self.labelButton_9.setGeometry(QtCore.QRect(Lb_x + 9 * Lb_row_shift + 9 * Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_9.setObjectName("labelButton_9")
        self.labelButton_9.setText(_translate("Form", "mouth"))
        self.labelButton_9.setStyleSheet("background-color: %s;" % number_color[9] + " color: black")
        self.labelButton_9.clicked.connect(partial(Form.switch_labels, 9))


        # Second Row
        self.labelButton_10 = QtWidgets.QPushButton(Form)
        self.labelButton_10.setGeometry(QtCore.QRect(Lb_x,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_10.setObjectName("labelButton_10")
        self.labelButton_10.setText(_translate("Form", "eye_glass"))
        self.labelButton_10.setStyleSheet("background-color: %s;" % number_color[10] + " color: black")
        self.labelButton_10.clicked.connect(partial(Form.switch_labels, 10))


        self.labelButton_11 = QtWidgets.QPushButton(Form)
        self.labelButton_11.setGeometry(QtCore.QRect(Lb_x + 1*Lb_row_shift + 1*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_11.setObjectName("labelButton_11")
        self.labelButton_11.setText(_translate("Form", "ear_rings"))
        self.labelButton_11.setStyleSheet("background-color: %s;" % number_color[11] + " color: black")
        self.labelButton_11.clicked.connect(partial(Form.switch_labels, 11))

    def add_label_buttons_seg19(self,Form):  # 19个 mask的 颜色按钮
        self.color_Button = QtWidgets.QPushButton(Form)  # 当前选定的颜色
        self.color_Button.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), Lb_y, 60, 60))
        self.color_Button.setObjectName("labelButton_0")
        self.color_Button.setStyleSheet("background-color: %s;" % number_color[1])  # 默认为 idx = 1


        self.labelButton_0 = QtWidgets.QPushButton(Form)
        self.labelButton_0.setGeometry(QtCore.QRect(Lb_x, Lb_y, Lb_width, Lb_height))
        self.labelButton_0.setObjectName("labelButton_0")
        self.labelButton_0.setText(_translate("Form", "background"))
        self.labelButton_0.setStyleSheet("background-color: %s;" % number_color[0]+ " color: black")
        self.labelButton_0.clicked.connect(partial(Form.switch_labels, 0))


        self.labelButton_1 = QtWidgets.QPushButton(Form)
        self.labelButton_1.setGeometry(QtCore.QRect(Lb_x + 1*Lb_row_shift + 1*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_1.setObjectName("labelButton_1")
        self.labelButton_1.setText(_translate("Form", "skin"))
        self.labelButton_1.setStyleSheet("background-color: %s;" % number_color[1] + " color: black")
        self.labelButton_1.clicked.connect(partial(Form.switch_labels, 1))


        self.labelButton_2 = QtWidgets.QPushButton(Form)
        self.labelButton_2.setGeometry(QtCore.QRect(Lb_x + 2*Lb_row_shift + 2*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_2.setObjectName("labelButton_2")
        self.labelButton_2.setText(_translate("Form", "nose"))
        self.labelButton_2.setStyleSheet("background-color: %s;" % number_color[2] + " color: black")
        self.labelButton_2.clicked.connect(partial(Form.switch_labels, 2))
    

        self.labelButton_3 = QtWidgets.QPushButton(Form)
        self.labelButton_3.setGeometry(QtCore.QRect(Lb_x + 3*Lb_row_shift + 3*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_3.setObjectName("labelButton_3")
        self.labelButton_3.setText(_translate("Form", "eye_g"))
        self.labelButton_3.setStyleSheet("background-color: %s;" % number_color[3] + " color: black")
        self.labelButton_3.clicked.connect(partial(Form.switch_labels, 3))


        self.labelButton_4 = QtWidgets.QPushButton(Form)
        self.labelButton_4.setGeometry(QtCore.QRect(Lb_x + 4*Lb_row_shift + 4*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_4.setObjectName("labelButton_4")
        self.labelButton_4.setText(_translate("Form", "l_eye"))
        self.labelButton_4.setStyleSheet("background-color: %s;" % number_color[4] + " color: black")
        self.labelButton_4.clicked.connect(partial(Form.switch_labels, 4))


        self.labelButton_5 = QtWidgets.QPushButton(Form)
        self.labelButton_5.setGeometry(QtCore.QRect(Lb_x + 5*Lb_row_shift + 5*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_5.setObjectName("labelButton_5")
        self.labelButton_5.setText(_translate("Form", "r_eye"))
        self.labelButton_5.setStyleSheet("background-color: %s;" % number_color[5] + " color: black")
        self.labelButton_5.clicked.connect(partial(Form.switch_labels, 5))


        self.labelButton_6 = QtWidgets.QPushButton(Form)
        self.labelButton_6.setGeometry(QtCore.QRect(Lb_x + 6*Lb_row_shift + 6*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_6.setObjectName("labelButton_6")
        self.labelButton_6.setText(_translate("Form", "l_brow"))
        self.labelButton_6.setStyleSheet("background-color: %s;" % number_color[6] + " color: black")
        self.labelButton_6.clicked.connect(partial(Form.switch_labels, 6))


        self.labelButton_7 = QtWidgets.QPushButton(Form)
        self.labelButton_7.setGeometry(QtCore.QRect(Lb_x + 7*Lb_row_shift + 7*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_7.setObjectName("labelButton_7")
        self.labelButton_7.setText(_translate("Form", "r_brow"))
        self.labelButton_7.setStyleSheet("background-color: %s;" % number_color[7] + " color: black")
        self.labelButton_7.clicked.connect(partial(Form.switch_labels, 7))


        self.labelButton_8 = QtWidgets.QPushButton(Form)
        self.labelButton_8.setGeometry(QtCore.QRect(Lb_x + 8*Lb_row_shift + 8*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_8.setObjectName("labelButton_8")
        self.labelButton_8.setText(_translate("Form", "l_ear"))
        self.labelButton_8.setStyleSheet("background-color: %s;" % number_color[8] + " color: black")
        self.labelButton_8.clicked.connect(partial(Form.switch_labels, 8))

        self.labelButton_9 = QtWidgets.QPushButton(Form)
        self.labelButton_9.setGeometry(QtCore.QRect(Lb_x + 9 * Lb_row_shift + 9 * Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_9.setObjectName("labelButton_9")
        self.labelButton_9.setText(_translate("Form", "r_ear"))
        self.labelButton_9.setStyleSheet("background-color: %s;" % number_color[9] + " color: black")
        self.labelButton_9.clicked.connect(partial(Form.switch_labels, 9))


        # Second Row
        self.labelButton_10 = QtWidgets.QPushButton(Form)
        self.labelButton_10.setGeometry(QtCore.QRect(Lb_x,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_10.setObjectName("labelButton_10")
        self.labelButton_10.setText(_translate("Form", "mouth"))
        self.labelButton_10.setStyleSheet("background-color: %s;" % number_color[10] + " color: black")
        self.labelButton_10.clicked.connect(partial(Form.switch_labels, 10))


        self.labelButton_11 = QtWidgets.QPushButton(Form)
        self.labelButton_11.setGeometry(QtCore.QRect(Lb_x + 1*Lb_row_shift + 1*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_11.setObjectName("labelButton_11")
        self.labelButton_11.setText(_translate("Form", "u_lip"))
        self.labelButton_11.setStyleSheet("background-color: %s;" % number_color[11] + " color: black")
        self.labelButton_11.clicked.connect(partial(Form.switch_labels, 11))

        self.labelButton_12 = QtWidgets.QPushButton(Form)
        self.labelButton_12.setGeometry(QtCore.QRect(Lb_x + 2*Lb_row_shift + 2*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_12.setObjectName("labelButton_12")
        self.labelButton_12.setText(_translate("Form", "l_lip"))
        self.labelButton_12.setStyleSheet("background-color: %s;" % number_color[12] + " color: black")
        self.labelButton_12.clicked.connect(partial(Form.switch_labels, 12))
        
        self.labelButton_13 = QtWidgets.QPushButton(Form)
        self.labelButton_13.setGeometry(QtCore.QRect(Lb_x + 3*Lb_row_shift + 3*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_13.setObjectName("labelButton_13")
        self.labelButton_13.setText(_translate("Form", "hair"))
        self.labelButton_13.setStyleSheet("background-color: %s;" % number_color[13] + " color: black")
        self.labelButton_13.clicked.connect(partial(Form.switch_labels, 13))
        
        self.labelButton_14 = QtWidgets.QPushButton(Form)
        self.labelButton_14.setGeometry(QtCore.QRect(Lb_x + 4*Lb_row_shift + 4*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_14.setObjectName("labelButton_14")
        self.labelButton_14.setText(_translate("Form", "hat"))
        self.labelButton_14.setStyleSheet("background-color: %s;" % number_color[14] + " color: black")
        self.labelButton_14.clicked.connect(partial(Form.switch_labels, 14))
        
        self.labelButton_15 = QtWidgets.QPushButton(Form)
        self.labelButton_15.setGeometry(QtCore.QRect(Lb_x + 5*Lb_row_shift + 5*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        
        self.labelButton_15.setObjectName("labelButton_15")
        self.labelButton_15.setText(_translate("Form", "ear_r"))
        self.labelButton_15.setStyleSheet("background-color: %s;" % number_color[15] + " color: black")
        self.labelButton_15.clicked.connect(partial(Form.switch_labels, 15))


        self.labelButton_16 = QtWidgets.QPushButton(Form)
        self.labelButton_16.setGeometry(QtCore.QRect(Lb_x + 6*Lb_row_shift + 6*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_16.setObjectName("labelButton_16")
        self.labelButton_16.setText(_translate("Form", "neck_l"))
        self.labelButton_16.setStyleSheet("background-color: %s;" % number_color[16] + " color: black")
        self.labelButton_16.clicked.connect(partial(Form.switch_labels, 16))

        self.labelButton_17 = QtWidgets.QPushButton(Form)
        self.labelButton_17.setGeometry(QtCore.QRect(Lb_x + 7*Lb_row_shift + 7*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_17.setObjectName("labelButton_17")
        self.labelButton_17.setText(_translate("Form", "neck"))
        self.labelButton_17.setStyleSheet("background-color: %s;" % number_color[17] + " color: black")
        self.labelButton_17.clicked.connect(partial(Form.switch_labels, 17))

        self.labelButton_18 = QtWidgets.QPushButton(Form)
        self.labelButton_18.setGeometry(QtCore.QRect(Lb_x + 8*Lb_row_shift + 8*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_18.setObjectName("labelButton_18")
        self.labelButton_18.setText(_translate("Form", "cloth"))
        self.labelButton_18.setStyleSheet("background-color: %s;" % number_color[18] + " color: black")
        self.labelButton_18.clicked.connect(partial(Form.switch_labels, 18))


    def add_ops_log_textBox(self,Form):  # 操作日志框

        self.opsLogLabel = QtWidgets.QLabel(Form)
        self.opsLogLabel.setObjectName("opsLogLabel")
        self.opsLogLabel.setGeometry(QtCore.QRect(Lb_x + 10*SCALE * Lb_row_shift + 10*SCALE * Lb_width + 40*SCALE, Lb_y + 50, 150*SCALE, 20*SCALE))
        self.opsLogLabel.setText('Logging ')
        font = self.brushsizeLabel.font()
        font.setPointSize(10)
        font.setBold(True)
        self.opsLogLabel.setFont(font)

        self.opsLogTextBox = QtWidgets.QPlainTextEdit(Form)
        self.opsLogTextBox.setReadOnly(True)
        self.opsLogTextBox.setObjectName("opsLogTextBox")
        self.opsLogTextBox.setGeometry(QtCore.QRect(Lb_x + 10*SCALE * Lb_row_shift + 10 *SCALE* Lb_width + 150*SCALE, Lb_y+35, 225*SCALE, 40*SCALE))
               
    def add_ref_img_button(self, Form):  # 右下角当前reference 的图片
        self.ref_img_button = QtWidgets.QPushButton(Form)
        self.ref_img_button.setGeometry(QtCore.QRect(1770*SCALE , 800*SCALE, 100*SCALE, 100*SCALE))
        self.ref_img_button.setStyleSheet("background-color: transparent")
        self.ref_img_button.setFixedSize(100, 100)
        self.ref_img_button.setIcon(QIcon(None))
        self.ref_img_button.setIconSize(QSize(100, 100))
        

    def cb_event(self, id, ifchecked):

        if id.text() == 'ALL':
            if ifchecked:
                for cb in self.checkBoxGroup.buttons():
                    cb.setChecked(True)
            else:
                for cb in self.checkBoxGroup.buttons():
                    cb.setChecked(False)
        self.change_cb_state()

    def change_cb_state(self):
        checkbox_status = [cb.isChecked() for cb in self.checkBoxGroup.buttons()]
        checkbox_status = checkbox_status[:len(my_number_object)]
        #self.obj_dic_back = copy.deepcopy(self.obj_dic)
        self.checkbox_status = checkbox_status



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
