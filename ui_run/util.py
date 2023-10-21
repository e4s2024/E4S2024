import numpy as np

########################  FACADES

# number_object = {
#                 0: 'background',###############
#                 1: 'facade',
#                 2: 'ledge',
#                 3: 'molding',
#                 4: 'pillar',
#                 5: 'deco', ###############
#                 6: 'cornice',
#                 7: 'window', #########
#                 8: 'sill',
#                 9: 'balcony', #########
#                 10: 'door',
#                 11: 'feb',
#                 12: 'fel',
#                 13: 'shop',   #########
#                 14: 'awning',
#                 15: 'sign',  #########
#                 16: 'tree', #########
#                 17: 'obs',  #########
# }
#
#
# number_color = {
#                 0: '#808080',
#                 1: '#751076',
#                 2: '#ff17ad'
#                 3: '#e8856e',
#                 4: '#ff4f03',
#                 5: '#854539',
#                 6: '#023e7f',
#                 7: '#05e8ff',
#                 8: '#027880',
#                 9: '#787bff',
#                 10: '#a56729',
#                 11: '#6729a5',
#                 12: '#a550a5',
#                 13: '#5dff73',
#                 14: '#fffe00',
#                 15: '#bde82a',
#                 16: '#3f6604',
#                 17: '#ff0000',
#
#
# }
#
#
# def color_pred(pred):
#
#     num_labels=18
#     color = np.array([[128,128,128],
#                     [117,16,118],
#                     [255,23,173],
#                     [232,133,110],
#                     [255,79,3],
#                     [133,69,57],
#                     [2,62,127],
#                     [5,232,255],
#                     [2,120,128],
#                     [120,123,255],
#                     [165,103,41],
#                     [103,41,165],
#                     [165,80,165],
#                     [93,255,115],
#                     [255,254,0],
#                     [189,232,42],
#                     [63,102,4],
#                     [255,0,0],
#                     ])
#     h, w = np.shape(pred)
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     #     print(color.shape)
#     for ii in range(num_labels):
#         #         print(ii)
#         mask = pred == ii
#         rgb[mask, None] = color[ii, :]
#     # Correct unk
#     unk = pred == 255
#     rgb[unk, None] = color[0, :]
#
#     return rgb


########################  FACES


number_object = {
                0: 'background',
                1: 'skin',
                2: 'nose',
                3: 'eye_g',
                4: 'l_eye',
                5: 'r_eye',
                6: 'l_brow',
                7: 'r_brow',
                8: 'l_ear',
                9: 'r_ear',
                10: 'mouth',
                11: 'u_lip',
                12: 'l_lip',
                13: 'hair',
                14: 'hat',
                15: 'ear_r',
                16: 'neck_l',
                17: 'neck',
                18: 'cloth',
}


number_color = {
                0: '#000000',
                1: '#cc0000',
                2: '#4c9900',
                3: '#cccc00',
                4: '#3333ff',
                5: '#cc00cc',
                6: '#00ffff',
                7: '#ffcccc',
                8: '#663300',
                9: '#ff0000',
                10: '#66cc00',
                11: '#ffff00',
                12: '#000099',
                13: '#0000cc',
                14: '#ff3399',
                15: '#00cccc',
                16: '#003300',
                17: '#ff9933',
                18: '#00cc00',

}  # celeba-hq 19个mask的颜色

#
# face_gray_color = np.array([[0,  0,  0],
#                     [204, 0,  0],
#                     [76, 153, 0],
#                     [204, 204, 0],##
#                     [51, 51, 255],##
#                     [204, 0, 204],##
#                     [0, 255, 255],##
#                     [51, 255, 255],##
#                     [102, 51, 0],##
#                     [255, 0, 0],##
#                     [102, 204, 0],##
#                     [255, 255, 0],##
#                     [0, 0, 153],##
#                     [0, 0, 204],##
#                     [255, 51, 153],##
#                     [0, 204, 204],##
#                     [0, 51, 0],##
#                     [255, 153, 51],
#                     [0, 204, 0],
#                     ])



def color_pred(pred):

    num_labels=19
    color = np.array([[0,  0,  0],
                    [204, 0,  0],
                    [76, 153, 0],
                    [204, 204, 0],##
                    [51, 51, 255],##
                    [204, 0, 204],##
                    [0, 255, 255],##
                    [255, 204, 204],##
                    [102, 51, 0],##
                    [255, 0, 0],##
                    [102, 204, 0],##
                    [255, 255, 0],##
                    [0, 0, 153],##
                    [0, 0, 204],##
                    [255, 51, 153],##
                    [0, 204, 204],##
                    [0, 51, 0],##
                    [255, 153, 51],
                    [0, 204, 0],
                    ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]

    return rgb



# ==============================

def celebAHQ_masks_to_faceParser_mask_detailed(celebA_mask):
    """将 celebAHQ_mask 格式的mask 转为 faceParser格式（聚合某些 facial component）
    
    保持一些细节，例如牙齿

    Args:
        celebA_mask (PIL image): with shape [H,W]
    Return:
        返回转化后的mask，同样是shape [H,W]，但是类别数更少
    """

    # assert len(celebA_mask.size) == 2, "mask 必须是[H,W]格式的数据"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(celebA_mask, 11), np.equal(celebA_mask, 12))
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8
    
    mouth = np.equal(celebA_mask, 10)   # 牙齿
    converted_mask[mouth] = 9

    eye_glass = np.equal(celebA_mask, 3)   # 眼镜
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(celebA_mask, 15)   # 耳环
    converted_mask[ear_rings] = 11
    
    # r_ear = np.equal(celebA_mask, 9)  # 右耳
    # converted_mask[r_ear] = 12
    
    return converted_mask

my_number_object = {
        0: 'background',
        1: 'lip',
        2: 'eyebrows', 
        3: 'eyes',
        4: 'hair',
        5: 'nose',
        6: 'skin',
        7: 'ears',
        8: 'belowface',
        9: 'mouth',
        10: 'eye_glass',
        11: 'ear_rings',
}

COMPS = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface','mouth','eye_glass','ear_rings']
