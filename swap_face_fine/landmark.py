import numpy as np
import PIL


def get_landmark(img, use_fa=False):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    if use_fa:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = None
        detector = None
    else:
        import dlib
        fa = None
        predictor = dlib.shape_predictor("./pretrained/E4S/shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()

    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    if fa is not None:
        # image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(img, return_bboxes=True)
        if len(lms) == 0:
            return None
        return lms[0]

    if detector is None:
        detector = dlib.get_frontal_face_detector()

    dets = detector(img)

    for k, d in enumerate(dets):
        
        # # 上下扩张50个pixel, 左右扩张20个pixel
        # d = dlib.rectangle(left=max(0, d.left()-20), top=max(0, d.top()-50), right=min(d.right()+20, img.shape[1]), bottom=min(d.bottom(), img.shape[0]))
    
        # (x, y, w, h) = rect_to_bb(d)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        shape = predictor(img, d)
        
        # shape_np = shape_to_np(shape)
        # for (x, y) in shape_np:
        #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        break
    else:
        return None
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm