from pygments.formatters import img
import cvzone
img_path = 'test/0203_10_jpg.rf.50d1b2e10f87fb2e83649c725132f33a.jpg'

def efficienDet_predict(img_list):
    import torch
    from torch.backends import cudnn

    from backbone import EfficientDetBackbone
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    from efficientdet.utils import BBoxTransform, ClipBoxes
    from utils.utils import preprocess, invert_affine, postprocess

    compound_coef = 0
    force_input_size = None  # set None to use default size


    threshold = 0.3
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


    method = 'cv!'

    if method == 'cv':
        obj_list = ['Circle1', 'Circle2', 'Circle3', 'defects']
        weight_file = ['0_efficientdet-d0_circle123.pth']
    else:
        obj_list = ['Defect1', 'Defect2', 'Defect3', 'defect']
        weight_file = ['0_efficientdet-d0_df123.pth']

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load('test/' + weight_file[-1]))
    model.requires_grad_(False)
    model.eval()
    count = 0

    for img_path in img_list:
        count += 1
        if count > 50:
            break

        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)


        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

        out = invert_affine(framed_metas, out)



        for i in range(len(ori_imgs)):
            if len(out[i]['rois']) == 0:
                continue
            ori_imgs[i] = ori_imgs[i].copy()
            for j in range(len(out[i]['rois'])):
                bounding_color = (0, 255, 0)


                obj = obj_list[out[i]['class_ids'][j]]
                label_text = obj
                if 'defect' in obj.lower():
                    bounding_color = (255, 0, 0)
                    label_text = 'Defect'
                score = float(out[i]['scores'][j])

                (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
                cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), bounding_color, 2)

                cv2.putText(ori_imgs[i], '{}, {:.1f}'.format(obj, score),
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2)

                plt.imshow(ori_imgs[i])

            head, tail = os.path.split(img_path)
            filename = os.path.join("Results", tail)
            plt.savefig(filename)
            plt.show()


if __name__ == '__main__':
    import os
    img_path = r'C:\datasets\microfastener\test'
    img_list = []
    for filename in os.listdir(img_path):
        full_name = os.path.join(img_path, filename)
        if '.jpg' in full_name:
            img_list.append(full_name)

    efficienDet_predict(img_list)
