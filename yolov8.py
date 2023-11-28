from ultralytics import YOLO
import torch
import cv2

from Helper.Errors import show_error

from Helper.Intersection import circle_rectangle_intersection, defect_classification
from Model.GlobalVariables import svt_model, colors_list, percent_defect


def SVT_classification(model, _file_path):
    predictions = model.predict(
        source=_file_path,
        conf=0.25
    )[0]

    cls = predictions.boxes.cls.to(torch.int32).tolist()
    boxes = predictions.boxes.data[:, :4].to(torch.int32).tolist()
    conf = predictions.boxes.conf

    img = predictions.orig_img
    labels = predictions.names



    dict_intersection_area = {}
    defect_area_list = []
    defect_box_list = []

    #  Find the boundingbox of the defect, calculate the area of this part first.
    for idx, box in enumerate(boxes):
        current_label = labels[cls[idx]]
        if current_label == 'defect':
            defect_box = (box[0], box[1], box[2], box[3])
            defect_area = abs(defect_box[2] - defect_box[0]) * abs(defect_box[3] - defect_box[1])

            #  Add to ds the errors detected in the current image (In case 1 image has many errors)
            defect_box_list.append(defect_box)
            defect_area_list.append(defect_area)

        else:
            # Draw circles
            midpoint = (box[0] + int((box[2] - box[0]) / 2), box[1] + int((box[3] - box[1]) / 2))
            r = int((box[2] - box[0]) / 2)
            cv2.circle(img, midpoint, r, colors_list[0], 2)
            cv2.putText(img,
                        labels[cls[idx]],
                        (box[0] + r, box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors_list[0],
                        1,
                        cv2.LINE_AA)
            # cv2.circle(img, midpoint, r, colors_list[0], 1)


    # Handling calculation of intersection area and drawing bounding box of defect
    for defect_box, defect_area in zip(defect_box_list, defect_area_list):
        # Calculate the area of intersection of the circle and the defect bounding box
        for idx, box in enumerate(boxes):
            current_label = labels[cls[idx]]
            if current_label != 'defect':  # Circle1, circle2, circle3
                midpoint = (box[0] + int((box[2] - box[0]) / 2), box[1] + int((box[3] - box[1]) / 2))
                r = int((box[2] - box[0]) / 2)

                intersection_are = circle_rectangle_intersection(midpoint, r,
                                                                 defect_box[0],
                                                                 defect_box[1],
                                                                 defect_box[2],
                                                                 defect_box[3])
                dict_intersection_area[current_label] = intersection_are

        defect_type = defect_classification(defect_area, dict_intersection_area, percent_defect)
        cv2.rectangle(img,
                      (defect_box[0], defect_box[1]),
                      (defect_box[2], defect_box[3]),
                      colors_list[int(defect_type[-1])],
                      2)

        cv2.putText(img,
                    defect_type,
                    (defect_box[0], defect_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors_list[int(defect_type[-1])],
                    1,
                    cv2.LINE_AA)

        defect_rate = {}
        for key, val in dict_intersection_area.items():
            defect_rate[key] = val/defect_area

        print(f'Area of defect bounding box: {defect_area}. Area of intersection: {dict_intersection_area} => Type: {defect_type}')
        print(f'defect rate: {defect_rate} => Type: {defect_type}')
    return img


if __name__ == '__main__':

    _filename = '0203_4_jpg.rf.2a3d5e9043aad797ee8a65601e79a6cb.jpg'
    _file_path = f'Images-Sample/{_filename}'

    try:
        img = SVT_classification(svt_model, _file_path)
        cv2.imshow('result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        show_error(e)
    finally:
        pass




