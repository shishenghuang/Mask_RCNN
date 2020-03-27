#
# This file is part of https://github.com/martinruenz/maskfusion
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import os

import numpy as np
#import scipy.misc

from PIL import Image

# Common parameters used in both offline runner and the online version
score_threshold = 0.55
SPECIAL_ASSIGNMENTS = {} #{'person': 255}
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# If not empty, ignore every class not listed here. Example: ['teddy bear']
filter_classes = []
# Conversion from COCO to ours sun38.
class_conversion = {
    'BG':0, 'person':31, 'bicycle':-1, 'car':-1, 'motorcycle':-1, 'airplane':-1,
    'bus':-1, 'train':-1, 'truck':-1, 'boat':-1, 'traffic light':-1,
    'fire hydrant':3, 'stop sign':-1, 'parking meter':-1, 'bench':5, 'bird':-1,
    'cat':-1, 'dog':-1, 'horse':-1, 'sheep':-1, 'cow':-1, 'elephant':-1, 'bear':-1,
    'zebra':-1, 'giraffe':-1, 'backpack': 37, 'umbrella':29, 'handbag':29, 'tie':-1,
    'suitcase':29, 'frisbee':-1, 'skis':-1, 'snowboard':-1, 'sports ball':-1,
    'kite':-1, 'baseball bat':-1, 'baseball glove':-1, 'skateboard':-1,
    'surfboard':-1, 'tennis racket':-1, 'bottle':29, 'wine glass':29, 'cup':29,
    'fork':29, 'knife':29, 'spoon':29, 'bowl':29, 'banana':29, 'apple':29,
    'sandwich':29, 'orange':29, 'broccoli':29, 'carrot':29, 'hot dog':29, 'pizza':29,
    'donut':-1, 'cake':-1, 'chair':5, 'couch':6, 'potted plant':-1, 'bed':4,
    'dining table':7, 'toilet':33, 'tv': 25, 'laptop':29, 'mouse':29, 'remote':29,
    'keyboard':29, 'cell phone':29, 'microwave': 15, 'oven': 15, 'toaster':-1,
    'sink':34, 'refrigerator':24, 'book': 23, 'clock':29, 'vase':32, 'scissors':-1,
    'teddy bear':29, 'hair drier':29, 'toothbrush':29
}
filter_classes += [k for k, v in class_conversion.items() if v != -1]
class_conversion = {class_names.index(k):v for k, v in class_conversion.items()}
print(filter_classes)

def merge_instances(result):
    m = 0
    while m < result['masks'].shape[2]:
        class_id = result['class_ids'][m]

        multiple_instances = True
        while multiple_instances:
            multiple_instances = False
            # Find other instance
            for m2 in range(m + 1, result['masks'].shape[2]):
                class_id2 = result['class_ids'][m2]
                if class_id == class_id2:
                    multiple_instances = True
                    break

            # Merge
            if multiple_instances:
                result['scores'][m] = max(result['scores'][m], result['scores'][m2])
                mask = result['masks'][:, :, m]
                mask2 = result['masks'][:, :, m2]
                mask[mask2==1] = 1
                result['scores'] = np.delete(result['scores'], m2, 0)
                result['class_ids'] = np.delete(result['class_ids'], m2, 0)
                r['rois'] = np.delete(r['rois'], m2, 0)
                result['masks'] = np.delete(result['masks'], m2, 2)

        m += 1


# Note, this is not used within "generate_id_image" due to speed concerns
def filter_result(result, class_filter=[]):
    n = len(result['class_ids'])
    to_delete = []

    for m in range(n):
        class_id = result['class_ids'][m]
        if len(class_filter) > 0 and not(class_id in class_filter):
            to_delete.append(m)

    result['masks'] = np.delete(result['masks'], to_delete, 2)
    result['scores'] = np.delete(result['scores'], to_delete, 0)
    result['class_ids'] = np.delete(result['class_ids'], to_delete, 0)
    result['rois'] = np.delete(result['rois'], to_delete, 0)


def generate_id_image(result, min_score, class_filter=[], special_assignments=[]):
    masks = result['masks']
    scores = result['scores']
    class_ids = result['class_ids']
    rois = result['rois']
    h, w = masks.shape[0:2]
    n = len(class_ids)

    if(n > 256):
        raise RuntimeError("Too many masks in image.")

    id_image = np.zeros([h,w], np.uint8)
    exported_class_ids = []
    exported_rois = []

    for m in range(n):
        class_id = class_ids[m]
        if len(class_filter) == 0 or class_id in class_filter:
            if scores[m] >= min_score:
                mask = masks[:,:,m]
                val = len(exported_class_ids)+1
                if len(special_assignments) > 0 and class_id in special_assignments:
                    val = special_assignments[class_id]
                id_image[mask == 1] = val
                #exported_class_ids.append(str(class_id))
                exported_class_ids.append(int(class_id))
                exported_rois.append(rois[m,:].tolist())

    return id_image, exported_class_ids, exported_rois


def save_id_image(id_image, output_dir, base_name, exported_class_ids=[], export_classes=False, exported_rois=[]):

    #scipy.misc.toimage(id_image, cmin=0.0, cmax=255).save(path)
    Image.fromarray(id_image).save(os.path.join(output_dir, base_name + ".png"))

    if export_classes:
        exported_class_ids_str = [str(id) for id in exported_class_ids]
        with open(os.path.join(output_dir, base_name + ".txt"), "w") as file:
            file.write(" ".join(exported_class_ids_str))
            if len(exported_rois) > 0:
                for roi in exported_rois:
                    roi_str = [str(r) for r in roi]
                    file.write("\n" + " ".join(roi_str))
