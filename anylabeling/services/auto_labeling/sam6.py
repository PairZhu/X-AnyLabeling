import logging
import os
import traceback

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .model import Model
from .types import AutoLabelingResult
from segment_anything import sam_model_registry, SamPredictor
import hashlib
from ...global_data import GlobalData
import PIL.Image
import PIL.ImageEnhance
import torch


class Sam6(Model):
    """Segmentation model using SegmentAnything"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "model_type",
            "display_name",
            "checkpoint_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)

        # Get encoder and decoder model paths
        checkpoint_abs_path = self.get_model_abs_path(
            self.config, "checkpoint_path"
        )
        if not checkpoint_abs_path or not os.path.isfile(checkpoint_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize the checkpoint of sam.",
                )
            )

        # Load models
        self.sam = sam_model_registry[self.config["model_type"]](
            checkpoint=checkpoint_abs_path
        )

        try:
            self.sam.cuda()
        except Exception as e:
            logging.warning(e)
            self.sam.cpu()

        self.model = SamPredictor(self.sam)
        self.current_image = None
        self.marks = []
        self.last_masks = None

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        # 判断是否是上次的超集
        is_subset = True
        for mark in marks:
            if mark not in self.marks:
                is_subset = False
                break
        if not is_subset:
            self.last_masks = None
        self.marks = marks

    def post_process(self, masks, trans):
        """
        Post process masks
        """
        # Find contours
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        approx_points = []
        for approx in approx_contours:
            points = approx.reshape(-1, 2).astype(np.float32)
            points[:, 0] = (points[:, 0] + trans["offset_x"]) / trans["zoom"]
            points[:, 1] = (points[:, 1] + trans["offset_y"]) / trans["zoom"]
            points = points.tolist()
            new_points = [points[0]]
            for point in points:
                if point != new_points[-1]:
                    new_points.append(point)
            points = new_points
            if len(points) < 3:
                continue
            points.append(points[0])
            approx_points.append(points)

        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for points in approx_points:
                # Create shape
                shape = Shape(flags={})
                for point in points:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = "AUTOLABEL_OBJECT"
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for points in approx_points:
                # Get min/max
                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            # Create shape
            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.add_point(QtCore.QPointF(x_min, y_max))
            shape.shape_type = "rectangle"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "AUTOLABEL_OBJECT"
            shape.selected = False
            shapes.append(shape)

        return shapes

    def trans_image(self, image):
        brightness = GlobalData["brightness"] or 1
        contrast = GlobalData["contrast"] or 1
        # No need to convert between RGB and BGR, because the image here is in RGB format
        pil_image = PIL.Image.fromarray(image)
        pil_image = PIL.ImageEnhance.Brightness(pil_image).enhance(brightness)
        pil_image = PIL.ImageEnhance.Contrast(pil_image).enhance(contrast)
        image = np.array(pil_image)
        height, width = image.shape[:2]
        win_zoom = GlobalData["get_win_zoom"]() / 100
        zoom = GlobalData["zoom"] / 100
        # Make zoom a little smaller than the actual zoom
        real_zoom = zoom / win_zoom - 0.05
        if real_zoom <= 1:
            return image, {"zoom": 1, "offset_x": 0, "offset_y": 0}
        image = cv2.resize(image, (0, 0), fx=real_zoom, fy=real_zoom)
        offset_x = round(GlobalData["scroll_x"] / win_zoom)
        offset_y = round(GlobalData["scroll_y"] / win_zoom)
        image = np.ascontiguousarray(
            image[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
        )
        trans_coords = {
            "zoom": real_zoom,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }
        return image, trans_coords

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        cv_image = qt_img_to_rgb_cv_img(image, filename)
        cv_image, trans_coords = self.trans_image(cv_image)

        prompt = {
            "point_labels": np.array([]),
            "multimask_output": False,
            "mask_input": np.zeros((1, 256, 256), dtype=np.float32),
        }

        imghash = hashlib.md5(cv_image).hexdigest()
        if self.current_image != imghash:
            try:
                self.model.set_image(cv_image)
                self.current_image = imghash
            except Exception as e:
                logging.warning("Could not set image")
                logging.warning(e)
                return AutoLabelingResult([], replace=False)
        elif self.last_masks is not None:
            prompt["mask_input"] = self.last_masks

        for mark in self.marks:
            if mark["type"] == "point":
                prompt["point_coords"] = np.append(
                    prompt.get("point_coords", np.empty((0, 2))),
                    [
                        [
                            round(
                                mark["data"][0] * trans_coords["zoom"]
                                - trans_coords["offset_x"]
                            ),
                            round(
                                mark["data"][1] * trans_coords["zoom"]
                                - trans_coords["offset_y"]
                            ),
                        ]
                    ],
                    axis=0,
                )
                prompt["point_labels"] = np.append(
                    prompt["point_labels"], mark["label"]
                )
            elif mark["type"] == "rectangle":
                prompt["box"] = np.array(
                    [
                        round(
                            mark["data"][0] * trans_coords["zoom"]
                            - trans_coords["offset_x"]
                        ),
                        round(
                            mark["data"][1] * trans_coords["zoom"]
                            - trans_coords["offset_y"]
                        ),
                        round(
                            mark["data"][2] * trans_coords["zoom"]
                            - trans_coords["offset_x"]
                        ),
                        round(
                            mark["data"][3] * trans_coords["zoom"]
                            - trans_coords["offset_y"]
                        ),
                    ]
                )

        try:
            masks, _, low_res_logits = self.model.predict(**prompt)
            self.last_masks = low_res_logits
            shapes = self.post_process(masks.squeeze(), trans_coords)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.sam = None
        self.model = None
        self.current_image = None
        self.marks = []
        self.last_masks = None
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
