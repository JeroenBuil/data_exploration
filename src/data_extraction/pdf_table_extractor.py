import csv
import glob as glob
import io
from pathlib import Path, PureWindowsPath

import easyocr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import torch
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.autonotebook import tqdm
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from fitz import Rect


class PdfTableExtractor:

    def __init__(
        self,
    ):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Init table detection transformer
        self.table_detection_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        self.table_detection_model.to(self.device)

        # Init table structure recognition transformer
        self.table_structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-structure-recognition-v1.1-all"
        )
        self.table_structure_model.to(self.device)

        # Init OCR reader
        self.ocr_reader = easyocr.Reader(
            ["en"]
        )  # this needs to run only once to load the model into memory

    def _get_class_map(self, data_type):

        if data_type == "detection":
            class_map = {0: "table", 1: "table rotated", 2: "no object"}
        elif data_type == "structure":
            class_map = {
                0: "table",
                1: "table column",
                2: "table row",
                3: "table column header",
                4: "table projected row header",
                5: "table spanning cell",
                6: "no object",
            }
        return class_map

    def _load_pdf_pages(self, path):
        """Loads the pdf from the path and returns a list of pages as pil images

        Returns:
            list: list of pil images, each representing a page of the pdf
        """
        # Check if the file is a PDF
        if path.suffix != ".pdf":
            raise ImportError("File is not a .pdf")

        else:
            pdf = pdfium.PdfDocument(path)

            if len(pdf) < 1:
                raise Exception("No pages could be detected in this pdf")

            pages = []
            # Loop over pages and render
            for i in range(len(pdf)):
                page = pdf[i]
                pages.append(page.render(scale=4).to_pil())

            return pages

    def _box_cxcywh_to_xyxy(self, bbox):
        # for output bounding box post-processing
        x_c, y_c, w, h = bbox.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def _rescale_bboxes(self, bbox, image_size: tuple):
        # for output bounding box post-processing
        img_w, img_h = image_size
        b = self._box_cxcywh_to_xyxy(bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def transform_image_to_tensor(self, image: Image.Image, maxResize: int = 800) -> torch.Tensor:
        image_transform = transforms.Compose(
            [
                MaxResize(maxResize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image_tensor = image_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def _outputs_to_objects(self, model_outputs, image_size: tuple, class_idx2name: dict):

        # Find use softmax to find table prediction bboxes
        m = model_outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = model_outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [
            elem.tolist() for elem in self._rescale_bboxes(bbox=pred_bboxes, image_size=image_size)
        ]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = class_idx2name[int(label)]
            if not class_label == "no object":
                objects.append(
                    {
                        "label": class_label,
                        "score": float(score),
                        "bbox": [float(elem) for elem in bbox],
                    }
                )

        return objects

    def _detect_tables(self, page_image: Image.Image) -> list:
        # Resize image
        image_tensor = self.transform_image_to_tensor(image=page_image)

        # Run table detection model
        with torch.no_grad():
            model_outputs = self.table_detection_model(image_tensor)

        # Convert model output to actual table outputs
        detected_tables = self._outputs_to_objects(
            model_outputs=model_outputs,
            image_size=page_image.size,
            class_idx2name=self._get_class_map("detection"),
        )

        return detected_tables

    def _fig2img(fig: plt.Figure):
        """Convert a Matplotlib figure to a PIL Image and return it"""

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def visualize_detected_tables(page_image: Image.Image, out_path=None) -> Image.Image:
        plt.imshow(page_image, interpolation="lanczos")

        detected_tables = self._detect_tables(page_image=page_image)

        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        ax = plt.gca()

        for det_table in detected_tables:
            bbox = det_table["bbox"]

            if det_table["label"] == "table":
                facecolor = (1, 0, 0.45)
                edgecolor = (1, 0, 0.45)
                alpha = 0.3
                linewidth = 2
                hatch = "//////"
            elif det_table["label"] == "table rotated":
                facecolor = (0.95, 0.6, 0.1)
                edgecolor = (0.95, 0.6, 0.1)
                alpha = 0.3
                linewidth = 2
                hatch = "//////"
            else:
                continue

            rect = patches.Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=linewidth,
                edgecolor="none",
                facecolor=facecolor,
                alpha=0.1,
            )
            ax.add_patch(rect)
            rect = patches.Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor="none",
                linestyle="-",
                alpha=alpha,
            )
            ax.add_patch(rect)
            rect = patches.Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=0,
                edgecolor=edgecolor,
                facecolor="none",
                linestyle="-",
                hatch=hatch,
                alpha=0.2,
            )
            ax.add_patch(rect)

        plt.xticks([], [])
        plt.yticks([], [])

        legend_elements = [
            Patch(
                facecolor=(1, 0, 0.45),
                edgecolor=(1, 0, 0.45),
                label="Table",
                hatch="//////",
                alpha=0.3,
            ),
            Patch(
                facecolor=(0.95, 0.6, 0.1),
                edgecolor=(0.95, 0.6, 0.1),
                label="Table (rotated)",
                hatch="//////",
                alpha=0.3,
            ),
        ]
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.5, -0.02),
            loc="upper center",
            borderaxespad=0,
            fontsize=10,
            ncol=2,
        )
        plt.gcf().set_size_inches(10, 10)
        plt.axis("off")

        if out_path is not None:
            plt.savefig(out_path, bbox_inches="tight", dpi=150)

        return self._fig2img(fig)

    def _iob(bbox1, bbox2):
        """
        Compute the intersection area over box area, for bbox1.
        """
        intersection = Rect(bbox1).intersect(bbox2)

        bbox1_area = Rect(bbox1).get_area()
        if bbox1_area > 0:
            return intersection.get_area() / bbox1_area

        return 0

    def _detected_tables_to_table_crops(
        self,
        page_image: Image.Image,
        tokens: list,
        detected_tables: list,
        class_thresholds: dict,
        padding=10,
    ) -> list:
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """

        table_crops = []
        for obj in detected_tables:
            if obj["score"] < class_thresholds[obj["label"]]:
                continue

            cropped_table = {}

            bbox = obj["bbox"]
            bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

            cropped_img = page_image.crop(bbox)

            table_tokens = [token for token in tokens if self._iob(token["bbox"], bbox) >= 0.5]
            for token in table_tokens:
                token["bbox"] = [
                    token["bbox"][0] - bbox[0],
                    token["bbox"][1] - bbox[1],
                    token["bbox"][2] - bbox[0],
                    token["bbox"][3] - bbox[1],
                ]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj["label"] == "table rotated":
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token["bbox"]
                    bbox = [
                        cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2],
                    ]
                    token["bbox"] = bbox

            cropped_table["image"] = cropped_img
            cropped_table["tokens"] = table_tokens

            table_crops.append(cropped_table)

        return table_crops

    def _extract_tables_from_page(self, page_image: Image.Image) -> list:
        # Detect tables
        detected_tables = self._detect_tables(page_image=page_image)

        # Crop tables out of image
        tokens = (
            []
        )  # Currently unused, but if a table is detected to be rotated, you can rotate it during the detect_tables_to_table_crops step
        detection_class_thresholds = {"table": 0.5, "table rotated": 0.5, "no object": 10}
        crop_padding = 5
        tables_crops = self._detected_tables_to_table_crops(
            page_image=page_image,
            tokens=tokens,
            detected_tables=detected_tables,
            class_thresholds=detection_class_thresholds,
            padding=crop_padding,
        )

        # Convert cropped images to RGB images and add to cropped_table list
        table_images = [table["image"].convert("RGB") for table in tables_crops]

        return table_images

    def _extract_cells_from_table(self, table_image: Image.Image) -> list:

        image_tensor = self.transform_image_to_tensor(image=table_image)

        # forward pass
        with torch.no_grad():
            model_outputs = self.table_structure_model(image_tensor)

        # Convert model output to detected cells bboxes
        detected_cells = self._outputs_to_objects(
            model_outputs=model_outputs,
            image_size=table_image.size,
            class_idx2name=self._get_class_map("structure"),
        )

        return detected_cells

    def visualise_cells(self, page: Image.Image):

        table_images = self._extract_tables_from_page(page=page)

        for table_image in table_images:

            cells = self._extract_cells_from_table(table_image=table_image)

            table_visualized = (
                table_image.copy()
            )  # Make a copy of the image as we're drawing boundary boxes over it in this function and we want to preserve the original image
            draw = ImageDraw.Draw(table_visualized)

            # Draw bboxes around cells
            for cell in cells:
                draw.rectangle(cell["bbox"], outline="red")

            display(table_visualized)

    def _get_cell_coordinates_by_row(self, cells: list) -> list:
        # Extract rows and columns
        rows = [entry for entry in cells if entry["label"] == "table row"]
        columns = [entry for entry in cells if entry["label"] == "table column"]

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x["bbox"][1])
        columns.sort(key=lambda x: x["bbox"][0])

        # Function to find cell coordinates
        def find_cell_coordinates(row, column):
            cell_bbox = [column["bbox"][0], row["bbox"][1], column["bbox"][2], row["bbox"][3]]
            return cell_bbox

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({"column": column["bbox"], "cell": cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x["column"][0])

            # Append row information to cell_coordinates
            cell_coordinates.append(
                {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
            )

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x["row"][1])

        return cell_coordinates

    def ocr_table_cells(self, table_image: Image.Image, cells: list) -> pd.DataFrame:
        cell_coordinates = self._get_cell_coordinates_by_row(cells)

        # OCR row by row
        data = dict()
        max_num_columns = 0
        with tqdm(total=len(cell_coordinates), desc="Cell nr", position=2, leave=False) as pbar:
            for idx, row in enumerate(cell_coordinates):
                row_text = []
                for cell in row["cells"]:
                    # crop cell out of image
                    cell_image = np.array(table_image.crop(cell["cell"]))
                    # apply OCR
                    result = self.ocr_reader.readtext(np.array(cell_image))
                    if len(result) > 0:
                        # print([x[1] for x in list(result)])
                        cell_text = " ".join([x[1] for x in result])
                        row_text.append(cell_text)

                if len(row_text) > max_num_columns:
                    max_num_columns = len(row_text)

                data[idx] = row_text

                pbar.update(1)

        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
            data[row] = row_data

        # put data into dataframe
        data_list = [row for row in data.values()]

        df = pd.DataFrame(data_list[1:], columns=data_list[0])
        return df

    def get_tables_from_pdf(
        self, path: Path, page_nr: None | int = None, skip_first_page: bool = False
    ) -> tuple[list, list]:
        if path.is_file() == False:
            raise ValueError(f"No file found on path:\n{path}")
        else:
            pages = self._load_pdf_pages(path=path)

        if page_nr is not None:
            pages = [pages[page_nr]]
            if skip_first_page == True:
                print(
                    "skip_first_page=True cannot be set when page_nr is given. skip_first_page is ignored"
                )
        if (skip_first_page == True) & (page_nr is None):
            try:
                pages = pages[1::]
            except:
                print(
                    f"{path} contains a pdf with only a single page, so 'skip_first_page=True' argument is ignored."
                )

        df_list = []
        table_images_list = []
        # Loop over pages
        with tqdm(total=len(pages), desc="Page nr", position=0, leave=False) as pbar1:
            for page in pages:
                # Get table images from page
                table_images = self._extract_tables_from_page(page_image=page)

                # Loop over table images
                with tqdm(
                    total=len(table_images), desc="Table nr", position=1, leave=False
                ) as pbar2:
                    for table_image in table_images:
                        # Get cells from table images
                        cells = self._extract_cells_from_table(table_image=table_image)
                        # Apply OCR on cells
                        df = self.ocr_table_cells(table_image=table_image, cells=cells)

                        df_list.append(df)
                        table_images_list.append(table_image)
                        pbar2.update(1)

                pbar1.update(1)
                pbar2.clear()
        pbar1.clear()

        return df_list, table_images_list


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image
