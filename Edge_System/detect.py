# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0
                                                     img.jpg
                                                     vid.mp4
                                                     screen
                                                     path/
                                                     list.txt
                                                     list.streams
                                                     'path/*.jpg'
                                                     'https://youtu.be/LNwODJXcvt4'
                                                     'rtsp://example.com/media.mp4'

Usage - formats:
    $ python detect.py --weights yolov5s.pt
                                 yolov5s.torchscript
                                 yolov5s.onnx
                                 yolov5s_openvino_model
                                 yolov5s.engine
                                 yolov5s.mlpackage
                                 yolov5s_saved_model
                                 yolov5s.pb
                                 yolov5s.tflite
                                 yolov5s_edgetpu.tflite
                                 yolov5s_paddle_model
"""

import argparse
import csv
import os
import platform
import sys

from glob import glob, has_magic
from pathlib import Path

import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(
        str(ROOT)
    )

ROOT = Path(
    os.path.relpath(
        ROOT,
        Path.cwd()
    )
)


from ultralytics.utils.plotting import (
    Annotator,
    colors,
    save_one_box
)

from models.common import DetectMultiBackend

from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams
)

from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)

from utils.torch_utils import (
    select_device,
    smart_inference_mode
)


# =========================================================
# 추가
# 객체 출현 변화 감지 및 Django 서버 전송
# =========================================================


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):

    source = str(source)

    is_file = (
        Path(source).suffix[1:]
        in (
            IMG_FORMATS
            + VID_FORMATS
        )
    )

    is_url = source.lower().startswith(
        (
            "rtsp://",
            "rtmp://",
            "http://",
            "https://"
        )
    )

    webcam = (
        source.isnumeric()
        or source.endswith(
            ".streams"
        )
        or (
            is_url
            and not is_file
        )
    )

    screenshot = (
        source
        .lower()
        .startswith(
            "screen"
        )
    )


    if not (
        webcam
        or screenshot
        or is_url
    ) and not (
        Path(source).exists()
        or (
            has_magic(source)
            and glob(
                source,
                recursive=True
            )
        )
    ):

        raise FileNotFoundError(
            f"Source path '{source}' does not exist"
        )


    save_img = (
        not nosave
        and not source.endswith(
            ".txt"
        )
    )


    if (
        is_url
        and is_file
    ):

        source = check_file(
            source
        )


    # =====================================================
    # 저장 디렉토리
    # =====================================================

    save_dir = increment_path(
        Path(project) / name,
        exist_ok=exist_ok
    )

    (
        save_dir / "labels"
        if save_txt
        else save_dir
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # 모델 로드
    # =====================================================

    device = select_device(
        device
    )

    model = DetectMultiBackend(
        weights,
        device=device,
        dnn=dnn,
        data=data,
        fp16=half
    )

    stride = model.stride
    names = model.names
    pt = model.pt

    imgsz = check_img_size(
        imgsz,
        s=stride
    )


    # =====================================================
    # Dataloader
    # =====================================================

    bs = 1


    if webcam:

        view_img = check_imshow(
            warn=True
        )

        dataset = LoadStreams(
            source,
            img_size=imgsz,
            stride=stride,
            auto=pt,
            vid_stride=vid_stride
        )

        bs = len(
            dataset
        )


    elif screenshot:

        dataset = LoadScreenshots(
            source,
            img_size=imgsz,
            stride=stride,
            auto=pt
        )


    else:

        dataset = LoadImages(
            source,
            img_size=imgsz,
            stride=stride,
            auto=pt,
            vid_stride=vid_stride
        )


    vid_path = [
        None
    ] * bs

    vid_writer = [
        None
    ] * bs


    # =====================================================
    # CSV 설정
    # =====================================================

    csv_path = (
        save_dir
        / "predictions.csv"
    )


    def write_to_csv(
        image_name,
        prediction,
        confidence
    ):

        data = {
            "Image Name":
                image_name,

            "Prediction":
                prediction,

            "Confidence":
                confidence
        }

        file_exists = os.path.isfile(
            csv_path
        )

        with open(
            csv_path,
            mode="a",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=data.keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                data
            )


    # =====================================================
    # Inference 시작
    # =====================================================

    model.warmup(
        imgsz=(
            1
            if pt
            or model.triton
            else bs,
            3,
            *imgsz
        )
    )


    seen = 0

    windows = []

    dt = (
        Profile(
            device=device
        ),
        Profile(
            device=device
        ),
        Profile(
            device=device
        )
    )


    for (
        path,
        im,
        im0s,
        vid_cap,
        s
    ) in dataset:


        # =================================================
        # 전처리
        # =================================================

        with dt[0]:

            im = torch.from_numpy(
                im
            ).to(
                model.device
            )

            im = (
                im.half()
                if model.fp16
                else im.float()
            )

            im /= 255


            if len(
                im.shape
            ) == 3:

                im = im[
                    None
                ]


            if (
                model.xml
                and im.shape[0] > 1
            ):

                ims = torch.chunk(
                    im,
                    im.shape[0],
                    0
                )


        # =================================================
        # Inference
        # =================================================

        with dt[1]:

            if (
                model.xml
                and im.shape[0] > 1
            ):

                pred = None

                for image in ims:

                    if pred is None:

                        pred = model(
                            image,
                            augment=augment
                        ).unsqueeze(
                            0
                        )

                    else:

                        pred = torch.cat(
                            (
                                pred,

                                model(
                                    image,
                                    augment=augment
                                ).unsqueeze(
                                    0
                                )
                            ),
                            dim=0
                        )


                pred = [
                    pred,
                    None
                ]


            else:

                pred = model(
                    im,
                    augment=augment
                )


        # =================================================
        # NMS
        # =================================================

        with dt[2]:

            pred = non_max_suppression(
                pred,
                conf_thres,
                iou_thres,
                classes,
                agnostic_nms,
                max_det=max_det
            )


        # =================================================
        # Prediction 처리
        # =================================================

        for i, det in enumerate(
            pred
        ):

            seen += 1


            # =================================================
            # 추가
            # 현재 프레임 객체 존재 상태
            #
            # 처음에는 모든 클래스가 0
            # =================================================


            if webcam:

                p = path[i]

                im0 = (
                    im0s[i]
                    .copy()
                )

                frame = (
                    dataset.count
                )

                s += (
                    f"{i}: "
                )


            else:

                p = path

                im0 = (
                    im0s.copy()
                )

                frame = getattr(
                    dataset,
                    "frame",
                    0
                )


            p = Path(
                p
            )


            save_path = str(
                save_dir
                / p.name
            )


            txt_path = (
                str(
                    save_dir
                    / "labels"
                    / p.stem
                )
                + (
                    ""
                    if dataset.mode == "image"
                    else f"_{frame}"
                )
            )


            s += (
                "{:g}x{:g} "
                .format(
                    *im.shape[2:]
                )
            )


            gn = torch.tensor(
                im0.shape
            )[
                [
                    1,
                    0,
                    1,
                    0
                ]
            ]


            imc = (
                im0.copy()
                if save_crop
                else im0
            )


            annotator = Annotator(
                im0,
                line_width=line_thickness,
                example=str(
                    names
                )
            )


            # =================================================
            # 객체 검출 결과 존재
            # =================================================

            if len(det):

                det[:, :4] = scale_boxes(
                    im.shape[2:],
                    det[:, :4],
                    im0.shape
                ).round()


                # =============================================
                # 검출 결과 출력
                # =============================================

                for c in det[
                    :,
                    5
                ].unique():

                    n = (
                        det[
                            :,
                            5
                        ]
                        == c
                    ).sum()


                    s += (
                        f"{n} "
                        f"{names[int(c)]}"
                        f"{'s' * (n > 1)}, "
                    )


                # =============================================
                # 객체별 처리
                # =============================================

                for (
                    *xyxy,
                    conf,
                    cls
                ) in reversed(
                    det
                ):

                    c = int(
                        cls
                    )


                    # =========================================
                    # 추가
                    # 현재 프레임에서 해당 클래스가 존재함
                    # =========================================


                    label = (
                        names[c]
                    )

                    confidence = float(
                        conf
                    )

                    confidence_str = (
                        f"{confidence:.2f}"
                    )


                    # =========================================
                    # CSV 저장
                    # =========================================

                    if save_csv:

                        write_to_csv(
                            p.name,
                            label,
                            confidence_str
                        )


                    # =========================================
                    # TXT 저장
                    # =========================================

                    if save_txt:

                        if save_format == 0:

                            coords = (
                                (
                                    xyxy2xywh(
                                        torch.tensor(
                                            xyxy
                                        ).view(
                                            1,
                                            4
                                        )
                                    )
                                    / gn
                                )
                                .view(
                                    -1
                                )
                                .tolist()
                            )


                        else:

                            coords = (
                                (
                                    torch.tensor(
                                        xyxy
                                    )
                                    .view(
                                        1,
                                        4
                                    )
                                    / gn
                                )
                                .view(
                                    -1
                                )
                                .tolist()
                            )


                        line = (
                            (
                                cls,
                                *coords,
                                conf
                            )
                            if save_conf
                            else (
                                cls,
                                *coords
                            )
                        )


                        with open(
                            f"{txt_path}.txt",
                            "a"
                        ) as f:

                            f.write(
                                (
                                    "%g "
                                    * len(
                                        line
                                    )
                                )
                                .rstrip()
                                % line
                                + "\n"
                            )


                    # =========================================
                    # Bounding Box 표시
                    # =========================================

                    if (
                        save_img
                        or save_crop
                        or view_img
                    ):

                        c = int(
                            cls
                        )


                        label = (
                            None
                            if hide_labels
                            else (
                                names[c]
                                if hide_conf
                                else (
                                    f"{names[c]} "
                                    f"{conf:.2f}"
                                )
                            )
                        )


                        annotator.box_label(
                            xyxy,
                            label,
                            color=colors(
                                c,
                                True
                            )
                        )


                    # =========================================
                    # Crop 저장
                    # =========================================

                    if save_crop:

                        save_one_box(
                            xyxy,
                            imc,
                            file=(
                                save_dir
                                / "crops"
                                / names[c]
                                / f"{p.stem}.jpg"
                            ),
                            BGR=True
                        )


            # =================================================
            # Annotation 적용
            # =================================================

            im0 = annotator.result()


            # =================================================
            # 화면 출력
            # =================================================

            if view_img:

                if (
                    platform.system()
                    == "Linux"
                    and p not in windows
                ):

                    windows.append(
                        p
                    )

                    cv2.namedWindow(
                        str(
                            p
                        ),
                        cv2.WINDOW_NORMAL
                        | cv2.WINDOW_KEEPRATIO
                    )

                    cv2.resizeWindow(
                        str(
                            p
                        ),
                        im0.shape[1],
                        im0.shape[0]
                    )


                cv2.imshow(
                    str(
                        p
                    ),
                    im0
                )

                cv2.waitKey(
                    1
                )


            # =================================================
            # 결과 저장
            # =================================================

            if save_img:

                if dataset.mode == "image":

                    cv2.imwrite(
                        save_path,
                        im0
                    )


                else:

                    if (
                        vid_path[i]
                        != save_path
                    ):

                        vid_path[i] = (
                            save_path
                        )


                        if isinstance(
                            vid_writer[i],
                            cv2.VideoWriter
                        ):

                            vid_writer[
                                i
                            ].release()


                        if vid_cap:

                            fps = vid_cap.get(
                                cv2.CAP_PROP_FPS
                            )

                            w = int(
                                vid_cap.get(
                                    cv2.CAP_PROP_FRAME_WIDTH
                                )
                            )

                            h = int(
                                vid_cap.get(
                                    cv2.CAP_PROP_FRAME_HEIGHT
                                )
                            )


                        else:

                            fps = 30

                            w = (
                                im0.shape[1]
                            )

                            h = (
                                im0.shape[0]
                            )


                        save_path = str(
                            Path(
                                save_path
                            ).with_suffix(
                                ".mp4"
                            )
                        )


                        vid_writer[i] = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            ),
                            fps,
                            (
                                w,
                                h
                            )
                        )


                    vid_writer[
                        i
                    ].write(
                        im0
                    )


        # =====================================================
        # 시간 출력
        # =====================================================

        LOGGER.info(
            f"{s}"
            f"{'' if len(det) else '(no detections), '}"
            f"{dt[1].dt * 1e3:.1f}ms"
        )


    # =========================================================
    # 결과 출력
    # =========================================================

    t = tuple(
        x.t
        / seen
        * 1e3
        for x in dt
    )


    LOGGER.info(
        "Speed: "
        "%.1fms pre-process, "
        "%.1fms inference, "
        "%.1fms NMS per image "
        f"at shape {(1, 3, *imgsz)}"
        % t
    )


    if (
        save_txt
        or save_img
    ):

        s = (
            (
                f"\n"
                f"{len(list(save_dir.glob('labels/*.txt')))} "
                f"labels saved to "
                f"{save_dir / 'labels'}"
            )
            if save_txt
            else ""
        )


        LOGGER.info(
            f"Results saved to "
            f"{colorstr('bold', save_dir)}"
            f"{s}"
        )


    if update:

        strip_optimizer(
            weights[0]
            if isinstance(
                weights,
                (
                    list,
                    tuple
                )
            )
            else weights
        )


def parse_opt():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT
        / "yolov5s.pt",
        help="model path or triton URL"
    )


    parser.add_argument(
        "--source",
        type=str,
        default=ROOT
        / "data/images",
        help="file/dir/URL/glob/screen/0(webcam)"
    )


    parser.add_argument(
        "--data",
        type=str,
        default=ROOT
        / "data/coco128.yaml",
        help="(optional) dataset.yaml path"
    )


    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[
            640
        ],
        help="inference size h,w"
    )


    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="confidence threshold"
    )


    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold"
    )


    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="maximum detections per image"
    )


    parser.add_argument(
        "--device",
        default="",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )


    parser.add_argument(
        "--view-img",
        action="store_true",
        help="show results"
    )


    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="save results to *.txt"
    )


    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help=(
            "whether to save boxes coordinates "
            "in YOLO format or Pascal-VOC format "
            "when save-txt is True"
        )
    )


    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="save results in CSV format"
    )


    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="save confidences in --save-txt labels"
    )


    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="save cropped prediction boxes"
    )


    parser.add_argument(
        "--nosave",
        action="store_true",
        help="do not save images/videos"
    )


    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3"
    )


    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="class-agnostic NMS"
    )


    parser.add_argument(
        "--augment",
        action="store_true",
        help="augmented inference"
    )


    parser.add_argument(
        "--update",
        action="store_true",
        help="update all models"
    )


    parser.add_argument(
        "--project",
        default=ROOT
        / "runs/detect",
        help="save results to project/name"
    )


    parser.add_argument(
        "--name",
        default="exp",
        help="save results to project/name"
    )


    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok"
    )


    parser.add_argument(
        "--line-thickness",
        default=3,
        type=int,
        help="bounding box thickness"
    )


    parser.add_argument(
        "--hide-labels",
        default=False,
        action="store_true",
        help="hide labels"
    )


    parser.add_argument(
        "--hide-conf",
        default=False,
        action="store_true",
        help="hide confidences"
    )


    parser.add_argument(
        "--half",
        action="store_true",
        help="use FP16 half-precision inference"
    )


    parser.add_argument(
        "--dnn",
        action="store_true",
        help="use OpenCV DNN for ONNX inference"
    )


    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="video frame-rate stride"
    )


    opt = parser.parse_args()


    opt.imgsz *= (
        2
        if len(
            opt.imgsz
        ) == 1
        else 1
    )


    print_args(
        vars(
            opt
        )
    )


    return opt


def main(
    opt
):

    check_requirements(
        ROOT
        / "requirements.txt",
        exclude=(
            "tensorboard",
            "ultralytics-thop"
        )
    )


    run(
        **vars(
            opt
        )
    )


if __name__ == "__main__":

    opt = parse_opt()

    main(
        opt
    )