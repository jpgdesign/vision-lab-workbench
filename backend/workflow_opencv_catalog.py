"""Small, framework-free catalog for image and temporal OpenCV operations."""
from copy import deepcopy


def _p(key, label, default, kind=None, **extra):
    return {"key": key, "label": label, "default": default,
            "type": kind or ("boolean" if isinstance(default, bool) else "number" if isinstance(default, (int, float)) else "json" if isinstance(default, list) else "string"), **extra}


OPENCV_SPECS = [
    ("cv_crop", "OpenCV · 裁切 ROI", "依相對座標裁切；框與關節可映射回原圖。ROI 格式為 [左, 上, 右, 下]，範圍 0–1。", [
        _p("roi", "裁切範圍 [左, 上, 右, 下]", [0., 0., 1., 1.])]),
    ("cv_resize", "OpenCV · 縮放", "等比例縮放至指定範圍，或拉伸至指定大小；下游模型使用實際縮放後影像。", [
        _p("width", "目標寬度", 640, min=16, max=2048, step=1),
        _p("height", "目標高度", 480, min=16, max=2048, step=1),
        _p("mode", "縮放方式", "fit", "select", options=["fit", "stretch"])]),
    ("cv_rotate", "OpenCV · 旋轉", "旋轉影像並保留座標映射；正值為逆時針，展開畫布可避免邊角裁切。", [
        _p("angle", "逆時針角度", 0., min=-180, max=180, step=1), _p("expand", "展開畫布", True)]),
    ("cv_color", "OpenCV · 灰階／對比", "灰階、直方圖均衡或 CLAHE 局部對比增強；輸出統一為可接模型的 RGB 畫面。", [
        _p("mode", "色彩處理", "gray", "select", options=["gray", "equalize", "clahe"])]),
    ("cv_blur", "OpenCV · 模糊／降噪", "Gaussian、median 或 bilateral 濾波；核心大小須為奇數。", [
        _p("method", "濾波方法", "gaussian", "select", options=["gaussian", "median", "bilateral"]),
        _p("kernel_size", "核心大小（奇數）", 5, min=1, max=31, step=1),
        _p("sigma", "Gaussian sigma／bilateral 強度", 1.2, min=.1, max=100, step=.1)]),
    ("cv_threshold", "OpenCV · 二值化", "固定閾值、Otsu 或局部自適應二值化；可接形態學與輪廓節點。", [
        _p("method", "二值化方法", "binary", "select", options=["binary", "otsu", "adaptive"]),
        _p("threshold", "固定閾值", 127, min=0, max=255, step=1), _p("invert", "反轉黑白", False),
        _p("block_size", "自適應區塊（奇數）", 11, min=3, max=101, step=1),
        _p("constant", "自適應偏移量 C", 2., min=-100, max=100, step=.5)]),
    ("cv_edges", "OpenCV · Canny 邊緣", "以 Canny 找影像邊緣；低閾值不可大於高閾值。", [
        _p("low", "低閾值", 100, min=0, max=1000, step=1), _p("high", "高閾值", 200, min=0, max=1000, step=1),
        _p("aperture", "Sobel 核心（3／5／7）", 3, min=3, max=7, step=1), _p("l2_gradient", "L2 梯度", False)]),
    ("cv_morphology", "OpenCV · 形態學", "開運算去雜點、閉運算補孔洞，或侵蝕／膨脹／形態梯度；常接在遮罩後。", [
        _p("operation", "運算", "opening", "select", options=["opening", "closing", "erode", "dilate", "gradient"]),
        _p("kernel_size", "核心大小（奇數）", 3, min=1, max=31, step=1),
        _p("iterations", "迭代次數", 1, min=1, max=10, step=1)]),
    ("cv_hsv", "OpenCV · HSV 顏色遮罩", "依 HSV 範圍產生遮罩。H 為 0–179，S/V 為 0–255；H 下限大於上限時跨越紅色色環。", [
        _p("lower", "HSV 下限", [0, 80, 50]), _p("upper", "HSV 上限", [179, 255, 255]), _p("invert", "反轉遮罩", False)]),
    ("cv_contours", "OpenCV · 輪廓／幾何物件", "由二值影像找外輪廓並簡化多邊形，輸出外框供追蹤／區域規則使用；這不是語意辨識。", [
        _p("min_area", "最小輪廓面積（像素²）", 100., min=0, max=4194304, step=1),
        _p("max_area_ratio", "最大面積比例", 1., min=.001, max=1, step=.01),
        _p("threshold", "輪廓前二值化閾值", 127, min=0, max=255, step=1),
        _p("max_objects", "最多輪廓", 40, min=1, max=100, step=1), _p("label", "幾何物件名稱", "contour"),
        _p("approx_epsilon", "多邊形簡化比例", .01, min=.001, max=.2, step=.001)]),
    ("cv_background", "OpenCV · 背景減除／移動", "MOG2／KNN 背景減除；暖機後依前景比例判定 motion/still。僅適合固定攝影機，啟用陰影偵測時排除陰影。", [
        _p("method", "背景模型", "MOG2", "select", options=["MOG2", "KNN"]),
        _p("history", "背景歷史影格", 200, min=2, max=2000, step=1),
        _p("var_threshold", "MOG2 變異閾值／KNN 距離平方閾值", 16., min=1, max=1000, step=1),
        _p("detect_shadows", "偵測並排除陰影", True), _p("learning_rate", "更新率（-1＝自動）", -1., min=-1, max=1, step=.01),
        _p("warmup_frames", "暖機影格數", 10, min=1, max=500, step=1),
        _p("min_motion_ratio", "移動前景比例門檻", .01, min=.0001, max=1, step=.001),
        _p("max_gap_seconds", "重建背景的斷訊間隔（秒）", 1., min=.05, max=30, step=.1)]),
    ("cv_optical_flow", "OpenCV · 稠密光流", "Farneback 光流以來源時間計算平均像素速度；相機移動也會產生 motion，不能當作動作分類。色相表示方向、亮度表示速度。", [
        _p("max_side", "光流計算最長邊", 320, min=64, max=960, step=1),
        _p("levels", "金字塔層數", 3, min=1, max=6, step=1), _p("window_size", "光流視窗（奇數）", 15, min=3, max=51, step=1),
        _p("iterations", "迭代次數", 3, min=1, max=10, step=1),
        _p("min_speed_px_s", "平均移動速度門檻（來源像素／秒）", 10., min=.01, max=10000, step=.5),
        _p("max_gap_seconds", "重新取樣的斷訊間隔（秒）", 1., min=.05, max=30, step=.1)]),
]
OPENCV_KINDS = frozenset(spec[0] for spec in OPENCV_SPECS)
OPENCV_PARAMS = {kind: params for kind, _, _, params in OPENCV_SPECS}


def opencv_catalog(node, param, port):
    result = []
    for kind, label, description, definitions in OPENCV_SPECS:
        outputs = [port("frame", "frame")]
        if kind == "cv_contours": outputs.append(port("detections", "detections"))
        if kind in {"cv_background", "cv_optical_flow"}: outputs.append(port("actions", "actions"))
        params = []
        for definition in definitions:
            values = deepcopy(definition)
            values["kind"] = values.pop("type")
            params.append(param(**values))
        result.append(node(kind, label, "OpenCV", description, [port("frame", "frame")], outputs, params))
    return result
