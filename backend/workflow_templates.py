"""Original example graphs; linked projects are references, not copied applications."""
from .workflow_graph import normalize_graph


def build(identifier, name, description, nodes, edges, sources):
    return {"id": identifier, "name": name, "description": description, "sources": [
        {"title": title, "url": url} for title, url in sources],
        "graph": normalize_graph({"version": 1, "nodes": [
            {"id": ident, "type": kind, "position": {"x": x, "y": y}, "params": params}
            for ident, kind, x, y, params in nodes], "edges": [
                {"id": f"e-{i}", "source": a, "sourceHandle": out, "target": b, "targetHandle": inp}
                for i, (a, out, b, inp) in enumerate(edges)]})}


ZONE = ("ZoneGuard · 功能概念參照", "https://github.com/K-saif/ZoneGuard")
SAFETY = ("AI Safety Monitoring · 功能概念參照（專有授權，未使用程式碼）", "https://github.com/nusRying/ai-safety-monitoring")
PYSKL = ("PYSKL · ST-GCN++ 官方模型", "https://github.com/kennymckormick/pyskl")
MMACTION = ("MMAction2 · 骨架與 RGB 行為辨識", "https://github.com/open-mmlab/mmaction2")
SUPERVISION = ("Supervision · 追蹤與區域分析參考", "https://github.com/roboflow/supervision")
RELATION = ("RelateAnything · 原作者專案", "https://github.com/Maelic/RelateAnything")
ASFORMER = ("ASFormer · 時序動作分段研究參考", "https://github.com/ChinaYi/ASFormer")


def templates():
    base = [("input", "source", 30, 160, {}), ("out", "output", 1300, 160, {})]
    frame = [("input", "frame", "out", "frame")]
    values = [
        build("zone-guard", "區域闖入與停留", "偵測 → 追蹤 → 多邊形區域 → 停留／人數事件。先在畫面上設定監控區域。", base + [
            ("detect", "detector", 330, 160, {}), ("track", "tracker", 650, 160, {}), ("zone", "zone", 970, 160, {})],
            frame + [("input","frame","detect","frame"),("detect","detections","track","observations"),("track","tracks","zone","tracks"),
                     ("zone","tracks","out","observations"),("zone","events","out","events")], [ZONE, SUPERVISION]),
        build("skeleton-actions", "骨架 → 時序動作 → 狀態事件", "YOLO26 Pose → 同人追蹤 → ST-GCN++ NTU120 → 動作規則；需要至少 30 個有效關節影格。", base + [
            ("pose","pose",330,160,{}), ("track","tracker",650,160,{}), ("action","skeleton_action",970,50,{}),
            ("rule","condition",1300,400,{"labels":["falling", "squat down", "hopping (one foot jumping)"],"min_duration":0.})],
            frame + [("input","frame","pose","frame"),("pose","poses","track","observations"),("track","tracks","action","tracks"),
                     ("track","tracks","out","observations"),("action","actions","rule","actions"),("action","actions","out","actions"),("rule","events","out","events")], [PYSKL, MMACTION]),
        build("posture-fall", "姿態轉換／跌倒候選", "幾何姿態規則示範：站立後快速轉躺姿並持續。這是規則候選事件，不是訓練式跌倒分類器。", base + [
            ("pose","pose",330,160,{}),("track","tracker",650,160,{}),("posture","posture",970,160,{}),
            ("rule","condition",1300,450,{"labels":["fall_candidate"],"min_score":.3})],
            frame + [("input","frame","pose","frame"),("pose","poses","track","observations"),("track","tracks","posture","tracks"),
                     ("track","tracks","out","observations"),("posture","actions","out","actions"),("posture","actions","rule","actions"),("rule","events","out","events")], [SAFETY]),
        build("zone-sequence", "檢查區 → 作業區 → 出口", "將同一個人的進區事件串成流程；跳步或逾時產生事件。區域與順序可依現場修改。", base + [
            ("pose","pose",330,160,{}),("track","tracker",650,160,{}),
            ("zone","zone",970,160,{"zones":[{"name":"check","points":[[.02,.05],[.30,.05],[.30,.98],[.02,.98]]},
                {"name":"work","points":[[.35,.05],[.65,.05],[.65,.98],[.35,.98]]},
                {"name":"exit","points":[[.70,.05],[.98,.05],[.98,.98],[.70,.98]]}]}),
            ("sequence","sequence",1300,450,{})],
            frame + [("input","frame","pose","frame"),("pose","poses","track","observations"),("track","tracks","zone","tracks"),
                     ("zone","tracks","out","observations"),("zone","events","sequence","signals"),("zone","events","out","events"),("sequence","events","out","events")], [ZONE, ASFORMER]),
        build("rgb-action-rules", "影片動作 → 規則事件", "直接用 X3D 片段分類並套用動作規則。預設是整個場景；個人模式必須另外接人體追蹤。", base + [
            ("action","video_action",450,160,{}), ("rule","condition",850,160,{"labels":["smoking", "punching person (boxing)", "climbing ladder"],"min_score":.3})],
            frame + [("input","frame","action","frame"),("action","actions","rule","actions"),("action","actions","out","actions"),("rule","events","out","events")], [MMACTION, ("PyTorchVideo · X3D", "https://github.com/facebookresearch/pytorchvideo")]),
        build("ppe-safety", "骨架可見性＋PPE 檢查", "人體骨架與 Grounding DINO 裝備偵測合併。安全帽缺少必須連續、頭部可見，遮擋標示未知。可改用自行微調的 PPE 模型。", base + [
            ("pose","pose",330,40,{}),("track","tracker",650,40,{}),
            ("detect","detector",330,420,{"model_id":"grounding_dino_base","prompt":"helmet . hard hat . safety vest ."}),
            ("ppe","ppe",970,160,{})],
            frame + [("input","frame","pose","frame"),("pose","poses","track","observations"),("input","frame","detect","frame"),
                     ("track","tracks","ppe","tracks"),("detect","detections","ppe","detections"),("track","tracks","out","observations"),
                     ("detect","detections","out","observations"),("ppe","events","out","events")], [SAFETY]),
        build("object-relations", "物件 → 追蹤 → 關係辨識", "任意偵測器提供外框，再用本機 RelateAnything ConvNeXt-L 判定關係；可換成自己的微調物件模型。", base + [
            ("detect","detector",330,160,{}),("track","tracker",650,160,{"labels":[]}), ("relation","relation",970,160,{})],
            frame + [("input","frame","detect","frame"),("detect","detections","track","observations"),("input","frame","relation","frame"),
                     ("track","tracks","relation","objects"),("track","tracks","out","observations"),("relation","relations","out","relations")], [RELATION]),
        build("cloud-review", "事件＋雲端視覺判讀", "區域事件提供上下文，取樣影像由已設定的 API 模型判讀。需有效金鑰與視覺模型 ID；取樣影像會送往服務商。", base + [
            ("detect","detector",330,160,{}),("track","tracker",650,160,{}),("zone","zone",970,160,{}),("review","vlm",970,500,{})],
            frame + [("input","frame","detect","frame"),("detect","detections","track","observations"),("track","tracks","zone","tracks"),
                     ("input","frame","review","frame"),("zone","events","review","events"),("zone","tracks","out","observations"),
                     ("zone","events","out","events"),("review","actions","out","actions")], [SAFETY]),
    ]
    return {"templates": values}
