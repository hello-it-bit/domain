import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import random

import decord
decord.bridge.set_bridge("native")

HEADER = ["path", "start_frame", "end_frame", "fps", "mask_id", "caption"]

FIXED_START = 0
FIXED_END = 49  

def infer_from_template(template_csv: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with template_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fps = int(row["fps"])
                mask_id = int(row["mask_id"])
                return fps, mask_id
    except Exception as e:
        print(f"[warn] 无法从 {template_csv} 推断 fps/mask_id：{e}")
    return None, None


def parse_replace_pairs(pairs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--replace_prefix 参数格式应为 旧=新，收到：{p}")
        old, new = p.split("=", 1)
        out.append((old.rstrip("/"), new.rstrip("/")))
    return out


def normalize_path(p: str, path_prefix: Optional[str], replacements: List[Tuple[str, str]]) -> str:
    s = str(p or "").strip()
    s = s.lstrip("/") 

    for old, new in replacements:
        if s.startswith(old + "/") or s == old:
            s = new + s[len(old):]
            break

    if path_prefix:
        s = str(Path(path_prefix) / s)

    return s


def get_num_frames(video_path: str) -> int:
    vr = decord.VideoReader(video_path)
    return len(vr)


def pick_path_and_caption(item: Dict) -> Tuple[str, str]:

    raw_path = item.get("video_path") or item.get("path") or ""
    caption = item.get("video_caption") or item.get("text") or ""
    return str(raw_path), str(caption)


def to_row(item: Dict,
           fps: int,
           mask_id: int,
           path_prefix: Optional[str],
           replacements: List[Tuple[str, str]]) -> Dict[str, str]:
    raw_path, caption = pick_path_and_caption(item)
    path = normalize_path(raw_path, path_prefix, replacements)

    return {
        "path": path,
        "start_frame": str(FIXED_START),
        "end_frame": str(FIXED_END),
        "fps": str(int(fps)),
        "mask_id": str(int(mask_id)),
        "caption": caption,
    }


def validate_and_maybe_shorten(row: Dict[str, str], auto_shorten: bool, log: List[str]) -> bool:
    path = row["path"]
    if not path.lower().endswith(".mp4"):
        log.append(f"SKIP(non-mp4): {path}")
        return False

    p = Path(path)
    if not p.is_file():
        log.append(f"DROP(missing): {path}")
        return False

    try:
        num = get_num_frames(path)
    except Exception as ex:
        log.append(f"DROP(read_fail): {path} ({ex})")
        return False

    end = int(row["end_frame"])
    if end > num:
        if auto_shorten:
            row["end_frame"] = str(num)  
            log.append(f"SHORTEN(end>num): {path} end {end}->{num}")
            return num > 0
        else:
            log.append(f"DROP(end>num): {path} end={end}, num={num}")
            return False

    return True


def write_csv(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def passes_filter(item: Dict, filter_type: Optional[str]) -> bool:
    if filter_type is None:
        return True
    t = item.get("type")
    c = item.get("class")
    return (t == filter_type) or (c == filter_type)


def main():
    ap = argparse.ArgumentParser(
        description="从 JSON 批量生成 crop.csv（固定 start=0, end=49 右开），做存在/帧数校验，并复制为 val.csv。"
    )
    ap.add_argument("--input_dir", type=str, required=True, help="包含多个 JSON 的目录（每个 JSON 是条目数组）")
    ap.add_argument("--output_dir", type=str, required=True, help="输出目录：crop1.csv / val1.csv")
    ap.add_argument("--fps", type=int, default=None, help="帧率；若不指定，会尝试用 --template_csv 推断，否则默认 8")
    ap.add_argument("--mask_id", type=int, default=None, help="mask_id；若不指定，会尝试用 --template_csv 推断，否则默认 2")
    ap.add_argument("--template_csv", type=str, default=None, help="一份已有的 crop.csv，用于自动读取 fps/mask_id")
    ap.add_argument("--path_prefix", type=str, default=None, help="可选：给 JSON 的 path 统一加的前缀（数据根）")
    ap.add_argument("--replace_prefix", action="append", default=[],
                    help="替换错误前缀：格式 旧=新，可多次提供")
    ap.add_argument("--shuffle", action="store_true", help="在汇总前打乱每个 JSON 的条目顺序（可选）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（在 --shuffle 时生效）")
    ap.add_argument("--filter_type", type=str, default=None,
                    help="仅保留满足 item['type']==此值 或 item['class']==此值 的记录。默认不过滤。")
    ap.add_argument("--auto-shorten", action="store_true",
                    help="若视频帧数不足 end_frame，则自动把 end_frame 缩到真实帧数；默认直接丢弃该条")
    ap.add_argument("--report", type=str, default=None, help="修复/丢弃日志输出到该文件；缺省打印到控制台")
    args = ap.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps, mask_id = args.fps, args.mask_id
    if (fps is None or mask_id is None) and args.template_csv:
        inf_fps, inf_mask = infer_from_template(Path(args.template_csv))
        if fps is None:
            fps = inf_fps
        if mask_id is None:
            mask_id = inf_mask
    if fps is None:
        fps = 8
    if mask_id is None:
        mask_id = 2

    report_fp = None
    try:
        if args.report:
            rp = Path(args.report)
            rp.parent.mkdir(parents=True, exist_ok=True)
            report_fp = rp.open("w", encoding="utf-8")

        def log(msg: str):
            (report_fp.write(msg + "\n") if report_fp else print(msg))

        crop_rows: List[Dict[str, str]] = []
        replacements = parse_replace_pairs(args.replace_prefix)

        json_files = sorted(input_dir.glob("*.json"))
        if not json_files:
            log(f"[info] {input_dir} 下未发现 JSON 文件。")
            return

        for jf in json_files:
            try:
                data = json.loads(Path(jf).read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    log(f"[warn] {jf.name} 不是数组，已跳过。")
                    continue

                items = [d for d in data if isinstance(d, dict) and passes_filter(d, args.filter_type)]
                if not items:
                    log(f"[warn] {jf.name} 无可用条目（可能被 filter_type 过滤），已跳过。")
                    continue

                if args.shuffle:
                    random.shuffle(items)

                kept = 0
                for it in items:
                    r = to_row(it, fps=fps, mask_id=mask_id,
                               path_prefix=args.path_prefix, replacements=replacements)
                    row_log: List[str] = []
                    if validate_and_maybe_shorten(r, auto_shorten=args.auto_shorten, log=row_log):
                        crop_rows.append(r)
                        kept += 1
                    for m in row_log:
                        log(m)

                log(f"[ok] {jf.name}: crop_keep {kept} / {len(items)}")

            except Exception as e:
                log(f"[error] 处理 {jf.name} 失败：{e}")

        crop_csv = output_dir / "crop1.csv"
        write_csv(crop_csv, crop_rows)

        val_csv = output_dir / "val1.csv"
        shutil.copyfile(crop_csv, val_csv)

        log("[done] 结果：")
        log(f" - {crop_csv} ({len(crop_rows)} 行)")
        log(f" - {val_csv}   ({len(crop_rows)} 行，来自 crop.csv 复制)")

    finally:
        if report_fp:
            report_fp.close()


if __name__ == "__main__":
    main()
