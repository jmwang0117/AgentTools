import cv2
import json
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI


class Qwen3VLDetector:
    """
    使用 Qwen3-VL 进行视频目标检测和 Grounding
    
    坐标系统：Qwen3-VL 使用 0-1000 标准化坐标
    - 输入：无需手动缩放图像，模型内部自动处理
    - 输出：0-1000 范围的标准化坐标，与实际图像尺寸无关
    - 绘制：自动转换为像素坐标进行可视化
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3-vl-plus",
        save_dir: str = "annotated_frames",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"初始化 {model_name}，坐标系统: 0-1000")

    @staticmethod
    def encode_image_to_base64_url(image: np.ndarray) -> str:
        """
        将 OpenCV 图像编码为 base64 URL 格式
        注意：无需手动缩放，模型内部会自动处理
        """
        _, buffer = cv2.imencode(".jpg", image)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    def detect_objects(
        self,
        image: np.ndarray,
        prompt: str = "检测图像中的所有物体，返回每个物体的标签和边界框坐标"
    ) -> List[Dict[str, Any]]:
        """
        调用 Qwen3-VL API 进行目标检测
        
        Args:
            image: OpenCV 图像 (BGR 格式)
            prompt: 检测提示词
            
        Returns:
            检测结果列表，格式: [{"label": "apple", "bbox": [x1, y1, x2, y2]}, ...]
            bbox 为 0-1000 坐标系统
        """
        base64_url = self.encode_image_to_base64_url(image)
        
        system_prompt = """提取图像中所有物体的位置和类别信息，以JSON格式输出。

输出格式要求：
{
  "objects": [
    {
      "label": "字符串类型，物体类别名称",
      "bbox": [x1, y1, x2, y2]  // 数组类型，坐标范围[0-1000]，左上角(x1,y1)，右下角(x2,y2)
    }
  ]
}

参考示例：
Q: 检测图中的水果
A: {"objects": [{"label": "apple", "bbox": [100, 200, 300, 400]}, {"label": "banana", "bbox": [500, 600, 700, 800]}]}

如果图中没有检测到物体，返回空数组：{"objects": []}"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )

            json_string = completion.choices[0].message.content
            result = json.loads(json_string)
            
            objects = result.get("objects", [])
            
            if objects:
                print(f"✓ 检测到 {len(objects)} 个物体: {[obj.get('label') for obj in objects]}")
                for obj in objects:
                    print(f"  - {obj.get('label')}: bbox={obj.get('bbox')} (0-1000)")
            else:
                print("⚠ 未检测到任何物体")

            return objects

        except json.JSONDecodeError as e:
            print(f"[错误] JSON 解析失败: {e}")
            return []
        except Exception as e:
            print(f"[错误] API 调用失败: {e}")
            return []

    @staticmethod
    def bbox_to_pixels(bbox: List[float], img_h: int, img_w: int) -> List[int]:
        """
        将 Qwen3-VL 的 0-1000 坐标转换为像素坐标
        
        Args:
            bbox: [x1, y1, x2, y2] in 0-1000 range
            img_h: 图像高度（像素）
            img_w: 图像宽度（像素）
            
        Returns:
            [x1, y1, x2, y2] in pixel coordinates
        """
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * img_w / 1000),
            int(y1 * img_h / 1000),
            int(x2 * img_w / 1000),
            int(y2 * img_h / 1000)
        ]

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """在图像上绘制检测结果"""
        img_h, img_w = image.shape[:2]
        output = image.copy()

        for det in detections:
            bbox = det.get("bbox")
            label = det.get("label", "unknown")
            
            if not bbox or len(bbox) != 4:
                continue

            # 转换为像素坐标
            x1, y1, x2, y2 = self.bbox_to_pixels(bbox, img_h, img_w)

            # 绘制矩形框
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 绘制标签背景
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(
                output, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )

        return output

    def process_video(
        self,
        video_path: str,
        prompt: str,
        max_frames: Optional[int] = None,
        frame_interval: int = 1,
        start_frame: int = 0,
        save_json: bool = True
    ):
        """
        处理视频文件，逐帧检测并保存结果
        
        Args:
            video_path: 视频文件路径
            prompt: 检测提示词
            max_frames: 最多处理的帧数
            frame_interval: 帧间隔（每隔多少帧处理一次）
            start_frame: 起始帧
            save_json: 是否保存 JSON 结果
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 跳转到指定帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"跳转到第 {start_frame} 帧")

        frame_count = start_frame
        saved_count = 0
        all_results = []

        print(f"开始处理视频: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_count - start_frame) % frame_interval == 0:
                print(f"处理第 {frame_count} 帧...")

                detections = self.detect_objects(frame, prompt)
                annotated_frame = self.draw_detections(frame, detections)

                # 保存图像
                save_path = self.save_dir / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(save_path), annotated_frame)
                
                # 记录结果（同时保存 0-1000 坐标和像素坐标）
                img_h, img_w = frame.shape[:2]
                detections_with_pixels = []
                for det in detections:
                    det_copy = det.copy()
                    bbox_0_1000 = det.get("bbox", [])
                    if len(bbox_0_1000) == 4:
                        bbox_pixels = self.bbox_to_pixels(bbox_0_1000, img_h, img_w)
                        det_copy["bbox_0_1000"] = bbox_0_1000
                        det_copy["bbox_pixels"] = bbox_pixels
                        del det_copy["bbox"]
                    detections_with_pixels.append(det_copy)
                
                frame_result = {
                    "frame": frame_count,
                    "image_path": str(save_path),
                    "image_size": {"width": img_w, "height": img_h},
                    "coordinate_system": "0-1000",
                    "model": self.model_name,
                    "detections": detections_with_pixels
                }
                all_results.append(frame_result)
                
                saved_count += 1

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        
        # 保存 JSON 结果
        if save_json and all_results:
            json_path = self.save_dir / "detections.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"✓ 已保存JSON结果到 {json_path}")
        
        print(f"完成！已保存 {saved_count} 帧到 {self.save_dir}")

    def process_webcam(self, prompt: str, duration_sec: int = 30):
        """从摄像头实时检测并保存结果"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("无法打开摄像头")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = duration_sec * fps
        frame_count = 0

        print(f"开始摄像头录制，持续 {duration_sec} 秒...")

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_objects(frame, prompt)
            annotated_frame = self.draw_detections(frame, detections)

            save_path = self.save_dir / f"webcam_{frame_count:06d}.png"
            cv2.imwrite(str(save_path), annotated_frame)
            frame_count += 1

        cap.release()
        print(f"完成！已保存 {frame_count} 帧")


if __name__ == "__main__":
    API_KEY = "#"
    VIDEO_PATH = "#"

    # 初始化检测器
    detector = Qwen3VLDetector(
        api_key=API_KEY,
        model_name="qwen3-vl-flash",  # 或 qwen3-vl-plus
        save_dir="./AgentTools/output_fruit_detection"
    )

    PROMPT = "检测图像中的所有水果，包括香蕉、苹果等"

    detector.process_video(
        video_path=VIDEO_PATH,
        prompt=PROMPT,
        start_frame=50,
        max_frames=20,
        frame_interval=20,
        save_json=True
    )

