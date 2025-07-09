# app.py
import threading
import time
import cv2
import numpy as np
import ast
import tkinter as tk
from config.config import load_config
from models.rknn_model import RKNNModel
from models.freshness_model import FreshnessModel
from hardware.camera import Camera
from hardware.serial_comm import SerialComm
from ui.main_window import MainWindow
from api.deepseek_api import DeepSeekAPI
from utils.logger import get_logger
from models.coco_utils import COCO_test_helper  # 修正导入

class EggDetectionApp:
    def __init__(self, root):
        self.logger = get_logger("EggDetectionApp")
        self.config = load_config()
        self.root = root
        self.running = True
        self.detection_enabled = False
        self.detection_running = False
        self.current_egg_id = 0
        self.record_count = 0
        self.last_shell_status = None
        self.current_freshness = "未检测"
        self.last_detection_results = {1: None, 2: None}
        self.deepseek_analysis = {}
        self.CLASSES = ['egg', 'break']

        # 初始化模块
        self.ui = MainWindow(root)
        self.rknn_model = RKNNModel(self.config['models']['rknn']['path'])
        self.freshness_model = FreshnessModel(self.config['models']['freshness']['path'])
        self.cameras = {
            1: Camera(**self.config['cameras']['front']),
            2: Camera(**self.config['cameras']['back'])
        }
        self.serial = SerialComm(**self.config['serial'])
        self.deepseek = DeepSeekAPI(**self.config['deepseek'])
        self.co_helper = COCO_test_helper()

        # 设置UI回调
        self.setup_ui_callbacks()

        # 启动线程
        self.start_threads()

        # 初始化状态
        self.ui.update_status("系统初始化完成，等待开始检测")

    def setup_ui_callbacks(self):
        """绑定UI按钮事件"""
        self.ui.ui['btn_detect'].config(command=self.start_detection)
        self.ui.ui['btn_reset'].config(command=self.reset_system)
        self.ui.ui['btn_exit'].config(command=self.exit_system)

    def start_threads(self):
        """启动后台线程"""
        threading.Thread(target=self.serial_receive_loop, daemon=True).start()
        threading.Thread(target=self.display_loop, daemon=True).start()

    def start_detection(self):
        """开始检测流程"""
        if not self.detection_enabled:
            self.current_egg_id = 0
            self.serial.send("Start")
            self.detection_enabled = True
            self.ui.update_status("启动成功，等待 openA 指令...")
            self.ui.ui['btn_detect'].config(state=tk.DISABLED)

    def serial_receive_loop(self):
        """串口接收循环"""
        while self.running:
            try:
                for message in self.serial.receive():
                    self.logger.info(f"接收到串口消息: {message}")
                    self.serial.send("Start")
                    self.root.after(10, self.process_serial_message, message)
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"串口接收错误: {e}")
                self.ui.update_status(f"串口接收错误: {str(e)}")
                time.sleep(0.1)

    def process_serial_message(self, message):
        """处理串口消息"""
        try:
            if message == "openA" and self.detection_enabled:
                self.logger.info("接收到 openA 指令，开始检测...")
                self.ui.update_status("接收到 openA 指令，开始检测...")
                if not self.detection_running:
                    threading.Thread(target=self.detect_egg, daemon=True).start()
            elif message.startswith("[") and message.endswith("]"):
                if self.last_shell_status in ["完好", "不确定"]:
                    try:
                        data = ast.literal_eval(message)
                        if isinstance(data, list) and len(data) == 24:
                            result, grade = self.freshness_model.predict(data)
                            self.serial.send(result)
                            self.ui.ui['fresh_info_label'].config(text=f"新鲜度检测结果：{grade}")
                            self.ui.update_status(f"鸡蛋{self.current_egg_id}检测完成 - 外壳{self.last_shell_status}，新鲜度:{grade}")
                            self.last_shell_status = None
                            self.detection_enabled = True
                            self.serial.send("Start")
                        else:
                            self.logger.error(f"新鲜度数据错误 - 长度为 {len(data)}，应为24")
                            self.ui.update_status(f"新鲜度数据错误 - 长度为 {len(data)}，应为24")
                    except SyntaxError as e:
                        self.logger.error(f"列表数据格式错误: {e}, 原始数据: {message}")
                        self.ui.update_status("列表数据格式错误")
        except Exception as e:
            self.logger.error(f"处理串口消息错误: {e}")
            self.ui.update_status(f"处理串口消息错误: {str(e)}")

    def detect_egg(self):
        """执行鸡蛋检测逻辑"""
        if self.detection_running:
            return
        self.detection_running = True
        self.current_egg_id += 1
        self.ui.update_status(f"开始检测鸡蛋 {self.current_egg_id}...")

        front_broken = False
        back_broken = False
        if self.last_detection_results[1]:
            _, _, classes1, _ = self.last_detection_results[1]
            front_broken = any(self.CLASSES[int(cls)] == 'break' for cls in classes1)
        if self.last_detection_results[2]:
            _, _, classes2, _ = self.last_detection_results[2]
            back_broken = any(self.CLASSES[int(cls)] == 'break' for cls in classes2)

        egg_broken = front_broken or back_broken
        if egg_broken:
            self.serial.send("AN")
            self.last_shell_status = "破损"
            self.ui.update_status(f"鸡蛋 {self.current_egg_id} 外壳破损，记录新鲜度为'无'")
            self.ui.ui['shell_info_label'].config(text="外壳检测结果：破损")
            self.add_egg_record("破损", "无")
            self.detection_enabled = True
            self.detection_running = False
            self.serial.send("Start")
        else:
            self.serial.send("AY")
            self.last_shell_status = "完好"
            self.ui.update_status(f"鸡蛋 {self.current_egg_id} 外壳完好，等待新鲜度检测...")
            self.ui.ui['shell_info_label'].config(text="外壳检测结果：完好")
            self.detection_running = False

    def draw_detection_boxes(self, frame, boxes, classes, scores):
        """在图像上绘制检测框"""
        for box, score, cl in zip(boxes, scores, classes):
            left, top, right, bottom = [int(_b) for _b in box]
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, '{0} {1:.2f}'.format(self.CLASSES[int(cl)], score),
                        (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def post_process(self, outputs):
        """后处理检测结果"""
        OBJ_THRESH = 0.5
        NMS_THRESH = 0.45
        MODEL_SIZE = tuple(self.config['models']['rknn']['input_size'])

        def filter_boxes(boxes, box_confidences, box_class_probs):
            box_confidences = box_confidences.reshape(-1)
            candidate, class_num = box_class_probs.shape
            class_max_score = np.max(box_class_probs, axis=-1)
            classes = np.argmax(box_class_probs, axis=-1)
            _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
            scores = (class_max_score * box_confidences)[_class_pos]
            boxes = boxes[_class_pos]
            classes = classes[_class_pos]
            return boxes, classes, scores

        def nms_boxes(boxes, scores):
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            areas = w * h
            order = scores.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x[i], x[order[1:]])
                yy1 = np.maximum(y[i], y[order[1:]])
                xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
                yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
                w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
                h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
                inter = w1 * h1
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= NMS_THRESH)[0]
                order = order[inds + 1]
            return np.array(keep)

        def dfl(position):
            n, c, h, w = position.shape
            p_num = 4
            mc = c // p_num
            y = position.reshape(n, p_num, mc, h, w)
            y = np.exp(y - np.max(y, axis=2, keepdims=True))
            y = y / np.sum(y, axis=2, keepdims=True)
            acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
            return (y * acc_metrix).sum(2)

        def box_process(position):
            grid_h, grid_w = position.shape[2:4]
            col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
            col = col.reshape(1, 1, grid_h, grid_w)
            row = row.reshape(1, 1, grid_h, grid_w)
            grid = np.concatenate((col, row), axis=1)
            stride = np.array([MODEL_SIZE[1] // grid_h, MODEL_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
            position = dfl(position)
            box_xy = grid + 0.5 - position[:, 0:2, :, :]
            box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
            xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
            return xyxy

        boxes, scores, classes_conf = [], [], []
        defualt_branch = 3
        pair_per_branch = len(outputs) // defualt_branch

        for i in range(defualt_branch):
            boxes.append(box_process(outputs[pair_per_branch * i]))
            classes_conf.append(outputs[pair_per_branch * i + 1])
            scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
        nboxes, nclasses, nscores = [], [], []

        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s)
            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        return boxes, classes, scores

    def add_egg_record(self, shell_status, freshness):
        """添加鸡蛋检测记录并发送到DeepSeek API"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.record_count += 1
        self.ui.add_record(self.record_count, shell_status, freshness, current_time)

        egg_data = {
            "egg_id": self.record_count,
            "shell_status": shell_status,
            "freshness": freshness,
            "detection_time": current_time
        }
        threading.Thread(target=self.send_to_deepseek, args=(egg_data,), daemon=True).start()

    def send_to_deepseek(self, egg_data):
        """发送鸡蛋数据到DeepSeek API"""
        analysis = self.deepseek.analyze_egg(egg_data)
        self.deepseek_analysis[egg_data['egg_id']] = analysis
        self.root.after(0, self.display_analysis_gradually, egg_data['egg_id'], analysis)

    def display_analysis_gradually(self, egg_id, analysis):
        """逐字显示DeepSeek分析结果"""
        try:
            self.ui.ui['analysis_text'].configure(state=tk.NORMAL)
            self.ui.ui['analysis_text'].delete(1.0, tk.END)
            content = f"鸡蛋 {egg_id} 分析结果:\n"
            self.ui.ui['analysis_text'].insert(tk.END, content)

            def type_character(index=0):
                if index < len(analysis):
                    self.ui.ui['analysis_text'].insert(tk.END, analysis[index])
                    self.ui.ui['analysis_text'].see(tk.END)
                    self.root.after(10, type_character, index + 1)
                else:
                    self.ui.ui['analysis_text'].configure(state=tk.DISABLED)

            type_character()
            self.logger.info(f"开始逐字显示分析结果：鸡蛋 {egg_id}")
        except Exception as e:
            self.logger.error(f"逐字显示分析结果失败：{e}")
            self.ui.ui['analysis_text'].configure(state=tk.NORMAL)
            self.ui.ui['analysis_text'].delete(1.0, tk.END)
            self.ui.ui['analysis_text'].insert(tk.END, f"鸡蛋 {egg_id} 分析失败")
            self.ui.ui['analysis_text'].configure(state=tk.DISABLED)

    def display_loop(self):
        """显示摄像头画面并持续推理检测框"""
        MODEL_SIZE = tuple(self.config['models']['rknn']['input_size'])
        while self.running:
            try:
                frames = {pos: cam.read() for pos, cam in self.cameras.items()}
                if any(frame is None for frame in frames.values()):
                    self.ui.update_status("摄像头读取失败")
                    time.sleep(0.1)
                    continue

                for pos, frame in frames.items():
                    img = self.co_helper.letter_box(im=frame.copy(), new_shape=MODEL_SIZE, pad_color=(0, 0, 0))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    outputs = self.rknn_model.infer(np.expand_dims(img, axis=0))
                    boxes, classes, scores = self.post_process(outputs)

                    status = "无检测目标"
                    if boxes is not None:
                        boxes_real = self.co_helper.get_real_box(boxes)
                        self.draw_detection_boxes(frame, boxes_real, classes, scores)
                        broken = any(self.CLASSES[int(cls)] == 'break' for cls in classes)
                        status = "有裂纹" if broken else "无裂纹"
                        self.last_detection_results[pos] = (frame.copy(), boxes_real, classes, scores)

                    self.ui.update_shell_status(pos, status)
                    self.ui.update_image(pos, frame)

                time.sleep(0.05)
            except Exception as e:
                self.logger.error(f"显示循环错误: {e}")
                self.ui.update_status(f"显示异常: {str(e)}")
                time.sleep(1)

    def reset_system(self):
        """复位系统"""
        self.running = True
        self.detection_enabled = False
        self.detection_running = False
        self.current_egg_id = 0
        self.record_count = 0
        self.last_shell_status = None
        self.current_freshness = "未检测"
        self.last_detection_results = {1: None, 2: None}
        self.deepseek_analysis = {}

        self.ui.update_status("系统已复位，等待开始检测")
        self.ui.ui['btn_detect'].config(state=tk.NORMAL)
        self.ui.ui['shell_result_label1'].config(text="1号位状态：未检测")
        self.ui.ui['shell_result_label2'].config(text="2号位状态：未检测")
        self.ui.ui['shell_info_label'].config(text="外壳检测结果: 等待检测")
        self.ui.ui['fresh_info_label'].config(text="新鲜度检测结果: 等待检测")

        blank_img = Image.new('RGB', (320, 320), (238, 238, 238))
        blank_photo = ImageTk.PhotoImage(blank_img)
        self.ui.ui['image_label_1'].config(image=blank_photo)
        self.ui.ui['image_label_1'].image = blank_photo
        self.ui.ui['image_label_2'].config(image=blank_photo)
        self.ui.ui['image_label_2'].image = blank_photo

        for item in self.ui.ui['record_table'].get_children():
            self.ui.ui['record_table'].delete(item)
        self.ui.ui['analysis_text'].configure(state=tk.NORMAL)
        self.ui.ui['analysis_text'].delete(1.0, tk.END)
        self.ui.ui['analysis_text'].configure(state=tk.DISABLED)

    def exit_system(self):
        """退出系统"""
        self.running = False
        for cam in self.cameras.values():
            cam.release()
        self.serial.close()
        self.rknn_model.release()
        self.root.quit()