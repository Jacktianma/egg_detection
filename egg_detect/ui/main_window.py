# ui/main_window.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from utils.logger import get_logger
import cv2
class MainWindow:
    def __init__(self, root):
        self.logger = get_logger("MainWindow")
        self.root = root
        self.root.title("壳・心双鉴：飞凌 ELF2 赋能边缘 AI 与高光谱的蛋品双维检测")
        self.root.geometry("1920x1080")
        self.root.resizable(False, False)
        self.ui = self.create_ui()

    def create_ui(self):
        """创建优化后的UI布局并返回UI组件的引用"""
        # 主框架
        main_frame = tk.Frame(self.root, bg='#f5f5f5')
        main_frame.place(x=0, y=0, width=1920, height=1080)

        # 标题框架
        title_frame = tk.Frame(main_frame, bg='#3a7ca5', bd=0)
        title_frame.place(x=20, y=10, width=1860, height=80)
        title_label = tk.Label(
            title_frame,
            text="壳・心双鉴：飞凌 ELF2 赋能边缘 AI 与高光谱的蛋品双维检测",
            font=("微软雅黑", 30, "bold"),
            fg="white",
            bg="#3a7ca5"
        )
        title_label.pack(pady=8)

        # 左侧区域 - 1号位外壳检测
        left_frame1 = tk.Frame(main_frame, bg='#3a7ca5', bd=0, highlightthickness=1, highlightbackground="#ddd")
        left_frame1.place(x=20, y=100, width=500, height=500)
        title_label_1 = tk.Label(
            left_frame1,
            text="鸡蛋裂纹实时检测(1号位)",
            font=("微软雅黑", 14, "bold"),
            fg="white",
            bg="#3a7ca5"
        )
        title_label_1.pack(pady=8)
        image_frame1 = tk.Frame(left_frame1, bg='#eee', bd=0)
        image_frame1.pack(pady=3, padx=10, fill=tk.BOTH, expand=True)
        image_label_1 = tk.Label(
            image_frame1,
            bg="#eee",
            width=320,
            height=320
        )
        image_label_1.pack(pady=5)
        shell_result_frame1 = tk.Frame(left_frame1, bg='white')
        shell_result_frame1.pack(pady=5, padx=10, fill=tk.X)
        shell_result_label1 = tk.Label(
            shell_result_frame1,
            text="1号位状态：未检测",
            font=("微软雅黑", 16, "bold"),
            fg="#3a7ca5",
            bg="white"
        )
        shell_result_label1.pack(pady=5, padx=10)

        # 左侧区域 - 2号位外壳检测
        left_frame2 = tk.Frame(main_frame, bg='#3a7ca5', bd=0, highlightthickness=1, highlightbackground="#ddd")
        left_frame2.place(x=550, y=100, width=500, height=500)
        title_label_2 = tk.Label(
            left_frame2,
            text="鸡蛋裂纹实时检测(2号位)",
            font=("微软雅黑", 14, "bold"),
            fg="white",
            bg="#3a7ca5"
        )
        title_label_2.pack(pady=8)
        image_frame2 = tk.Frame(left_frame2, bg='#eee', bd=0)
        image_frame2.pack(pady=3, padx=10, fill=tk.BOTH, expand=True)
        image_label_2 = tk.Label(
            image_frame2,
            bg="#eee",
            width=320,
            height=320
        )
        image_label_2.pack(pady=5)
        shell_result_frame2 = tk.Frame(left_frame2, bg='white')
        shell_result_frame2.pack(pady=5, padx=10, fill=tk.X)
        shell_result_label2 = tk.Label(
            shell_result_frame2,
            text="2号位状态：未检测",
            font=("微软雅黑", 16, "bold"),
            fg="#3a7ca5",
            bg="white"
        )
        shell_result_label2.pack(pady=5, padx=10)

        # 右侧区域 - 记录表格
        right_frame = tk.Frame(main_frame, bg='#3a7ca5', bd=0, highlightthickness=1, highlightbackground="#ddd")
        right_frame.place(x=1080, y=100, width=800, height=500)
        record_title_frame = tk.Frame(right_frame, bg='#3a7ca5')
        record_title_frame.pack(pady=8, padx=10, fill=tk.X)
        record_title = tk.Label(
            record_title_frame,
            text="鸡蛋检测记录",
            font=("微软雅黑", 14, "bold"),
            fg="white",
            bg="#3a7ca5"
        )
        record_title.pack(pady=0)
        record_frame = tk.Frame(right_frame, bg="#3a7ca5")
        record_frame.pack(pady=3, padx=10, fill=tk.BOTH, expand=True)
        columns = ("序号", "外壳状态", "新鲜度", "检测时间")
        record_table = ttk.Treeview(record_frame, columns=columns, show="headings", height=10)
        record_table.heading("序号", text="序号", anchor="center")
        record_table.heading("外壳状态", text="外壳状态", anchor="center")
        record_table.heading("新鲜度", text="新鲜度", anchor="center")
        record_table.heading("检测时间", text="检测时间", anchor="center")
        record_table.column("序号", width=50, anchor="center")
        record_table.column("外壳状态", width=100, anchor="center")
        record_table.column("新鲜度", width=100, anchor="center")
        record_table.column("检测时间", width=200, anchor="center")
        y_scrollbar = ttk.Scrollbar(record_frame, orient="vertical", command=record_table.yview)
        record_table.configure(yscrollcommand=y_scrollbar.set)
        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar = ttk.Scrollbar(record_frame, orient="horizontal", command=record_table.xview)
        record_table.configure(xscrollcommand=x_scrollbar.set)
        x_scrollbar.pack(side="bottom", fill="x")
        record_table.pack(side="left", fill="both", expand=True)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="white", foreground="#333", rowheight=22, fieldbackground="white")
        style.map('Treeview', background=[('selected', 'white')], foreground=[('selected', 'white')])
        style.configure("Treeview.Heading", font=('微软雅黑', 10, 'bold'), background="#f0f0f0", foreground="#333")

        # 综合结果框架
        result_frame = tk.Frame(main_frame, bg='white', bd=0, highlightthickness=1, highlightbackground="#ddd")
        result_frame.place(x=20, y=600, width=1860, height=80)
        shell_info_frame = tk.Frame(result_frame, bg='white')
        shell_info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fresh_info_frame = tk.Frame(result_frame, bg='white')
        fresh_info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        shell_info_label = tk.Label(
            shell_info_frame,
            text="外壳检测结果: 等待检测",
            font=("微软雅黑", 16, "bold"),
            fg="#3a7ca5",
            bg="white"
        )
        shell_info_label.pack(pady=20)
        fresh_info_label = tk.Label(
            fresh_info_frame,
            text="新鲜度检测结果: 等待检测",
            font=("微软雅黑", 16, "bold"),
            fg="#3a7ca5",
            bg="white"
        )
        fresh_info_label.pack(pady=20)

        # DeepSeek分析框架
        analysis_frame = tk.Frame(main_frame, bg='white', bd=0, highlightthickness=1, highlightbackground="#ddd")
        analysis_frame.place(x=20, y=690, width=1860, height=190)
        analysis_title = tk.Label(
            analysis_frame,
            text="DeepSeek AI 分析结果",
            font=("微软雅黑", 14, "bold"),
            fg="white",
            bg="#3a7ca5"
        )
        analysis_title.pack(pady=8, fill=tk.X)
        analysis_text = tk.Text(
            analysis_frame,
            height=8,
            font=("微软雅黑", 12),
            bg="white",
            fg="#333",
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        analysis_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=analysis_text.yview)
        analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        analysis_scrollbar.pack(side="right", fill="y")

        # 系统状态框架
        status_frame = tk.Frame(main_frame, bg='white', bd=0, highlightthickness=1, highlightbackground="#ddd")
        status_frame.place(x=20, y=880, width=1860, height=50)
        status_label = tk.Label(
            status_frame,
            text="系统状态：准备就绪",
            font=("微软雅黑", 14, "bold"),
            fg="#3a7ca5",
            bg="white"
        )
        status_label.pack(pady=10)

        # 按钮框架
        button_frame = tk.Frame(main_frame, bg='white', bd=0)
        button_frame.place(x=20, y=930, width=1860, height=80)
        btn_detect = tk.Button(
            button_frame,
            text="开始检测",
            width=22,
            bg="#3a7ca5",
            fg="white",
            relief="flat",
            cursor="hand2",
            font=("微软雅黑", 16)
        )
        btn_detect.pack(side=tk.LEFT, padx=30, pady=15, expand=True)
        btn_reset = tk.Button(
            button_frame,
            text="复位系统",
            width=22,
            bg="#f0ad4e",
            fg="white",
            relief="flat",
            cursor="hand2",
            font=("微软雅黑", 16)
        )
        btn_reset.pack(side=tk.LEFT, padx=30, pady=15, expand=True)
        btn_exit = tk.Button(
            button_frame,
            text="退出系统",
            width=22,
            bg="#d9534f",
            fg="white",
            relief="flat",
            cursor="hand2",
            font=("微软雅黑", 16)
        )
        btn_exit.pack(side=tk.LEFT, padx=30, pady=15, expand=True)

        # 按钮悬停效果
        def on_enter(e, btn, color):
            btn['background'] = color
        def on_leave(e, btn, color):
            btn['background'] = color
        btn_detect.bind("<Enter>", lambda e: on_enter(e, btn_detect, "#2a6d95"))
        btn_detect.bind("<Leave>", lambda e: on_leave(e, btn_detect, "#3a7ca5"))
        btn_reset.bind("<Enter>", lambda e: on_enter(e, btn_reset, "#ec971f"))
        btn_reset.bind("<Leave>", lambda e: on_leave(e, btn_reset, "#f0ad4e"))
        btn_exit.bind("<Enter>", lambda e: on_enter(e, btn_exit, "#c9302c"))
        btn_exit.bind("<Leave>", lambda e: on_leave(e, btn_exit, "#d9534f"))

        # 返回UI组件字典
        ui_components = {
            'image_label_1': image_label_1,
            'image_label_2': image_label_2,
            'shell_result_label1': shell_result_label1,
            'shell_result_label2': shell_result_label2,
            'btn_detect': btn_detect,
            'btn_reset': btn_reset,
            'btn_exit': btn_exit,
            'status_label': status_label,
            'record_table': record_table,
            'shell_info_label': shell_info_label,
            'fresh_info_label': fresh_info_label,
            'analysis_text': analysis_text
        }
        self.logger.info("UI layout created successfully")
        return ui_components

    def update_image(self, position, frame):
        """更新UI显示图像"""
        try:
            display_frame = cv2.resize(frame, (320, 320))
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(image=img)
            label = self.ui[f'image_label_{position}']
            label.config(image=photo)
            label.image = photo
        except Exception as e:
            self.logger.error(f"更新图像错误: {e}")

    def update_status(self, message):
        """更新状态栏信息"""
        self.ui['status_label'].config(text=f"系统状态：{message}")
        self.root.update()

    def update_shell_status(self, position, status):
        """更新外壳状态标签"""
        self.ui[f'shell_result_label{position}'].config(text=f"{position}号位状态：{status}")

    def add_record(self, record_id, shell_status, freshness, timestamp):
        """添加鸡蛋检测记录"""
        self.ui['record_table'].insert("", "end", values=(record_id, shell_status, freshness, timestamp))
        self.ui['record_table'].see(self.ui['record_table'].get_children()[-1])
        self.logger.info(f"添加记录: 序号={record_id}, 外壳={shell_status}, 新鲜度={freshness}, 时间={timestamp}")