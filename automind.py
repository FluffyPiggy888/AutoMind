import pygame
import pyttsx3
import numpy as np
import time
import threading
import random
from collections import deque
import pyaudio
import sys
import os

# ******************** 常量配置 ********************
# 界面参数
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
BG_COLOR = (20, 30, 50)
PANEL_COLOR = (40, 60, 90)
TEXT_COLOR = (220, 220, 220)
WARNING_COLOR = (255, 165, 0)
CRITICAL_COLOR = (255, 50, 50)
NORMAL_COLOR = (100, 200, 100)

# 信号参数
AUDIO_SAMPLE_RATE = 16000  # 音频采样率 (Hz)
AUDIO_CHUNK = 1024         # 音频块大小

# 疲劳阈值
YAWN_COUNT_THRESHOLD = 3   # 10分钟内哈欠次数阈值
YAWN_ENERGY_THRESHOLD = 0.02  # 哈欠能量阈值

# ******************** 疲劳分析器 ********************
class FatigueAnalyzer:
    def __init__(self):
        # 初始化状态变量
        self.fatigue_level = "NORMAL"
        self.yawn_count = 0
        self.last_update = time.time()
        self.steering_value = 0.0
        self.driver_state = "专注驾驶"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.voice_engine = pyttsx3.init()
        self.last_alert_time = time.time()
        # 添加缺失的属性
        self.torque_history = deque(maxlen=100)  # 存储方向盘扭矩历史数据
        self.simulation_speed = 1.0  # 模拟速度初始值
        
    def add_audio_data(self, audio_data):
        """添加音频数据到缓冲区"""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
    
    def _voice_alert(self, message):
        """语音提醒功能"""
        current_time = time.time()
        # 控制提醒频率，避免过于频繁（至少间隔30秒）
        if current_time - self.last_alert_time >= 30:
            self.voice_engine.say(message)
            self.voice_engine.runAndWait()
            self.last_alert_time = current_time
            
    def update(self):
        """更新疲劳状态分析"""
        current_time = time.time()
        
        # 每3秒分析一次
        if current_time - self.last_update < 3:  
            return
        
        self.last_update = current_time
        
        # 1. 哈欠检测 (使用真实麦克风数据)
        yawn_detected = 0
        if len(self.audio_buffer) >= AUDIO_SAMPLE_RATE * 3:  # 至少3秒音频
            # 计算平均能量
            energy = np.mean(self.audio_buffer**2)
            
            # 检测哈欠 (能量超过阈值)
            if energy > YAWN_ENERGY_THRESHOLD:
                yawn_detected = 1
                self.yawn_count += 1
                
            # 清空已分析的音频缓冲区
            self.audio_buffer = np.array([], dtype=np.float32)
        
        # 2. 模拟方向盘行为 (基于当前疲劳状态)
        if self.fatigue_level == "NORMAL":
            # 正常驾驶：有规律的微操作
            self.steering_value = np.sin(time.time() * 1.5 * self.simulation_speed) * 1.2 + random.uniform(-0.3, 0.3)
            self.driver_state = random.choices(["专注驾驶", "轻微走神"], weights=[8, 2])[0]
        elif self.fatigue_level == "WARNING":
            # 疲劳驾驶：转向幅度减小且不规律
            if random.random() < 0.7:
                self.steering_value = np.sin(time.time() * 0.8 * self.simulation_speed) * 0.8 + random.uniform(-0.2, 0.2)
            else:
                # 偶尔出现大角度转向（修正方向）
                self.steering_value = random.uniform(-2.0, 2.0)
            self.driver_state = random.choices(["走神", "短暂瞌睡"], weights=[7, 3])[0]
        else:
            # 严重疲劳：长时间无操作 + 偶尔急转
            if random.random() < 0.9:
                self.steering_value = 0.0
            else:
                self.steering_value = random.uniform(-3.0, 3.0)
            self.driver_state = random.choices(["瞌睡", "严重瞌睡"], weights=[6, 4])[0]
        
        # 记录方向盘数据
        self.torque_history.append(self.steering_value)
        
        # 3. 疲劳状态决策
        if self.yawn_count >= YAWN_COUNT_THRESHOLD + 1:
            self.fatigue_level = "CRITICAL"
        elif self.yawn_count >= YAWN_COUNT_THRESHOLD:
            self.fatigue_level = "WARNING"
        else:
            self.fatigue_level = "NORMAL"
        
        # 每10分钟重置哈欠计数
        if current_time % 600 < 3:
            self.yawn_count = 0

# ******************** 音频接口 ********************
class AudioInterface:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.running = True
        
        # 初始化音频输入
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
        
        # 启动音频采集线程
        self.thread = threading.Thread(target=self._capture_audio)
        self.thread.daemon = True
        self.thread.start()
    
    def _capture_audio(self):
        """从麦克风捕获音频数据"""
        while self.running:
            try:
                # 读取音频数据
                audio_data = np.frombuffer(
                    self.stream.read(AUDIO_CHUNK, exception_on_overflow=False),
                    dtype=np.float32
                )
                self.analyzer.add_audio_data(audio_data)
            except Exception as e:
                print(f"音频采集错误: {e}")
            
            time.sleep(0.01)
    
    def stop(self):
        """停止并释放资源"""
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

# ******************** 可视化界面 ********************
class DemoUI:
    def __init__(self, analyzer):
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI疲劳驾驶监测系统 - 纯软件演示版")
        self.clock = pygame.time.Clock()
        self.analyzer = analyzer
        
        # 字体设置
        self.font_large = pygame.font.SysFont("microsoftyahei", 60)  # 使用微软雅黑
        self.font_medium = pygame.font.SysFont("microsoftyahei", 36)
        self.font_small = pygame.font.SysFont("microsoftyahei", 24)
        
        # 状态颜色映射
        self.status_colors = {
            "NORMAL": NORMAL_COLOR,
            "WARNING": WARNING_COLOR,
            "CRITICAL": CRITICAL_COLOR
        }
        
        # 创建图标
        self.car_icon = self._create_car_icon()
        self.driver_icons = self._create_driver_icons()
        
        # 公司Logo
        self.logo = self._create_logo()
    
    def _create_car_icon(self):
        """创建汽车图标"""
        icon = pygame.Surface((120, 60), pygame.SRCALPHA)
        # 车身
        pygame.draw.rect(icon, (70, 130, 200), (10, 15, 100, 30), border_radius=8)
        # 车窗
        pygame.draw.rect(icon, (180, 220, 255), (20, 20, 80, 20), border_radius=5)
        # 车轮
        pygame.draw.circle(icon, (40, 40, 50), (30, 50), 8)
        pygame.draw.circle(icon, (40, 40, 50), (90, 50), 8)
        return icon
    
    def _create_driver_icons(self):
        """创建不同状态的驾驶员图标"""
        icons = {}
        
        # 正常状态
        icon_normal = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.circle(icon_normal, NORMAL_COLOR, (50, 50), 40, 3)
        pygame.draw.circle(icon_normal, (200, 180, 150), (50, 50), 30)  # 脸部
        pygame.draw.circle(icon_normal, (50, 50, 80), (35, 45), 5)      # 左眼
        pygame.draw.circle(icon_normal, (50, 50, 80), (65, 45), 5)      # 右眼
        pygame.draw.arc(icon_normal, (80, 50, 50), (40, 55, 20, 10), 0, np.pi, 2)  # 微笑
        icons["normal"] = icon_normal
        
        # 警告状态
        icon_warning = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.circle(icon_warning, WARNING_COLOR, (50, 50), 40, 3)
        pygame.draw.circle(icon_warning, (200, 180, 150), (50, 50), 30)
        pygame.draw.circle(icon_warning, (50, 50, 80), (35, 45), 5)
        pygame.draw.circle(icon_warning, (50, 50, 80), (65, 45), 5)
        pygame.draw.line(icon_warning, (80, 50, 50), (40, 60), (60, 60), 2)  # 平嘴
        icons["warning"] = icon_warning
        
        # 危险状态
        icon_critical = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.circle(icon_critical, CRITICAL_COLOR, (50, 50), 40, 3)
        pygame.draw.circle(icon_critical, (200, 180, 150), (50, 50), 30)
        pygame.draw.circle(icon_critical, (50, 50, 80), (35, 40), 5)  # 半闭眼
        pygame.draw.circle(icon_critical, (50, 50, 80), (65, 40), 5)
        pygame.draw.arc(icon_critical, (80, 50, 50), (35, 60, 30, 20), 0, np.pi, 2)  # 打哈欠
        icons["critical"] = icon_critical
        
        return icons
    
    def _create_logo(self):
        """创建公司Logo"""
        logo = pygame.Surface((200, 60), pygame.SRCALPHA)
        # 绘制公司名称
        text = self.font_small.render("Automind", True, (70, 170, 255))
        logo.blit(text, (40, 15))
        # 绘制装饰线
        pygame.draw.line(logo, (0, 200, 200), (10, 55), (190, 55), 2)
        return logo
    
    def draw_main_panel(self):
        """绘制主面板"""
        # 主面板背景
        panel_rect = pygame.Rect(50, 50, SCREEN_WIDTH-100, SCREEN_HEIGHT-100)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect, border_radius=15)
        pygame.draw.rect(self.screen, (80, 110, 150), panel_rect, 3, border_radius=15)
        
        # 显示标题
        title = self.font_large.render("AI疲劳驾驶监测系统", True, TEXT_COLOR)
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 70))
        
        # 显示公司Logo
        self.screen.blit(self.logo, (SCREEN_WIDTH - 250, 70))
        
        # 显示当前状态
        status = self.analyzer.fatigue_level
        color = self.status_colors[status]
        status_text = self.font_large.render(f"状态: {status}", True, color)
        self.screen.blit(status_text, (SCREEN_WIDTH//2 - status_text.get_width()//2, 150))
        
        # 显示驾驶员状态
        state_text = self.font_medium.render(f"驾驶员状态: {self.analyzer.driver_state}", True, color)
        self.screen.blit(state_text, (SCREEN_WIDTH//2 - state_text.get_width()//2, 220))
        
        # 显示状态说明
        if status == "NORMAL":
            desc = "驾驶员状态正常，请继续保持安全驾驶"
        elif status == "WARNING":
            desc = "警告：检测到疲劳迹象，建议休息"
        else:
            desc = "危险：严重疲劳，请立即在安全区域停车休息！"
        
        desc_text = self.font_medium.render(desc, True, color)
        self.screen.blit(desc_text, (SCREEN_WIDTH//2 - desc_text.get_width()//2, 270))
        
        # 传感器数据显示区
        pygame.draw.rect(self.screen, (30, 45, 70), (80, 320, SCREEN_WIDTH-160, 180), border_radius=10)
        
        # 方向盘数据
        torque_text = self.font_medium.render(f"方向盘行为: {self.analyzer.steering_value:.2f}", True, TEXT_COLOR)
        self.screen.blit(torque_text, (100, 340))
        
        # 哈欠检测
        yawn_text = self.font_medium.render(f"检测到哈欠: {self.analyzer.yawn_count}次", True, TEXT_COLOR)
        self.screen.blit(yawn_text, (100, 390))
        
        # 麦克风状态
        mic_status = "就绪 - 请尝试打哈欠进行测试"
        mic_text = self.font_medium.render(f"麦克风状态: {mic_status}", True, TEXT_COLOR)
        self.screen.blit(mic_text, (100, 440))
        
        # 绘制方向盘扭矩历史图
        if self.analyzer.torque_history:
            # 绘制坐标轴
            pygame.draw.line(self.screen, (100, 150, 200), 
                            (100, 490), 
                            (SCREEN_WIDTH-100, 490), 2)
            
            # 绘制数据线
            max_torque = max(1, max(abs(t) for t in self.analyzer.torque_history))
            points = []
            for i, t in enumerate(self.analyzer.torque_history):
                x = 100 + i * (SCREEN_WIDTH-200) / len(self.analyzer.torque_history)
                y = 490 - t/max_torque * 30
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 200), False, points, 3)
        
        # 绘制汽车和驾驶员图标
        self.screen.blit(self.car_icon, (SCREEN_WIDTH//2 - 60, 520))
        driver_icon = self.driver_icons["normal"]
        if status == "WARNING":
            driver_icon = self.driver_icons["warning"]
        elif status == "CRITICAL":
            driver_icon = self.driver_icons["critical"]
        self.screen.blit(driver_icon, (SCREEN_WIDTH//2 - 50, 460))
    
    def draw_control_panel(self):
        """绘制控制面板"""
        panel_rect = pygame.Rect(50, SCREEN_HEIGHT - 80, SCREEN_WIDTH-100, 50)
        pygame.draw.rect(self.screen, (30, 40, 60), panel_rect, border_radius=10)
        
        # 控制说明
        controls = [
            "空格键: 重置演示",
            "H键: 模拟打哈欠",
            "+/-: 调整模拟速度",
            "ESC键: 退出"
        ]
        
        for i, text in enumerate(controls):
            ctrl_text = self.font_small.render(text, True, (180, 200, 255))
            self.screen.blit(ctrl_text, (100 + i*240, SCREEN_HEIGHT - 65))
    
    def draw_info_panel(self):
        """绘制信息面板"""
        # 技术信息
        tech_info = [
            "技术特点:",
            "- 纯软件方案，无需硬件",
            "- 实时麦克风哈欠检测",
            "- 多级疲劳状态分析",
            f"模拟速度: {self.analyzer.simulation_speed:.1f}x"
        ]
        
        for i, text in enumerate(tech_info):
            info_text = self.font_small.render(text, True, (180, 220, 255))
            self.screen.blit(info_text, (SCREEN_WIDTH - 280, 320 + i*30))
    
    def run(self):
        """运行主循环"""
        running = True
        last_analysis_time = time.time()
        last_yawn_sim_time = 0
        
        while running:
            current_time = time.time()
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # 重置演示
                        self.analyzer.yawn_count = 0
                        self.analyzer.fatigue_level = "NORMAL"
                        self.analyzer.driver_state = "专注驾驶"
                    elif event.key == pygame.K_h and current_time - last_yawn_sim_time > 2:
                        # 模拟打哈欠
                        self.analyzer.yawn_count += 1
                        last_yawn_sim_time = current_time
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # 增加模拟速度
                        self.analyzer.simulation_speed = min(3.0, self.analyzer.simulation_speed + 0.2)
                    elif event.key == pygame.K_MINUS:
                        # 降低模拟速度
                        self.analyzer.simulation_speed = max(0.5, self.analyzer.simulation_speed - 0.2)
            
            # 更新分析器
            self.analyzer.update()
            
            # 绘制界面
            self.screen.fill(BG_COLOR)
            self.draw_main_panel()
            self.draw_control_panel()
            self.draw_info_panel()
            
            # 显示公司信息
            footer = self.font_small.render("奥途智能科技 - 纯软件AI解决方案 | 帝国理工孵化项目", True, (150, 150, 180))
            self.screen.blit(footer, (SCREEN_WIDTH//2 - footer.get_width()//2, SCREEN_HEIGHT - 30))
            
            # 实时状态提示
            if current_time - last_yawn_sim_time < 1:
                yawn_text = self.font_medium.render("模拟哈欠已触发!", True, WARNING_COLOR)
                self.screen.blit(yawn_text, (SCREEN_WIDTH//2 - yawn_text.get_width()//2, 350))
            
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        return 0



# ******************** 主程序入口 ********************
def main():
    print("启动纯软件疲劳驾驶监测演示...")
    print("提示：请确保已连接麦克风并允许程序访问")
    
    # 初始化分析器
    analyzer = FatigueAnalyzer()
    
    # 初始化音频接口
    try:
        audio_interface = AudioInterface(analyzer)
    except Exception as e:
        print(f"音频初始化失败: {e}")
        print("将继续运行模拟模式，但麦克风功能不可用")
        audio_interface = None
    
    # 启动UI
    ui = DemoUI(analyzer)
    exit_code = ui.run()
    # 清理资源
    if audio_interface:
        audio_interface.stop()
    print("演示已安全退出")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())