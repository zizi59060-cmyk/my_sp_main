#!/bin/bash
# 延迟以确保桌面环境和外设加载完成
sleep 10

# 切换工作目录
cd /home/wheeltec/projects/sp_vision_25-main || exit

# 确保主程序有执行权限
chmod +x ./build/auto_aim_test_all
chmod +x ./watchdog.sh

# 1️⃣ 在新终端中启动主程序并保持窗口
gnome-terminal -- bash -c "./build/auto_aim_test_all; echo '程序已结束，按任意键关闭...'; read -n1"

# 2️⃣ 在后台运行 watchdog（记录日志）
screen -L -Logfile logs/$(date '+%Y-%m-%d_%H-%M-%S').screenlog -d -m bash -c "./watchdog.sh"

