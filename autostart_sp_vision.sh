#!/bin/bash
sleep 10
cd /home/wheeltec/projects/my_sp_main || exit
mkdir -p logs

chmod +x ./build/auto_aim_test_all

# watchdog（可选）
mkdir -p logs
WD="$(find /home/wheeltec/projects/my_sp_main -maxdepth 3 -type f -name "*.sh" -iname "*watchdog*" | head -n 1)"
if [ -n "$WD" ]; then
  chmod +x "$WD"
  screen -L -Logfile "logs/$(date '+%Y-%m-%d_%H-%M-%S').screenlog" -d -m bash -lc "\"$WD\""
else
  echo "[WARN] watchdog script not found, skip."
fi

gnome-terminal -- bash -lc "./build/auto_aim_test_all; echo '程序已结束，按任意键退出'; read -n 1"
screen -L -Logfile logs/$(date '+%Y-%m-%d_%H-%M-%S').screenlog -d -m bash -lc "./watchdog.sh"
