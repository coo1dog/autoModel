@echo off
cd /d "%~dp0"
echo [INFO] 正在启动 AutoML Web UI...

:: 尝试激活 Conda 环境 (根据以前的报错推测路径)
if exist "D:\anaconda3\Scripts\activate.bat" (
    echo [INFO] Found Anaconda, activating environment 'paddle_cpu'...
    call D:\anaconda3\Scripts\activate.bat paddle_cpu
) else (
    echo [INFO] Anaconda activate script not found at default location. Using system environment.
)

echo [INFO] Running Streamlit...
echo [INFO] Please wait for the browser to open...
streamlit run src/web_ui.py

if errorlevel 1 (
    echo.
    echo [ERROR] 启动失败！
    echo [原因可能] 系统找不到 'streamlit' 命令。
    echo [解决方法] 请回到 VS Code，在下方终端手动输入: streamlit run src/web_ui.py
)
pause