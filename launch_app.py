import os
import platform
import subprocess
import sys

ENV_NAME = "streamlit"
APP_FILE = "interface.py" 

def run_command(command, shell=True):
    """Utility to run a command and stream output live."""
    process = subprocess.Popen(command, shell=shell)
    process.communicate()

def launch_on_windows():
    print("Launching Streamlit app on Windows...")
    command = f'cmd.exe /k "conda activate {ENV_NAME} && streamlit run {APP_FILE}"'
    run_command(command)

def launch_on_unix():
    print("Launching Streamlit app on Unix-like OS...")
    conda_init = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    if not os.path.exists(conda_init):
        conda_init = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")
    if not os.path.exists(conda_init):
        print("❌ Could not find conda.sh. Make sure conda is installed and initialized.")
        sys.exit(1)
    
    command = f'bash -c "source {conda_init} && conda activate {ENV_NAME} && streamlit run {APP_FILE}"'
    run_command(command)

if __name__ == "__main__":
    os_type = platform.system()
    if os_type == "Windows":
        launch_on_windows()
    elif os_type in ["Linux", "Darwin"]:  # Darwin = macOS
        launch_on_unix()
    else:
        print(f"Unsupported OS: {os_type}")

