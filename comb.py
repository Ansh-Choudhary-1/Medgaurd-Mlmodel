import subprocess
import os

def run_sepsis_server():
    # Adjust the path to point to your file location
    subprocess.Popen(['python', os.path.join(os.getcwd(), '_sep.py')])

def run_diagnosis_server():
    # Adjust the path to point to your file location
    subprocess.Popen(['python', os.path.join(os.getcwd(), 'xrayy.py')])

if __name__ == '__main__':
    try:
        # Start both servers
        run_sepsis_server()
        run_diagnosis_server()

        print("Both servers are running...")
    except Exception as e:
        print(f"Error occurred: {e}")
