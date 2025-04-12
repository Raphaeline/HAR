# import uvicorn
# from api.api import app

# if __name__ == "__main__":
#     uvicorn.run("api.api:app", host="127.0.0.1", port=8000, reload=True)


# from gui.gui_main import run_gui
import subprocess

if __name__ == '__main__':
    # Pastikan path ke gui_main.py sudah benar relatif terhadap direktori eksekusi main.py
    subprocess.run(["python", "gui/gui_main.py"])
