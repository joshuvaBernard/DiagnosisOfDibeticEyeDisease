from loader import ImageViewer
from tkinter import Tk

if __name__ == "__main__":
    root = Tk()
    while True:
        app = ImageViewer(root)
        root.mainloop()

