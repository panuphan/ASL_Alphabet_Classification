from tkinter import filedialog
from tkinter import * 
import tkinter as tk
from PIL import Image, ImageTk
import os
import webbrowser

selectedpic =""
IMAGE_PATH = 'ASLbg.jpg'
WIDTH, HEIGTH = 524,393

root = tk.Tk()
root.title('ASL Recognition')
root.geometry('{}x{}'.format(WIDTH, HEIGTH))

canvas = tk.Canvas(root, width=WIDTH, height=HEIGTH)
canvas.pack()

img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas.background = img  # Keep a reference in case this code is put in a function.
bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)

def browsefile():
    global selectedpic
    new_window = Toplevel(root)
    new_window.title("Choosen Image")
    selectedpic=filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    print(selectedpic)
    im = Image.open(selectedpic)
    tkimage = ImageTk.PhotoImage(Image.open(selectedpic).resize((400,300),Image.ANTIALIAS))
    myvar=Label(new_window,image = tkimage)
    myvar.image = tkimage
    myvar.pack()

def openwebMethod1():
    method1url = "https://www.google.com"
    one=1
    webbrowser.open(method1url,new=one) 

def openwebMethod2():
    method2url = "https://www.youtube.com/"
    two=2
    webbrowser.open(method2url,new=two) 

def openwebMethod3():
    method3url = "https://www.facebook.com/"
    three=3
    webbrowser.open(method3url,new=three) 

# def method1(selectedpic):
#     filemethod1 = filedialog.askopenfile(initialdir=os.getcwd,title = "Select file",filetypes = (("all files","*.*")))
#     if not filemethod1:
#         return
#     method1_window = Toplevel(root)
#     method1_window.title("Method 1")
#     method1 = ImageTk.PhotoImage(Image.open(filemethod1))
#     panel2 = Label(method1_window,image = im)
#     panel2.image = im
#     panel2.place(x=0, y=0)
#     panel2.pack()
#     print(filemethod1)


#Button Region 
Browsebtn = tk.Button(root, text ='  Browse  ', command = lambda:browsefile())
Browsebtn_window = canvas.create_window(287, 141, anchor=tk.CENTER, window=Browsebtn)
Method1btn = tk.Button(root, text ='Method 1', command = lambda:openwebMethod1())
Method1btn_window = canvas.create_window(95, 315, anchor=tk.NW, window=Method1btn)
Method2btn = tk.Button(root, text ='Method 2', command = lambda:openwebMethod2())
Method2btn_window = canvas.create_window(245, 315, anchor=tk.NW, window=Method2btn)
Method3btn = tk.Button(root, text ='Method 3', command = lambda:openwebMethod3())
Method3btn_window = canvas.create_window(395, 315, anchor=tk.NW, window=Method3btn)

root.mainloop()
