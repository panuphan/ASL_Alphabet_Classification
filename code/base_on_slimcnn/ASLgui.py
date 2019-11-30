from tkinter import filedialog
from tkinter import * 
import tkinter as tk
from PIL import Image, ImageTk
import os
import webbrowser
from mymodel import *

selectedpic =""
IMAGE_PATH = 'code/base_on_slimcnn/gui/ASLbg.jpg'
WIDTH, HEIGTH = 524,393

slimcnn_model = 'model/slimcnn/2019-12-01_01.23.00-model'
AlexNet_model = 'model/myAlexNet/2019-11-30_19.37.12-model'
VGG16_model = 'model/myVGG16/2019-11-30_23.18.29-model'

model_1, train_log_1 = reload_model(slimcnn_model) 
model_2, train_log_2=  reload_model(AlexNet_model) 
model_3, train_log_3 = reload_model(VGG16_model) 

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
    global new_window
    new_window = Toplevel(root)
    new_window.title("Choosen Image")
    selectedpic=filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    print(selectedpic)
    im = Image.open(selectedpic)
    tkimage = ImageTk.PhotoImage(Image.open(selectedpic).resize((400,300),Image.ANTIALIAS))
    myvar=Label(new_window,image = tkimage)
    myvar.image = tkimage
    myvar.pack()

    # selectedpic = filedialog.askopenfile(initialdir=os.getcwd,title = "Select file",filetypes = (("jpeg","*.jpg"),("png","*.png"),("all files","*.*")))
    # if not selectedpic:
    #     return
    # new_window = Toplevel(root)
    # new_window.title("Choosen Image")

    # im = ImageTk.PhotoImage(Image.open(selectedpic).resize((400,300),Image.ANTIALIAS))
    # panel = Label(new_window,image = im)
    # panel.image = im
    # panel.pack()
    # print(selectedpic)

def openwebMethod1():
    global selectedpic
    global new_window
    # print(f"selectedpic: {selectedpic}")
    
    img = read_image(selectedpic)
    predictions = model_1.predict(img)
    pred=read_label(predictions)
    # print('[MODEL_01] predictions:', pred)    
    result1 = Text(new_window,width= 50, height=3)
    result1.pack()
    result1.insert(tk.END, "             slimcnn - Predict result :   "+pred+"  \n")

def openwebMethod2():
    global selectedpic
    global new_window
    # print(f"selectedpic: {selectedpic}")
    
    img = read_image(selectedpic)
    predictions = model_2.predict(img)
    pred=read_label(predictions)
    # print('[MODEL_02] predictions:', pred)    
    result1 = Text(new_window,width= 50, height=3)
    result1.pack()
    result1.insert(tk.END, "             AlexNet - Predict result :   "+pred+"  \n")

def openwebMethod3():
    global selectedpic
    global new_window
    # print(f"selectedpic: {selectedpic}")
    
    img = read_image(selectedpic)
    predictions = model_3.predict(img)
    pred=read_label(predictions)
    # print('[MODEL_03] predictions:', pred)    
    result1 = Text(new_window,width= 50, height=3)
    result1.pack()
    result1.insert(tk.END, "             VGG16 - Predict result :   "+pred+"  \n")

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

T = tk.Text(root, height=2, width=30)
T.pack()
T.insert(tk.END, "Just a text Widget\nin two lines\n")

root.mainloop()
