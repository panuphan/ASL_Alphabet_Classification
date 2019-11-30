from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
a=""
root = Tk()
root.geometry("500x500")
def selection():
    global a
    a = filedialog.askopenfile(initialdir = "/",title = "Select file",filetypes = (("jpeg","*.jpg"),("all files","*.*")))
    im = Image.open(a)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(root,image = tkimage)
    myvar.image = tkimage
    myvar.pack()
    print(a)
Button(text = ' Browse ' ,bd = 3 ,font = ('',10),padx=5,pady=5, command=selection).grid(row=1,column=1)

root.mainloop()