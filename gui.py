from Tkinter import *
from PIL import Image,ImageTk
import backEnd
import numpy as np
import cv2
import os
import tkFileDialog
import copy

class gui(object):

    #initializes the gui
    def __init__(self,title="Math OCR",width=1000,height=1000):
        self.title=title
        self.width=width
        self.height=height
        self.filePath=""
        self.frameWidth,self.frameHeight=self.width/3,self.height/3
        self.net=backEnd.load("double2")
        self.imageImported=False
        camWidth,camHeight=1680,1200
        self.cap=cv2.VideoCapture(0)
        self.isCam=self.cap.isOpened()
        self.savedDisplay=None
        self.result=None
        self.inputList,self.indexList=backEnd.loadGrid()
        if self.isCam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)

    #creating the bulk of the interface
    def run(self):
        self.root=Tk()
        self.root.wm_title(self.title)

        menuFrame=LabelFrame(self.root,relief="flat")
        menuFrame.grid(row=0,columnspan=3)

        leftFrame=LabelFrame(self.root,text="Image")
        leftFrame.grid(row=1,column=0,padx=5,pady=5)

        rigthFrame=LabelFrame(self.root,text="Result")
        rigthFrame.grid(row=1,column=2,padx=5,pady=5)

        topFrame=LabelFrame(menuFrame,text="Import File")
        topFrame.grid(row=0,column=1,padx=10)

        fileEntryLabel=Label(topFrame,text="File Path: ")
        fileEntryLabel.grid(row=0,column=0)

        topRightFrame=LabelFrame(menuFrame,text="Export File")
        topRightFrame.grid(row=0,column=2,padx=10)

        topRightMostFrame=LabelFrame(menuFrame,text="Help")
        topRightMostFrame.grid(row=0,column=0,padx=10)

        self.filePath=StringVar()
        self.fileEntry=Entry(topFrame,textvariable=self.filePath,width=100)
        self.fileEntry.grid(row=0,column=1,padx=2)

        importButton=Button(topFrame,text="Import",command=self.importFromFile)
        importButton.grid(row=0,column=2)

        self.imageLabel=Label(leftFrame,padx=self.width/4,pady=self.height/4,
            text="Import an Image!")
        self.imageLabel.grid()

        solveButton=Button(self.root,text="Solve!",command=self.solve)
        solveButton.grid(row=1,column=1)

        importBrowse=Button(topFrame,text="Browse",command=self.getFileName)
        importBrowse.grid(row=0,column=3)

        self.resultLabel=Label(rigthFrame,padx=self.width/4,pady=self.height/4,
            text="No Solutions Yet!")
        self.resultLabel.grid()

        self.cameraButton=Button(topFrame,text="Camera",command=self.camera)
        self.cameraButton.grid(row=0,column=4)

        exportButton=Button(topRightFrame,text="Export",
            command=self.exportImage)
        exportButton.grid(padx=10)

        helpButton=Button(topRightMostFrame,text="?",command=self.helpScreen)
        helpButton.grid(padx=10)

        self.root.mainloop()

    #creates a seprate pop up window containing image from the webcam
    def camera(self):
        if self.isCam:
            cameraWindow=Toplevel()
            cameraFrame=LabelFrame(cameraWindow)
            cameraFrame.grid()
            self.cameraLabel=Label(cameraFrame,padx=self.width/4,
                pady=self.height/4)
            self.cameraLabel.grid(row=1)
            self.captureButton=Button(cameraFrame,text="Capture!",
                command=self.capture)
            self.captureButton.grid(row=0)
            self.getImage()
        else: 
            top = Toplevel(padx=10,pady=10)
            top.title("Error")

            msg = Message(top, text="Error! No Camera Detected")
            msg.pack()

            button = Button(top, text="Ok", command=top.destroy)
            button.pack()

    #creates a help screen with all recognized characters
    def helpScreen(self):
        top = Toplevel(padx=10,pady=10)
        top.title("Accepted Inputs")

        inputFrame=LabelFrame(top)
        inputFrame.grid()
        index=0
        for index in xrange(len(self.inputList)):
            img=Image.fromarray(self.inputList[index])
            imgtk = ImageTk.PhotoImage(image=img)
            imgFrame=LabelFrame(inputFrame,text=self.indexList[index])
            imgFrame.grid(row=index/10,column=index%10,padx=2,pady=2)
            imgLabel=Label(imgFrame,image=imgtk)
            imgLabel.image=imgtk
            imgLabel.grid()

    #exports the current solution into user specified directory
    def exportImage(self):
        if self.rawResult!=None:
            fileName=tkFileDialog.asksaveasfilename(defaultextension=".png")
            if fileName!="":
                self.rawResult.save(fileName,"PNG")
        else:
            top = Toplevel(padx=10,pady=10)
            top.title("Error")

            msg = Message(top, text="Error! No Solution Yet")
            msg.pack()

            button = Button(top, text="Ok", command=top.destroy)
            button.pack()

    #gets the image and processes it from camera
    def getImage(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2Image=np.fliplr(cv2Image)
        self.rawImage=cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(cv2Image)
        self.cameraDisplay=copy.deepcopy(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cameraLabel.imgtk = imgtk
        self.cameraLabel.configure(image=imgtk)
        self.cameraLabel.after(10, self.getImage)

    #captures the image from camera for solving use
    def capture(self):
        self.savedRaw=self.rawImage
        self.savedDisplay=self.cameraDisplay
        cv2.imwrite("eq1.png",self.savedRaw)
        self.importImage()

    #imports an image form specified file directory
    def importFromFile(self):
        self.filePath.set(self.fileEntry.get())
        if os.path.exists(self.filePath.get()):
            self.savedRaw=cv2.imread(self.filePath.get(),0)
            self.savedDisplay=Image.open(self.filePath.get())
            self.importImage()
        else:
            top = Toplevel(padx=10,pady=10)
            top.title("Error")

            msg = Message(top, text="Error! Not a Valid Path!")
            msg.pack()

            button = Button(top, text="Ok", command=top.destroy)
            button.pack()

    #importsd the image proper and resizes it to fit display window
    def importImage(self):
        x1,y1,x2,y2=self.savedDisplay.getbbox()
        if (y2-y1)<=self.frameHeight and (x2-x1)<=self.frameWidth:
            ratio=1
        elif (y2-y1)>(x2-x1):
            ratio=(y2-y1)/self.frameHeight
        else: ratio=(x2-x1)/self.frameWidth
        self.savedDisplay=self.savedDisplay.resize((int((x2-x1)/ratio),
            int((y2-y1)/ratio)))
        self.savedDisplay=ImageTk.PhotoImage(self.savedDisplay)
        self.imageImported=True
        self.imageLabel.config(image=self.savedDisplay)

    #solves the imported image and displays the result
    def solve(self):
        if self.imageImported==True:
            result=backEnd.process(self.savedRaw,self.net)
            print result
            self.num=backEnd.displaySolution(result)
            self.rawResult=Image.open("temp%d.png"%self.num)
            self.result=ImageTk.PhotoImage(self.rawResult)
            self.resultLabel.config(image=self.result)

    #gets the file directory from user input
    def getFileName(self):
        self.filePath.set(tkFileDialog.askopenfilename())

gui().run()