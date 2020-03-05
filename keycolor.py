import matplotlib.pyplot as plt
import tkinter.filedialog
import numpy as np
from tkinter import *
from tkinter import ttk
import cv2
import argparse
from tkinter.filedialog import askopenfilename
from sklearn.cluster import KMeans
from PIL import Image,ImageTk
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
help = "# of clusters")
args = vars(ap.parse_args())



def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist # return the histogram

def plot_colors(hist, centroids):
	num = 1	# initialize the bar chart representing the relative frequency of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	positionx=10
	positiony=30
	for (percent, color) in zip(hist, centroids):	# loop over the percentage of each cluster and the color of# each cluster
		endX = startX + (percent * 300)	# plot the relative percentage of each cluster
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
		cv2.putText(bar, "{}={:.2f}%".format(num,percent*100), (positionx, positiony), cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 255), 0, cv2.LINE_AA)
		positionx = positionx+70
		num=num+1
	return bar #แสดงแทบสี

def aboutme():
	root = Tk()
	root.title("โปรแกรมจำแนกสีจากภาพ")
	root.option_add("*Font","consolas 10")

	l0=Label(root, text="    ")
	l0.grid(row=3,column=0)
	l1=Label(root, text="จัดทำโดย")
	l1.grid(row=1,column=2)
	l2=Label(root, text="นายวิธาน   วงษาบุตร (603410061-2)")
	l2.grid(row=2,column=2)
	l3=Label(root, text="สาขาวิทยาการคอมพิวเตอร์และสารสนเทศ")
	l3.grid(row=3,column=2)
	l5=Label(root, text="อ้างอิง : https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering")
	l5.grid(row=4,column=2)
	l6=Label(root, text="https://www.geeksforgeeks.org/python-gui-tkinter")
	l6.grid(row=5,column=2)
	l4=Label(root, text="     ")
	l4.grid(row=6,column=3)

	Button(root, text="ปิดหน้าต่าง", command=root.destroy).grid(row=7,column=2)
	rw.mainloop()



def open_File():
    filename = askopenfilename(filetypes=[("images","*.jpg")])
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()#แสดงภาพต้นฉบับ
    plt.axis("off")
    plt.title('Original')
    plt.imshow(image)

    image = image.reshape((image.shape[0] * image.shape[1], 3))#เปลี่ยนรูปร่างของภาพให้เป็นรายการ Pixel
    clt = KMeans(n_clusters = args["clusters"])    #จับกลุ่มของสี
    clt.fit(image)


    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    plt.figure()
    plt.axis("off")
    plt.title('Color(s) {}'.format(args["clusters"]))
	#plt.title('{}'.format(title))
    plt.imshow(bar)
    plt.show()

rw = Tk()
rw.title("โปรแกรมจำแนกสีจากภาพ")
rw.option_add("*Font","consolas 14")

l0=Label(rw, text="    ",background="black")
l0.grid(row=0,column=1)
l1=Label(rw, text="กรุณาเลือกภาพ",background="black")
l1.grid(row=3,column=2)
l2=Label(rw, text="จำนวนสีที่ต้องการจำแนก {} สี".format(args["clusters"]),background="black",fg="white")
l2.grid(row=4,column=2)
l3=Label(rw, text="     ",background="black")
l3.grid(row=5,column=3)
l4=Label(rw, text="     ",background="black")
l4.grid(row=8,column=2)

btn1=Button(rw,text="Browse...", width =30,padx=30,pady=10,bg='#FFCC00')
btn1.grid(row=6, column=2)
btn1.config(command=open_File)

btn2=Button(rw,text="About Me", width =30,padx=30,pady=10,bg='#FFCC00')
btn2.grid(row=7, column=2)
btn2.config(command=aboutme)

img = ImageTk.PhotoImage(Image.open("img.png"))
panel = Label(rw, image = img,background="black")
panel.grid(row=1,column=2)

rw['bg']='black'
rw.iconbitmap(r'icon.ico')
rw.mainloop()
