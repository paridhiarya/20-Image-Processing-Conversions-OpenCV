import cv2
import easygui
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import *

root = tk.Tk()
root.geometry('400x400')
root.title('Image Processing - Console:')
root.configure(background= 'cadetblue3')

label = Label(root, background = 'aquamarine4', font=("Times New Roman", 30, "bold"))

def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype("uint8") # creating table for exponent
    channel = cv2.LUT(channel, table)
    return channel

def tone(img, number):
    for i in range(3):
        if i == number:
            img[:, :, i] = exponential_function(img[:, :, i], 1.05) # applying exponential function on slice
        else:
            img[:, :, i] = 0 # setting values of all other slices to 0
    return img

def u():
    Imagepath = easygui.fileopenbox()
    c(Imagepath)

def c(Imagepath):
    orig = cv2.imread(Imagepath)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    
    if orig is None:
        print("No image found.")
        exit()
    
    R1 = cv2.resize(orig, (930, 510))
    
    img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    R2 = cv2.resize(img_gray, (930, 510))
    
    smooth = cv2.medianBlur(img_gray, 5)
    R3 = cv2.resize(smooth, (930, 510))
    
    edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    R4 = cv2.resize(edges, (930, 510))
    
    img_filter = cv2.bilateralFilter(orig, 9, 300, 300)
    R5 = cv2.resize(img_filter, (930, 510))
    
    cartoon = cv2.bitwise_and(img_filter, img_filter, mask=edges)
    R6 = cv2.resize(cartoon, (930, 510))
    
    bright = cv2.convertScaleAbs(orig, alpha=1.5, beta=10)
    R7 = cv2.resize(bright, (930, 510))
    
    dark = cv2.convertScaleAbs(orig, alpha=1.5, beta=1)
    R8 = cv2.resize(dark, (930, 510))
    
    hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    R9 =  cv2.resize(hsv, (930, 510))
    
    pencil_gray, pencil_rgb = cv2.pencilSketch(orig, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    R10 = cv2.resize(pencil_gray, (930, 510))
    R11 = cv2.resize(pencil_rgb, (930, 510))

    
    #DUO TONE FILTERS
    origcopy = np.array(orig)
    green = tone(origcopy, 1)
    R12 = cv2.resize(green, (930, 510))
    
    origcopy = np.array(orig)
    red = tone(origcopy, 2)
    R13 = cv2.resize(red, (930, 510))
    
    origcopy = np.array(orig)
    blue = tone(origcopy, 0)
    R14 = cv2.resize(blue, (930, 510))
    
    #SEPIA
    origcopy = np.array(orig, dtype=np.float64) # converting to float to prevent loss
    origcopy = cv2.transform(origcopy, np.matrix([[0.272, 0.534, 0.131],
                                        [0.349, 0.686, 0.168],
                                        [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    origcopy[np.where(origcopy > 255)] = 255 # normalizing values greater than 255 to 255
    sepia = np.array(origcopy, dtype=np.uint8) # converting back to int
    R15 = cv2.resize(sepia, (930, 510))
    
    #Embossing
    height, width = orig.shape[:2]
    y = np.ones((height, width), np.uint8) * 128
    output = np.zeros((height, width), np.uint8)
    # generating the kernels
    kernel1 = np.array([[0, -1, -1], # kernel for embossing bottom left side
                        [1, 0, -1],
                        [1, 1, 0]])
    kernel2 = np.array([[-1, -1, 0], # kernel for embossing bottom right side
                        [-1, 0, 1],
                        [0, 1, 1]])
    # you can generate kernels for embossing top as well
    output1 = cv2.add(cv2.filter2D(img_gray, -1, kernel1), y) # emboss on bottom left side
    output2 = cv2.add(cv2.filter2D(img_gray, -1, kernel2), y) # emboss on bottom right side
    for i in range(height):
        for j in range(width):
            output[i, j] = max(output1[i, j], output2[i, j]) # combining both embosses to produce stronger emboss
        
    R16 = cv2.resize(output1, (930, 510))
    R17 = cv2.resize(output2, (930, 510))
    R18 = cv2.resize(output, (930, 510))
    graycopy = np.array(img_gray)
    ret, x = cv2.threshold(graycopy,127,255,cv2.THRESH_BINARY)
    R19 = cv2.resize(x, (930, 510))
    
    invert = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    R20 = cv2.resize(invert, (930, 510))
    
    images = [R1, R2, R3, R4, R5, R6]
    titles = ['Original', 'Grayscale', 'GrayScale-Smoothened', 'Edge-Detection', 'Bilateral-Filter', 'Cartoon']
    fig, axes = plt.subplots(3, 2, figsize=(8, 14), subplot_kw={'xticks': [], 'yticks':[]}, gridspec_kw = dict(hspace = 0.4, wspace = 0.4))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap = 'gray')
        ax.set_title(titles[i])
    
    images2 = [R7, R8, R9, R10, R11, R12]
    titles2 = ['Increased-Brightness', 'Decreased-Brightness', 'HSV', 'Grayscale-Pencil-Sketch', 'Colored-Pencil-Sketch', 'Green-Duo-Tone']
    fig, axes = plt.subplots(3, 2, figsize=(8, 14), subplot_kw={'xticks': [], 'yticks':[]}, gridspec_kw = dict(hspace = 0.4, wspace = 0.4))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images2[i], cmap = 'gray')
        ax.set_title(titles2[i])
    
    images3 = [R13, R14, R15, R16, R17, R18]
    titles3 = ['Red-Duo-Tone', 'Blue-Duo-Tone', 'Sepia', 'Emboss-Bottom-Right', 'Emboss-Bottom-Left', 'Combined-Emboss']
    fig, axes = plt.subplots(3, 2, figsize=(8, 14), subplot_kw={'xticks': [], 'yticks':[]}, gridspec_kw = dict(hspace = 0.4, wspace = 0.4))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images3[i], cmap = 'gray')
        ax.set_title(titles3[i])
    
    images4 = [R19, R20]
    titles4 = ['Threshold', 'Inversion']
    fig, axes = plt.subplots(2, 1, figsize=(8, 14), subplot_kw={'xticks': [], 'yticks':[]}, gridspec_kw = dict(hspace = 0.4, wspace = 0.4))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images4[i], cmap = 'gray')
        ax.set_title(titles4[i])
        
    images = images + images2 + images3 + images4
    titles = titles + titles2 + titles3 + titles4
    save4 = Button(root, text="Save Output Images", command= lambda : save(images, titles, 20))
    save4.configure(background="cadetblue4", foreground="white", font=('Times New Roman', 20, 'bold'))
    save4.pack(side=TOP)  
    
    plt.show()
    
def save(imgs, titles, n):
    path = "C:\\Users\\Paridhi\\Dropbox\\PC\\Documents\\Projects\\Image Processing\\Output"
    os.chdir(path) 
    for i in range(1, n+1):
        name = "Result-" + titles[i-1] + ".jpg"
        cv2.imwrite(name, cv2.cvtColor(imgs[i-1], cv2.COLOR_RGB2BGR))
    I = "The images have been saved at: " + path
    tk.messagebox.showinfo(title=None, message=I)
    
a = Button(root, text="Perform Image Conversions", command=u) 
a.configure(background="aquamarine4", foreground="white", font=('Times New Roman', 20, 'bold'))
a.pack(side=TOP, pady=50)

root.mainloop()

    