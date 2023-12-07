import tkinter as tk
from tkinter import *
from PIL import *
import matplotlib.pyplot as plt
from pygame import mixer
import time
import cv2
import pathlib
from fastai.vision.all import *
from PIL import Image
from fastai.learner import load_learner
from fastcore.all import *
from fastai import *
from fastai.vision import *
from fastai.vision.widgets import *
import ipywidgets as widgets





mixer.init()
sound = mixer.Sound(r'C:\roshin\miniproject\talkback\welcome.mp3')
sound.play()



print("\n\n \tOpening Currency Classification...\n\n")

#designing the template using tkinter library
window = tk.Tk()  
window.title("currency classification")
window.geometry('1920x1080')

window.configure(background ='lightcyan')

bg = PhotoImage(file = r'')

#Show image using label
label1 = tk.Label(window, image = bg)
label1.place(x = 0, y = 0)


window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 
message = tk.Label( 
    window, text ="Currency Detection",  
    bg ="lightsalmon", fg = "black", width = 56,  
    height = 3, font = ('times', 30, 'bold'))
message.place(x = 75, y = 20)





#Trainig code goes here ###############################################

# def enter_train(event):
#     tb=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_train_model.mp3')
#     tb.stop()
#     tb.play()
#     trainImg['background']='green'

# def leave_train(event):
#     trainImg['background']='turquoise'

# def train_model():
#     # path = r'C:\roshin\miniproject\dataset\dataset_main'
#     # failed = verify_images(get_image_files(path))
#     # failed.map(Path.unlink)
#     # len(failed)

#     data = ImageDataLoaders.from_folder(r'C:\roshin\miniproject\dataset\dataset1',
#                                     train='training',
#                                     valid='validation',
#                                     bs=16,
#                                     seed=42,
#                                     item_tfms=Resize(360,method=ResizeMethod.Squish))
    
#     audio=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_training_model.mp3')
#     audio.play()
    
#     learn = vision_learner(data,resnet34, metrics= [error_rate,accuracy])

#     learn.fine_tune(9, 0.001, freeze_epochs=3,cbs=[SaveModelCallback()])
#     print("Training for 9 epochs...")

#     #constructing confusion matrix
#     interp = ClassificationInterpretation.from_learner(learn)
#     interp.plot_confusion_matrix()
#     plt.savefig(r'C:\roshin\miniproject\confusion_matrix.png')

#     #save the trained model to the disk...
#     learn.export(r'C:\roshin\miniproject\currency_classifier.pkl')

#     audio=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_training_success.mp3')
#     audio.play()



#detection code goes here #####################################################
def enter_detect(event):
    global tb
    tb=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_detect.mp3')
    tb.stop()
    tb.play()
    detect['background']='green'

def leave_detect(event):
    # global tb
    # tb=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_leaving_detection.mp3')
    # tb.stop()
    # tb.play()
    detect['background']='turquoise'

def currency_detection():
    audio=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_camera_start.mp3')
    audio.play()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    start_time = time.time()
    audio=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_detecting_opening.mp3')
    audio.stop()
    audio.play()
    #cv2.moveWindow('Show your currency', 500, 500)
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame")
            break

        cv2.imshow('Show your currency ', frame)

        if time.time() - start_time >= 5:
            cv2.imwrite('captured_frame.jpg', frame)
            print("Picture captured!")
            audio=mixer.Sound(r'C:\roshin\miniproject\talkback\beep-04.mp3')
            audio.play()
            break

        # Wait for a short duration between frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    temp = pathlib.PosixPath

    pathlib.PosixPath = pathlib.WindowsPath

    p = pathlib.Path('C:\\roshin\\miniproject\\currency_classifier.pkl')
    model = load_learner(p)

    pathlib.PosixPath = temp

    img = PILImage.create(r'C:\roshin\miniproject\captured_frame.jpg')
    print("predicting...")
    pred_class,_,_ = model.predict(img)

    print(f'Currency is Rs. {pred_class}/- ')

    if pred_class=='10':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\ten.mp3')
        sound.play()
    if pred_class=='20':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\twenty.mp3')
        sound.play()
    if pred_class=='50':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\fifty.mp3')
        sound.play()
    if pred_class=='100':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\100.mp3')
        sound.play()
    if pred_class=='200':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\200.mp3')
        sound.play()
    if pred_class=='500':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\500.mp3')
        sound.play()
    if pred_class=='2000':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\2000.mp3')
        sound.play()
    if pred_class=='Background':
        sound = mixer.Sound(r'C:\roshin\miniproject\talkback\tb_background.mp3')
        sound.play()
        

#check model code goes here#############################################

def enter_checkmodel(event):
    tb=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_model_availability.mp3')
    tb.stop()
    tb.play()
    checkmodel['background']='green'

def leave_checkmodel(event):
    checkmodel['background']='turquoise'
    
def load_model():
    print("checking model...")
    try:
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        p = pathlib.Path('C:\\roshin\\miniproject\\currency_classifier.pkl')
        load_learner(p)
        print("Trained model successfully loaded from the disk...")
        tb=mixer.Sound(r"C:\roshin\miniproject\talkback\tb_check_model_ok.mp3")
        tb.play()
    except:
        print("Please train a model ...no trained model is existing in the disk!")
        tb=mixer.Sound(r"C:\roshin\miniproject\talkback\tb_check_model_not_ok.mp3")
        tb.play()


#details code goes here ################################

# def enter_graph(event):
#     graph['background']='green'

# def leave_graph(event):
#     graph['background']='turquoise'
# def graph_show():
#     print("Displaying Graph of training") 
#     img = cv2.imread(r"C:\roshin\miniproject\graph.png")
#     winname = "Training graph (Press 'q' to exit)"
#     cv2.namedWindow(winname)        # Create a named window
#     cv2.moveWindow(winname, 360,30)  # Move it to (360,30)
#     cv2.imshow(winname, img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#user manual code ################
def enter_usermanual(event):
    user_manual['background']='green'
    tb=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_usermanual_info.mp3')
    tb.play()

def leave_usermanual(event):
    user_manual['background']='turquoise'
    

is_playing = False
def user_manual():
    global tb, is_playing
    print("User Manual being played !!!")
    if not is_playing:
        tb = mixer.Sound(r'C:\roshin\miniproject\talkback\tb_usermanual.mp3')
        tb.play()
        user_manual['background']='#FF7F24'
        is_playing = True
    else:
        tb.stop()
        is_playing = False












#confusion matrix code ###########################################
def confusion_matrix():
    print("Displaying Confusion Matrix...") 
    img = cv2.imread(r"C:\roshin\miniproject\confusion_matrix.png")
    winname = "confusion matrix (Press 'q' to exit)"
    cv2.namedWindow(winname)        
    cv2.moveWindow(winname, 360,30)  
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def enter_confusion(event):
    confusionmatrix['background'] = 'green'

def leave_confusion(event):
    confusionmatrix['background'] = 'turquoise'


#textsize buttons enter and leave methods ############

def enter_textminus(event):
    textminus['background'] = 'green'
    tb=mixer.Sound(r"C:\roshin\miniproject\talkback\tb_textsize_decrease.mp3")
    tb.play()

def leave_textminus(event):
    textminus['background'] = '#AB82FF'


def enter_textdefault(event):
    textdefault['background'] = 'green'
    tb=mixer.Sound(r"C:\roshin\miniproject\talkback\tb_textsize_default.mp3")
    tb.play()

def leave_textdefault(event):
    textdefault['background'] = '#AB82FF'

def enter_textplus(event):
    textplus['background'] = 'green'
    tb=mixer.Sound(r"C:\roshin\miniproject\talkback\tb_textsize_increase.mp3")
    tb.play()

def leave_textplus(event):
    textplus['background'] = '#AB82FF'

#code for exit button ############################
def enter_exit(event):
    tb=mixer.Sound(r"C:\roshin\miniproject\talkback\tb_exit.mp3")
    tb.stop()
    tb.play()
    exit['background'] = 'green'

def leave_exit(event):
    exit['background'] = 'gold'

def prgm_close():
    print("\n\tTerminated\n")
    audio=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_closing.mp3')
    audio.play()
    window.quit()

##########################################
#text size dealing code ##################

def update_text_size(text_size):
    widgets_to_adjust = window.winfo_children()
    new_font = ("times", text_size)
    for widget in widgets_to_adjust:
        if widget != message:
            # if widget != textplus:
            #     if widget != textminus:
            #         if widget != textdefault:
            widget.config(font=new_font)
def default_text_size():
    global text_size
    text_size=15
    update_text_size(text_size)
def text_size_plus():
    global text_size
    text_size=text_size+1
    update_text_size(text_size)

def text_size_minus():
    global text_size
    text_size=text_size-1
    update_text_size(text_size)

default_text_size()




#code for buttons
# trainImg = tk.Button(window, text ="Train Model",  
# command = train_model, fg ="black", bg ="turquoise",  
# width = 25, height = 5, activebackground = "Red",  
# font =('times', text_size, ' bold ')) 
# trainImg.place(x = 100, y = 300)
# trainImg.bind("<Enter>",enter_train)
# trainImg.bind("<Leave>",leave_train)

detect = tk.Button(window, text ="DETECT!!",  
command = currency_detection, fg ="black", bg ="turquoise",  
width = 30, height = 10, activebackground = "Red",  
font =('times', text_size+3, ' bold ')) 
detect.place(x = 540, y = 300)
detect.bind("<Enter>",enter_detect)
detect.bind("<Leave>",leave_detect)

user_manual = tk.Button(window, text ="Manual",  
command = user_manual, fg ="black", bg ="turquoise",  
width = 25, height = 5, activebackground = "Red",  
font =('times', text_size, ' bold ')) 
user_manual.place(x = 1100, y = 300)
user_manual.bind("<Enter>",enter_usermanual)
user_manual.bind("<Leave>",leave_usermanual)

checkmodel = tk.Button(window, text ="Check Model",  
command = load_model, fg ="black", bg ="turquoise",  
width = 25, height = 5, activebackground = "Red",  
font =('times', text_size, ' bold ')) 
checkmodel.place(x = 100, y = 300)
checkmodel.bind("<Enter>",enter_checkmodel)
checkmodel.bind("<Leave>",leave_checkmodel)


# graph = tk.Button(window, text ="Training details",  
# command = graph_show, fg ="black", bg ="turquoise",  
# width = 20, height = 3, activebackground = "Red",  
# font =('times', text_size, ' bold ')) 
# graph.place(x = 100, y = 500)
# graph.bind("<Enter>",enter_graph)
# graph.bind("<Leave>",leave_graph)


textminus = tk.Button(window, text ="A-",  
command = text_size_minus, fg ="black", bg ="#AB82FF",  
width = 8, height = 1, activebackground = "Red",  
font =('times', text_size, ' bold ')) 
textminus.place(x = 1000, y = 180)
textminus.bind("<Enter>",enter_textminus)
textminus.bind("<Leave>",leave_textminus)


textdefault = tk.Button(window, text ="A",  
command = default_text_size, fg ="black", bg ="#AB82FF",  
width = 8, height = 1, activebackground = "Red",  
font =('times', text_size, ' bold ')) 
textdefault.place(x = 1135, y = 180)
textdefault.bind("<Enter>",enter_textdefault)
textdefault.bind("<Leave>",leave_textdefault)

textplus = tk.Button(window, text ="A+",  
command = text_size_plus, fg ="black", bg ="#AB82FF",  
width = 8, height = 1, activebackground = "Red",  
font =('times', text_size, ' bold '))
textplus.place(x = 1270, y = 180)
textplus.bind("<Enter>",enter_textplus)
textplus.bind("<Leave>",leave_textplus)


confusionmatrix = tk.Button(window, text ="Confusion Matrix",  
command = confusion_matrix, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', text_size, ' bold ')) 
confusionmatrix.place(x = 100, y = 600)
confusionmatrix.bind("<Enter>",enter_confusion)
confusionmatrix.bind("<Leave>",leave_confusion)


exit = tk.Button(window, text ="Exit",  
command = prgm_close, fg ="black", bg ="gold",  
width = 20, height = 3, activebackground = "Red",  
font =('times', text_size, ' bold ')) 
exit.place(x = 1150, y = 630)
exit.bind("<Enter>",enter_exit)
exit.bind("<Leave>",leave_exit)

###########################################################



window.mainloop() 
