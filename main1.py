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
from gtts import gTTS
from io import BytesIO

global count,tb
global language

mixer.init()
tb = mixer.Sound(r'talkback\welcome.mp3')
tb.play()

#text to audio generation--no saving of audio file only generate and play
def audio_tts(text,lang,after=""):
    try:
        tts = gTTS(text=str(text)+after, slow=False,lang=lang)
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream) 
        # Seek to the beginning of the stream
        audio_stream.seek(0)
        mixer.init()
        mixer.music.stop()
        mixer.music.load(audio_stream)
        mixer.music.play()
    except Exception as e:
        print("Exception occured in text to audio conversion",e)

def sum_up():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return   
    sum_list = [] 
    start_time = time.time()
    capture_count = 0    
    while capture_count < 3:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break        
        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', 300, 300)
        if time.time() - start_time >= 5:
            global tb
            tb.stop()
            tb=mixer.Sound(r"talkback\beep-04.mp3")
            tb.play()
            img = frame
            print(f"Frame {capture_count + 1} captured!")
            start_time = time.time()  # Reset the timer
            capture_count += 1

            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            p = pathlib.Path('C:\\roshin\\maintest\\currency_classifier.pkl')
            model = load_learner(p)
            pathlib.PosixPath = temp           
            img = PILImage.create(img)
            print("predicting...")
            pred_class, _, _ = model.predict(img)
            print(pred_class)
            
            if pred_class in ['10', '20', '50', '100', '200', '500']:
                sum_list.append(int(pred_class))
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 
    if sum_list:
        return sum(sum_list)

def lang_english():
    global count,tb
    global language
    language="English"

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

    def create_popup():
        popup = tk.Toplevel()
        popup.title("Change Language")
        popup.geometry("600x480")  # Increase the size of the popup window
        popup.configure(bg="lightcyan")

        global language
        button_frame = tk.Frame(popup, bg="lightcyan")
        button_frame.pack(pady=50)

        def on_lang_change(lang):
            global language
            language=lang
            print(f'language chosen is {language}')
            popup.destroy()
            if language=="English":
                pass
            else:
                window.destroy()
                lang_malayalam() if language=="Malayalam" else lang_hindi()

        def enter_lang_eng(event):
            global language
            if language!="English":
                global tb
                b_eng['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_english.mp3")
                tb.play()
                

        def enter_lang_mal(event):
            global language
            if language!="Malayalam":
                global tb
                b_mal['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_malayalam.mp3")
                tb.play()
        
        def enter_lang_hin(event):
            global language
            if language!="Hindi":
                global tb
                b_hin['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_hindi.mp3")
                tb.play()
        
        def leave_lang_eng(event):
            b_eng['background'] = 'lightblue'
        def leave_lang_mal(event):
            b_mal['background'] = 'lightblue'
        def leave_lang_hin(event):
            b_hin['background'] = 'lightblue'
       
        b_eng = tk.Button(button_frame, text="English", font=("Helvetica", 20), bg="lightblue", bd=10, height=2, width=30,command=lambda:on_lang_change("English"))
        b_eng.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_eng.bind("<Enter>",enter_lang_eng)
        b_eng.bind("<Leave>",leave_lang_eng)

        b_mal = tk.Button(button_frame, text="മലയാളം", font=("Helvetica", 20), bg="lightblue", bd=10, height=2,width=30,command=lambda:on_lang_change("Malayalam"))
        b_mal.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_mal.bind("<Enter>",enter_lang_mal)
        b_mal.bind("<Leave>",leave_lang_mal)
    
        b_hin = tk.Button(button_frame, text="हिंदी", font=("Helvetica", 20), bg="lightblue", bd=10,height=2, width=30,command=lambda:on_lang_change("Hindi"))
        b_hin.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_hin.bind("<Enter>",enter_lang_hin)
        b_hin.bind("<Leave>",leave_lang_hin)

        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

    def enter_changelang(event):
        global tb
        changelang['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_changelang.mp3")
        tb.play()

    def leave_changelang(event):
        changelang['background'] = 'turquoise'

    #window ui background color customization
    global color_count
    color_count=0
    colors=['white','lightsalmon','gray','blue','black']

    def change_color():
        global color_count
        if color_count >= 4:
            color_count=-1
        color = colors[color_count+1] 
        color_count+=1
        window.config(bg=color)

    def enter_invertbg(event):
        global tb
        invertbg['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_change_bg.mp3")
        tb.play()

    def leave_invertbg(event):
        invertbg['background'] = '#0074D9'

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
        tb.stop()
        detect['background']='blue'
        tb=mixer.Sound(r'talkback\tb_detect.mp3')
        tb.play()

    def leave_detect(event):
        # global tb
        # if tb is not None:
        #     tb.stop()
        # tb=mixer.Sound(r'C:\roshin\miniproject\talkback\tb_leaving_detection.mp3')
        # tb.stop()
        # tb.play()
        detect['background']='turquoise'

    def currency_detection():
        global tb
        tb.stop()
        tb=mixer.Sound(r'talkback\tb_camera_start.mp3')
        tb.play()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            return

        start_time = time.time()
        tb=mixer.Sound(r'talkback\tb_detecting_opening.mp3')
        tb.stop()
        tb.play()
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
                tb=mixer.Sound(r'talkback\beep-04.mp3')
                tb.play()
                break

            # Wait for a short duration between frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        p = pathlib.Path('C:\\roshin\\maintest\\currency_classifier.pkl')
        model = load_learner(p)
        pathlib.PosixPath = temp

        img = PILImage.create(r'C:\roshin\maintest\captured_frame.jpg')
        print("predicting...")
        pred_class,_,_ = model.predict(img)

        print("No currency detected") if pred_class =='Background' else print(f'Currency is Rs. {pred_class}/- ')

        if pred_class=='10':
            sound = mixer.Sound(r'talkback\\10.mp3')
            sound.play()
        elif pred_class=='20':
            sound = mixer.Sound(r'talkback\\20.mp3')
            sound.play()
        elif pred_class=='50':
            sound = mixer.Sound(r'talkback\\50.mp3')
            sound.play()
        elif pred_class=='100':
            sound = mixer.Sound(r'talkback\\100.mp3')
            sound.play()
        elif pred_class=='200':
            sound = mixer.Sound(r'talkback\\200.mp3')
            sound.play()
        elif pred_class=='500':
            sound = mixer.Sound(r'talkback\\500.mp3')
            sound.play()
        else:
            sound = mixer.Sound(r'talkback\background.mp3')
            sound.play()
            
    # #check model code goes here#############################################

    # def enter_checkmodel(event):
    #     global tb
    #     tb.stop()
    #     checkmodel['background']='green'
    #     tb=mixer.Sound(r'talkback\tb_model_availability.mp3')
    #     tb.play()
        

    # def leave_checkmodel(event):
    #     checkmodel['background']='turquoise'
        
    # def load_model():
    #     print("checking model...")
    #     try:
    #         temp = pathlib.PosixPath
    #         pathlib.PosixPath = pathlib.WindowsPath
    #         p = pathlib.Path('C:\\roshin\\miniproject\\currency_classifier.pkl')
    #         load_learner(p)
    #         print("Trained model successfully loaded from the disk...")
    #         tb=mixer.Sound(r"talkback\tb_check_model_ok.mp3")
    #         tb.play()
    #     except:
    #         print("Please train a model ...no trained model is existing in the disk!")
    #         tb=mixer.Sound(r"talkback\tb_check_model_not_ok.mp3")
    #         tb.play()

    #user manual code ################
    def enter_usermanual(event):
        global tb
        tb.stop()
        user_manual['background']='green'
        tb=mixer.Sound(r'talkback\tb_usermanual_info.mp3')
        tb.play()

    def leave_usermanual(event):
        user_manual['background']='turquoise'
        
    def user_manual():
        global tb
        tb.stop()
        print("User Manual being played !!!")
        tb = mixer.Sound(r'talkback\tb_usermanual.mp3')
        tb.play()
        user_manual['background']='#FF7F24'

    #currency sum code ###########################################
    def currency_sum():
            total_sum=sum_up()
            print(f"Net Amount is Rs.{total_sum}/-")
            audio_tts(total_sum,lang="en",after=" Rupees")

    def enter_sum(event):
        confusionmatrix['background'] = 'green'
        global tb
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_sum_up.mp3")
        tb.play()

    def leave_sum(event):
        confusionmatrix['background'] = 'turquoise'

    #textsize buttons enter and leave methods ############

    def enter_textminus(event):
        global tb
        textminus['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_textsize_decrease.mp3")
        tb.play()

    def leave_textminus(event):
        textminus['background'] = '#AB82FF'

    def enter_textdefault(event):
        global tb
        textdefault['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_textsize_default.mp3")
        tb.play()

    def leave_textdefault(event):
        textdefault['background'] = '#AB82FF'

    def enter_textplus(event):
        global tb
        textplus['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_textsize_increase.mp3")
        tb.play()

    def leave_textplus(event):
        textplus['background'] = '#AB82FF'

    #code for exit button ############################
    def enter_exit(event):
        global tb
        tb.stop()
        tb=mixer.Sound(r"talkback\tb_exit.mp3")
        tb.stop()
        tb.play()
        exit['background'] = 'green'

    def leave_exit(event):
        exit['background'] = 'gold'

    def prgm_close():
        print("\n\tTerminated\n")
        window.destroy()

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
        global count
        count=0
        text_size=15
        update_text_size(text_size)
    def text_size_plus():
        global count
        if count>=5:
            return
        else:
            global text_size
            count+=1
            text_size=text_size+1
            update_text_size(text_size)

    def text_size_minus():
        global count
        if count<=-5:
            return
        else:
            global text_size
            text_size=text_size-1
            count-=1
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

    # checkmodel = tk.Button(window, text ="Check Model",  
    # command = load_model, fg ="black", bg ="turquoise",  
    # width = 25, height = 5, activebackground = "Red",  
    # font =('times', text_size, ' bold ')) 
    # checkmodel.place(x = 100, y = 300)
    # checkmodel.bind("<Enter>",enter_checkmodel)
    # checkmodel.bind("<Leave>",leave_checkmodel)

    changelang = tk.Button(window, text ="Change Language",  
    command = create_popup, fg ="black", bg ="turquoise",  
    width = 25, height = 5, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    changelang.place(x = 100, y = 300)
    changelang.bind("<Enter>",enter_changelang)
    changelang.bind("<Leave>",leave_changelang)

    confusionmatrix = tk.Button(window, text ="Currency Sum",  
    command = currency_sum, fg ="black", bg ="turquoise",  
    width = 25, height = 5, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    confusionmatrix.place(x = 1100, y = 300)
    confusionmatrix.bind("<Enter>",enter_sum)
    confusionmatrix.bind("<Leave>",leave_sum)

    textminus = tk.Button(window, text ="A-",  
    command = text_size_minus, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    textminus.place(x = 900, y = 180)
    textminus.bind("<Enter>",enter_textminus)
    textminus.bind("<Leave>",leave_textminus)

    textdefault = tk.Button(window, text ="A",  
    command = default_text_size, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    textdefault.place(x = 1035, y = 180)
    textdefault.bind("<Enter>",enter_textdefault)
    textdefault.bind("<Leave>",leave_textdefault)

    textplus = tk.Button(window, text ="A+",  
    command = text_size_plus, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold '))
    textplus.place(x = 1170, y = 180)
    textplus.bind("<Enter>",enter_textplus)
    textplus.bind("<Leave>",leave_textplus)

    invertbg = tk.Button(window, text ="🌙",  
    command = change_color, fg ="black", bg ="#0074D9",  
    width = 6, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold '))
    invertbg.place(x = 1320, y = 180)
    invertbg.bind("<Enter>",enter_invertbg)
    invertbg.bind("<Leave>",leave_invertbg)

    user_manual = tk.Button(window, text ="Manual",  
    command = user_manual, fg ="black", bg ="turquoise",  
    width = 20, height = 3, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    user_manual.place(x = 100, y = 600)
    user_manual.bind("<Enter>",enter_usermanual)
    user_manual.bind("<Leave>",leave_usermanual)

    exit = tk.Button(window, text ="Exit",  
    command = prgm_close, fg ="black", bg ="gold",  
    width = 20, height = 3, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    exit.place(x = 1150, y = 630)
    exit.bind("<Enter>",enter_exit)
    exit.bind("<Leave>",leave_exit)

    ###########################################################
    window.mainloop() 

def lang_malayalam():
    global count,tb
    global language
    language="Malayalam"

    window = tk.Tk()  
    window.title("currency classification")
    window.geometry('1920x1080')
    window.configure(background ='lightcyan')
    bg = PhotoImage(file = r'')
    label1 = tk.Label(window, image = bg)
    label1.place(x = 0, y = 0)
    window.grid_rowconfigure(0, weight = 1) 
    window.grid_columnconfigure(0, weight = 1) 
    message = tk.Label( 
        window, text ="കറൻസി മൂല്യനിർണ്ണയം",  
        bg ="lightsalmon", fg = "black", width = 56,  
        height = 3, font = ('times', 30, 'bold'))
    message.place(x = 75, y = 20)

    def create_popup():
        popup = tk.Toplevel()
        popup.title("Change Language")
        popup.geometry("600x480")  # Increase the size of the popup window
        popup.configure(bg="lightcyan")

        global language
        button_frame = tk.Frame(popup, bg="lightcyan")
        button_frame.pack(pady=50)

        def on_lang_change(lang):
            global language
            language=lang
            print(f'language chosen is {language}')
            popup.destroy()
            if language=="Malayalam":
                pass
            else:
                window.destroy()
                lang_english() if language=="English" else lang_hindi()

        def enter_lang_eng(event):
            global language
            if language!="English":
                global tb
                b_eng['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_english.mp3")
                tb.play()
                
        def enter_lang_mal(event):
            global language
            if language!="Malayalam":
                global tb
                b_mal['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_malayalam.mp3")
                tb.play()
        
        def enter_lang_hin(event):
            global language
            if language!="Hindi":
                global tb
                b_hin['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_hindi.mp3")
                tb.play()
        
        def leave_lang_eng(event):
            b_eng['background'] = 'lightblue'
        def leave_lang_mal(event):
            b_mal['background'] = 'lightblue'
        def leave_lang_hin(event):
            b_hin['background'] = 'lightblue'
        
        b_eng = tk.Button(button_frame, text="English", font=("Helvetica", 20), bg="lightblue", bd=10, height=2, width=30,command=lambda:on_lang_change("English"))
        b_eng.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_eng.bind("<Enter>",enter_lang_eng)
        b_eng.bind("<Leave>",leave_lang_eng)

        b_mal = tk.Button(button_frame, text="മലയാളം", font=("Helvetica", 20), bg="lightblue", bd=10, height=2,width=30,command=lambda:on_lang_change("Malayalam"))
        b_mal.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_mal.bind("<Enter>",enter_lang_mal)
        b_mal.bind("<Leave>",leave_lang_mal)
    
        b_hin = tk.Button(button_frame, text="हिंदी", font=("Helvetica", 20), bg="lightblue", bd=10,height=2, width=30,command=lambda:on_lang_change("Hindi"))
        b_hin.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_hin.bind("<Enter>",enter_lang_hin)
        b_hin.bind("<Leave>",leave_lang_hin)

        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

    def enter_changelang(event):
        global tb
        changelang['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_changelang_ml.mp3")
        tb.play()

    def leave_changelang(event):
        changelang['background'] = 'turquoise'

    #window ui background color customization
    global color_count
    color_count=0
    colors=['white','lightsalmon','gray','blue','black']

    def change_color():
        global color_count
        if color_count >= 4:
            color_count=-1
        color = colors[color_count+1] 
        color_count+=1
        window.config(bg=color)

    def enter_invertbg(event):
        global tb
        invertbg['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_change_bg_ml.mp3")
        tb.play()

    def leave_invertbg(event):
        invertbg['background'] = '#0074D9'

    #detection code goes here #####################################################
    def enter_detect(event):
        global tb
        tb.stop()
        detect['background']='blue'
        tb=mixer.Sound(r'talkback_ml\tb_detect_ml.mp3')
        tb.play()

    def leave_detect(event):
        detect['background']='turquoise'

    def currency_detection():
        global tb
        tb.stop()
        tb=mixer.Sound(r'talkback_ml\tb_camera_start_ml.mp3')
        tb.play()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            return

        start_time = time.time()
        tb=mixer.Sound(r'talkback_ml\tb_detecting_opening_ml.mp3')
        tb.stop()
        tb.play()
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
                tb=mixer.Sound(r'talkback\beep-04.mp3')
                tb.play()
                break

            # Wait for a short duration between frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        p = pathlib.Path('C:\\roshin\\maintest\\currency_classifier.pkl')
        model = load_learner(p)
        pathlib.PosixPath = temp

        img = PILImage.create(r'C:\roshin\maintest\captured_frame.jpg')
        print("predicting...")
        pred_class,_,_ = model.predict(img)

        print("No currency detected") if pred_class =='Background' else print(f'Currency is Rs. {pred_class}/- ')

        if pred_class=='10':
            sound = mixer.Sound(r'talkback_ml\\10_ml.mp3')
            sound.play()
        elif pred_class=='20':
            sound = mixer.Sound(r'talkback_ml\\20_ml.mp3')
            sound.play()
        elif pred_class=='50':
            sound = mixer.Sound(r'talkback_ml\\50_ml.mp3')
            sound.play()
        elif pred_class=='100':
            sound = mixer.Sound(r'talkback_ml\\100_ml.mp3')
            sound.play()
        elif pred_class=='200':
            sound = mixer.Sound(r'talkback_ml\\200_ml.mp3')
            sound.play()
        elif pred_class=='500':
            sound = mixer.Sound(r'talkback_ml\\500_ml.mp3')
            sound.play()
        else:
            sound = mixer.Sound(r'talkback_ml\background_ml.mp3')
            sound.play()
            
    #user manual code ################
    def enter_usermanual(event):
        global tb
        tb.stop()
        user_manual['background']='green'
        tb=mixer.Sound(r'talkback_ml\tb_usermanual_info_ml.mp3')
        tb.play()

    def leave_usermanual(event):
        user_manual['background']='turquoise'

    def user_manual():
        global tb
        tb.stop()
        print("User Manual being played !!!")
        tb = mixer.Sound(r'talkback_ml\tb_usermanual_ml.mp3')
        tb.play()
        user_manual['background']='#FF7F24'
        

    #currency sum code ###########################################
    def currency_sum():
        total_sum=sum_up()
        print(f"Net Amount is Rs.{total_sum}/-")
        audio_tts(total_sum,lang="ml",after=" രൂപ")

    def enter_sum(event):
        confusionmatrix['background'] = 'green'
        global tb
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_sum_up_ml.mp3")
        tb.play()

    def leave_sum(event):
        confusionmatrix['background'] = 'turquoise'

    #textsize buttons enter and leave methods ############
    def enter_textminus(event):
        global tb
        textminus['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_textsize_decrease_ml.mp3")
        tb.play()

    def leave_textminus(event):
        textminus['background'] = '#AB82FF'

    def enter_textdefault(event):
        global tb
        textdefault['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_textsize_default_ml.mp3")
        tb.play()

    def leave_textdefault(event):
        textdefault['background'] = '#AB82FF'

    def enter_textplus(event):
        global tb
        textplus['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_textsize_increase_ml.mp3")
        tb.play()

    def leave_textplus(event):
        textplus['background'] = '#AB82FF'

    #code for exit button ############################
    def enter_exit(event):
        global tb
        tb.stop()
        tb=mixer.Sound(r"talkback_ml\tb_exit_ml.mp3")
        tb.stop()
        tb.play()
        exit['background'] = 'green'

    def leave_exit(event):
        exit['background'] = 'gold'

    def prgm_close():
        print("\n\tTerminated\n")
        window.destroy()

    #text size dealing code ##################
    def update_text_size(text_size):
        widgets_to_adjust = window.winfo_children()
        new_font = ("times", text_size)
        for widget in widgets_to_adjust:
            if widget != message:
                widget.config(font=new_font)
    def default_text_size():
        global text_size
        global count
        count=0
        text_size=15
        update_text_size(text_size)
    def text_size_plus():
        global count
        if count>=5:
            return
        else:
            global text_size
            count+=1
            text_size=text_size+1
            update_text_size(text_size)

    def text_size_minus():
        global count
        if count<=-5:
            return
        else:
            global text_size
            text_size=text_size-1
            count-=1
            update_text_size(text_size)

    default_text_size()

#################################################################################

    detect = tk.Button(window, text ="മൂല്യനിർണ്ണയം!!",  
    command = currency_detection, fg ="black", bg ="turquoise",  
    width = 30, height = 10, activebackground = "Red",  
    font =('times', text_size+3, ' bold ')) 
    detect.place(x = 540, y = 300)
    detect.bind("<Enter>",enter_detect)
    detect.bind("<Leave>",leave_detect)

    changelang = tk.Button(window, text ="ഭാഷ മാറ്റുക",  
    command = create_popup, fg ="black", bg ="turquoise",  
    width = 25, height = 5, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    changelang.place(x = 100, y = 300)
    changelang.bind("<Enter>",enter_changelang)
    changelang.bind("<Leave>",leave_changelang)

    confusionmatrix = tk.Button(window, text ="തുക സംഗ്രഹിക്കുക",  
    command = currency_sum, fg ="black", bg ="turquoise",  
    width = 25, height = 5, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    confusionmatrix.place(x = 1100, y = 300)
    confusionmatrix.bind("<Enter>",enter_sum)
    confusionmatrix.bind("<Leave>",leave_sum)

    textminus = tk.Button(window, text ="അ-",  
    command = text_size_minus, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    textminus.place(x = 900, y = 180)
    textminus.bind("<Enter>",enter_textminus)
    textminus.bind("<Leave>",leave_textminus)

    textdefault = tk.Button(window, text ="അ",  
    command = default_text_size, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    textdefault.place(x = 1035, y = 180)
    textdefault.bind("<Enter>",enter_textdefault)
    textdefault.bind("<Leave>",leave_textdefault)

    textplus = tk.Button(window, text ="അ+",  
    command = text_size_plus, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold '))
    textplus.place(x = 1170, y = 180)
    textplus.bind("<Enter>",enter_textplus)
    textplus.bind("<Leave>",leave_textplus)

    invertbg = tk.Button(window, text ="🌙",  
    command = change_color, fg ="black", bg ="#0074D9",  
    width = 6, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold '))
    invertbg.place(x = 1320, y = 180)
    invertbg.bind("<Enter>",enter_invertbg)
    invertbg.bind("<Leave>",leave_invertbg)

    user_manual = tk.Button(window, text ="ഉപയോഗ നിർദേശങ്ങൾ",  
    command = user_manual, fg ="black", bg ="turquoise",  
    width = 25, height = 3, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    user_manual.place(x = 100, y = 600)
    user_manual.bind("<Enter>",enter_usermanual)
    user_manual.bind("<Leave>",leave_usermanual)

    exit = tk.Button(window, text ="പുറത്തുകടക്കുക",  
    command = prgm_close, fg ="black", bg ="gold",  
    width = 20, height = 3, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    exit.place(x = 1150, y = 630)
    exit.bind("<Enter>",enter_exit)
    exit.bind("<Leave>",leave_exit)

    ###########################################################
    window.mainloop() 

def lang_hindi():
    global count,tb
    global language
    language="Hindi"

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
        window, text ="मुद्रा मूल्य पहचान",  
        bg ="lightsalmon", fg = "black", width = 56,  
        height = 3, font = ('times', 30, 'bold'))
    message.place(x = 75, y = 20)

    def create_popup():
        popup = tk.Toplevel()
        popup.title("Change Language")
        popup.geometry("600x480")  # Increase the size of the popup window
        popup.configure(bg="lightcyan")

        global language
        button_frame = tk.Frame(popup, bg="lightcyan")
        button_frame.pack(pady=50)

        def on_lang_change(lang):
            global language
            language=lang
            print(f'language chosen is {language}')
            popup.destroy()
            if language=="Hindi":
                pass
            else:
                window.destroy()
                lang_malayalam() if language=="Malayalam" else lang_english()

        def enter_lang_eng(event):
            global language
            if language!="English":
                global tb
                b_eng['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_english.mp3")
                tb.play()
                

        def enter_lang_mal(event):
            global language
            if language!="Malayalam":
                global tb
                b_mal['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_malayalam.mp3")
                tb.play()
        
        def enter_lang_hin(event):
            global language
            if language!="Hindi":
                global tb
                b_hin['background'] = 'green'
                tb.stop()
                tb=mixer.Sound("tb_hindi.mp3")
                tb.play()
        
        def leave_lang_eng(event):
            b_eng['background'] = 'lightblue'
        def leave_lang_mal(event):
            b_mal['background'] = 'lightblue'
        def leave_lang_hin(event):
            b_hin['background'] = 'lightblue'
        
        b_eng = tk.Button(button_frame, text="English", font=("Helvetica", 20), bg="lightblue", bd=10, height=2, width=30,command=lambda:on_lang_change("English"))
        b_eng.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_eng.bind("<Enter>",enter_lang_eng)
        b_eng.bind("<Leave>",leave_lang_eng)

        b_mal = tk.Button(button_frame, text="മലയാളം", font=("Helvetica", 20), bg="lightblue", bd=10, height=2,width=30,command=lambda:on_lang_change("Malayalam"))
        b_mal.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_mal.bind("<Enter>",enter_lang_mal)
        b_mal.bind("<Leave>",leave_lang_mal)
    
        b_hin = tk.Button(button_frame, text="हिंदी", font=("Helvetica", 20), bg="lightblue", bd=10,height=2, width=30,command=lambda:on_lang_change("Hindi"))
        b_hin.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        b_hin.bind("<Enter>",enter_lang_hin)
        b_hin.bind("<Leave>",leave_lang_hin)

        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

    def enter_changelang(event):
        global tb
        changelang['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_changelang_hi.mp3")
        tb.play()

    def leave_changelang(event):
        changelang['background'] = 'turquoise'

    #window ui background color customization
    global color_count
    color_count=0
    colors=['white','lightsalmon','gray','blue','black']

    def change_color():
        global color_count
        if color_count >= 4:
            color_count=-1
        color = colors[color_count+1] 
        color_count+=1
        window.config(bg=color)

    def enter_invertbg(event):
        global tb
        invertbg['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_change_bg_hi.mp3")
        tb.play()

    def leave_invertbg(event):
        invertbg['background'] = '#0074D9'

    #detection code goes here #####################################################
    def enter_detect(event):
        global tb
        tb.stop()
        detect['background']='blue'
        tb=mixer.Sound(r'talkback_hi\tb_detect_hi.mp3')
        tb.play()

    def leave_detect(event):
        detect['background']='turquoise'

    def currency_detection():
        global tb
        tb.stop()
        tb=mixer.Sound(r'talkback_hi\tb_camera_start_hi.mp3')
        tb.play()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            return

        start_time = time.time()
        tb=mixer.Sound(r'talkback_hi\tb_detecting_opening_hi.mp3')
        tb.stop()
        tb.play()
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
                tb=mixer.Sound(r'talkback\beep-04.mp3')
                tb.play()
                break

            # Wait for a short duration between frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        p = pathlib.Path('C:\\roshin\\maintest\\currency_classifier.pkl')
        model = load_learner(p)
        pathlib.PosixPath = temp

        img = PILImage.create(r'C:\roshin\maintest\captured_frame.jpg')
        print("predicting...")
        pred_class,_,_ = model.predict(img)

        print("No currency detected") if pred_class =='Background' else print(f'Currency is Rs. {pred_class}/- ')

        if pred_class=='10':
            sound = mixer.Sound(r'talkback_hi\\10_hi.mp3')
            sound.play()
        elif pred_class=='20':
            sound = mixer.Sound(r'talkback_hi\\20_hi.mp3')
            sound.play()
        elif pred_class=='50':
            sound = mixer.Sound(r'talkback_hi\\50_hi.mp3')
            sound.play()
        elif pred_class=='100':
            sound = mixer.Sound(r'talkback_hi\\100_hi.mp3')
            sound.play()
        elif pred_class=='200':
            sound = mixer.Sound(r'talkback_hi\\200_hi.mp3')
            sound.play()
        elif pred_class=='500':
            sound = mixer.Sound(r'talkback_hi\\500_hi.mp3')
            sound.play()
        else:
            sound = mixer.Sound(r'talkback_hi\background_hi.mp3')
            sound.play()
            
    #user manual code ################
    def enter_usermanual(event):
        global tb
        tb.stop()
        user_manual['background']='green'
        tb=mixer.Sound(r'talkback_hi\tb_usermanual_info_hi.mp3')
        tb.play()

    def leave_usermanual(event):
        user_manual['background']='turquoise'
        
    def user_manual():
        global tb
        tb.stop()
        print("User Manual being played !!!")
        tb = mixer.Sound(r'talkback_hi\tb_usermanual_hi.mp3')
        tb.play()
        user_manual['background']='#FF7F24'

    #currency sum code ###########################################
    def currency_sum():
        total_sum = sum_up()
        print(f"Net Amount is Rs.{total_sum}/-")
        audio_tts(total_sum,lang="hi",after=" रुपया")

    def enter_sum(event):
        confusionmatrix['background'] = 'green'
        global tb
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_sum_up_hi.mp3")
        tb.play()

    def leave_sum(event):
        confusionmatrix['background'] = 'turquoise'

    #textsize buttons enter and leave methods ############
    def enter_textminus(event):
        global tb
        textminus['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_textsize_decrease_hi.mp3")
        tb.play()

    def leave_textminus(event):
        textminus['background'] = '#AB82FF'

    def enter_textdefault(event):
        global tb
        textdefault['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_textsize_default_hi.mp3")
        tb.play()

    def leave_textdefault(event):
        textdefault['background'] = '#AB82FF'

    def enter_textplus(event):
        global tb
        textplus['background'] = 'green'
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_textsize_increase_hi.mp3")
        tb.play()

    def leave_textplus(event):
        textplus['background'] = '#AB82FF'

    #code for exit button ############################
    def enter_exit(event):
        global tb
        tb.stop()
        tb=mixer.Sound(r"talkback_hi\tb_exit_hi.mp3")
        tb.stop()
        tb.play()
        exit['background'] = 'green'

    def leave_exit(event):
        exit['background'] = 'gold'

    def prgm_close():
        print("\n\tTerminated\n")
        window.destroy()

    #text size dealing code ##################
    def update_text_size(text_size):
        widgets_to_adjust = window.winfo_children()
        new_font = ("times", text_size)
        for widget in widgets_to_adjust:
            if widget != message:
                widget.config(font=new_font)
    def default_text_size():
        global text_size
        global count
        count=0
        text_size=15
        update_text_size(text_size)
    def text_size_plus():
        global count
        if count>=5:
            return
        else:
            global text_size
            count+=1
            text_size=text_size+1
            update_text_size(text_size)

    def text_size_minus():
        global count
        if count<=-5:
            return
        else:
            global text_size
            text_size=text_size-1
            count-=1
            update_text_size(text_size)

    default_text_size()

#############################################################################
    detect = tk.Button(window, text ="नोट मूल्य!!",  
    command = currency_detection, fg ="black", bg ="turquoise",  
    width = 30, height = 10, activebackground = "Red",  
    font =('times', text_size+3, ' bold ')) 
    detect.place(x = 540, y = 300)
    detect.bind("<Enter>",enter_detect)
    detect.bind("<Leave>",leave_detect)

    changelang = tk.Button(window, text ="भाषा बदलें",  
    command = create_popup, fg ="black", bg ="turquoise",  
    width = 25, height = 5, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    changelang.place(x = 100, y = 300)
    changelang.bind("<Enter>",enter_changelang)
    changelang.bind("<Leave>",leave_changelang)

    confusionmatrix = tk.Button(window, text ="धन का योग करो",  
    command = currency_sum, fg ="black", bg ="turquoise",  
    width = 25, height = 5, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    confusionmatrix.place(x = 1100, y = 300)
    confusionmatrix.bind("<Enter>",enter_sum)
    confusionmatrix.bind("<Leave>",leave_sum)

    textminus = tk.Button(window, text ="अ-",  
    command = text_size_minus, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    textminus.place(x = 900, y = 180)
    textminus.bind("<Enter>",enter_textminus)
    textminus.bind("<Leave>",leave_textminus)

    textdefault = tk.Button(window, text ="अ",  
    command = default_text_size, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    textdefault.place(x = 1035, y = 180)
    textdefault.bind("<Enter>",enter_textdefault)
    textdefault.bind("<Leave>",leave_textdefault)

    textplus = tk.Button(window, text ="अ+",  
    command = text_size_plus, fg ="black", bg ="#AB82FF",  
    width = 8, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold '))
    textplus.place(x = 1170, y = 180)
    textplus.bind("<Enter>",enter_textplus)
    textplus.bind("<Leave>",leave_textplus)

    invertbg = tk.Button(window, text ="🌙",  
    command = change_color, fg ="black", bg ="#0074D9",  
    width = 6, height = 1, activebackground = "Red",  
    font =('times', text_size, ' bold '))
    invertbg.place(x = 1320, y = 180)
    invertbg.bind("<Enter>",enter_invertbg)
    invertbg.bind("<Leave>",leave_invertbg)

    user_manual = tk.Button(window, text ="उपयोग करने के निर्देश",  
    command = user_manual, fg ="black", bg ="turquoise",  
    width = 20, height = 3, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    user_manual.place(x = 100, y = 600)
    user_manual.bind("<Enter>",enter_usermanual)
    user_manual.bind("<Leave>",leave_usermanual)

    exit = tk.Button(window, text ="बाहर निकलना",  
    command = prgm_close, fg ="black", bg ="gold",  
    width = 20, height = 3, activebackground = "Red",  
    font =('times', text_size, ' bold ')) 
    exit.place(x = 1150, y = 630)
    exit.bind("<Enter>",enter_exit)
    exit.bind("<Leave>",leave_exit)

    ###########################################################
    window.mainloop() 


#####################starting of the code
if __name__=="__main__":
    lang_english()
