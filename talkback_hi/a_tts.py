from gtts import gTTS
import os

name="tb_usermanual_hi"
text="मुद्रा मूल्य पहचान बटन का पता लगाने के लिए स्क्रीन पर धीरे-धीरे आगे बढ़ें। पता लगाने वाले बटन पर पहुंचने पर, क्लिक करें और कुछ सेकंड तक प्रतीक्षा करें। मुद्रा को अपने हाथ में डिवाइस से ठीक पहले पकड़ने के लिए निर्देशों का पालन करें। एक बार जब आप एक बीप सुन लें तो मुद्रा हटा लें और आपको ध्वनि के माध्यम से मुद्रा मूल्य के बारे में सूचित कर दिया जाएगा। कृपया ध्यान रखें कि यह प्रणाली केवल भारतीय मुद्रा का वर्गीकरण और पता लगाती है।"
file=r"C:\roshin\maintest\talkback_hi"
cur=name+'.mp3'
fullpath=os.path.join(file,cur)
tts=gTTS(text=text,slow=False,lang="hi")
tts.save(fullpath)
print("saved successfully😊")