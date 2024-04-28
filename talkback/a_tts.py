from gtts import gTTS

name="tb_changelang"
text="Change language"

tts=gTTS(text=text,slow=False)
tts.save(name+'.mp3')
print("saved successfully😊")