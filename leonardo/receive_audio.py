from pyo import *

s = Server().boot()
#s.start()

player1 = None
player2 = None

def play_sound(address, file_path):
    global player1, player2
    print(f"Received file path: {file_path}")

    if file_path.split("/")[-1] in ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav", "7.wav", "8.wav", "9.wav", "10.wav"]:
        player1 = SfPlayer(file_path, loop=False).out()
    else:
        player2 = SfPlayer(file_path, loop=False).out()

osc = OscDataReceive(port=9000, address="/var", function=play_sound)

s.gui(locals())
