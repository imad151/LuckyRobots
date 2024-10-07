import luckyrobots as lr

@lr.on("robot_output")
def printoutput(msg):
    print(f"robot output: {msg}")

@lr.on("message")
def printmessage(msg):
    print(f"robot message: {msg}")

lr.start()