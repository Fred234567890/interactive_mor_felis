import zmq

#  Create socket to talk to server
context = zmq.Context()
print("Connecting to Felis")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

socket.send_string("initialize")
message = socket.recv()
print(message)