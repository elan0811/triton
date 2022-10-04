import socket
import subprocess
import signal
import os
from pathlib import Path

class Tritonserver_handler():

	def __init__(self, ip, port, model_repo, shell_script_loc, debug=False):
		self.ip = ip # "127.0.0.1"
		self.port  = port   #= 35491
		self.request_count = 0
		self.proc = None
		self.buffer = 1024
		self.model_repo = model_repo
		self.shell_script_loc = shell_script_loc  #'/home/lilun/work/projects/poc/py_server/triton.sh'
		self.clientConnection = None
		self.debug = debug

	def triton_server_start(self):
		#self.proc = subprocess.Popen([self.shell_script_loc, self.model_repo], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
		print(self.shell_script_loc)
		print(self.model_repo)
		Path(self.model_repo).mkdir(parents=True, exist_ok=True)

		self.proc = subprocess.Popen(['sh', self.shell_script_loc, self.model_repo], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
		self.clientConnection.send(str.encode("Start tritonserver ......"))
		if self.debug == True:
			with self.proc.stdout:
				for line in iter(self.proc.stdout.readline, b''):
					print (line)
			self.proc.wait()

	def triton_server_stop(self):
		os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
		self.clientConnection.send(str.encode("Kill tritonserver ......"))

	def handle_request(self):

		serverSocket = socket.socket()
		print("Server socket created")
		serverSocket.bind((self.ip, self.port))
		print("Server socket bound with with ip {} port {}".format(self.ip, self.port))
		serverSocket.listen()

		while(True):

			(self.clientConnection, clientAddress) = serverSocket.accept()

			while(True):

				data = self.clientConnection.recv(self.buffer)
				if(data==b'Triton_Start'):
					self.request_count = self.request_count + 1
					if self.request_count <= 1:
						self.triton_server_start()
					print("Connection closed")
					break
				if(data==b'Triton_Stop'):
					if self.proc is not None:
						self.triton_server_stop()
						self.request_count = 0
					else:    
						self.clientConnection.send(str.encode('No tritonserver is running ......'))
					break

				elif (data!=b''):
					self.clientConnection.send(str.encode('Not supported request....'))
					break

if __name__ == '__main__':

	triton_server_ip = '127.0.0.1'
	trition_server_port = 35491
	path_to_repo = '/home/lilun/work/projects/nvidia_tensorRT/triton_torch'
	navigator_specific_model_store_path = 'navigator_workspace/interim-model-store'
	triton_server_run_script = '/home/lilun/work/projects/poc/py_server/triton.sh'
	triton_server = Tritonserver_handler(
		triton_server_ip, 
		trition_server_port, 
		os.path.join(path_to_repo, navigator_specific_model_store_path), 
		triton_server_run_script
		)

	triton_server.handle_request()	
