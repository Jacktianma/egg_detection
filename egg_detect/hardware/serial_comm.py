# hardware/serial_comm.py
import serial
from utils.logger import get_logger

class SerialComm:
    def __init__(self, port, baudrate, timeout):
        self.logger = get_logger("SerialComm")
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        if not self.ser.is_open:
            raise RuntimeError("Failed to open serial port")
        self.logger.info("Serial port opened successfully")

    def send(self, message):
        try:
            full_message = f"{message}\n"
            self.ser.write(full_message.encode())
            self.logger.info(f"Sent: {full_message.strip()}")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise

    def receive(self):
        buffer = bytearray()
        while self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
            buffer.extend(data)
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                yield line.decode(errors='ignore').strip()

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            self.logger.info("Serial port closed")