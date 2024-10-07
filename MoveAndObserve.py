import logging
from ultralytics import YOLO
import os
import luckyrobots as lr

# Set up logging
logger = logging.getLogger('RobotLogger')
logger.setLevel(logging.INFO)  # Log only INFO level and higher messages

file_handler = logging.FileHandler('robot_log.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load the YOLO model
model = YOLO("YOLOv8n.pt")
model.overrides['verbose'] = False # Disable extra printing from YOLO

class MoveAndDetect:
    def __init__(self):
        self.OvenDetected = False
        self.detected_objects = set()  # To keep track of detected objects

    def DoInitialMovement(self):
        commands = [['A 32', 'W 900 1'],
                    ['A 0', 'W 10000 1']]
        
        # Log initial movement
        logger.info("Starting initial movement")
        lr.send_message(commands)

    def HandleRobotOutput(self, robot_images: list):
        if not self.OvenDetected:
            if robot_images and isinstance(robot_images, dict) and 'rgb_cam1' in robot_images:
                image_path = robot_images['rgb_cam1'].get('file_path')
                if os.path.exists(image_path):
                    # Perform object detection using YOLO
                    results = model(image_path)

                    # Iterate through detected objects
                    for class_id in results[0].boxes.cls:
                        try:
                            class_id_int = int(class_id)  # Convert tensor to int
                            object_name = model.names[class_id_int]
                            
                            # Log the object if it's detected for the first time
                            if object_name not in self.detected_objects:
                                self.detected_objects.add(object_name)
                                logger.info(f"Object Detected: {object_name}")

                            # Check if the oven is detected
                            if object_name == 'dining table':
                                self.OvenDetected = True
                                logger.info("Dining Table detected. Moving Towards kitchen.")
                                self.MoveToKitchen()
                                break
                        except Exception as e:
                            logger.error(f"Error processing detection: {e}")
        else:
            return

    def MoveToKitchen(self):
        commands = [["W 7200 1"]]
        lr.send_message(commands)

    def TaskCompleted(self, id: str, message=""):
        logger.info(f'Task Completed: {id}')
        if id == '1234':
            logger.info('Robot is in kitchen now')

if __name__ == '__main__':
    myClass = MoveAndDetect()
    lr.on("start")(myClass.DoInitialMovement)
    lr.on("robot_output")(myClass.HandleRobotOutput)
    lr.on("task_complete")(myClass.TaskCompleted)

    lr.start()
