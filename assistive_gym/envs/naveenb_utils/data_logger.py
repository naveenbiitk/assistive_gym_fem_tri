class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            self.file = open(file_path, 'w')
        except FileNotFoundError:
            self.file = None
            print("Logger::Logger: invalid input file path")
        self.delimiter = ','
        self.header()

    def header(self):
        d = self.delimiter
        self.file.write("joint_name" + d + "time" + d + "position" + d + "velocity" + d + "force\n")

    def write_robot_joint_state(self, joint_name, time, position, velocity, force):
        if self.file:
            d = self.delimiter
            self.file.write(joint_name + d + str(time) + d + str(position) + d + str(velocity) + d + str(force))
            self.file.write('\n')

    def close(self):
        if self.file:
            self.file.close()