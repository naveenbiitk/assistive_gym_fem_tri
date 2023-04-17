class Frames:
    def __init__(self):
        armchair_frames = {
            "001": 2000,
            "002": 1800,
            "003": 2100,
            "004": 1900,
            "005": 2000,
            "006": 1900,
            "007": 1800,
            "008": 1400,
            "009": 1600,
            "010": 1500,
            "011": 1500,
            "012": 1700,
            "013": 1900,
            "014": 2000,
            "015": 2300,
            "016": 2200,
            "017": 2400,
            "018": 1700,
            "019": 1700
        }

        full_file_name_armchair_frames = {}

        for key, value in armchair_frames.items():
            full_file_name_armchair_frames.update({"armchair" + str(key) + "_stageII.pkl": value})

        self.armchair_frames = full_file_name_armchair_frames