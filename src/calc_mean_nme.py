import os

if __name__ == '__main__':
    evaluation_path = "./evaluation_cropped"
    pitch_dict = {}
    yaw_dict = {}
    for i in range(-1, 2):
        pitch_dict[str(i)] = 0.0
    for i in range(-4, 5):
        yaw_dict[str(i)] = 0.0
    subjects_evaluated = 0
    faultyOBJs = [3, 30]

    for subject_n in range(1, 54):
        if subject_n in faultyOBJs:
            continue
        else:
            subjects_evaluated += 1
            subject_number_str = f"0{str(subject_n)}" if subject_n<10 else str(subject_n)
            with open(f"{evaluation_path}/{subject_number_str}.txt") as f:
                for line in f:
                    fp, error = line.split(" ")
                    if error=="no_detected_face\n":
                        break
                    pitch, yaw = os.path.splitext(os.path.basename(fp))[0].split("_")[-2:]
                    pitch_dict[pitch] += float(error)
                    yaw_dict[yaw] += float(error)

    for key in pitch_dict:
        pitch_dict[key] /= (subjects_evaluated*9)
    for key in yaw_dict:
        yaw_dict[key] /= (subjects_evaluated*3)
    print(pitch_dict)
    print("\n")
    print(yaw_dict)