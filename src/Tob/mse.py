if __name__ == '__main__':
    with open("./gt.txt") as f:
        gt = f.readlines()

    with open("./output.txt") as f:
        output = f.readlines()
    
    thingy = 0
    for gt_line, output_line in zip(gt, output):
        thingy += abs((float(gt_line)-float(output_line)))
    print(thingy/4426)