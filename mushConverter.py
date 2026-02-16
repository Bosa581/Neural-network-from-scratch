import random

class Mushroom_converer:
    def __init__(self, file):
        # open the original dataset file
        self.file = open("agaricus-lepiota.data","r")
        self.feature_map = {
            "class": ['e','p'],
            "cap-shape": ['b','c','x','f','k','s'],
            "cap-surface": ['f','g','y','s'],
            "cap-color": ['n','b','c','g','r','p','u','e','w','y'],
            "bruises": ['t','f'],
            "odor": ['a','l','c','y','f','m','n','p','s'],
            "gill-attachment": ['a','d','f','n'],
            "gill-spacing": ['c','w','d'],
            "gill-size": ['b','n'],
            "gill-color": ['k','n','b','h','g','r','o','p','u','e','w','y'],
            "stalk-shape": ['e','t'],
            "stalk-root": ['b','c','u','e','z','r','?'],
            "stalk-surface-above-ring": ['f','y','k','s'],
            "stalk-surface-below-ring": ['f','y','k','s'],
            "stalk-color-above-ring": ['n','b','c','g','o','p','e','w','y'],
            "stalk-color-below-ring": ['n','b','c','g','o','p','e','w','y'],
            "veil-type": ['p','u'],
            "veil-color": ['n','o','w','y'],
            "ring-number": ['n','o','t'],
            "ring-type": ['c','e','f','l','n','p','s','z'],
            "spore-print-color": ['k','n','b','h','r','o','u','w','y'],
            "population": ['a','c','n','s','v','y'],
            "habitat": ['g','l','m','p','u','w','d']
        }
        self.feature_order = list(self.feature_map.keys())

    def convert(self):
        # list that will store all one-hot encoded mushrooms
        converted_data_set = []
         # read each line (each mushroom) from the dataset
        for line in self.file:
            line = line.strip()  # remove any leading/trailing whitespace
            attributes = line.split(",") # split values into a list of letters
            oneHotLine = []

            for i in range(len(attributes)):
                attr = attributes[i]  # get the attribute value
                feature = self.feature_order[i] # get the feature name based at this index

                # SPECIAL CASE: class feature produces *two bits* (edible, poisonous)
                if feature == "class":
                    if attr == 'e':
                        oneHotLine.extend([1, 0])  # edible
                    else:
                        oneHotLine.extend([0, 1])  # poisonous

                else:
                    feature_values = self.feature_map[feature] # possible values for this feature
                    # create one-hot encoding for this attribute
                    oneHot = [0] * len(feature_values) # initialize a zero vector for this feature
                    index = feature_values.index(attr) # find the index of the attribute value
                     # set the corresponding index to 1
                    oneHot[index] = 1
                    oneHotLine.extend(oneHot)  # add this one-hot vector to the mushroom's full vector

             # add the fully encoded mushroom to the dataset
            converted_data_set.append(oneHotLine) # add the completed one-hot encoded mushroom to the dataset

        return converted_data_set #return the full dataset of one-hot encoded mushrooms
    

if __name__ == "__main__":
    converter = Mushroom_converer("agaricus-lepiota.data")
    converted_data = converter.convert()
    random.shuffle(converted_data) # shuffle the dataset to ensure randomness

    # write the converted data to a new file
    with open("training.txt","w") as f: 
        for i in range(0, 5686): # generate training set (70% of 8124)
            line = " ".join(map(str,converted_data[i]))
            f.write(line + "\n")

    with open("val.txt","w") as f:
        for i in range(5686, 7108): #generate validation set (15% of 8124)
            line = " ".join(map(str,converted_data[i]))
            f.write(line + "\n")

    with open("testing.txt","w") as f:
        for i in range(7108, 8124): #generate testing set (15% of 8124)
            line = " ".join(map(str,converted_data[i]))
            f.write(line + "\n")

    print("Number of mushrooms converted:", len(converted_data))
    print("Length of mushroom vector:", len(converted_data[0]))
